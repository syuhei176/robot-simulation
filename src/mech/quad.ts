/**
 * 四足機構（Rapier 3D 動的歩行）の Mechanism 実装。
 * `runQuadrupedGait` をラップ。総質量スライダーは「機体スケール」を兼ねる: 密度一定の相似縮小で
 * mass ∝ s³（s = (mass/BASE_TOTAL)^(1/3)）とし、脚長・歩容 ∝ s、PD ゲイン ∝ s⁵、横安定化 ∝ s⁴、
 * substeps ∝ 1/s でスケールする。これで小型・軽量な機体（150g 級）が SCS0009 のトルク上限内で歩け、
 * 同じダッシュボードで「小型四足×安サーボ」も「大型四足×高トルクサーボ」も評価できる。
 * s=1（mass=BASE_TOTAL=1.2kg）では全係数が 1 で従来の機体に一致する。
 */
import {
  runQuadrupedGait,
  DEFAULT_QUAD_DYN_CONFIG,
  bodyScale,
  scaledBodyOverrides,
  type QuadDynReplay,
} from '../sim3d/quadruped-dynamics.ts';
import type { CourseSpec } from '../sim3d/course.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';
import type { Mechanism, MechReplay, MechRunCtx, MechScore, StatRow } from './Mechanism.ts';

const D = DEFAULT_QUAD_DYN_CONFIG;
// スケール基準（s=1 になる質量）= 基準機体 1.2kg。歩容スライダーもこの基準機体の単位で表す。
const G = D.gait;
// ダッシュボード/最適化の既定機体は SCS0009 級の小型四足（150g）。s=(0.15/1.2)^(1/3)=0.5。
const DEFAULT_MASS = 0.15;

class QuadReplay implements MechReplay {
  readonly duration: number;
  private readonly replay: QuadDynReplay;
  private readonly torqueCapNm: number;
  private readonly motorName: string;
  private readonly course: CourseSpec;
  // パラメータプロパティは使わない（node の型ストリップ実行で未対応のため・明示フィールドにする）。
  constructor(replay: QuadDynReplay, torqueCapNm: number, motorName: string, course: CourseSpec) {
    this.replay = replay;
    this.torqueCapNm = torqueCapNm;
    this.motorName = motorName;
    this.course = course;
    this.duration = replay.summary.config.duration;
  }

  bindView(view: StairDynamicsView): void {
    view.setMechanism('quad');
    view.buildQuadReplay(this.replay.layout);
    view.showCourse(this.course);
  }

  applyTime(view: StairDynamicsView, t: number): void {
    const frames = this.replay.frames;
    if (frames.length === 0) return;
    const dur = this.duration;
    const tt = dur > 0 ? ((t % dur) + dur) % dur : 0;
    let hi = 1;
    while (hi < frames.length && frames[hi].t < tt) hi++;
    view.applyQuadFrame(frames[Math.min(frames.length - 1, hi)]);
  }

  private frameAt(t: number): QuadDynReplay['frames'][number] | null {
    const frames = this.replay.frames;
    if (frames.length === 0) return null;
    const dur = this.duration;
    const tt = dur > 0 ? ((t % dur) + dur) % dur : 0;
    let hi = 1;
    while (hi < frames.length && frames[hi].t < tt) hi++;
    return frames[Math.min(frames.length - 1, hi)];
  }

  liveStats(t: number): StatRow[] {
    const frame = this.frameAt(t);
    if (!frame) return [];
    const d = frame.diag;
    const ratio = this.torqueCapNm > 1e-9 ? d.demandNm / this.torqueCapNm : Infinity;
    return [
      {
        label: '現在 前進',
        value: `${(d.forwardX * 100).toFixed(1)} cm`,
        kind: d.fallen ? 'bad' : null,
      },
      {
        label: '胴の傾き',
        value: `${d.tiltDeg.toFixed(0)}°`,
        kind: d.fallen ? 'bad' : d.tiltDeg > 30 ? 'warn' : null,
      },
      {
        label: '現在 τ/cap',
        value: `${d.demandNm.toFixed(2)} / ${this.torqueCapNm.toFixed(2)} (${ratio.toFixed(1)}x)`,
        kind: d.saturated ? 'warn' : null,
      },
    ];
  }

  score(): MechScore {
    const s = this.replay.summary;
    // 目的: 転ばず・直立を保ったまま前進距離を最大化。転倒は強く減点（前へ滑り込んでも報われない）。
    // 過度な傾き(>25°)は微減点。トルク cap は sim 側で clamp 済み＝弱いモーターは自然に前進せず低 fitness。
    // 前進はコース goal（＋踊り場少し）で頭打ちにする: コース端の先（床のない void）へ飛んで距離を
    // 稼ぐ報酬ハックを防ぐ。平地は goalX が遠く実質キャップなし＝従来の前進距離最大化と一致。
    const raw = Number.isFinite(s.forwardDistanceM) ? s.forwardDistanceM : -1;
    const progressM = Math.min(raw, this.course.goalX + 0.5);
    const tilt = Number.isFinite(s.maxTiltDeg) ? s.maxTiltDeg : 90;
    let fitness = progressM;
    if (s.fell) fitness -= 0.5;
    fitness -= 0.003 * Math.max(0, tilt - 25);
    return { fitness, progressM, feasible: s.success };
  }

  resultStats(): StatRow[] {
    const s = this.replay.summary;
    return [
      { label: 'モーター', value: this.motorName },
      {
        label: '前進距離',
        value: `${(s.forwardDistanceM * 100).toFixed(1)} cm`,
        kind: s.success ? 'good' : 'warn',
      },
      {
        label: '転倒',
        value: s.fell ? `${(s.fellTime ?? 0).toFixed(1)}s で転倒` : 'なし',
        kind: s.fell ? 'bad' : 'good',
      },
      { label: 'モーター τ上限', value: `${this.torqueCapNm.toFixed(2)} N·m` },
      {
        label: '判定',
        value: s.success ? '歩行OK' : s.fell ? '転倒' : '失速（前進せず）',
        kind: s.success ? 'good' : 'bad',
      },
    ];
  }
}

export const quadMechanism: Mechanism = {
  id: 'quad',
  name: '機構: 四足',
  subtitle: 'Rapier 3D 動的歩行（地形適応クロール歩容）',
  supportsCourse: true,
  params: [
    {
      key: 'mass',
      label: '総質量(機体スケール)',
      min: 0.1,
      max: 2.5,
      step: 0.05,
      default: DEFAULT_MASS,
      unit: 'kg',
      // 機体設計の選択であって歩容の制御量ではない。歩容最適化中は既定値に固定（--mass で上書き可）。
      // 密度一定の相似縮小で機体スケール s=(mass/BASE_TOTAL)^(1/3) を兼ねる（脚長・PD 等が連動）。
      optimize: false,
    },
    {
      key: 'period',
      label: '歩容周期',
      min: 0.6,
      max: 2.4,
      step: 0.05,
      default: G.period,
      unit: 's',
    },
    {
      key: 'strideM',
      label: 'ストライド',
      min: 0.02,
      max: 0.12,
      step: 0.005,
      default: G.strideM,
      unit: 'm',
    },
    {
      key: 'liftM',
      label: '遊脚持ち上げ',
      min: 0.01,
      max: 0.1, // 段差・障害物を越えるには高く上げる必要があるため上限を拡張
      step: 0.005,
      default: G.liftM,
      unit: 'm',
    },
    {
      key: 'standM',
      label: '保持高さ',
      min: 0.12,
      max: 0.175,
      step: 0.005,
      default: G.standM,
      unit: 'm',
    },
    { key: 'stanceDuty', label: '接地比', min: 0.5, max: 0.9, step: 0.05, default: G.stanceDuty },
  ],
  async run(ctx: MechRunCtx): Promise<MechReplay> {
    // 機体スケール s（密度一定の相似縮小: mass ∝ s³）。胴・脚・PD・横安定化・substeps は共有関数で
    // スケール（QuadEnv と同一ソース）。歩容スライダーだけは基準機体(1.2kg)の単位なのでここで s 連動させる。
    const s = bodyScale(ctx.params.mass);
    const sqrtS = Math.sqrt(s); // 動的相似（Froude）: 時間 ∝ √s。歩容周期を縮めて前進を稼ぐ。
    // 地形のあるコースは端まで歩くのに時間が要るので、ゴールまでの距離に応じて duration を伸ばす
    // （平地は従来どおり既定 5s でトルク/前進を測る回帰互換）。
    const duration =
      ctx.course.stepRise > 0
        ? Math.min(32, Math.max(8, (ctx.course.goalX + 0.8) / 0.08 + 4))
        : D.duration;
    const replay = await runQuadrupedGait(
      {
        ...scaledBodyOverrides(ctx.params.mass, ctx.torqueCapNm),
        course: ctx.course,
        duration,
        // 歩容スライダーは基準機体(1.2kg)の単位。機体スケール s で幾何相似に縮める（長さ ∝ s, 周期 ∝ √s）。
        gait: {
          period: ctx.params.period * sqrtS,
          strideM: ctx.params.strideM * s,
          liftM: ctx.params.liftM * s,
          standM: ctx.params.standM * s,
          stanceDuty: ctx.params.stanceDuty,
        },
      },
      60,
    );
    return new QuadReplay(replay, ctx.torqueCapNm, ctx.motorName, ctx.course);
  },
};
