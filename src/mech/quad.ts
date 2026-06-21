/**
 * 四足機構（Rapier 3D 動的歩行）の Mechanism 実装。
 * `runQuadrupedGait` をラップ。総質量は trunk:leg 比を保ってスケールし、歩容パラメータはライブ調整。
 * 歩容は平地前提のため supportsCourse=false（段差走破は後続段）。
 */
import {
  runQuadrupedGait,
  DEFAULT_QUAD_DYN_CONFIG,
  type QuadDynReplay,
} from '../sim3d/quadruped-dynamics.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';
import type { Mechanism, MechReplay, MechRunCtx, MechScore, StatRow } from './Mechanism.ts';

const BASE_TRUNK = DEFAULT_QUAD_DYN_CONFIG.trunk.mass;
const BASE_SEG = DEFAULT_QUAD_DYN_CONFIG.leg.segMass;
const BASE_TOTAL = BASE_TRUNK + 8 * BASE_SEG;
const G = DEFAULT_QUAD_DYN_CONFIG.gait;

class QuadReplay implements MechReplay {
  readonly duration: number;
  private readonly replay: QuadDynReplay;
  private readonly torqueCapNm: number;
  private readonly motorName: string;
  // パラメータプロパティは使わない（node の型ストリップ実行で未対応のため・明示フィールドにする）。
  constructor(replay: QuadDynReplay, torqueCapNm: number, motorName: string) {
    this.replay = replay;
    this.torqueCapNm = torqueCapNm;
    this.motorName = motorName;
    this.duration = replay.summary.config.duration;
  }

  bindView(view: StairDynamicsView): void {
    view.setMechanism('quad');
    view.buildQuadReplay(this.replay.layout);
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
    const progressM = Number.isFinite(s.forwardDistanceM) ? s.forwardDistanceM : -1;
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
  subtitle: 'Rapier 3D 動的歩行（クロール歩容）',
  supportsCourse: false,
  params: [
    {
      key: 'mass',
      label: '総質量',
      min: 0.8,
      max: 2.5,
      step: 0.05,
      default: BASE_TOTAL,
      unit: 'kg',
      // 機体設計の選択であって歩容の制御量ではない。歩容最適化中は既定値に固定。
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
      max: 0.1,
      step: 0.005,
      default: G.strideM,
      unit: 'm',
    },
    {
      key: 'liftM',
      label: '遊脚持ち上げ',
      min: 0.01,
      max: 0.06,
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
    const massFactor = ctx.params.mass / BASE_TOTAL;
    const replay = await runQuadrupedGait(
      {
        trunk: { mass: BASE_TRUNK * massFactor },
        leg: { segMass: BASE_SEG * massFactor },
        motor: { maxTorqueNm: ctx.torqueCapNm },
        gait: {
          period: ctx.params.period,
          strideM: ctx.params.strideM,
          liftM: ctx.params.liftM,
          standM: ctx.params.standM,
          stanceDuty: ctx.params.stanceDuty,
        },
      },
      60,
    );
    return new QuadReplay(replay, ctx.torqueCapNm, ctx.motorName);
  },
};
