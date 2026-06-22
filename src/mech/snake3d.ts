/**
 * 汎用 3D ヘビ（MuJoCo）の Mechanism 実装。
 * 関節構成（JointSpec＝軸パターン）と歩容（yaw/pitch のパラメタ化波）を分離した「キャンバス」を
 * ダッシュボードへ出す。pattern を切り替え pitchAmp を上げると平面うねり→3D 運動へ連続的に動く。
 */
import {
  runSnake3D,
  DEFAULT_SNAKE3D_CONFIG,
  type AxisPattern,
  type Snake3DReplay,
} from '../sim3d/snake3d-dynamics.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';
import type { Mechanism, MechReplay, MechRunCtx, MechScore, StatRow } from './Mechanism.ts';

const D = DEFAULT_SNAKE3D_CONFIG;

/** スライダー値（0/1/2）→ 軸パターン。 */
const PATTERNS: AxisPattern[] = ['all-yaw', 'alt-yaw-pitch', 'all-pitch'];
function patternFrom(v: number): AxisPattern {
  return PATTERNS[Math.max(0, Math.min(PATTERNS.length - 1, Math.round(v)))];
}

class Snake3DMechReplay implements MechReplay {
  readonly duration: number;
  private readonly replay: Snake3DReplay;
  private readonly motorName: string;
  private readonly torqueCapNm: number;
  constructor(replay: Snake3DReplay, motorName: string, torqueCapNm: number) {
    this.replay = replay;
    this.motorName = motorName;
    this.torqueCapNm = torqueCapNm;
    this.duration = replay.summary.config.duration;
  }

  bindView(view: StairDynamicsView): void {
    view.setMechanism('snake3d');
    view.buildSnake3D(this.replay.layout);
    // 頭（末尾リンク）の全軌跡を渡して trail / 目盛り / 開始マーカーを描く（移動が一目で分かる）。
    const headIdx = this.replay.layout.length - 1;
    const headPts = this.replay.frames.map(
      (fr): [number, number, number] => fr.bodies[headIdx]?.p ?? [0, 0, 0],
    );
    view.setSnake3DTrail(headPts);
  }

  private frameAt(t: number): Snake3DReplay['frames'][number] | null {
    const frames = this.replay.frames;
    if (frames.length === 0) return null;
    const dur = this.duration;
    const tt = dur > 0 ? ((t % dur) + dur) % dur : 0;
    let hi = 1;
    while (hi < frames.length && frames[hi].t < tt) hi++;
    return frames[Math.min(frames.length - 1, hi)];
  }

  applyTime(view: StairDynamicsView, t: number): void {
    const frame = this.frameAt(t);
    if (frame) view.applySnake3DFrame(frame);
  }

  liveStats(t: number): StatRow[] {
    const frame = this.frameAt(t);
    if (!frame) return [];
    const d = frame.diag;
    const ratio = this.torqueCapNm > 1e-9 ? d.demandNm / this.torqueCapNm : Infinity;
    return [
      { label: '現在 移動', value: `${(d.travelM * 100).toFixed(1)} cm` },
      {
        label: '現在 τ/cap',
        value: `${d.demandNm.toFixed(2)} / ${this.torqueCapNm.toFixed(2)} (${ratio.toFixed(1)}x)`,
        kind: d.saturated ? 'warn' : null,
      },
    ];
  }

  score(): MechScore {
    const s = this.replay.summary;
    return { fitness: s.travelM, progressM: s.travelM, feasible: s.success };
  }

  resultStats(): StatRow[] {
    const s = this.replay.summary;
    return [
      { label: 'モーター', value: this.motorName },
      {
        label: '移動距離',
        value: `${(s.travelM * 100).toFixed(1)} cm`,
        kind: s.success ? 'good' : 'warn',
      },
      { label: 'ピーク τ', value: `${s.maxDemandNm.toFixed(2)} N·m` },
      { label: 'モーター τ上限', value: `${this.torqueCapNm.toFixed(2)} N·m` },
      {
        label: '判定',
        value: s.success ? '移動OK' : '移動不足',
        kind: s.success ? 'good' : 'bad',
      },
    ];
  }
}

export const snake3dMechanism: Mechanism = {
  id: 'snake3d',
  name: '機構: 蛇3D (MuJoCo)',
  subtitle: 'MuJoCo 汎用蛇: 関節構成(JointSpec)×歩容を分離したキャンバス',
  supportsCourse: false, // 現状は平地のみ（柱登攀は後続）
  params: [
    // --- 関節構成（モルフォロジー: 歩容ではないので最適化対象外） ---
    {
      key: 'pattern',
      label: '関節(0=yaw,1=交互,2=pitch)',
      min: 0,
      max: 2,
      step: 1,
      default: 0,
      optimize: false,
    },
    // --- 歩容 ---
    {
      key: 'yawAmp',
      label: '水平振幅',
      min: 0,
      max: 0.9,
      step: 0.05,
      default: D.yawAmp,
      unit: 'rad',
    },
    {
      key: 'pitchAmp',
      label: '垂直振幅',
      min: 0,
      max: 0.9,
      step: 0.05,
      default: D.pitchAmp,
      unit: 'rad',
    },
    {
      key: 'period',
      label: '波の周期',
      min: 0.8,
      max: 2.4,
      step: 0.05,
      default: D.period,
      unit: 's',
    },
    { key: 'waveLength', label: '波長(リンク数)', min: 4, max: 12, step: 1, default: D.waveLength },
  ],
  async run(ctx: MechRunCtx): Promise<MechReplay> {
    const replay = await runSnake3D(
      {
        pattern: patternFrom(ctx.params.pattern),
        yawAmp: ctx.params.yawAmp,
        pitchAmp: ctx.params.pitchAmp,
        period: ctx.params.period,
        waveLength: Math.round(ctx.params.waveLength),
        motor: { ...D.motor, maxTorqueNm: ctx.torqueCapNm },
      },
      60,
    );
    return new Snake3DMechReplay(replay, ctx.motorName, ctx.torqueCapNm);
  },
};
