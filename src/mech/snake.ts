/**
 * 蛇機構（Rapier 物理アタック）の Mechanism 実装。
 * `runPhysicalStairAttemptReplay` をラップし、参照軌道追従の蛇をコース上で走らせて診断する。
 */
import { DEFAULT_STAIR_REPLAY_CONFIG } from '../sim3d/stair-kinematic-replay.ts';
import { sampleStairDiagnostics } from '../sim3d/stair-feasibility.ts';
import { runPhysicalStairAttemptReplay } from '../sim3d/stair-physical-attempt.ts';
import type { StairDynamicsReplay, StairFailureKind } from '../sim3d/stair-dynamics.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';
import type { Mechanism, MechReplay, MechRunCtx, MechScore, StatRow } from './Mechanism.ts';

function fmtFinite(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '∞';
  return value.toFixed(digits);
}

/** ピーク比が物理発散で非有限になった場合に有限値（大きく扱う）へ丸める。 */
function safeRatio(value: number): number {
  return Number.isFinite(value) ? value : 5;
}

function failureLabel(kind: StairFailureKind): string {
  if (kind === 'torque') return 'トルク';
  if (kind === 'fall') return '落下/干渉';
  if (kind === 'slip') return '滑り';
  return '高μ';
}

function formatFailureKinds(kinds: StairFailureKind[]): string {
  return kinds.length > 0 ? kinds.map(failureLabel).join(' / ') : 'pass';
}

class SnakeReplay implements MechReplay {
  readonly duration: number;
  private readonly replay: StairDynamicsReplay;
  private readonly friction: number;
  // パラメータプロパティは使わない（node の型ストリップ実行で未対応のため・明示フィールドにする）。
  constructor(replay: StairDynamicsReplay, friction: number) {
    this.replay = replay;
    this.friction = friction;
    this.duration = replay.summary.config.duration;
  }

  bindView(view: StairDynamicsView): void {
    view.setMechanism('snake');
    view.setReplay(this.replay);
  }

  applyTime(view: StairDynamicsView, t: number): void {
    view.applyTime(t);
  }

  liveStats(t: number): StatRow[] {
    const d = sampleStairDiagnostics(this.replay, t);
    if (!d) return [];
    const torque = d.motorDemandNm ?? d.torqueNm;
    const ratio = d.motorTorqueRatio ?? d.torqueRatio;
    const n = this.replay.summary.config.morphology.n;
    return [
      {
        label: '状態',
        value: formatFailureKinds(d.failureKinds),
        kind: d.failureKinds.length > 0 ? 'bad' : 'good',
      },
      {
        label: '現在 τ/cap',
        value: `${fmtFinite(torque)} / ${fmtFinite(d.torqueLimitNm)} (${fmtFinite(ratio, 1)}x)`,
        kind: ratio > 1 ? 'bad' : ratio > 0.8 ? 'warn' : null,
      },
      {
        label: '現在 μ/設定',
        value: `${fmtFinite(d.requiredMu)} / ${fmtFinite(d.friction)}`,
        kind: d.requiredMu > d.friction ? 'bad' : d.requiredMu > 0.8 ? 'warn' : null,
      },
      {
        label: '滑り速度',
        value: `${(d.slipSpeedMps * 100).toFixed(1)} cm/s`,
        kind: d.failureKinds.includes('slip') ? 'bad' : null,
      },
      {
        label: '最小 clearance',
        value: `${(d.minClearanceM * 1000).toFixed(1)} mm`,
        kind: d.minClearanceM < -0.006 ? 'bad' : null,
      },
      {
        label: '支持リンク',
        value: `${d.supportCount}/${n}`,
        kind: d.failureKinds.includes('fall') ? 'bad' : null,
      },
    ];
  }

  score(): MechScore {
    const s = this.replay.summary;
    // 目的: head が前進した距離を最大化しつつ、モーター cap 超過（torque>cap）と滑り（μ要求>床μ）を減点。
    // ピーク値は recordFps に依らず全ステップで積算される summary フィールドを使う（headless 評価でも正確）。
    // 高すぎる段では物理が発散しピーク比が Infinity になりうるので、有限にクランプして段階的なペナルティにする
    // （集団全体が -Infinity だと CMA-ES が勾配を失うため）。fitness は前進量が支配的になるよう設計。
    const cap = s.config.motor.maxTorqueNm;
    const torqueRatio = cap > 1e-9 ? safeRatio(s.maxDemandTorqueNm / cap) : 5;
    const muRatio = this.friction > 1e-9 ? safeRatio(s.maxMuDemand / this.friction) : 5;
    const torquePen = 0.2 * Math.min(3, Math.max(0, torqueRatio - 1));
    const muPen = 0.2 * Math.min(3, Math.max(0, muRatio - 1));
    // 末尾でリンクが吹き飛ぶ（数値発散）と head x が前進していても歩容として無価値なので、
    // 頭の最終 z が非物理（コース外・|z|>1m）なら前進を無効化し、安定して前へ進む歩容へ誘導する。
    // 階段最上段でも head z は最大 0.54m 程度なので 1m 閾は正当な登攀を妨げない。
    const headX = s.finalHeadTip[0];
    const headZ = s.finalHeadTip[1];
    const stable = Number.isFinite(headX) && Number.isFinite(headZ) && Math.abs(headZ) < 1;
    const progressM = stable ? headX : -1;
    return { fitness: progressM - torquePen - muPen, progressM, feasible: s.success };
  }

  resultStats(): StatRow[] {
    const s = this.replay.summary;
    const f = s.feasibility;
    const head = s.finalHeadTip;
    const rows: StatRow[] = [
      {
        label: '判定',
        value: s.success ? '走破' : '未走破',
        kind: s.success ? 'good' : 'bad',
      },
    ];
    if (f) {
      rows.push(
        {
          label: 'ピーク τ/cap',
          value: `${fmtFinite(f.maxTorqueNm)} N·m (${fmtFinite(f.maxTorqueRatio, 1)}x)`,
          kind: f.maxTorqueRatio > 1 ? 'bad' : f.maxTorqueRatio > 0.8 ? 'warn' : null,
        },
        {
          label: 'ピーク μ要求',
          value: `${fmtFinite(f.maxRequiredMu)} @ ${f.worstMuTime.toFixed(2)}s`,
          kind: f.maxRequiredMu > this.friction ? 'bad' : null,
        },
        {
          label: '初回破綻',
          value: f.firstFailureTime === null ? 'なし' : `${f.firstFailureTime.toFixed(2)}s`,
          kind: f.firstFailureTime === null ? 'good' : 'bad',
        },
      );
    }
    rows.push({
      label: 'final head',
      value: `${(head[0] * 100).toFixed(1)}, ${(head[1] * 100).toFixed(1)} cm`,
    });
    return rows;
  }
}

export const snakeMechanism: Mechanism = {
  id: 'snake',
  name: '機構: 蛇',
  subtitle: 'Rapier physical attempt + diagnostics',
  supportsCourse: true,
  params: [
    {
      key: 'referenceSpeed',
      label: '参照速度',
      min: 0.05,
      max: 0.4,
      step: 0.01,
      default: 0.18,
      unit: 'm/s',
    },
    {
      key: 'clearance',
      label: '持ち上げ clearance',
      min: 0.02,
      max: 0.12,
      step: 0.005,
      default: 0.06,
      unit: 'm',
    },
    // 床摩擦は環境（コース側）の性質で歩容の制御量ではない。歩容最適化中は既定値に固定。
    {
      key: 'friction',
      label: '床摩擦 μ',
      min: 0.15,
      max: 1.4,
      step: 0.05,
      default: 0.6,
      optimize: false,
    },
  ],
  async run(ctx: MechRunCtx): Promise<MechReplay> {
    const friction = ctx.params.friction;
    const replay = await runPhysicalStairAttemptReplay(
      {
        course: ctx.course,
        friction,
        clearance: ctx.params.clearance,
        referenceSpeed: ctx.params.referenceSpeed,
        motor: { ...DEFAULT_STAIR_REPLAY_CONFIG.motor, maxTorqueNm: ctx.torqueCapNm },
      },
      45,
    );
    return new SnakeReplay(replay, friction);
  },
};
