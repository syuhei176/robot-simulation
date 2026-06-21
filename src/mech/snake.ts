/**
 * 蛇機構（Rapier 物理アタック）の Mechanism 実装。
 * `runPhysicalStairAttemptReplay` をラップし、参照軌道追従の蛇をコース上で走らせて診断する。
 */
import { DEFAULT_STAIR_REPLAY_CONFIG } from '../sim3d/stair-kinematic-replay.ts';
import { sampleStairDiagnostics } from '../sim3d/stair-feasibility.ts';
import { runPhysicalStairAttemptReplay } from '../sim3d/stair-physical-attempt.ts';
import type { StairDynamicsReplay, StairFailureKind } from '../sim3d/stair-dynamics.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';
import type { Mechanism, MechReplay, MechRunCtx, StatRow } from './Mechanism.ts';

function fmtFinite(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '∞';
  return value.toFixed(digits);
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
  constructor(
    private readonly replay: StairDynamicsReplay,
    private readonly friction: number,
  ) {
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
    { key: 'friction', label: '床摩擦 μ', min: 0.15, max: 1.4, step: 0.05, default: 0.6 },
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
