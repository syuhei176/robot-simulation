import {
  DEFAULT_STAIR_DYNAMICS_CONFIG,
  runFrictionSweep,
  runStairDynamics,
  type StairDynamicsConfig,
  type StairDynamicsSummary,
} from '../src/sim3d/stair-dynamics.ts';
import { runPhysicalStairAttemptReplay } from '../src/sim3d/stair-physical-attempt.ts';

const NM_PER_KGCM = 0.0980665;

function fmt(n: number, digits = 2): string {
  return n.toFixed(digits);
}

function tq(nm: number): string {
  return `${fmt(nm, 3)} N·m (${fmt(nm / NM_PER_KGCM, 1)} kg·cm)`;
}

function maybe(n: number, digits = 2): string {
  return Number.isFinite(n) ? fmt(n, digits) : '∞';
}

function printSummary(summary: StairDynamicsSummary): void {
  const c = summary.config;
  const dynamicRatio =
    summary.staticArcPeakNm > 1e-9 ? summary.maxDemandTorqueNm / summary.staticArcPeakNm : Infinity;

  console.log('=== Rapier 階段ダイナミクス（Phase C 初版） ===');
  console.log(
    `形状: ${c.morphology.n}リンク / 総長${fmt(c.morphology.totalLength * 100, 0)}cm / 総質量${fmt(c.morphology.totalMass, 2)}kg`,
  );
  console.log(
    `段: 蹴上げ=${fmt(c.stair.rise * 100, 0)}cm / 踏面=${fmt(c.stair.treadDepth * 100, 0)}cm / lift-off前方=${fmt(c.stair.forward * 100, 0)}cm / μ=${fmt(c.friction, 2)}`,
  );
  console.log(
    `制御: dt=${fmt(c.dt * 1000, 1)}ms / stiffness=${fmt(c.motor.stiffness, 2)} / damping=${fmt(c.motor.damping, 2)} / torque cap=${tq(c.motor.maxTorqueNm)}`,
  );
  console.log(`牽引プローブ: t>=2.8s に tail +x ${fmt(c.tractionProbeForceN, 1)}N`);
  console.log('');
  console.log('--- 結果 ---');
  console.log(`持ち上げリンク数: ${summary.liftLinks}`);
  console.log(`静的円弧ピーク: ${tq(summary.staticArcPeakNm)}`);
  console.log(
    `動的PD要求ピーク: ${tq(summary.maxDemandTorqueNm)}  joint=${summary.maxDemandJoint}  静的比=${fmt(dynamicRatio, 2)}x`,
  );
  console.log(
    `実印加ピーク: ${tq(summary.maxAppliedTorqueNm)}  joint=${summary.maxAppliedJoint}  saturation steps=${summary.saturatedSteps}`,
  );
  console.log(
    `接触ピーク: normal=${fmt(summary.maxContactNormalForceN, 1)}N / tangent impulse=${fmt(summary.maxContactTangentForceN, 1)}N / probe μ≈${fmt(summary.maxProbeMuDemand, 2)} / μ要求≈${fmt(summary.maxMuDemand, 2)}`,
  );
  console.log(
    `head tip final=(${fmt(summary.finalHeadTip[0] * 100, 1)}, ${fmt(summary.finalHeadTip[1] * 100, 1)})cm  maxZ=${fmt(summary.maxHeadTipZ * 100, 1)}cm  success=${summary.success}`,
  );
}

const baseline = await runStairDynamics();
printSummary(baseline);

console.log('');
console.log('=== 摩擦スイープ（同じ歩容・同じ制御） ===');
console.log('  μ   | success | max τ要求 | μ要求 | friction-limited samples');
const sweep = await runFrictionSweep([0.3, 0.5, 0.8, 1.1], DEFAULT_STAIR_DYNAMICS_CONFIG);
for (const s of sweep) {
  console.log(
    ` ${fmt(s.config.friction, 1)} | ${String(s.success).padStart(7)} | ${tq(s.maxDemandTorqueNm).padStart(19)} | ${fmt(s.maxMuDemand, 2).padStart(5)} | ${String(s.frictionLimitedSamples).padStart(8)}`,
  );
}

console.log('');
console.log('=== 完全階段 Rapier physical attempt（失敗も実際に動かす） ===');
console.log(
  '注意: キネマティック軌道を目標にするが、実体は剛体リンク + torque cap付き関節 + μ制限付き牽引で動く。',
);
const attempt = await runPhysicalStairAttemptReplay(
  {
    friction: 0.6,
    motor: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.motor,
      maxTorqueNm: 0.25,
    },
  },
  45,
);
const f = attempt.summary.feasibility;
if (!f) throw new Error('physical attempt diagnostics were not produced');
console.log(`物理条件: torque cap=${tq(0.25)} / μ=${fmt(0.6, 2)}`);
console.log(
  `状態: ${attempt.summary.success ? 'success' : 'failed'}  first failure=${f.firstFailureTime === null ? 'なし' : `${fmt(f.firstFailureTime, 2)}s`}`,
);
console.log(
  `failure frames: torque=${f.failureCounts.torque} / fall-or-collision=${f.failureCounts.fall} / slip=${f.failureCounts.slip} / high-mu=${f.failureCounts.mu}`,
);
console.log(
  `motor demand peak=${tq(attempt.summary.maxDemandTorqueNm)} / applied peak=${tq(attempt.summary.maxAppliedTorqueNm)} / saturation steps=${attempt.summary.saturatedSteps}`,
);
console.log(
  `diagnostic torque peak=${tq(f.maxTorqueNm)} (${maybe(f.maxTorqueRatio, 2)}x cap) @ ${fmt(f.worstTorqueTime, 2)}s`,
);
console.log(
  `peak μ要求=${maybe(f.maxRequiredMu, 2)} @ ${fmt(f.worstMuTime, 2)}s / max slip=${fmt(f.maxSlipSpeedMps * 100, 1)}cm/s`,
);
console.log(
  `final head=(${fmt(attempt.summary.finalHeadTip[0] * 100, 1)}, ${fmt(attempt.summary.finalHeadTip[1] * 100, 1)})cm / min clearance=${fmt(f.minClearanceM * 1000, 1)}mm / max unsupported span=${fmt(f.maxUnsupportedSpanM * 100, 1)}cm`,
);

// ---- (A) ゆっくり歩容で SG90 に収める：gaitTimeScale 掃引 ----------------
// 加速 ∝ 1/scale² なので、歩容を遅くするほど慣性トルクが下がり静的floorへ漸近する。
const SG90_STALL_NM = 0.176; // 1.8 kg·cm（プラギア・FBなし）
const GAIT_MOTION_BASE_S = 2.8; // gaitTimeScale=1 で持ち上げ→配置が終わる時刻
const GAIT_SCALES = [1, 2, 4, 6, 8, 12];

console.log('');
console.log('=== SG90 ゆっくり歩容スイープ（torque cap = SG90 ストール 0.176 N·m 固定） ===');
console.log('  歩容を遅くして動的要求ピークが SG90 ストール以下に収まるケイデンスを探す。');

async function sg90CadenceSweep(label: string, base: Partial<StairDynamicsConfig>): Promise<void> {
  console.log('');
  console.log(`--- ${label} ---`);
  console.log('  gait× | 1段秒 | 動的要求ピーク         | SG90比 | ピーク時刻/関節 | success');
  for (const g of GAIT_SCALES) {
    const motionEnd = GAIT_MOTION_BASE_S * g;
    const summary = await runStairDynamics({
      ...base,
      gaitTimeScale: g,
      duration: motionEnd + 1.2,
      tractionProbeForceN: 0, // 固定3Nプローブは軽量機には過大。歩容そのものの保持トルクを分離する
      demandWarmupS: 0.25 * g, // 生成直後の整定スパイク（t≈0.04s）を除外
      motor: { ...DEFAULT_STAIR_DYNAMICS_CONFIG.motor, maxTorqueNm: SG90_STALL_NM },
    });
    const ratio = summary.maxDemandTorqueNm / SG90_STALL_NM;
    const flag = summary.maxDemandTorqueNm <= SG90_STALL_NM ? ' ← SG90に収まる' : '';
    const peakPhase = summary.maxDemandTimeS / motionEnd; // 0=開始, 1=配置完了（歩容内の位相）
    console.log(
      `  ${fmt(g, 0).padStart(4)} | ${fmt(motionEnd, 1).padStart(5)} | ${tq(summary.maxDemandTorqueNm).padStart(21)} | ${fmt(ratio, 2).padStart(5)}x | ${fmt(summary.maxDemandTimeS, 2).padStart(5)}s(位相${fmt(peakPhase, 2)})/j${summary.maxDemandJoint} | ${String(summary.success).padStart(5)}${flag}`,
    );
  }
}

// A: 実階段(蹴上18cm)のまま、静的では SG90 に乗る軽量90cm/0.15kg
await sg90CadenceSweep('A: 実階段18cm × 90cm/0.15kg', {
  morphology: { ...DEFAULT_STAIR_DYNAMICS_CONFIG.morphology, totalMass: 0.15 },
});

// B: 幾何相似ダウンスケール（s=0.6）＝トイ階段。質量∝s³・段も×s・body寸も×s
const SC = 0.6;
await sg90CadenceSweep(
  `B: トイ階段 s=${SC}（${fmt(0.9 * SC * 100, 0)}cm/${fmt(SC ** 3, 3)}kg, 蹴上${fmt(0.18 * SC * 100, 1)}cm）`,
  {
    morphology: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.morphology,
      totalLength: 0.9 * SC,
      totalMass: 1.0 * SC ** 3,
      bodyWidth: 0.04 * SC,
      bodyThickness: 0.028 * SC,
    },
    stair: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.stair,
      rise: 0.18 * SC,
      forward: 0.1 * SC,
      treadDepth: 0.25 * SC,
    },
    clearance: 0.06 * SC,
  },
);

console.log('');
console.log(
  '注: ピークは gait×1〜12 で平坦＝準静的な保持トルク（慣性項はこの質量・段では無視できる）。',
);
console.log(
  '注: つまり「ゆっくり化」では下がらない。効くのは保持トルク自体＝質量・持ち上げ姿勢・支持の取り方。',
);
console.log(
  '注: 前報の 7.5x/3.9x は固定3N牽引プローブ（軽量機には過大, weight比1.4〜2x）＋生成直後の整定スパイクの寄与で、歩容本体の要求ではなかった。',
);
