import {
  DEFAULT_STAIR_DYNAMICS_CONFIG,
  runFrictionSweep,
  runStairDynamics,
  type StairDynamicsSummary,
} from '../src/sim3d/stair-dynamics.ts';

const NM_PER_KGCM = 0.0980665;

function fmt(n: number, digits = 2): string {
  return n.toFixed(digits);
}

function tq(nm: number): string {
  return `${fmt(nm, 3)} N·m (${fmt(nm / NM_PER_KGCM, 1)} kg·cm)`;
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
