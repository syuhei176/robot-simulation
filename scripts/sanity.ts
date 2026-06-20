/**
 * Three.js を介さず物理モデルだけを検証するヘッドレステスト。
 * CPG でしばらく駆動し、重心が体軸方向にきちんと前進するかを確認する。
 *   実行: pnpm sanity   (Node 22+/24 の型ストリップ機能を利用)
 */
import { DEFAULT_SIM, DEFAULT_GAIT } from '../src/config.ts';
import { SnakePhysics } from '../src/sim/SnakePhysics.ts';
import { CPGController } from '../src/control/CPGController.ts';

const sim = new SnakePhysics(DEFAULT_SIM);
const cpg = new CPGController(sim.jointCount, { ...DEFAULT_GAIT });

const start = sim.centerOfMass();
const seconds = 10;
const steps = Math.round(seconds / DEFAULT_SIM.dt);

let maxSpeed = 0;
for (let i = 0; i < steps; i++) {
  sim.setJointTargets(cpg.update(DEFAULT_SIM.dt));
  sim.step(DEFAULT_SIM.dt);
  const v = sim.centerOfMassVelocity();
  maxSpeed = Math.max(maxSpeed, Math.hypot(v[0], v[1]));
}

const end = sim.centerOfMass();
const dx = end[0] - start[0];
const dy = end[1] - start[1];
const dist = Math.hypot(dx, dy);

const finite = sim.pos.every(Number.isFinite);

console.log('=== microbot snake sanity ===');
console.log(`links=${DEFAULT_SIM.links} joints=${sim.jointCount} time=${seconds}s`);
console.log(`displacement: dx=${dx.toFixed(3)} dy=${dy.toFixed(3)} |d|=${dist.toFixed(3)} m`);
console.log(
  `avg speed=${(dist / seconds).toFixed(3)} m/s  max node-COM speed=${maxSpeed.toFixed(3)}`,
);
console.log(`numerically stable (all finite): ${finite}`);

if (!finite) {
  throw new Error('FAIL: 数値が発散しました。剛性/サブステップを見直してください。');
}
if (dist < 0.3) {
  console.warn('WARN: ほとんど前進していません。歩容/摩擦パラメータの調整が必要です。');
} else {
  console.log('OK: 前進を確認しました。');
}
