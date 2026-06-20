/**
 * phaseLag を +x の共鳴帯 [0.8, 1.0] に絞った場合、振幅・周波数によらず
 * 安定して +x へ前進するかを検証する。
 */
import { DEFAULT_SIM } from '../src/config.ts';
import { SnakePhysics } from '../src/sim/SnakePhysics.ts';

const dt = DEFAULT_SIM.dt;
const steps = 800;

function run(A: number, F: number, phaseLag: number): number {
  const sim = new SnakePhysics(DEFAULT_SIM);
  const targets = new Float64Array(sim.jointCount);
  let phase = 0;
  const x0 = sim.centerOfMass()[0];
  for (let s = 0; s < steps; s++) {
    phase += 2 * Math.PI * F * dt;
    for (let j = 0; j < targets.length; j++) targets[j] = A * Math.sin(phase + j * phaseLag);
    sim.setJointTargets(targets);
    sim.step(dt);
  }
  return sim.centerOfMass()[0] - x0;
}

for (const F of [0.4, 0.7, 1.0, 1.3]) {
  for (const A of [0.3, 0.55, 0.8]) {
    const row = [0.8, 0.85, 0.9, 0.95, 1.0]
      .map((p) => `p${p.toFixed(2)}:${run(A, F, p).toFixed(1).padStart(6)}`)
      .join('  ');
    console.log(`F=${F.toFixed(1)} A=${A.toFixed(2)}  ${row}`);
  }
}
