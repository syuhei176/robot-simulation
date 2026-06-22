/// <reference types="node" />
/**
 * 保存済み RL 方策（public/policies/<stem>.json）を再ロードし、決定論ロールアウトで前進量を測って
 * 基盤歩容（残差0）と比較する。「RL が基盤を超えた」が再現する成果物であることの検証。
 *
 *   node scripts/verify-policy.ts                 # 既定 snake3d-challenge-mg996r
 *   node scripts/verify-policy.ts --stem snake3d-challenge-mg996r
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { getServo } from '../src/sim3d/servos.ts';
import {
  makeProgressionTerrain,
  makeStraightChallengeTerrain,
  type SnakeTerrainBox,
} from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, defaultSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';

/** 検証に使う名前付きコース（汎用性を見るため複数で評価する）。 */
const NAMED_COURSES: Record<string, () => SnakeTerrainBox[]> = {
  flat: () => [],
  progression: makeProgressionTerrain,
  challenge: makeStraightChallengeTerrain,
};

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

interface PolicyFile {
  motor: string;
  course: string;
  obsDim: number;
  actDim: number;
  hidden: number;
  weights: { policy: number[][]; value: number[][]; logStd: number[] };
  baseForwardM?: number;
  forwardM: number;
}

function evalDet(env: SnakeEnv, policy: Policy): { forwardM: number; dyM: number } {
  let obs = env.reset();
  const startX = env.progressMetric();
  let done = false;
  while (!done) {
    const res = env.step(policy.actMean(obs));
    obs = res.obs;
    done = res.done;
  }
  return { forwardM: env.progressMetric() - startX, dyM: env.getReplay().summary.netDispM[1] };
}

function evalBase(env: SnakeEnv): { forwardM: number; dyM: number } {
  env.reset();
  const startX = env.progressMetric();
  const zeros = new Float32Array(env.actDim);
  let done = false;
  while (!done) done = env.step(zeros).done;
  return { forwardM: env.progressMetric() - startX, dyM: env.getReplay().summary.netDispM[1] };
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  let stem = 'snake3d-general-mg996r';
  for (let i = 0; i < argv.length; i++) if (argv[i] === '--stem') stem = argv[++i];

  const file = JSON.parse(
    readFileSync(join(ROOT, 'public', 'policies', `${stem}.json`), 'utf8'),
  ) as PolicyFile;
  const servo = getServo(file.motor);

  await tf.setBackend('cpu');
  await tf.ready();

  const policy = new Policy(file.obsDim, file.actDim, file.hidden);
  policy.importWeights(file.weights);

  console.log(`=== 検証: ${stem} === 学習コース=${file.course} / モーター=${servo.name}`);
  console.log(`  名前付きコースで横断評価（汎用性の確認）。RL は基盤(残差0)を各コースで上回るか:`);

  // 名前付きコースを横断して、同一方策が各コースで基盤を上回るかを見る（汎用性）。
  for (const [name, makeTerrain] of Object.entries(NAMED_COURSES)) {
    const envCfg = defaultSnakeEnvConfig({
      terrain: makeTerrain(),
      motor: { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm },
    });
    envCfg.episodeSteps = 1800;
    const env = await SnakeEnv.create(envCfg);
    const r1 = evalDet(env, policy);
    const r2 = evalDet(env, policy);
    const base = evalBase(env);
    const repro = Math.abs(r1.forwardM - r2.forwardM) < 1e-6 ? 'OK' : '不一致';
    const gainPct = base.forwardM !== 0 ? ((r1.forwardM - base.forwardM) / base.forwardM) * 100 : 0;
    const verdict = r1.forwardM > base.forwardM ? '上回る ✅' : '下回る ❌';
    console.log(
      `  [${name.padEnd(11)}] 基盤 ${(base.forwardM * 100).toFixed(0).padStart(4)}cm → RL ${(r1.forwardM * 100).toFixed(0).padStart(4)}cm ` +
        `(${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(0)}%) ${verdict} ｜ dy 基盤 ${(base.dyM * 100).toFixed(0)} / RL ${(r1.dyM * 100).toFixed(0)}cm（再現 ${repro}）`,
    );
    env.dispose();
  }

  policy.dispose();
}

await main();
