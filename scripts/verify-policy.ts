/// <reference types="node" />
/**
 * 保存済み RL 方策（public/policies/<stem>.json）を再ロードし、各コース×離散方位で決定論ロールアウトして
 * 「指令方向への前進」「達成方位が指令に追従（＝操舵できている）」「再現性」を基盤歩容と比較検証する。
 *
 *   node scripts/verify-policy.ts                              # 既定 snake3d-general-mg996r
 *   node scripts/verify-policy.ts --stem snake3d-general-mg996r
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { getServo } from '../src/sim3d/servos.ts';
import {
  makeProgressionTerrain,
  makeStraightChallengeTerrain,
  makeRoomWalls,
  roomTargetWall,
  type SnakeTerrainBox,
} from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, defaultSnakeEnvConfig, roomSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';

/** 検証に使う名前付きコース（汎用性を見るため複数で評価する）。 */
const NAMED_COURSES: Record<string, () => SnakeTerrainBox[]> = {
  flat: () => [],
  progression: makeProgressionTerrain,
  challenge: makeStraightChallengeTerrain,
};
const EVAL_HEADINGS_DEG = [-25, 0, 25] as const;
const ROOM_EVAL_HEADINGS_DEG = [-90, -45, 0, 45, 90] as const;
const DEG = Math.PI / 180;

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

interface PolicyFile {
  motor: string;
  course: string;
  obsDim: number;
  actDim: number;
  hidden: number;
  weights: { policy: number[][]; value: number[][]; logStd: number[] };
}

interface RolloutResult {
  forwardM: number;
  crossM: number;
  achievedDeg: number;
  reached: boolean;
  goalDistM: number;
}

function rollout(
  env: SnakeEnv,
  headingRad: number,
  act: (obs: Float32Array) => Float32Array,
): RolloutResult {
  let obs = env.reset(undefined, headingRad);
  const startProj = env.progressMetric();
  let done = false;
  while (!done) {
    const r = env.step(act(obs));
    obs = r.obs;
    done = r.done;
  }
  const summary = env.getReplay().summary;
  const [dx, dy] = summary.netDispM;
  return {
    forwardM: env.progressMetric() - startProj,
    crossM: Math.abs(-dx * Math.sin(headingRad) + dy * Math.cos(headingRad)),
    achievedDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
    reached: summary.reached ?? false,
    goalDistM: summary.goalDistM ?? 0,
  };
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
  const zeros = new Float32Array(file.actDim);

  console.log(`=== 検証: ${stem} === 学習コース=${file.course} / モーター=${servo.name}`);

  if (file.course === 'room') {
    // 部屋ナビ: 壁のみのクリーンな部屋で θ ごとに「到達・残距離・達成方位・再現性」を基盤と比較。
    console.log(`  部屋ナビ（壁のみ）で各方位の到達・残距離・達成方位・再現性を基盤(残差0)と比較:`);
    const envCfg = roomSnakeEnvConfig({
      terrain: makeRoomWalls(),
      motor: { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm },
    });
    const env = await SnakeEnv.create(envCfg);
    for (const deg of ROOM_EVAL_HEADINGS_DEG) {
      const rad = deg * DEG;
      const r1 = rollout(env, rad, (o) => policy.actMean(o));
      const r2 = rollout(env, rad, (o) => policy.actMean(o));
      const base = rollout(env, rad, () => zeros);
      const repro = Math.abs(r1.goalDistM - r2.goalDistM) < 1e-6 ? 'OK' : '不一致';
      const wall = roomTargetWall(env.startCom[0], env.startCom[1], rad).wall;
      console.log(
        `  [指令${String(deg).padStart(3)}°→${wall.padEnd(5)}] 基盤 ${base.reached ? '到達' : '未到達'}(残${(base.goalDistM * 100).toFixed(0)}cm) → ` +
          `RL ${r1.reached ? '到達✓' : '未到達'}(残${(r1.goalDistM * 100).toFixed(0)}cm) 達成方位${r1.achievedDeg.toFixed(0)}°（再現${repro}）`,
      );
    }
    env.dispose();
    policy.dispose();
    return;
  }

  console.log(`  各コース×方位で「指令方向前進・達成方位の追従・再現性」を基盤(残差0)と比較:`);

  for (const [name, makeTerrain] of Object.entries(NAMED_COURSES)) {
    const envCfg = defaultSnakeEnvConfig({
      terrain: makeTerrain(),
      motor: { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm },
    });
    envCfg.episodeSteps = 1800;
    const env = await SnakeEnv.create(envCfg);
    for (const deg of EVAL_HEADINGS_DEG) {
      const rad = deg * DEG;
      const r1 = rollout(env, rad, (o) => policy.actMean(o));
      const r2 = rollout(env, rad, (o) => policy.actMean(o));
      const base = rollout(env, rad, () => zeros);
      const repro = Math.abs(r1.forwardM - r2.forwardM) < 1e-6 ? 'OK' : '不一致';
      const gainPct =
        base.forwardM !== 0 ? ((r1.forwardM - base.forwardM) / base.forwardM) * 100 : 0;
      const steer =
        Math.abs(r1.achievedDeg - deg) < Math.abs(base.achievedDeg - deg) ? '操舵✅' : '—';
      console.log(
        `  [${name.padEnd(11)} 指令${String(deg).padStart(3)}°] 基盤 ${(base.forwardM * 100).toFixed(0).padStart(4)}cm(方位${base.achievedDeg.toFixed(0)}°) → ` +
          `RL ${(r1.forwardM * 100).toFixed(0).padStart(4)}cm(方位${r1.achievedDeg.toFixed(0)}°/${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(0)}%) ` +
          `横ずれ ${(r1.crossM * 100).toFixed(0)}cm ${steer}（再現${repro}）`,
      );
    }
    env.dispose();
  }

  policy.dispose();
}

await main();
