/// <reference types="node" />
/**
 * 部屋ナビ（S17）の教師-生徒蒸留（Mac / Node オフライン）。
 *
 * 問題: センサーのみ観測（関節＋頭IMU＋目標ヘディング誤差）で目標方位へ操舵する方策を直接 PPO で学ぶと、
 * 指令に追従せず固定 veer に collapse する（S16 から続く難問）。観測が「自分が今どっちを向いているか」を
 * 厳密には持たないため。
 *
 * 解法（teacher-student distillation）:
 *  1) 教師 = 特権観測（ゴールへの真の方位 sin/cos ＋ 壁距離）を足した方策を PPO で学ぶ。方位が自明なので
 *     クリーンに操舵を獲得する。
 *  2) 生徒 = センサーのみ 56 次元の方策。教師をロールアウトしながら DAgger（生徒が訪れた状態を教師がラベル）で
 *     教師の行動を behavioral cloning する。生徒は obs[:56]（＝特権の先頭56次元＝センサーと完全一致）だけ見る。
 *  3) 生徒の重み（センサーのみ・ブラウザ互換）を public/policies/snake3d-room-<motor>.json に保存し、各方位の
 *     replay＋俯瞰SVG を書き出す（ダッシュボード/検証はこの生徒を使う）。
 *
 *   node scripts/distill-snake.ts --motor mg996r
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { getServo } from '../src/sim3d/servos.ts';
import { makeRoomBank, makeRoomWalls, roomTargetWall } from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, roomSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';
import { writeRoomTraceSvg } from './room-trace.ts';

const DEG = Math.PI / 180;
const ROOM_EVAL_HEADINGS_DEG = [-90, -45, 0, 45, 90] as const;

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');

// 教師 PPO の探索 std アニーリング（train-snake と同じ方針）。
const LOGSTD_START = -0.9;
const LOGSTD_END = -2.3;
function annealLogStd(i: number, iters: number): number {
  const frac = Math.max(0, Math.min(1, (i / iters - 0.3) / 0.55));
  return LOGSTD_START + (LOGSTD_END - LOGSTD_START) * frac;
}

function headTag(deg: number): string {
  if (deg === 0) return 'h0';
  return deg > 0 ? `hp${deg}` : `hm${-deg}`;
}

const round4 = (x: number): number => Number(x.toFixed(4));
const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;

interface Options {
  motor: string;
  teacherIters: number;
  daggerIters: number;
  episodesPerDagger: number;
  bcEpochs: number;
  hidden: number;
}

function parseArgs(argv: string[]): Options {
  const o: Options = {
    motor: 'mg996r',
    teacherIters: 60,
    daggerIters: 4,
    episodesPerDagger: 12,
    bcEpochs: 6,
    hidden: 96,
  };
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const key = argv[i].slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error(`--${key} に値がありません`);
    if (key === 'motor') o.motor = val;
    else if (key === 'teacher-iters') o.teacherIters = Number(val);
    else if (key === 'dagger-iters') o.daggerIters = Number(val);
    else if (key === 'episodes-per-dagger') o.episodesPerDagger = Number(val);
    else if (key === 'bc-epochs') o.bcEpochs = Number(val);
    else if (key === 'hidden') o.hidden = Number(val);
    else throw new Error(`未知のオプション --${key}`);
  }
  return o;
}

interface EvalOut {
  reached: boolean;
  goalDistM: number;
  achievedDeg: number;
}

/** env で act(obs) を決定論ロールアウト。obsSlice で生徒に渡す観測を切り出す（特権の先頭56次元）。 */
function rollout(
  env: SnakeEnv,
  headingRad: number,
  act: (obs: Float32Array) => Float32Array,
  obsSlice: number,
  record = false,
): EvalOut {
  let obs = env.reset(undefined, headingRad);
  if (record) env.enableRecording();
  let done = false;
  while (!done) {
    const a = act(obsSlice > 0 ? (obs.subarray(0, obsSlice) as Float32Array) : obs);
    const r = env.step(a);
    obs = r.obs;
    done = r.done;
  }
  const s = env.getReplay().summary;
  const [dx, dy] = s.netDispM;
  return {
    reached: s.reached ?? false,
    goalDistM: s.goalDistM ?? 0,
    achievedDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
  };
}

/** 各方位で到達率を測る（簡易・選択用）。obsSlice>0 で生徒、=0 で教師（フル観測）。 */
function reachedRate(
  env: SnakeEnv,
  act: (obs: Float32Array) => Float32Array,
  obsSlice: number,
): { count: number; meanDist: number } {
  let count = 0;
  let dist = 0;
  for (const deg of ROOM_EVAL_HEADINGS_DEG) {
    const r = rollout(env, deg * DEG, act, obsSlice);
    if (r.reached) count++;
    dist += r.goalDistM;
  }
  return { count, meanDist: dist / ROOM_EVAL_HEADINGS_DEG.length };
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const servo = getServo(opts.motor);
  const motor = { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm };

  await tf.setBackend('cpu');
  await tf.ready();

  // ---- 特権 env（教師の学習・ロールアウト用。obs = sensor56 + [ゴール方位 sin/cos, 壁距離/6]）----
  const trainCfg = roomSnakeEnvConfig({ motor });
  trainCfg.privilegedObs = true;
  trainCfg.terrainBank = makeRoomBank();
  const trainEnv = await SnakeEnv.create(trainCfg);

  const evalCfg = roomSnakeEnvConfig({ terrain: makeRoomWalls(), motor });
  evalCfg.privilegedObs = true;
  const evalEnv = await SnakeEnv.create(evalCfg);

  const privObsDim = trainEnv.obsDim;
  const sensorDim = privObsDim - 3; // 特権は末尾3次元のみ。先頭 sensorDim がセンサーのみ生徒の観測。
  const actDim = trainEnv.actDim;

  console.log(`=== 部屋ナビ 教師-生徒蒸留 === モーター=${servo.name}`);
  console.log(
    `  特権obs=${privObsDim} センサーobs=${sensorDim} actDim=${actDim} hidden=${opts.hidden}`,
  );

  // ================= 1) 教師を PPO で学習（特権観測でクリーン操舵） =================
  const teacher = new Policy(privObsDim, actDim, opts.hidden, LOGSTD_START);
  const ppo = new PPO(teacher, {
    ...DEFAULT_PPO,
    rolloutSteps: 2048,
    lr: 1.5e-4,
    entCoef: 0.001,
    minibatch: 256,
    epochs: 4,
  });
  console.log(`\n--- 教師 PPO (${opts.teacherIters} iter) ---`);
  let teacherBest = teacher.exportWeights();
  let teacherBestScore = -Infinity;
  for (let i = 0; i < opts.teacherIters; i++) {
    teacher.setLogStd(annealLogStd(i, opts.teacherIters));
    const s = ppo.runIteration(trainEnv);
    if (!Number.isFinite(s.policyLoss)) throw new Error('教師の損失が NaN（発散）');
    if ((i + 1) % 5 === 0 || i === opts.teacherIters - 1) {
      const { count, meanDist } = reachedRate(evalEnv, (o) => teacher.actMean(o), 0);
      const score = count - meanDist; // 到達数優先・残距離で微調整
      if (score > teacherBestScore) {
        teacherBestScore = score;
        teacherBest = teacher.exportWeights();
      }
      ppo.resetRollout();
      console.log(
        `  iter ${String(s.iteration).padStart(3)}: return=${s.meanEpisodeReturn.toFixed(1).padStart(7)} 到達 ${count}/5 残距離 ${(meanDist * 100).toFixed(0)}cm`,
      );
    }
  }
  teacher.importWeights(teacherBest);
  const tEval = reachedRate(evalEnv, (o) => teacher.actMean(o), 0);
  console.log(`  教師ベスト: 到達 ${tEval.count}/5 残距離 ${(tEval.meanDist * 100).toFixed(0)}cm`);

  // ================= 2) DAgger で生徒へ蒸留（センサーのみ） =================
  const student = new Policy(sensorDim, actDim, opts.hidden);
  const optimizer = tf.train.adam(1e-3);
  const dataX: Float32Array[] = []; // 生徒観測（sensorDim）
  const dataY: Float32Array[] = []; // 教師の平均行動（actDim）
  console.log(`\n--- DAgger 蒸留 (${opts.daggerIters} 周) ---`);

  for (let d = 0; d < opts.daggerIters; d++) {
    // データ収集: 周0 は教師が運転（教師分布）、以降は生徒が運転（生徒が訪れた状態を教師がラベル＝分布シフト矯正）。
    const driveByStudent = d > 0;
    for (let ep = 0; ep < opts.episodesPerDagger; ep++) {
      let obs = trainEnv.reset(); // ランダム θ（straightProb）＋ランダム地形
      let done = false;
      while (!done) {
        const teacherAct = teacher.actMean(obs);
        dataX.push((obs.subarray(0, sensorDim) as Float32Array).slice());
        dataY.push(teacherAct.slice());
        const stepAct = driveByStudent
          ? student.actMean(obs.subarray(0, sensorDim) as Float32Array)
          : teacherAct;
        const r = trainEnv.step(stepAct);
        obs = r.obs;
        done = r.done;
      }
    }

    // BC 学習（集めた全データを shuffle してミニバッチ MSE）。
    const N = dataX.length;
    const idx = Array.from({ length: N }, (_, i) => i);
    const bs = 256;
    let lastLoss = 0;
    for (let epoch = 0; epoch < opts.bcEpochs; epoch++) {
      for (let i = N - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1)); // Fisher-Yates シャッフル
        const t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
      }
      for (let b = 0; b < N; b += bs) {
        const end = Math.min(N, b + bs);
        const m = end - b;
        const xb = new Float32Array(m * sensorDim);
        const yb = new Float32Array(m * actDim);
        for (let k = 0; k < m; k++) {
          xb.set(dataX[idx[b + k]], k * sensorDim);
          yb.set(dataY[idx[b + k]], k * actDim);
        }
        const xt = tf.tensor2d(xb, [m, sensorDim]);
        const yt = tf.tensor2d(yb, [m, actDim]);
        lastLoss = student.bcStep(xt, yt, optimizer);
        xt.dispose();
        yt.dispose();
      }
    }
    const se = reachedRate(evalEnv, (o) => student.actMean(o), sensorDim);
    console.log(
      `  周${d + 1}: データ ${N} / BC損失 ${lastLoss.toFixed(4)} → 生徒 到達 ${se.count}/5 残距離 ${(se.meanDist * 100).toFixed(0)}cm`,
    );
  }

  // ================= 3) 生徒の成果物を書き出し（センサーのみ・ブラウザ互換） =================
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('\n=== 生徒（センサーのみ）決定論評価・各方位 ===');
  for (const deg of ROOM_EVAL_HEADINGS_DEG) {
    const rad = deg * DEG;
    const res = rollout(evalEnv, rad, (o) => student.actMean(o), sensorDim, true);
    const replay = evalEnv.getReplay();
    const wall = roomTargetWall(evalEnv.startCom[0], evalEnv.startCom[1], rad).wall;
    console.log(
      `  [${String(deg).padStart(3)}°→${wall.padEnd(5)}] ${res.reached ? '到達✓' : '未到達'} 残距離 ${(res.goalDistM * 100).toFixed(0)}cm 達成方位 ${res.achievedDeg.toFixed(0)}°`,
    );
    writeRoomTraceSvg(
      join(OUT_DIR, `room-trace-${headTag(deg)}-${servo.id}.svg`),
      evalEnv.getActiveTerrain(),
      evalEnv.startCom,
      deg,
      replay,
      res.reached,
    );
    const meta = {
      mechanism: 'snake3d',
      course: 'room',
      cmdHeadingDeg: deg,
      motor: servo.id,
      motorName: servo.name,
      mass: trainCfg.sim.totalMass,
      base: 'climb-gait',
      policy: 'room',
      reached: res.reached,
      goalDistM: round4(res.goalDistM),
      achievedDeg: round4(res.achievedDeg),
    };
    writeFileSync(
      join(OUT_DIR, `snake3d-room-${headTag(deg)}-${servo.id}.replay.json`),
      JSON.stringify({ meta, replay }, round5) + '\n',
    );
  }

  // 生徒の重み（センサーのみ 56 次元・ライブ駆動に使う単一方策）。
  writeFileSync(
    join(OUT_DIR, `snake3d-room-${servo.id}.json`),
    JSON.stringify({
      kind: 'rl-policy',
      mechanism: 'snake3d',
      course: 'room',
      motor: servo.id,
      motorName: servo.name,
      obsDim: sensorDim,
      actDim,
      hidden: opts.hidden,
      headingMaxDeg: trainCfg.headingMaxRad / DEG,
      distilled: true,
      weights: student.exportWeights(),
    }) + '\n',
  );
  rebuildPolicyManifest();
  console.log(
    `\n  書き出し: snake3d-room-h<deg>-${servo.id}.replay.json ＋ snake3d-room-${servo.id}.json（+ manifest）`,
  );

  trainEnv.dispose();
  evalEnv.dispose();
  teacher.dispose();
  student.dispose();
  ppo.dispose();
}

/** public/policies/*.replay.json を走査して manifest.json を作り直す。 */
function rebuildPolicyManifest(): void {
  const entries = readdirSync(OUT_DIR)
    .filter((f) => f.endsWith('.replay.json'))
    .map((file) => {
      const rec = JSON.parse(readFileSync(join(OUT_DIR, file), 'utf8')) as {
        meta: Record<string, unknown>;
      };
      return { file, ...rec.meta };
    });
  writeFileSync(join(OUT_DIR, 'manifest.json'), JSON.stringify(entries, null, 2) + '\n');
}

await main();
