/// <reference types="node" />
/**
 * 3D 四足の強化学習（Mac / Node オフライン）。
 *   実行例:
 *     node scripts/train-quad.ts --iters 50 --base-tuned auto       # 残差RL on CMA-ES tuned 土台（推奨）
 *     node scripts/train-quad.ts --iters 80                         # 残差RL on 参照歩容
 *     node scripts/train-quad.ts --iters 80 --base-gait false       # end-to-end（静止姿勢中心・比較用）
 *
 * `QuadEnv`（方策が 8 関節を制御）を `Policy`+`PPO`（TF.js, CPU）で学習する。既定は **残差RL**＝IK クロール
 * 歩容に方策の補正を上乗せ（`--base-tuned auto` で CMA-ES の速い tuned 歩容を土台にできる）。重い探索は
 * CLI 側で回し、学習した方策の重み＋メタを `public/policies/quad-<course>-<motor>.json` に保存する。
 * 既定は 150g・SCS0009 の小型四足を平地で歩かせるタスク。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { COURSES, type CourseId } from '../src/sim3d/course.ts';
import { getServo } from '../src/sim3d/servos.ts';
import { QuadEnv, DEFAULT_QUAD_ENV, type QuadEnvConfig } from '../src/env/QuadEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');
const TUNED_DIR = join(ROOT, 'public', 'tuned');

interface Options {
  iters: number;
  rollout: number;
  mass: number;
  motor: string;
  course: CourseId;
  episodeSteps: number;
  hidden: number;
  lr: number;
  entCoef: number; // PPO エントロピー係数（下げると平均方策がコミットして決定論再生が安定）
  evalEvery: number; // 何イテレーションごとに決定論評価してベスト方策を更新するか
  baseGait: boolean; // 残差RL（土台 IK 歩容に上乗せ）か、end-to-end（静止姿勢中心）か
  baseTuned: string | null; // 土台に使う tuned 歩容 JSON（'auto'=public/tuned/quad-<course>-<motor>.json）
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    iters: 80,
    rollout: 2048,
    mass: DEFAULT_QUAD_ENV.mass,
    motor: 'scs0009',
    course: 'flat',
    episodeSteps: DEFAULT_QUAD_ENV.episodeSteps,
    hidden: 64,
    lr: DEFAULT_PPO.lr,
    entCoef: DEFAULT_PPO.entCoef,
    evalEvery: 5,
    baseGait: true,
    baseTuned: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error(`--${key} に値がありません`);
    switch (key) {
      case 'iters':
        opts.iters = Number(val);
        break;
      case 'rollout':
        opts.rollout = Number(val);
        break;
      case 'mass':
        opts.mass = Number(val);
        break;
      case 'motor':
        opts.motor = val;
        break;
      case 'course':
        opts.course = val as CourseId;
        break;
      case 'episode-steps':
        opts.episodeSteps = Number(val);
        break;
      case 'hidden':
        opts.hidden = Number(val);
        break;
      case 'lr':
        opts.lr = Number(val);
        break;
      case 'ent-coef':
        opts.entCoef = Number(val);
        break;
      case 'eval-every':
        opts.evalEvery = Number(val);
        break;
      case 'base-gait':
        opts.baseGait = val !== 'false' && val !== '0';
        break;
      case 'base-tuned':
        opts.baseTuned = val;
        break;
      default:
        throw new Error(`未知のオプション --${key}`);
    }
  }
  return opts;
}

/** 学習済み方策を決定論（平均行動）で1エピソード走らせ、前進量と転倒を測る。 */
function evaluate(
  env: QuadEnv,
  policy: Policy,
): { forwardM: number; steps: number; fell: boolean } {
  let obs = env.reset();
  const startX = env.progressMetric();
  let steps = 0;
  let done = false;
  let fell = false;
  while (!done) {
    const action = policy.actMean(obs);
    const res = env.step(action);
    obs = res.obs;
    done = res.done;
    steps++;
    // 終端が時間切れでなく早期なら転倒とみなす。
    if (done && steps < env.maxSteps) fell = true;
  }
  return { forwardM: env.progressMetric() - startX, steps, fell };
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const servo = getServo(opts.motor);
  const cap = servo.stallNm;
  const course = COURSES[opts.course]();

  // 残差RL の土台に tuned 歩容を使う場合は CMA-ES の結果から歩容パラメータを読む。
  let gaitParams: QuadEnvConfig['gaitParams'];
  let baseLabel = '参照歩容';
  if (opts.baseTuned) {
    const file =
      opts.baseTuned === 'auto'
        ? join(TUNED_DIR, `quad-${opts.course}-${servo.id}.json`)
        : opts.baseTuned;
    const rec = JSON.parse(readFileSync(file, 'utf8')) as { params: Record<string, number> };
    const p = rec.params;
    gaitParams = {
      period: p.period,
      strideM: p.strideM,
      liftM: p.liftM,
      standM: p.standM,
      stanceDuty: p.stanceDuty,
    };
    baseLabel = `tuned歩容(${file.split('/').pop()})`;
  }

  await tf.setBackend('cpu');
  await tf.ready();

  const env = await QuadEnv.create({
    mass: opts.mass,
    torqueCapNm: cap,
    course,
    episodeSteps: opts.episodeSteps,
    baseGait: opts.baseGait,
    gaitParams,
  });
  const policy = new Policy(env.obsDim, env.actDim, opts.hidden);
  const ppo = new PPO(policy, {
    ...DEFAULT_PPO,
    rolloutSteps: opts.rollout,
    lr: opts.lr,
    entCoef: opts.entCoef,
    minibatch: 256,
    epochs: 4,
  });

  console.log(
    `=== 3D 四足 RL (PPO) === コース=${opts.course} / モーター=${servo.name} (cap ${cap.toFixed(3)} N·m) / 質量=${opts.mass}kg`,
  );
  console.log(
    `  方策=${opts.baseGait ? `残差RL（土台=${baseLabel}）` : 'end-to-end（静止姿勢中心）'}`,
  );
  console.log(
    `  obsDim=${env.obsDim} actDim=${env.actDim} hidden=${opts.hidden} / rollout=${opts.rollout} episodeSteps=${opts.episodeSteps} / tf=${tf.getBackend()}`,
  );

  interface HistRow {
    iter: number;
    return: number;
    forwardM: number;
    std: number;
  }
  const history: HistRow[] = [];
  // デプロイされるのは平均（決定論）方策。確率的ロールアウトの前進はノイズで嵩上げされるため、
  // ベストは「定期的な決定論評価で最も前進し転倒しない方策」で選ぶ（mean-vs-sample 乖離対策）。
  let bestEvalForward = -Infinity;
  let bestWeights = policy.exportWeights();

  for (let i = 0; i < opts.iters; i++) {
    const s = ppo.runIteration(env);
    history.push({
      iter: s.iteration,
      return: round4(s.meanEpisodeReturn),
      forwardM: round4(s.meanEpisodeForward),
      std: round4(s.std),
    });
    if (!Number.isFinite(s.policyLoss) || !Number.isFinite(s.valueLoss)) {
      throw new Error('損失が NaN/Inf になりました（学習発散）');
    }
    // 定期的に決定論評価し、ベスト方策を更新。
    let evalNote = '';
    if ((i + 1) % opts.evalEvery === 0 || i === opts.iters - 1) {
      const ev = evaluate(env, policy);
      const det = ev.fell ? -1 : ev.forwardM;
      if (det > bestEvalForward) {
        bestEvalForward = det;
        bestWeights = policy.exportWeights();
      }
      evalNote = ` | det=${(ev.forwardM * 100).toFixed(1)}cm${ev.fell ? '(転倒)' : ''}`;
    }
    console.log(
      `  iter ${String(s.iteration).padStart(3)}: return=${s.meanEpisodeReturn.toFixed(3).padStart(8)} ` +
        `forward=${(s.meanEpisodeForward * 100).toFixed(1).padStart(7)}cm pLoss=${s.policyLoss.toFixed(4)} vLoss=${s.valueLoss.toFixed(3)} std=${s.std.toFixed(3)}${evalNote}`,
    );
  }

  // ベスト（決定論で最良）重みで最終評価。
  policy.importWeights(bestWeights);
  const evalRes = evaluate(env, policy);
  console.log('');
  console.log(
    `=== 結果 === 決定論評価: 前進 ${(evalRes.forwardM * 100).toFixed(1)}cm / ${evalRes.steps}ステップ / ${evalRes.fell ? '転倒' : '転倒なし'}`,
  );

  // ---- 書き出し ----
  const record = {
    kind: 'rl-policy',
    mechanism: 'quad',
    course: opts.course,
    motor: servo.id,
    motorName: servo.name,
    mass: opts.mass,
    torqueCapNm: round4(cap),
    obsDim: env.obsDim,
    actDim: env.actDim,
    hidden: opts.hidden,
    baseGait: opts.baseGait,
    base: opts.baseGait ? baseLabel : 'stance',
    gaitParams,
    env: {
      episodeSteps: opts.episodeSteps,
      controlFrames: DEFAULT_QUAD_ENV.controlFrames,
      clockFreq: DEFAULT_QUAD_ENV.clockFreq,
      maxHipDelta: DEFAULT_QUAD_ENV.maxHipDelta,
      maxKneeDelta: DEFAULT_QUAD_ENV.maxKneeDelta,
      standFrac: DEFAULT_QUAD_ENV.standFrac,
    },
    eval: {
      forwardM: round4(evalRes.forwardM),
      steps: evalRes.steps,
      fell: evalRes.fell,
    },
    weights: bestWeights,
    history,
    config: { iters: opts.iters, rollout: opts.rollout, lr: opts.lr },
  };
  mkdirSync(OUT_DIR, { recursive: true });
  const fileName = `quad-${opts.course}-${servo.id}.json`;
  writeFileSync(join(OUT_DIR, fileName), JSON.stringify(record) + '\n');
  console.log(`  書き出し: public/policies/${fileName}`);

  env.dispose();
  policy.dispose();
  ppo.dispose();
}

function round4(x: number): number {
  return Number(x.toFixed(4));
}

await main();
