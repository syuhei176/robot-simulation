/// <reference types="node" />
/**
 * 3D 蛇（MuJoCo）の強化学習（Mac / Node オフライン）— 進行性地形の走破。
 *   実行例:
 *     node scripts/train-snake.ts --iters 60 --motor mg996r            # 進行性地形（障害物→階段→テーブル）
 *     node scripts/train-snake.ts --iters 60 --course flat             # 平地（比較・前進のみ）
 *
 * `SnakeEnv`（方策が n-1 関節の歩容残差を制御）を `Policy`+`PPO`（TF.js, CPU）で学習する。土台は前進登坂歩容
 * （alt-yaw-pitch・yaw前進＋pitch持ち上げ）＝**残差RL**。action=0 でも障害物＋階段を越えるので、方策は前方地形
 * プレビューを手がかりに「いつ・どの関節を余分に動かして段を越え・横ドリフトを抑えるか」を学ぶ。学習した方策の
 * 決定論ロールアウトを frames 記録して `public/policies/snake3d-<course>-<motor>.replay.json` に保存（ダッシュボード
 * の RL ボタンで TF.js 無し再生）。
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { getServo } from '../src/sim3d/servos.ts';
import { SnakeEnv, defaultSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');

interface Options {
  iters: number;
  rollout: number;
  motor: string;
  course: 'progression' | 'flat';
  episodeSteps: number;
  hidden: number;
  lr: number;
  entCoef: number;
  evalEvery: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    iters: 60,
    rollout: 2048,
    motor: 'mg996r',
    course: 'progression',
    episodeSteps: 1100, // 階段を登りきって踊り場→テーブルに到達できる長さ（基盤歩容で ~303cm）

    hidden: 64,
    lr: DEFAULT_PPO.lr,
    entCoef: DEFAULT_PPO.entCoef,
    evalEvery: 5,
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
      case 'motor':
        opts.motor = val;
        break;
      case 'course':
        opts.course = val as Options['course'];
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
      default:
        throw new Error(`未知のオプション --${key}`);
    }
  }
  return opts;
}

/** 学習済み方策を決定論（平均行動）で1エピソード走らせ、前進量を測る。record でフレーム記録。 */
function evaluate(
  env: SnakeEnv,
  policy: Policy,
  record = false,
): { forwardM: number; steps: number; fell: boolean } {
  let obs = env.reset();
  if (record) env.enableRecording();
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
    if (done && steps < env.maxSteps) fell = true; // 早期終了＝数値破綻
  }
  return { forwardM: env.progressMetric() - startX, steps, fell };
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const servo = getServo(opts.motor);
  const cap = servo.stallNm;

  // progression は既定の進行性地形、flat は地形なし。terrain:undefined を渡すと既定を消すので分岐する。
  const simOverrides: Parameters<typeof defaultSnakeEnvConfig>[0] = {
    motor: { stiffness: 3, damping: 0.15, maxTorqueNm: cap },
  };
  if (opts.course === 'flat') simOverrides.terrain = [];
  const envCfg = defaultSnakeEnvConfig(simOverrides);
  envCfg.episodeSteps = opts.episodeSteps;

  await tf.setBackend('cpu');
  await tf.ready();

  const env = await SnakeEnv.create(envCfg);
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
    `=== 3D 蛇 RL (PPO) === コース=${opts.course} / モーター=${servo.name} (cap ${cap.toFixed(3)} N·m)`,
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
    let evalNote = '';
    if ((i + 1) % opts.evalEvery === 0 || i === opts.iters - 1) {
      const ev = evaluate(env, policy);
      const det = ev.fell ? -1 : ev.forwardM;
      if (det > bestEvalForward) {
        bestEvalForward = det;
        bestWeights = policy.exportWeights();
      }
      evalNote = ` | det=${(ev.forwardM * 100).toFixed(1)}cm${ev.fell ? '(破綻)' : ''}`;
    }
    console.log(
      `  iter ${String(s.iteration).padStart(3)}: return=${s.meanEpisodeReturn.toFixed(3).padStart(8)} ` +
        `forward=${(s.meanEpisodeForward * 100).toFixed(1).padStart(7)}cm pLoss=${s.policyLoss.toFixed(4)} vLoss=${s.valueLoss.toFixed(3)} std=${s.std.toFixed(3)}${evalNote}`,
    );
  }

  policy.importWeights(bestWeights);
  const evalRes = evaluate(env, policy, true);
  const replay = env.getReplay();
  console.log('');
  console.log(
    `=== 結果 === 決定論評価: 前進 ${(evalRes.forwardM * 100).toFixed(1)}cm / ${evalRes.steps}ステップ / ${evalRes.fell ? '破綻' : '完走'}`,
  );

  // ---- 書き出し ----
  const meta = {
    mechanism: 'snake3d',
    course: opts.course,
    motor: servo.id,
    motorName: servo.name,
    mass: envCfg.sim.totalMass,
    base: 'climb-gait',
    forwardM: round4(evalRes.forwardM),
    fell: evalRes.fell,
  };
  mkdirSync(OUT_DIR, { recursive: true });
  const stem = `snake3d-${opts.course}-${servo.id}`;
  writeFileSync(
    join(OUT_DIR, `${stem}.json`),
    JSON.stringify({
      kind: 'rl-policy',
      ...meta,
      obsDim: env.obsDim,
      actDim: env.actDim,
      hidden: opts.hidden,
      weights: bestWeights,
      history,
      config: { iters: opts.iters, rollout: opts.rollout, lr: opts.lr },
    }) + '\n',
  );
  const round5 = (_k: string, v: unknown): unknown =>
    typeof v === 'number' ? Number(v.toFixed(5)) : v;
  writeFileSync(
    join(OUT_DIR, `${stem}.replay.json`),
    JSON.stringify({ meta, replay }, round5) + '\n',
  );
  rebuildPolicyManifest();
  console.log(`  書き出し: public/policies/${stem}.json + ${stem}.replay.json（+ manifest.json）`);

  env.dispose();
  policy.dispose();
  ppo.dispose();
}

function round4(x: number): number {
  return Number(x.toFixed(4));
}

/** public/policies/*.replay.json を走査して manifest.json（ダッシュボードの RL 再生一覧）を作り直す。 */
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
