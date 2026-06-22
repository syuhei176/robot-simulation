/// <reference types="node" />
/**
 * 3D 蛇（MuJoCo）の強化学習（Mac / Node オフライン）— 進行性地形の走破。
 *   実行例:
 *     node scripts/train-snake.ts --iters 60 --motor mg996r                          # 進行性地形（障害物→階段→テーブル壁）
 *     node scripts/train-snake.ts --iters 80 --course challenge --episode-steps 1800 # 壁なし直進チャレンジ（基盤超え用）
 *     node scripts/train-snake.ts --iters 60 --course flat                           # 平地（比較・前進のみ）
 *
 * challenge コースは前進 x を壁でキャップしない。基盤歩容は地形に進行方向を蹴られて斜行し前進を浪費するので、
 * 横オフセット＋体軸ヘディング観測を使って +x へ操舵し直す閉ループ方策＝RL が基盤の前進量を明確に上回れる。
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
import {
  makeProgressionTerrain,
  makeStraightChallengeTerrain,
  makeCourseBank,
  type SnakeTerrainBox,
} from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, defaultSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';

// 探索 std のアニーリング: 前半は探索を保ち、後半で縮小して決定論（mean）に性能を担わせる。
// （std 固定だと方策が探索ノイズに依存し、確率的ロールアウトは良いのに det 評価が低く荒れる。）
const LOGSTD_START = -0.9; // std≈0.41
const LOGSTD_END = -2.3; // std≈0.10
function annealLogStd(i: number, iters: number): number {
  const frac = clamp((i / iters - 0.3) / 0.55, 0, 1); // 30%まで探索→85%で下限
  return LOGSTD_START + (LOGSTD_END - LOGSTD_START) * frac;
}

/** 汎用性の評価・録画に使う名前付きコース（このどれでも基盤を下回らない単一方策を目指す）。 */
const EVAL_COURSES: Array<{ name: string; terrain: () => SnakeTerrainBox[] }> = [
  { name: 'flat', terrain: () => [] },
  { name: 'progression', terrain: makeProgressionTerrain },
  { name: 'challenge', terrain: makeStraightChallengeTerrain },
];

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');

interface Options {
  iters: number;
  rollout: number;
  motor: string;
  course: 'progression' | 'flat' | 'challenge' | 'general';
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
    // 残差RLは報酬の振れ幅が大きく PPO が崩れやすい。lr を下げ entCoef を絞り（std 膨張を抑える）、
    // 勾配ノルムクリップ（PPO 側）＋初期 std 低下と併せて安定収束させる。
    lr: 1.5e-4,
    entCoef: 0.001,
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

/** 基盤歩容（残差=0）を1エピソード走らせ、前進量と正味横ドリフトを測る（RL の比較対象）。 */
function evaluateBase(env: SnakeEnv): { forwardM: number; dyM: number } {
  env.reset();
  const startX = env.progressMetric();
  const zeros = new Float32Array(env.actDim);
  let done = false;
  while (!done) done = env.step(zeros).done;
  return { forwardM: env.progressMetric() - startX, dyM: env.getReplay().summary.netDispM[1] };
}

const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;

/**
 * コース汎用な単一方策を学習する。訓練 env は**地形バンク（ドメインランダム化）**で reset 毎にコースが変わる。
 * 評価は固定地形の名前付きコース（flat/progression/challenge）で行い、**どのコースでも基盤を下回らない**よう
 * 「コース横断の最小改善率」でベスト方策を選ぶ。最後に各コースで決定論ロールアウトを録画し、コースごとの
 * replay（同一の汎用方策）を保存する＝ダッシュボードでコースを切り替えても同じ方策が再生される。
 */
async function trainGeneral(
  opts: Options,
  servo: ReturnType<typeof getServo>,
  cap: number,
): Promise<void> {
  const motorOverride = { stiffness: 3, damping: 0.15, maxTorqueNm: cap };

  // 訓練 env: 地形バンク（平地＋進行性＋直進チャレンジ＋ランダム変種）でランダム化。
  const trainCfg = defaultSnakeEnvConfig({ motor: motorOverride });
  trainCfg.episodeSteps = opts.episodeSteps;
  trainCfg.terrainBank = makeCourseBank();
  const trainEnv = await SnakeEnv.create(trainCfg);

  // 評価 env: 各名前付きコースを固定地形で（訓練 env とは別インスタンス＝評価が学習ロールアウトを汚さない）。
  const evalEnvs: Array<{ name: string; env: SnakeEnv; base: number }> = [];
  for (const c of EVAL_COURSES) {
    const cfg = defaultSnakeEnvConfig({ terrain: c.terrain(), motor: motorOverride });
    cfg.episodeSteps = opts.episodeSteps;
    const env = await SnakeEnv.create(cfg);
    evalEnvs.push({ name: c.name, env, base: evaluateBase(env).forwardM });
  }

  const policy = new Policy(trainEnv.obsDim, trainEnv.actDim, opts.hidden, LOGSTD_START);
  const ppo = new PPO(policy, {
    ...DEFAULT_PPO,
    rolloutSteps: opts.rollout,
    lr: opts.lr,
    entCoef: opts.entCoef,
    minibatch: 256,
    epochs: 4,
  });

  console.log(
    `=== 3D 蛇 RL (PPO) === コース=general（ドメインランダム化 ${trainCfg.terrainBank.length}地形）/ モーター=${servo.name} (cap ${cap.toFixed(3)} N·m)`,
  );
  console.log(
    `  obsDim=${trainEnv.obsDim} actDim=${trainEnv.actDim} hidden=${opts.hidden} / rollout=${opts.rollout} episodeSteps=${opts.episodeSteps} / tf=${tf.getBackend()}`,
  );
  console.log(
    `  基盤(残差0): ` +
      evalEnvs.map((e) => `${e.name} ${(e.base * 100).toFixed(0)}cm`).join(' / ') +
      ' ← どのコースでも下回らない汎用方策を目指す',
  );

  const history: Array<{ iter: number; return: number; forwardM: number; std: number }> = [];
  let bestScore = -Infinity;
  let bestWeights = policy.exportWeights();

  for (let i = 0; i < opts.iters; i++) {
    policy.setLogStd(annealLogStd(i, opts.iters));
    const s = ppo.runIteration(trainEnv);
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
      const fwds = evalEnvs.map((e) => {
        const ev = evaluate(e.env, policy);
        return ev.fell ? 0 : ev.forwardM;
      });
      const ratios = fwds.map((f, k) => (evalEnvs[k].base > 0 ? f / evalEnvs[k].base : f));
      // 最小改善率（最悪コースを下回らない）を主、平均でタイブレーク＝どのコースも壊さない方策を選ぶ。
      const score =
        Math.min(...ratios) + 0.05 * (ratios.reduce((a, b) => a + b, 0) / ratios.length);
      if (score > bestScore) {
        bestScore = score;
        bestWeights = policy.exportWeights();
      }
      evalNote =
        ' | ' + evalEnvs.map((e, k) => `${e.name}=${(fwds[k] * 100).toFixed(0)}`).join(' ');
    }
    console.log(
      `  iter ${String(s.iteration).padStart(3)}: return=${s.meanEpisodeReturn.toFixed(2).padStart(8)} ` +
        `forward=${(s.meanEpisodeForward * 100).toFixed(0).padStart(5)}cm vLoss=${s.valueLoss.toFixed(3)} std=${s.std.toFixed(3)}${evalNote}`,
    );
  }

  // ---- 各コースで決定論ロールアウトを録画して保存（同一の汎用方策） ----
  policy.importWeights(bestWeights);
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('\n=== 結果（汎用方策・決定論評価） ===');
  for (const e of evalEnvs) {
    const res = evaluate(e.env, policy, true);
    const replay = e.env.getReplay();
    const gainPct = e.base > 0 ? ((res.forwardM - e.base) / e.base) * 100 : 0;
    console.log(
      `  [${e.name.padEnd(11)}] 基盤 ${(e.base * 100).toFixed(0)}cm → RL ${(res.forwardM * 100).toFixed(0)}cm ` +
        `(${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(0)}%) ${res.fell ? '破綻' : '完走'}`,
    );
    const meta = {
      mechanism: 'snake3d',
      course: e.name,
      motor: servo.id,
      motorName: servo.name,
      mass: trainCfg.sim.totalMass,
      base: 'climb-gait',
      policy: 'general', // 単一の汎用方策（全コース共通）の録画であることを示す
      forwardM: round4(res.forwardM),
      baseForwardM: round4(e.base),
      fell: res.fell,
    };
    writeFileSync(
      join(OUT_DIR, `snake3d-${e.name}-${servo.id}.replay.json`),
      JSON.stringify({ meta, replay }, round5) + '\n',
    );
  }

  // 汎用方策の重み（全コース共通）を1ファイルに保存。
  writeFileSync(
    join(OUT_DIR, `snake3d-general-${servo.id}.json`),
    JSON.stringify({
      kind: 'rl-policy',
      mechanism: 'snake3d',
      course: 'general',
      motor: servo.id,
      motorName: servo.name,
      obsDim: trainEnv.obsDim,
      actDim: trainEnv.actDim,
      hidden: opts.hidden,
      weights: bestWeights,
      history,
      config: {
        iters: opts.iters,
        rollout: opts.rollout,
        lr: opts.lr,
        bank: trainCfg.terrainBank.length,
      },
    }) + '\n',
  );
  rebuildPolicyManifest();
  console.log(
    `\n  書き出し: 各コース snake3d-<course>-${servo.id}.replay.json（同一汎用方策）＋ snake3d-general-${servo.id}.json（+ manifest）`,
  );

  trainEnv.dispose();
  for (const e of evalEnvs) e.env.dispose();
  policy.dispose();
  ppo.dispose();
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const servo = getServo(opts.motor);
  const cap = servo.stallNm;

  await tf.setBackend('cpu');
  await tf.ready();

  if (opts.course === 'general') {
    await trainGeneral(opts, servo, cap);
    return;
  }

  // progression は既定の進行性地形、flat は地形なし。terrain:undefined を渡すと既定を消すので分岐する。
  const simOverrides: Parameters<typeof defaultSnakeEnvConfig>[0] = {
    motor: { stiffness: 3, damping: 0.15, maxTorqueNm: cap },
  };
  if (opts.course === 'flat') simOverrides.terrain = [];
  else if (opts.course === 'challenge') simOverrides.terrain = makeStraightChallengeTerrain();
  const envCfg = defaultSnakeEnvConfig(simOverrides);
  envCfg.episodeSteps = opts.episodeSteps;

  const env = await SnakeEnv.create(envCfg);
  const policy = new Policy(env.obsDim, env.actDim, opts.hidden, LOGSTD_START);
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

  // 基盤歩容（残差=0・開ループ）の前進量＝RL が超えるべき基準線。
  const base = evaluateBase(env);
  console.log(
    `  基盤歩容（残差0）: 前進 ${(base.forwardM * 100).toFixed(1)}cm / 横ドリフト dy=${(base.dyM * 100).toFixed(1)}cm ← これを超える`,
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
    policy.setLogStd(annealLogStd(i, opts.iters)); // 探索 std をスケジュールに沿って縮小
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
      ppo.resetRollout(); // 評価で env を reset したので学習ロールアウトの継続状態を破棄（整合）
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
  const gainCm = (evalRes.forwardM - base.forwardM) * 100;
  const gainPct = base.forwardM > 0 ? (gainCm / (base.forwardM * 100)) * 100 : 0;
  console.log('');
  console.log(
    `=== 結果 === 決定論評価: 前進 ${(evalRes.forwardM * 100).toFixed(1)}cm / ${evalRes.steps}ステップ / ${evalRes.fell ? '破綻' : '完走'}`,
  );
  console.log(
    `  基盤 ${(base.forwardM * 100).toFixed(1)}cm → RL ${(evalRes.forwardM * 100).toFixed(1)}cm ` +
      `（${gainCm >= 0 ? '+' : ''}${gainCm.toFixed(1)}cm / ${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(0)}%）` +
      `／RL 横ドリフト dy=${(replay.summary.netDispM[1] * 100).toFixed(1)}cm（基盤 ${(base.dyM * 100).toFixed(1)}cm）`,
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
    baseForwardM: round4(base.forwardM), // 基盤歩容（残差0）の前進量＝RL が超えた基準線
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

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
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
