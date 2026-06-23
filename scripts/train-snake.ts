/// <reference types="node" />
/**
 * 3D 蛇（MuJoCo）の強化学習（Mac / Node オフライン）— 実機センサー観測で「任意方位へ操舵しながら地形を走る」汎用方策。
 *   実行例:
 *     node scripts/train-snake.ts --course general --motor mg996r --iters 80 --episode-steps 1800  # コース汎用＋操舵
 *     node scripts/train-snake.ts --course challenge --motor mg996r --iters 60                      # 単一コース（デバッグ）
 *
 * `SnakeEnv`（方策が n-1 関節の歩容残差を制御）を `Policy`+`PPO`（TF.js, CPU）で学習する。土台は前進登坂歩容
 * （alt-yaw-pitch・yaw前進＋pitch持ち上げ）＝**残差RL**。観測は実機相当（サーボ present position/velocity/load
 * ＋頭IMU）で、目的地は**目標ヘディング指令（相対方位）**で条件付け。エピソード毎に地形（バンク）と目標方位
 * （±headingMaxRad）をドメインランダム化し、報酬は「指令方向への前進 − 指令光線からの横ずれ − …」。
 *
 * 学習した方策の決定論ロールアウトを、各コース×離散方位 {−25°,0°,+25°} で frames 記録して
 * `public/policies/snake3d-<course>-h<deg>-<motor>.replay.json` に保存（ダッシュボードの方位スライダーで再生）。
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
  makeRoomBank,
  makeRoomWalls,
  roomTargetWall,
  type SnakeTerrainBox,
} from '../src/sim3d/snake3d-dynamics.ts';
import {
  SnakeEnv,
  defaultSnakeEnvConfig,
  roomSnakeEnvConfig,
  type SnakeEnvConfig,
} from '../src/env/SnakeEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';
import { writeRoomTraceSvg } from './room-trace.ts';

// 探索 std のアニーリング: 前半は探索を保ち、後半で縮小して決定論（mean）に性能を担わせる。
const LOGSTD_START = -0.9; // std≈0.41
const LOGSTD_END = -2.3; // std≈0.10
function annealLogStd(i: number, iters: number): number {
  const frac = clamp((i / iters - 0.3) / 0.55, 0, 1); // 30%まで探索→85%で下限
  return LOGSTD_START + (LOGSTD_END - LOGSTD_START) * frac;
}

/** 評価・録画に使う離散方位 [deg]（操舵を確認できる左/直進/右）。+x コース用。 */
const EVAL_HEADINGS_DEG = [-25, 0, 25] as const;
/** 部屋ナビ課題の評価方位 [deg]（±90° まで・曲がれる範囲）。 */
const ROOM_EVAL_HEADINGS_DEG = [-90, -45, 0, 45, 90] as const;
const DEG = Math.PI / 180;

/** 録画ファイル名の方位タグ（h0 / hp25 / hm25）。 */
function headTag(deg: number): string {
  if (deg === 0) return 'h0';
  return deg > 0 ? `hp${deg}` : `hm${-deg}`;
}

/** 汎用性の評価・録画に使う名前付きコース（このどれでも・どの方位でも操舵して走る単一方策を目指す）。 */
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
  course: 'progression' | 'flat' | 'challenge' | 'general' | 'room';
  episodeSteps: number;
  hidden: number;
  lr: number;
  entCoef: number;
  evalEvery: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    iters: 80,
    rollout: 2048,
    motor: 'mg996r',
    course: 'general',
    episodeSteps: 1800, // 段を登りきり踊り場を遠くまで走る長さ（斜め操舵でも地形上に乗る）
    hidden: 96, // 観測が実機センサーで増えた（56次元）分の容量
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

interface EvalResult {
  forwardM: number; // 指令方向への前進量 [m]
  crossM: number; // 指令光線からの最終横ずれ [m]（操舵追従の誤差）
  achievedDeg: number; // 達成方位 [deg]
  steps: number;
  fell: boolean;
  reached: boolean; // 部屋ナビ: ゴール壁に到達したか
  goalDistM: number; // 部屋ナビ: 終端の目標壁までの距離 [m]
}

/** 指令方位 headingRad で決定論（平均行動）ロールアウト。record でフレーム記録。 */
function evaluate(env: SnakeEnv, policy: Policy, headingRad: number, record = false): EvalResult {
  let obs = env.reset(undefined, headingRad);
  if (record) env.enableRecording();
  const startProj = env.progressMetric();
  let steps = 0;
  let done = false;
  while (!done) {
    const r = env.step(policy.actMean(obs));
    obs = r.obs;
    done = r.done;
    steps++;
  }
  const summary = env.getReplay().summary;
  // 早期終了＝数値破綻。ただしゴール到達による早期 done は破綻ではない（部屋ナビ）。
  const fell = done && steps < env.maxSteps && !(summary.reached ?? false);
  const [dx, dy] = summary.netDispM;
  const crossM = Math.abs(-dx * Math.sin(headingRad) + dy * Math.cos(headingRad));
  return {
    forwardM: env.progressMetric() - startProj,
    crossM,
    achievedDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
    steps,
    fell,
    reached: summary.reached ?? false,
    goalDistM: summary.goalDistM ?? 0,
  };
}

/** 基盤歩容（残差=0）を指令方位で評価（操舵不可なので非0方位では指令方向前進が落ちる＝RL の上乗せが出る）。 */
function evaluateBase(env: SnakeEnv, headingRad: number): EvalResult {
  env.reset(undefined, headingRad);
  const startProj = env.progressMetric();
  const zeros = new Float32Array(env.actDim);
  let done = false;
  let steps = 0;
  while (!done) {
    done = env.step(zeros).done;
    steps++;
  }
  const summary = env.getReplay().summary;
  const [dx, dy] = summary.netDispM;
  const crossM = Math.abs(-dx * Math.sin(headingRad) + dy * Math.cos(headingRad));
  return {
    forwardM: env.progressMetric() - startProj,
    crossM,
    achievedDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
    steps,
    fell: false,
    reached: summary.reached ?? false,
    goalDistM: summary.goalDistM ?? 0,
  };
}

const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;

/**
 * 各コース×離散方位で eval 平均の選択スコア。
 * 部屋ナビ: 到達=+2・目標壁までの距離を減点（到達率優先）。+x コース: 前進 − 追従誤差。
 */
function selectionScore(
  evalEnvs: Array<{ name: string; env: SnakeEnv }>,
  policy: Policy,
  headingsDeg: readonly number[],
  room: boolean,
): { score: number; note: string } {
  let sum = 0;
  let count = 0;
  let reachedCount = 0;
  const parts: string[] = [];
  for (const e of evalEnvs) {
    for (const deg of headingsDeg) {
      const r = evaluate(e.env, policy, deg * DEG);
      let v: number;
      if (room) {
        v = r.fell ? -1 : (r.reached ? 2 : 0) - r.goalDistM / 3; // 到達優先＋残距離を減点
        if (r.reached) reachedCount++;
      } else {
        v = r.fell ? -1 : r.forwardM - r.crossM; // 前進 − 追従誤差。破綻は強く減点
        if (deg === 0) parts.push(`${e.name}=${(r.forwardM * 100).toFixed(0)}`);
      }
      sum += v;
      count++;
    }
  }
  if (room) parts.push(`到達${reachedCount}/${count}`);
  return { score: count > 0 ? sum / count : -Infinity, note: parts.join(' ') };
}

/** 学習済み方策を各コース×離散方位で録画し、replay と manifest を書き出す（共通）。 */
function recordCourses(
  evalEnvs: Array<{ name: string; env: SnakeEnv }>,
  policy: Policy,
  servo: ReturnType<typeof getServo>,
  totalMass: number,
  policyTag: string,
  headingsDeg: readonly number[],
  room: boolean,
): void {
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('\n=== 結果（決定論評価・コース×方位） ===');
  for (const e of evalEnvs) {
    for (const deg of headingsDeg) {
      const rad = deg * DEG;
      const base = evaluateBase(e.env, rad);
      const res = evaluate(e.env, policy, rad, true);
      const replay = e.env.getReplay();
      if (room) {
        console.log(
          `  [${e.name.padEnd(6)} ${String(deg).padStart(3)}°→${roomTargetWall(e.env.startCom[0], e.env.startCom[1], rad).wall.padEnd(5)}] ` +
            `RL ${res.reached ? '到達✓' : '未到達'} 残距離 ${(res.goalDistM * 100).toFixed(0)}cm 達成方位 ${res.achievedDeg.toFixed(0)}° ` +
            `(基盤 ${base.reached ? '到達' : '未到達'}) ${res.fell ? '破綻' : ''}`,
        );
        writeRoomTraceSvg(
          join(OUT_DIR, `room-trace-${headTag(deg)}-${servo.id}.svg`),
          e.env.getActiveTerrain(),
          e.env.startCom,
          deg,
          replay,
          res.reached,
        );
      } else {
        const gainPct =
          base.forwardM > 0 ? ((res.forwardM - base.forwardM) / base.forwardM) * 100 : 0;
        console.log(
          `  [${e.name.padEnd(11)} ${String(deg).padStart(3)}°] 基盤 ${(base.forwardM * 100).toFixed(0).padStart(4)}cm → ` +
            `RL ${(res.forwardM * 100).toFixed(0).padStart(4)}cm (${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(0)}%) ` +
            `達成方位 ${res.achievedDeg.toFixed(0)}° 横ずれ ${(res.crossM * 100).toFixed(0)}cm ${res.fell ? '破綻' : '完走'}`,
        );
      }
      const meta = {
        mechanism: 'snake3d',
        course: e.name,
        cmdHeadingDeg: deg, // 目標ヘディング指令（操舵）
        motor: servo.id,
        motorName: servo.name,
        mass: totalMass,
        base: 'climb-gait',
        policy: policyTag,
        forwardM: round4(res.forwardM),
        baseForwardM: round4(base.forwardM),
        achievedDeg: round4(res.achievedDeg),
        crossM: round4(res.crossM),
        reached: room ? res.reached : undefined,
        goalDistM: room ? round4(res.goalDistM) : undefined,
        fell: res.fell,
      };
      writeFileSync(
        join(OUT_DIR, `snake3d-${e.name}-${headTag(deg)}-${servo.id}.replay.json`),
        JSON.stringify({ meta, replay }, round5) + '\n',
      );
    }
  }
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const servo = getServo(opts.motor);
  const cap = servo.stallNm;
  const motorOverride = { stiffness: 3, damping: 0.15, maxTorqueNm: cap };

  await tf.setBackend('cpu');
  await tf.ready();

  const room = opts.course === 'room';
  const headings = room ? ROOM_EVAL_HEADINGS_DEG : EVAL_HEADINGS_DEG;

  // 訓練 env: room は部屋バンク、general は +x 地形バンク、単一コースは固定地形。方位は毎エピソードランダム化。
  let trainCfg: SnakeEnvConfig;
  if (room) {
    trainCfg = roomSnakeEnvConfig({ motor: motorOverride });
    trainCfg.terrainBank = makeRoomBank();
  } else {
    trainCfg = defaultSnakeEnvConfig({ motor: motorOverride });
    if (opts.course === 'general') trainCfg.terrainBank = makeCourseBank();
    else if (opts.course === 'flat') trainCfg.sim.terrain = [];
    else if (opts.course === 'challenge') trainCfg.sim.terrain = makeStraightChallengeTerrain();
    // progression は defaultSnakeEnvConfig の既定地形のまま。
  }
  trainCfg.episodeSteps = opts.episodeSteps;
  const trainEnv = await SnakeEnv.create(trainCfg);

  // 評価 env: room は壁のみのクリーンな部屋（決定論）。+x は名前付きコースを固定地形で。
  const evalEnvs: Array<{ name: string; env: SnakeEnv }> = [];
  if (room) {
    const cfg = roomSnakeEnvConfig({ terrain: makeRoomWalls(), motor: motorOverride });
    cfg.episodeSteps = opts.episodeSteps;
    evalEnvs.push({ name: 'room', env: await SnakeEnv.create(cfg) });
  } else {
    const evalCourseList =
      opts.course === 'general' ? EVAL_COURSES : EVAL_COURSES.filter((c) => c.name === opts.course);
    for (const c of evalCourseList) {
      const cfg = defaultSnakeEnvConfig({ terrain: c.terrain(), motor: motorOverride });
      cfg.episodeSteps = opts.episodeSteps;
      evalEnvs.push({ name: c.name, env: await SnakeEnv.create(cfg) });
    }
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
    `=== 3D 蛇 RL (PPO) === コース=${opts.course} / モーター=${servo.name} (cap ${cap.toFixed(3)} N·m) / 操舵±${(trainCfg.headingMaxRad / DEG).toFixed(0)}°`,
  );
  console.log(
    `  obsDim=${trainEnv.obsDim} actDim=${trainEnv.actDim} hidden=${opts.hidden} / rollout=${opts.rollout} episodeSteps=${opts.episodeSteps} / tf=${tf.getBackend()}`,
  );
  // 基盤（残差0）の直進(0°)前進＝目安。
  for (const e of evalEnvs) {
    const b0 = evaluateBase(e.env, 0);
    if (room) {
      console.log(
        `  基盤(0°) ${e.name}: ${b0.reached ? '到達✓' : '未到達'} 残距離 ${(b0.goalDistM * 100).toFixed(0)}cm`,
      );
    } else {
      console.log(`  基盤(0°) ${e.name}: ${(b0.forwardM * 100).toFixed(0)}cm`);
    }
  }

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
      const { score, note } = selectionScore(evalEnvs, policy, headings, room);
      if (score > bestScore) {
        bestScore = score;
        bestWeights = policy.exportWeights();
      }
      ppo.resetRollout(); // 評価で env を reset したので学習ロールアウトの継続状態を破棄（整合）
      evalNote = ` | 0°: ${note} (score ${score.toFixed(2)})`;
    }
    console.log(
      `  iter ${String(s.iteration).padStart(3)}: return=${s.meanEpisodeReturn.toFixed(2).padStart(8)} ` +
        `forward=${(s.meanEpisodeForward * 100).toFixed(0).padStart(5)}cm vLoss=${s.valueLoss.toFixed(3)} std=${s.std.toFixed(3)}${evalNote}`,
    );
  }

  // ---- ベスト方策で各コース×方位を録画・保存 ----
  policy.importWeights(bestWeights);
  const policyTag = opts.course === 'general' ? 'general' : opts.course;
  recordCourses(evalEnvs, policy, servo, trainCfg.sim.totalMass, policyTag, headings, room);

  // 方策の重み（全コース・全方位共通の単一方策）を1ファイルに保存。
  const weightsStem = opts.course === 'general' ? 'snake3d-general' : `snake3d-${opts.course}`;
  writeFileSync(
    join(OUT_DIR, `${weightsStem}-${servo.id}.json`),
    JSON.stringify({
      kind: 'rl-policy',
      mechanism: 'snake3d',
      course: opts.course,
      motor: servo.id,
      motorName: servo.name,
      obsDim: trainEnv.obsDim,
      actDim: trainEnv.actDim,
      hidden: opts.hidden,
      headingMaxDeg: trainCfg.headingMaxRad / DEG,
      weights: bestWeights,
      history,
      config: {
        iters: opts.iters,
        rollout: opts.rollout,
        lr: opts.lr,
        bank: trainCfg.terrainBank?.length ?? 1,
      },
    }) + '\n',
  );
  rebuildPolicyManifest();
  console.log(
    `\n  書き出し: 各コース×方位 snake3d-<course>-h<deg>-${servo.id}.replay.json ＋ ${weightsStem}-${servo.id}.json（+ manifest）`,
  );

  trainEnv.dispose();
  for (const e of evalEnvs) e.env.dispose();
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
