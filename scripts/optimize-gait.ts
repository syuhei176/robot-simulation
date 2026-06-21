/// <reference types="node" />
/**
 * オフライン歩容最適化（Mac / Node ヘッドレス Rapier）。
 *   実行例:
 *     node scripts/optimize-gait.ts --mech quad --motor sts3215 --gens 20
 *     node scripts/optimize-gait.ts --mech snake --course stairs --motor sts3215
 *
 * 機構（Mechanism）の歩容パラメータ（= ダッシュボードのスライダーと同じ key のうち
 * `optimize !== false` のもの）を、3D 実コース上で **CMA-ES** により最適化する。
 * 目的関数は機構が宣言する `replay.score().fitness`（前進量 − 物理破綻ペナルティ）で、
 * 機構非依存。重い探索は CLI 側で回し、結果（調整済みパラメータ＋改善履歴）を
 * `public/tuned/<mech>-<course>-<motor>.json` に書き出す。ダッシュボードはこれを再生する。
 *
 * CMA-ES は外部依存なしの自前実装（対称行列の Jacobi 固有値分解つき・seed 再現可）。
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { COURSES, type CourseId } from '../src/sim3d/course.ts';
import { getServo } from '../src/sim3d/servos.ts';
import { getMechanism } from '../src/mech/registry.ts';
import { defaultParamValues, type MechParam } from '../src/mech/Mechanism.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TUNED_DIR = join(ROOT, 'public', 'tuned');

// ===================== CLI =====================
interface Options {
  mech: string;
  course: CourseId;
  motor: string;
  gens: number;
  popsize: number | null; // null = N から自動
  seed: number;
  sigma: number;
  params: string[] | null; // 明示指定（null = optimize!==false の全 param）
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    mech: 'quad',
    course: 'stairs',
    motor: 'sts3215',
    gens: 20,
    popsize: null,
    seed: 1,
    sigma: 0.3,
    params: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error(`--${key} に値がありません`);
    switch (key) {
      case 'mech':
        opts.mech = val;
        break;
      case 'course':
        opts.course = val as CourseId;
        break;
      case 'motor':
        opts.motor = val;
        break;
      case 'gens':
        opts.gens = Number(val);
        break;
      case 'popsize':
        opts.popsize = Number(val);
        break;
      case 'seed':
        opts.seed = Number(val);
        break;
      case 'sigma':
        opts.sigma = Number(val);
        break;
      case 'params':
        opts.params = val.split(',').map((s) => s.trim());
        break;
      default:
        throw new Error(`未知のオプション --${key}`);
    }
  }
  return opts;
}

// ===================== 乱数（seed 再現可） =====================
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 標準正規乱数を1つ返す（Box-Muller）。 */
function gaussianSampler(rand: () => number): () => number {
  return () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// ===================== 線形代数（小さい対称行列向け） =====================
type Mat = number[][];

function identity(n: number): Mat {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

function matVec(m: Mat, v: number[]): number[] {
  return m.map((row) => row.reduce((acc, x, j) => acc + x * v[j], 0));
}

/**
 * 対称行列の固有値分解（cyclic Jacobi 法）。
 * 返り値: values=固有値, vectors=固有ベクトルを列に持つ直交行列 V（A = V diag(values) Vᵀ）。
 * CMA-ES の共分散は最大5×5程度なので Jacobi で十分・正確。
 */
function jacobiEigen(input: Mat): { values: number[]; vectors: Mat } {
  const n = input.length;
  const a = input.map((row) => [...row]);
  const v = identity(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((_, i) => a[i][i]), vectors: v };
}

// ===================== CMA-ES =====================
interface CmaSample {
  x: number[]; // [0,1]^N に clamp 済みの正規化点（評価された実体）
  fitness: number;
}

/**
 * 正規化空間 [0,1]^N 上の (μ/μ_w, λ)-CMA-ES。fitness を **最大化** する。
 * 箱制約は「サンプルを [0,1] に clamp し、その clamp 後の点を評価にも更新にも使う」方式
 * （平均が箱の外へ出ないシンプルな境界処理。低次元の歩容調整に十分）。
 */
class Cmaes {
  readonly n: number;
  readonly lambda: number;
  readonly mu: number;
  private readonly weights: number[];
  private readonly muEff: number;
  private readonly cSigma: number;
  private readonly dSigma: number;
  private readonly cc: number;
  private readonly c1: number;
  private readonly cmu: number;
  private readonly chiN: number;
  private mean: number[];
  private sigma: number;
  private cov: Mat;
  private pSigma: number[];
  private pc: number[];
  private b: Mat;
  private d: number[]; // 固有値の平方根
  private invSqrtC: Mat;
  private gen = 0;
  private readonly gauss: () => number;

  constructor(mean0: number[], sigma0: number, lambda: number, rand: () => number) {
    const n = mean0.length;
    this.n = n;
    this.lambda = lambda;
    this.mu = Math.floor(lambda / 2);
    const w = Array.from({ length: this.mu }, (_, i) => Math.log(this.mu + 0.5) - Math.log(i + 1));
    const wSum = w.reduce((a, b) => a + b, 0);
    this.weights = w.map((x) => x / wSum);
    this.muEff = 1 / this.weights.reduce((a, b) => a + b * b, 0);
    this.cSigma = (this.muEff + 2) / (n + this.muEff + 5);
    this.dSigma = 1 + 2 * Math.max(0, Math.sqrt((this.muEff - 1) / (n + 1)) - 1) + this.cSigma;
    this.cc = (4 + this.muEff / n) / (n + 4 + (2 * this.muEff) / n);
    this.c1 = 2 / ((n + 1.3) * (n + 1.3) + this.muEff);
    this.cmu = Math.min(
      1 - this.c1,
      (2 * (this.muEff - 2 + 1 / this.muEff)) / ((n + 2) * (n + 2) + this.muEff),
    );
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
    this.mean = [...mean0];
    this.sigma = sigma0;
    this.cov = identity(n);
    this.pSigma = new Array(n).fill(0);
    this.pc = new Array(n).fill(0);
    this.b = identity(n);
    this.d = new Array(n).fill(1);
    this.invSqrtC = identity(n);
    this.gauss = gaussianSampler(rand);
  }

  /** 1世代ぶんの候補（clamp 済み正規化点）を生成する。 */
  ask(): number[][] {
    const samples: number[][] = [];
    for (let k = 0; k < this.lambda; k++) {
      const z = Array.from({ length: this.n }, () => this.gauss());
      const dz = z.map((zi, i) => this.d[i] * zi);
      const y = matVec(this.b, dz);
      const x = y.map((yi, i) => clamp01(this.mean[i] + this.sigma * yi));
      samples.push(x);
    }
    return samples;
  }

  /** 評価済み候補で分布を更新する。 */
  tell(samples: CmaSample[]): void {
    this.gen++;
    const sorted = [...samples].sort((a, b) => b.fitness - a.fitness); // 最大化 → 降順
    const oldMean = [...this.mean];
    // 重み付き再結合（clamp 後の点をそのまま使う）。
    const newMean = new Array(this.n).fill(0);
    for (let i = 0; i < this.mu; i++) {
      const xi = sorted[i].x;
      for (let j = 0; j < this.n; j++) newMean[j] += this.weights[i] * xi[j];
    }
    const meanShift = newMean.map((m, j) => (m - oldMean[j]) / this.sigma); // = Σ w_i y_i
    this.mean = newMean;

    // p_sigma 更新
    const csFactor = Math.sqrt(this.cSigma * (2 - this.cSigma) * this.muEff);
    const cInvShift = matVec(this.invSqrtC, meanShift);
    this.pSigma = this.pSigma.map((p, j) => (1 - this.cSigma) * p + csFactor * cInvShift[j]);
    const psNorm = Math.hypot(...this.pSigma);

    // h_sigma
    const hsThresh = psNorm / Math.sqrt(1 - Math.pow(1 - this.cSigma, 2 * this.gen)) / this.chiN;
    const hSigma = hsThresh < 1.4 + 2 / (this.n + 1) ? 1 : 0;

    // p_c 更新
    const ccFactor = Math.sqrt(this.cc * (2 - this.cc) * this.muEff);
    this.pc = this.pc.map((p, j) => (1 - this.cc) * p + hSigma * ccFactor * meanShift[j]);

    // 共分散更新（rank-one + rank-mu）
    const deltaHsig = (1 - hSigma) * this.cc * (2 - this.cc);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        let rankMu = 0;
        for (let m = 0; m < this.mu; m++) {
          const y = (sorted[m].x[i] - oldMean[i]) / this.sigma;
          const yj = (sorted[m].x[j] - oldMean[j]) / this.sigma;
          rankMu += this.weights[m] * y * yj;
        }
        this.cov[i][j] =
          (1 - this.c1 - this.cmu) * this.cov[i][j] +
          this.c1 * (this.pc[i] * this.pc[j] + deltaHsig * this.cov[i][j]) +
          this.cmu * rankMu;
      }
    }
    // 対称化（数値誤差の蓄積を防ぐ）
    for (let i = 0; i < this.n; i++) {
      for (let j = i + 1; j < this.n; j++) {
        const avg = (this.cov[i][j] + this.cov[j][i]) / 2;
        this.cov[i][j] = avg;
        this.cov[j][i] = avg;
      }
    }

    // ステップサイズ更新
    this.sigma *= Math.exp((this.cSigma / this.dSigma) * (psNorm / this.chiN - 1));

    this.updateEigen();
  }

  private updateEigen(): void {
    const { values, vectors } = jacobiEigen(this.cov);
    const eig = values.map((v) => Math.max(v, 1e-20));
    this.b = vectors;
    this.d = eig.map((v) => Math.sqrt(v));
    // invSqrtC = B diag(1/d) Bᵀ
    const inv = identity(this.n);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        let sum = 0;
        for (let k = 0; k < this.n; k++) sum += vectors[i][k] * (1 / this.d[k]) * vectors[j][k];
        inv[i][j] = sum;
      }
    }
    this.invSqrtC = inv;
  }

  bestMean(): number[] {
    return this.mean.map(clamp01);
  }
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

// ===================== パラメータ空間 =====================
interface Dim {
  key: string;
  min: number;
  max: number;
}

function denorm(dims: Dim[], x: number[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (let i = 0; i < dims.length; i++) {
    out[dims[i].key] = dims[i].min + x[i] * (dims[i].max - dims[i].min);
  }
  return out;
}

function norm(dims: Dim[], values: Record<string, number>): number[] {
  return dims.map((d) => clamp01((values[d.key] - d.min) / (d.max - d.min)));
}

// ===================== 評価 =====================
async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const mech = getMechanism(opts.mech);
  const servo = getServo(opts.motor);
  const torqueCapNm = servo.stallNm;
  const effectiveCourse: CourseId = mech.supportsCourse ? opts.course : 'flat';
  const courseSpec = COURSES[effectiveCourse]();

  // 探索次元 = optimize!==false の param（--params 指定があればさらに絞る）。
  const tunable = mech.params.filter((p: MechParam) => p.optimize !== false);
  const selected = opts.params ? tunable.filter((p) => opts.params!.includes(p.key)) : tunable;
  if (selected.length === 0) throw new Error('最適化対象のパラメータがありません');
  const dims: Dim[] = selected.map((p) => ({ key: p.key, min: p.min, max: p.max }));

  const fixed = defaultParamValues(mech); // 非探索 param は既定値で固定
  const lambda = opts.popsize ?? 4 + Math.floor(3 * Math.log(dims.length));

  const evaluate = async (paramOverrides: Record<string, number>) => {
    const params = { ...fixed, ...paramOverrides };
    const replay = await mech.run({
      course: courseSpec,
      torqueCapNm,
      motorName: servo.name,
      params,
    });
    return replay.score();
  };

  console.log(
    `=== 歩容最適化 CMA-ES === 機構=${mech.id} / コース=${effectiveCourse} / モーター=${servo.name} (cap ${torqueCapNm.toFixed(2)} N·m)`,
  );
  console.log(
    `  探索次元(${dims.length}): ${dims.map((d) => d.key).join(', ')} / λ=${lambda} / 世代=${opts.gens} / seed=${opts.seed}`,
  );

  const baseline = await evaluate({});
  console.log(
    `  baseline: fitness=${baseline.fitness.toFixed(4)} progress=${(baseline.progressM * 100).toFixed(1)}cm feasible=${baseline.feasible}`,
  );

  const rand = mulberry32(opts.seed);
  const mean0 = norm(dims, fixed); // 既定歩容から探索を開始
  const cma = new Cmaes(mean0, opts.sigma, lambda, rand);

  interface HistRow {
    gen: number;
    best: number;
    mean: number;
    progressM: number;
  }
  const history: HistRow[] = [];
  // ベストは「捕捉時点で denorm した新規スナップショット」を持つ（サンプル配列のエイリアシングを避ける）。
  let bestParams = denorm(dims, mean0);
  let bestFitness = baseline.fitness;
  let bestProgress = baseline.progressM;

  for (let g = 0; g < opts.gens; g++) {
    const xs = cma.ask();
    const samples: CmaSample[] = [];
    let genBest = -Infinity;
    let genBestProgress = 0;
    let sum = 0;
    for (const x of xs) {
      const sc = await evaluate(denorm(dims, x));
      samples.push({ x, fitness: sc.fitness });
      sum += sc.fitness;
      if (sc.fitness > genBest) {
        genBest = sc.fitness;
        genBestProgress = sc.progressM;
      }
      if (sc.fitness > bestFitness) {
        bestFitness = sc.fitness;
        bestProgress = sc.progressM;
        bestParams = denorm(dims, x);
      }
    }
    cma.tell(samples);
    history.push({
      gen: g,
      best: round4(genBest),
      mean: round4(sum / samples.length),
      progressM: round4(genBestProgress),
    });
    console.log(
      `  gen ${String(g).padStart(2)}: best=${genBest.toFixed(4)} mean=${(sum / samples.length).toFixed(4)} (累計best=${bestFitness.toFixed(4)}, ${(bestProgress * 100).toFixed(1)}cm)`,
    );
  }

  // 累計ベストの実体（full 精度の float）をそのまま採用し最終評価する。snake の 18cm 階段のような
  // 実現可能性の崖では小数6桁の丸めですら結果が激変するほどカオス的なため、一切丸めない。ランナーは
  // 同一 float なら決定論的なので、保存する params → tuned スコアは厳密に再現する（ダッシュボードが
  // 同じ値で再生してスコアが一致する）。
  const tunedParams: Record<string, number> = { ...fixed, ...bestParams };
  const tuned = await evaluate(tunedParams);

  console.log('');
  console.log('=== 結果 ===');
  for (const dim of dims) {
    console.log(
      `  ${dim.key.padEnd(14)} ${fixed[dim.key].toFixed(4)} → ${tunedParams[dim.key].toFixed(4)}`,
    );
  }
  console.log(
    `  fitness ${baseline.fitness.toFixed(4)} → ${tuned.fitness.toFixed(4)} / progress ${(baseline.progressM * 100).toFixed(1)} → ${(tuned.progressM * 100).toFixed(1)}cm / feasible ${baseline.feasible}→${tuned.feasible}`,
  );

  // ---- 書き出し ----
  const record = {
    mechanism: mech.id,
    course: effectiveCourse,
    motor: servo.id,
    motorName: servo.name,
    torqueCapNm: round4(torqueCapNm),
    optimizedKeys: dims.map((d) => d.key),
    // params は評価した精度のまま保存する（丸めると再生時にスコアが再現しなくなるため）。
    params: tunedParams,
    baseline: scoreRecord(baseline),
    tuned: scoreRecord(tuned),
    history,
    config: { gens: opts.gens, popsize: lambda, seed: opts.seed, sigma: opts.sigma },
  };
  mkdirSync(TUNED_DIR, { recursive: true });
  const fileName = `${mech.id}-${effectiveCourse}-${servo.id}.json`;
  writeFileSync(join(TUNED_DIR, fileName), JSON.stringify(record, null, 2) + '\n');
  rebuildManifest();
  console.log('');
  console.log(`  書き出し: public/tuned/${fileName}（+ manifest.json）`);
}

function scoreRecord(s: { fitness: number; progressM: number; feasible: boolean }) {
  return { fitness: round4(s.fitness), progressM: round4(s.progressM), feasible: s.feasible };
}

function round4(x: number): number {
  return Number(x.toFixed(4));
}

/** public/tuned/*.json を走査して manifest.json（ダッシュボードの利用可能一覧）を作り直す。 */
function rebuildManifest(): void {
  const entries = readdirSync(TUNED_DIR)
    .filter((f) => f.endsWith('.json') && f !== 'manifest.json')
    .map((file) => {
      const rec = JSON.parse(readFileSync(join(TUNED_DIR, file), 'utf8'));
      return {
        file,
        mechanism: rec.mechanism,
        course: rec.course,
        motor: rec.motor,
        torqueCapNm: rec.torqueCapNm,
        baseline: rec.baseline,
        tuned: rec.tuned,
      };
    });
  writeFileSync(join(TUNED_DIR, 'manifest.json'), JSON.stringify(entries, null, 2) + '\n');
}

await main();
