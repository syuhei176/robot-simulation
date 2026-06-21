import { DEFAULT_SIM, type SimParams } from '../config.ts';
import { SnakePhysics } from '../sim/SnakePhysics.ts';

/** ステップ結果。done は時間切れ or 数値破綻。 */
export interface StepResult {
  obs: Float32Array;
  reward: number;
  done: boolean;
}

export interface EnvConfig {
  /** 1 エピソードの制御ステップ数 */
  episodeSteps: number;
  /** 1 制御ステップあたりの物理フレーム数（制御周期 = controlSubsteps / 60 s） */
  controlSubsteps: number;
  /** 前進距離に対する報酬係数 */
  forwardReward: number;
  /** 進行方向(+x)から逸れることへのペナルティ係数（直進を促す） */
  headingPenalty: number;
  /** 制御の激しさ（関節を曲げる量）へのペナルティ係数 */
  energyPenalty: number;
  /** 位相クロックの周波数 [Hz]。方策にリズムを与える外部時計。 */
  clockFreq: number;
  /** 各関節の目標角の上限 [rad]（行動を tanh で squash する範囲） */
  maxJointAngle: number;
}

export const DEFAULT_ENV: EnvConfig = {
  episodeSteps: 200,
  controlSubsteps: 4,
  forwardReward: 6.0,
  headingPenalty: 0.05,
  energyPenalty: 0.01,
  clockFreq: 0.7,
  maxJointAngle: 1.0,
};

const TAU = Math.PI * 2;
const tanh = Math.tanh;

/**
 * end-to-end 強化学習環境：方策が各関節の目標角を直接出力する。
 *
 *  行動 a ∈ R^(関節数) を tanh で [-maxJointAngle, +maxJointAngle] に squash し、
 *  各関節の目標角としてそのまま物理へ与える（serpenoid の数式は使わない）。
 *  フィードフォワード方策は「同じ観測→同じ行動」なので、振動歩容を生むリズムの
 *  基準として観測に位相クロック(sin/cos)を入れる。方策はこのクロックを手がかりに、
 *  関節間の位相差・振幅を含む進行波を自力で学習する。
 *
 *  目標角は制御ステップ間で急変させず、物理サブステップにかけて線形補間する
 *  （SnakePhysics は φ̇ から推進力を得るため、角速度のスパイクを避ける必要がある）。
 *  報酬は +x 方向への前進量 − 進行方向のズレ − 関節を曲げるエネルギー。
 */
export class MicrobotEnv {
  readonly physics: SnakePhysics;
  readonly obsDim: number;
  readonly actDim: number;

  private readonly sim: SimParams;
  private readonly cfg: EnvConfig;
  private readonly jointCount: number;
  /** 現在適用中の関節目標角（制御ステップ間の補間の始点にもなる） */
  private readonly targets: Float64Array;
  private readonly nextTargets: Float64Array;
  private readonly blend: Float64Array;
  private phase = 0;
  private stepCount = 0;
  private prevX = 0;
  private readonly obs: Float32Array;

  constructor(env: EnvConfig = DEFAULT_ENV, sim: SimParams = DEFAULT_SIM) {
    this.sim = { ...sim };
    this.cfg = env;
    this.physics = new SnakePhysics(this.sim);
    this.jointCount = this.physics.jointCount;
    this.actDim = this.jointCount;
    // 観測: 位相クロック(2) + COM速度(2) + heading(2) + 各関節角(jointCount)
    this.obsDim = 6 + this.jointCount;
    this.targets = new Float64Array(this.jointCount);
    this.nextTargets = new Float64Array(this.jointCount);
    this.blend = new Float64Array(this.jointCount);
    this.obs = new Float32Array(this.obsDim);
  }

  /** 1 制御ステップが表す実時間 [s]（= controlSubsteps * dt）。描画の等速再生に使う。 */
  get controlPeriod(): number {
    return this.cfg.controlSubsteps * this.sim.dt;
  }

  /** RLEnv 契約: 進捗指標 = COM の前進 x [m]。 */
  progressMetric(): number {
    return this.physics.centerOfMass()[0];
  }

  reset(): Float32Array {
    this.physics.reset();
    // 初期位相は決定論的に 0。ランダム化すると対称な初期状態から進行方向(+x/-x)が
    // ランダムに決まり、同一方策でも報酬符号が揺れて学習が壊れる。
    this.phase = 0;
    this.stepCount = 0;
    this.prevX = this.physics.centerOfMass()[0];
    this.targets.fill(0);
    this.nextTargets.fill(0);
    return this.writeObs();
  }

  /** raw 行動 (長さ jointCount) を受け取り 1 制御ステップ進める。 */
  step(action: ArrayLike<number>): StepResult {
    const { maxJointAngle: maxAng, controlSubsteps: subs } = this.cfg;

    // 行動 → 各関節の目標角（squash）
    for (let j = 0; j < this.jointCount; j++) {
      this.nextTargets[j] = maxAng * tanh(action[j] ?? 0);
    }

    const dt = this.sim.dt;
    const clockW = TAU * this.cfg.clockFreq;
    for (let k = 0; k < subs; k++) {
      // 目標角を前ステップ値から新目標へ線形補間（角速度スパイクを避ける）
      const alpha = (k + 1) / subs;
      for (let j = 0; j < this.jointCount; j++) {
        this.blend[j] = this.targets[j] + alpha * (this.nextTargets[j] - this.targets[j]);
      }
      this.physics.setJointTargets(this.blend);
      this.physics.step(dt);
      this.phase += clockW * dt;
    }

    // エネルギー: 関節をどれだけ曲げているか（平均二乗）
    let bend = 0;
    for (let j = 0; j < this.jointCount; j++) bend += this.nextTargets[j] * this.nextTargets[j];
    bend /= this.jointCount || 1;

    // 次ステップの補間始点として確定
    for (let j = 0; j < this.jointCount; j++) this.targets[j] = this.nextTargets[j];

    const com = this.physics.centerOfMass();
    const dx = com[0] - this.prevX;
    this.prevX = com[0];

    // 報酬: 前進量 − 進行方向のズレ − エネルギー
    const headingErr = 1 - Math.cos(this.physics.heading());
    let reward =
      this.cfg.forwardReward * dx -
      this.cfg.headingPenalty * headingErr -
      this.cfg.energyPenalty * bend;

    this.stepCount++;
    const finite = Number.isFinite(com[0]) && Number.isFinite(com[1]);
    if (!finite) reward -= 5;
    const done = !finite || this.stepCount >= this.cfg.episodeSteps;

    return { obs: this.writeObs(), reward, done };
  }

  private writeObs(): Float32Array {
    const v = this.physics.centerOfMassVelocity();
    const h = this.physics.heading();
    const o = this.obs;
    o[0] = Math.sin(this.phase);
    o[1] = Math.cos(this.phase);
    o[2] = v[0]; // 前進速度 (+x)
    o[3] = v[1]; // 横速度
    o[4] = Math.sin(h);
    o[5] = Math.cos(h);
    const inv = 1 / this.cfg.maxJointAngle;
    for (let j = 0; j < this.jointCount; j++) o[6 + j] = this.targets[j] * inv;
    return o;
  }
}
