/**
 * 3D 四足の end-to-end 強化学習環境（平地・小型四足）。
 *
 * `runQuadrupedGait` を「1 制御ステップずつ進められる」形にほどいたもの。IK クロール歩容は使わず、
 * 方策が 8 関節（hip×4 + knee×4）の目標角を直接出力する（action ∈ R^8 を tanh で squash し、
 * 静止立脚姿勢を中心に ±maxDelta の関節角オフセットとして与える）。物理は `runQuadrupedGait` と
 * 同じ忠実トルク駆動（PD を ±cap で clamp ＋ 受動ダンピング）と横安定化を再利用する。脚は pitch のみで
 * 横（ロール/ヨー）を制御できないため横安定化は必須前提（矢状面の歩行・トルク充足を切り出して学ぶ）。
 *
 * 機体は総質量＝機体スケールの相似縮小（`scaledBodyOverrides`）。既定は 150g・SCS0009 cap。
 * 観測に位相クロック(sin/cos)を入れてリズムの基準を与え、方策はそれを手がかりに前進歩容を自力で学ぶ。
 * 報酬 = 前進量 − 傾き − 行動エネルギー − トルク飽和、転倒/数値破綻で終了。
 */
import RAPIER from '@dimforge/rapier3d-compat';
import { G } from '../sim3d/chain.ts';
import { COURSES, buildCourseColliders, terrainTopAt, type CourseSpec } from '../sim3d/course.ts';
import {
  buildQuad,
  driveJoint,
  stabilizeLateral,
  pitchAboutY,
  tiltFromUp,
  legIK,
  resolveConfig,
  scaledBodyOverrides,
  DEG,
  KNEE_SIGN,
  MOTOR_AXIS_SIGN,
  type QuadAssembly,
  type QuadDynConfig,
} from '../sim3d/quadruped-dynamics.ts';
import type { RLEnv, StepResult } from '../rl/RLEnv.ts';

export interface QuadEnvConfig {
  mass: number; // 総質量＝機体スケール [kg]
  torqueCapNm: number; // 関節トルク上限 [N·m]（サーボ stall）
  course: CourseSpec; // 走行コース（既定は平地）
  episodeSteps: number; // 1 エピソードの制御ステップ数
  controlFrames: number; // 1 制御ステップが含む cfg.dt フレーム数（制御周期 = controlFrames/240 s）
  clockFreq: number; // 位相クロック周波数 [Hz]
  maxHipDelta: number; // hip 目標角の可動幅 [rad]（静止姿勢中心の ±）
  maxKneeDelta: number; // knee 目標角の可動幅 [rad]
  standFrac: number; // 静止立脚の足先高さ＝(thigh+shin)*standFrac（<1 で膝を曲げる）
  forwardReward: number; // 前進量への係数
  tiltPenalty: number; // 傾き(>20°)への係数
  energyPenalty: number; // 行動エネルギー（関節オフセット二乗）への係数
  satPenalty: number; // トルク飽和率への係数
  aliveBonus: number; // 生存ボーナス（転倒回避を促す）
  fallPenalty: number; // 転倒の終端ペナルティ
}

export const DEFAULT_QUAD_ENV: Omit<QuadEnvConfig, 'course'> = {
  mass: 0.15,
  torqueCapNm: 0.2256, // SCS0009 stall
  episodeSteps: 200,
  controlFrames: 4, // 制御周期 = 4/240 ≈ 60 Hz
  clockFreq: 1.2,
  maxHipDelta: 0.6,
  maxKneeDelta: 0.6,
  standFrac: 0.9,
  forwardReward: 20,
  tiltPenalty: 0.01,
  energyPenalty: 0.002,
  satPenalty: 0.3,
  aliveBonus: 0.01,
  fallPenalty: 2,
};

const TAU = Math.PI * 2;
const tanh = Math.tanh;

export class QuadEnv implements RLEnv {
  readonly obsDim: number;
  readonly actDim: number;

  private readonly cfg: QuadEnvConfig;
  private readonly dyn: QuadDynConfig;
  private readonly physicsDt: number;
  private readonly substeps: number;
  private readonly course: CourseSpec;
  // 静止立脚姿勢の関節相対目標（hip rel, knee rel）。action はこの中心からのオフセット。
  private readonly hip0: number;
  private readonly knee0: number;
  private readonly maxDelta: Float64Array; // 各 action 次元の可動幅（hip/knee 交互）

  private world!: RAPIER.World;
  private asm!: QuadAssembly;
  private startX = 0;
  private standZ = 0;
  private fallClearance = 0;
  private prevTargets: Float64Array; // 前ステップの 8 関節目標（補間始点）
  private curTargets: Float64Array;
  private blend: Float64Array;
  private phase = 0;
  private stepCount = 0;
  private prevX = 0;
  private readonly obs: Float32Array;

  private constructor(cfg: QuadEnvConfig) {
    this.cfg = cfg;
    this.course = cfg.course;
    this.dyn = resolveConfig(scaledBodyOverrides(cfg.mass, cfg.torqueCapNm));
    this.substeps = Math.max(1, Math.round(this.dyn.substeps));
    this.physicsDt = this.dyn.dt / this.substeps;

    // 静止立脚姿勢: 足先を hip 直下の standFrac*reach に置く IK（trunkPitch=0 基準）。
    const reach = this.dyn.leg.thigh + this.dyn.leg.shin;
    const { p1, p2 } = legIK(
      0,
      -reach * cfg.standFrac,
      this.dyn.leg.thigh,
      this.dyn.leg.shin,
      KNEE_SIGN,
    );
    this.hip0 = p1; // hip rel = p1 - trunkPitch(0)
    this.knee0 = p2 - p1;

    this.actDim = 8; // hip/knee × 4 脚
    // 観測: 位相(2) + pitch + pitchRate + vForward + vVert + clearance(1) + 関節相対角(8) = 15
    this.obsDim = 7 + 8;
    this.obs = new Float32Array(this.obsDim);
    this.prevTargets = new Float64Array(this.actDim);
    this.curTargets = new Float64Array(this.actDim);
    this.blend = new Float64Array(this.actDim);
    this.maxDelta = new Float64Array(this.actDim);
    for (let leg = 0; leg < 4; leg++) {
      this.maxDelta[leg * 2] = cfg.maxHipDelta;
      this.maxDelta[leg * 2 + 1] = cfg.maxKneeDelta;
    }
  }

  /** RAPIER 初期化（非同期・一度きり）を済ませてから env を生成する。reset() は同期。 */
  static async create(overrides: Partial<QuadEnvConfig> = {}): Promise<QuadEnv> {
    await RAPIER.init();
    const cfg: QuadEnvConfig = {
      ...DEFAULT_QUAD_ENV,
      course: overrides.course ?? COURSES.flat(),
      ...overrides,
    };
    const env = new QuadEnv(cfg);
    env.reset();
    return env;
  }

  /** 1 制御ステップが表す実時間 [s]（描画の等速再生に使う）。 */
  get controlPeriod(): number {
    return this.cfg.controlFrames * this.dyn.dt;
  }

  /** 1 エピソードの制御ステップ上限（評価で時間切れ判定に使う）。 */
  get maxSteps(): number {
    return this.cfg.episodeSteps;
  }

  progressMetric(): number {
    return this.asm.trunk.translation().x;
  }

  reset(): Float32Array {
    if (this.world) this.world.free();
    this.world = new RAPIER.World({ x: 0, y: 0, z: -G });
    this.world.timestep = this.physicsDt;
    this.world.numSolverIterations = 8;
    buildCourseColliders(this.world, this.course, this.dyn.friction, 3);
    this.asm = buildQuad(this.world, this.dyn);

    this.startX = this.asm.trunk.translation().x;
    this.standZ = this.asm.trunk.translation().z;
    this.fallClearance = this.standZ * 0.55;
    this.phase = 0;
    this.stepCount = 0;
    this.prevX = this.startX;
    // 静止立脚姿勢を始点にする（action=0 で立脚維持）。
    for (let leg = 0; leg < 4; leg++) {
      this.prevTargets[leg * 2] = this.hip0;
      this.prevTargets[leg * 2 + 1] = this.knee0;
      this.curTargets[leg * 2] = this.hip0;
      this.curTargets[leg * 2 + 1] = this.knee0;
    }
    return this.writeObs();
  }

  step(action: ArrayLike<number>): StepResult {
    // 行動 → 8 関節の相対目標角（静止姿勢中心に ±maxDelta で squash）。
    for (let leg = 0; leg < 4; leg++) {
      const hi = leg * 2;
      const ki = leg * 2 + 1;
      this.curTargets[hi] = this.hip0 + this.maxDelta[hi] * tanh(action[hi] ?? 0);
      this.curTargets[ki] = this.knee0 + this.maxDelta[ki] * tanh(action[ki] ?? 0);
    }

    const totalSub = this.cfg.controlFrames * this.substeps;
    const clockW = TAU * this.cfg.clockFreq;
    let satCount = 0;
    let satSamples = 0;
    for (let sub = 0; sub < totalSub; sub++) {
      // 目標角を前ステップ値から新目標へ線形補間（角速度スパイクを避ける）。
      const alpha = (sub + 1) / totalSub;
      for (let j = 0; j < this.actDim; j++) {
        this.blend[j] = this.prevTargets[j] + alpha * (this.curTargets[j] - this.prevTargets[j]);
      }
      for (let leg = 0; leg < 4; leg++) {
        const L = this.asm.legs[leg];
        const hip = driveJoint(
          L.hipJoint,
          this.asm.trunk,
          L.thigh,
          this.blend[leg * 2],
          MOTOR_AXIS_SIGN,
          this.dyn.motor,
          this.physicsDt,
        );
        const knee = driveJoint(
          L.kneeJoint,
          L.thigh,
          L.shin,
          this.blend[leg * 2 + 1],
          MOTOR_AXIS_SIGN,
          this.dyn.motor,
          this.physicsDt,
        );
        satSamples += 2;
        if (hip.demand - hip.applied > 1e-9) satCount++;
        if (knee.demand - knee.applied > 1e-9) satCount++;
      }
      stabilizeLateral(
        this.asm.trunk,
        this.dyn.lateralStabK,
        this.dyn.lateralStabD,
        this.physicsDt,
      );
      this.world.step();
      this.phase += clockW * this.physicsDt;
    }
    // 次ステップの補間始点として確定。
    this.prevTargets.set(this.curTargets);

    // --- 報酬 ---
    const trunk = this.asm.trunk;
    const x = trunk.translation().x;
    const z = trunk.translation().z;
    const dx = x - this.prevX;
    this.prevX = x;
    const tiltDeg = tiltFromUp(trunk) / DEG;

    let energy = 0;
    for (let j = 0; j < this.actDim; j++) {
      const a = tanh(action[j] ?? 0);
      energy += a * a;
    }
    energy /= this.actDim;
    const satFrac = satSamples > 0 ? satCount / satSamples : 0;

    const finite = Number.isFinite(x) && Number.isFinite(z);
    const clearance = z - terrainTopAt(this.course, x);
    const fell = !finite || clearance < this.fallClearance || tiltFromUp(trunk) > 55 * DEG;

    let reward =
      this.cfg.forwardReward * dx -
      this.cfg.tiltPenalty * Math.max(0, tiltDeg - 20) -
      this.cfg.energyPenalty * energy -
      this.cfg.satPenalty * satFrac +
      this.cfg.aliveBonus;
    if (fell) reward -= this.cfg.fallPenalty;

    this.stepCount++;
    const done = fell || this.stepCount >= this.cfg.episodeSteps;
    return { obs: this.writeObs(), reward, done };
  }

  private writeObs(): Float32Array {
    const trunk = this.asm.trunk;
    const lin = trunk.linvel();
    const ang = trunk.angvel();
    const pitch = pitchAboutY(trunk);
    const z = trunk.translation().z;
    const o = this.obs;
    o[0] = Math.sin(this.phase);
    o[1] = Math.cos(this.phase);
    o[2] = pitch / 0.5;
    o[3] = ang.y / 5;
    o[4] = lin.x / 0.5;
    o[5] = lin.z / 0.5;
    o[6] =
      this.standZ > 0 ? (z - terrainTopAt(this.course, trunk.translation().x)) / this.standZ : 0;
    const trunkPitch = pitch;
    for (let leg = 0; leg < 4; leg++) {
      const L = this.asm.legs[leg];
      const thighPitch = pitchAboutY(L.thigh);
      const shinPitch = pitchAboutY(L.shin);
      o[7 + leg * 2] = thighPitch - trunkPitch; // hip rel
      o[7 + leg * 2 + 1] = shinPitch - thighPitch; // knee rel
    }
    // NaN ガード（数値破綻時に学習を壊さない）。
    for (let i = 0; i < this.obsDim; i++) if (!Number.isFinite(o[i])) o[i] = 0;
    return o;
  }

  dispose(): void {
    if (this.world) this.world.free();
  }
}
