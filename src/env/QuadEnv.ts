/**
 * 3D 四足の強化学習環境（平地・小型四足）。
 *
 * `runQuadrupedGait` を「1 制御ステップずつ進められる」形にほどいたもの。方策が 8 関節
 * （hip×4 + knee×4）を制御する。2 モード:
 *  - baseGait=true（**残差RL**・既定）: 各関節目標 = IK クロール歩容のターゲット(時間依存) + 方策の残差
 *    (±maxDelta·tanh(action))。土台の歩容が action=0 でも前進する（≈17.5cm）ので、方策は「土台をどう
 *    補正すれば速く/安定するか」だけを学べばよい。決定論方策（平均行動）も最初から歩く＝ end-to-end が
 *    嵌った「ノイズ依存の縮退均衡（平均は静止のまま）」を回避できる。
 *  - baseGait=false（end-to-end・比較用）: 目標 = 静止立脚姿勢 + 残差。歩容の振動をゼロから生成する必要があり難しい。
 *
 * 物理は `runQuadrupedGait` と同じ忠実トルク駆動（PD を ±cap で clamp ＋ 受動ダンピング）と横安定化を
 * 再利用する。脚は pitch のみで横（ロール/ヨー）を制御できないため横安定化は必須前提（矢状面の歩行・トルク
 * 充足を切り出して学ぶ）。機体は総質量＝機体スケールの相似縮小（`scaledBodyOverrides`、既定 150g・SCS0009 cap）。
 * 観測に位相クロック(sin/cos)を入れて歩容リズムの基準を与える。報酬 = 前進量 − 傾き − 行動エネルギー
 * − トルク飽和、転倒/数値破綻で終了。
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
  footTargetRelHip,
  resolveConfig,
  scaledBodyOverrides,
  bodyScale,
  DEFAULT_QUAD_DYN_CONFIG,
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
  baseGait: boolean; // true=残差RL（IK歩容に上乗せ）, false=end-to-end（静止姿勢中心）
  // 土台 IK 歩容のパラメータ（基準機体 1.2kg の単位。機体スケール s で内部スケール）。
  // 既定は遅い参照歩容。CMA-ES の tuned 歩容（period 0.6/stride 0.11 等）を渡すと速い土台になり、
  // 残差RL が「速い歩容」をフィードバックで頑健化する形になる（cadence 固定の上限を引き上げる）。
  gaitParams?: Partial<{
    period: number;
    strideM: number;
    liftM: number;
    standM: number;
    stanceDuty: number;
  }>;
  clockFreq: number; // 位相クロック周波数 [Hz]
  maxHipDelta: number; // hip 残差の可動幅 [rad]（土台ターゲット中心の ±）
  maxKneeDelta: number; // knee 残差の可動幅 [rad]
  standFrac: number; // 静止立脚の足先高さ＝(thigh+shin)*standFrac（end-to-end モードの中心）
  forwardReward: number; // 前進量への係数
  tiltPenalty: number; // 傾き(>20°)への係数
  energyPenalty: number; // 行動エネルギー（残差二乗）への係数
  satPenalty: number; // トルク飽和率への係数
  aliveBonus: number; // 生存ボーナス（転倒回避を促す）
  fallPenalty: number; // 転倒の終端ペナルティ
}

export const DEFAULT_QUAD_ENV: Omit<QuadEnvConfig, 'course'> = {
  mass: 0.15,
  torqueCapNm: 0.2256, // SCS0009 stall
  episodeSteps: 200,
  controlFrames: 4, // 制御周期 = 4/240 ≈ 60 Hz
  baseGait: true, // 残差RL を既定にする
  clockFreq: 1.2, // ≈ スケール後の歩容周波数（150g で gait period≈0.85s）
  // 残差は土台の歩容に上乗せする「補正」なので可動幅は小さめ（end-to-end の ±0.6 より控えめ）。
  maxHipDelta: 0.35,
  maxKneeDelta: 0.35,
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
  private readonly dyn: QuadDynConfig; // 機体スケール＋スケール済み歩容を含む解決済み config
  private readonly physicsDt: number;
  private readonly substeps: number;
  private readonly course: CourseSpec;
  // 静止立脚姿勢の関節相対目標（end-to-end モードの中心）。
  private readonly hip0: number;
  private readonly knee0: number;
  private readonly vBody: number; // 土台歩容の基準前進速度（足が世界で固定される速度）
  private readonly maxDelta: Float64Array; // 各 action 次元の残差可動幅（hip/knee 交互）

  private world!: RAPIER.World;
  private asm!: QuadAssembly;
  private startX = 0;
  private standZ = 0;
  private fallClearance = 0;
  private prevResidual: Float64Array; // 前ステップの 8 関節残差（補間始点）
  private curResidual: Float64Array;
  private residualBlend: Float64Array;
  private phase = 0;
  private simTime = 0; // 土台歩容に渡す経過時間 [s]
  private stepCount = 0;
  private prevX = 0;
  private readonly obs: Float32Array;

  private constructor(cfg: QuadEnvConfig) {
    this.cfg = cfg;
    this.course = cfg.course;
    // 機体スケール s に応じて歩容（長さ ∝ s, 周期 ∝ √s）もスケールして土台歩容を作る（quad.ts と同じ規約）。
    // 歩容パラメータは基準機体単位で、既定は参照歩容・cfg.gaitParams で上書き（tuned 歩容を土台にできる）。
    const s = bodyScale(cfg.mass);
    const Dg = { ...DEFAULT_QUAD_DYN_CONFIG.gait, ...cfg.gaitParams };
    const gait = {
      period: Dg.period * Math.sqrt(s),
      strideM: Dg.strideM * s,
      liftM: Dg.liftM * s,
      standM: Dg.standM * s,
      stanceDuty: Dg.stanceDuty,
    };
    this.dyn = resolveConfig({ ...scaledBodyOverrides(cfg.mass, cfg.torqueCapNm), gait });
    this.substeps = Math.max(1, Math.round(this.dyn.substeps));
    this.physicsDt = this.dyn.dt / this.substeps;
    // 立脚で足が後方へ stride 掃く間に胴が stride 進む速度 = 足が世界で固定される基準速度。
    this.vBody = gait.strideM / (gait.stanceDuty * gait.period);

    // 静止立脚姿勢: 足先を hip 直下の standFrac*reach に置く IK（trunkPitch=0 基準・end-to-end モード用）。
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
    this.prevResidual = new Float64Array(this.actDim);
    this.curResidual = new Float64Array(this.actDim);
    this.residualBlend = new Float64Array(this.actDim);
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
    this.simTime = 0;
    this.stepCount = 0;
    this.prevX = this.startX;
    // 残差ゼロから始める（action=0 で土台の歩容そのもの／end-to-end なら静止立脚）。
    this.prevResidual.fill(0);
    this.curResidual.fill(0);
    return this.writeObs();
  }

  /**
   * 脚 legIndex の「土台ターゲット」（残差を足す前の中心）を返す。
   * baseGait=true は IK クロール歩容（胴速度レギュレータ付き・scripted 歩容と同じ）、false は静止立脚。
   */
  private baseTargets(
    legIndex: number,
    trunkPitch: number,
    trunkX: number,
  ): { hip: number; knee: number } {
    if (!this.cfg.baseGait) return { hip: this.hip0, knee: this.knee0 };
    const g = this.dyn.gait;
    // 胴速度レギュレータ: 足を世界の接地点に収束させ過走/滑りを抑える（±stride で clamp）。
    const bodyErrX = Math.max(
      -g.strideM,
      Math.min(g.strideM, trunkX - this.startX - this.vBody * this.simTime),
    );
    const { fx, fz } = footTargetRelHip(this.dyn, this.simTime, this.asm.legs[legIndex].phase);
    const { p1, p2 } = legIK(fx + bodyErrX, fz, this.dyn.leg.thigh, this.dyn.leg.shin, KNEE_SIGN);
    return { hip: p1 - trunkPitch, knee: p2 - p1 };
  }

  step(action: ArrayLike<number>): StepResult {
    // 行動 → 8 関節の残差（土台ターゲットに上乗せする補正・±maxDelta で squash）。
    for (let j = 0; j < this.actDim; j++) {
      this.curResidual[j] = this.maxDelta[j] * tanh(action[j] ?? 0);
    }

    const totalSub = this.cfg.controlFrames * this.substeps;
    const clockW = TAU * this.cfg.clockFreq;
    let satCount = 0;
    let satSamples = 0;
    for (let sub = 0; sub < totalSub; sub++) {
      // 残差を前ステップ値から新値へ線形補間（角速度スパイクを避ける）。土台ターゲットは毎サブステップ再計算。
      const alpha = (sub + 1) / totalSub;
      for (let j = 0; j < this.actDim; j++) {
        this.residualBlend[j] =
          this.prevResidual[j] + alpha * (this.curResidual[j] - this.prevResidual[j]);
      }
      const trunkPitch = pitchAboutY(this.asm.trunk);
      const trunkX = this.asm.trunk.translation().x;
      for (let leg = 0; leg < 4; leg++) {
        const L = this.asm.legs[leg];
        const base = this.baseTargets(leg, trunkPitch, trunkX);
        const hip = driveJoint(
          L.hipJoint,
          this.asm.trunk,
          L.thigh,
          base.hip + this.residualBlend[leg * 2],
          MOTOR_AXIS_SIGN,
          this.dyn.motor,
          this.physicsDt,
        );
        const knee = driveJoint(
          L.kneeJoint,
          L.thigh,
          L.shin,
          base.knee + this.residualBlend[leg * 2 + 1],
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
      this.simTime += this.physicsDt;
      this.phase += clockW * this.physicsDt;
    }
    // 次ステップの補間始点として確定。
    this.prevResidual.set(this.curResidual);

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
