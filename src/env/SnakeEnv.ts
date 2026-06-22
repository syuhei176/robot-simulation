/**
 * 3D 蛇（MuJoCo）の強化学習環境 — 実機センサー相当の観測で「任意方位へ操舵しながら地形を走る」汎用方策を学ぶ。
 *
 * `runSnake3D` の制御（パラメタ化セルペノイド波 PD ＋ 車輪相当の異方力場 ＋ 実接触）を「1 制御ステップずつ
 * 進められる」形にほどき、**残差RL**で駆動する: 各関節の目標角 = 前進登坂歩容のターゲット(時間依存) + 方策の残差
 * (±maxJointDelta·tanh(action))。土台の歩容（alt-yaw-pitch・yaw前進＋pitch持ち上げ）は action=0 でも障害物＋
 * 階段を越えるので、方策は「いつ・どの関節を余分に動かして段を越え、目標方位へ操舵し直すか」を学べばよい。
 *
 * **観測は実機（サーボ＋頭IMU）で取れる信号だけ**にしてある（sim-to-real）: 関節角/速度/負荷（present
 * position/velocity/load）＋頭IMU（姿勢・ジャイロ）＋位相クロック＋目標ヘディング誤差。シミュ特権情報
 * （前方地形プレビュー・絶対COM位置・頭尾から計算する厳密ヘディング）は使わない。
 *
 * 「目的地」は絶対(x,y)ではなく**目標ヘディング指令（相対方位）**で与える（localization 不要＝実機で成立）。
 * 報酬は「指令方向への前進 − 指令光線からの横ずれ − エネルギー/トルク飽和 ＋ 生存」。目標方位=0 が直進に一致。
 * エピソード毎に目標方位をランダム化して操舵を学ぶ。物理は `runSnake3D` と同じ z-up・MuJoCo。
 */
import { getMujoco, type MujocoModule } from '../sim3d/mujoco-engine.ts';
import {
  buildMjcf,
  qRotate,
  expandAxes,
  snakeTerrainTopAt,
  makeProgressionTerrain,
  DEFAULT_SNAKE3D_CONFIG,
  type Snake3DConfig,
  type JointAxis,
  type SnakeBodyLayout,
  type Snake3DReplay,
  type SnakeFrame,
  type SnakeTerrainBox,
} from '../sim3d/snake3d-dynamics.ts';
import type { RLEnv, StepResult } from '../rl/RLEnv.ts';

export interface SnakeEnvConfig {
  sim: Snake3DConfig; // 身体・歩容・地形・モータ（既定は進行性地形＋前進登坂歩容）
  // ドメインランダム化用の地形バンク（任意）。指定すると reset 毎にこの中から1つをランダムに選ぶ
  //（各地形のモデルを事前コンパイルして保持）。コース汎用な単一方策を学習するために使う。未指定なら sim.terrain 固定。
  terrainBank?: SnakeTerrainBox[][];
  episodeSteps: number; // 1 エピソードの制御ステップ数
  controlFrames: number; // 1 制御ステップが含む sim.dt フレーム数（制御周期 = controlFrames·dt）
  maxJointDelta: number; // 関節残差の可動幅 [rad]（土台ターゲット中心の ±）
  // 操舵行動のスケール [rad]: 方策の最後の行動次元（曲率指令）を ±steerActionScale·tanh(a) の yaw 定常バイアスへ
  // 写像し、全 yaw 関節へ一律加算する。これで方策は「目標方位に応じて体をどれだけ曲げるか」を残差とは独立に学べる
  //（関節残差 ±maxJointDelta では操舵に必要な ~0.2rad を出すとうねりの余地が無くなるため、操舵は専用次元に分離）。
  steerActionScale: number;
  headingMaxRad: number; // 目標ヘディングのランダム化範囲 ±[rad]（操舵の学習。0 なら常に直進=+x）
  headYawEmaTau: number; // 頭IMU yaw の EMA フィルタ時定数 [s]（うねりの振動を均して走行方位を推定）
  forwardReward: number; // 指令方向への前進量への係数
  lateralPenalty: number; // 指令方向に対する横速度 |⊥Δ| への係数（大 veer の矯正）
  centerlinePenalty: number; // 指令光線（開始点を通る方位線）からの横ずれ |⊥offset| への係数（操舵の追従）
  centerlineCap: number; // 横ずれ罰を頭打ちにする距離 [m]（学習初期の暴走防止）
  energyPenalty: number; // 行動エネルギー（残差二乗平均）への係数
  satPenalty: number; // トルク飽和率への係数
  aliveBonus: number; // 生存ボーナス
}

/** 既定: 進行性地形＋前進登坂歩容（障害物＋階段を越える土台）＋ ±30° の操舵ランダム化。 */
export function defaultSnakeEnvConfig(overrides: Partial<Snake3DConfig> = {}): SnakeEnvConfig {
  const sim: Snake3DConfig = {
    ...DEFAULT_SNAKE3D_CONFIG,
    pattern: 'alt-yaw-pitch',
    yawAmp: 0.6,
    pitchAmp: 0.35,
    yawPitchPhase: 0,
    waveLength: 10,
    period: 1.2,
    waveSign: -1, // +x へ進む
    terrain: makeProgressionTerrain(),
    duration: 0, // env が制御するので未使用
    ...overrides,
  };
  return {
    sim,
    episodeSteps: 500,
    controlFrames: 5, // 制御周期 = 5/240 ≈ 48 Hz
    // 残差幅。残差RLは action=0 で良い基盤（前進）なので、必要なのは段越えの追加動作と操舵補正だけ。
    // 幅を絞ると方策が基盤から大きく逸脱できず、学習の「床」が基盤付近に保たれて崩壊（後退）しにくい。
    maxJointDelta: 0.2,
    // 操舵行動のスケール。実測校正（手動）で yaw 定常バイアス 0.22rad で明確に曲がり 0.45rad で巻き込む。
    // 余裕を持って ±0.3rad を上限に与え、方策は前進報酬とのトレードオフで必要なぶんだけ使う（直進時は ~0）。
    steerActionScale: 0.3,
    headingMaxRad: 0.5236, // ±30°。エピソード毎に目標方位を一様サンプル（θ≈0 も含み「直進時は操舵しない」を学ぶ）
    headYawEmaTau: 0.6, // ≈半歩容周期。頭yawはうねりで±30°超振れるので均して平均方位を取り出す
    forwardReward: 20,
    // **大veer の矯正**: per-step の指令直交速度 |⊥Δ| への強い罰。ドリフト「率」を罰するので大きく逸れた時の
    // 矯正圧が強い（challenge の大 veer をここで綺麗に直す）。速度比例＝有界で PPO 安定。
    lateralPenalty: 8.0,
    // **定常ドリフトの矯正**: 指令光線からの横ずれ（積分距離）を罰す。**cap を広く**取り（3.0m）「ずれるほど悪い」
    // 勾配をエピソード全域で効かせる＝平地の小さな定常ドリフト(±13°)も直させる（cap 0.8 では即飽和し勾配が消えて
    // いた）。飽和値 centerlinePenalty·cap ≈ 前進報酬1step分に収め、無制限罰の崩壊(S14)も避ける（有界で安定）。
    centerlinePenalty: 0.03,
    centerlineCap: 3.0,
    energyPenalty: 0.002,
    satPenalty: 0.05,
    aliveBonus: 0.01,
  };
}

const tanh = Math.tanh;
function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

type Quat = [number, number, number, number]; // x,y,z,w

/** クォータニオン共役（[x,y,z,w]）。 */
function quatConj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** クォータニオン積 a⊗b（[x,y,z,w]）。 */
function quatMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export class SnakeEnv implements RLEnv {
  readonly obsDim: number;
  readonly actDim: number;

  private readonly cfg: SnakeEnvConfig;
  private readonly sim: Snake3DConfig;
  private readonly mj: MujocoModule;
  private readonly axes: JointAxis[];
  private readonly n: number;
  private readonly nJoints: number;
  private readonly halfW: number;
  private readonly substeps: number;
  private readonly physicsDt: number;
  private readonly spatialPhase: number;
  private readonly groundGate: number;
  private readonly layout: SnakeBodyLayout[];
  private readonly emaAlpha: number; // 頭yaw EMA の係数（controlPeriod / (tau+controlPeriod)）

  // MuJoCo モデルは地形バンクの各地形ごとに1回だけコンパイルして保持、data はエピソード毎に作り直す。
  // ドメインランダム化では reset 毎に地形（=モデル）をランダムに切り替える。バンク未指定なら要素1個。
  private readonly terrains: SnakeTerrainBox[][];
  private readonly models: Array<ReturnType<MujocoModule['MjModel']['from_xml_string']>>;
  private model: ReturnType<MujocoModule['MjModel']['from_xml_string']>;
  private activeTerrain: SnakeTerrainBox[];
  private data: InstanceType<MujocoModule['MjData']>;

  private readonly residual: Float64Array;
  private readonly jointTau: Float64Array; // 直近に印加した関節トルク（present load 観測用）
  private readonly prevPos: Array<[number, number, number]>;
  private readonly obs: Float32Array;
  private simSubstep = 0;
  private stepCount = 0;
  private startComX = 0;
  private startComY = 0;
  private prevComX = 0;
  private prevComY = 0;

  // 操舵指令（相対方位）と、頭IMU の派生状態。
  private cmdHeadingRad = 0;
  private cmdDirX = 1;
  private cmdDirY = 0;
  // 操舵の yaw 定常バイアス [rad]。yaw 関節の土台ターゲットへ一律加算して体を一定曲率で曲げる。
  // 毎ステップ方策の操舵行動（最後の行動次元）から設定される＝曲がる動きを RL が獲得する（外部からは与えない）。
  private yawSteerBias = 0;
  private yawEmaCos = 1; // 頭yaw を単位ベクトルで EMA（角度のラップ回避）
  private yawEmaSin = 0;
  private prevHeadQuat: Quat = [0, 0, 0, 1]; // ジャイロ（角速度）算出用の前ステップ頭姿勢

  // 記録（決定論ロールアウト → ダッシュボード再生用）。
  private recording = false;
  private recordEvery = 2;
  private frames: SnakeFrame[] = [];
  private epMaxDemand = 0;
  private epMaxApplied = 0;
  private epSatSteps = 0;

  private constructor(mj: MujocoModule, cfg: SnakeEnvConfig) {
    this.mj = mj;
    this.cfg = cfg;
    this.sim = cfg.sim;
    this.n = this.sim.n;
    this.nJoints = this.n - 1;
    this.halfW = this.sim.bodyWidth / 2;
    this.substeps = Math.max(1, Math.round(this.sim.substeps));
    this.physicsDt = this.sim.dt / this.substeps;
    this.spatialPhase = (2 * Math.PI) / this.sim.waveLength;
    this.groundGate = this.halfW * 1.6;
    this.axes = expandAxes(this.sim.pattern, this.nJoints);
    this.layout = Array.from({ length: this.n }, () => ({
      half: [this.sim.segLen / 2, this.halfW, this.halfW] as [number, number, number],
    }));
    const controlPeriod = cfg.controlFrames * this.sim.dt;
    this.emaAlpha = controlPeriod / (cfg.headYawEmaTau + controlPeriod);

    // 地形バンク（未指定なら sim.terrain 単体）。各地形のモデルを事前コンパイル。
    this.terrains =
      cfg.terrainBank && cfg.terrainBank.length > 0 ? cfg.terrainBank : [this.sim.terrain];
    this.models = this.terrains.map((terrain) =>
      mj.MjModel.from_xml_string(buildMjcf({ ...this.sim, terrain }, this.axes, this.physicsDt)),
    );
    this.model = this.models[0];
    this.activeTerrain = this.terrains[0];
    this.data = new mj.MjData(this.model);

    // 行動 = 各関節の歩容残差(nJoints) ＋ 操舵（yaw 曲率）1次元。最後の次元が目標方位に応じた曲げ量を担う。
    this.actDim = this.nJoints + 1;
    // 観測（実機相当）: 関節角(nJoints)+関節速度(nJoints)+関節負荷(nJoints)
    //   +頭IMU姿勢[fwd xy, pitch, roll](4)+頭IMUジャイロ(3)+位相(2)+目標ヘディング誤差 sin/cos(2)。
    this.obsDim = 3 * this.nJoints + 11;
    this.obs = new Float32Array(this.obsDim);
    this.residual = new Float64Array(this.nJoints);
    this.jointTau = new Float64Array(this.nJoints);
    this.prevPos = Array.from({ length: this.n }, () => [0, 0, 0] as [number, number, number]);
  }

  static async create(cfg?: SnakeEnvConfig): Promise<SnakeEnv> {
    const mj = await getMujoco();
    const env = new SnakeEnv(mj, cfg ?? defaultSnakeEnvConfig());
    env.reset();
    return env;
  }

  get controlPeriod(): number {
    return this.cfg.controlFrames * this.sim.dt;
  }

  get maxSteps(): number {
    return this.cfg.episodeSteps;
  }

  /**
   * 実行中に目標ヘディング指令（相対方位）を差し替える（ライブ操舵用）。reset 時のみだった設定を
   * エピソードの途中でも更新できるようにする。観測の「目標ヘディング誤差」が即座に変わるので、
   * 次ステップから方策が新しい方位へ操舵し直す。報酬の指令座標系も追従する（ライブ再生では未使用）。
   */
  setCommandHeading(rad: number): void {
    this.cmdHeadingRad = rad;
    this.cmdDirX = Math.cos(rad);
    this.cmdDirY = Math.sin(rad);
  }

  /** 各リンクのカプセル寸法（ライブ描画でビューを組むのに使う）。 */
  getLayout(): SnakeBodyLayout[] {
    return this.layout;
  }

  /** 現在の各リンク姿勢（ライブ描画用・`SnakeFrame['bodies']` と同形）。 */
  currentBodies(): SnakeFrame['bodies'] {
    const xpos = this.data.xpos;
    const xquat = this.data.xquat;
    const bodies: SnakeFrame['bodies'] = [];
    for (let i = 0; i < this.n; i++) {
      const bid = this.bodyId(i);
      bodies.push({
        p: [xpos[bid * 3], xpos[bid * 3 + 1], xpos[bid * 3 + 2]],
        q: [xquat[bid * 4 + 1], xquat[bid * 4 + 2], xquat[bid * 4 + 3], xquat[bid * 4]],
      });
    }
    return bodies;
  }

  private bodyId(link: number): number {
    return link + 1; // world=0, L0=1, ...
  }

  private com(): [number, number] {
    const xpos = this.data.xpos;
    let x = 0;
    let y = 0;
    for (let i = 0; i < this.n; i++) {
      x += xpos[this.bodyId(i) * 3];
      y += xpos[this.bodyId(i) * 3 + 1];
    }
    return [x / this.n, y / this.n];
  }

  /** 頭リンク（先頭リンク n-1）の姿勢クォータニオン [x,y,z,w]（頭IMU 相当）。 */
  private headQuat(): Quat {
    const xquat = this.data.xquat;
    const bid = this.bodyId(this.n - 1);
    return [xquat[bid * 4 + 1], xquat[bid * 4 + 2], xquat[bid * 4 + 3], xquat[bid * 4]];
  }

  /** 指令方向への前進量（射影）。`progressMetric()-開始値` がコマンド方向の前進になる。 */
  progressMetric(): number {
    const [cx, cy] = this.com();
    return cx * this.cmdDirX + cy * this.cmdDirY;
  }

  /**
   * エピソードを初期化。地形バンクが複数なら毎回ランダムに地形（=事前コンパイル済みモデル）を選ぶ。
   * 目標ヘディングも毎回 ±headingMaxRad で一様ランダム化（操舵の学習）。`forceTerrainIdx`/`forceHeadingRad`
   * を渡せば固定（決定論評価・録画用）。
   */
  reset(forceTerrainIdx?: number, forceHeadingRad?: number): Float32Array {
    const idx =
      forceTerrainIdx !== undefined
        ? forceTerrainIdx
        : this.models.length > 1
          ? Math.floor(Math.random() * this.models.length)
          : 0;
    this.model = this.models[idx];
    this.activeTerrain = this.terrains[idx];
    this.data.delete();
    this.data = new this.mj.MjData(this.model);
    this.mj.mj_forward(this.model, this.data);

    // 目標ヘディング指令（相対方位）。0=+x。±headingMaxRad の一様サンプル（forceHeadingRad で固定）。
    this.cmdHeadingRad =
      forceHeadingRad !== undefined
        ? forceHeadingRad
        : (Math.random() * 2 - 1) * this.cfg.headingMaxRad;
    this.cmdDirX = Math.cos(this.cmdHeadingRad);
    this.cmdDirY = Math.sin(this.cmdHeadingRad);

    const xpos = this.data.xpos;
    for (let i = 0; i < this.n; i++) {
      const bid = this.bodyId(i);
      this.prevPos[i][0] = xpos[bid * 3];
      this.prevPos[i][1] = xpos[bid * 3 + 1];
      this.prevPos[i][2] = xpos[bid * 3 + 2];
    }
    const [cx, cy] = this.com();
    this.startComX = cx;
    this.startComY = cy;
    this.prevComX = cx;
    this.prevComY = cy;
    this.simSubstep = 0;
    this.stepCount = 0;
    this.residual.fill(0);
    this.yawSteerBias = 0;
    this.jointTau.fill(0);

    // 頭IMU 派生状態の初期化: yaw EMA を初期 yaw に、ジャイロ用の前姿勢を現姿勢に。
    const hq = this.headQuat();
    const fwd0 = qRotate(hq, [1, 0, 0]);
    const yaw0 = Math.atan2(fwd0[1], fwd0[0]);
    this.yawEmaCos = Math.cos(yaw0);
    this.yawEmaSin = Math.sin(yaw0);
    this.prevHeadQuat = hq;

    this.frames = [];
    this.epMaxDemand = 0;
    this.epMaxApplied = 0;
    this.epSatSteps = 0;
    return this.writeObs();
  }

  enableRecording(recordEvery = 2): void {
    this.recording = true;
    this.recordEvery = Math.max(1, Math.round(recordEvery));
    this.frames = [];
  }

  getReplay(): Snake3DReplay {
    const [cx, cy] = this.com();
    const dx = cx - this.startComX;
    const dy = cy - this.startComY; // 開始からの正味変位
    const travelM = Math.hypot(dx, dy);
    const forwardProj = dx * this.cmdDirX + dy * this.cmdDirY; // 指令方向への正味前進
    return {
      layout: this.layout,
      frames: this.frames,
      summary: {
        // 録画再生用の地形は「このエピソードで実際に使った地形」（バンクからランダム選択された地形）。
        config: {
          ...this.sim,
          terrain: this.activeTerrain,
          duration: this.simSubstep * this.physicsDt,
        },
        travelM,
        netDispM: [dx, dy],
        headingDeg: (Math.atan2(dy, dx) * 180) / Math.PI, // 達成方位（実測）
        maxDemandNm: this.epMaxDemand,
        maxAppliedNm: this.epMaxApplied,
        saturatedSteps: this.epSatSteps,
        success: forwardProj > this.sim.segLen * this.n * 0.5,
      },
    };
  }

  /** 関節 j の土台ターゲット（残差を足す前のセルペノイド波）。runSnake3D の歩容と同式＋yaw 操舵バイアス。 */
  private baseTarget(j: number, ts: number): number {
    const isYaw = this.axes[j] === 'yaw';
    const amp = isYaw ? this.sim.yawAmp : this.sim.pitchAmp;
    const phase = isYaw ? 0 : this.sim.yawPitchPhase;
    const bias = isYaw ? this.yawSteerBias : 0; // yaw 関節へ一律オフセット＝方策の操舵行動が与える一定曲率
    return (
      amp *
        Math.sin(
          (2 * Math.PI * ts) / this.sim.period - this.sim.waveSign * j * this.spatialPhase + phase,
        ) +
      bias
    );
  }

  step(action: ArrayLike<number>): StepResult {
    // 前 nJoints は各関節の歩容残差、最後の1次元は操舵（yaw 曲率）。
    for (let j = 0; j < this.nJoints; j++) {
      this.residual[j] = this.cfg.maxJointDelta * tanh(action[j] ?? 0);
    }
    this.yawSteerBias = this.cfg.steerActionScale * tanh(action[this.nJoints] ?? 0);

    let stepDemand = 0;
    let stepApplied = 0;
    let satCount = 0;
    let satSamples = 0;

    for (let frame = 0; frame < this.cfg.controlFrames; frame++) {
      // 車輪相当の異方力場（dt 毎に更新・局所地形上面に接地するリンクのみ）。runSnake3D と同式。
      const drive: Array<[number, number]> = [];
      {
        const xpos = this.data.xpos;
        const xquat = this.data.xquat;
        for (let i = 0; i < this.n; i++) {
          const bid = this.bodyId(i);
          const px = xpos[bid * 3];
          const py = xpos[bid * 3 + 1];
          const pz = xpos[bid * 3 + 2];
          const vx = (px - this.prevPos[i][0]) / this.sim.dt;
          const vy = (py - this.prevPos[i][1]) / this.sim.dt;
          this.prevPos[i][0] = px;
          this.prevPos[i][1] = py;
          this.prevPos[i][2] = pz;
          const top =
            this.activeTerrain.length > 0 ? snakeTerrainTopAt(this.activeTerrain, px, py) : 0;
          if (pz - top > this.groundGate) {
            drive.push([0, 0]);
            continue;
          }
          const q: Quat = [
            xquat[bid * 4 + 1],
            xquat[bid * 4 + 2],
            xquat[bid * 4 + 3],
            xquat[bid * 4],
          ];
          const fwd = qRotate(q, [1, 0, 0]);
          const lat = qRotate(q, [0, 1, 0]);
          const vFwd = vx * fwd[0] + vy * fwd[1];
          const vLat = vx * lat[0] + vy * lat[1];
          const ff = -this.sim.kFwd * vFwd;
          const fl = -this.sim.kLat * vLat;
          drive.push([fwd[0] * ff + lat[0] * fl, fwd[1] * ff + lat[1] * fl]);
        }
      }

      for (let sub = 0; sub < this.substeps; sub++) {
        const qpos = this.data.qpos;
        const qvel = this.data.qvel;
        const qfrc = this.data.qfrc_applied;
        const xfrc = this.data.xfrc_applied;
        const ts = this.simSubstep * this.physicsDt;
        for (let j = 0; j < this.nJoints; j++) {
          const target = this.baseTarget(j, ts) + this.residual[j];
          const angle = qpos[7 + j];
          const vel = qvel[6 + j];
          const raw = this.sim.motor.stiffness * (target - angle) - this.sim.motor.damping * vel;
          const tau = clamp(raw, -this.sim.motor.maxTorqueNm, this.sim.motor.maxTorqueNm);
          qfrc[6 + j] = tau;
          this.jointTau[j] = tau; // present load（最後のサブステップの印加トルク）を観測へ
          stepDemand = Math.max(stepDemand, Math.abs(raw));
          stepApplied = Math.max(stepApplied, Math.abs(tau));
          satSamples++;
          if (Math.abs(raw) - Math.abs(tau) > 1e-9) satCount++;
        }
        for (let i = 0; i < this.n; i++) {
          const bid = this.bodyId(i);
          xfrc[bid * 6 + 0] = drive[i][0];
          xfrc[bid * 6 + 1] = drive[i][1];
          xfrc[bid * 6 + 2] = 0;
        }
        this.mj.mj_step(this.model, this.data);
        this.simSubstep++;
      }
    }

    // --- 報酬（指令方向 cmdDir の座標系で評価） ---
    const [cx, cy] = this.com();
    const dx = cx - this.prevComX;
    const dy = cy - this.prevComY;
    this.prevComX = cx;
    this.prevComY = cy;
    const forwardProj = dx * this.cmdDirX + dy * this.cmdDirY; // 指令方向への前進
    const lateralProj = -dx * this.cmdDirY + dy * this.cmdDirX; // 指令直交（横ずれ速度）

    // エネルギー罰は関節残差のみ（操舵行動は罰さない＝曲がることをコストにしない）。
    let energy = 0;
    for (let j = 0; j < this.nJoints; j++) {
      const a = tanh(action[j] ?? 0);
      energy += a * a;
    }
    energy /= this.nJoints;
    const satFrac = satSamples > 0 ? satCount / satSamples : 0;

    // 指令光線（開始点を通る方位線）からの横ずれ＝追従誤差の積分。上限付き（暴走防止）。
    const ox = cx - this.startComX;
    const oy = cy - this.startComY;
    const crossOffset = -ox * this.cmdDirY + oy * this.cmdDirX;
    const offMag = Math.min(Math.abs(crossOffset), this.cfg.centerlineCap);
    const blewUp = !Number.isFinite(cx) || !Number.isFinite(cy) || Math.abs(cx) > 50;
    let reward =
      this.cfg.forwardReward * forwardProj -
      this.cfg.lateralPenalty * Math.abs(lateralProj) -
      this.cfg.centerlinePenalty * offMag -
      this.cfg.energyPenalty * energy -
      this.cfg.satPenalty * satFrac +
      this.cfg.aliveBonus;
    if (blewUp) reward -= 5;

    this.stepCount++;
    const done = blewUp || this.stepCount >= this.cfg.episodeSteps;

    if (this.recording) {
      const saturated = satCount > 0;
      this.epMaxDemand = Math.max(this.epMaxDemand, stepDemand);
      this.epMaxApplied = Math.max(this.epMaxApplied, stepApplied);
      if (saturated) this.epSatSteps++;
      if (this.stepCount % this.recordEvery === 0 || done) {
        const xpos = this.data.xpos;
        const xquat = this.data.xquat;
        const bodies: SnakeFrame['bodies'] = [];
        for (let i = 0; i < this.n; i++) {
          const bid = this.bodyId(i);
          bodies.push({
            p: [xpos[bid * 3], xpos[bid * 3 + 1], xpos[bid * 3 + 2]],
            q: [xquat[bid * 4 + 1], xquat[bid * 4 + 2], xquat[bid * 4 + 3], xquat[bid * 4]],
          });
        }
        this.frames.push({
          t: this.simSubstep * this.physicsDt,
          bodies,
          diag: {
            travelM: Math.hypot(ox, oy), // 開始からの総移動距離（向きに依らない）
            demandNm: stepDemand,
            appliedNm: stepApplied,
            saturated,
          },
        });
      }
    }

    return { obs: this.writeObs(), reward, done };
  }

  private writeObs(): Float32Array {
    const o = this.obs;
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;

    // 関節角・関節速度・関節負荷（実機サーボの present position / velocity / load）。
    let k = 0;
    for (let j = 0; j < this.nJoints; j++) o[k++] = qpos[7 + j]; // 角 [rad]
    for (let j = 0; j < this.nJoints; j++) o[k++] = qvel[6 + j] / 10; // 速度（粗く正規化）
    for (let j = 0; j < this.nJoints; j++) o[k++] = this.jointTau[j] / this.sim.motor.maxTorqueNm; // 負荷 τ/cap（符号付き [-1,1]）

    // 頭IMU: 姿勢（先頭リンクのワールド姿勢から、IMU が出せる量だけ）。
    const hq = this.headQuat();
    const fwd = qRotate(hq, [1, 0, 0]); // 機体前方の向き
    const lat = qRotate(hq, [0, 1, 0]); // 機体左方の向き
    o[k++] = fwd[0]; // yaw 単位ベクトル x（生の頭向き＝うねりで振れる）
    o[k++] = fwd[1]; // yaw 単位ベクトル y
    o[k++] = fwd[2]; // ピッチ指標（鼻先の上下＝段差で持ち上がると +）
    o[k++] = lat[2]; // ロール指標（機体横軸の上下）

    // 頭IMU: 角速度（ジャイロ）。前ステップ→現ステップの頭姿勢差分を body frame で取り、制御周期で割る。
    const relq = quatMul(quatConj(this.prevHeadQuat), hq); // body-frame の微小回転
    const sign = relq[3] >= 0 ? 1 : -1; // double-cover を解消
    const invDt = 1 / (this.controlPeriod * 5); // ≈±数rad/s を ~O(1) に正規化
    o[k++] = 2 * sign * relq[0] * invDt; // ωx (roll rate)
    o[k++] = 2 * sign * relq[1] * invDt; // ωy (pitch rate)
    o[k++] = 2 * sign * relq[2] * invDt; // ωz (yaw rate)
    this.prevHeadQuat = hq;

    // 位相クロック（コントローラ内部時計）。
    const ts = this.simSubstep * this.physicsDt;
    o[k++] = Math.sin((2 * Math.PI * ts) / this.sim.period);
    o[k++] = Math.cos((2 * Math.PI * ts) / this.sim.period);

    // 目標ヘディング誤差（操舵指令）: 指令方位 − EMA フィルタした頭yaw。頭yaw はうねりで振れるので
    // 単位ベクトルで EMA して平均方位（≈走行方位）を取り出し、指令との差を sin/cos で渡す。
    const yawNow = Math.atan2(fwd[1], fwd[0]);
    this.yawEmaCos += this.emaAlpha * (Math.cos(yawNow) - this.yawEmaCos);
    this.yawEmaSin += this.emaAlpha * (Math.sin(yawNow) - this.yawEmaSin);
    const yawFilt = Math.atan2(this.yawEmaSin, this.yawEmaCos);
    const headErr = this.cmdHeadingRad - yawFilt;
    o[k++] = Math.sin(headErr);
    o[k++] = Math.cos(headErr);

    for (let i = 0; i < this.obsDim; i++) if (!Number.isFinite(o[i])) o[i] = 0;
    return o;
  }

  dispose(): void {
    this.data.delete();
    for (const m of this.models) m.delete(); // バンクの全モデルを解放
  }
}
