/**
 * 3D 蛇（MuJoCo）の強化学習環境 — 進行性地形（小障害物 → 階段 → テーブル）の走破。
 *
 * `runSnake3D` の制御（パラメタ化セルペノイド波 PD ＋ 車輪相当の異方力場 ＋ 実接触）を「1 制御ステップずつ
 * 進められる」形にほどき、**残差RL**で駆動する: 各関節の目標角 = 前進登坂歩容のターゲット(時間依存) + 方策の残差
 * (±maxJointDelta·tanh(action))。土台の歩容（alt-yaw-pitch・yaw前進＋pitch持ち上げ）は action=0 でも障害物＋
 * 階段を越えるので（Stage1 で検証済み）、方策は「いつ・どの関節を余分に持ち上げ/曲げて段を越えるか」を、
 * 前方地形プレビュー観測を手がかりに学べばよい。決定論方策も最初から前進する＝end-to-end の縮退を回避。
 *
 * 物理は `runSnake3D` と同じ z-up・MuJoCo。観測=位相クロック＋COM速度＋頭クリアランス＋頭ピッチ＋前方地形
 * プレビュー＋関節角。報酬=前進(+x) − 横ドリフト − 行動エネルギー − トルク飽和 ＋ 生存。数値破綻で終了。
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
  previewOffsets: number[]; // 前方地形プレビューの COM 前方オフセット [m]
  forwardReward: number; // 前進(+x)量への係数
  lateralPenalty: number; // 横ドリフト速度 |Δy| への係数（うねりの横揺れを軽く抑える）
  centerlinePenalty: number; // センターライン（開始 y）からの絶対オフセット |y−y0| への係数（直進補正）
  centerlineCap: number; // センターライン罰を頭打ちにする距離 [m]（学習初期の暴走防止）
  energyPenalty: number; // 行動エネルギー（残差二乗平均）への係数
  satPenalty: number; // トルク飽和率への係数
  aliveBonus: number; // 生存ボーナス
}

/** 既定: 進行性地形＋前進登坂歩容（Stage1 で障害物＋階段を越えると確認した土台）。 */
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
    // 残差幅。残差RLは action=0 で良い基盤（直進チャレンジで前進 555cm）なので、必要なのは小さな操舵補正だけ。
    // 幅を絞ると方策が基盤から大きく逸脱できず、学習の「床」が基盤付近に保たれて崩壊（後退）しにくい。
    maxJointDelta: 0.2,
    previewOffsets: [0.2, 0.4, 0.7, 1.0],
    forwardReward: 20,
    lateralPenalty: 1.0, // per-step 横速度の軽い減衰（うねり自体は許す）。直進補正は centerlinePenalty が担う
    // |y−y0| を罰して斜行を中心へ戻させる。ただし上限付き（centerlineCap で飽和）＝学習初期に大ドリフトしても
    // 罰が暴走せず（無制限だと基盤の -4.5m で penalty が前進報酬を桁違いに上回り PPO が崩壊する）、前進報酬が支配を保つ。
    centerlinePenalty: 0.1,
    centerlineCap: 0.6, // この距離[m]で罰を頭打ちに（最大 centerlinePenalty·centerlineCap ≈ 前進報酬と同程度）
    energyPenalty: 0.002,
    satPenalty: 0.05,
    aliveBonus: 0.01,
  };
}

const tanh = Math.tanh;
function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
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

  // MuJoCo モデルは地形バンクの各地形ごとに1回だけコンパイルして保持、data はエピソード毎に作り直す。
  // ドメインランダム化では reset 毎に地形（=モデル）をランダムに切り替える。バンク未指定なら要素1個。
  private readonly terrains: SnakeTerrainBox[][];
  private readonly models: Array<ReturnType<MujocoModule['MjModel']['from_xml_string']>>;
  private model: ReturnType<MujocoModule['MjModel']['from_xml_string']>;
  private activeTerrain: SnakeTerrainBox[];
  private data: InstanceType<MujocoModule['MjData']>;

  private readonly residual: Float64Array;
  private readonly prevPos: Array<[number, number, number]>;
  private readonly obs: Float32Array;
  private simSubstep = 0;
  private stepCount = 0;
  private startComX = 0;
  private startComY = 0;
  private prevComX = 0;
  private prevComY = 0;

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

    // 地形バンク（未指定なら sim.terrain 単体）。各地形のモデルを事前コンパイル。
    this.terrains =
      cfg.terrainBank && cfg.terrainBank.length > 0 ? cfg.terrainBank : [this.sim.terrain];
    this.models = this.terrains.map((terrain) =>
      mj.MjModel.from_xml_string(buildMjcf({ ...this.sim, terrain }, this.axes, this.physicsDt)),
    );
    this.model = this.models[0];
    this.activeTerrain = this.terrains[0];
    this.data = new mj.MjData(this.model);

    this.actDim = this.nJoints; // 各関節の歩容残差
    // 観測: 位相(2)+COM速度(2)+頭クリアランス(1)+頭ピッチ(1)+横オフセット(1)+体軸ヘディング(2)
    //       +前方地形プレビュー(K)+関節角(nJoints)。横オフセット＋ヘディングが直進補正の閉ループ信号。
    this.obsDim = 9 + cfg.previewOffsets.length + this.nJoints;
    this.obs = new Float32Array(this.obsDim);
    this.residual = new Float64Array(this.actDim);
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

  /** 歩行面（足場/壁）の地形上面 [m]。天板など頭上の張り出し（底面が高い箱）は除外する。 */
  private groundTopAt(x: number, y: number): number {
    let top = 0;
    for (const b of this.activeTerrain) {
      if (b.cz - b.halfZ > 0.12) continue; // 頭上の張り出し（テーブル天板）は歩行面でない
      if (Math.abs(x - b.cx) <= b.halfX && Math.abs(y - b.cy) <= b.halfY) {
        top = Math.max(top, b.cz + b.halfZ);
      }
    }
    return top;
  }

  progressMetric(): number {
    return this.com()[0];
  }

  /**
   * エピソードを初期化。地形バンクが複数なら毎回ランダムに地形（=事前コンパイル済みモデル）を選ぶ
   *（ドメインランダム化）。`forceTerrainIdx` を渡せば特定地形に固定（決定論評価・録画用）。
   */
  reset(forceTerrainIdx?: number): Float32Array {
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
    const dy = cy - this.startComY; // 開始からの正味横ドリフト
    const travelM = Math.hypot(dx, dy);
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
        headingDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
        maxDemandNm: this.epMaxDemand,
        maxAppliedNm: this.epMaxApplied,
        saturatedSteps: this.epSatSteps,
        success: dx > this.sim.segLen * this.n * 0.5,
      },
    };
  }

  /** 関節 j の土台ターゲット（残差を足す前のセルペノイド波）。runSnake3D の歩容と同式。 */
  private baseTarget(j: number, ts: number): number {
    const isYaw = this.axes[j] === 'yaw';
    const amp = isYaw ? this.sim.yawAmp : this.sim.pitchAmp;
    const phase = isYaw ? 0 : this.sim.yawPitchPhase;
    return (
      amp *
      Math.sin(
        (2 * Math.PI * ts) / this.sim.period - this.sim.waveSign * j * this.spatialPhase + phase,
      )
    );
  }

  step(action: ArrayLike<number>): StepResult {
    for (let j = 0; j < this.actDim; j++) {
      this.residual[j] = this.cfg.maxJointDelta * tanh(action[j] ?? 0);
    }

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
          const q: [number, number, number, number] = [
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

    // --- 報酬 ---
    const [cx, cy] = this.com();
    const dx = cx - this.prevComX;
    const dy = cy - this.prevComY;
    this.prevComX = cx;
    this.prevComY = cy;

    let energy = 0;
    for (let j = 0; j < this.actDim; j++) {
      const a = tanh(action[j] ?? 0);
      energy += a * a;
    }
    energy /= this.actDim;
    const satFrac = satSamples > 0 ? satCount / satSamples : 0;

    const offset = cy - this.startComY; // センターライン（開始 y）からの横ずれ
    const offMag = Math.min(Math.abs(offset), this.cfg.centerlineCap); // 上限付き（暴走防止）
    const blewUp = !Number.isFinite(cx) || !Number.isFinite(cy) || Math.abs(cx) > 50;
    let reward =
      this.cfg.forwardReward * dx -
      this.cfg.lateralPenalty * Math.abs(dy) -
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
            travelM: cx - this.startComX,
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
    const [cx, cy] = this.com();
    const xpos = this.data.xpos;
    const xquat = this.data.xquat;
    const headBid = this.bodyId(this.n - 1);
    const headX = xpos[headBid * 3];
    const headY = xpos[headBid * 3 + 1];
    const headZ = xpos[headBid * 3 + 2];
    const hq: [number, number, number, number] = [
      xquat[headBid * 4 + 1],
      xquat[headBid * 4 + 2],
      xquat[headBid * 4 + 3],
      xquat[headBid * 4],
    ];
    const headFwd = qRotate(hq, [1, 0, 0]);

    const ts = this.simSubstep * this.physicsDt;
    o[0] = Math.sin((2 * Math.PI * ts) / this.sim.period);
    o[1] = Math.cos((2 * Math.PI * ts) / this.sim.period);
    o[2] = (cx - this.prevComX) / 0.02; // 1 制御ステップの前進量を粗く正規化
    o[3] = (cy - this.prevComY) / 0.02;
    o[4] = (headZ - this.groundTopAt(headX, headY)) / 0.05; // 頭クリアランス
    o[5] = headFwd[2]; // 頭ピッチ（+1=真上, -1=真下）
    // 直進補正の閉ループ信号: センターライン（開始 y）からの横オフセットと、体軸ヘディング。
    o[6] = clamp((cy - this.startComY) / 0.5, -3, 3); // 横オフセット（どれだけ斜行したか）
    const tailBid = this.bodyId(0);
    let axX = headX - xpos[tailBid * 3];
    let axY = headY - xpos[tailBid * 3 + 1];
    const axLen = Math.hypot(axX, axY) || 1;
    axX /= axLen;
    axY /= axLen;
    o[7] = axX; // 体軸ヘディング x（真っ直ぐ +x なら ≈±1）
    o[8] = axY; // 体軸ヘディング y（veer 量＝符号付きで操舵方向が分かる）
    // 前方地形プレビュー: COM 前方各オフセットの歩行面高さ − COM 直下の歩行面高さ。
    const baseTop = this.groundTopAt(cx, cy);
    const pBase = 9;
    for (let i = 0; i < this.cfg.previewOffsets.length; i++) {
      o[pBase + i] = (this.groundTopAt(cx + this.cfg.previewOffsets[i], cy) - baseTop) / 0.06;
    }
    // 関節角。
    const qpos = this.data.qpos;
    const jBase = pBase + this.cfg.previewOffsets.length;
    for (let j = 0; j < this.nJoints; j++) o[jBase + j] = qpos[7 + j];

    for (let i = 0; i < this.obsDim; i++) if (!Number.isFinite(o[i])) o[i] = 0;
    return o;
  }

  dispose(): void {
    this.data.delete();
    for (const m of this.models) m.delete(); // バンクの全モデルを解放
  }
}
