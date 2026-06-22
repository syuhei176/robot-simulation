/**
 * 汎用 3D ヘビ（MuJoCo）— 関節構成（JointSpec）と歩容（gait）を分離した「キャンバス」。
 *
 * 関節構成は宣言的な軸パターンで決める:
 *  - all-yaw       … 全関節が水平(z軸)。平面の横うねり。
 *  - all-pitch     … 全関節が垂直(y軸)。縦うねり・尺取り（段差越え）。
 *  - alt-yaw-pitch … yaw/pitch 交互。3D 形状（サイドワインド・螺旋）が出せる“歩容コンプリート”な土台。
 * 歩容は `(時刻, 関節, 軸) → 目標角` のパラメタ化セルペノイド波で与える: yaw/pitch それぞれに振幅・位相を持たせ、
 * pitchAmp=0 で横うねり、yaw=pitch かつ位相差 π/2 でサイドワインド…と単一式で多歩容を表す。
 * 前進は車輪相当の異方抵抗（前方=低・横=高、接地リンクのみ）で生む。MuJoCo は接線異方摩擦を持たないため外力で与える。
 *
 * 物理は z-up。レンダラは quad と同じ z-up→y-up 変換。MuJoCo の xquat は (w,x,y,z) 順。
 */
import { getMujoco, type MujocoModule } from './mujoco-engine.ts';

/** 関節の軸パターン（モルフォロジーの宣言）。universal-2dof は将来拡張（関節2倍）。 */
export type AxisPattern = 'all-yaw' | 'all-pitch' | 'alt-yaw-pitch';
export type JointAxis = 'yaw' | 'pitch';

/**
 * 地形の箱（矢状面 x-z ＋ 横幅 y）。MuJoCo の box geom 兼、接地ゲート用の局所地形上面の真実。
 * 蛇は実接触で段に体を押し付けて登るので、地形は「異方力場の抽象」ではなく本物の剛体として置く。
 */
export interface SnakeTerrainBox {
  cx: number;
  cz: number;
  cy: number;
  halfX: number;
  halfY: number;
  halfZ: number;
}

/** (x,y) を覆う地形箱の最大上面 z（無ければ地面 0）。接地判定と前方プレビュー観測に使う。 */
export function snakeTerrainTopAt(boxes: SnakeTerrainBox[], x: number, y: number): number {
  let top = 0;
  for (const b of boxes) {
    if (Math.abs(x - b.cx) <= b.halfX && Math.abs(y - b.cy) <= b.halfY) {
      top = Math.max(top, b.cz + b.halfZ);
    }
  }
  return top;
}

/**
 * だんだん難しくなる進行性コース（小障害物 → 階段 → 低いテーブル）を +x 方向に並べる。
 * 蛇は x∈[0, n·segLen] に横たわって生成され、+x へ進む（waveSign=-1）。各要素は横幅 y を広く取り
 * （回り込み防止）、高さは控えめ（蛇の段差越えは四足より難しい）。テーブルは到達するが突破は必須にしない。
 */
export function makeProgressionTerrain(): SnakeTerrainBox[] {
  const halfY = 3.0; // 横（y）に広く張って蛇が回り込めないように（蛇は y へ大きくドリフトしうる）
  const boxes: SnakeTerrainBox[] = [];
  const box = (cx: number, cz: number, halfX: number, halfZ: number): void => {
    boxes.push({ cx, cz, cy: 0, halfX, halfY, halfZ });
  };
  // 1) 小障害物 ×3（高さ 2cm）。頭の初期位置(≈+1.12)の少し先から。
  for (let i = 0; i < 3; i++) box(1.7 + i * 0.26, 0.01, 0.05, 0.01);
  // 2) 階段 ×3（rise 2cm・踏面 0.18m、上面 2/4/6cm）。各段は床から立つ実体ブロック。
  const stairStartX = 2.7;
  const tread = 0.18;
  for (let i = 0; i < 3; i++) {
    const h = (i + 1) * 0.02;
    box(stairStartX + i * tread + tread / 2, h / 2, tread / 2, h / 2);
  }
  const topZ = 0.06; // 階段上面
  // 階段上〜テーブル下までを覆う踊り場（蛇は z=topZ の床を進み、テーブルの脚はこの上に立つ）。
  const stairTopX = stairStartX + 3 * tread; // 3.24
  const plateauEndX = stairTopX + 1.4;
  box((stairTopX + plateauEndX) / 2, topZ / 2, (plateauEndX - stairTopX) / 2, topZ / 2);
  // 3) 低いテーブル（脚2本＋天板）。脚は踊り場(z=topZ)から立つ実体の壁＝今は越えられない最終要素。
  // 将来「脚をつたって天板へ登る」モデルの土台。
  const tableX = stairTopX + 0.65;
  const legTop = topZ + 0.16; // 天板下端＝踊り場から 16cm（体径 3cm の ~5倍。今は越えられない最終壁）
  const legHalf = 0.02;
  box(tableX - 0.18, (topZ + legTop) / 2, legHalf, (legTop - topZ) / 2); // 手前脚（踊り場→天板下）
  box(tableX + 0.18, (topZ + legTop) / 2, legHalf, (legTop - topZ) / 2); // 奥脚
  box(tableX, legTop + 0.015, 0.24, 0.015); // 天板（厚 3cm）
  return boxes;
}

/**
 * 「直進チャレンジ」コース（小障害物 → 2cm 階段 → 長い踊り場・壁なし）を +x 方向に並べる。
 *
 * makeProgressionTerrain との違い: テーブル壁を置かず踊り場を遠くまで延ばす。狙いは前進(x)に上限を
 * 作らないこと。基盤歩容は障害物＋階段で進行方向を蹴られ、その後も開ループのまま斜行を続ける
 * （平地では真っ直ぐ進めるのに、コースでは ~-40° へ veer して前進を浪費する）。前進 x が壁でキャップ
 * されないので、「進行方向を検知して +x へ操舵し直す」閉ループ方策＝RL が、浪費される横移動を前進へ
 * 変換して基盤を明確に上回れる。`plateauEndX` まで踊り場（z=topZ）が続く。
 */
export function makeStraightChallengeTerrain(plateauEndX = 12.0): SnakeTerrainBox[] {
  const halfY = 3.0; // 横（y）に広く張る（接触で回り込めないように＝壁ではなく床として）
  const boxes: SnakeTerrainBox[] = [];
  const box = (cx: number, cz: number, halfX: number, halfZ: number): void => {
    boxes.push({ cx, cz, cy: 0, halfX, halfY, halfZ });
  };
  // 1) 小障害物 ×3（高さ 2cm）。
  for (let i = 0; i < 3; i++) box(1.7 + i * 0.26, 0.01, 0.05, 0.01);
  // 2) 階段 ×3（rise 2cm・踏面 0.18m、上面 2/4/6cm）＝基盤が登れる高さ。
  const stairStartX = 2.7;
  const tread = 0.18;
  for (let i = 0; i < 3; i++) {
    const h = (i + 1) * 0.02;
    box(stairStartX + i * tread + tread / 2, h / 2, tread / 2, h / 2);
  }
  // 3) 階段上面（6cm）から遠くまで続く踊り場。壁なし＝前進 x に上限を作らない。
  const topZ = 0.06;
  const stairTopX = stairStartX + 3 * tread; // 3.24
  box((stairTopX + plateauEndX) / 2, topZ / 2, (plateauEndX - stairTopX) / 2, topZ / 2);
  return boxes;
}

/**
 * ランダムなコースを1つ生成する（ドメインランダム化の素）。障害物数・段高(≤2cm=基盤が登れる)・段数・
 * 踊り場長・終端壁の有無をランダムに振る。蛇は常に x∈[0,1.05] の平地に生成され +x へ進むので、
 * 地形は x≥1.5 から置く。`rng` は [0,1) を返す乱数（既定 Math.random）。
 */
export function makeRandomCourse(rng: () => number = Math.random): SnakeTerrainBox[] {
  const halfY = 3.0;
  const boxes: SnakeTerrainBox[] = [];
  const box = (cx: number, cz: number, halfX: number, halfZ: number): void => {
    boxes.push({ cx, cz, cy: 0, halfX, halfY, halfZ });
  };
  // 小障害物 0〜3 個（高さ 2cm・位置を少し揺らす）。
  const nBumps = Math.floor(rng() * 4);
  for (let i = 0; i < nBumps; i++) box(1.5 + i * 0.26 + rng() * 0.12, 0.01, 0.05, 0.01);
  // 階段 0〜3 段（rise 1.5〜2cm＝基盤が登れる高さ）。
  const nStairs = Math.floor(rng() * 4);
  const rise = 0.015 + rng() * 0.005;
  const stairStartX = 2.5 + rng() * 0.4;
  const tread = 0.18;
  let topZ = 0;
  for (let i = 0; i < nStairs; i++) {
    const h = (i + 1) * rise;
    box(stairStartX + i * tread + tread / 2, h / 2, tread / 2, h / 2);
    topZ = h;
  }
  // 踊り場（階段がある時のみ・上面 topZ）を遠くまで（長さ 3〜10m）伸ばす。
  const stairTopX = stairStartX + nStairs * tread;
  const plateauEndX = stairTopX + 3 + rng() * 7;
  if (topZ > 0)
    box((stairTopX + plateauEndX) / 2, topZ / 2, (plateauEndX - stairTopX) / 2, topZ / 2);
  // 30% で終端に高い全幅壁（蛇は越えられず手前で止まる＝壁に直進して止まる挙動を学ぶ）。
  if (rng() < 0.3) {
    const wallX = stairTopX + 0.5 + rng() * 1.0;
    const wallH = 0.12 + rng() * 0.06;
    box(wallX, topZ + wallH / 2, 0.03, wallH / 2);
  }
  return boxes;
}

/**
 * ドメインランダム化用のコースバンク（コース汎用な単一方策を学習するための地形集合）。
 * 平地・進行性・直進チャレンジの3つの名前付きコース（汎用性の評価対象）に加えてランダム変種を混ぜる。
 * 平地を必ず含むのが肝: 「ドリフトしていない時は操舵しない」を学ばせ、特定コースへの過学習（常時操舵バイアス）を防ぐ。
 */
export function makeCourseBank(nRandom = 9, rng: () => number = Math.random): SnakeTerrainBox[][] {
  const bank: SnakeTerrainBox[][] = [
    [], // 平地
    makeProgressionTerrain(),
    makeStraightChallengeTerrain(),
  ];
  for (let i = 0; i < nRandom; i++) bank.push(makeRandomCourse(rng));
  return bank;
}

export interface Snake3DConfig {
  // ---- 関節構成（JointSpec） ----
  pattern: AxisPattern;
  n: number; // リンク数
  segLen: number; // 1リンク長 [m]
  bodyWidth: number; // カプセル直径 [m]
  // ---- 機体 ----
  totalMass: number;
  groundFriction: number;
  // ---- 地形（空なら平地） ----
  terrain: SnakeTerrainBox[];
  // 進行方向: 波の空間位相の符号。+1=現状(−x へ進む)、−1=+x へ進む（地形コースは +x に並ぶので −1）。
  waveSign: number;
  // ---- 歩容（パラメタ化セルペノイド波） ----
  yawAmp: number; // 水平関節の振幅 [rad]
  pitchAmp: number; // 垂直関節の振幅 [rad]
  period: number; // 時間周期 [s]
  waveLength: number; // 空間波長（リンク数）
  yawPitchPhase: number; // yaw に対する pitch の位相差 [rad]（サイドワインドは π/2）
  // ---- 車輪相当の異方抵抗 ----
  kLat: number; // 横方向抵抗 [N/(m/s)]（高い＝横滑りしない＝前進へ変換）
  kFwd: number; // 前方抵抗 [N/(m/s)]（低い＝前へ滑る）
  motor: { stiffness: number; damping: number; maxTorqueNm: number };
  dt: number;
  substeps: number;
  duration: number;
}

export const DEFAULT_SNAKE3D_CONFIG: Snake3DConfig = {
  pattern: 'all-yaw',
  n: 16,
  segLen: 0.07,
  bodyWidth: 0.03,
  totalMass: 0.6,
  groundFriction: 0.1,
  terrain: [],
  waveSign: 1,
  yawAmp: 0.5,
  pitchAmp: 0.35, // 既定で 3D（サイドワインド）が出る垂直振幅。pattern=all-yaw では未使用。
  period: 1.4,
  waveLength: 10, // 長めの波長＋pitch=0.35＋位相 90° で alt-yaw-pitch がクリーンにサイドワインド（~98cm）。
  yawPitchPhase: Math.PI / 2,
  kLat: 12,
  kFwd: 0.6,
  motor: { stiffness: 3, damping: 0.15, maxTorqueNm: 1.0 },
  dt: 1 / 240,
  substeps: 4,
  duration: 8,
};

/** 軸パターンを関節ごとの軸列（長さ n-1）へ展開する。 */
export function expandAxes(pattern: AxisPattern, nJoints: number): JointAxis[] {
  const axes: JointAxis[] = [];
  for (let j = 0; j < nJoints; j++) {
    if (pattern === 'all-yaw') axes.push('yaw');
    else if (pattern === 'all-pitch') axes.push('pitch');
    else axes.push(j % 2 === 0 ? 'yaw' : 'pitch');
  }
  return axes;
}

export interface SnakeBodyLayout {
  half: [number, number, number]; // カプセル相当の半寸法（local x=長手, y/z=半径）
}

export interface SnakeFrameDiag {
  travelM: number; // 開始からの水平移動距離 [m]（向きに依らない総移動）
  demandNm: number;
  appliedNm: number;
  saturated: boolean;
}

export interface SnakeFrame {
  t: number;
  bodies: Array<{ p: [number, number, number]; q: [number, number, number, number] }>;
  diag: SnakeFrameDiag;
}

export interface Snake3DSummary {
  config: Snake3DConfig;
  travelM: number; // 正味変位の大きさ |Δ(COM)|（向きに依らない総移動）
  netDispM: [number, number]; // 正味変位ベクトル [Δx, Δy]（サイドワインドの斜め移動を表す）
  headingDeg: number; // 進行方向（+x を 0°、+y を 90° とする方位）
  maxDemandNm: number;
  maxAppliedNm: number;
  saturatedSteps: number;
  success: boolean;
}

export interface Snake3DReplay {
  layout: SnakeBodyLayout[];
  frames: SnakeFrame[];
  summary: Snake3DSummary;
}

type V3 = [number, number, number];
type Quat = [number, number, number, number]; // x,y,z,w

export function qRotate(q: Quat, v: V3): V3 {
  const [x, y, z, w] = q;
  const ix = w * v[0] + y * v[2] - z * v[1];
  const iy = w * v[1] + z * v[0] - x * v[2];
  const iz = w * v[2] + x * v[1] - y * v[0];
  const iw = -x * v[0] - y * v[1] - z * v[2];
  return [
    ix * w + iw * -x + iy * -z - iz * -y,
    iy * w + iw * -y + iz * -x - ix * -z,
    iz * w + iw * -z + ix * -y - iy * -x,
  ];
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function f(x: number): string {
  return x.toFixed(6);
}

export function buildMjcf(cfg: Snake3DConfig, axes: JointAxis[], physicsDt: number): string {
  const halfW = cfg.bodyWidth / 2;
  const capHalf = Math.max(0.001, cfg.segLen / 2 - halfW);
  const linkMass = cfg.totalMass / cfg.n;
  const z0 = halfW + 0.002;
  const mu = cfg.groundFriction;

  const lines: string[] = [];
  for (let i = 0; i < cfg.n; i++) {
    const indent = '      ' + '  '.repeat(i);
    const pos = i === 0 ? `0 0 ${f(z0)}` : `${f(cfg.segLen)} 0 0`;
    lines.push(`${indent}<body name="L${i}" pos="${pos}">`);
    if (i === 0) {
      lines.push(`${indent}  <freejoint/>`);
    } else {
      const axis = axes[i - 1] === 'yaw' ? '0 0 1' : '0 1 0';
      lines.push(
        `${indent}  <joint name="J${i - 1}" type="hinge" axis="${axis}" pos="${f(-cfg.segLen / 2)} 0 0" damping="0.004" armature="0.001"/>`,
      );
    }
    lines.push(
      `${indent}  <geom name="g${i}" type="capsule" fromto="${f(-capHalf)} 0 0 ${f(capHalf)} 0 0" size="${f(halfW)}" mass="${f(linkMass)}" friction="${mu} 0.005 0.0001"/>`,
    );
  }
  for (let i = cfg.n - 1; i >= 0; i--) lines.push('      ' + '  '.repeat(i) + '</body>');

  // 地形箱（実接触の剛体）。蛇はこれに体を押し付けて段を登る。摩擦は床と同じ μ。
  const terrainLines = cfg.terrain.map(
    (b, i) =>
      `    <geom name="terrain${i}" type="box" pos="${f(b.cx)} ${f(b.cy)} ${f(b.cz)}" size="${f(b.halfX)} ${f(b.halfY)} ${f(b.halfZ)}" friction="${mu} 0.01 0.001" rgba="0.36 0.42 0.5 1"/>`,
  );

  return `<mujoco model="snake3d">
  <option gravity="0 0 -9.81" timestep="${f(physicsDt)}" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="8 8 0.1" pos="0 0 0" friction="${mu} 0.005 0.0001"/>
${terrainLines.join('\n')}
${lines.join('\n')}
  </worldbody>
</mujoco>`;
}

export async function runSnake3D(
  overrides: Partial<Snake3DConfig> = {},
  recordFps = 60,
): Promise<Snake3DReplay> {
  const mj: MujocoModule = await getMujoco();
  const cfg: Snake3DConfig = { ...DEFAULT_SNAKE3D_CONFIG, ...overrides };
  const { n } = cfg;
  const halfW = cfg.bodyWidth / 2;
  const substeps = Math.max(1, Math.round(cfg.substeps));
  const physicsDt = cfg.dt / substeps;
  const axes = expandAxes(cfg.pattern, n - 1);

  const model = mj.MjModel.from_xml_string(buildMjcf(cfg, axes, physicsDt));
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);

  const layout: SnakeBodyLayout[] = Array.from({ length: n }, () => ({
    half: [cfg.segLen / 2, halfW, halfW],
  }));

  const bodyId = (link: number): number => link + 1; // world=0, L0=1, ...
  const spatialPhase = (2 * Math.PI) / cfg.waveLength;
  const groundGate = halfW * 1.6; // この高さ未満のリンクのみ接地とみなし異方抵抗を掛ける

  const com = (): [number, number] => {
    const xpos = data.xpos;
    let x = 0;
    let y = 0;
    for (let i = 0; i < n; i++) {
      x += xpos[bodyId(i) * 3];
      y += xpos[bodyId(i) * 3 + 1];
    }
    return [x / n, y / n];
  };
  const [startX, startY] = com();

  const prevPos: V3[] = [];
  {
    const xpos = data.xpos;
    for (let i = 0; i < n; i++) {
      const bid = bodyId(i);
      prevPos.push([xpos[bid * 3], xpos[bid * 3 + 1], xpos[bid * 3 + 2]]);
    }
  }

  const frames: SnakeFrame[] = [];
  const steps = Math.ceil(cfg.duration / cfg.dt);
  const recordEvery = recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * cfg.dt))) : Infinity;

  let maxDemandNm = 0;
  let maxAppliedNm = 0;
  let saturatedSteps = 0;

  for (let step = 0; step < steps; step++) {
    const t = step * cfg.dt;

    // 車輪相当の異方抵抗（制御刻みで更新・接地リンクのみ）: 横成分を強く・前方を弱く抑える外力。
    const driveForce: V3[] = [];
    {
      const xpos = data.xpos;
      const xquat = data.xquat;
      for (let i = 0; i < n; i++) {
        const bid = bodyId(i);
        const px = xpos[bid * 3];
        const py = xpos[bid * 3 + 1];
        const pz = xpos[bid * 3 + 2];
        const vx = (px - prevPos[i][0]) / cfg.dt;
        const vy = (py - prevPos[i][1]) / cfg.dt;
        prevPos[i][0] = px;
        prevPos[i][1] = py;
        prevPos[i][2] = pz;
        // 接地判定は「局所地形上面からの高さ」で行う（段の上・縁でも牽引が効くように）。
        // 段の手前では top=0（地面）なので、リンクは接地扱い→水平牽引が段の縦面へ押し付け、
        // pitch の持ち上げと相まって体が段へ乗り上がる（実接触は MuJoCo が解く）。
        const top = cfg.terrain.length > 0 ? snakeTerrainTopAt(cfg.terrain, px, py) : 0;
        if (pz - top > groundGate) {
          driveForce.push([0, 0, 0]); // 局所地形から持ち上がっているリンクには掛けない
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
        const ff = -cfg.kFwd * vFwd;
        const fl = -cfg.kLat * vLat;
        driveForce.push([fwd[0] * ff + lat[0] * fl, fwd[1] * ff + lat[1] * fl, 0]);
      }
    }

    let demand = 0;
    let applied = 0;
    let saturated = false;

    for (let sub = 0; sub < substeps; sub++) {
      const qpos = data.qpos;
      const qvel = data.qvel;
      const qfrc = data.qfrc_applied;
      const xfrc = data.xfrc_applied;
      const ts = t + sub * physicsDt;
      // パラメタ化セルペノイド波の関節目標 → PD（±cap）。軸ごとに振幅・位相を変える。
      for (let j = 0; j < n - 1; j++) {
        const isYaw = axes[j] === 'yaw';
        const amp = isYaw ? cfg.yawAmp : cfg.pitchAmp;
        const phase = isYaw ? 0 : cfg.yawPitchPhase;
        const target =
          amp * Math.sin((2 * Math.PI * ts) / cfg.period - cfg.waveSign * j * spatialPhase + phase);
        const angle = qpos[7 + j];
        const vel = qvel[6 + j];
        const raw = cfg.motor.stiffness * (target - angle) - cfg.motor.damping * vel;
        const tau = clamp(raw, -cfg.motor.maxTorqueNm, cfg.motor.maxTorqueNm);
        qfrc[6 + j] = tau;
        demand = Math.max(demand, Math.abs(raw));
        applied = Math.max(applied, Math.abs(tau));
        saturated ||= Math.abs(raw) - Math.abs(tau) > 1e-9;
      }
      for (let i = 0; i < n; i++) {
        const bid = bodyId(i);
        xfrc[bid * 6 + 0] = driveForce[i][0];
        xfrc[bid * 6 + 1] = driveForce[i][1];
        xfrc[bid * 6 + 2] = 0;
      }
      mj.mj_step(model, data);
    }

    maxDemandNm = Math.max(maxDemandNm, demand);
    maxAppliedNm = Math.max(maxAppliedNm, applied);
    if (saturated) saturatedSteps++;

    if (step % recordEvery === 0) {
      const xpos = data.xpos;
      const xquat = data.xquat;
      const [cx, cy] = com();
      const travelM = Math.hypot(cx - startX, cy - startY);
      const bodies: SnakeFrame['bodies'] = [];
      for (let i = 0; i < n; i++) {
        const bid = bodyId(i);
        bodies.push({
          p: [xpos[bid * 3], xpos[bid * 3 + 1], xpos[bid * 3 + 2]],
          q: [xquat[bid * 4 + 1], xquat[bid * 4 + 2], xquat[bid * 4 + 3], xquat[bid * 4]],
        });
      }
      frames.push({
        t,
        bodies,
        diag: { travelM, demandNm: demand, appliedNm: applied, saturated },
      });
    }
  }

  const [cx, cy] = com();
  const dx = cx - startX;
  const dy = cy - startY;
  const travelM = Math.hypot(dx, dy);
  const summary: Snake3DSummary = {
    config: cfg,
    travelM,
    netDispM: [dx, dy],
    headingDeg: (Math.atan2(dy, dx) * 180) / Math.PI,
    maxDemandNm,
    maxAppliedNm,
    saturatedSteps,
    success: travelM > cfg.segLen * cfg.n * 0.3,
  };

  data.delete();
  model.delete();
  return { layout, frames, summary };
}
