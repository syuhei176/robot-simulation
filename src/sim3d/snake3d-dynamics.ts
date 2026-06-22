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

export interface Snake3DConfig {
  // ---- 関節構成（JointSpec） ----
  pattern: AxisPattern;
  n: number; // リンク数
  segLen: number; // 1リンク長 [m]
  bodyWidth: number; // カプセル直径 [m]
  // ---- 機体 ----
  totalMass: number;
  groundFriction: number;
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
  yawAmp: 0.5,
  pitchAmp: 0,
  period: 1.4,
  waveLength: 8,
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
  travelM: number;
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

function qRotate(q: Quat, v: V3): V3 {
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

function buildMjcf(cfg: Snake3DConfig, axes: JointAxis[], physicsDt: number): string {
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

  return `<mujoco model="snake3d">
  <option gravity="0 0 -9.81" timestep="${f(physicsDt)}" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" friction="${mu} 0.005 0.0001"/>
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
        if (pz > groundGate) {
          driveForce.push([0, 0, 0]); // 持ち上がっているリンクには掛けない
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
        const target = amp * Math.sin((2 * Math.PI * ts) / cfg.period - j * spatialPhase + phase);
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
  const travelM = Math.hypot(cx - startX, cy - startY);
  const summary: Snake3DSummary = {
    config: cfg,
    travelM,
    maxDemandNm,
    maxAppliedNm,
    saturatedSteps,
    success: travelM > cfg.segLen * cfg.n * 0.3,
  };

  data.delete();
  model.delete();
  return { layout, frames, summary };
}
