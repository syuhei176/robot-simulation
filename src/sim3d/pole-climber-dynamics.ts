/**
 * 3D ヘビ（pole-climber）の動的シミュレーション（**MuJoCo wasm**）。
 *
 * 設計（モジュラー型ヘビロボット由来）:
 *  - N リンクを **ローカル yaw / pitch を交互に持つ 1自由度 hinge 関節** で連結する（広瀬 ACM・CMU modular snake と同型）。
 *    交互直交関節は最小DOFで 3D 形状（ヘリックス）を作れる。2D ヘビ（全関節 y軸）の自然な 3D 拡張。
 *  - **解析的ヘリックス**（柱を z 軸とする一定半径・一定ピッチ）を MJCF の rest pose に焼き込む: 全 hinge=0 で
 *    螺旋になるよう各 body の親相対変換を与える。縦ピッチを直径以上に取り、contype/conaffinity でリンク同士の
 *    自己衝突を無効化（接触は柱・床のみ）＝コイルが自己交差で爆発しない。
 *  - 関節は螺旋を **保持するだけ**（hinge を 0 付近へ PD・ほぼ静的トルク＝安サーボ向き）。yaw を gripTighten だけ
 *    締めて柱を握り、接触法線力 N を生む。capstan 効果で 1.5 巻でも N は十分。保持トルクは `qfrc_applied`。
 *  - 推進は **車輪**（v1 は抽象化）: 各リンク前方へ μ·N で頭打ちした牽引 `xfrc_applied` を与え「ねじ登る」。
 *    見た目の車輪は牽引量に応じて回す（spin 角を記録）。
 *
 * feasibility が数値で出る: 保持トルク → 締付け N → 最大牽引 μ·N が自重を支えれば登れる。
 * 物理は z-up。レンダラは quad と同じ z-up→y-up 変換で描く。MuJoCo の xquat は (w,x,y,z) 順。
 */
import { getMujoco, type MujocoModule } from './mujoco-engine.ts';

export interface PoleSpec {
  radius: number;
  height: number;
}

export interface PoleClimberConfig {
  n: number; // リンク数
  bodyWidth: number; // カプセル直径 [m]
  totalMass: number; // 総質量 [kg]
  pole: PoleSpec;
  friction: number; // 体↔柱の摩擦 μ
  wraps: number; // 巻き数（capstan グリップ）
  gripTighten: number; // 巻き関節(local y)の締付け [rad]（柱を握る）
  driveForceN: number; // 各リンク前方へかける車輪駆動力 [N]（摩擦が滑り/把持を自然に頭打ち）
  climbSpeed: number; // 車輪 spin 表示用の公称速度 [m/s]
  motor: { stiffness: number; damping: number; maxTorqueNm: number };
  dt: number; // 制御刻み [s]
  substeps: number; // 1制御刻みあたりの MuJoCo ステップ数
  duration: number;
}

export const DEFAULT_POLE_CLIMBER_CONFIG: PoleClimberConfig = {
  n: 26,
  bodyWidth: 0.022,
  totalMass: 0.5,
  pole: { radius: 0.05, height: 0.9 },
  friction: 0.35, // 車輪=体軸方向は低摩擦（転がり）。握りは normal 力＋締付けで作るので低くてよい。
  wraps: 1.7,
  gripTighten: 0.2,
  driveForceN: 4, // 1リンクあたりの最大車輪駆動力 [N]（速度制御の cap）
  climbSpeed: 0.03, // 目標 climb 速度 [m/s]（COM 上昇率の閉ループ目標）
  motor: { stiffness: 6, damping: 0.4, maxTorqueNm: 1.5 },
  dt: 1 / 240,
  substeps: 6,
  duration: 8,
};

const TURN_SEPARATION = 2.5; // 巻き間の縦ピッチ = bodyWidth × これ（自己衝突は無効だが上向き成分も確保）
const GAP = 0; // リンク内面を柱表面にちょうど接する半径に置く（初期めり込みなし）。握りは締付けで作る。
const DRIVE_KP = 250; // climb 速度の閉ループゲイン [N/(m/s)]（車輪駆動・重力滑落も能動的に支える）

// ---- 描画用レイアウト＆フレーム（renderer-free な生データ） ----
export interface ClimberBodyLayout {
  half: [number, number, number]; // カプセル相当の半寸法（local x=長手, y/z=半径）
  wheelRadius: number;
}

export interface ClimberFrameDiag {
  climbedM: number;
  demandNm: number;
  appliedNm: number;
  saturated: boolean;
  gripN: number;
  muDemand: number;
  fallen: boolean;
}

export interface ClimberFrame {
  t: number;
  bodies: Array<{ p: [number, number, number]; q: [number, number, number, number] }>;
  wheelSpin: number[];
  diag: ClimberFrameDiag;
}

export interface PoleClimberSummary {
  config: PoleClimberConfig;
  climbedM: number;
  helixRadiusM: number;
  bodyLengthM: number;
  maxDemandNm: number;
  maxAppliedNm: number;
  saturatedSteps: number;
  maxMuDemand: number;
  minGripN: number;
  fell: boolean;
  fellTime: number | null;
  success: boolean;
}

export interface PoleClimberReplay {
  layout: ClimberBodyLayout[];
  pole: PoleSpec;
  frames: ClimberFrame[];
  summary: PoleClimberSummary;
}

// ---- 最小限のクォータニオン／ベクトル演算 ----
type V3 = [number, number, number];
type Quat = [number, number, number, number]; // x,y,z,w

function qMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function qConj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

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

function qFromBasis(x: V3, y: V3, z: V3): Quat {
  const m00 = x[0];
  const m10 = x[1];
  const m20 = x[2];
  const m01 = y[0];
  const m11 = y[1];
  const m21 = y[2];
  const m02 = z[0];
  const m12 = z[1];
  const m22 = z[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s];
  }
  if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  }
  if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
  return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
}

function vDot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vCross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function vUnit(a: V3): V3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function jointAxisLocal(jointIdx: number): V3 {
  return jointIdx % 2 === 0 ? [0, 0, 1] : [0, 1, 0]; // 偶=yaw(local z), 奇=pitch(local y)
}

/** 数値を MJCF 用に整形（指数表記を避け固定小数で）。 */
function f(x: number): string {
  return x.toFixed(6);
}

interface HelixPlacement {
  placed: V3[];
  rots: Quat[];
  r: number;
  arc: number;
  linkLen: number;
}

/** 解析的ヘリックス上にリンク中心と姿勢を配置（柱は z 軸・中心 (0,0)）。 */
function placeHelix(cfg: PoleClimberConfig): HelixPlacement {
  const { n } = cfg;
  const halfW = cfg.bodyWidth / 2;
  const r = cfg.pole.radius + halfW + GAP;
  const pitchPerTurn = cfg.bodyWidth * TURN_SEPARATION;
  const c = pitchPerTurn / (2 * Math.PI);
  const totalPhi = cfg.wraps * 2 * Math.PI;
  const dPhi = totalPhi / n;
  const arc = totalPhi * Math.hypot(r, c);
  const linkLen = arc / n;
  const z0 = 0.04 + halfW;

  const placed: V3[] = [];
  const rots: Quat[] = [];
  for (let i = 0; i < n; i++) {
    const phi = (i + 0.5) * dPhi;
    const cphi = Math.cos(phi);
    const sphi = Math.sin(phi);
    placed.push([r * cphi, r * sphi, z0 + c * phi]);
    const tangent = vUnit([-r * sphi, r * cphi, c]); // local x
    const radialOut: V3 = [cphi, sphi, 0];
    const normal = vUnit([
      radialOut[0] - vDot(radialOut, tangent) * tangent[0],
      radialOut[1] - vDot(radialOut, tangent) * tangent[1],
      radialOut[2] - vDot(radialOut, tangent) * tangent[2],
    ]); // local z（外向き）
    const binormal = vUnit(vCross(normal, tangent)); // local y
    rots.push(qFromBasis(tangent, binormal, normal));
  }
  return { placed, rots, r, arc, linkLen };
}

/** ヘリックス配置を rest pose に焼き込んだ MJCF を生成（hinge=0 で螺旋）。 */
function buildMjcf(cfg: PoleClimberConfig, place: HelixPlacement, physicsDt: number): string {
  const { placed, rots, linkLen } = place;
  const { n } = cfg;
  const halfW = cfg.bodyWidth / 2;
  const capHalf = Math.max(0.001, linkLen / 2 - halfW);
  const linkMass = cfg.totalMass / n;
  const mu = cfg.friction;

  // body 開始タグ（親相対 pos/quat, hinge）。quat は MJCF 順 (w x y z)。
  const lines: string[] = [];
  for (let i = 0; i < n; i++) {
    let pos: V3;
    let quat: Quat;
    if (i === 0) {
      pos = placed[0];
      quat = rots[0];
    } else {
      const dp: V3 = [
        placed[i][0] - placed[i - 1][0],
        placed[i][1] - placed[i - 1][1],
        placed[i][2] - placed[i - 1][2],
      ];
      pos = qRotate(qConj(rots[i - 1]), dp);
      quat = qMul(qConj(rots[i - 1]), rots[i]);
    }
    const indent = '      ' + '  '.repeat(i);
    lines.push(
      `${indent}<body name="L${i}" pos="${f(pos[0])} ${f(pos[1])} ${f(pos[2])}" quat="${f(quat[3])} ${f(quat[0])} ${f(quat[1])} ${f(quat[2])}">`,
    );
    if (i === 0) {
      lines.push(`${indent}  <freejoint/>`);
    } else {
      const ax = jointAxisLocal(i - 1);
      lines.push(
        `${indent}  <joint name="J${i - 1}" type="hinge" axis="${ax[0]} ${ax[1]} ${ax[2]}" pos="${f(-linkLen / 2)} 0 0" damping="0.003" armature="0.0008"/>`,
      );
    }
    // リンク geom: 自己衝突しない（contype=1 conaffinity=0）＝柱/床のみと接触。
    lines.push(
      `${indent}  <geom name="g${i}" type="capsule" fromto="${f(-capHalf)} 0 0 ${f(capHalf)} 0 0" size="${f(halfW)}" mass="${f(linkMass)}" friction="${mu} 0.005 0.0001" contype="1" conaffinity="0"/>`,
    );
  }
  // すべての body を閉じる（深いネスト）。
  for (let i = n - 1; i >= 0; i--) lines.push('      ' + '  '.repeat(i) + '</body>');

  return `<mujoco model="pole-climber">
  <option gravity="0 0 -9.81" timestep="${f(physicsDt)}" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="1 1 0.1" pos="0 0 0" contype="2" conaffinity="0" friction="${mu} 0.005 0.0001"/>
    <geom name="pole" type="cylinder" fromto="0 0 0 0 0 ${f(cfg.pole.height)}" size="${f(cfg.pole.radius)}" contype="1" conaffinity="1" friction="${mu} 0.005 0.0001"/>
${lines.join('\n')}
  </worldbody>
</mujoco>`;
}

export async function runPoleClimber(
  overrides: Partial<PoleClimberConfig> = {},
  recordFps = 60,
): Promise<PoleClimberReplay> {
  const mj: MujocoModule = await getMujoco();
  const cfg: PoleClimberConfig = { ...DEFAULT_POLE_CLIMBER_CONFIG, ...overrides };
  const { n } = cfg;
  const halfW = cfg.bodyWidth / 2;
  const place = placeHelix(cfg);
  const substeps = Math.max(1, Math.round(cfg.substeps));
  const physicsDt = cfg.dt / substeps;

  const model = mj.MjModel.from_xml_string(buildMjcf(cfg, place, physicsDt));
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);

  // 保持目標: hinge=0（螺旋維持）＋ 巻き（半径）を司る odd 関節(local y)を gripTighten 締めて柱を握る。
  // even 関節(local z)は climb 角なので締付けは掛けない（締めると半径でなく登り角が変わる）。
  const tightenAmt: number[] = [];
  for (let j = 0; j < n - 1; j++) tightenAmt.push(j % 2 === 1 ? cfg.gripTighten : 0);

  const layout: ClimberBodyLayout[] = Array.from({ length: n }, () => ({
    half: [place.linkLen / 2, halfW, halfW],
    wheelRadius: halfW * 0.9,
  }));

  // 注意: data.qpos 等の typed-array ビューは wasm のメモリ確保（mj_contactForce 等）でヒープが伸びると
  // detach し得る。キャッシュせず必要時に data から都度取り直す（下のヘルパ経由）。
  const bodyId = (link: number): number => link + 1; // world=0, L0=1, ...

  const comZ = (): number => {
    const xpos = data.xpos;
    let z = 0;
    for (let i = 0; i < n; i++) z += xpos[bodyId(i) * 3 + 2];
    return z / n;
  };
  const startComZ = comZ();
  const wheelSpin = new Array<number>(n).fill(0);
  let prevClimbed = 0; // 前制御ステップの COM 上昇（climb 速度フィードバック用）

  const frames: ClimberFrame[] = [];
  const steps = Math.ceil(cfg.duration / cfg.dt);
  const recordEvery = recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * cfg.dt))) : Infinity;

  let maxDemandNm = 0;
  let maxAppliedNm = 0;
  let saturatedSteps = 0;
  const maxMuDemand = 0; // 接触力 N が当バインディングから取れないため μ 要求は未計測（接触点数で代用）
  let minGripN = Infinity;
  let fellTime: number | null = null;

  for (let step = 0; step < steps; step++) {
    const t = step * cfg.dt;
    const gripRamp = clamp(t / 0.4, 0, 1);
    const climbRamp = clamp((t - 0.9) / 0.6, 0, 1);

    let demand = 0;
    let applied = 0;
    let saturated = false;

    // 車輪駆動: COM 上昇率を目標 climb 速度へ閉ループ制御（重力での滑落も能動的に支える）。
    // 1リンクあたり ±driveForceN で頭打ち。低摩擦＝体軸方向は転がり、握りは normal 力で別途確保。
    const climbedNow = comZ() - startComZ;
    const vClimb = (climbedNow - prevClimbed) / cfg.dt;
    prevClimbed = climbedNow;
    const vClimbTarget = cfg.climbSpeed * climbRamp;
    const perLinkDrive = clamp(
      (DRIVE_KP * (vClimbTarget - vClimb)) / n,
      -cfg.driveForceN,
      cfg.driveForceN,
    );

    for (let sub = 0; sub < substeps; sub++) {
      const qpos = data.qpos; // [free(7), hinge(n-1)]（ビューは都度取得＝heap 成長で detach しないよう）
      const qvel = data.qvel; // [free(6), hinge(n-1)]
      const qfrc = data.qfrc_applied;
      const xfrc = data.xfrc_applied; // body ごと 6（world frame wrench）
      const xquat = data.xquat; // body ごと 4（w,x,y,z）
      // 1) 保持トルク（各 hinge を target へ PD・±cap）→ qfrc_applied。
      for (let j = 0; j < n - 1; j++) {
        const angle = qpos[7 + j];
        const vel = qvel[6 + j];
        const target = tightenAmt[j] * gripRamp;
        const raw = cfg.motor.stiffness * (target - angle) - cfg.motor.damping * vel;
        const tau = clamp(raw, -cfg.motor.maxTorqueNm, cfg.motor.maxTorqueNm);
        qfrc[6 + j] = tau;
        demand = Math.max(demand, Math.abs(raw));
        applied = Math.max(applied, Math.abs(tau));
        saturated ||= Math.abs(raw) - Math.abs(tau) > 1e-9;
      }

      // 2) 牽引（抽象車輪）: 全リンク前方へ driveForceN を印加。柱接触の摩擦コーンが MuJoCo 側で
      //    滑り/把持を自然に頭打ちするので、μ·N の手動 cap は不要（接触読みをホットループから排除）。
      for (let i = 0; i < n; i++) {
        const bid = bodyId(i);
        const q: Quat = [
          xquat[bid * 4 + 1],
          xquat[bid * 4 + 2],
          xquat[bid * 4 + 3],
          xquat[bid * 4],
        ];
        const fwd = qRotate(q, [1, 0, 0]);
        xfrc[bid * 6 + 0] = fwd[0] * perLinkDrive;
        xfrc[bid * 6 + 1] = fwd[1] * perLinkDrive;
        xfrc[bid * 6 + 2] = fwd[2] * perLinkDrive;
        // 車輪 spin（描画用）: 体軸前方速度 ≈ climb速度 / sin(lead) を概算。
        wheelSpin[i] += (vClimbTarget * 6 * physicsDt) / layout[i].wheelRadius;
      }

      mj.mj_step(model, data);
    }

    maxDemandNm = Math.max(maxDemandNm, demand);
    maxAppliedNm = Math.max(maxAppliedNm, applied);
    if (saturated) saturatedSteps++;

    const climbedM = comZ() - startComZ;
    const xpos = data.xpos;
    let maxR = 0;
    for (let i = 0; i < n; i++) {
      const bid = bodyId(i);
      maxR = Math.max(maxR, Math.hypot(xpos[bid * 3], xpos[bid * 3 + 1]));
    }
    const fallen = climbedM < -0.08 || maxR > place.r * 3;
    if (fallen && fellTime === null) fellTime = t;

    if (step % recordEvery === 0) {
      // グリップ指標は柱との接触点数（接触は柱のみ＝geom 判定不要・純数値で確実）。
      // mj_contactForce は wasm バッファ必須で当バインディングからは安全に呼べないため力 N は出さない。
      const gripN = data.ncon;
      const muDemand = 0;
      if (gripN > 0) minGripN = Math.min(minGripN, gripN);

      const xquat = data.xquat;
      const bodies: ClimberFrame['bodies'] = [];
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
        wheelSpin: [...wheelSpin],
        diag: {
          climbedM,
          demandNm: demand,
          appliedNm: applied,
          saturated,
          gripN,
          muDemand,
          fallen,
        },
      });
    }
  }

  const climbedM = comZ() - startComZ;
  const fell = fellTime !== null;
  const success = !fell && climbedM > cfg.pole.height * 0.3;
  const summary: PoleClimberSummary = {
    config: cfg,
    climbedM,
    helixRadiusM: place.r,
    bodyLengthM: place.arc,
    maxDemandNm,
    maxAppliedNm,
    saturatedSteps,
    maxMuDemand,
    minGripN: Number.isFinite(minGripN) ? minGripN : 0,
    fell,
    fellTime,
    success,
  };

  data.delete();
  model.delete();
  return { layout, pole: cfg.pole, frames, summary };
}
