/**
 * 「トゲ車輪ウニ」= 放射状の長いスポーク（棘）が回転して進む リムレスホイール／whegs 型の 3D 動的シム（Rapier）。
 *
 * 脚で真下に立って歩く形（urchin-dynamics）は段差≈脚を持ち上げる高さが限界（合成コースで 3cm 級）だが、
 * このトゲ車輪は **スポークの先端で「転がり歩き」し、段差の角にスポークを引っかけて梃子で登る**ため、
 * **スポーク長に近い高さ（脚で立つ限界の数倍）の段差を越えられる**。接地したスポークが脚になるので姿勢に依らない。
 *
 * モデル: ハブ＋N本のスポークを **1つの剛体** にする（スポークは剛＝リムレスホイール）。横転を防ぐため
 * 同軸に2列(y=±width)。軸(y)まわりに **トルク上限付きの速度制御モーター** で回す（cap=サーボ τ上限・直結）。
 * 横方向は「軸を world y に保つ」復元トルクのみ与え（スピンは自由）、矢状面(x-z)の登坂・トルク充足を切り出す。
 * 接触・摩擦・登坂は Rapier が解く。レンダラ/リプレイ互換のため戻り値は四足と同じ {@link QuadDynReplay}。
 */
import RAPIER, { type RigidBody, type World } from '@dimforge/rapier3d-compat';
import { G } from './chain.ts';
import { COURSES, buildCourseColliders, type CourseSpec } from './course.ts';
import {
  DEG,
  type QuadDynReplay,
  type QuadFrame,
  type QuadFrameDiag,
  type QuadBodyLayout,
} from './quadruped-dynamics.ts';

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

type Quat = { x: number; y: number; z: number; w: number };
type Vec3 = [number, number, number];

/** クォータニオン積 a∘b。 */
function qMul(a: Quat, b: Quat): Quat {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

/** ベクトル v を q で回す。 */
function qRot(q: Quat, v: Vec3): Vec3 {
  const { x, y, z, w } = q;
  // t = 2 q.xyz × v ; v' = v + w t + q.xyz × t
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

/** y 軸まわり angle [rad] の回転クォータニオン。 */
function qAxisY(angle: number): Quat {
  return { x: 0, y: Math.sin(angle / 2), z: 0, w: Math.cos(angle / 2) };
}

export interface SpokeUrchinOverrides {
  spokeCount?: number; // スポーク本数（放射）
  spokeLen?: number; // スポーク長（=実効転がり半径）[m]
  width?: number; // 同軸2列の片側オフセット [m]（横安定）
  mass?: number; // 総質量 [kg]
  torqueCapNm?: number; // 軸トルク上限 [N·m]（サーボ直結）
  targetOmega?: number; // 目標角速度 [rad/s]（前進=+y まわり正）
  friction?: number;
  course?: CourseSpec;
  duration?: number;
  dt?: number;
  substeps?: number;
  lateralStabK?: number;
  lateralStabD?: number;
}

const DEFAULTS = {
  spokeCount: 8,
  spokeLen: 0.15,
  width: 0.05,
  mass: 0.25,
  torqueCapNm: 0.226,
  targetOmega: 5,
  friction: 0.95,
  duration: 6,
  dt: 1 / 240,
  substeps: 8,
  lateralStabK: 6,
  lateralStabD: 0.4,
};

interface SpokeLocal {
  pos: Vec3;
  quat: Quat;
  half: Vec3;
}

interface SpokeAssembly {
  body: RigidBody;
  spokeLocals: SpokeLocal[]; // 描画用: 各スポークのボディ局所変換
  hubHalf: Vec3;
}

/** トゲ車輪（ハブ＋N×2スポーク）を1剛体で組む。 */
function buildSpokeWheel(
  world: World,
  o: Required<
    Pick<SpokeUrchinOverrides, 'spokeCount' | 'spokeLen' | 'width' | 'mass' | 'friction'>
  >,
  startX: number,
  standZ: number,
): SpokeAssembly {
  const { spokeCount: n, spokeLen: L, width, mass, friction } = o;
  const r = Math.max(0.006, L * 0.05); // スポークの太さ（細い棒）
  const hubR = Math.max(0.02, L * 0.18);
  const hubHalf: Vec3 = [hubR, width, hubR];
  const spokeMassTotal = mass * 0.5;
  const hubMass = mass - spokeMassTotal;
  const spokeMass = spokeMassTotal / (2 * n);

  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(startX, 0, standZ)
      // 初期に1本のスポークが真下(-z)を向くよう全体を回す（θ=0 が +x。-z は θ=-90°→ -π/2 だけ回す）。
      .setRotation(qAxisY(-Math.PI / 2 + Math.PI / n))
      .setLinearDamping(0.05)
      .setAngularDamping(0.06)
      .setCanSleep(false),
  );

  // ハブ
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(hubHalf[0], hubHalf[1], hubHalf[2])
      .setMass(hubMass)
      .setFriction(friction),
    body,
  );

  const spokeLocals: SpokeLocal[] = [];
  const spokeHalf: Vec3 = [r, r, L / 2];
  for (let i = 0; i < n; i++) {
    const theta = (2 * Math.PI * i) / n;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    // 局所 z（棒の長軸）を放射方向 (cosθ,0,sinθ) へ向ける: y まわり α=π/2−θ。
    const quat = qAxisY(Math.PI / 2 - theta);
    for (const side of [-1, +1] as const) {
      const pos: Vec3 = [(L / 2) * cos, side * width, (L / 2) * sin];
      world.createCollider(
        RAPIER.ColliderDesc.cuboid(spokeHalf[0], spokeHalf[1], spokeHalf[2])
          .setTranslation(pos[0], pos[1], pos[2])
          .setRotation(quat)
          .setMass(spokeMass)
          .setFriction(friction)
          .setRestitution(0),
        body,
      );
      spokeLocals.push({ pos, quat, half: spokeHalf });
    }
  }
  body.setAdditionalSolverIterations(4);
  return { body, spokeLocals, hubHalf };
}

function spokeLayout(asm: SpokeAssembly): QuadBodyLayout[] {
  const layout: QuadBodyLayout[] = [{ kind: 'trunk', half: asm.hubHalf }];
  for (const s of asm.spokeLocals) layout.push({ kind: 'shin', half: s.half });
  return layout;
}

/** ハブ＋各スポークのワールド変換を記録（剛体1つの姿勢から局所変換を合成）。 */
function captureSpokeFrame(asm: SpokeAssembly, t: number, diag: QuadFrameDiag): QuadFrame {
  const p = asm.body.translation();
  const q = asm.body.rotation();
  const Q: Quat = { x: q.x, y: q.y, z: q.z, w: q.w };
  const bodies: QuadFrame['bodies'] = [{ p: [p.x, p.y, p.z], q: [q.x, q.y, q.z, q.w] }];
  for (const s of asm.spokeLocals) {
    const wp = qRot(Q, s.pos);
    const wq = qMul(Q, s.quat);
    bodies.push({ p: [p.x + wp[0], p.y + wp[1], p.z + wp[2]], q: [wq.x, wq.y, wq.z, wq.w] });
  }
  return { t, bodies, diag };
}

export async function runSpokeUrchin(
  overrides: SpokeUrchinOverrides = {},
  recordFps = 60,
): Promise<QuadDynReplay> {
  await RAPIER.init();
  const o = { ...DEFAULTS, ...overrides };
  const substeps = Math.max(1, Math.round(o.substeps));
  const physicsDt = o.dt / substeps;

  const world = new RAPIER.World({ x: 0, y: 0, z: -G });
  world.timestep = physicsDt;
  world.numSolverIterations = 8;

  const course = overrides.course ?? COURSES.flat();
  buildCourseColliders(world, course, o.friction, 3);

  const startX = -0.3;
  const standZ = o.spokeLen + 0.01; // スポーク先端がほぼ z=0
  const asm = buildSpokeWheel(world, o, startX, standZ);

  // 速度制御の比例ゲイン: 静止時に full cap を出す（cap/targetOmega の数倍）。
  const kp = o.targetOmega > 1e-6 ? (3 * o.torqueCapNm) / o.targetOmega : 0;

  const layout = spokeLayout(asm);
  const frames: QuadFrame[] = [];
  const steps = Math.ceil(o.duration / o.dt);
  const recordEvery = recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * o.dt))) : Infinity;

  let maxDemandNm = 0;
  let maxAppliedNm = 0;
  let saturatedSteps = 0;
  let minHubZ = standZ;
  let maxTiltDeg = 0;
  let fellTime: number | null = null;

  for (let step = 0; step < steps; step++) {
    const t = step * o.dt;
    let demand = 0;
    let applied = 0;
    let saturated = false;

    for (let sub = 0; sub < substeps; sub++) {
      const q = asm.body.rotation();
      const Q: Quat = { x: q.x, y: q.y, z: q.z, w: q.w };
      const axle = qRot(Q, [0, 1, 0]); // ワールドでの軸方向
      const w = asm.body.angvel();
      const spin = w.x * axle[0] + w.y * axle[1] + w.z * axle[2];
      const rawTau = kp * (o.targetOmega - spin);
      const tau = clamp(rawTau, -o.torqueCapNm, o.torqueCapNm);
      const imp = tau * physicsDt;
      asm.body.applyTorqueImpulse({ x: axle[0] * imp, y: axle[1] * imp, z: axle[2] * imp }, true);
      // 横安定化: 軸を world y へ戻す復元（axle×ŷ=(-az,0,ax)・y成分0＝スピンに触れない）＋ロール/ヨー減衰。
      const sTauX = o.lateralStabK * -axle[2] - o.lateralStabD * w.x;
      const sTauZ = o.lateralStabK * axle[0] - o.lateralStabD * w.z;
      asm.body.applyTorqueImpulse({ x: sTauX * physicsDt, y: 0, z: sTauZ * physicsDt }, true);

      demand = Math.max(demand, Math.abs(rawTau));
      applied = Math.max(applied, Math.abs(tau));
      saturated ||= Math.abs(rawTau) - Math.abs(tau) > 1e-9;
      world.step();
    }

    const p = asm.body.translation();
    const qn = asm.body.rotation();
    const axle = qRot({ x: qn.x, y: qn.y, z: qn.z, w: qn.w }, [0, 1, 0]);
    const tilt = Math.acos(clamp(axle[1], -1, 1)); // 軸が y からどれだけ傾いたか
    const forwardX = p.x - startX;
    // 横転（軸が倒れた）を fell とする。
    const fallen = tilt > 50 * DEG;
    if (fallen && fellTime === null) fellTime = t;

    maxDemandNm = Math.max(maxDemandNm, demand);
    maxAppliedNm = Math.max(maxAppliedNm, applied);
    if (saturated) saturatedSteps++;
    minHubZ = Math.min(minHubZ, p.z);
    maxTiltDeg = Math.max(maxTiltDeg, tilt / DEG);

    if (step % recordEvery === 0) {
      frames.push(
        captureSpokeFrame(asm, t, {
          demandNm: demand,
          appliedNm: applied,
          saturated,
          trunkZ: p.z,
          tiltDeg: tilt / DEG,
          forwardX,
          fallen,
        }),
      );
    }
  }

  const pEnd = asm.body.translation();
  const forwardDistanceM = pEnd.x - startX;
  const fell = fellTime !== null;
  const success = !fell && forwardDistanceM > o.spokeLen;

  const summary: QuadDynReplay['summary'] = {
    // config は QuadDynConfig 完全形を要求するが、再生は duration しか見ないので最小限を満たす形で詰める。
    config: {
      trunk: { length: 2 * o.spokeLen, width: 2 * o.width, height: 2 * o.spokeLen, mass: o.mass },
      leg: { thigh: o.spokeLen / 2, shin: o.spokeLen / 2, segMass: 0, radius: 0.01 },
      hipInset: 0,
      motor: {
        stiffness: 0,
        damping: 0,
        passiveDamping: 0,
        maxTorqueNm: o.torqueCapNm,
        mode: 'torque',
      },
      friction: o.friction,
      dt: o.dt,
      substeps: o.substeps,
      lateralStabK: o.lateralStabK,
      lateralStabD: o.lateralStabD,
      duration: o.duration,
      gait: { period: 1, strideM: 0, liftM: 0, standM: o.spokeLen, stanceDuty: 0.5 },
    },
    forwardDistanceM,
    maxDemandNm,
    maxAppliedNm,
    saturatedSteps,
    minTrunkZ: minHubZ,
    maxTiltDeg,
    fell,
    fellTime,
    success,
  };

  world.free();
  return { layout, frames, summary };
}
