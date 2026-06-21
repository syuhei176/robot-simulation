/**
 * 「本物のウニ形」放射対称 多脚機構の 3D 動的歩行（Rapier）。中心ハブの **円環上に N 脚を角度
 * θᵢ=2πi/N で放射配置**（前後軸が無い放射対称）。直列クモ（multiped: 行×左右で全脚が矢状面 x-z を漕ぐ）
 * と対照させ、放射形が直進コースを歩けるかを検証する。
 *
 * 脚の揺動面を legPlane で選ぶ:
 *  - 'radial'（本物の splay ウニ）: 各脚は **自分の放射鉛直面**（半径 dᵢ=(cosθ,sinθ) と鉛直 ẑ の面）で揺動。
 *    関節軸＝接線 aᵢ=(−sinθ,cosθ,0)。歩容は接地足を世界に固定する自然方式（後方へ fx 掃く）だが、
 *    **1自由度脚では足を世界 x に固定しつつ動かせるのは進行軸に揃った前後の脚(θ≈0,π)だけ**で、斜め・横の
 *    脚は自分の放射線上しか動けず横滑り(scrub)する。よって脚数を増やしても増えるのは擦る横脚ばかりで、
 *    前進はほぼ伸びず **這う**（平地でクモの ~1/8 速）。これが splay(開脚)の本質的代償。
 *  - 'heading'（操舵ウニ＝脚にヨー DOF を足した全方向型に相当）: 全脚の揺動面を進行方向 x へ向ける(軸=ŷ)。
 *    放射配置のまま全脚が足を世界に固定して漕げる → クモ並みに歩く。実機ではサーボが脚あたり1個増えるコスト。
 *
 * 静力学的には放射形はむしろ有利: 足先がほぼ hip 直下で脚が near-vertical → 保持トルクが小さく、
 * 支持多角形は半径 ringRadius のほぼ円形 → どの向きにも転びにくい（弱サーボの静止保持に向く）。
 *
 * θ=0 のとき a=ŷ で四足/多足の pitch 駆動に一致する一般化。レンダラ/リプレイ互換のため戻り値は
 * 四足と同じ {@link QuadDynReplay}（layout/frames/summary）。
 */
import RAPIER, {
  type RevoluteImpulseJoint,
  type RigidBody,
  type World,
} from '@dimforge/rapier3d-compat';
import { G } from './chain.ts';
import { COURSES, buildCourseColliders, terrainTopAt } from './course.ts';
import {
  resolveConfig,
  footTargetRelHip,
  legIK,
  stabilizeLateral,
  tiltFromUp,
  captureFrame,
  KNEE_SIGN,
  DEG,
  type QuadDynConfig,
  type QuadDynOverrides,
  type QuadDynReplay,
  type QuadBodyLayout,
  type QuadFrame,
} from './quadruped-dynamics.ts';

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** ウニ固有の追加つまみ（残りの物理は QuadDynConfig を流用）。 */
export interface UrchinOverrides extends QuadDynOverrides {
  /** 脚数（放射対称に円環配置）。 */
  legCount?: number;
  /** hip リングの半径 [m]（ハブ半径）。大きいほど支持が広く安定だが脚同士の間隔が要る。 */
  ringRadius?: number;
  /** 歩容: tripod=1つ飛ばしで交互2群（常に約半数接地）, wave=リング順に1脚ずつ。 */
  urchinGait?: 'tripod' | 'wave';
  /**
   * 脚の揺動面。
   * - 'radial'（本物の splay ウニ）: 各脚は自分の放射鉛直面で揺動（関節軸=接線）。1自由度では進行軸に
   *   揃った前後の脚しか足を世界に固定できず、斜め・横の脚は横滑り(scrub)するため極端に遅い（這う）。
   * - 'heading'（操舵ウニ＝ヨー自由度相当）: 全脚が進行方向 +x へ漕ぐ（軸=ŷ）。放射配置のままクモ並みに歩く。
   *   実在の全方向ウニ型が脚にヨー DOF を足すのに対応＝サーボが脚あたり1個増えるコスト。
   */
  legPlane?: 'radial' | 'heading';
}

const DEFAULT_LEG_COUNT = 8;
const DEFAULT_RING_RADIUS = 0.06;

/** 1脚の放射配置（角度）と歩容位相。 */
interface UrchinLegSpec {
  name: string;
  theta: number; // 放射角 [rad]
  phase: number; // 遊脚タイミング [0,1)
}

/** 脚数・歩容から放射レイアウト（角度＋位相）を生成する。 */
export function makeUrchinLegs(n: number, gait: 'tripod' | 'wave'): UrchinLegSpec[] {
  const legs: UrchinLegSpec[] = [];
  for (let i = 0; i < n; i++) {
    const theta = (2 * Math.PI * i) / n;
    // tripod: 偶数/奇数で逆位相(0/0.5)の2群＝隣り合う脚が交互に接地（n 偶数で常に約半数支持）。
    // wave: リング順に等間隔で1脚ずつ遊脚（最も静的に安定だが遅い）。
    const phase = gait === 'tripod' ? (i % 2) * 0.5 : i / n;
    legs.push({ name: `u${i}`, theta, phase });
  }
  return legs;
}

/** 剛体の局所 z 軸が「半径方向 d と鉛直 ẑ が張る面」内でどれだけ傾いたか [rad]（鉛直=0, +d 側で正）。
 *  θ=0（d=x̂, 軸 a=ŷ）で pitchAboutY に一致する一般化。 */
function pitchAboutAxis(body: RigidBody, cos: number, sin: number): number {
  const q = body.rotation();
  const lzx = 2 * (q.x * q.z + q.w * q.y);
  const lzy = 2 * (q.y * q.z - q.w * q.x);
  const lzz = 1 - 2 * (q.x * q.x + q.y * q.y);
  const alongD = lzx * cos + lzy * sin;
  return Math.atan2(alongD, lzz);
}

interface UrchinLeg {
  thigh: RigidBody;
  shin: RigidBody;
  hipJoint: RevoluteImpulseJoint;
  kneeJoint: RevoluteImpulseJoint;
  phase: number;
  cos: number;
  sin: number;
  hipWX: number; // ハブ中心からの hip オフセット x（trunk のヨーは安定化で≈0）
  hipWY: number; // 同 y
}

interface UrchinAssembly {
  trunk: RigidBody;
  legs: UrchinLeg[];
  bodies: RigidBody[]; // 記録順: trunk, (thigh,shin)×N
}

/** 放射多脚機体（ハブ＋N脚）を組む。各脚の関節軸を接線方向にして放射鉛直面で揺動させる。 */
function buildUrchin(
  world: World,
  cfg: QuadDynConfig,
  legSpecs: UrchinLegSpec[],
  ringRadius: number,
  legPlane: 'radial' | 'heading',
): UrchinAssembly {
  const { trunk: T, leg: L } = cfg;
  const hubH = T.height;
  const standZ = hubH / 2 + L.thigh + L.shin + 0.002; // 足先がほぼ z=0
  const hipZ = standZ - hubH / 2;
  const r = L.radius;

  // ハブ: 半径 ringRadius の正方箱（質量は scaled trunk.mass を流用）。
  const trunk = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(0, 0, standZ)
      .setLinearDamping(0.1)
      .setAngularDamping(0.4)
      .setCanSleep(false),
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(ringRadius, ringRadius, hubH / 2)
      .setMass(T.mass)
      .setFriction(cfg.friction),
    trunk,
  );
  trunk.setAdditionalSolverIterations(4);

  const legs: UrchinLeg[] = [];
  const bodies: RigidBody[] = [trunk];

  for (const spec of legSpecs) {
    // hip は常に放射リング上（放射対称の配置）。揺動面(=駆動の cos/sin と関節軸)だけ legPlane で選ぶ。
    const hipWX = ringRadius * Math.cos(spec.theta);
    const hipWY = ringRadius * Math.sin(spec.theta);
    // radial: 揺動面は放射方向 d（軸=接線 a）。heading: 全脚が進行方向 x へ漕ぐ（軸=ŷ）。
    const cos = legPlane === 'heading' ? 1 : Math.cos(spec.theta);
    const sin = legPlane === 'heading' ? 0 : Math.sin(spec.theta);
    const axis = { x: -sin, y: cos, z: 0 }; // 揺動軸（radial=接線, heading=ŷ）

    // 脚は鉛直に吊る（揺動軸 a まわりに回すと放射鉛直面 (d,ẑ) 内で振れる）。
    const thigh = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(hipWX, hipWY, hipZ - L.thigh / 2)
        .setAngularDamping(0.1)
        .setCanSleep(false),
    );
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(r, r, L.thigh / 2)
        .setMass(L.segMass)
        .setFriction(cfg.friction),
      thigh,
    );

    const shin = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(hipWX, hipWY, hipZ - L.thigh - L.shin / 2)
        .setAngularDamping(0.1)
        .setCanSleep(false),
    );
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(r, r, L.shin / 2)
        .setMass(L.segMass)
        .setFriction(cfg.friction)
        .setRestitution(0),
      shin,
    );

    const hipJoint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: hipWX, y: hipWY, z: -hubH / 2 },
        { x: 0, y: 0, z: L.thigh / 2 },
        axis,
      ),
      trunk,
      thigh,
      true,
    ) as RevoluteImpulseJoint;
    hipJoint.setContactsEnabled(false);
    hipJoint.configureMotorModel(RAPIER.MotorModel.ForceBased);

    const kneeJoint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: 0, y: 0, z: -L.thigh / 2 },
        { x: 0, y: 0, z: L.shin / 2 },
        axis,
      ),
      thigh,
      shin,
      true,
    ) as RevoluteImpulseJoint;
    kneeJoint.setContactsEnabled(false);
    kneeJoint.configureMotorModel(RAPIER.MotorModel.ForceBased);

    legs.push({ thigh, shin, hipJoint, kneeJoint, phase: spec.phase, cos, sin, hipWX, hipWY });
    bodies.push(thigh, shin);
  }

  return { trunk, legs, bodies };
}

/** 関節を揺動軸 a まわりの PD で駆動し、±cap でハード clamp（torque モード固定・物理忠実）。
 *  torque モードはトルクをボディへ直接インパルスで与えるので joint ハンドルは不要（四足 driveJoint と同様）。 */
function driveJointAxis(
  parent: RigidBody,
  child: RigidBody,
  target: number,
  cos: number,
  sin: number,
  motor: QuadDynConfig['motor'],
  physicsDt: number,
): { demand: number; applied: number } {
  const rel = pitchAboutAxis(child, cos, sin) - pitchAboutAxis(parent, cos, sin);
  const wc = child.angvel();
  const wp = parent.angvel();
  // 揺動軸 a=(-sin,cos,0) まわりの相対角速度。
  const relVel = (wc.x - wp.x) * -sin + (wc.y - wp.y) * cos;
  const raw = motor.stiffness * (target - rel) - motor.damping * relVel;
  const active = clamp(raw, -motor.maxTorqueNm, motor.maxTorqueNm);
  const total = active - motor.passiveDamping * relVel;
  const imp = total * physicsDt;
  child.applyTorqueImpulse({ x: -sin * imp, y: cos * imp, z: 0 }, true);
  parent.applyTorqueImpulse({ x: sin * imp, y: -cos * imp, z: 0 }, true);
  return { demand: Math.abs(raw), applied: Math.abs(active) };
}

function urchinLayout(cfg: QuadDynConfig, ringRadius: number, n: number): QuadBodyLayout[] {
  const { leg: L, trunk: T } = cfg;
  const layout: QuadBodyLayout[] = [
    { kind: 'trunk', half: [ringRadius, ringRadius, T.height / 2] },
  ];
  for (let i = 0; i < n; i++) {
    layout.push({ kind: 'thigh', half: [L.radius, L.radius, L.thigh / 2] });
    layout.push({ kind: 'shin', half: [L.radius, L.radius, L.shin / 2] });
  }
  return layout;
}

/** 放射ウニの動的歩行を回す。戻り値は四足互換の QuadDynReplay。 */
export async function runUrchinGait(
  overrides: UrchinOverrides = {},
  recordFps = 60,
): Promise<QuadDynReplay> {
  await RAPIER.init();
  const cfg = resolveConfig(overrides);
  const n = Math.max(4, Math.round(overrides.legCount ?? DEFAULT_LEG_COUNT));
  const ringRadius = overrides.ringRadius ?? DEFAULT_RING_RADIUS;
  const legPlane = overrides.legPlane ?? 'radial';
  const legSpecs = makeUrchinLegs(n, overrides.urchinGait ?? 'tripod');

  const substeps = Math.max(1, Math.round(cfg.substeps));
  const physicsDt = cfg.dt / substeps;
  const world = new RAPIER.World({ x: 0, y: 0, z: -G });
  world.timestep = physicsDt;
  world.numSolverIterations = 8;

  const course = overrides.course ?? COURSES.flat();
  buildCourseColliders(world, course, cfg.friction, 3);

  const asm = buildUrchin(world, cfg, legSpecs, ringRadius, legPlane);
  const startX = asm.trunk.translation().x;
  const standZ = asm.trunk.translation().z;
  const fallClearance = standZ * 0.55;
  const fallTilt = 55 * DEG;
  const bodyLen = 2 * ringRadius;

  const layout = urchinLayout(cfg, ringRadius, n);
  const frames: QuadFrame[] = [];
  const steps = Math.ceil(cfg.duration / cfg.dt);
  const recordEvery = recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * cfg.dt))) : Infinity;

  let maxDemandNm = 0;
  let maxAppliedNm = 0;
  let saturatedSteps = 0;
  let minTrunkZ = standZ;
  let maxTiltDeg = 0;
  let fellTime: number | null = null;

  const vBody = cfg.gait.strideM / (cfg.gait.stanceDuty * cfg.gait.period);

  for (let step = 0; step < steps; step++) {
    const t = step * cfg.dt;
    let demand = 0;
    let applied = 0;
    let saturated = false;

    for (let sub = 0; sub < substeps; sub++) {
      const ts = t + sub * physicsDt;
      const trunkX = asm.trunk.translation().x;
      // 進行方向(+x)の胴速度レギュレータ（足を世界の接地点へ収束させ過走/滑りを抑える）。
      const bodyErrX = clamp(trunkX - startX - vBody * ts, -cfg.gait.strideM, cfg.gait.strideM);
      // 胴基準の地形高さ = 全 hip 直下地形の平均（段の縁での基準ジャンプを緩和）。
      let terrainBody = 0;
      for (const leg of asm.legs) terrainBody += terrainTopAt(course, trunkX + leg.hipWX);
      terrainBody /= asm.legs.length;

      for (const leg of asm.legs) {
        const { fx, fz } = footTargetRelHip(cfg, ts, leg.phase);
        // 接地足を「世界に固定」する自然な歩容: 足先の世界 x を後方へ fx だけ掃く（クモと同一）。
        // 脚は自分の放射線 d 上しか足を置けないので fr·cosθ = wdx ⇒ fr = wdx/cosθ。
        //   - 進行軸に近い脚(|cosθ|大): クモと同じ全ストロークで漕いで +x へ寄与
        //   - 真横の脚(|cosθ|≈0): fr が発散するので outrigger 化（fr=0 でその場足踏み・支持専従）
        //     ＝ これが splay(開脚)の物理的代償（横を向いた脚は前進に使えない）。
        const wdx = fx + bodyErrX;
        const reach = cfg.gait.strideM * 1.5;
        const fr = Math.abs(leg.cos) < 0.2 ? 0 : clamp(wdx / leg.cos, -reach, reach);
        // 足先の世界 x（地形参照用）= 胴 + hipのx + (fr·d)のx成分。
        const footWorldX = trunkX + leg.hipWX + fr * leg.cos;
        const terrainDz = terrainTopAt(course, footWorldX) - terrainBody;
        const { p1, p2 } = legIK(fr, fz + terrainDz, cfg.leg.thigh, cfg.leg.shin, KNEE_SIGN);
        // 胴の（この脚の面内での）傾きを相殺。
        const trunkPitch = pitchAboutAxis(asm.trunk, leg.cos, leg.sin);
        const hip = driveJointAxis(
          asm.trunk,
          leg.thigh,
          p1 - trunkPitch,
          leg.cos,
          leg.sin,
          cfg.motor,
          physicsDt,
        );
        const knee = driveJointAxis(
          leg.thigh,
          leg.shin,
          p2 - p1,
          leg.cos,
          leg.sin,
          cfg.motor,
          physicsDt,
        );
        demand = Math.max(demand, hip.demand, knee.demand);
        applied = Math.max(applied, hip.applied, knee.applied);
        saturated ||= hip.demand - hip.applied > 1e-9 || knee.demand - knee.applied > 1e-9;
      }
      stabilizeLateral(asm.trunk, cfg.lateralStabK, cfg.lateralStabD, physicsDt);
      world.step();
    }

    const trunkZ = asm.trunk.translation().z;
    const tilt = tiltFromUp(asm.trunk);
    const forwardX = asm.trunk.translation().x - startX;
    const bodyClearance = trunkZ - terrainTopAt(course, asm.trunk.translation().x);
    const fallen = bodyClearance < fallClearance || tilt > fallTilt;
    if (fallen && fellTime === null) fellTime = t;

    maxDemandNm = Math.max(maxDemandNm, demand);
    maxAppliedNm = Math.max(maxAppliedNm, applied);
    if (saturated) saturatedSteps++;
    minTrunkZ = Math.min(minTrunkZ, trunkZ);
    maxTiltDeg = Math.max(maxTiltDeg, tilt / DEG);

    if (step % recordEvery === 0) {
      frames.push(
        captureFrame(asm, t, {
          demandNm: demand,
          appliedNm: applied,
          saturated,
          trunkZ,
          tiltDeg: tilt / DEG,
          forwardX,
          fallen,
        }),
      );
    }
  }

  const forwardDistanceM = asm.trunk.translation().x - startX;
  const fell = fellTime !== null;
  const success = !fell && forwardDistanceM > bodyLen * 0.5;

  // summary.config は四足互換のため legs に放射脚を畳んで入れる必要はない（描画は layout/frames が担う）。
  const summary = {
    config: { ...cfg, duration: cfg.duration } as QuadDynConfig,
    forwardDistanceM,
    maxDemandNm,
    maxAppliedNm,
    saturatedSteps,
    minTrunkZ,
    maxTiltDeg,
    fell,
    fellTime,
    success,
  };

  world.free();
  return { layout, frames, summary };
}
