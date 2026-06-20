/**
 * 四足（quadruped）の 3D 動的歩行シミュレーション（Rapier）。
 *
 * トランク（胴）＋4脚（各 thigh+shin の2リンク・hip/膝の2関節, すべて y 軸 pitch）を剛体で作り、
 * IK クロール歩容を「忠実トルク」で駆動する:
 *  1. 足先のカートシアン軌道（立脚は胴に対し後方へ等速で掃く→接地摩擦で前進、遊脚は持ち上げて戻す）
 *  2. 2リンク IK で hip/膝の目標角に変換
 *  3. PD トルクを ±maxTorqueNm でハード clamp し、作用反作用で実トルクとして印加（受動ダンピング併用）
 * cap が保持・推進に足りなければ崩れ前進できない＝cap→歩行性能が単調（弱いサーボは失速）。
 * 接触・摩擦・転倒は Rapier が解く。蛇の physical attempt の四足版。
 *
 * 脚は pitch のみで横方向(ロール/ヨー)を制御できないため、横バランスは別機構前提として
 * トランクのロール/ヨーだけ外部トルクで安定化し、矢状面の歩行・トルク充足を切り出して評価する。
 */
import RAPIER, {
  type RevoluteImpulseJoint,
  type RigidBody,
  type World,
} from '@dimforge/rapier3d-compat';
import { G } from './chain.ts';

export interface QuadDynConfig {
  trunk: { length: number; width: number; height: number; mass: number };
  leg: { thigh: number; shin: number; segMass: number; radius: number };
  hipInset: number; // トランク角からhipを内側へ寄せる量 [m]
  // mode='torque': 能動PDトルクを ±maxTorqueNm でハード clamp ＋ 受動ダンピング(ギア摩擦相当, cap外)を
  //   作用反作用で直接かける（物理忠実・単調）。極端に軽い脚が強capで暴走するのを受動ダンピングが抑える。
  // mode='position': 内蔵位置モーターで目標角を詰めて出力を ±cap に頭打ち（旧式・中間帯で非単調）。
  motor: {
    stiffness: number; // 能動P（cap対象）
    damping: number; // 能動D（cap対象）
    passiveDamping: number; // 受動ダンピング係数 [N·m/(rad/s)]（cap外・常時）＝ギア摩擦相当
    maxTorqueNm: number;
    mode: 'torque' | 'position';
  };
  friction: number;
  dt: number; // 制御・記録の時間刻み [s]
  substeps: number; // 1制御刻みあたりの物理サブステップ数（torque モードの陽解法を安定化）
  // 横安定化: 脚が pitch(y軸)のみで横方向(ロール/ヨー)を制御できないため、横バランスは別機構前提として
  // トランクのロール・ヨーを外部トルクで水平/直進に保つ。これで矢状面の歩行・トルク充足を切り出して評価する。
  // 0 で無効（純物理）。矢状面(pitch)の歩行はこの安定化では駆動されない＝サーボのトルク充足が結果を決める。
  lateralStabK: number; // ロール/ヨー復元剛性 [N·m/rad]
  lateralStabD: number; // 同ダンピング [N·m/(rad/s)]
  duration: number;
  // IK クロール歩容: 足先のカートシアン軌道で指定。立脚は足を胴に対し後方へ掃き、接地摩擦で胴を前進させる。
  gait: {
    period: number; // 1脚の1周期 [s]
    strideM: number; // 1歩の足の前後ストローク [m]
    liftM: number; // 遊脚の持ち上げ高さ [m]
    standM: number; // hip から足先までの保持高さ [m]（< thigh+shin で膝を曲げる）
    stanceDuty: number; // 接地が占める周期割合（残りが遊脚）。0.75=常時3脚接地
  };
}

export const DEFAULT_QUAD_DYN_CONFIG: QuadDynConfig = {
  // 合計質量 ≈ 0.52 + 8×0.085 = 1.2kg。脚 segMass は関節慣性を現実的にし（軽すぎる脚は強い
  // サーボのトルクで暴走する＝関節にサーボ＋ギアの反映慣性がある実機に近づける）、忠実トルクでも
  // 安定して歩けるようにするための値。
  trunk: { length: 0.24, width: 0.16, height: 0.05, mass: 0.52 },
  leg: { thigh: 0.09, shin: 0.09, segMass: 0.085, radius: 0.015 },
  hipInset: 0.02,
  motor: { stiffness: 6, damping: 0.3, passiveDamping: 0.02, maxTorqueNm: 3.0, mode: 'torque' },
  friction: 0.9,
  dt: 1 / 240,
  substeps: 8,
  lateralStabK: 4,
  lateralStabD: 0.3,
  duration: 5,
  // IK クロール: 1脚ずつ遊脚、常時3脚接地で静的に安定。足は胴に対し後方へ掃いて前進。
  gait: { period: 1.2, strideM: 0.05, liftM: 0.03, standM: 0.165, stanceDuty: 0.75 },
};

export interface QuadBodyLayout {
  kind: 'trunk' | 'thigh' | 'shin';
  half: [number, number, number]; // 箱の半寸法 [hx, hy, hz]
}

export interface QuadFrameDiag {
  demandNm: number; // このフレームの最大要求トルク
  appliedNm: number; // 同・実印加（cap後）
  saturated: boolean;
  trunkZ: number; // 胴の高さ [m]
  tiltDeg: number; // 胴の上軸が鉛直からどれだけ傾いたか [deg]
  forwardX: number; // 胴の前進量 [m]
  fallen: boolean;
}

export interface QuadFrame {
  t: number;
  bodies: Array<{ p: [number, number, number]; q: [number, number, number, number] }>;
  diag: QuadFrameDiag;
}

export interface QuadDynSummary {
  config: QuadDynConfig;
  forwardDistanceM: number; // 前進した距離 [m]
  maxDemandNm: number;
  maxAppliedNm: number;
  saturatedSteps: number;
  minTrunkZ: number;
  maxTiltDeg: number;
  fell: boolean;
  fellTime: number | null;
  success: boolean; // 転ばず一定距離前進できたか
}

export interface QuadDynReplay {
  layout: QuadBodyLayout[];
  frames: QuadFrame[];
  summary: QuadDynSummary;
}

// クロール: 4脚を1/4周期ずつずらし、1脚ずつ振る（常時3脚支持＝静的安定）
const LEGS = [
  { name: 'FL', sx: +1, sy: +1, phase: 0 },
  { name: 'RR', sx: -1, sy: -1, phase: 0.25 },
  { name: 'FR', sx: +1, sy: -1, phase: 0.5 },
  { name: 'RL', sx: -1, sy: +1, phase: 0.75 },
] as const;

const DEG = Math.PI / 180;
// configureMotorPosition の角度符号（pitchAboutY 規約に対する内蔵モーターの向き）
const MOTOR_AXIS_SIGN = -1;
// 2リンク IK の肘の向き（膝が前/後ろどちらに曲がるか）。歩行方向が正になるよう選ぶ。
const KNEE_SIGN = 1;

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** 剛体の局所 z 軸が矢状面(x-z)でどれだけ傾いているか（鉛直=0, 前傾で正）。 */
function pitchAboutY(body: RigidBody): number {
  const q = body.rotation();
  // localZ=(0,0,1) を q で回した world 方向
  const x = 2 * (q.x * q.z + q.w * q.y);
  const z = 1 - 2 * (q.x * q.x + q.y * q.y);
  return Math.atan2(x, z);
}

/**
 * トランクの横方向(ロール=x軸まわり / ヨー=z軸まわり)を水平・直進へ戻す外部トルクをインパルスで与える。
 * 脚は pitch(y) のみで横を制御できないため、横バランスは別機構前提というモデリング。矢状面(pitch)の
 * 歩行・推進はこの安定化では駆動されない（=サーボのトルク充足が前進可否を決める）。
 */
function stabilizeLateral(trunk: RigidBody, k: number, d: number, physicsDt: number): void {
  if (k <= 0 && d <= 0) return;
  const q = trunk.rotation();
  // up = local z を world へ。lateral lean(ロール) ≈ up の y 成分。
  const upY = 2 * (q.y * q.z - q.w * q.x);
  // forward = local x を world へ。heading(ヨー) = atan2(fwd_y, fwd_x)。
  const fwdX = 1 - 2 * (q.y * q.y + q.z * q.z);
  const fwdY = 2 * (q.x * q.y + q.w * q.z);
  const yaw = Math.atan2(fwdY, fwdX);
  const w = trunk.angvel();
  // upY ≈ -roll(x軸まわり) なので復元トルクは +k·upY。ヨーは -k·yaw。
  const tauX = k * upY - d * w.x; // ロール復元（upのy成分を0へ）
  const tauZ = -k * yaw - d * w.z; // ヨー復元（直進へ）
  trunk.applyTorqueImpulse({ x: tauX * physicsDt, y: 0, z: tauZ * physicsDt }, true);
}

/** 胴の上軸(local z)が world 上(z)からどれだけ傾いたか [rad]。 */
function tiltFromUp(body: RigidBody): number {
  const q = body.rotation();
  const upZ = 1 - 2 * (q.x * q.x + q.y * q.y); // world z 成分
  return Math.acos(clamp(upZ, -1, 1));
}

interface Leg {
  thigh: RigidBody;
  shin: RigidBody;
  hipJoint: RevoluteImpulseJoint;
  kneeJoint: RevoluteImpulseJoint;
  phase: number;
}

interface QuadAssembly {
  trunk: RigidBody;
  legs: Leg[];
  bodies: RigidBody[]; // 記録順: trunk, (thigh,shin)×4
}

function buildQuad(world: World, cfg: QuadDynConfig): QuadAssembly {
  const { trunk: T, leg: L } = cfg;
  const standZ = T.height / 2 + L.thigh + L.shin + 0.002; // 足先がほぼ z=0
  const r = L.radius;

  const trunk = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(0, 0, standZ)
      .setLinearDamping(0.1)
      .setAngularDamping(0.4)
      .setCanSleep(false),
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(T.length / 2, T.width / 2, T.height / 2)
      .setMass(T.mass)
      .setFriction(cfg.friction),
    trunk,
  );
  trunk.setAdditionalSolverIterations(4);

  const legs: Leg[] = [];
  const bodies: RigidBody[] = [trunk];

  for (const spec of LEGS) {
    const hipX = spec.sx * (T.length / 2 - cfg.hipInset);
    const hipY = spec.sy * (T.width / 2);
    const hipZ = standZ - T.height / 2;

    const thigh = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(hipX, hipY, hipZ - L.thigh / 2)
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
        .setTranslation(hipX, hipY, hipZ - L.thigh - L.shin / 2)
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

    // hip: trunk角(下端) ↔ thigh上端、軸 y
    const hipJoint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: hipX, y: hipY, z: -T.height / 2 },
        { x: 0, y: 0, z: L.thigh / 2 },
        { x: 0, y: 1, z: 0 },
      ),
      trunk,
      thigh,
      true,
    ) as RevoluteImpulseJoint;
    hipJoint.setContactsEnabled(false);
    hipJoint.configureMotorModel(RAPIER.MotorModel.ForceBased);
    // 膝: thigh下端 ↔ shin上端、軸 y
    const kneeJoint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: 0, y: 0, z: -L.thigh / 2 },
        { x: 0, y: 0, z: L.shin / 2 },
        { x: 0, y: 1, z: 0 },
      ),
      thigh,
      shin,
      true,
    ) as RevoluteImpulseJoint;
    kneeJoint.setContactsEnabled(false);
    kneeJoint.configureMotorModel(RAPIER.MotorModel.ForceBased);

    legs.push({ thigh, shin, hipJoint, kneeJoint, phase: spec.phase });
    bodies.push(thigh, shin);
  }

  return { trunk, legs, bodies };
}

/**
 * 遊脚の足先軌道を滑らかにする補間（0→1）。
 */
function smoothstep(u: number): number {
  const x = clamp(u, 0, 1);
  return x * x * (3 - 2 * x);
}

/**
 * 矢状面(x-z)の足先目標（hip 基準）。立脚は足を前(+x)から後(-x)へ等速で掃き、接地摩擦で
 * 胴を前進させる。遊脚は後→前へ戻しつつ放物線で持ち上げる。常時3脚接地（stanceDuty）。
 * 返り値: fx=前後, fz=上下（足は hip より下なので負）。
 */
function footTargetRelHip(
  cfg: QuadDynConfig,
  t: number,
  phase: number,
): { fx: number; fz: number } {
  const g = cfg.gait;
  const psi = (((t / g.period + phase) % 1) + 1) % 1;
  const half = g.strideM / 2;
  if (psi < g.stanceDuty) {
    // 立脚: 足を +half → -half へ後方へ掃く（=胴を前進）。
    const u = psi / g.stanceDuty;
    return { fx: half - g.strideM * u, fz: -g.standM };
  }
  // 遊脚: -half → +half へ戻し、放物線で持ち上げる。
  const u = (psi - g.stanceDuty) / (1 - g.stanceDuty);
  return { fx: -half + g.strideM * smoothstep(u), fz: -g.standM + g.liftM * Math.sin(Math.PI * u) };
}

/**
 * 2リンク平面 IK。hip 基準の足先目標 (fx, fz)（fz<0）を thigh/shin の world pitch (p1,p2) に。
 * リンクの hip→distal 方向は pitchAboutY=p に対して (-sin p, -cos p)。kneeSign で肘の向きを選ぶ。
 */
function legIK(
  fx: number,
  fz: number,
  l1: number,
  l2: number,
  kneeSign: number,
): { p1: number; p2: number } {
  const dMin = Math.abs(l1 - l2) + 1e-3;
  const dMax = l1 + l2 - 1e-3;
  const dRaw = Math.hypot(fx, fz);
  const d = clamp(dRaw, dMin, dMax);
  // hip→foot 方向 (fx,fz)/d = (-sin psi, -cos psi)
  const psi = Math.atan2(-fx, -fz);
  // thigh と hip-foot 線のなす角
  const cosA = clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1);
  const a = Math.acos(cosA);
  const p1 = psi + kneeSign * a;
  // 膝位置から shin の pitch を直接求める（符号曖昧を避ける）
  const kneeX = -l1 * Math.sin(p1);
  const kneeZ = -l1 * Math.cos(p1);
  const p2 = Math.atan2(-(fx - kneeX), -(fz - kneeZ));
  return { p1, p2 };
}

/**
 * 足先目標＋IK から hip/膝の相対角目標を出す。rel=pitch(child)-pitch(parent) 規約に合わせ、
 * hip 目標 = p1 - trunkPitch（胴の傾きを相殺）、膝目標 = p2 - p1。
 */
function gaitTargets(
  cfg: QuadDynConfig,
  trunkPitch: number,
  t: number,
  phase: number,
): { hip: number; knee: number } {
  const { fx, fz } = footTargetRelHip(cfg, t, phase);
  const { p1, p2 } = legIK(fx, fz, cfg.leg.thigh, cfg.leg.shin, KNEE_SIGN);
  return { hip: p1 - trunkPitch, knee: p2 - p1 };
}

/**
 * 関節を PD で駆動し、トルク cap を効かせる。2方式:
 *
 * - mode='torque'（物理忠実・単調）: PD トルクを ±maxTorqueNm でハード clamp し、親子リンクへ
 *   実トルクをインパルス(τ·physicsDt)で与える（addTorque は reset まで永続=累積するため
 *   applyTorqueImpulse を使う）。pitchAboutY は +y まわりの回転角なので child へ +τ / parent へ −τ。
 *   cap が足りなければ保持・推進に負けて崩れる＝cap→歩行性能が単調。
 * - mode='position'（旧式・比較用）: 内蔵位置モーターで目標角を詰めて出力を ±cap に頭打ち。
 *   中間トルク帯で cap→前進距離 が非単調になる。
 */
function driveJoint(
  joint: RevoluteImpulseJoint,
  parent: RigidBody,
  child: RigidBody,
  target: number,
  positionAxisSign: number,
  motor: QuadDynConfig['motor'],
  physicsDt: number,
): { demand: number; applied: number } {
  const rel = pitchAboutY(child) - pitchAboutY(parent);
  const relVel = child.angvel().y - parent.angvel().y;
  const raw = motor.stiffness * (target - rel) - motor.damping * relVel;

  if (motor.mode === 'position') {
    let effectiveTarget = target;
    let applied = raw;
    if (Math.abs(raw) > motor.maxTorqueNm) {
      const capped = Math.sign(raw) * motor.maxTorqueNm;
      effectiveTarget = rel + (capped + motor.damping * relVel) / motor.stiffness;
      applied = capped;
    }
    joint.configureMotorPosition(
      positionAxisSign * effectiveTarget,
      motor.stiffness,
      motor.damping,
    );
    return { demand: Math.abs(raw), applied: Math.abs(applied) };
  }

  // 能動トルク（±cap）＋ 受動ダンピング（cap外・常時）。後者がギア摩擦相当で暴走を抑える。
  const active = clamp(raw, -motor.maxTorqueNm, motor.maxTorqueNm);
  const total = active - motor.passiveDamping * relVel;
  const impulse = total * physicsDt;
  child.applyTorqueImpulse({ x: 0, y: impulse, z: 0 }, true);
  parent.applyTorqueImpulse({ x: 0, y: -impulse, z: 0 }, true);
  return { demand: Math.abs(raw), applied: Math.abs(active) };
}

function layoutOf(cfg: QuadDynConfig): QuadBodyLayout[] {
  const { trunk: T, leg: L } = cfg;
  const layout: QuadBodyLayout[] = [
    { kind: 'trunk', half: [T.length / 2, T.width / 2, T.height / 2] },
  ];
  for (let i = 0; i < LEGS.length; i++) {
    layout.push({ kind: 'thigh', half: [L.radius, L.radius, L.thigh / 2] });
    layout.push({ kind: 'shin', half: [L.radius, L.radius, L.shin / 2] });
  }
  return layout;
}

function captureFrame(asm: QuadAssembly, t: number, diag: QuadFrameDiag): QuadFrame {
  return {
    t,
    bodies: asm.bodies.map((b) => {
      const p = b.translation();
      const q = b.rotation();
      return { p: [p.x, p.y, p.z], q: [q.x, q.y, q.z, q.w] };
    }),
    diag,
  };
}

/** ネストしたグループも部分指定できる runQuadrupedGait の上書き型。 */
export interface QuadDynOverrides {
  trunk?: Partial<QuadDynConfig['trunk']>;
  leg?: Partial<QuadDynConfig['leg']>;
  motor?: Partial<QuadDynConfig['motor']>;
  gait?: Partial<QuadDynConfig['gait']>;
  hipInset?: number;
  friction?: number;
  dt?: number;
  substeps?: number;
  duration?: number;
}

export async function runQuadrupedGait(
  overrides: QuadDynOverrides = {},
  recordFps = 60,
): Promise<QuadDynReplay> {
  await RAPIER.init();
  const cfg: QuadDynConfig = {
    ...DEFAULT_QUAD_DYN_CONFIG,
    ...overrides,
    trunk: { ...DEFAULT_QUAD_DYN_CONFIG.trunk, ...overrides.trunk },
    leg: { ...DEFAULT_QUAD_DYN_CONFIG.leg, ...overrides.leg },
    motor: { ...DEFAULT_QUAD_DYN_CONFIG.motor, ...overrides.motor },
    gait: { ...DEFAULT_QUAD_DYN_CONFIG.gait, ...overrides.gait },
  };

  const substeps = Math.max(1, Math.round(cfg.substeps));
  const physicsDt = cfg.dt / substeps;
  const world = new RAPIER.World({ x: 0, y: 0, z: -G });
  world.timestep = physicsDt;
  world.numSolverIterations = 8;

  // 地面
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(3, 3, 0.05).setTranslation(0, 0, -0.05).setFriction(cfg.friction),
  );

  const asm = buildQuad(world, cfg);
  const startX = asm.trunk.translation().x;
  const standZ = asm.trunk.translation().z;
  const fallZ = standZ * 0.55; // 胴がここまで落ちたら転倒
  const fallTilt = 55 * DEG;

  const layout = layoutOf(cfg);
  const frames: QuadFrame[] = [];
  const steps = Math.ceil(cfg.duration / cfg.dt);
  const recordEvery = recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * cfg.dt))) : Infinity;

  let maxDemandNm = 0;
  let maxAppliedNm = 0;
  let saturatedSteps = 0;
  let minTrunkZ = standZ;
  let maxTiltDeg = 0;
  let fellTime: number | null = null;

  for (let step = 0; step < steps; step++) {
    const t = step * cfg.dt;

    let demand = 0;
    let applied = 0;
    let saturated = false;
    // 物理サブステップごとに IK 目標と PD を再計算（torque モードの陽解法を安定化）。
    for (let sub = 0; sub < substeps; sub++) {
      const ts = t + sub * physicsDt;
      const trunkPitch = pitchAboutY(asm.trunk);
      for (const leg of asm.legs) {
        const tgt = gaitTargets(cfg, trunkPitch, ts, leg.phase);
        const hip = driveJoint(
          leg.hipJoint,
          asm.trunk,
          leg.thigh,
          tgt.hip,
          MOTOR_AXIS_SIGN,
          cfg.motor,
          physicsDt,
        );
        const knee = driveJoint(
          leg.kneeJoint,
          leg.thigh,
          leg.shin,
          tgt.knee,
          MOTOR_AXIS_SIGN,
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
    const fallen = trunkZ < fallZ || tilt > fallTilt;
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
  // 成功 = 転ばず、体長の半分以上 前進した
  const success = !fell && forwardDistanceM > cfg.trunk.length * 0.5;

  const summary: QuadDynSummary = {
    config: cfg,
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
