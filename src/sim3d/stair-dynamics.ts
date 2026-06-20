import RAPIER, {
  type Collider,
  type RevoluteImpulseJoint,
  type RigidBody,
  type TempContactManifold,
  type World,
} from '@dimforge/rapier3d-compat';

import { G, reachArc, staticTorques } from './chain.ts';

export interface Morphology3d {
  n: number;
  totalLength: number;
  totalMass: number;
  bodyWidth: number;
  bodyThickness: number;
}

export interface StairGeometry {
  rise: number;
  forward: number;
  treadDepth: number;
  stepCount: number;
}

export interface JointMotorConfig {
  stiffness: number;
  damping: number;
  maxTorqueNm: number;
}

export interface StairDynamicsConfig {
  morphology: Morphology3d;
  stair: StairGeometry;
  friction: number;
  dt: number;
  duration: number;
  motor: JointMotorConfig;
  clearance: number;
  tractionProbeForceN: number;
}

export interface StairDynamicsSummary {
  config: StairDynamicsConfig;
  liftLinks: number;
  staticArcPeakNm: number;
  maxDemandTorqueNm: number;
  maxAppliedTorqueNm: number;
  maxDemandJoint: number;
  maxAppliedJoint: number;
  saturatedSteps: number;
  maxContactNormalForceN: number;
  maxContactTangentForceN: number;
  maxMuDemand: number;
  maxProbeMuDemand: number;
  frictionLimitedSamples: number;
  finalHeadTip: [number, number];
  maxHeadTipZ: number;
  success: boolean;
}

export interface StairDynamicsLinkFrame {
  x: number;
  z: number;
  angle: number;
}

export interface StairDynamicsFrame {
  t: number;
  links: StairDynamicsLinkFrame[];
  headTip: [number, number];
}

export interface StairDynamicsReplay {
  summary: StairDynamicsSummary;
  frames: StairDynamicsFrame[];
}

interface ChainBody {
  body: RigidBody;
  collider: Collider;
}

interface ChainAssembly {
  links: ChainBody[];
  joints: RevoluteImpulseJoint[];
}

interface TargetKeyframes {
  prepare: number[];
  overEdge: number[];
  place: number[];
}

export const DEFAULT_STAIR_DYNAMICS_CONFIG: StairDynamicsConfig = {
  morphology: {
    n: 8,
    totalLength: 0.9,
    totalMass: 1.0,
    bodyWidth: 0.04,
    bodyThickness: 0.028,
  },
  stair: {
    rise: 0.18,
    forward: 0.1,
    treadDepth: 0.25,
    stepCount: 3,
  },
  friction: 0.8,
  dt: 1 / 240,
  duration: 4,
  motor: {
    stiffness: 5,
    damping: 0.18,
    maxTorqueNm: 8,
  },
  clearance: 0.06,
  tractionProbeForceN: 3,
};

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function smoothstep(t: number): number {
  const u = clamp(t, 0, 1);
  return u * u * (3 - 2 * u);
}

function mixAngles(a: number[], b: number[], t: number): number[] {
  return a.map((v, i) => v + normalizeAngle(b[i] - v) * t);
}

function normalizeAngle(rad: number): number {
  let x = rad;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

function quatFromSagittalAngle(theta: number): RAPIER.Quaternion {
  return new RAPIER.Quaternion(0, -Math.sin(theta / 2), 0, Math.cos(theta / 2));
}

function sagittalAngle(body: RigidBody): number {
  const q = body.rotation();
  const dirX = 1 - 2 * (q.y * q.y + q.z * q.z);
  const dirZ = 2 * (q.x * q.z - q.y * q.w);
  return normalizeAngle(Math.atan2(dirZ, dirX));
}

function sagittalAngularVelocity(body: RigidBody): number {
  return -body.angvel().y;
}

function chooseLiftLinks(config: StairDynamicsConfig): number {
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const candidates = [
    [config.stair.forward * 0.4, config.stair.rise + config.clearance],
    [config.stair.forward + 0.05, config.stair.rise + config.clearance * 0.75],
    [config.stair.forward + 0.06, config.stair.rise + 0.01],
  ];
  const maxChord = Math.max(...candidates.map(([x, z]) => Math.hypot(x, z)));

  let liftLinks = 1;
  while (liftLinks < config.morphology.n && liftLinks * linkLen < 1.15 * maxChord) {
    liftLinks++;
  }
  return liftLinks;
}

function buildTargets(config: StairDynamicsConfig, liftLinks: number): TargetKeyframes {
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const lengths = new Array<number>(liftLinks).fill(linkLen);

  const fullTarget = (front: number[]): number[] => [
    ...new Array<number>(config.morphology.n - liftLinks).fill(0),
    ...front,
  ];

  const arc = (x: number, z: number): number[] => {
    const result = reachArc(lengths, x, z);
    if (!result.reachable) {
      throw new Error(
        `unreachable stair target: x=${x.toFixed(3)} z=${z.toFixed(3)} with ${liftLinks} lift links`,
      );
    }
    return result.absAngles;
  };

  return {
    prepare: fullTarget(arc(config.stair.forward * 0.4, config.stair.rise + config.clearance)),
    overEdge: fullTarget(
      arc(config.stair.forward + 0.05, config.stair.rise + config.clearance * 0.75),
    ),
    place: fullTarget(arc(config.stair.forward + 0.06, config.stair.rise + 0.01)),
  };
}

function targetAt(t: number, targets: TargetKeyframes): number[] {
  if (t < 0.4) return targets.prepare;
  if (t < 1.8) return mixAngles(targets.prepare, targets.overEdge, smoothstep((t - 0.4) / 1.4));
  if (t < 2.8) return mixAngles(targets.overEdge, targets.place, smoothstep((t - 1.8) / 1.0));
  return targets.place;
}

function createStairs(world: World, config: StairDynamicsConfig): Collider[] {
  const env: Collider[] = [];
  const halfDepth = config.stair.treadDepth / 2;
  const halfWidth = 0.35;
  const baseThickness = 0.04;

  env.push(
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(1.4, halfWidth, baseThickness / 2)
        .setTranslation(-0.7, 0, -baseThickness / 2)
        .setFriction(config.friction),
    ),
  );

  for (let i = 0; i < config.stair.stepCount; i++) {
    const height = (i + 1) * config.stair.rise;
    env.push(
      world.createCollider(
        RAPIER.ColliderDesc.cuboid(halfDepth, halfWidth, height / 2)
          .setTranslation(i * config.stair.treadDepth + halfDepth, 0, height / 2)
          .setFriction(config.friction),
      ),
    );
  }

  return env;
}

function centerlinePoints(
  config: StairDynamicsConfig,
  liftLinks: number,
  absAngles: number[],
): Array<[number, number]> {
  const n = config.morphology.n;
  const linkLen = config.morphology.totalLength / n;
  const halfThickness = config.morphology.bodyThickness / 2;
  const baseZ = halfThickness + 0.006;
  const liftOffX = -config.stair.forward;
  const tailX = liftOffX - (n - liftLinks) * linkLen;

  const points: Array<[number, number]> = [[tailX, baseZ]];
  for (let i = 0; i < n; i++) {
    const [x, z] = points[i];
    points.push([x + linkLen * Math.cos(absAngles[i]), z + linkLen * Math.sin(absAngles[i])]);
  }
  return points;
}

function createChain(
  world: World,
  config: StairDynamicsConfig,
  liftLinks: number,
  initialAbsAngles: number[],
): ChainAssembly {
  const n = config.morphology.n;
  const linkLen = config.morphology.totalLength / n;
  const linkMass = config.morphology.totalMass / n;
  const halfWidth = config.morphology.bodyWidth / 2;
  const halfThickness = config.morphology.bodyThickness / 2;
  const points = centerlinePoints(config, liftLinks, initialAbsAngles);
  const chain: ChainBody[] = [];
  const joints: RevoluteImpulseJoint[] = [];

  for (let i = 0; i < n; i++) {
    const [x0, z0] = points[i];
    const [x1, z1] = points[i + 1];
    const theta = Math.atan2(z1 - z0, x1 - x0);
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation((x0 + x1) / 2, 0, (z0 + z1) / 2)
        .setRotation(quatFromSagittalAngle(theta))
        .setLinearDamping(0.2)
        .setAngularDamping(0.05)
        .setCanSleep(false),
    );

    const collider = world.createCollider(
      RAPIER.ColliderDesc.cuboid(linkLen / 2, halfWidth, halfThickness)
        .setMass(linkMass)
        .setFriction(config.friction)
        .setRestitution(0),
      body,
    );

    body.setAdditionalSolverIterations(2);
    chain.push({ body, collider });
  }

  for (let i = 0; i < n - 1; i++) {
    const joint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: linkLen / 2, y: 0, z: 0 },
        { x: -linkLen / 2, y: 0, z: 0 },
        { x: 0, y: 1, z: 0 },
      ),
      chain[i].body,
      chain[i + 1].body,
      true,
    ) as RevoluteImpulseJoint;
    joint.setContactsEnabled(false);
    joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
    joints.push(joint);
  }

  return { links: chain, joints };
}

function configureJointMotors(
  assembly: ChainAssembly,
  targetAbsAngles: number[],
  motor: JointMotorConfig,
): {
  maxDemand: number;
  maxApplied: number;
  demandJoint: number;
  appliedJoint: number;
  saturated: boolean;
} {
  let maxDemand = 0;
  let maxApplied = 0;
  let demandJoint = 0;
  let appliedJoint = 0;
  let saturated = false;

  for (const link of assembly.links) {
    link.body.resetTorques(true);
    link.body.resetForces(true);
  }

  for (let j = 0; j < assembly.links.length - 1; j++) {
    const parent = assembly.links[j].body;
    const child = assembly.links[j + 1].body;
    const current = normalizeAngle(sagittalAngle(child) - sagittalAngle(parent));
    const target = normalizeAngle(targetAbsAngles[j + 1] - targetAbsAngles[j]);
    const relVel = sagittalAngularVelocity(child) - sagittalAngularVelocity(parent);
    const demand = motor.stiffness * normalizeAngle(target - current) - motor.damping * relVel;
    const applied = clamp(demand, -motor.maxTorqueNm, motor.maxTorqueNm);
    assembly.joints[j].configureMotorPosition(-target, motor.stiffness, motor.damping);

    if (Math.abs(demand) > maxDemand) {
      maxDemand = Math.abs(demand);
      demandJoint = j;
    }
    if (Math.abs(applied) > maxApplied) {
      maxApplied = Math.abs(applied);
      appliedJoint = j;
    }
    saturated ||= Math.abs(demand - applied) > 1e-9;
  }

  return { maxDemand, maxApplied, demandJoint, appliedJoint, saturated };
}

function applyTractionProbe(assembly: ChainAssembly, t: number, config: StairDynamicsConfig): void {
  if (t < 2.8 || config.tractionProbeForceN <= 0) return;

  // Small forward body load after the head is placed on the tread. This is not a gait;
  // it is a repeatable probe to make contact friction show up in the log.
  assembly.links[0].body.addForce({ x: config.tractionProbeForceN, y: 0, z: 0 }, true);
}

function readContacts(
  world: World,
  chain: ChainBody[],
  env: Collider[],
  dt: number,
  friction: number,
): {
  maxNormalForce: number;
  maxTangentForce: number;
  totalNormalForce: number;
  maxMuDemand: number;
  frictionLimitedSamples: number;
} {
  let maxNormalForce = 0;
  let maxTangentForce = 0;
  let totalNormalForce = 0;
  let maxMuDemand = 0;
  let frictionLimitedSamples = 0;

  const readManifold = (manifold: TempContactManifold): void => {
    for (let i = 0; i < manifold.numContacts(); i++) {
      const normalImpulse = Math.max(0, manifold.contactImpulse(i));
      const tangentImpulse = Math.hypot(
        manifold.contactTangentImpulseX(i),
        manifold.contactTangentImpulseY(i),
      );
      const normalForce = normalImpulse / dt;
      const tangentForce = tangentImpulse / dt;
      const muDemand = normalImpulse > 1e-9 ? tangentImpulse / normalImpulse : 0;

      totalNormalForce += normalForce;
      maxNormalForce = Math.max(maxNormalForce, normalForce);
      maxTangentForce = Math.max(maxTangentForce, tangentForce);
      maxMuDemand = Math.max(maxMuDemand, muDemand);
      if (normalImpulse > 1e-9 && muDemand > friction * 0.9) frictionLimitedSamples++;
    }
  };

  for (const link of chain) {
    for (const obstacle of env) {
      world.narrowPhase.contactPair(link.collider.handle, obstacle.handle, readManifold);
    }
  }

  return { maxNormalForce, maxTangentForce, totalNormalForce, maxMuDemand, frictionLimitedSamples };
}

function headTip(chain: ChainBody[], linkLen: number): [number, number] {
  const head = chain[chain.length - 1].body;
  const theta = sagittalAngle(head);
  const p = head.translation();
  return [p.x + (linkLen / 2) * Math.cos(theta), p.z + (linkLen / 2) * Math.sin(theta)];
}

function captureFrame(chain: ChainBody[], linkLen: number, t: number): StairDynamicsFrame {
  return {
    t,
    links: chain.map(({ body }) => {
      const p = body.translation();
      return { x: p.x, z: p.z, angle: sagittalAngle(body) };
    }),
    headTip: headTip(chain, linkLen),
  };
}

function staticArcPeak(config: StairDynamicsConfig, liftLinks: number): number {
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const linkMass = config.morphology.totalMass / config.morphology.n;
  const lengths = new Array<number>(liftLinks).fill(linkLen);
  const masses = new Array<number>(liftLinks).fill(linkMass);
  const arc = reachArc(lengths, config.stair.forward, config.stair.rise);
  return staticTorques(lengths, masses, arc.absAngles).peak;
}

async function simulateStairDynamics(
  overrides: Partial<StairDynamicsConfig> = {},
  recordFps = 0,
): Promise<StairDynamicsReplay> {
  await RAPIER.init();

  const config: StairDynamicsConfig = {
    ...DEFAULT_STAIR_DYNAMICS_CONFIG,
    ...overrides,
    morphology: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.morphology,
      ...overrides.morphology,
    },
    stair: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.stair,
      ...overrides.stair,
    },
    motor: {
      ...DEFAULT_STAIR_DYNAMICS_CONFIG.motor,
      ...overrides.motor,
    },
  };

  const liftLinks = chooseLiftLinks(config);
  const targets = buildTargets(config, liftLinks);
  const world = new RAPIER.World({ x: 0, y: 0, z: -G });
  world.timestep = config.dt;
  world.numSolverIterations = 8;
  world.numInternalPgsIterations = 1;

  try {
    const env = createStairs(world, config);
    const assembly = createChain(world, config, liftLinks, targets.prepare);
    const linkLen = config.morphology.totalLength / config.morphology.n;
    const steps = Math.ceil(config.duration / config.dt);
    const frames: StairDynamicsFrame[] = [];
    const recordEverySteps =
      recordFps > 0 ? Math.max(1, Math.round(1 / (recordFps * config.dt))) : Infinity;

    let maxDemandTorqueNm = 0;
    let maxAppliedTorqueNm = 0;
    let maxDemandJoint = 0;
    let maxAppliedJoint = 0;
    let saturatedSteps = 0;
    let maxContactNormalForceN = 0;
    let maxContactTangentForceN = 0;
    let maxMuDemand = 0;
    let maxProbeMuDemand = 0;
    let frictionLimitedSamples = 0;
    let maxHeadTipZ = -Infinity;

    for (let step = 0; step < steps; step++) {
      const t = step * config.dt;
      if (step % recordEverySteps === 0) {
        frames.push(captureFrame(assembly.links, linkLen, t));
      }
      const torques = configureJointMotors(assembly, targetAt(t, targets), config.motor);
      applyTractionProbe(assembly, t, config);
      world.step();

      if (torques.maxDemand > maxDemandTorqueNm) {
        maxDemandTorqueNm = torques.maxDemand;
        maxDemandJoint = torques.demandJoint;
      }
      if (torques.maxApplied > maxAppliedTorqueNm) {
        maxAppliedTorqueNm = torques.maxApplied;
        maxAppliedJoint = torques.appliedJoint;
      }
      if (torques.saturated) saturatedSteps++;

      const contacts = readContacts(world, assembly.links, env, config.dt, config.friction);
      const probeMuDemand =
        t >= 2.8 && config.tractionProbeForceN > 0 && contacts.totalNormalForce > 1e-9
          ? config.tractionProbeForceN / contacts.totalNormalForce
          : 0;
      maxContactNormalForceN = Math.max(maxContactNormalForceN, contacts.maxNormalForce);
      maxContactTangentForceN = Math.max(maxContactTangentForceN, contacts.maxTangentForce);
      maxProbeMuDemand = Math.max(maxProbeMuDemand, probeMuDemand);
      maxMuDemand = Math.max(maxMuDemand, contacts.maxMuDemand, probeMuDemand);
      frictionLimitedSamples += contacts.frictionLimitedSamples;

      const [, tipZ] = headTip(assembly.links, linkLen);
      maxHeadTipZ = Math.max(maxHeadTipZ, tipZ);
    }

    const finalHeadTip = headTip(assembly.links, linkLen);
    frames.push(captureFrame(assembly.links, linkLen, config.duration));
    const success =
      finalHeadTip[0] >= 0.02 &&
      finalHeadTip[1] >= config.stair.rise + config.morphology.bodyThickness * 0.25;

    return {
      summary: {
        config,
        liftLinks,
        staticArcPeakNm: staticArcPeak(config, liftLinks),
        maxDemandTorqueNm,
        maxAppliedTorqueNm,
        maxDemandJoint,
        maxAppliedJoint,
        saturatedSteps,
        maxContactNormalForceN,
        maxContactTangentForceN,
        maxMuDemand,
        maxProbeMuDemand,
        frictionLimitedSamples,
        finalHeadTip,
        maxHeadTipZ,
        success,
      },
      frames,
    };
  } finally {
    world.free();
  }
}

export async function runStairDynamics(
  overrides: Partial<StairDynamicsConfig> = {},
): Promise<StairDynamicsSummary> {
  return (await simulateStairDynamics(overrides)).summary;
}

export async function runStairDynamicsReplay(
  overrides: Partial<StairDynamicsConfig> = {},
  fps = 30,
): Promise<StairDynamicsReplay> {
  return simulateStairDynamics(overrides, fps);
}

export async function runFrictionSweep(
  frictions: number[],
  overrides: Partial<StairDynamicsConfig> = {},
): Promise<StairDynamicsSummary[]> {
  const out: StairDynamicsSummary[] = [];
  for (const friction of frictions) {
    out.push(await runStairDynamics({ ...overrides, friction }));
  }
  return out;
}
