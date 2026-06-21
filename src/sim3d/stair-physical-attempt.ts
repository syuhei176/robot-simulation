import RAPIER, {
  type Collider,
  type RevoluteImpulseJoint,
  type RigidBody,
  type TempContactManifold,
  type World,
} from '@dimforge/rapier3d-compat';

import { G } from './chain.ts';
import { buildCourseColliders, terrainTopAt } from './course.ts';
import { addStairFeasibilityDiagnostics } from './stair-feasibility.ts';
import { createKinematicStairReplay } from './stair-kinematic-replay.ts';
import { resolveCourse } from './stair-dynamics.ts';
import type {
  JointMotorConfig,
  StairDynamicsConfig,
  StairDynamicsFrame,
  StairDynamicsReplay,
  StairDynamicsSummary,
  StairFrameTelemetry,
} from './stair-dynamics.ts';

interface ChainBody {
  body: RigidBody;
  collider: Collider;
}

interface ChainAssembly {
  links: ChainBody[];
  joints: RevoluteImpulseJoint[];
}

interface ContactSnapshot {
  maxNormalForceN: number;
  maxTangentForceN: number;
  maxMuDemand: number;
  frictionLimitedSamples: number;
}

const DEFAULT_ATTEMPT_FPS = 45;
const TARGET_FPS = 90;
const DRIVE_POSITION_GAIN = 18;
const DRIVE_VELOCITY_GAIN = 5;
const DRIVE_FORCE_SAFETY = 0.9;

let rapierReady: Promise<void> | null = null;

function ensureRapierReady(): Promise<void> {
  rapierReady ??= RAPIER.init();
  return rapierReady;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
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

function createChainFromFrame(
  world: World,
  config: StairDynamicsConfig,
  frame: StairDynamicsFrame,
): ChainAssembly {
  const n = config.morphology.n;
  const linkLen = config.morphology.totalLength / n;
  const linkMass = config.morphology.totalMass / n;
  const halfWidth = config.morphology.bodyWidth / 2;
  const halfThickness = config.morphology.bodyThickness / 2;
  const links: ChainBody[] = [];
  const joints: RevoluteImpulseJoint[] = [];

  for (const link of frame.links) {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(link.x, 0, link.z)
        .setRotation(quatFromSagittalAngle(link.angle))
        .setLinearDamping(0.32)
        .setAngularDamping(0.08)
        .setCanSleep(false),
    );

    const collider = world.createCollider(
      RAPIER.ColliderDesc.cuboid(linkLen / 2, halfWidth, halfThickness)
        .setMass(linkMass)
        .setFriction(config.friction)
        .setRestitution(0),
      body,
    );

    body.setAdditionalSolverIterations(4);
    links.push({ body, collider });
  }

  for (let i = 0; i < linkCount(config) - 1; i++) {
    const joint = world.createImpulseJoint(
      RAPIER.JointData.revolute(
        { x: linkLen / 2, y: 0, z: 0 },
        { x: -linkLen / 2, y: 0, z: 0 },
        { x: 0, y: 1, z: 0 },
      ),
      links[i].body,
      links[i + 1].body,
      true,
    ) as RevoluteImpulseJoint;
    joint.setContactsEnabled(false);
    joint.setLimits(-2.35, 2.35);
    joints.push(joint);
  }

  return { links, joints };
}

function linkCount(config: StairDynamicsConfig): number {
  return config.morphology.n;
}

function sampleReplayFrame(replay: StairDynamicsReplay, t: number): StairDynamicsFrame {
  const frames = replay.frames;
  if (t <= frames[0].t) return frames[0];
  if (t >= frames[frames.length - 1].t) return frames[frames.length - 1];

  let hi = 1;
  while (hi < frames.length && frames[hi].t < t) hi++;
  const a = frames[hi - 1];
  const b = frames[hi];
  const u = clamp((t - a.t) / Math.max(1e-9, b.t - a.t), 0, 1);

  return {
    t,
    links: a.links.map((link, i) => {
      const next = b.links[i] ?? link;
      return {
        x: link.x + (next.x - link.x) * u,
        z: link.z + (next.z - link.z) * u,
        angle: link.angle + normalizeAngle(next.angle - link.angle) * u,
      };
    }),
    headTip: [
      a.headTip[0] + (b.headTip[0] - a.headTip[0]) * u,
      a.headTip[1] + (b.headTip[1] - a.headTip[1]) * u,
    ],
  };
}

function frameCom(frame: StairDynamicsFrame): [number, number] {
  let x = 0;
  let z = 0;
  for (const link of frame.links) {
    x += link.x;
    z += link.z;
  }
  return [x / frame.links.length, z / frame.links.length];
}

function assemblyCom(assembly: ChainAssembly): [number, number] {
  let x = 0;
  let z = 0;
  for (const link of assembly.links) {
    const p = link.body.translation();
    x += p.x;
    z += p.z;
  }
  return [x / assembly.links.length, z / assembly.links.length];
}

function assemblyComVelocity(assembly: ChainAssembly): [number, number] {
  let x = 0;
  let z = 0;
  for (const link of assembly.links) {
    const v = link.body.linvel();
    x += v.x;
    z += v.z;
  }
  return [x / assembly.links.length, z / assembly.links.length];
}

function supportedLinkIndices(config: StairDynamicsConfig, assembly: ChainAssembly): number[] {
  const halfThickness = config.morphology.bodyThickness / 2;
  const course = resolveCourse(config);
  const supported: number[] = [];
  for (let i = 0; i < assembly.links.length; i++) {
    const body = assembly.links[i].body;
    const p = body.translation();
    const centerClearance = p.z - halfThickness - terrainTopAt(course, p.x);
    const flatEnough = Math.abs(normalizeAngle(sagittalAngle(body))) < 0.75;
    if (flatEnough && centerClearance > -0.025 && centerClearance < 0.045) {
      supported.push(i);
    }
  }
  return supported;
}

function applyJointTorques(
  assembly: ChainAssembly,
  target: StairDynamicsFrame,
  motor: JointMotorConfig,
): Omit<StairFrameTelemetry, 'driveForceN'> {
  let motorDemandNm = 0;
  let motorAppliedNm = 0;
  let motorJoint = 0;
  let motorSaturated = false;

  for (let j = 0; j < assembly.links.length - 1; j++) {
    const parent = assembly.links[j].body;
    const child = assembly.links[j + 1].body;
    const current = normalizeAngle(sagittalAngle(child) - sagittalAngle(parent));
    const targetRel = normalizeAngle(target.links[j + 1].angle - target.links[j].angle);
    const relVel = sagittalAngularVelocity(child) - sagittalAngularVelocity(parent);
    const demand = motor.stiffness * normalizeAngle(targetRel - current) - motor.damping * relVel;
    const applied = clamp(demand, -motor.maxTorqueNm, motor.maxTorqueNm);

    parent.addTorque({ x: 0, y: applied, z: 0 }, true);
    child.addTorque({ x: 0, y: -applied, z: 0 }, true);

    if (Math.abs(demand) > motorDemandNm) {
      motorDemandNm = Math.abs(demand);
      motorJoint = j;
    }
    motorAppliedNm = Math.max(motorAppliedNm, Math.abs(applied));
    motorSaturated ||= Math.abs(demand - applied) > 1e-9;
  }

  return {
    motorDemandNm,
    motorAppliedNm,
    motorTorqueRatio: motor.maxTorqueNm > 1e-9 ? motorDemandNm / motor.maxTorqueNm : Infinity,
    motorJoint,
    motorSaturated,
  };
}

function applyTractionDrive(
  assembly: ChainAssembly,
  config: StairDynamicsConfig,
  target: StairDynamicsFrame,
  targetNext: StairDynamicsFrame,
): number {
  const supported = supportedLinkIndices(config, assembly);
  if (supported.length === 0) return 0;

  const [targetX] = frameCom(target);
  const [targetNextX] = frameCom(targetNext);
  const [actualX] = assemblyCom(assembly);
  const [actualVx] = assemblyComVelocity(assembly);
  const targetVx = (targetNextX - targetX) / config.dt;
  const demand =
    DRIVE_POSITION_GAIN * (targetX - actualX) + DRIVE_VELOCITY_GAIN * (targetVx - actualVx);
  const maxDrive = config.friction * config.morphology.totalMass * G * DRIVE_FORCE_SAFETY;
  const driveForce = clamp(demand, -maxDrive, maxDrive);
  const perLinkForce = driveForce / supported.length;

  for (const i of supported) {
    assembly.links[i].body.addForce({ x: perLinkForce, y: 0, z: 0 }, true);
  }

  return Math.abs(driveForce);
}

function headTip(chain: ChainBody[], linkLen: number): [number, number] {
  const head = chain[chain.length - 1].body;
  const theta = sagittalAngle(head);
  const p = head.translation();
  return [p.x + (linkLen / 2) * Math.cos(theta), p.z + (linkLen / 2) * Math.sin(theta)];
}

function captureFrame(
  assembly: ChainAssembly,
  config: StairDynamicsConfig,
  t: number,
  telemetry?: StairFrameTelemetry,
): StairDynamicsFrame {
  const linkLen = config.morphology.totalLength / config.morphology.n;
  return {
    t,
    links: assembly.links.map(({ body }) => {
      const p = body.translation();
      return { x: p.x, z: p.z, angle: sagittalAngle(body) };
    }),
    headTip: headTip(assembly.links, linkLen),
    telemetry,
  };
}

function readContacts(
  world: World,
  chain: ChainBody[],
  env: Collider[],
  dt: number,
  friction: number,
): ContactSnapshot {
  let maxNormalForceN = 0;
  let maxTangentForceN = 0;
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
      maxNormalForceN = Math.max(maxNormalForceN, normalForce);
      maxTangentForceN = Math.max(maxTangentForceN, tangentForce);
      maxMuDemand = Math.max(maxMuDemand, muDemand);
      if (normalImpulse > 1e-9 && muDemand > friction * 0.9) frictionLimitedSamples++;
    }
  };

  for (const link of chain) {
    for (const obstacle of env) {
      world.narrowPhase.contactPair(link.collider.handle, obstacle.handle, readManifold);
    }
  }

  return { maxNormalForceN, maxTangentForceN, maxMuDemand, frictionLimitedSamples };
}

function physicalSuccess(config: StairDynamicsConfig, finalFrame: StairDynamicsFrame): boolean {
  const course = resolveCourse(config);
  const topZ = Math.max(...course.profile.map(([, z]) => z)); // 目的プラトーの高さ
  const goalX = course.goalX;
  const minZ = Math.min(...finalFrame.links.map((link) => link.z));
  const minX = Math.min(...finalFrame.links.map((link) => link.x));
  const maxX = Math.max(...finalFrame.links.map((link) => link.x));
  return (
    minZ >= topZ - config.morphology.bodyThickness * 0.2 &&
    minX >= goalX - 0.15 &&
    maxX > goalX + 0.35
  );
}

export async function runPhysicalStairAttemptReplay(
  overrides: Partial<StairDynamicsConfig> = {},
  fps = DEFAULT_ATTEMPT_FPS,
): Promise<StairDynamicsReplay> {
  await ensureRapierReady();

  const targetReplay = createKinematicStairReplay(overrides, TARGET_FPS);
  const config = targetReplay.summary.config;
  const world = new RAPIER.World({ x: 0, y: 0, z: -G });
  world.timestep = config.dt;
  world.numSolverIterations = 12;
  world.numInternalPgsIterations = 2;

  try {
    const env = buildCourseColliders(world, resolveCourse(config), config.friction);
    const assembly = createChainFromFrame(world, config, targetReplay.frames[0]);
    const frames: StairDynamicsFrame[] = [];
    const steps = Math.ceil(config.duration / config.dt);
    const recordEverySteps = Math.max(1, Math.round(1 / (fps * config.dt)));
    let lastTelemetry: StairFrameTelemetry = {
      motorDemandNm: 0,
      motorAppliedNm: 0,
      motorTorqueRatio: 0,
      motorJoint: 0,
      motorSaturated: false,
      driveForceN: 0,
    };

    let maxDemandTorqueNm = 0;
    let maxAppliedTorqueNm = 0;
    let maxDemandJoint = 0;
    let maxAppliedJoint = 0;
    let saturatedSteps = 0;
    let maxContactNormalForceN = 0;
    let maxContactTangentForceN = 0;
    let maxMuDemand = 0;
    let frictionLimitedSamples = 0;
    let maxHeadTipZ = -Infinity;

    frames.push(captureFrame(assembly, config, 0, lastTelemetry));

    for (let step = 0; step < steps; step++) {
      const t = step * config.dt;
      const target = sampleReplayFrame(targetReplay, t);
      const targetNext = sampleReplayFrame(targetReplay, Math.min(config.duration, t + config.dt));

      for (const link of assembly.links) {
        link.body.resetForces(true);
        link.body.resetTorques(true);
      }

      const motor = applyJointTorques(assembly, target, config.motor);
      const driveForceN = applyTractionDrive(assembly, config, target, targetNext);
      lastTelemetry = { ...motor, driveForceN };
      world.step();

      if (lastTelemetry.motorDemandNm > maxDemandTorqueNm) {
        maxDemandTorqueNm = lastTelemetry.motorDemandNm;
        maxDemandJoint = lastTelemetry.motorJoint;
      }
      if (lastTelemetry.motorAppliedNm > maxAppliedTorqueNm) {
        maxAppliedTorqueNm = lastTelemetry.motorAppliedNm;
        maxAppliedJoint = lastTelemetry.motorJoint;
      }
      if (lastTelemetry.motorSaturated) saturatedSteps++;

      const contacts = readContacts(world, assembly.links, env, config.dt, config.friction);
      maxContactNormalForceN = Math.max(maxContactNormalForceN, contacts.maxNormalForceN);
      maxContactTangentForceN = Math.max(maxContactTangentForceN, contacts.maxTangentForceN);
      maxMuDemand = Math.max(maxMuDemand, contacts.maxMuDemand);
      frictionLimitedSamples += contacts.frictionLimitedSamples;

      const [, tipZ] = headTip(assembly.links, config.morphology.totalLength / config.morphology.n);
      maxHeadTipZ = Math.max(maxHeadTipZ, tipZ);

      if ((step + 1) % recordEverySteps === 0) {
        frames.push(
          captureFrame(assembly, config, Math.min(config.duration, t + config.dt), lastTelemetry),
        );
      }
    }

    const finalFrame = captureFrame(assembly, config, config.duration, lastTelemetry);
    frames.push(finalFrame);
    const finalHeadTip = finalFrame.headTip;
    const summary: StairDynamicsSummary = {
      config,
      liftLinks: targetReplay.summary.liftLinks,
      staticArcPeakNm: targetReplay.summary.staticArcPeakNm,
      maxDemandTorqueNm,
      maxDemandTimeS: 0,
      maxAppliedTorqueNm,
      maxDemandJoint,
      maxAppliedJoint,
      saturatedSteps,
      maxContactNormalForceN,
      maxContactTangentForceN,
      maxMuDemand,
      maxProbeMuDemand: 0,
      frictionLimitedSamples,
      finalHeadTip,
      maxHeadTipZ,
      success: physicalSuccess(config, finalFrame),
    };

    return addStairFeasibilityDiagnostics({ summary, frames });
  } finally {
    world.free();
  }
}
