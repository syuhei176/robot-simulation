import { G } from './chain.ts';
import { terrainTopAt as courseTopAt } from './course.ts';
import { resolveCourse } from './stair-dynamics.ts';
import {
  DEFAULT_STAIR_REPLAY_CONFIG,
  createKinematicStairReplay,
} from './stair-kinematic-replay.ts';
import type {
  StairDynamicsConfig,
  StairDynamicsFrame,
  StairDynamicsReplay,
  StairFailureKind,
  StairFeasibilitySummary,
  StairFrameDiagnostics,
} from './stair-dynamics.ts';

const CONTACT_BAND_M = 0.018;
const PENETRATION_LIMIT_M = -0.006;
const HIGH_MU_LIMIT = 0.8;
const SLIP_SPEED_LIMIT_MPS = 0.04;

interface LinkKinematics {
  x: number;
  z: number;
  angle: number;
  vx: number;
  vz: number;
  ax: number;
  az: number;
  alpha: number;
  minClearanceM: number;
  supported: boolean;
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

function terrainTopAt(config: StairDynamicsConfig, x: number): number {
  return courseTopAt(resolveCourse(config), x);
}

function clearanceAtPoint(
  config: StairDynamicsConfig,
  halfThickness: number,
  x: number,
  z: number,
): number {
  return z - halfThickness - terrainTopAt(config, x);
}

function frameDt(frames: StairDynamicsFrame[], i: number): number {
  const prev = frames[Math.max(0, i - 1)];
  const next = frames[Math.min(frames.length - 1, i + 1)];
  return Math.max(1e-6, next.t - prev.t);
}

function linearVelocity(
  frames: StairDynamicsFrame[],
  i: number,
  linkIndex: number,
): [number, number] {
  const prev = frames[Math.max(0, i - 1)].links[linkIndex];
  const next = frames[Math.min(frames.length - 1, i + 1)].links[linkIndex];
  const dt = frameDt(frames, i);
  return [(next.x - prev.x) / dt, (next.z - prev.z) / dt];
}

function linearAcceleration(
  frames: StairDynamicsFrame[],
  i: number,
  linkIndex: number,
): [number, number] {
  if (i === 0 || i === frames.length - 1) return [0, 0];
  const prev = frames[i - 1].links[linkIndex];
  const cur = frames[i].links[linkIndex];
  const next = frames[i + 1].links[linkIndex];
  const dtPrev = Math.max(1e-6, frames[i].t - frames[i - 1].t);
  const dtNext = Math.max(1e-6, frames[i + 1].t - frames[i].t);
  const ax = (2 * ((next.x - cur.x) / dtNext - (cur.x - prev.x) / dtPrev)) / (dtPrev + dtNext);
  const az = (2 * ((next.z - cur.z) / dtNext - (cur.z - prev.z) / dtPrev)) / (dtPrev + dtNext);
  return [ax, az];
}

function angularAcceleration(frames: StairDynamicsFrame[], i: number, linkIndex: number): number {
  if (i === 0 || i === frames.length - 1) return 0;
  const prev = frames[i - 1].links[linkIndex];
  const cur = frames[i].links[linkIndex];
  const next = frames[i + 1].links[linkIndex];
  const dtPrev = Math.max(1e-6, frames[i].t - frames[i - 1].t);
  const dtNext = Math.max(1e-6, frames[i + 1].t - frames[i].t);
  const wPrev = normalizeAngle(cur.angle - prev.angle) / dtPrev;
  const wNext = normalizeAngle(next.angle - cur.angle) / dtNext;
  return (2 * (wNext - wPrev)) / (dtPrev + dtNext);
}

function linkClearance(
  config: StairDynamicsConfig,
  link: { x: number; z: number; angle: number },
  linkLen: number,
  halfThickness: number,
): number {
  let minClearance = Infinity;
  const cos = Math.cos(link.angle);
  const sin = Math.sin(link.angle);
  for (const offset of [-0.5, -0.25, 0, 0.25, 0.5]) {
    const x = link.x + offset * linkLen * cos;
    const z = link.z + offset * linkLen * sin;
    minClearance = Math.min(minClearance, clearanceAtPoint(config, halfThickness, x, z));
  }
  return minClearance;
}

function readLinkKinematics(
  config: StairDynamicsConfig,
  frames: StairDynamicsFrame[],
  frameIndex: number,
): LinkKinematics[] {
  const frame = frames[frameIndex];
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const halfThickness = config.morphology.bodyThickness / 2;

  return frame.links.map((link, linkIndex) => {
    const [vx, vz] = linearVelocity(frames, frameIndex, linkIndex);
    const [ax, az] = linearAcceleration(frames, frameIndex, linkIndex);
    const minClearanceM = linkClearance(config, link, linkLen, halfThickness);
    const centerBottom = link.z - halfThickness;
    const centerClearance = centerBottom - terrainTopAt(config, link.x);
    const supported =
      Math.abs(normalizeAngle(link.angle)) < 0.35 &&
      centerClearance >= PENETRATION_LIMIT_M &&
      centerClearance <= CONTACT_BAND_M;

    return {
      x: link.x,
      z: link.z,
      angle: link.angle,
      vx,
      vz,
      ax,
      az,
      alpha: angularAcceleration(frames, frameIndex, linkIndex),
      minClearanceM,
      supported,
    };
  });
}

function unsupportedSpan(links: LinkKinematics[], linkLen: number): number {
  let run = 0;
  let maxRun = 0;
  for (const link of links) {
    if (link.supported) {
      run = 0;
    } else {
      run += linkLen;
      maxRun = Math.max(maxRun, run);
    }
  }
  return maxRun;
}

function requiredTorque(
  config: StairDynamicsConfig,
  links: LinkKinematics[],
): {
  torqueNm: number;
  torqueJoint: number;
} {
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const linkMass = config.morphology.totalMass / config.morphology.n;
  const halfLen = linkLen / 2;
  const inertiaY = (linkMass * (linkLen * linkLen + config.morphology.bodyThickness ** 2)) / 12;
  let torqueNm = 0;
  let torqueJoint = 0;

  for (let joint = 0; joint < links.length - 1; joint++) {
    const parent = links[joint];
    const jointX = parent.x + halfLen * Math.cos(parent.angle);
    const jointZ = parent.z + halfLen * Math.sin(parent.angle);
    let moment = 0;
    let hasUnsupportedDistal = false;

    for (let i = joint + 1; i < links.length; i++) {
      const link = links[i];
      if (link.supported) break;
      hasUnsupportedDistal = true;
      const dx = link.x - jointX;
      const dz = link.z - jointZ;
      const fx = linkMass * link.ax;
      const fz = linkMass * (link.az + G);
      moment += dx * fz - dz * fx + inertiaY * link.alpha;
    }

    if (hasUnsupportedDistal && Math.abs(moment) > torqueNm) {
      torqueNm = Math.abs(moment);
      torqueJoint = joint;
    }
  }

  return { torqueNm, torqueJoint };
}

function supportXRange(
  links: LinkKinematics[],
  linkLen: number,
): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  const halfLen = linkLen / 2;
  for (const link of links) {
    if (!link.supported) continue;
    const dx = halfLen * Math.cos(link.angle);
    min = Math.min(min, link.x - Math.abs(dx), link.x + Math.abs(dx));
    max = Math.max(max, link.x - Math.abs(dx), link.x + Math.abs(dx));
  }
  return Number.isFinite(min) ? { min, max } : null;
}

function diagnoseFrame(
  config: StairDynamicsConfig,
  frames: StairDynamicsFrame[],
  frameIndex: number,
): StairFrameDiagnostics {
  const links = readLinkKinematics(config, frames, frameIndex);
  const linkLen = config.morphology.totalLength / config.morphology.n;
  const torque = requiredTorque(config, links);
  const supportedLinks = links.map((link, i) => (link.supported ? i : -1)).filter((i) => i >= 0);
  const supportCount = supportedLinks.length;
  const supportRange = supportXRange(links, linkLen);
  const maxUnsupportedSpanM = unsupportedSpan(links, linkLen);
  const minClearanceM = Math.min(...links.map((link) => link.minClearanceM));
  const comX = links.reduce((sum, link) => sum + link.x, 0) / links.length;
  const comZ = links.reduce((sum, link) => sum + link.z, 0) / links.length;
  const vComX = links.reduce((sum, link) => sum + link.vx, 0) / links.length;
  const vComZ = links.reduce((sum, link) => sum + link.vz, 0) / links.length;
  const aComX = links.reduce((sum, link) => sum + link.ax, 0) / links.length;
  const totalMass = config.morphology.totalMass;
  const normalForce = supportCount > 0 ? totalMass * G : 0;
  const liftPower = totalMass * G * Math.max(0, vComZ);
  const tractionFromLift = liftPower / Math.max(0.025, Math.abs(vComX));
  const tractionFromAccel = totalMass * Math.max(0, aComX);
  const requiredMu =
    normalForce > 1e-9 ? Math.max(0, tractionFromLift + tractionFromAccel) / normalForce : Infinity;
  const slipSpeedMps = Math.max(
    0,
    ...links.filter((link) => link.supported).map((link) => Math.abs(link.vx)),
  );
  const slipCandidates = links
    .map((link, i) => (link.supported && Math.abs(link.vx) > SLIP_SPEED_LIMIT_MPS ? i : -1))
    .filter((i) => i >= 0);
  const frictionExceeded = requiredMu > config.friction;
  const loadedSliding =
    slipCandidates.length > 0 && requiredMu > Math.min(config.friction * 0.75, 0.5);
  const slippingLinks = frictionExceeded || loadedSliding ? slipCandidates : [];
  const torqueRatio =
    config.motor.maxTorqueNm > 1e-9 ? torque.torqueNm / config.motor.maxTorqueNm : Infinity;
  const penetrationLinks = links
    .map((link, i) => (link.minClearanceM < PENETRATION_LIMIT_M ? i : -1))
    .filter((i) => i >= 0);
  const supportMargin = linkLen * 0.35;
  const supportOutside =
    supportRange === null ||
    comX < supportRange.min - supportMargin ||
    comX > supportRange.max + supportMargin;
  const longUnsupported =
    maxUnsupportedSpanM > Math.max(0.42, config.morphology.totalLength * 0.55);
  const fallingLinks =
    penetrationLinks.length > 0 || supportCount === 0 || supportOutside || longUnsupported
      ? links
          .map((link, i) => (!link.supported || penetrationLinks.includes(i) ? i : -1))
          .filter((i) => i >= 0)
      : [];

  const failureKinds: StairFailureKind[] = [];
  if (torqueRatio > 1) failureKinds.push('torque');
  if (fallingLinks.length > 0) failureKinds.push('fall');
  if (frictionExceeded || loadedSliding) failureKinds.push('slip');
  if (requiredMu > HIGH_MU_LIMIT) failureKinds.push('mu');

  return {
    torqueNm: torque.torqueNm,
    torqueLimitNm: config.motor.maxTorqueNm,
    torqueRatio,
    torqueJoint: torque.torqueJoint,
    requiredMu,
    friction: config.friction,
    slipSpeedMps,
    minClearanceM,
    maxUnsupportedSpanM,
    supportCount,
    com: [comX, comZ],
    supportedLinks,
    slippingLinks,
    fallingLinks,
    failureKinds,
  };
}

function summarizeDiagnostics(frames: StairDynamicsFrame[]): StairFeasibilitySummary {
  const failureCounts: Record<StairFailureKind, number> = {
    torque: 0,
    fall: 0,
    slip: 0,
    mu: 0,
  };
  let firstFailureTime: number | null = null;
  let maxTorqueNm = 0;
  let maxTorqueRatio = 0;
  let maxRequiredMu = 0;
  let maxSlipSpeedMps = 0;
  let minClearanceM = Infinity;
  let maxUnsupportedSpanM = 0;
  let minSupportCount = Infinity;
  let worstTorqueTime = 0;
  let worstMuTime = 0;
  let worstSlipTime = 0;
  let worstClearanceTime = 0;

  for (const frame of frames) {
    const diagnostics = frame.diagnostics;
    if (!diagnostics) continue;

    if (diagnostics.failureKinds.length > 0 && firstFailureTime === null) {
      firstFailureTime = frame.t;
    }
    for (const kind of diagnostics.failureKinds) {
      failureCounts[kind]++;
    }
    const torqueNm = diagnostics.motorDemandNm ?? diagnostics.torqueNm;
    const torqueRatio = diagnostics.motorTorqueRatio ?? diagnostics.torqueRatio;
    if (torqueNm > maxTorqueNm) {
      maxTorqueNm = torqueNm;
      worstTorqueTime = frame.t;
    }
    if (torqueRatio > maxTorqueRatio) {
      maxTorqueRatio = torqueRatio;
    }
    if (diagnostics.requiredMu > maxRequiredMu) {
      maxRequiredMu = diagnostics.requiredMu;
      worstMuTime = frame.t;
    }
    if (diagnostics.slipSpeedMps > maxSlipSpeedMps) {
      maxSlipSpeedMps = diagnostics.slipSpeedMps;
      worstSlipTime = frame.t;
    }
    if (diagnostics.minClearanceM < minClearanceM) {
      minClearanceM = diagnostics.minClearanceM;
      worstClearanceTime = frame.t;
    }
    maxUnsupportedSpanM = Math.max(maxUnsupportedSpanM, diagnostics.maxUnsupportedSpanM);
    minSupportCount = Math.min(minSupportCount, diagnostics.supportCount);
  }

  return {
    passed: firstFailureTime === null,
    firstFailureTime,
    failureCounts,
    maxTorqueNm,
    maxTorqueRatio,
    maxRequiredMu,
    maxSlipSpeedMps,
    minClearanceM,
    maxUnsupportedSpanM,
    minSupportCount: Number.isFinite(minSupportCount) ? minSupportCount : 0,
    worstTorqueTime,
    worstMuTime,
    worstSlipTime,
    worstClearanceTime,
  };
}

export function addStairFeasibilityDiagnostics(replay: StairDynamicsReplay): StairDynamicsReplay {
  const frames = replay.frames.map((frame, i, allFrames) => ({
    ...frame,
    diagnostics: withTelemetry(diagnoseFrame(replay.summary.config, allFrames, i), frame),
  }));
  const feasibility = summarizeDiagnostics(frames);

  return {
    summary: {
      ...replay.summary,
      maxDemandTorqueNm: feasibility.maxTorqueNm,
      maxAppliedTorqueNm: Math.min(
        feasibility.maxTorqueNm,
        replay.summary.config.motor.maxTorqueNm,
      ),
      saturatedSteps: feasibility.failureCounts.torque,
      maxMuDemand: feasibility.maxRequiredMu,
      frictionLimitedSamples: feasibility.failureCounts.slip,
      success: replay.summary.success && feasibility.passed,
      feasibility,
    },
    frames,
  };
}

function withTelemetry(
  diagnostics: StairFrameDiagnostics,
  frame: StairDynamicsFrame,
): StairFrameDiagnostics {
  if (!frame.telemetry) return diagnostics;
  const failureKinds = new Set(diagnostics.failureKinds);
  if (frame.telemetry.motorSaturated) failureKinds.add('torque');

  return {
    ...diagnostics,
    motorDemandNm: frame.telemetry.motorDemandNm,
    motorAppliedNm: frame.telemetry.motorAppliedNm,
    motorTorqueRatio: frame.telemetry.motorTorqueRatio,
    motorJoint: frame.telemetry.motorJoint,
    motorSaturated: frame.telemetry.motorSaturated,
    driveForceN: frame.telemetry.driveForceN,
    failureKinds: [...failureKinds],
  };
}

export function createDiagnosedStairReplay(
  overrides: Partial<StairDynamicsConfig> = {},
  fps = 45,
): StairDynamicsReplay {
  return addStairFeasibilityDiagnostics(createKinematicStairReplay(overrides, fps));
}

export function sampleStairDiagnostics(
  replay: StairDynamicsReplay | null,
  time: number,
): StairFrameDiagnostics | null {
  if (!replay || replay.frames.length === 0) return null;
  const duration = replay.summary.config.duration || DEFAULT_STAIR_REPLAY_CONFIG.duration;
  const t = ((time % duration) + duration) % duration;
  const found = replay.frames.findIndex((frame) => frame.t >= t);
  if (found === -1) return replay.frames[replay.frames.length - 1].diagnostics ?? null;
  const index = clamp(found, 0, replay.frames.length - 1);
  return replay.frames[index]?.diagnostics ?? null;
}
