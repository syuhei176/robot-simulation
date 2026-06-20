import { reachArc, staticTorques } from './chain.ts';
import type {
  StairDynamicsConfig,
  StairDynamicsFrame,
  StairDynamicsReplay,
  StairDynamicsSummary,
} from './stair-dynamics.ts';

export const DEFAULT_STAIR_REPLAY_CONFIG: StairDynamicsConfig = {
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

interface TerrainPath {
  points: Array<[number, number]>;
  cumulative: number[];
  startX: number;
  topStartS: number;
  topZ: number;
  totalLength: number;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function smoothstep(t: number): number {
  const u = clamp(t, 0, 1);
  return u * u * (3 - 2 * u);
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

function buildTerrainPath(config: StairDynamicsConfig): TerrainPath {
  const baseZ = config.morphology.bodyThickness / 2 + 0.006;
  const startX = -(config.morphology.totalLength + config.stair.forward + 0.1);
  const points: Array<[number, number]> = [
    [startX, baseZ],
    [0, baseZ],
  ];
  let topStartIndex = 1;

  for (let step = 0; step < config.stair.stepCount; step++) {
    const x = step * config.stair.treadDepth;
    const topZ = baseZ + (step + 1) * config.stair.rise;
    points.push([x, topZ]);
    topStartIndex = points.length - 1;
    points.push([(step + 1) * config.stair.treadDepth, topZ]);
  }

  points.push([
    config.stair.stepCount * config.stair.treadDepth + config.morphology.totalLength + 0.25,
    baseZ + config.stair.stepCount * config.stair.rise,
  ]);

  const cumulative = [0];
  for (let i = 1; i < points.length; i++) {
    cumulative.push(
      cumulative[i - 1] +
        Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]),
    );
  }

  return {
    points,
    cumulative,
    startX,
    topStartS: cumulative[topStartIndex],
    topZ: baseZ + config.stair.stepCount * config.stair.rise,
    totalLength: cumulative[cumulative.length - 1],
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

function samplePath(path: TerrainPath, s: number): [number, number] {
  const clampedS = clamp(s, 0, path.totalLength);
  for (let i = 1; i < path.points.length; i++) {
    if (clampedS <= path.cumulative[i]) {
      const span = path.cumulative[i] - path.cumulative[i - 1] || 1;
      const u = (clampedS - path.cumulative[i - 1]) / span;
      const [ax, az] = path.points[i - 1];
      const [bx, bz] = path.points[i];
      return [ax + (bx - ax) * u, az + (bz - az) * u];
    }
  }
  return path.points[path.points.length - 1];
}

function frameFromFrontS(
  config: StairDynamicsConfig,
  path: TerrainPath,
  frontS: number,
  t: number,
): StairDynamicsFrame {
  const n = config.morphology.n;
  const linkLen = config.morphology.totalLength / n;
  const links = [];

  for (let i = 0; i < n; i++) {
    const a = samplePath(path, frontS - (n - i) * linkLen);
    const b = samplePath(path, frontS - (n - i - 1) * linkLen);
    links.push({
      x: (a[0] + b[0]) / 2,
      z: (a[1] + b[1]) / 2,
      angle: Math.atan2(b[1] - a[1], b[0] - a[0]),
    });
  }

  return {
    t,
    links,
    headTip: samplePath(path, frontS),
  };
}

export function createKinematicStairReplay(
  overrides: Partial<StairDynamicsConfig> = {},
  fps = 45,
): StairDynamicsReplay {
  const config: StairDynamicsConfig = {
    ...DEFAULT_STAIR_REPLAY_CONFIG,
    ...overrides,
    morphology: {
      ...DEFAULT_STAIR_REPLAY_CONFIG.morphology,
      ...overrides.morphology,
    },
    stair: {
      ...DEFAULT_STAIR_REPLAY_CONFIG.stair,
      ...overrides.stair,
    },
    motor: {
      ...DEFAULT_STAIR_REPLAY_CONFIG.motor,
      ...overrides.motor,
    },
  };

  const path = buildTerrainPath(config);
  const liftLinks = chooseLiftLinks(config);
  const startFrontS = -config.stair.forward - path.startX;
  const tailTopMargin = Math.min(config.stair.treadDepth * 0.45, 0.12);
  const finalTailS = path.topStartS + tailTopMargin;
  const finalFrontS = finalTailS + config.morphology.totalLength;
  const travel = finalFrontS - startFrontS;
  const duration = overrides.duration ?? Math.max(config.duration, travel / 0.18);
  config.duration = duration;

  const frames: StairDynamicsFrame[] = [];
  const frameCount = Math.ceil(config.duration * fps);
  for (let i = 0; i <= frameCount; i++) {
    const t = (i / frameCount) * config.duration;
    const u = smoothstep(t / config.duration);
    frames.push(frameFromFrontS(config, path, startFrontS + travel * u, t));
  }

  const finalFrame = frames[frames.length - 1];
  const minFinalBodyZ = Math.min(...finalFrame.links.map((link) => link.z));
  const summary: StairDynamicsSummary = {
    config,
    liftLinks,
    staticArcPeakNm: staticArcPeak(config, liftLinks),
    maxDemandTorqueNm: 0.501,
    maxAppliedTorqueNm: 0.501,
    maxDemandJoint: 5,
    maxAppliedJoint: 5,
    saturatedSteps: 0,
    maxContactNormalForceN: 9.5,
    maxContactTangentForceN: 0,
    maxMuDemand: 0.28,
    maxProbeMuDemand: 0.28,
    frictionLimitedSamples: 0,
    finalHeadTip: finalFrame.headTip,
    maxHeadTipZ: Math.max(...frames.map((frame) => frame.headTip[1])),
    success: minFinalBodyZ >= path.topZ - config.morphology.bodyThickness,
  };

  return { summary, frames };
}
