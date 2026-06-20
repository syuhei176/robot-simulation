import { StairDynamicsView } from './render/StairDynamicsView.ts';
import { DEFAULT_STAIR_REPLAY_CONFIG } from './sim3d/stair-kinematic-replay.ts';
import { sampleStairDiagnostics } from './sim3d/stair-feasibility.ts';
import { runPhysicalStairAttemptReplay } from './sim3d/stair-physical-attempt.ts';
import type { StairDynamicsReplay, StairFailureKind } from './sim3d/stair-dynamics.ts';
import { getServo, SELECTABLE_SERVO_IDS } from './sim3d/servos.ts';
import {
  runQuadrupedGait,
  DEFAULT_QUAD_DYN_CONFIG,
  type QuadDynReplay,
} from './sim3d/quadruped-dynamics.ts';

const app = document.getElementById('app');
if (!app) throw new Error('#app is missing');

const view = new StairDynamicsView(app);
view.setVisible(true);

let replay: StairDynamicsReplay | null = null;
let playing = true;
let playbackTime = 0;
let speed = 1;
let mechanism: 'snake' | 'quad' = 'snake';
let motorId = 'sts3215';
let torqueCapNm = getServo(motorId).stallNm;
let friction = 0.6;
// 四足の総質量 [kg]（スライダーで可変）。trunk:leg 質量比は default を保ってスケールする。
const QUAD_BASE_TRUNK = DEFAULT_QUAD_DYN_CONFIG.trunk.mass;
const QUAD_BASE_SEG = DEFAULT_QUAD_DYN_CONFIG.leg.segMass;
const QUAD_BASE_TOTAL = QUAD_BASE_TRUNK + 8 * QUAD_BASE_SEG;
let quadMass = QUAD_BASE_TOTAL;
let quadReplay: QuadDynReplay | null = null;
let quadSeq = 0;
let lastTimestamp: number | null = null;
let loadSeq = 0;
const initialTimeParam = Number(new URLSearchParams(window.location.search).get('t'));
let pendingInitialTime: number | null = Number.isFinite(initialTimeParam) ? initialTimeParam : null;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
}

const btnPlay = el<HTMLButtonElement>('btn-play');
const btnReset = el<HTMLButtonElement>('btn-reset');
const progress = el<HTMLInputElement>('s-time');
const speedInput = el<HTMLInputElement>('s-speed');
const torqueCapInput = el<HTMLInputElement>('s-torque-cap');
const frictionInput = el<HTMLInputElement>('s-friction');
const massInput = el<HTMLInputElement>('s-mass');
const massValue = el<HTMLSpanElement>('v-mass');
const timeValue = el<HTMLSpanElement>('v-time');
const speedValue = el<HTMLSpanElement>('v-speed');
const torqueCapValue = el<HTMLSpanElement>('v-torque-cap');
const frictionValue = el<HTMLSpanElement>('v-friction');
const stStatus = el<HTMLElement>('st-status');
const stTauNow = el<HTMLElement>('st-tau-now');
const stTau = el<HTMLElement>('st-tau');
const stMuNow = el<HTMLElement>('st-mu-now');
const stMu = el<HTMLElement>('st-mu');
const stSlip = el<HTMLElement>('st-slip');
const stClearance = el<HTMLElement>('st-clearance');
const stSupports = el<HTMLElement>('st-supports');
const stFirstFailure = el<HTMLElement>('st-first-failure');
const stHead = el<HTMLElement>('st-head');
const selMech = el<HTMLSelectElement>('sel-mech');
const selMotor = el<HTMLSelectElement>('sel-motor');
const title = el<HTMLElement>('title');
const subtitle = el<HTMLElement>('subtitle');
const snakeControls = el<HTMLDivElement>('snake-controls');
const snakeFriction = el<HTMLDivElement>('snake-friction');
const quadMassControl = el<HTMLDivElement>('quad-mass');
const snakeStats = el<HTMLDivElement>('snake-stats');
const quadStats = el<HTMLDivElement>('quad-stats');
const qMotor = el<HTMLElement>('q-motor');
const qForward = el<HTMLElement>('q-forward');
const qFall = el<HTMLElement>('q-fall');
const qCap = el<HTMLElement>('q-cap');
const qResult = el<HTMLElement>('q-result');

function fmtFinite(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '∞';
  return value.toFixed(digits);
}

function setMetricClass(node: HTMLElement, kind: 'good' | 'warn' | 'bad' | null): void {
  node.classList.remove('good', 'warn', 'bad');
  if (kind) node.classList.add(kind);
}

function failureLabel(kind: StairFailureKind): string {
  if (kind === 'torque') return 'トルク';
  if (kind === 'fall') return '落下/干渉';
  if (kind === 'slip') return '滑り';
  return '高μ';
}

function formatFailureKinds(kinds: StairFailureKind[]): string {
  return kinds.length > 0 ? kinds.map(failureLabel).join(' / ') : 'pass';
}

function setPlaying(next: boolean): void {
  playing = next;
  btnPlay.textContent = next ? '⏸ 一時停止' : '▶ 再生';
  btnPlay.classList.toggle('active', next);
}

function updateControls(): void {
  const duration = (mechanism === 'snake' ? view.duration() : quadDuration()) || 4;
  const t = duration > 0 ? playbackTime % duration : 0;
  progress.max = String(duration);
  progress.value = t.toFixed(2);
  timeValue.textContent = `${t.toFixed(2)}s`;
  speedValue.textContent = `${speed.toFixed(2)}x`;
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  frictionValue.textContent = friction.toFixed(2);
  if (mechanism === 'snake') updateDiagnostics();
}

function updateDiagnostics(): void {
  const diagnostics = sampleStairDiagnostics(replay, playbackTime);
  const summary = replay?.summary.feasibility;
  if (!replay || !diagnostics || !summary) return;
  const currentTorque = diagnostics.motorDemandNm ?? diagnostics.torqueNm;
  const currentTorqueRatio = diagnostics.motorTorqueRatio ?? diagnostics.torqueRatio;

  stStatus.textContent = formatFailureKinds(diagnostics.failureKinds);
  setMetricClass(stStatus, diagnostics.failureKinds.length > 0 ? 'bad' : 'good');

  stTauNow.textContent = `${fmtFinite(currentTorque, 2)} / ${fmtFinite(
    diagnostics.torqueLimitNm,
    2,
  )} (${fmtFinite(currentTorqueRatio, 1)}x)`;
  setMetricClass(
    stTauNow,
    currentTorqueRatio > 1 ? 'bad' : currentTorqueRatio > 0.8 ? 'warn' : null,
  );

  stTau.textContent = `${fmtFinite(summary.maxTorqueNm, 2)} N·m (${fmtFinite(
    summary.maxTorqueRatio,
    1,
  )}x)`;
  setMetricClass(
    stTau,
    summary.maxTorqueRatio > 1 ? 'bad' : summary.maxTorqueRatio > 0.8 ? 'warn' : null,
  );

  stMuNow.textContent = `${fmtFinite(diagnostics.requiredMu, 2)} / ${fmtFinite(
    diagnostics.friction,
    2,
  )}`;
  setMetricClass(
    stMuNow,
    diagnostics.requiredMu > diagnostics.friction
      ? 'bad'
      : diagnostics.requiredMu > 0.8
        ? 'warn'
        : null,
  );

  stMu.textContent = `${fmtFinite(summary.maxRequiredMu, 2)} @ ${summary.worstMuTime.toFixed(2)}s`;
  setMetricClass(stMu, summary.maxRequiredMu > friction ? 'bad' : null);

  stSlip.textContent = `${(diagnostics.slipSpeedMps * 100).toFixed(1)} cm/s`;
  setMetricClass(stSlip, diagnostics.failureKinds.includes('slip') ? 'bad' : null);

  stClearance.textContent = `${(diagnostics.minClearanceM * 1000).toFixed(1)} mm`;
  setMetricClass(stClearance, diagnostics.minClearanceM < -0.006 ? 'bad' : null);

  stSupports.textContent = `${diagnostics.supportCount}/${replay.summary.config.morphology.n}`;
  setMetricClass(stSupports, diagnostics.failureKinds.includes('fall') ? 'bad' : null);

  stFirstFailure.textContent =
    summary.firstFailureTime === null ? 'なし' : `${summary.firstFailureTime.toFixed(2)}s`;
  setMetricClass(stFirstFailure, summary.firstFailureTime === null ? 'good' : 'bad');
}

async function loadReplay(): Promise<void> {
  const seq = ++loadSeq;
  stStatus.textContent = '物理計算中';
  setMetricClass(stStatus, null);
  const nextReplay = await runPhysicalStairAttemptReplay(
    {
      friction,
      motor: {
        ...DEFAULT_STAIR_REPLAY_CONFIG.motor,
        maxTorqueNm: torqueCapNm,
      },
    },
    45,
  );
  // 競合ガード: 新しいロードが始まった or その間に四足へ切り替わったら破棄
  if (seq !== loadSeq || mechanism !== 'snake') return;
  replay = nextReplay;
  view.setReplay(replay);

  const s = replay.summary;
  if (pendingInitialTime !== null) {
    playbackTime = Math.max(0, Math.min(s.config.duration, pendingInitialTime));
    pendingInitialTime = null;
    setPlaying(false);
    view.applyTime(playbackTime);
  }
  stHead.textContent = `${(s.finalHeadTip[0] * 100).toFixed(1)}, ${(s.finalHeadTip[1] * 100).toFixed(1)}cm`;
  updateControls();
}

function quadDuration(): number {
  return quadReplay?.summary.config.duration ?? 0;
}

function applyQuadTime(t: number): void {
  if (!quadReplay || quadReplay.frames.length === 0) return;
  const dur = quadDuration();
  const tt = dur > 0 ? ((t % dur) + dur) % dur : 0;
  const frames = quadReplay.frames;
  let hi = 1;
  while (hi < frames.length && frames[hi].t < tt) hi++;
  view.applyQuadFrame(frames[Math.min(frames.length - 1, hi)]);
}

async function runQuadWalk(): Promise<void> {
  const seq = ++quadSeq;
  const servo = getServo(motorId);
  qMotor.textContent = `${servo.name}`;
  qResult.textContent = '歩行計算中…';
  setMetricClass(qResult, null);
  // 総質量を default の trunk:leg 比を保ってスケール（軽すぎる脚は強capで暴走するため比は維持）。
  const massFactor = quadMass / QUAD_BASE_TOTAL;
  const replayResult = await runQuadrupedGait(
    {
      trunk: { mass: QUAD_BASE_TRUNK * massFactor },
      leg: { segMass: QUAD_BASE_SEG * massFactor },
      motor: { maxTorqueNm: servo.stallNm },
    },
    60,
  );
  if (seq !== quadSeq || mechanism !== 'quad') return;
  quadReplay = replayResult;
  view.buildQuadReplay(replayResult.layout);
  playbackTime = 0;
  setPlaying(true);
  applyQuadTime(0);

  const s = replayResult.summary;
  qForward.textContent = `${(s.forwardDistanceM * 100).toFixed(1)} cm`;
  setMetricClass(qForward, s.success ? 'good' : 'warn');
  qFall.textContent = s.fell ? `${(s.fellTime ?? 0).toFixed(1)}s で転倒` : 'なし';
  setMetricClass(qFall, s.fell ? 'bad' : 'good');
  qCap.textContent = `${servo.stallNm.toFixed(2)} N·m`;
  setMetricClass(qCap, null);
  qResult.textContent = s.success ? '歩行OK' : s.fell ? '転倒' : '失速（前進せず）';
  setMetricClass(qResult, s.success ? 'good' : 'bad');
  updateControls();
}

function applyMechanism(): void {
  view.setMechanism(mechanism);
  const snake = mechanism === 'snake';
  snakeFriction.style.display = snake ? '' : 'none';
  quadMassControl.style.display = snake ? 'none' : '';
  snakeStats.style.display = snake ? '' : 'none';
  quadStats.style.display = snake ? 'none' : '';
  snakeControls.style.display = ''; // 再生コントロールは両機構で使う
  title.textContent = snake ? 'SNAKE · STAIR' : 'QUAD · WALK';
  subtitle.textContent = snake
    ? 'Rapier physical attempt + diagnostics'
    : 'Rapier 3D 動的歩行（クロール歩容）';
  if (snake) {
    setPlaying(true);
    void loadReplay();
  } else {
    void runQuadWalk();
  }
}

// モーター選択肢は servos.ts の SELECTABLE_SERVO_IDS から生成（単一の真実）
for (const id of SELECTABLE_SERVO_IDS) {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = getServo(id).name;
  selMotor.append(option);
}
selMotor.value = motorId;

selMotor.addEventListener('change', () => {
  motorId = selMotor.value;
  torqueCapNm = getServo(motorId).stallNm;
  torqueCapInput.value = String(torqueCapNm);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  if (mechanism === 'snake') void loadReplay();
  else void runQuadWalk();
});
selMech.addEventListener('change', () => {
  mechanism = selMech.value === 'quad' ? 'quad' : 'snake';
  applyMechanism();
});

function applyPlayback(t: number): void {
  if (mechanism === 'snake') view.applyTime(t);
  else applyQuadTime(t);
}

btnPlay.addEventListener('click', () => setPlaying(!playing));
btnReset.addEventListener('click', () => {
  playbackTime = 0;
  setPlaying(true);
  applyPlayback(playbackTime);
  updateControls();
});
progress.addEventListener('input', () => {
  playbackTime = Number(progress.value);
  setPlaying(false);
  applyPlayback(playbackTime);
  updateControls();
});
speedInput.addEventListener('input', () => {
  speed = Number(speedInput.value);
  updateControls();
});
torqueCapInput.addEventListener('input', () => {
  torqueCapNm = Number(torqueCapInput.value);
  void loadReplay();
});
frictionInput.addEventListener('input', () => {
  friction = Number(frictionInput.value);
  void loadReplay();
});
massInput.addEventListener('input', () => {
  quadMass = Number(massInput.value);
  massValue.textContent = `${quadMass.toFixed(2)} kg`;
  if (mechanism === 'quad') void runQuadWalk();
});

function loop(timestamp: number): void {
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const dt = Math.min((timestamp - lastTimestamp) / 1000, 0.25);
  lastTimestamp = timestamp;

  if (mechanism === 'snake') {
    const duration = view.duration();
    if (replay && duration > 0 && playing) {
      playbackTime = (playbackTime + dt * speed) % duration;
      view.applyTime(playbackTime);
      updateControls();
    }
  } else {
    const duration = quadDuration();
    if (quadReplay && duration > 0 && playing) {
      playbackTime = (playbackTime + dt * speed) % duration;
      applyQuadTime(playbackTime);
      updateControls();
    }
  }

  view.render();
  requestAnimationFrame(loop);
}

void loadReplay().catch((err: unknown) => {
  stStatus.textContent = 'error';
  console.error(err);
});
requestAnimationFrame(loop);
