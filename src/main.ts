import * as tf from '@tensorflow/tfjs';
import { DEFAULT_SIM, DEFAULT_GAIT, type SimParams, type GaitParams } from './config.ts';
import { SnakePhysics } from './sim/SnakePhysics.ts';
import { CPGController } from './control/CPGController.ts';
import { SceneView } from './render/SceneView.ts';
import { StairDynamicsView } from './render/StairDynamicsView.ts';
import { MicrobotEnv } from './env/MicrobotEnv.ts';
import { Policy } from './rl/Policy.ts';
import { PPO } from './rl/PPO.ts';
import { DEFAULT_STAIR_REPLAY_CONFIG } from './sim3d/stair-kinematic-replay.ts';
import { sampleStairDiagnostics } from './sim3d/stair-feasibility.ts';
import { runPhysicalStairAttemptReplay } from './sim3d/stair-physical-attempt.ts';
import type { StairDynamicsReplay, StairFailureKind } from './sim3d/stair-dynamics.ts';

const SAVE_KEY = 'microbot-ppo';

const sim: SimParams = { ...DEFAULT_SIM };
const gait: GaitParams = { ...DEFAULT_GAIT };

const app = document.getElementById('app')!;
const view = new SceneView(app, sim.links + 1, sim.restLength);
const stairView = new StairDynamicsView(app);

// --- 手動 CPG 経路 ---
const cpgPhysics = new SnakePhysics(sim);
const cpg = new CPGController(cpgPhysics.jointCount, gait);

// --- 強化学習経路 ---
const trainEnv = new MicrobotEnv();
const demoEnv = new MicrobotEnv();
let policy = new Policy(trainEnv.obsDim, trainEnv.actDim);
let ppo = new PPO(policy);
let demoObs = demoEnv.reset();

type Mode = 'cpg' | 'rl' | 'stair';

let mode: Mode = 'rl';
let training = false;
const rewardHistory: number[] = [];
let stairReplay: StairDynamicsReplay | null = null;
let stairPlaying = true;
let stairTime = 0;
let stairSpeed = 1;
let stairTorqueCapNm = 0.25;
let stairFriction = 0.6;
let stairLoadSeq = 0;

// メインループの時間積分用（wall-clock 同期）
let lastTimestamp: number | null = null;
let accumulator = 0;
// タブ復帰などで大量のステップをまとめ実行しないための 1 フレーム上限 [s]
const MAX_FRAME_TIME = 0.25;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} が見つかりません`);
  return node as T;
}

// ---------- モード切替 ----------
const sectionCpg = el<HTMLDivElement>('section-cpg');
const sectionRl = el<HTMLDivElement>('section-rl');
const sectionStair = el<HTMLDivElement>('section-stair');
const btnModeCpg = el<HTMLButtonElement>('mode-cpg');
const btnModeRl = el<HTMLButtonElement>('mode-rl');
const btnModeStair = el<HTMLButtonElement>('mode-stair');

function setMode(next: Mode): void {
  mode = next;
  sectionCpg.classList.toggle('hidden', next !== 'cpg');
  sectionRl.classList.toggle('hidden', next !== 'rl');
  sectionStair.classList.toggle('hidden', next !== 'stair');
  btnModeCpg.classList.toggle('active', next === 'cpg');
  btnModeRl.classList.toggle('active', next === 'rl');
  btnModeStair.classList.toggle('active', next === 'stair');
  view.setVisible(next !== 'stair');
  stairView.setVisible(next === 'stair');
  view.resetFollow();
  accumulator = 0; // モード間で経過時間を持ち越さない
  lastTimestamp = null;
  if (next === 'stair') void loadStairReplay();
}
btnModeCpg.addEventListener('click', () => setMode('cpg'));
btnModeRl.addEventListener('click', () => setMode('rl'));
btnModeStair.addEventListener('click', () => setMode('stair'));

// ---------- CPG スライダー ----------
function bindSlider(
  sliderId: string,
  valueId: string,
  init: number,
  on: (v: number) => void,
): void {
  const slider = el<HTMLInputElement>(sliderId);
  const out = el<HTMLSpanElement>(valueId);
  slider.value = String(init);
  const apply = (): void => {
    const v = Number(slider.value);
    out.textContent = v.toFixed(2);
    on(v);
  };
  slider.addEventListener('input', apply);
  apply();
}
bindSlider('s-amp', 'v-amp', gait.amplitude, (v) => (gait.amplitude = v));
bindSlider('s-freq', 'v-freq', gait.frequency, (v) => (gait.frequency = v));
bindSlider('s-phase', 'v-phase', gait.phaseLag, (v) => (gait.phaseLag = v));
bindSlider('s-turn', 'v-turn', gait.turnBias, (v) => (gait.turnBias = v));

// ---------- RL コントロール ----------
const btnTrain = el<HTMLButtonElement>('btn-train');
const stIter = el<HTMLElement>('st-iter');
const stFwd = el<HTMLElement>('st-fwd');
const stStd = el<HTMLElement>('st-std');
const stSpeed = el<HTMLElement>('st-speed');
const stBackend = el<HTMLElement>('st-backend');
const curve = el<HTMLCanvasElement>('rl-curve');

btnTrain.addEventListener('click', () => {
  training = !training;
  btnTrain.textContent = training ? '⏸ 学習停止' : '▶ 学習開始';
  btnTrain.classList.toggle('go', !training);
  if (training) void trainLoop();
});

el<HTMLButtonElement>('btn-reset-weights').addEventListener('click', () => {
  training = false;
  btnTrain.textContent = '▶ 学習開始';
  btnTrain.classList.add('go');
  ppo.dispose();
  policy.dispose();
  policy = new Policy(trainEnv.obsDim, trainEnv.actDim);
  ppo = new PPO(policy);
  rewardHistory.length = 0;
  trainEnv.reset();
  demoObs = demoEnv.reset();
  drawCurve();
  stIter.textContent = '0';
  stFwd.textContent = '0.0';
});

el<HTMLButtonElement>('btn-save').addEventListener('click', () => void policy.save(SAVE_KEY));
el<HTMLButtonElement>('btn-load').addEventListener('click', async () => {
  const ok = await policy.load(SAVE_KEY);
  if (!ok) alert('保存された重みが見つかりません');
});

// ---------- 階段再生 ----------
const btnStairPlay = el<HTMLButtonElement>('btn-stair-play');
const btnStairReset = el<HTMLButtonElement>('btn-stair-reset');
const stairProgress = el<HTMLInputElement>('s-stair-time');
const stairSpeedInput = el<HTMLInputElement>('s-stair-speed');
const stairTorqueCapInput = el<HTMLInputElement>('s-stair-torque-cap');
const stairFrictionInput = el<HTMLInputElement>('s-stair-friction');
const stairTimeValue = el<HTMLSpanElement>('v-stair-time');
const stairSpeedValue = el<HTMLSpanElement>('v-stair-speed');
const stairTorqueCapValue = el<HTMLSpanElement>('v-stair-torque-cap');
const stairFrictionValue = el<HTMLSpanElement>('v-stair-friction');
const stStairStatus = el<HTMLElement>('st-stair-status');
const stStairTauNow = el<HTMLElement>('st-stair-tau-now');
const stStairTau = el<HTMLElement>('st-stair-tau');
const stStairMuNow = el<HTMLElement>('st-stair-mu-now');
const stStairMu = el<HTMLElement>('st-stair-mu');
const stStairSlip = el<HTMLElement>('st-stair-slip');
const stStairClearance = el<HTMLElement>('st-stair-clearance');
const stStairSupports = el<HTMLElement>('st-stair-supports');
const stStairFirstFailure = el<HTMLElement>('st-stair-first-failure');
const stStairHead = el<HTMLElement>('st-stair-head');

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

function stairOverrides(): Partial<StairDynamicsReplay['summary']['config']> {
  return {
    friction: stairFriction,
    motor: {
      ...DEFAULT_STAIR_REPLAY_CONFIG.motor,
      maxTorqueNm: stairTorqueCapNm,
    },
  };
}

function setStairPlaying(playing: boolean): void {
  stairPlaying = playing;
  btnStairPlay.textContent = playing ? '⏸ 一時停止' : '▶ 再生';
  btnStairPlay.classList.toggle('active', playing);
}

function updateStairControls(): void {
  const duration = stairView.duration();
  const clamped = duration > 0 ? stairTime % duration : 0;
  stairProgress.max = String(duration || 4);
  stairProgress.value = clamped.toFixed(2);
  stairTimeValue.textContent = `${clamped.toFixed(2)}s`;
  stairSpeedValue.textContent = `${stairSpeed.toFixed(2)}x`;
  stairTorqueCapValue.textContent = `${stairTorqueCapNm.toFixed(2)} N·m`;
  stairFrictionValue.textContent = stairFriction.toFixed(2);
  updateStairDiagnostics();
}

function updateStairDiagnostics(): void {
  const diagnostics = sampleStairDiagnostics(stairReplay, stairTime);
  const summary = stairReplay?.summary.feasibility;
  if (!stairReplay || !diagnostics || !summary) return;
  const currentTorque = diagnostics.motorDemandNm ?? diagnostics.torqueNm;
  const currentTorqueRatio = diagnostics.motorTorqueRatio ?? diagnostics.torqueRatio;

  stStairStatus.textContent = formatFailureKinds(diagnostics.failureKinds);
  setMetricClass(stStairStatus, diagnostics.failureKinds.length > 0 ? 'bad' : 'good');

  stStairTauNow.textContent = `${fmtFinite(currentTorque, 2)} / ${fmtFinite(
    diagnostics.torqueLimitNm,
    2,
  )} (${fmtFinite(currentTorqueRatio, 1)}x)`;
  setMetricClass(
    stStairTauNow,
    currentTorqueRatio > 1 ? 'bad' : currentTorqueRatio > 0.8 ? 'warn' : null,
  );

  stStairTau.textContent = `${fmtFinite(summary.maxTorqueNm, 2)} N·m (${fmtFinite(
    summary.maxTorqueRatio,
    1,
  )}x)`;
  setMetricClass(
    stStairTau,
    summary.maxTorqueRatio > 1 ? 'bad' : summary.maxTorqueRatio > 0.8 ? 'warn' : null,
  );

  stStairMuNow.textContent = `${fmtFinite(diagnostics.requiredMu, 2)} / ${fmtFinite(
    diagnostics.friction,
    2,
  )}`;
  setMetricClass(
    stStairMuNow,
    diagnostics.requiredMu > diagnostics.friction
      ? 'bad'
      : diagnostics.requiredMu > 0.8
        ? 'warn'
        : null,
  );

  stStairMu.textContent = `${fmtFinite(summary.maxRequiredMu, 2)} @ ${summary.worstMuTime.toFixed(
    2,
  )}s`;
  setMetricClass(stStairMu, summary.maxRequiredMu > stairFriction ? 'bad' : null);

  stStairSlip.textContent = `${(diagnostics.slipSpeedMps * 100).toFixed(1)} cm/s`;
  setMetricClass(stStairSlip, diagnostics.failureKinds.includes('slip') ? 'bad' : null);

  stStairClearance.textContent = `${(diagnostics.minClearanceM * 1000).toFixed(1)} mm`;
  setMetricClass(stStairClearance, diagnostics.minClearanceM < -0.006 ? 'bad' : null);

  stStairSupports.textContent = `${diagnostics.supportCount}/${stairReplay.summary.config.morphology.n}`;
  setMetricClass(stStairSupports, diagnostics.failureKinds.includes('fall') ? 'bad' : null);

  stStairFirstFailure.textContent =
    summary.firstFailureTime === null ? 'なし' : `${summary.firstFailureTime.toFixed(2)}s`;
  setMetricClass(stStairFirstFailure, summary.firstFailureTime === null ? 'good' : 'bad');
}

async function loadStairReplay(): Promise<void> {
  if (stairReplay) return;
  const seq = ++stairLoadSeq;
  stStairStatus.textContent = '物理計算中';
  try {
    const nextReplay = await runPhysicalStairAttemptReplay(stairOverrides(), 45);
    if (seq !== stairLoadSeq) return;
    stairReplay = nextReplay;
    stairView.setReplay(stairReplay);
    const s = stairReplay.summary;
    stStairHead.textContent = `${(s.finalHeadTip[0] * 100).toFixed(1)}, ${(s.finalHeadTip[1] * 100).toFixed(1)}cm`;
    updateStairControls();
  } catch (err) {
    stStairStatus.textContent = 'error';
    console.error(err);
  }
}

btnStairPlay.addEventListener('click', () => setStairPlaying(!stairPlaying));
btnStairReset.addEventListener('click', () => {
  stairTime = 0;
  setStairPlaying(true);
  stairView.applyTime(stairTime);
  updateStairControls();
});
stairProgress.addEventListener('input', () => {
  stairTime = Number(stairProgress.value);
  setStairPlaying(false);
  stairView.applyTime(stairTime);
  updateStairControls();
});
stairSpeedInput.addEventListener('input', () => {
  stairSpeed = Number(stairSpeedInput.value);
  updateStairControls();
});
stairTorqueCapInput.addEventListener('input', () => {
  stairTorqueCapNm = Number(stairTorqueCapInput.value);
  stairReplay = null;
  void loadStairReplay();
});
stairFrictionInput.addEventListener('input', () => {
  stairFriction = Number(stairFrictionInput.value);
  stairReplay = null;
  void loadStairReplay();
});

async function trainLoop(): Promise<void> {
  await tf.ready();
  while (training) {
    const stats = ppo.runIteration(trainEnv);
    rewardHistory.push(stats.meanEpisodeForward);
    stIter.textContent = String(stats.iteration);
    stFwd.textContent = stats.meanEpisodeForward.toFixed(2);
    stStd.textContent = stats.std.toFixed(3);
    drawCurve();
    await tf.nextFrame(); // UI / 描画に制御を返す
  }
}

function drawCurve(): void {
  const ctx = curve.getContext('2d');
  if (!ctx) return;
  const w = curve.width;
  const h = curve.height;
  ctx.clearRect(0, 0, w, h);
  if (rewardHistory.length < 2) return;
  const max = Math.max(...rewardHistory, 1);
  const min = Math.min(...rewardHistory, 0);
  const range = max - min || 1;
  ctx.strokeStyle = '#35c8ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  rewardHistory.forEach((v, i) => {
    const x = (i / (rewardHistory.length - 1)) * (w - 4) + 2;
    const y = h - 4 - ((v - min) / range) * (h - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ---------- メインループ（wall-clock 同期の固定タイムステップ積分） ----------
// 実経過時間を accumulator に貯め、各モードの周期ぶん溜まるごとに 1 ステップ進める。
// これでディスプレイのリフレッシュレートに依らず常に等速再生になる。
function loop(timestamp: number): void {
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const frameTime = Math.min((timestamp - lastTimestamp) / 1000, MAX_FRAME_TIME);
  lastTimestamp = timestamp;
  accumulator += frameTime;

  if (mode === 'stair') {
    const duration = stairView.duration();
    if (duration > 0 && stairPlaying) {
      stairTime = (stairTime + frameTime * stairSpeed) % duration;
      stairView.applyTime(stairTime);
      updateStairControls();
    }
    stairView.render();
    requestAnimationFrame(loop);
    return;
  }

  let activePhysics: SnakePhysics;
  if (mode === 'cpg') {
    // 物理周期 dt ごとに 1 ステップ
    while (accumulator >= sim.dt) {
      cpgPhysics.setJointTargets(cpg.update(sim.dt));
      cpgPhysics.step(sim.dt);
      accumulator -= sim.dt;
    }
    activePhysics = cpgPhysics;
  } else {
    // 制御周期 (= controlSubsteps * dt) ごとに方策を 1 ステップ再生
    const controlPeriod = demoEnv.controlPeriod;
    while (accumulator >= controlPeriod) {
      const action = policy.actMean(demoObs);
      const res = demoEnv.step(action);
      demoObs = res.done ? demoEnv.reset() : res.obs;
      accumulator -= controlPeriod;
    }
    activePhysics = demoEnv.physics;
  }

  view.sync(activePhysics);
  view.render();

  const v = activePhysics.centerOfMassVelocity();
  stSpeed.textContent = Math.hypot(v[0], v[1]).toFixed(2);

  requestAnimationFrame(loop);
}

void tf.ready().then(() => {
  stBackend.textContent = tf.getBackend();
});

const initialModeParam = new URLSearchParams(window.location.search).get('mode');
const initialMode: Mode =
  initialModeParam === 'stair' ? 'stair' : initialModeParam === 'cpg' ? 'cpg' : 'rl';
setMode(initialMode);
requestAnimationFrame(loop);
