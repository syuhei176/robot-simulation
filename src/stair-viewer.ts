import { StairDynamicsView } from './render/StairDynamicsView.ts';
import { createKinematicStairReplay } from './sim3d/stair-kinematic-replay.ts';
import type { StairDynamicsReplay } from './sim3d/stair-dynamics.ts';

const app = document.getElementById('app');
if (!app) throw new Error('#app is missing');

const view = new StairDynamicsView(app);
view.setVisible(true);

let replay: StairDynamicsReplay | null = null;
let playing = true;
let playbackTime = 0;
let speed = 1;
let lastTimestamp: number | null = null;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
}

const btnPlay = el<HTMLButtonElement>('btn-play');
const btnReset = el<HTMLButtonElement>('btn-reset');
const progress = el<HTMLInputElement>('s-time');
const speedInput = el<HTMLInputElement>('s-speed');
const timeValue = el<HTMLSpanElement>('v-time');
const speedValue = el<HTMLSpanElement>('v-speed');
const stStatus = el<HTMLElement>('st-status');
const stTau = el<HTMLElement>('st-tau');
const stRatio = el<HTMLElement>('st-ratio');
const stMu = el<HTMLElement>('st-mu');
const stHead = el<HTMLElement>('st-head');

function setPlaying(next: boolean): void {
  playing = next;
  btnPlay.textContent = next ? '⏸ 一時停止' : '▶ 再生';
  btnPlay.classList.toggle('active', next);
}

function updateControls(): void {
  const duration = view.duration() || 4;
  const t = duration > 0 ? playbackTime % duration : 0;
  progress.max = String(duration);
  progress.value = t.toFixed(2);
  timeValue.textContent = `${t.toFixed(2)}s`;
  speedValue.textContent = `${speed.toFixed(2)}x`;
}

async function loadReplay(): Promise<void> {
  replay = createKinematicStairReplay({}, 45);
  view.setReplay(replay);

  const s = replay.summary;
  const ratio = s.staticArcPeakNm > 1e-9 ? s.maxDemandTorqueNm / s.staticArcPeakNm : Infinity;
  stStatus.textContent = s.success ? 'success' : 'failed';
  stTau.textContent = `${s.maxDemandTorqueNm.toFixed(3)} N·m`;
  stRatio.textContent = `${ratio.toFixed(2)}x`;
  stMu.textContent = s.maxMuDemand.toFixed(2);
  stHead.textContent = `${(s.finalHeadTip[0] * 100).toFixed(1)}, ${(s.finalHeadTip[1] * 100).toFixed(1)}cm`;
  updateControls();
}

btnPlay.addEventListener('click', () => setPlaying(!playing));
btnReset.addEventListener('click', () => {
  playbackTime = 0;
  setPlaying(true);
  view.applyTime(playbackTime);
  updateControls();
});
progress.addEventListener('input', () => {
  playbackTime = Number(progress.value);
  setPlaying(false);
  view.applyTime(playbackTime);
  updateControls();
});
speedInput.addEventListener('input', () => {
  speed = Number(speedInput.value);
  updateControls();
});

function loop(timestamp: number): void {
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const dt = Math.min((timestamp - lastTimestamp) / 1000, 0.25);
  lastTimestamp = timestamp;

  const duration = view.duration();
  if (replay && duration > 0 && playing) {
    playbackTime = (playbackTime + dt * speed) % duration;
    view.applyTime(playbackTime);
    updateControls();
  }

  view.render();
  requestAnimationFrame(loop);
}

void loadReplay().catch((err: unknown) => {
  stStatus.textContent = 'error';
  console.error(err);
});
requestAnimationFrame(loop);
