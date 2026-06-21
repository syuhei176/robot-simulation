/**
 * 統合ダッシュボード — 機構(Mechanism) × コース(Course) × モーション(歩容パラメータ) を1画面で切替。
 *
 * 機構は registry から取得し、UI（機構/コース/モーター選択・Motion スライダー・統計）は
 * すべて Mechanism の宣言（params / liveStats / resultStats）から汎用的に生成する。
 * 新機構は src/mech に Mechanism を1つ足すだけでここに現れる。
 */
import { StairDynamicsView } from './render/StairDynamicsView.ts';
import { getServo, SELECTABLE_SERVO_IDS } from './sim3d/servos.ts';
import { COURSES, COURSE_OPTIONS, type CourseId } from './sim3d/course.ts';
import { MECHANISMS, getMechanism } from './mech/registry.ts';
import {
  defaultParamValues,
  type MechParam,
  type MechReplay,
  type StatRow,
} from './mech/Mechanism.ts';

const app = document.getElementById('app');
if (!app) throw new Error('#app is missing');

const view = new StairDynamicsView(app);
view.setVisible(true);

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
}

const selMech = el<HTMLSelectElement>('sel-mech');
const selMotor = el<HTMLSelectElement>('sel-motor');
const selCourse = el<HTMLSelectElement>('sel-course');
const btnPlay = el<HTMLButtonElement>('btn-play');
const btnReset = el<HTMLButtonElement>('btn-reset');
const progress = el<HTMLInputElement>('s-time');
const speedInput = el<HTMLInputElement>('s-speed');
const torqueCapInput = el<HTMLInputElement>('s-torque-cap');
const timeValue = el<HTMLSpanElement>('v-time');
const speedValue = el<HTMLSpanElement>('v-speed');
const torqueCapValue = el<HTMLSpanElement>('v-torque-cap');
const paramSliders = el<HTMLDivElement>('param-sliders');
const statsBox = el<HTMLDivElement>('stats');
const mechSubtitle = el<HTMLDivElement>('mech-subtitle');

// ---- 状態 ----
let mechId = MECHANISMS[0].id;
let courseId: CourseId = 'stairs';
let motorId = 'sts3215';
let torqueCapNm = getServo(motorId).stallNm;
let paramValues: Record<string, number> = defaultParamValues(getMechanism(mechId));
let replay: MechReplay | null = null;
let playing = true;
let playbackTime = 0;
let speed = 1;
let loadSeq = 0;
let lastTimestamp: number | null = null;

// ---- ドロップダウン生成（単一の真実から） ----
for (const mech of MECHANISMS) {
  const option = document.createElement('option');
  option.value = mech.id;
  option.textContent = mech.name;
  selMech.append(option);
}
selMech.value = mechId;

for (const id of SELECTABLE_SERVO_IDS) {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = getServo(id).name;
  selMotor.append(option);
}
selMotor.value = motorId;

for (const { id, label } of COURSE_OPTIONS) {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = label;
  selCourse.append(option);
}
selCourse.value = courseId;

// ---- 整形ヘルパ ----
function paramDecimals(step: number): number {
  if (step < 0.01) return 3;
  if (step < 0.1) return 2;
  return 1;
}

function formatParam(param: MechParam, value: number): string {
  const text = value.toFixed(paramDecimals(param.step));
  return param.unit ? `${text} ${param.unit}` : text;
}

// ---- Motion スライダーを mechanism.params から動的生成 ----
function buildParamSliders(): void {
  const mech = getMechanism(mechId);
  paramSliders.replaceChildren();
  for (const param of mech.params) {
    const label = document.createElement('label');
    const valueSpan = document.createElement('span');
    valueSpan.textContent = formatParam(param, paramValues[param.key]);
    label.append(document.createTextNode(`${param.label} `), valueSpan);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(param.min);
    input.max = String(param.max);
    input.step = String(param.step);
    input.value = String(paramValues[param.key]);
    input.addEventListener('input', () => {
      const v = Number(input.value);
      paramValues[param.key] = v;
      valueSpan.textContent = formatParam(param, v);
      scheduleRun();
    });

    paramSliders.append(label, input);
  }
}

// ---- 統計描画（StatRow[] を汎用グリッドへ） ----
function renderStats(rows: StatRow[]): void {
  statsBox.replaceChildren();
  for (const row of rows) {
    const label = document.createElement('div');
    label.textContent = row.label;
    const valueCell = document.createElement('div');
    const b = document.createElement('b');
    b.textContent = row.value;
    if (row.kind) b.classList.add(row.kind);
    valueCell.append(b);
    statsBox.append(label, valueCell);
  }
}

// ---- 再生コントロール ----
function setPlaying(next: boolean): void {
  playing = next;
  btnPlay.textContent = next ? '⏸ 一時停止' : '▶ 再生';
  btnPlay.classList.toggle('active', next);
}

function updateControls(): void {
  const duration = replay?.duration || 4;
  const t = duration > 0 ? playbackTime % duration : 0;
  progress.max = String(duration);
  progress.value = t.toFixed(2);
  timeValue.textContent = `${t.toFixed(2)}s`;
  speedValue.textContent = `${speed.toFixed(2)}x`;
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  if (replay) renderStats([...replay.liveStats(playbackTime), ...replay.resultStats()]);
}

// ---- 実行（機構を走らせてリプレイ生成） ----
async function run(): Promise<void> {
  const seq = ++loadSeq;
  const mech = getMechanism(mechId);
  renderStats([{ label: '状態', value: '計算中…' }]);
  const next = await mech.run({
    course: COURSES[courseId](),
    torqueCapNm,
    motorName: getServo(motorId).name,
    params: { ...paramValues },
  });
  // 競合ガード: 新しい実行が始まっていたら破棄
  if (seq !== loadSeq) return;
  replay = next;
  replay.bindView(view);
  playbackTime = 0;
  setPlaying(true);
  replay.applyTime(view, 0);
  updateControls();
}

// スライダーのドラッグ連打で物理計算を投げ過ぎないようデバウンス。
let runTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleRun(): void {
  if (runTimer !== null) clearTimeout(runTimer);
  runTimer = setTimeout(() => {
    runTimer = null;
    void run();
  }, 160);
}

// ---- 機構切替: スライダー再構築・コース有効化・再計算 ----
function applyMechanism(): void {
  const mech = getMechanism(mechId);
  mechSubtitle.textContent = mech.subtitle;
  paramValues = defaultParamValues(mech);
  buildParamSliders();
  selCourse.disabled = !mech.supportsCourse;
  selCourse.title = mech.supportsCourse
    ? 'コース（地形）'
    : 'この機構は現状 平地のみ（段差走破は後続段）';
  void run();
}

// ---- イベント配線 ----
selMech.addEventListener('change', () => {
  mechId = selMech.value;
  applyMechanism();
});
selMotor.addEventListener('change', () => {
  motorId = selMotor.value;
  torqueCapNm = getServo(motorId).stallNm;
  torqueCapInput.value = String(torqueCapNm);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  void run();
});
selCourse.addEventListener('change', () => {
  courseId = selCourse.value as CourseId;
  void run();
});
torqueCapInput.addEventListener('input', () => {
  torqueCapNm = Number(torqueCapInput.value);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  scheduleRun();
});
btnPlay.addEventListener('click', () => setPlaying(!playing));
btnReset.addEventListener('click', () => {
  playbackTime = 0;
  setPlaying(true);
  if (replay) replay.applyTime(view, playbackTime);
  updateControls();
});
progress.addEventListener('input', () => {
  playbackTime = Number(progress.value);
  setPlaying(false);
  if (replay) replay.applyTime(view, playbackTime);
  updateControls();
});
speedInput.addEventListener('input', () => {
  speed = Number(speedInput.value);
  updateControls();
});

// ---- メインループ ----
function loop(timestamp: number): void {
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const dt = Math.min((timestamp - lastTimestamp) / 1000, 0.25);
  lastTimestamp = timestamp;

  if (replay && replay.duration > 0 && playing) {
    playbackTime = (playbackTime + dt * speed) % replay.duration;
    replay.applyTime(view, playbackTime);
    updateControls();
  }
  view.render();
  requestAnimationFrame(loop);
}

// ---- 起動 ----
mechSubtitle.textContent = getMechanism(mechId).subtitle;
buildParamSliders();
torqueCapInput.value = String(torqueCapNm);
torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
void run().catch((err: unknown) => {
  renderStats([{ label: '状態', value: 'error', kind: 'bad' }]);
  console.error(err);
});
requestAnimationFrame(loop);
