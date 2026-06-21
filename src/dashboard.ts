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
const btnScripted = el<HTMLButtonElement>('btn-scripted');
const btnTuned = el<HTMLButtonElement>('btn-tuned');
const tunedInfo = el<HTMLDivElement>('tuned-info');
const improveCurve = el<HTMLCanvasElement>('improve-curve');

// ---- 学習済み歩容（オフライン CMA-ES の成果物）の型 ----
interface TunedScore {
  fitness: number;
  progressM: number;
  feasible: boolean;
}
/** public/tuned/manifest.json の1エントリ（利用可能な最適化結果の一覧）。 */
interface TunedManifestEntry {
  file: string;
  mechanism: string;
  course: string;
  motor: string;
  torqueCapNm: number;
  baseline: TunedScore;
  tuned: TunedScore;
}
/** public/tuned/<file>.json の全体（params と改善履歴を含む）。 */
interface TunedRecord extends TunedManifestEntry {
  optimizedKeys: string[];
  params: Record<string, number>;
  history: Array<{ gen: number; best: number; mean: number; progressM: number }>;
}

// ---- 状態 ----
let mechId = MECHANISMS[0].id;
let courseId: CourseId = 'combined';
let motorId = 'scs0009';
let torqueCapNm = getServo(motorId).stallNm;
let paramValues: Record<string, number> = defaultParamValues(getMechanism(mechId));
let replay: MechReplay | null = null;
let playing = true;
let playbackTime = 0;
let speed = 1;
let loadSeq = 0;
let lastTimestamp: number | null = null;
let manifest: TunedManifestEntry[] = [];
let useTuned = false;
const tunedCache = new Map<string, TunedRecord>();

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

// ---- 学習済み歩容（tuned）の解決・読み込み・適用 ----
/** 非コース機構（四足）は平地固定なので manifest 照合は 'flat' を使う。 */
function effectiveCourse(): string {
  return getMechanism(mechId).supportsCourse ? courseId : 'flat';
}

/** 現在の 機構×コース×モーター に対応する最適化結果を manifest から探す。 */
function findTunedEntry(): TunedManifestEntry | null {
  return (
    manifest.find(
      (e) => e.mechanism === mechId && e.motor === motorId && e.course === effectiveCourse(),
    ) ?? null
  );
}

async function loadTunedRecord(entry: TunedManifestEntry): Promise<TunedRecord> {
  const cached = tunedCache.get(entry.file);
  if (cached) return cached;
  const res = await fetch(`./tuned/${entry.file}`);
  if (!res.ok) throw new Error(`tuned ファイルを取得できません: ${entry.file}`);
  const record = (await res.json()) as TunedRecord;
  tunedCache.set(entry.file, record);
  return record;
}

/** tuned ボタンの有効/無効と選択状態を現在の選択に同期する（結果が無ければ scripted に戻す）。 */
function updateTunedAvailability(): void {
  const entry = findTunedEntry();
  btnTuned.disabled = entry === null;
  if (entry === null) useTuned = false;
  btnTuned.classList.toggle('active', useTuned);
  btnScripted.classList.toggle('active', !useTuned);
}

function syncTorqueUI(): void {
  torqueCapInput.value = String(torqueCapNm);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
}

// ---- 改善カーブ（世代ごとの best / mean fitness） ----
function drawImproveCurve(history: TunedRecord['history'] | null): void {
  const ctx = improveCurve.getContext('2d');
  if (!ctx) return;
  const w = improveCurve.width;
  const h = improveCurve.height;
  ctx.clearRect(0, 0, w, h);
  if (!history || history.length < 2) return;
  const vals = history.flatMap((p) => [p.best, p.mean]);
  const max = Math.max(...vals);
  const min = Math.min(...vals);
  const range = max - min || 1;
  const plot = (key: 'best' | 'mean', color: string, lineWidth: number): void => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    history.forEach((p, i) => {
      const x = (i / (history.length - 1)) * (w - 4) + 2;
      const y = h - 4 - ((p[key] - min) / range) * (h - 8);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot('mean', 'rgba(120, 144, 168, 0.7)', 1);
  plot('best', '#35c8ff', 1.5);
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

// ---- 選択変更の共通経路: tuned 可否を更新し、tuned/scripted に応じて歩容を反映して再計算 ----
async function reload(): Promise<void> {
  updateTunedAvailability();
  const mech = getMechanism(mechId);
  const entry = useTuned ? findTunedEntry() : null;
  if (entry) {
    // tuned: 最適化済みの厳密な float をそのまま適用（スライダーの step に丸めない＝崖系でも再現）。
    const record = await loadTunedRecord(entry);
    paramValues = { ...defaultParamValues(mech), ...record.params };
    // τ上限は丸めた JSON 値（torqueCapNm）ではなく、最適化時と同一のサーボ stall を使う。
    // 崖系（蛇 18cm 階段）では cap の 5e-6 差ですら結果が変わるため、単一の真実=サーボカタログに合わせる。
    torqueCapNm = getServo(motorId).stallNm;
    syncTorqueUI();
    drawImproveCurve(record.history);
    const b = record.baseline;
    const t = record.tuned;
    tunedInfo.textContent = `fitness ${b.fitness.toFixed(2)} → ${t.fitness.toFixed(2)} ・ 前進 ${(b.progressM * 100).toFixed(0)} → ${(t.progressM * 100).toFixed(0)}cm`;
  } else {
    drawImproveCurve(null);
    tunedInfo.textContent = '';
  }
  buildParamSliders();
  await run();
}

// ---- 機構切替: 既定歩容へ戻し・コース有効化・再計算（scripted から開始） ----
function applyMechanism(): void {
  const mech = getMechanism(mechId);
  mechSubtitle.textContent = mech.subtitle;
  useTuned = false;
  paramValues = defaultParamValues(mech);
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  selCourse.disabled = !mech.supportsCourse;
  selCourse.title = mech.supportsCourse
    ? 'コース（地形）'
    : 'この機構は現状 平地のみ（段差走破は後続段）';
  void reload();
}

// ---- イベント配線 ----
selMech.addEventListener('change', () => {
  mechId = selMech.value;
  applyMechanism();
});
selMotor.addEventListener('change', () => {
  motorId = selMotor.value;
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  void reload();
});
selCourse.addEventListener('change', () => {
  courseId = selCourse.value as CourseId;
  void reload();
});
btnScripted.addEventListener('click', () => {
  if (!useTuned) return;
  useTuned = false;
  paramValues = defaultParamValues(getMechanism(mechId));
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  void reload();
});
btnTuned.addEventListener('click', () => {
  if (useTuned || btnTuned.disabled) return;
  useTuned = true;
  void reload();
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
syncTorqueUI();

async function init(): Promise<void> {
  try {
    const res = await fetch('./tuned/manifest.json');
    if (res.ok) manifest = (await res.json()) as TunedManifestEntry[];
  } catch {
    // manifest 不在（オフライン最適化を未実施）は許容。tuned ボタンは無効のまま。
  }
  await reload();
}

void init().catch((err: unknown) => {
  renderStats([{ label: '状態', value: 'error', kind: 'bad' }]);
  console.error(err);
});
requestAnimationFrame(loop);
