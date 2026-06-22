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
import { recordedSnake3DReplay } from './mech/snake3d.ts';
import type { Snake3DReplay } from './sim3d/snake3d-dynamics.ts';
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
const btnRL = el<HTMLButtonElement>('btn-rl');
const tunedInfo = el<HTMLDivElement>('tuned-info');
const linkMaterials = el<HTMLAnchorElement>('link-materials');
const courseRow = selCourse.parentElement as HTMLDivElement; // sel-course を含む .row
const headingCtl = el<HTMLDivElement>('heading-ctl'); // 操舵スライダーの容れ物（RL モードのみ表示）
const headingInput = el<HTMLInputElement>('s-heading');
const headingValue = el<HTMLSpanElement>('v-heading');

// ---- RL 方策（オフライン PPO の成果物・決定論ロールアウトの記録）の型 ----
/** public/policies/manifest.json の1エントリ。 */
interface PolicyManifestEntry {
  file: string;
  mechanism: string;
  course: string;
  motor: string;
  mass: number;
  base: string;
  forwardM: number;
  baseForwardM?: number; // 基盤歩容（残差0）の前進量。あれば「基盤→RL」の上乗せを表示する
  cmdHeadingDeg?: number; // 目標ヘディング指令（操舵）[deg]。無ければ 0（直進）扱い
  achievedDeg?: number; // 達成方位（実測）[deg]
  fell: boolean;
}
/** public/policies/<file>.replay.json（meta + 記録リプレイ）。 */
interface PolicyReplayFile {
  meta: PolicyManifestEntry;
  replay: Snake3DReplay;
}

type MotionMode = 'scripted' | 'rl';

// 本プロジェクトは MuJoCo 蛇（snake3d）専用。registry は snake3d のみだが、将来の拡張に備え一覧から生成する。
const UI_MECHANISMS = MECHANISMS;

// ---- 状態 ----
// 既定は「直進チャレンジ × MG996R」。scripted=基盤歩容が地形に蹴られ斜行 → RL ボタンで汎用方策が
// +x へ操舵し直す様子（基盤 555cm → RL 839cm/+51%）を見せる。コースを切り替えても同じ汎用方策が再生される。
let mechId = UI_MECHANISMS[0].id;
let courseId: CourseId = 'challenge';
let motorId = 'mg996r';
let commandHeadingDeg = 0; // RL 方策に与える目標方位（操舵）。最近傍の録画方位を再生する
let torqueCapNm = getServo(motorId).stallNm;
let paramValues: Record<string, number> = defaultParamValues(getMechanism(mechId));
let replay: MechReplay | null = null;
let playing = true;
let playbackTime = 0;
let speed = 1;
let loadSeq = 0;
let lastTimestamp: number | null = null;
let policyManifest: PolicyManifestEntry[] = [];
let motionMode: MotionMode = 'scripted';
const policyCache = new Map<string, PolicyReplayFile>();
// scripted 物理計算は決定論的（同じ 機構×コース×モーター×cap×params なら同結果）だが、
// メインスレッドで数秒かかる。結果を入力キーでメモ化し、過去に見た構成への再選択を即時化する。
// frames が嵩むので LRU（最古を退避）で上限を設ける。
const REPLAY_CACHE_LIMIT = 24;
const replayCache = new Map<string, MechReplay>();

/** run() の入力（結果を一意に決める）から安定なキャッシュキーを作る。 */
function runKey(): string {
  return JSON.stringify({
    mech: mechId,
    course: courseId,
    motor: motorId,
    cap: torqueCapNm,
    params: paramValues,
  });
}

/** リプレイをキャッシュへ入れる（既存は最新へ詰め直し、上限超過なら最古を退避）。 */
function cacheReplay(key: string, value: MechReplay): void {
  replayCache.delete(key); // 末尾（最新）へ移動して LRU の鮮度を保つ
  replayCache.set(key, value);
  while (replayCache.size > REPLAY_CACHE_LIMIT) {
    const oldest = replayCache.keys().next().value;
    if (oldest === undefined) break;
    replayCache.delete(oldest);
  }
}

// ---- ドロップダウン生成（単一の真実から） ----
for (const mech of UI_MECHANISMS) {
  const option = document.createElement('option');
  option.value = mech.id;
  option.textContent = mech.name;
  selMech.append(option);
}
selMech.value = mechId;
// 蛇機構が1つだけなら機構セレクタは不要なので隠す（増えたら自動で出る）。
if (UI_MECHANISMS.length <= 1) selMech.style.display = 'none';

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

// ---- RL 成果物（記録リプレイ）の解決・読み込み ----
/** RL 成果物を照合するコース名。コース対応機構なら選択中のコース、非対応なら平地。 */
function effectiveCourse(): string {
  const mech = getMechanism(mechId);
  return mech.supportsCourse ? courseId : 'flat';
}

/** 現在の 機構×コース×モーター に対応する RL 方策（記録リプレイ）を全方位ぶん集める。 */
function findPolicyEntries(): PolicyManifestEntry[] {
  return policyManifest.filter(
    (e) => e.mechanism === mechId && e.motor === motorId && e.course === effectiveCourse(),
  );
}

/** 目標方位 deg に最も近い録画方位のエントリを選ぶ（操舵スライダー用）。無ければ null。 */
function pickPolicyEntry(deg: number): PolicyManifestEntry | null {
  const entries = findPolicyEntries();
  if (entries.length === 0) return null;
  return entries.reduce((best, e) =>
    Math.abs((e.cmdHeadingDeg ?? 0) - deg) < Math.abs((best.cmdHeadingDeg ?? 0) - deg) ? e : best,
  );
}

async function loadPolicyReplay(entry: PolicyManifestEntry): Promise<PolicyReplayFile> {
  const cached = policyCache.get(entry.file);
  if (cached) return cached;
  const res = await fetch(`./policies/${entry.file}`);
  if (!res.ok) throw new Error(`RL リプレイを取得できません: ${entry.file}`);
  const record = (await res.json()) as PolicyReplayFile;
  policyCache.set(entry.file, record);
  return record;
}

/** scripted/RL ボタンの有効/無効と選択状態を同期する（RL が無いコースを選んでいたら scripted に戻す）。 */
function updateModeAvailability(): void {
  const hasPolicy = findPolicyEntries().length > 0;
  btnRL.disabled = !hasPolicy;
  if (motionMode === 'rl' && !hasPolicy) motionMode = 'scripted';
  btnScripted.classList.toggle('active', motionMode === 'scripted');
  btnRL.classList.toggle('active', motionMode === 'rl');
  // 操舵スライダーは RL 方策があり RL モードの時だけ表示（基盤歩容は +x のみで操舵できない）。
  headingCtl.style.display = motionMode === 'rl' && hasPolicy ? '' : 'none';
  headingValue.textContent = `${commandHeadingDeg > 0 ? '+' : ''}${commandHeadingDeg}°`;
  headingInput.value = String(commandHeadingDeg);
}

function syncTorqueUI(): void {
  torqueCapInput.value = String(torqueCapNm);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
}

// 材料表（買い物リスト）リンクへ現在のサーボを引き継ぐ（?motor= で初期選択を合わせる）。
function syncMaterialsLink(): void {
  linkMaterials.href = `./materials.html?motor=${encodeURIComponent(motorId)}`;
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

/** ブラウザに1フレーム描画させてから重い同期計算へ入るための yield（「計算中…」を必ず見せる）。 */
function yieldToPaint(): Promise<void> {
  return new Promise((resolve) => {
    let done = false;
    const finish = (): void => {
      if (done) return;
      done = true;
      resolve();
    };
    // 通常は rAF（描画直前）で macrotask を仕込み、描画後に resolve する。
    requestAnimationFrame(() => setTimeout(finish, 0));
    // 非表示タブ等で rAF が発火しない場合でも計算へ進めるフォールバック。
    setTimeout(finish, 100);
  });
}

/** 生成/キャッシュ済みリプレイをビューへ適用し先頭から再生する（run・キャッシュヒット・RL 共通）。 */
function applyReplay(next: MechReplay): void {
  replay = next;
  next.bindView(view);
  playbackTime = 0;
  setPlaying(true);
  next.applyTime(view, 0);
  updateControls();
}

// ---- 実行（機構を走らせてリプレイ生成・結果はキャッシュ） ----
async function run(): Promise<void> {
  const seq = ++loadSeq;
  const key = runKey();
  const cached = replayCache.get(key);
  if (cached) {
    cacheReplay(key, cached); // LRU 鮮度を更新
    applyReplay(cached);
    return;
  }
  const mech = getMechanism(mechId);
  renderStats([{ label: '状態', value: '計算中…（数秒かかることがあります）' }]);
  // 物理ループはメインスレッドを同期占有する。まず1フレーム描画させて「計算中…」を見せてから計算する。
  await yieldToPaint();
  if (seq !== loadSeq) return; // paint 待ちの間に新しい実行が始まっていたら破棄
  const next = await mech.run({
    course: COURSES[courseId](),
    torqueCapNm,
    motorName: getServo(motorId).name,
    params: { ...paramValues },
  });
  // 競合ガード: 新しい実行が始まっていたら破棄
  if (seq !== loadSeq) return;
  cacheReplay(key, next);
  applyReplay(next);
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

// ---- RL: 記録済みリプレイ（決定論ロールアウトの frames）を読み込んで再生する ----
async function runRL(): Promise<void> {
  const entry = pickPolicyEntry(commandHeadingDeg);
  if (!entry) {
    motionMode = 'scripted';
    await reload();
    return;
  }
  const seq = ++loadSeq;
  renderStats([{ label: '状態', value: 'RL 記録を読み込み中…' }]);
  const record = await loadPolicyReplay(entry);
  if (seq !== loadSeq) return; // 競合ガード
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  // 操舵指令（目標方位）と達成方位を見せつつ、基盤→RL（+N%）の上乗せを明示する。
  const cmdDeg = entry.cmdHeadingDeg ?? 0;
  const cmdStr = `目標方位 ${cmdDeg > 0 ? '+' : ''}${cmdDeg}°`;
  const achStr =
    entry.achievedDeg !== undefined
      ? ` → 実方位 ${entry.achievedDeg > 0 ? '+' : ''}${entry.achievedDeg.toFixed(0)}°`
      : '';
  const rlCm = entry.forwardM * 100;
  if (entry.baseForwardM !== undefined && entry.baseForwardM > 0) {
    const baseCm = entry.baseForwardM * 100;
    const pct = ((rlCm - baseCm) / baseCm) * 100;
    tunedInfo.textContent =
      `RL方策 ${cmdStr}${achStr}・基盤 ${baseCm.toFixed(0)}cm → RL ${rlCm.toFixed(0)}cm ` +
      `(${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%)${entry.fell ? '（破綻）' : ''}`;
  } else {
    tunedInfo.textContent = `RL方策 ${cmdStr}${achStr}・前進 ${rlCm.toFixed(0)}cm${entry.fell ? '（破綻）' : ''}`;
  }
  buildParamSliders();
  const rl = recordedSnake3DReplay(
    record.replay as Snake3DReplay,
    getServo(motorId).name,
    torqueCapNm,
  );
  applyReplay(rl);
}

// ---- 選択変更の共通経路: モード可否を更新し、scripted（基盤歩容）/RL に応じて再計算 ----
async function reload(): Promise<void> {
  updateModeAvailability();
  if (motionMode === 'rl') {
    await runRL();
    return;
  }
  // scripted: 基盤登坂歩容を選択コースでライブ実行。
  tunedInfo.textContent = '基盤歩容（残差0・開ループ）をコース上でライブ実行';
  buildParamSliders();
  await run();
}

// ---- 機構に応じた周辺UI（副題・コース選択の有効/可視）の同期。起動時と機構切替の両方で使う。 ----
function syncMechChrome(): void {
  const mech = getMechanism(mechId);
  mechSubtitle.textContent = mech.subtitle;
  selCourse.disabled = !mech.supportsCourse;
  courseRow.style.display = mech.supportsCourse ? '' : 'none';
}

// ---- 機構切替: 既定歩容へ戻し・コース有効化・再計算（scripted から開始） ----
function applyMechanism(): void {
  motionMode = 'scripted';
  paramValues = defaultParamValues(getMechanism(mechId));
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  syncMechChrome();
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
  syncMaterialsLink();
  void reload();
});
selCourse.addEventListener('change', () => {
  courseId = selCourse.value as CourseId;
  void reload();
});
btnScripted.addEventListener('click', () => {
  if (motionMode === 'scripted') return;
  motionMode = 'scripted';
  paramValues = defaultParamValues(getMechanism(mechId));
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  void reload();
});
btnRL.addEventListener('click', () => {
  if (motionMode === 'rl' || btnRL.disabled) return;
  motionMode = 'rl';
  void reload();
});
headingInput.addEventListener('input', () => {
  commandHeadingDeg = Number(headingInput.value);
  headingValue.textContent = `${commandHeadingDeg > 0 ? '+' : ''}${commandHeadingDeg}°`;
  if (motionMode === 'rl') void runRL(); // 最近傍の録画方位を再生（fetch はキャッシュ）
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
syncMechChrome();
syncTorqueUI();
syncMaterialsLink();

async function init(): Promise<void> {
  try {
    const res = await fetch('./policies/manifest.json');
    if (res.ok) policyManifest = (await res.json()) as PolicyManifestEntry[];
  } catch {
    // policy manifest 不在（RL 未学習）は許容。RL ボタンは無効のまま。
  }
  await reload();
}

void init().catch((err: unknown) => {
  renderStats([{ label: '状態', value: 'error', kind: 'bad' }]);
  console.error(err);
});
requestAnimationFrame(loop);
