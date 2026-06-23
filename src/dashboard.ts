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
import { ROOM, type Snake3DReplay } from './sim3d/snake3d-dynamics.ts';
import { SnakeEnv, defaultSnakeEnvConfig, roomSnakeEnvConfig } from './env/SnakeEnv.ts';
import { makePolicyForward } from './rl/policyForward.ts';
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
const btnLive = el<HTMLButtonElement>('btn-live');
const tunedInfo = el<HTMLDivElement>('tuned-info');
const modeHint = el<HTMLDivElement>('mode-hint');
const linkMaterials = el<HTMLAnchorElement>('link-materials');
const courseRow = selCourse.parentElement as HTMLDivElement; // sel-course を含む .row
const headingCtl = el<HTMLDivElement>('heading-ctl'); // 操舵スライダーの容れ物（RL/ライブ モードで表示）
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
  policy?: string; // 学習方策の種別。'general' ならコース汎用の単一重み（ライブ駆動に使える）
  forwardM?: number; // +x コースの前進量。部屋ナビでは未使用（reached/goalDistM を使う）
  baseForwardM?: number; // 基盤歩容（残差0）の前進量。あれば「基盤→RL」の上乗せを表示する
  cmdHeadingDeg?: number; // 目標ヘディング指令（操舵）[deg]。無ければ 0（直進）扱い
  achievedDeg?: number; // 達成方位（実測）[deg]
  reached?: boolean; // 部屋ナビ: ゴール壁に到達したか
  goalDistM?: number; // 部屋ナビ: 終端の目標壁までの距離 [m]
  fell?: boolean;
}
/** public/policies/<file>.replay.json（meta + 記録リプレイ）。 */
interface PolicyReplayFile {
  meta: PolicyManifestEntry;
  replay: Snake3DReplay;
}
/** public/policies/<stem>.json（重み JSON・ライブ方策フォワード用）。 */
interface PolicyWeightsFile {
  obsDim: number;
  actDim: number;
  hidden: number;
  weights: { policy: number[][]; value: number[][]; logStd: number[] };
}

// scripted=基盤歩容のライブ実行 / rl=録画リプレイ再生 / live=方策をブラウザ内でリアルタイム駆動。
type MotionMode = 'scripted' | 'rl' | 'live';

const DEG = Math.PI / 180;

// 本プロジェクトは MuJoCo 蛇（snake3d）専用。registry は snake3d のみだが、将来の拡張に備え一覧から生成する。
const UI_MECHANISMS = MECHANISMS;

// ---- 状態 ----
// 既定は「直進チャレンジ × MG996R」。scripted=基盤歩容が地形に蹴られ斜行 → RL ボタンで汎用方策が
// +x へ操舵し直す様子（基盤 555cm → RL 839cm/+51%）を見せる。コースを切り替えても同じ汎用方策が再生される。
let mechId = UI_MECHANISMS[0].id;
let courseId: CourseId = 'challenge';
let motorId = 'mg996r';
let commandHeadingDeg = 0; // 方策に与える目標方位（操舵）。RL=最近傍の録画方位を再生 / ライブ=連続で即操舵
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
const weightsCache = new Map<string, PolicyWeightsFile>();

// ---- ライブ操舵（方策をブラウザ内でリアルタイムに 1 ステップずつ駆動）の状態 ----
// 録画再生(replay)とは別経路。SnakeEnv をブラウザ内に1つ持ち、毎フレーム制御周期ぶん step して
// 現在姿勢をビューへ流す。目標方位スライダーは env.setCommandHeading でライブに反映される。
let liveEnv: SnakeEnv | null = null;
let liveForward: ((obs: ArrayLike<number>) => Float32Array) | null = null;
let liveObs: Float32Array = new Float32Array(0);
let liveAccum = 0; // 実時間→制御ステップの蓄積（controlPeriod 毎に1ステップ進める）
let liveStartProj = 0; // エピソード開始時の指令方向射影（前進量表示の基準）
let liveStatsThrottle = 0;
const MAX_LIVE_STEPS_PER_FRAME = 3; // 1フレームで進める制御ステップ上限（重い時の暴走防止）
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

/**
 * ブラウザ内でリアルタイム駆動できる単一重み方策の重み stem。無ければ null。
 * 部屋ナビは room 専用方策、それ以外はコース汎用方策。manifest に該当 policy のエントリがある時だけ返す。
 */
function livePolicyStem(): string | null {
  const policyTag = effectiveCourse() === 'room' ? 'room' : 'general';
  const ok = policyManifest.some(
    (e) => e.mechanism === mechId && e.motor === motorId && e.policy === policyTag,
  );
  return ok ? `snake3d-${policyTag}-${motorId}` : null;
}

/** ライブ方策の重み JSON を取得（fetch はキャッシュ）。決定論フォワードに渡す。 */
async function loadPolicyWeights(stem: string): Promise<PolicyWeightsFile> {
  const cached = weightsCache.get(stem);
  if (cached) return cached;
  const res = await fetch(`./policies/${stem}.json`);
  if (!res.ok) throw new Error(`方策の重みを取得できません: ${stem}`);
  const file = (await res.json()) as PolicyWeightsFile;
  weightsCache.set(stem, file);
  return file;
}

/** 目標方位スライダーをモード×コースに合わせて構成（RL=録画方位に量子化 / ライブ=連続）。 */
function syncHeadingSlider(): void {
  // 部屋ナビは ±90°（曲がれる範囲）。+x コースは ±25/30°。
  const room = effectiveCourse() === 'room';
  if (motionMode === 'live') {
    // ライブは連続操舵。スライダーを動かすと即座に方位指令が変わる。
    const max = room ? 90 : 30;
    headingInput.min = String(-max);
    headingInput.max = String(max);
    headingInput.step = room ? '5' : '1';
    commandHeadingDeg = Math.max(-max, Math.min(max, Math.round(commandHeadingDeg)));
  } else {
    // RL は録画済みの離散方位の最近傍を再生する（部屋=±90/45°、+x=±25/5°）。
    const max = room ? 90 : 25;
    const step = room ? 45 : 5;
    headingInput.min = String(-max);
    headingInput.max = String(max);
    headingInput.step = String(step);
    commandHeadingDeg = Math.max(-max, Math.min(max, Math.round(commandHeadingDeg / step) * step));
  }
  headingInput.value = String(commandHeadingDeg);
  headingValue.textContent = `${commandHeadingDeg > 0 ? '+' : ''}${commandHeadingDeg}°`;
}

/** scripted/RL/ライブ ボタンの有効/無効と選択状態を同期する（成果物が無いモードは scripted に戻す）。 */
function updateModeAvailability(): void {
  const hasPolicy = findPolicyEntries().length > 0; // 選択コースの録画リプレイ（RL 再生）
  const hasLive = livePolicyStem() !== null; // ブラウザ内ライブ駆動できる単一重み方策
  btnRL.disabled = !hasPolicy;
  btnLive.disabled = !hasLive;
  if (motionMode === 'rl' && !hasPolicy) motionMode = 'scripted';
  if (motionMode === 'live' && !hasLive) motionMode = 'scripted';
  btnScripted.classList.toggle('active', motionMode === 'scripted');
  btnRL.classList.toggle('active', motionMode === 'rl');
  btnLive.classList.toggle('active', motionMode === 'live');
  // 操舵スライダーは方策がある RL/ライブ モードの時だけ表示（基盤歩容は +x のみで操舵できない）。
  const showHeading = (motionMode === 'rl' && hasPolicy) || (motionMode === 'live' && hasLive);
  headingCtl.style.display = showHeading ? '' : 'none';
  // 再生位置スライダーはライブでは無効（録画ではないのでシークできない）。
  progress.disabled = motionMode === 'live';
  // モードの説明（findability 改善）。録画/ライブが無い時は理由を出す。
  if (motionMode === 'scripted') {
    modeHint.textContent =
      '基盤歩容＝開ループの登坂歩容（操舵なし）。録画方策を見るなら「RL録画」、その場で操舵するなら「ライブ」を選択。';
  } else if (motionMode === 'rl') {
    modeHint.textContent =
      '録画済み方策を再生中。下の「目標方位」スライダーで方向を選ぶ（録画した方位の最近傍を再生）。';
  } else {
    modeHint.textContent =
      '方策をブラウザ内でリアルタイム駆動中。「目標方位」スライダーを動かすと即座に操舵します。';
  }
  if (!hasPolicy && !hasLive) {
    modeHint.textContent += ' ※この コース×モーター には録画方策がありません（基盤歩容のみ）。';
  }
  syncHeadingSlider();
  syncUrl();
}

/**
 * 現在の状態（コース・モーター・モード・目標方位）を URL クエリへ反映（共有・ブックマーク・リロード安定）。
 * 例: `?course=room&mode=rl&heading=-90` で教師の右旋回ビューに直行できる。既定値は省いて URL を短く保つ。
 */
function syncUrl(): void {
  const p = new URLSearchParams();
  p.set('course', courseId);
  if (motorId !== 'mg996r') p.set('motor', motorId);
  if (motionMode !== 'scripted') {
    p.set('mode', motionMode);
    if (commandHeadingDeg !== 0) p.set('heading', String(commandHeadingDeg));
  }
  history.replaceState(null, '', `${location.pathname}?${p.toString()}`);
}

/** 起動時に URL クエリから状態を復元する（reload 前に呼ぶ。mode の可否は updateModeAvailability が後で担保）。 */
function applyUrlState(): void {
  const p = new URLSearchParams(location.search);
  const motor = p.get('motor');
  if (motor && (SELECTABLE_SERVO_IDS as readonly string[]).includes(motor)) {
    motorId = motor;
    selMotor.value = motor;
    torqueCapNm = getServo(motorId).stallNm;
    syncTorqueUI();
    syncMaterialsLink();
  }
  const course = p.get('course');
  if (course && COURSE_OPTIONS.some((c) => c.id === course)) {
    courseId = course as CourseId;
    selCourse.value = course;
  }
  const mode = p.get('mode');
  if (mode === 'rl' || mode === 'live') motionMode = mode;
  const heading = p.get('heading');
  if (heading !== null && Number.isFinite(Number(heading))) commandHeadingDeg = Number(heading);
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
  if (entry.reached !== undefined) {
    // 部屋ナビ: 到達と残距離を表示。
    const dist = (entry.goalDistM ?? 0) * 100;
    tunedInfo.textContent = `RL方策 ${cmdStr}${achStr}・${entry.reached ? 'ゴール到達✓' : `未到達（残${dist.toFixed(0)}cm）`}`;
  } else if (entry.baseForwardM !== undefined && entry.baseForwardM > 0) {
    const rlCm = (entry.forwardM ?? 0) * 100;
    const baseCm = entry.baseForwardM * 100;
    const pct = ((rlCm - baseCm) / baseCm) * 100;
    tunedInfo.textContent =
      `RL方策 ${cmdStr}${achStr}・基盤 ${baseCm.toFixed(0)}cm → RL ${rlCm.toFixed(0)}cm ` +
      `(${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%)${entry.fell ? '（破綻）' : ''}`;
  } else {
    tunedInfo.textContent = `RL方策 ${cmdStr}${achStr}・前進 ${((entry.forwardM ?? 0) * 100).toFixed(0)}cm${entry.fell ? '（破綻）' : ''}`;
  }
  buildParamSliders();
  const rl = recordedSnake3DReplay(
    record.replay as Snake3DReplay,
    getServo(motorId).name,
    torqueCapNm,
  );
  applyReplay(rl);
}

// ---- ライブ操舵: 方策をブラウザ内 SnakeEnv でリアルタイム駆動（録画再生ではない） ----
/** ライブ用の SnakeEnv を破棄する（モード離脱・コース/モーター変更時）。 */
function stopLive(): void {
  if (liveEnv) {
    liveEnv.dispose();
    liveEnv = null;
  }
  liveForward = null;
}

/** ライブのエピソードを初期化（現在の目標方位で reset し前進量の基準を取り直す）。 */
function resetLiveEpisode(): void {
  if (!liveEnv) return;
  liveObs = liveEnv.reset(undefined, commandHeadingDeg * DEG);
  liveAccum = 0;
  liveStartProj = liveEnv.progressMetric();
}

/** ライブの統計行（モード・目標方位・指令方向への前進）。 */
function liveStatsRows(): StatRow[] {
  const fwd = liveEnv ? (liveEnv.progressMetric() - liveStartProj) * 100 : 0;
  return [
    { label: 'モード', value: 'ライブ（リアルタイム駆動）' },
    {
      label: '目標方位（操舵）',
      value: `${commandHeadingDeg > 0 ? '+' : ''}${commandHeadingDeg}°`,
    },
    { label: '指令方向 前進', value: `${fwd.toFixed(1)} cm` },
  ];
}

/** 選択コース×汎用方策でライブ駆動を開始（重みを純JSフォワードへロードし env を用意）。 */
async function startLive(): Promise<void> {
  const stem = livePolicyStem();
  if (!stem) {
    motionMode = 'scripted';
    updateModeAvailability();
    await run();
    return;
  }
  const seq = ++loadSeq;
  renderStats([{ label: '状態', value: 'ライブ方策を初期化中…' }]);
  torqueCapNm = getServo(motorId).stallNm;
  syncTorqueUI();
  let weights: PolicyWeightsFile;
  try {
    weights = await loadPolicyWeights(stem);
  } catch (err) {
    if (seq !== loadSeq) return;
    motionMode = 'scripted';
    updateModeAvailability();
    tunedInfo.textContent = 'ライブ用の重みが見つからないため基盤歩容に戻しました';
    console.error(err);
    await run();
    return;
  }
  if (seq !== loadSeq) return;

  // 学習時と同じモーター設定（stiffness/damping は固定、cap はサーボの stall）で env を作る。
  const servo = getServo(motorId);
  const room = effectiveCourse() === 'room';
  const terrain = COURSES[effectiveCourse() as CourseId]();
  const motor = { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm };
  const cfg = room
    ? roomSnakeEnvConfig({ terrain, motor })
    : defaultSnakeEnvConfig({ terrain, motor });
  cfg.episodeSteps = 1800; // 終端まで走り、到達したら自動 reset でループ
  const env = await SnakeEnv.create(cfg);
  if (seq !== loadSeq) {
    env.dispose();
    return;
  }

  stopLive(); // 直前のライブ env があれば破棄してから差し替え
  liveEnv = env;
  liveForward = makePolicyForward(weights.weights, weights.obsDim, weights.actDim);
  resetLiveEpisode();

  // ビューを組む（再生 replay は使わないので null 化）。グリッドは進行範囲を固定で張る（部屋は部屋枠）。
  replay = null;
  view.showSnake3D();
  view.buildSnake3D(env.getLayout());
  view.setSnake3DTerrain(terrain);
  if (room) view.setSnake3DLiveGrid([ROOM.backX, ROOM.frontX], [-ROOM.halfY, ROOM.halfY]);
  else view.setSnake3DLiveGrid([-1, 13], [-2, 2]);
  setPlaying(true);
  tunedInfo.textContent = room
    ? 'ライブ: 部屋ナビ方策をブラウザ内で駆動。スライダーで目標方位へ即操舵（±90°）。'
    : 'ライブ: 汎用方策をブラウザ内でリアルタイム駆動。スライダーで即操舵。';
  buildParamSliders();
  renderStats(liveStatsRows());
}

/** ライブの制御を1ステップ進める（目標方位を反映→方策の平均行動→env.step、終端で自動 reset）。 */
function liveStepOnce(): void {
  if (!liveEnv || !liveForward) return;
  // スライダーは目標方位だけを与える。曲がる動き（yaw 曲率）は方策が操舵行動として出力する＝RL が獲得した操舵。
  liveEnv.setCommandHeading(commandHeadingDeg * DEG);
  const action = liveForward(liveObs);
  const r = liveEnv.step(action);
  liveObs = r.obs;
  if (r.done) resetLiveEpisode();
}

/** 実時間 dt ぶんだけライブ env を制御周期単位で進め、現在姿勢をビューへ適用する。 */
function stepLive(dt: number): void {
  if (!liveEnv || !liveForward) return;
  liveAccum += dt * speed;
  const period = liveEnv.controlPeriod;
  let n = 0;
  while (liveAccum >= period && n < MAX_LIVE_STEPS_PER_FRAME) {
    liveStepOnce();
    liveAccum -= period;
    n++;
  }
  if (liveAccum > period) liveAccum = period; // バックログを溜め込まない（重い時も実時間付近を保つ）
  view.applySnake3DFrame({ bodies: liveEnv.currentBodies() });
  if (++liveStatsThrottle >= 12) {
    liveStatsThrottle = 0;
    renderStats(liveStatsRows());
  }
}

// ---- 選択変更の共通経路: モード可否を更新し、scripted / RL / ライブ に応じて再構成 ----
async function reload(): Promise<void> {
  updateModeAvailability();
  stopLive(); // ライブ env は毎回作り直す（コース/モーター変更を反映）。live なら startLive が再生成する
  if (motionMode === 'rl') {
    await runRL();
    return;
  }
  if (motionMode === 'live') {
    await startLive();
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
btnLive.addEventListener('click', () => {
  if (motionMode === 'live' || btnLive.disabled) return;
  motionMode = 'live';
  void reload();
});
headingInput.addEventListener('input', () => {
  commandHeadingDeg = Number(headingInput.value);
  headingValue.textContent = `${commandHeadingDeg > 0 ? '+' : ''}${commandHeadingDeg}°`;
  // ライブは次ステップで env.setCommandHeading が読むので即操舵（再構成不要）。
  if (motionMode === 'rl') void runRL(); // RL は最近傍の録画方位を再生（fetch はキャッシュ）
  syncUrl();
});
torqueCapInput.addEventListener('input', () => {
  torqueCapNm = Number(torqueCapInput.value);
  torqueCapValue.textContent = `${torqueCapNm.toFixed(2)} N·m`;
  scheduleRun();
});
btnPlay.addEventListener('click', () => setPlaying(!playing));
btnReset.addEventListener('click', () => {
  if (motionMode === 'live') {
    resetLiveEpisode();
    setPlaying(true);
    renderStats(liveStatsRows());
    return;
  }
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

  if (motionMode === 'live' && liveEnv && liveForward) {
    if (playing) stepLive(dt);
  } else if (replay && replay.duration > 0 && playing) {
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
  applyUrlState(); // URL クエリ（?course=&motor=&mode=&heading=）から初期状態を復元してから描画。
  await reload();
}

void init().catch((err: unknown) => {
  renderStats([{ label: '状態', value: 'error', kind: 'bad' }]);
  console.error(err);
});
requestAnimationFrame(loop);
