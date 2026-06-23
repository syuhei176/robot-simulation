/**
 * 材料表（買い物リスト）ページの描画。
 *
 * 計算・部品カタログは {@link bom.ts}（単一の真実）に閉じ込め、ここは DOM 生成に徹する。
 * サーボ選択（ドロップダウン）を変えると買い物リストの小計・合計が即時更新される。
 * 価格・寸法・必要トルクは bom.ts 経由で servos.ts / snake3d-dynamics.ts / chain.ts から来る。
 */
import { getServo, SELECTABLE_SERVO_IDS } from './sim3d/servos.ts';
import {
  buildBom,
  bomComparison,
  pickSnakeServos,
  purchaseLinks,
  servoComparison,
  servoSearchQuery,
  vendorLinks,
  LIFT_LINK_COUNT,
  REQUIRED_TORQUE_NM,
  SNAKE_JOINT_COUNT,
  SNAKE_LINK_COUNT,
  SNAKE_TOTAL_MASS_KG,
  type ServoPick,
  type VendorLink,
} from './sim3d/bom.ts';

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
}

const NM_PER_KGCM = 0.0980665;
const yen = (n: number): string => `¥${Math.round(n).toLocaleString('ja-JP')}`;
const nm = (n: number): string => `${n.toFixed(2)} N·m`;
const kgcm = (n: number): string => `${(n / NM_PER_KGCM).toFixed(1)} kg·cm`;
const yesNo = (b: boolean): string => (b ? 'あり' : 'なし');

const ROLE_LABEL: Record<keyof ServoPick, string> = {
  cheapest: '最安',
  recommended: '推奨',
  premium: '高性能',
};

// ---- 小さな DOM ヘルパ ------------------------------------------------------
type Cell = { text: string; cls?: string; html?: HTMLElement };
type Row = { cells: Cell[]; cls?: string };

function cellNode(tag: 'td' | 'th', cell: Cell): HTMLTableCellElement {
  const node = document.createElement(tag);
  if (cell.cls) node.className = cell.cls;
  if (cell.html) node.append(cell.html);
  else node.textContent = cell.text;
  return node;
}

function table(headers: Cell[], rows: Row[]): HTMLTableElement {
  const t = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headers.forEach((h) => headRow.append(cellNode('th', h)));
  thead.append(headRow);
  const tbody = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.cls) tr.className = row.cls;
    row.cells.forEach((c) => tr.append(cellNode('td', c)));
    tbody.append(tr);
  }
  t.append(thead, tbody);
  return t;
}

function badge(role: keyof ServoPick): HTMLSpanElement {
  const span = document.createElement('span');
  span.className = `badge ${role}`;
  span.textContent = ROLE_LABEL[role];
  return span;
}

/** 購入リンク（ベンダー検索）を " / " 区切りで並べた span を作る。新規タブで開く。 */
function linksSpan(links: VendorLink[]): HTMLSpanElement {
  const span = document.createElement('span');
  span.className = 'buy';
  links.forEach((link, i) => {
    if (i > 0) span.append(document.createTextNode(' / '));
    const a = document.createElement('a');
    a.href = link.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = link.label;
    span.append(a);
  });
  return span;
}

// ---- ① 前提（蛇の構成）-----------------------------------------------------
function renderSpec(): void {
  const cards: Array<{ k: string; v: string; u?: string }> = [
    { k: 'リンク（体節）', v: String(SNAKE_LINK_COUNT), u: '本' },
    { k: '関節 ＝ サーボ個数', v: String(SNAKE_JOINT_COUNT), u: '個' },
    { k: '機体総質量', v: SNAKE_TOTAL_MASS_KG.toFixed(2), u: 'kg' },
    {
      k: `必要トルク（${LIFT_LINK_COUNT}リンク持上）`,
      v: REQUIRED_TORQUE_NM.toFixed(2),
      u: `N·m / ${kgcm(REQUIRED_TORQUE_NM)}`,
    },
  ];
  const grid = document.createElement('div');
  grid.className = 'spec-grid';
  for (const c of cards) {
    const card = document.createElement('div');
    card.className = 'spec-card';
    const k = document.createElement('div');
    k.className = 'k';
    k.textContent = c.k;
    const v = document.createElement('div');
    v.className = 'v';
    v.textContent = c.v;
    if (c.u) {
      const u = document.createElement('span');
      u.className = 'u';
      u.textContent = ` ${c.u}`;
      v.append(u);
    }
    card.append(k, v);
    grid.append(card);
  }
  el('spec').replaceChildren(grid);
}

// ---- ② サーボ選択 + 合計カード ---------------------------------------------
function renderPicker(selectedId: string): void {
  const picker = el('picker');
  picker.replaceChildren();

  const field = document.createElement('label');
  field.className = 'field';
  field.append(document.createTextNode('サーボ機種'));
  const select = document.createElement('select');
  for (const id of SELECTABLE_SERVO_IDS) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = getServo(id).name;
    select.append(opt);
  }
  select.value = selectedId;
  select.addEventListener('change', () => render(select.value));
  field.append(select);

  const totalCard = document.createElement('div');
  totalCard.className = 'total-card';
  const k = document.createElement('div');
  k.className = 'k';
  k.textContent = '概算合計（サーボ＋制御＋電源＋構造）';
  const v = document.createElement('div');
  v.className = 'v';
  v.textContent = yen(buildBom(selectedId).totalJpy);
  totalCard.append(k, v);

  picker.append(field, totalCard);
}

// ---- ② 買い物リスト表 ------------------------------------------------------
function renderBom(selectedId: string): void {
  const bom = buildBom(selectedId);
  const rows: Row[] = bom.lines.map((line) => ({
    cells: [
      { text: line.name },
      { text: String(line.qty), cls: 'num' },
      { text: yen(line.unitJpy), cls: 'num' },
      { text: yen(line.subtotalJpy), cls: 'num' },
      { text: line.note, cls: 'note' },
      { text: '', cls: 'buy-cell', html: linksSpan(purchaseLinks(line)) },
    ],
  }));
  rows.push({
    cls: 'total-row',
    cells: [
      { text: '合計' },
      { text: '', cls: 'num' },
      { text: '', cls: 'num' },
      { text: yen(bom.totalJpy), cls: 'num' },
      { text: '', cls: 'note' },
      { text: '' },
    ],
  });
  const t = table(
    [
      { text: '部品' },
      { text: '個数', cls: 'num' },
      { text: '単価', cls: 'num' },
      { text: '小計', cls: 'num' },
      { text: '備考' },
      { text: '購入' },
    ],
    rows,
  );
  el('bom').replaceChildren(t);
}

// ---- ③ サーボ比較表 + 推奨理由 ---------------------------------------------
function renderServoCompare(selectedId: string): void {
  const rows = servoComparison().map((r) => {
    const name = document.createElement('span');
    name.textContent = r.servo.name;
    if (r.role) name.append(badge(r.role));
    return {
      cls: r.servo.id === selectedId ? 'recommended' : undefined,
      cells: [
        { text: '', html: name },
        { text: `${nm(r.servo.stallNm)} (${kgcm(r.servo.stallNm)})`, cls: 'num' },
        { text: `×${r.marginX.toFixed(1)}`, cls: `num ${r.meetsTorque ? 'ok' : 'ng'}` },
        { text: `${r.servo.massG} g`, cls: 'num' },
        { text: yesNo(r.servo.feedback), cls: r.servo.feedback ? 'ok' : 'note' },
        { text: yen(r.servo.priceJpy), cls: 'num' },
        { text: yen(r.servoSubtotalJpy), cls: 'num' },
        {
          text: '',
          cls: 'buy-cell',
          html: linksSpan(vendorLinks(servoSearchQuery(r.servo), ['amazon', 'aliexpress'])),
        },
      ],
    };
  });
  const t = table(
    [
      { text: 'サーボ' },
      { text: 'ストールτ', cls: 'num' },
      { text: '余裕', cls: 'num' },
      { text: '質量', cls: 'num' },
      { text: '位置帰還' },
      { text: '単価', cls: 'num' },
      { text: `小計(×${SNAKE_JOINT_COUNT})`, cls: 'num' },
      { text: '購入' },
    ],
    rows,
  );
  el('servo-compare').replaceChildren(t);

  const pick = pickSnakeServos();
  const why = el('why-recommended');
  why.replaceChildren();
  const strong = document.createElement('b');
  strong.textContent = `推奨は ${pick.recommended.name}。`;
  why.append(
    strong,
    document.createTextNode(
      ` 多関節（${SNAKE_JOINT_COUNT}個）を滑らかに協調させるには各サーボの現在角がわかる位置帰還（シリアルバス＋エンコーダ）が要る。` +
        `必要トルクを満たす最安は ${pick.cheapest.name} だが位置帰還が無くオープンループになる。` +
        `${pick.recommended.name} は同等トルク＋位置帰還を持ち、安価なまま協調制御に向く（${pick.recommended.note ?? ''}）。` +
        `より重い機体や将来拡張には高トルク＋位置帰還の ${pick.premium.name} を。`,
    ),
  );
}

// ---- ④ サーボ選択による合計比較 --------------------------------------------
function renderTotalCompare(selectedId: string): void {
  const rows = bomComparison().map((c) => {
    const name = document.createElement('span');
    name.append(badge(c.role), document.createTextNode(` ${c.servo.name}`));
    return {
      cls: c.servo.id === selectedId ? 'recommended' : undefined,
      cells: [
        { text: c.label },
        { text: '', html: name },
        {
          text: c.servo.feedback ? '位置帰還あり' : '位置帰還なし',
          cls: c.servo.feedback ? 'ok' : 'note',
        },
        { text: yen(c.servo.priceJpy * SNAKE_JOINT_COUNT), cls: 'num' },
        { text: yen(c.totalJpy), cls: 'num' },
      ],
    };
  });
  const t = table(
    [
      { text: '案' },
      { text: 'サーボ' },
      { text: '位置帰還' },
      { text: `サーボ小計(×${SNAKE_JOINT_COUNT})`, cls: 'num' },
      { text: '概算合計', cls: 'num' },
    ],
    rows,
  );
  el('total-compare').replaceChildren(t);
}

// ---- 注記 -------------------------------------------------------------------
function renderNotes(): void {
  const notes = [
    `必要トルクは「体の先端 ${LIFT_LINK_COUNT}リンク（約 1/4）を水平に持ち上げて支える最悪姿勢」の静的保持トルク（chain.ts）。効率的な歩容での常用はこれ以下。`,
    'サーボの価格・トルク・質量・位置帰還は servos.ts（単一の真実）、リンク/関節数・総質量は snake3d-dynamics.ts の DEFAULT_SNAKE3D_CONFIG から導出（重複定義なし）。',
    '制御部品はサーボの接続方式で変わる: シリアルバス（SCS/STS）は URT-1、PWM（SG90/MG90S/MG996R）は PCA9685。',
    '蛇以外の部品（マイコン・電源・フレーム・配線・ネジ類）の価格は 2026 年時点の国内通販の一般的な目安＝概算で、購入先・為替・数量で変動する。',
    '「購入」リンクは特定商品ページではなく、型番・部品名での Amazon／AliExpress の検索結果を開く（リンク切れ・誤リンクを避けるため）。実際の価格・在庫はリンク先で確認のこと。Feetech 系サーボや URT-1 は AliExpress が安い場合が多い。',
  ];
  const ul = el('notes');
  ul.replaceChildren();
  for (const text of notes) {
    const li = document.createElement('li');
    li.textContent = text;
    ul.append(li);
  }
}

// ---- 全体描画（サーボ選択ごとに再描画）-------------------------------------
function render(selectedId: string): void {
  renderPicker(selectedId);
  renderBom(selectedId);
  renderServoCompare(selectedId);
  renderTotalCompare(selectedId);
}

/** URL の ?motor= を検証して初期サーボを決める（無効/未指定なら推奨）。 */
function initialServoId(): string {
  const param = new URLSearchParams(window.location.search).get('motor');
  const selectable = SELECTABLE_SERVO_IDS as readonly string[];
  if (param && selectable.includes(param)) return param;
  return pickSnakeServos().recommended.id;
}

renderSpec();
renderNotes();
render(initialServoId());
