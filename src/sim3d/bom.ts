/**
 * BOM（材料表＝買い物リスト）の単一の真実。
 *
 * 「この蛇ロボを実際に作るには何を買えばいいか」を数値で出すための計算と部品カタログ。
 * 価格・寸法・必要トルクは既存モジュールから導き、重複定義しない:
 *   - サーボ価格/質量/位置帰還 … {@link servos.ts}
 *   - 蛇の構成（リンク数→関節数=サーボ個数・総質量・寸法）… {@link DEFAULT_SNAKE3D_CONFIG}
 *   - 必要トルク … {@link chain.ts} の静的保持トルク（姿勢だけで厳密に出る）
 * 蛇以外（制御基板・電源・フレーム等）の価格はここに「概算・前提」つきで1か所にまとめる。
 */
import { DEFAULT_SNAKE3D_CONFIG } from './snake3d-dynamics.ts';
import { staticTorques, horizontalCantilever } from './chain.ts';
import { getServo, SELECTABLE_SERVO_IDS, type Servo } from './servos.ts';

const CFG = DEFAULT_SNAKE3D_CONFIG;

// ---- 蛇の構成（単一の真実 = DEFAULT_SNAKE3D_CONFIG）-------------------------
/** リンク数（フレーム本数）。 */
export const SNAKE_LINK_COUNT = CFG.n; // 16
/** 関節数 = リンク数 − 1 ＝ 必要サーボ個数。 */
export const SNAKE_JOINT_COUNT = CFG.n - 1; // 15
/** 機体総質量 [kg]。 */
export const SNAKE_TOTAL_MASS_KG = CFG.totalMass; // 0.6

// ---- 必要トルク（静的保持トルクの一次見積り）-------------------------------
// 体の一部（約 1/4）を水平に持ち上げて支える「最悪姿勢」の保持トルク。段差越え・尺取りで
// 前方の数リンクを浮かせる瞬間が、平地走行では支配的な負荷になる。chain.ts の静力学を
// 再利用し、歩容に依らず姿勢だけで厳密に出す（効率的な歩容での常用はこれ以下＝ダッシュボードの
// 動的シミュで SCS0009 の τ上限内に収まることを確認済み）。
/** 持ち上げると仮定するリンク数（先端側・約 1/4）。 */
export const LIFT_LINK_COUNT = Math.ceil(CFG.n / 4);

function computeRequiredTorqueNm(): number {
  const linkMass = CFG.totalMass / CFG.n;
  const lengths = new Array<number>(LIFT_LINK_COUNT).fill(CFG.segLen);
  const masses = new Array<number>(LIFT_LINK_COUNT).fill(linkMass);
  return staticTorques(lengths, masses, horizontalCantilever(LIFT_LINK_COUNT)).peak;
}

/** 1関節あたりの必要保持トルク [N·m]（上記の最悪姿勢ピーク）。 */
export const REQUIRED_TORQUE_NM = computeRequiredTorqueNm();

// ---- サーボ推奨ロジック（買える5機種 = SELECTABLE_SERVO_IDS から選ぶ）-------
const SELECTABLE: Servo[] = SELECTABLE_SERVO_IDS.map(getServo);

const cheapestBy = (list: Servo[]): Servo =>
  list.reduce((a, b) => (b.priceJpy < a.priceJpy ? b : a));
const strongestBy = (list: Servo[]): Servo =>
  list.reduce((a, b) => (b.stallNm > a.stallNm ? b : a));

/** 3つの推奨枠（最安 / 推奨=標準 / 高性能）。すべて買える5機種の中から選ぶ。 */
export interface ServoPick {
  /** 必要トルクを満たす最安（位置帰還は問わない）。 */
  cheapest: Servo;
  /** 上記＋位置帰還を持つ最安。多関節の協調制御に位置帰還は必須なのでこれを標準に推す。 */
  recommended: Servo;
  /** 必要トルクを満たし位置帰還を持つ中で最高トルク（重い機体・将来拡張向け）。 */
  premium: Servo;
}

export function pickSnakeServos(): ServoPick {
  // 必要トルクを満たす機種のみ候補に（万一すべて不足する設定なら全機種から選ぶ）。
  const meeting = SELECTABLE.filter((s) => s.stallNm >= REQUIRED_TORQUE_NM);
  const pool = meeting.length > 0 ? meeting : SELECTABLE;
  const withFeedback = pool.filter((s) => s.feedback);
  const feedbackPool = withFeedback.length > 0 ? withFeedback : pool;
  return {
    cheapest: cheapestBy(pool),
    recommended: cheapestBy(feedbackPool),
    premium: strongestBy(feedbackPool),
  };
}

// ---- 購入リンク（ベンダー検索）---------------------------------------------
// 実在を確証できない特定商品URLは貼らず（リンク切れ・誤リンク防止）、部品名/型番をキーにした
// ベンダーの「検索結果」URLを生成する。型番が正確なサーボはこれで狙い撃ちできる。
export type VendorId = 'amazon' | 'aliexpress';

interface Vendor {
  label: string;
  url(query: string): string;
}

const VENDORS: Record<VendorId, Vendor> = {
  amazon: {
    label: 'Amazon',
    url: (q) => `https://www.amazon.co.jp/s?k=${encodeURIComponent(q)}`,
  },
  aliexpress: {
    label: 'AliExpress',
    url: (q) => `https://www.aliexpress.com/wholesale?SearchText=${encodeURIComponent(q)}`,
  },
};

export interface VendorLink {
  label: string;
  url: string;
}

export function vendorLinks(query: string, ids: readonly VendorId[]): VendorLink[] {
  return ids.map((id) => ({ label: VENDORS[id].label, url: VENDORS[id].url(query) }));
}

/** サーボの検索クエリ（"Feetech SCS0009 (6V)" → "Feetech SCS0009"。型番だけ残す）。 */
export function servoSearchQuery(servo: Servo): string {
  return servo.name.replace(/\s*\([^)]*\)\s*$/, '');
}

// ---- 蛇以外の部品（概算価格・前提つき・1か所にまとめる）----------------------
// 価格は 2026 年時点の国内通販の一般的な目安（概算）。サーボの接続方式（iface）で制御系が
// 変わるので、サーボ非依存の共通部品と、iface 依存の制御部品を分けて持つ。
export interface BomPart {
  name: string;
  qty: number;
  unitJpy: number;
  note: string;
  /** 購入リンクの検索キーワード（未指定なら name を使う）。 */
  query?: string;
  /** 購入リンクを出すベンダー（未指定なら Amazon のみ）。 */
  vendors?: VendorId[];
}

/** サーボ非依存の共通部品。フレーム本数はリンク数（単一の真実）から導く。 */
export const COMMON_PARTS: readonly BomPart[] = [
  {
    name: 'マイコン（ESP32 等）',
    qty: 1,
    unitJpy: 1500,
    note: '歩容生成・サーボ指令を出す制御基板（概算）',
    query: 'ESP32 開発ボード',
  },
  {
    name: 'バッテリ（LiPo 2S 7.4V・1000mAh級）',
    qty: 1,
    unitJpy: 1800,
    note: '電圧はサーボ定格に合わせ降圧して使う（概算）',
    query: 'LiPo 2S 7.4V バッテリー 1000mAh',
  },
  {
    name: '電源レギュレータ（UBEC 5–6V）',
    qty: 1,
    unitJpy: 800,
    note: 'サーボ電圧へ降圧。12V系サーボなら不要な構成も可（概算）',
    query: 'UBEC 5V 降圧 BEC',
  },
  {
    name: 'フレーム（3Dプリント・リンク）',
    qty: SNAKE_LINK_COUNT,
    unitJpy: 150,
    note: `リンク${SNAKE_LINK_COUNT}本分。PLA フィラメント費の概算（1本≈150円）`,
    query: 'PLA フィラメント 1kg',
  },
  {
    name: '配線・コネクタ一式',
    qty: 1,
    unitJpy: 1200,
    note: 'サーボ間ケーブル・電源配線（概算）',
    query: 'サーボ 延長ケーブル コネクタ セット',
  },
  {
    name: 'ネジ・ベアリング類',
    qty: 1,
    unitJpy: 1000,
    note: '関節軸・固定具一式（概算）',
    query: 'M3 ネジ ベアリング セット',
  },
];

/** サーボの接続方式に応じて1つ選ぶ制御部品（シリアルバス変換 or PWM ドライバ）。 */
export const DRIVER_BY_IFACE: Record<Servo['iface'], BomPart> = {
  'serial-bus': {
    name: 'シリアルバス変換（Feetech URT-1）',
    qty: 1,
    unitJpy: 1500,
    note: 'SCS/STS をデイジーチェーン接続。多関節を2本線で制御（概算）',
    query: 'Feetech URT-1',
    vendors: ['amazon', 'aliexpress'],
  },
  PWM: {
    name: 'PWM サーボドライバ（PCA9685 16ch）',
    qty: 1,
    unitJpy: 600,
    note: `16ch。関節${SNAKE_JOINT_COUNT}個を1枚で駆動（概算）`,
    query: 'PCA9685 16ch サーボ ドライバ',
  },
};

// ---- BOM の組み立て ---------------------------------------------------------
/** 買い物リストの1行（小計込み）。 */
export interface BomLine extends BomPart {
  subtotalJpy: number;
}

/** 選択サーボ1機種ぶんの完成 BOM。 */
export interface Bom {
  servo: Servo;
  lines: BomLine[];
  totalJpy: number;
}

const toLine = (part: BomPart): BomLine => ({ ...part, subtotalJpy: part.qty * part.unitJpy });

/** 選択サーボ（id）に対する買い物リスト全体を組み立てる。 */
export function buildBom(servoId: string): Bom {
  const servo = getServo(servoId);
  const servoPart: BomPart = {
    name: `サーボ: ${servo.name}`,
    qty: SNAKE_JOINT_COUNT,
    unitJpy: servo.priceJpy,
    note: `関節${SNAKE_JOINT_COUNT}個ぶん・${servo.feedback ? '位置帰還あり' : '位置帰還なし'}／${servo.iface}`,
  };
  const lines = [servoPart, DRIVER_BY_IFACE[servo.iface], ...COMMON_PARTS].map(toLine);
  const totalJpy = lines.reduce((sum, l) => sum + l.subtotalJpy, 0);
  return { servo, lines, totalJpy };
}

/** サーボ比較表の1行（買える5機種ぶん）。 */
export interface ServoComparisonRow {
  servo: Servo;
  meetsTorque: boolean;
  /** 余裕倍率 = ストールトルク ÷ 必要トルク。 */
  marginX: number;
  /** 関節数ぶんのサーボ小計 [JPY]。 */
  servoSubtotalJpy: number;
  /** 推奨枠（最安 / 推奨 / 高性能）。該当しなければ null。 */
  role: keyof ServoPick | null;
}

/** 買える5機種を必要トルク・価格・位置帰還で並べた比較データ。 */
export function servoComparison(): ServoComparisonRow[] {
  const pick = pickSnakeServos();
  return SELECTABLE.map((servo) => ({
    servo,
    meetsTorque: servo.stallNm >= REQUIRED_TORQUE_NM,
    marginX: servo.stallNm / REQUIRED_TORQUE_NM,
    servoSubtotalJpy: servo.priceJpy * SNAKE_JOINT_COUNT,
    role:
      servo.id === pick.recommended.id
        ? 'recommended'
        : servo.id === pick.cheapest.id
          ? 'cheapest'
          : servo.id === pick.premium.id
            ? 'premium'
            : null,
  }));
}

/** 最安 / 推奨 / 高性能 の3案で合計金額がどう変わるかの比較。 */
export interface BomComparisonRow {
  label: string;
  role: keyof ServoPick;
  servo: Servo;
  totalJpy: number;
}

export function bomComparison(): BomComparisonRow[] {
  const pick = pickSnakeServos();
  const rows: Array<{ label: string; role: keyof ServoPick }> = [
    { label: '最安', role: 'cheapest' },
    { label: '推奨（標準）', role: 'recommended' },
    { label: '高性能', role: 'premium' },
  ];
  return rows.map(({ label, role }) => ({
    label,
    role,
    servo: pick[role],
    totalJpy: buildBom(pick[role].id).totalJpy,
  }));
}
