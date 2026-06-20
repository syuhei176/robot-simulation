/**
 * 形状 → 必要モーターの逆算（静的保持トルク版）。
 *
 * 階段の一段を「次段へ前端を届かせる」姿勢でモデル化し、持ち上げ部の各関節が支える
 * 重力モーメントを計算する。2つの姿勢を評価:
 *   - リーチ円弧（効率的歩容）: lift-off を立てて被せる → 平常運用の保持トルク
 *   - 水平片持ち（最悪・上界）: 腕を水平に伸ばし切る → 設計荷重
 * 設計荷重に安全率を掛けて必要ストールトルクを出し、候補サーボへ当てる。さらに形状
 * （リンク数・総質量）を振って最小トルク形を探す。
 *
 *   実行: pnpm motor-sizing   (Node 型ストリップ)
 */
import { G, reachArc, staticTorques, horizontalCantilever } from '../src/sim3d/chain.ts';
import { getServo, pickServo, SELECTABLE_SERVO_IDS } from '../src/sim3d/servos.ts';
import { analyzeQuad, defaultPostures, defaultQuad } from '../src/sim3d/quadruped-static.ts';

const NM_PER_KGCM = 0.0980665; // 1 kg·cm = 0.0980665 N·m
const toKgcm = (nm: number) => nm / NM_PER_KGCM;

const DESIGN_SAFETY = 1.5; // 最悪姿勢（腕水平）を設計荷重とした上での安全率

interface Morphology {
  n: number; // リンク数
  totalLength: number; // 総長 [m]
  totalMass: number; // 総質量 [kg]（構造＋アクチュエータ＋電池＋積載）
}

interface Step {
  rise: number; // 蹴上げ [m]
  forward: number; // lift-off 点から次段着地点までの前方距離 [m]
}

interface Sizing {
  liftLinks: number;
  liftMass: number;
  reachable: boolean;
  tip: [number, number];
  arcPeak: number; // 円弧姿勢のピークトルク [N·m]（効率的歩容）
  arcPeakJoint: number;
  arcTau: number[];
  cantPeak: number; // 水平片持ち（上界）のピークトルク [N·m]（設計荷重）
}

/** 形状＋段から、必要な持ち上げ部とピークトルクを計算 */
function analyze(m: Morphology, step: Step): Sizing {
  const linkLen = m.totalLength / m.n;
  const linkMass = m.totalMass / m.n;
  const chord = Math.hypot(step.forward, step.rise);

  // 弦の 1.25 倍以上の連鎖長を確保できる最小リンク数を持ち上げる（緩い円弧で届かせる）
  let liftLinks = 1;
  while (liftLinks < m.n && liftLinks * linkLen < 1.25 * chord) liftLinks++;

  const lengths = new Array<number>(liftLinks).fill(linkLen);
  const masses = new Array<number>(liftLinks).fill(linkMass);

  const arc = reachArc(lengths, step.forward, step.rise);
  const arcT = staticTorques(lengths, masses, arc.absAngles);
  const cantT = staticTorques(lengths, masses, horizontalCantilever(liftLinks));

  return {
    liftLinks,
    liftMass: liftLinks * linkMass,
    reachable: arc.reachable,
    tip: arc.tip,
    arcPeak: arcT.peak,
    arcPeakJoint: arcT.peakJoint,
    arcTau: arcT.tau,
    cantPeak: cantT.peak,
  };
}

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

function tq(nm: number): string {
  return `${fmt(nm, 3)} N·m (${fmt(toKgcm(nm), 1)} kg·cm)`;
}

// ---- シナリオ設定 -------------------------------------------------------
const STEP: Step = { rise: 0.18, forward: 0.1 }; // 蹴上げ18cm・前方10cm（やや急な住宅階段）
const BASE: Morphology = { n: 8, totalLength: 0.9, totalMass: 1.0 }; // 90cm・1.0kg・8リンク

console.log('=== 形状→モーター逆算（静的保持トルク） ===');
console.log(
  `段: 蹴上げ=${STEP.rise * 100}cm  前方=${STEP.forward * 100}cm  弦=${fmt(Math.hypot(STEP.forward, STEP.rise) * 100, 1)}cm`,
);
console.log(
  `基準形状: ${BASE.n}リンク / 総長${BASE.totalLength * 100}cm / 総質量${BASE.totalMass}kg`,
);
console.log('');

// ---- 基準形状の詳細 -----------------------------------------------------
const s = analyze(BASE, STEP);
console.log('--- 基準形状の関節別 保持トルク（次段リーチ円弧姿勢） ---');
console.log(
  `持ち上げリンク数=${s.liftLinks}（先端側）  持ち上げ質量=${fmt(s.liftMass * 1000, 0)}g`,
);
console.log(
  `到達検証: tip=(${fmt(s.tip[0] * 100, 1)}, ${fmt(s.tip[1] * 100, 1)})cm  目標=(${STEP.forward * 100}, ${STEP.rise * 100})cm  reachable=${s.reachable}`,
);
s.arcTau.forEach((t, j) => {
  const name = j === 0 ? 'lift-off' : `+${j}`;
  const tag = j === s.arcPeakJoint ? '  ← この姿勢のピーク' : '';
  console.log(`  関節${j} (${name}): |τ|=${tq(Math.abs(t))}${tag}`);
});

console.log('');
console.log('--- 基準形状の3姿勢比較（ピークトルク） ---');
console.log('  平面（床に直線）           : ≈ 0 N·m（床が体重を支える）');
console.log(`  次段リーチ（円弧・効率歩容）: ${tq(s.arcPeak)}`);
console.log(`  水平片持ち（最悪・設計荷重）: ${tq(s.cantPeak)}`);

const required = s.cantPeak * DESIGN_SAFETY;
const servo = pickServo(required);
console.log('');
console.log('--- モーター逆算（基準形状） ---');
console.log(
  `設計荷重 = 最悪姿勢 ${fmt(s.cantPeak, 3)} × 安全率 ${DESIGN_SAFETY} = ${tq(required)}`,
);
if (servo) {
  console.log(
    `→ 最小で足りるサーボ: ${servo.name}（${fmt(servo.stallNm, 2)} N·m, ${servo.massG}g, ¥${servo.priceJpy}）`,
  );
  console.log(
    `→ ${BASE.n - 1}関節ぶん: サーボ質量 ${fmt((servo.massG * (BASE.n - 1)) / 1000, 2)}kg / 概算 ¥${(servo.priceJpy * (BASE.n - 1)).toLocaleString()}`,
  );
  console.log(
    `  （効率歩容では常用 ${fmt(s.arcPeak, 3)} N·m＝ストールの ${fmt((s.arcPeak / servo.stallNm) * 100, 0)}% 以下で動くので、発熱・消費とも軽い）`,
  );
} else {
  console.log('→ カタログ内に足りるサーボなし（より強力なサーボが必要）');
}

// ---- エネルギーは激安、の確認 ------------------------------------------
const ePerStep = BASE.totalMass * G * STEP.rise;
const flight = 13;
console.log('');
console.log('--- 登坂エネルギー（参考: トルクが問題でエネルギーは問題でない） ---');
console.log(
  `1段 = m·g·h = ${fmt(ePerStep, 2)} J   ${flight}段 = ${fmt(ePerStep * flight, 1)} J = ${fmt((ePerStep * flight) / 3600, 5)} Wh`,
);
console.log(
  '  → 登坂の仕事は微小。効くのは「保持トルク」と、休止中それを 0W にできるか（非バックドライブ/ブレーキ）。',
);

// ---- 形状スイープ：最小トルク形を探す ----------------------------------
console.log('');
console.log('=== 形状スイープ（総長0.9m固定、設計荷重=最悪姿勢×1.5でサーボ選定） ===');
console.log('   n 質量kg | 持上本数 持上g | 効率歩容πトルク | 最悪姿勢トルク | 最小サーボ');
const ns = [4, 6, 8, 12];
const massesKg = [0.6, 1.0, 1.8];
for (const mass of massesKg) {
  for (const n of ns) {
    const a = analyze({ n, totalLength: 0.9, totalMass: mass }, STEP);
    const sv = pickServo(a.cantPeak * DESIGN_SAFETY);
    const arcStr = `${fmt(toKgcm(a.arcPeak), 1)}kg·cm`;
    const cantStr = `${fmt(toKgcm(a.cantPeak), 1)}kg·cm`;
    const svStr = sv ? `${sv.name} ¥${sv.priceJpy}` : '足りるサーボなし';
    const reach = a.reachable ? '' : ' [届かず]';
    console.log(
      `  ${String(n).padStart(2)} ${fmt(mass, 1).padStart(5)} | ${String(a.liftLinks).padStart(6)} ${String(Math.round(a.liftMass * 1000)).padStart(5)} | ${arcStr.padStart(13)} | ${cantStr.padStart(12)} | ${svStr}${reach}`,
    );
  }
}

// ---- SG90 成立サイズ探索 ------------------------------------------------
const SG90 = getServo('sg90');

function sg90Row(label: string, m: Morphology, step: Step): void {
  const a = analyze(m, step);
  const design = a.cantPeak * DESIGN_SAFETY;
  const fits = design <= SG90.stallNm;
  const usagePct = (a.arcPeak / SG90.stallNm) * 100; // 効率歩容での常用負荷（ストール比）
  const verdict = fits
    ? `OK 常用${fmt(usagePct, 0)}%`
    : `NG (設計荷重${fmt(design / SG90.stallNm, 1)}xストール)`;
  const reach = a.reachable ? '' : ' [届かず]';
  console.log(
    `  ${label.padEnd(13)} | ${a.liftLinks}本/${String(Math.round(a.liftMass * 1000)).padStart(3)}g | π=${fmt(toKgcm(a.arcPeak), 2).padStart(5)} 最悪=${fmt(toKgcm(a.cantPeak), 2).padStart(5)}kg·cm | 設計=${fmt(toKgcm(design), 2).padStart(5)}kg·cm | ${verdict}${reach}`,
  );
}

console.log('');
console.log('=== SG90 成立サイズ探索（A: 実階段18cm固定で軽量化） ===');
console.log(`SG90 ストール=${tq(SG90.stallNm)} / 設計荷重がこれ以下なら OK`);
for (const mass of [1.0, 0.4, 0.25, 0.19, 0.15]) {
  sg90Row(`90cm ${fmt(mass, 2)}kg`, { n: 8, totalLength: 0.9, totalMass: mass }, STEP);
}
console.log(
  '  → 実階段(蹴上18cm)を残すと弦20.6cmが固定。90cm級では総質量≈0.19kg以下でないとSG90に乗らない。',
);

console.log('');
console.log('=== SG90 成立サイズ探索（B: 幾何相似ダウンスケール＝トイ階段, 質量∝s³・段も×s） ===');
console.log('  トルク∝s⁴で激減。ただし登るのは縮小した段（実18cm階段ではない）。');
const REF_LEN = 0.9;
const REF_MASS = 1.0;
for (const sc of [1.0, 0.8, 0.7, 0.66, 0.6, 0.5]) {
  const len = REF_LEN * sc;
  const mass = REF_MASS * sc ** 3;
  const step: Step = { rise: STEP.rise * sc, forward: STEP.forward * sc };
  sg90Row(
    `s=${fmt(sc, 2)} ${fmt(len * 100, 0)}cm`,
    { n: 8, totalLength: len, totalMass: mass },
    step,
  );
}

// ---- 機構×モーター比較（蛇 vs 四足, 静的設計荷重） --------------------------
const SELECTABLE = SELECTABLE_SERVO_IDS.map(getServo);

/** 設計荷重(最悪τ×1.5) に対し、4モーターが足りるか（✓/✗）を1行に。 */
function motorCells(worstNm: number): string {
  const design = worstNm * DESIGN_SAFETY;
  return SELECTABLE.map((sv) => (sv.stallNm >= design ? '  ✓  ' : '  ✗  ')).join('|');
}

function mechRow(mech: string, posture: string, worstNm: number, joints: number): void {
  console.log(
    `  ${mech.padEnd(4)} ${posture.padEnd(20)} | ${fmt(toKgcm(worstNm), 2).padStart(5)} | ${fmt(toKgcm(worstNm * DESIGN_SAFETY), 2).padStart(5)} |${motorCells(worstNm)}| ${String(joints).padStart(2)}`,
  );
}

console.log('');
console.log('=== 機構×モーター比較（静的: 設計荷重 = 最悪姿勢τ × 1.5, 1関節あたり） ===');
console.log(`総質量${BASE.totalMass}kg / 段 蹴上${STEP.rise * 100}cm・前方${STEP.forward * 100}cm`);
console.log(
  `  選択4モーター: ${SELECTABLE.map((s) => `${s.id.toUpperCase()}(${fmt(toKgcm(s.stallNm), 1)})`).join(' / ')} kg·cm`,
);
console.log(
  `  機構 姿勢                 | 最悪τ | 設計荷重|${SELECTABLE.map((s) => s.id.padStart(5).padEnd(5)).join('|')}| 関節`,
);

// 蛇: 基準形状の最悪（水平片持ち）。関節数 = リンク数-1。
mechRow('蛇', `階段保持(片持ち${s.liftLinks}links)`, s.cantPeak, BASE.n - 1);

// 四足: 同じ総質量・6モジュール級（4脚×1関節＋胴2）。3姿勢を評価。
const quad = analyzeQuad(defaultQuad(BASE.totalMass), defaultPostures(STEP.rise));
for (const leg of quad.legs) {
  mechRow('四足', `${leg.posture}`, leg.peakNm, quad.jointCount);
}
console.log(
  `  注: 四足は1脚1関節(6モジュール=4脚+胴2)・矢状面1脚モデル。支持体重=総質量÷支持脚数。`,
);
console.log(
  `  注: 蛇は持ち上げ部を片持ちで支えるため1関節が重い。四足は体重を脚で床へ預けるので立脚は軽いが、段差リーチで増える。`,
);

console.log('');
console.log(
  '注: これは静的保持トルクの一次見積り。動的ピーク（縁越えの瞬間加速）と接触は次段の Rapier で上乗せ評価する。',
);
console.log(
  '注: SG90はプラギア・FBなし・保持に常時電流。動的ピークは静的の約3倍（§6）なので、上の常用%に十分な余裕が要る。',
);
