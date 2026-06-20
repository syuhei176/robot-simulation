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

const NM_PER_KGCM = 0.0980665; // 1 kg·cm = 0.0980665 N·m
const toKgcm = (nm: number) => nm / NM_PER_KGCM;

interface Servo {
  name: string;
  stallNm: number;
  massG: number;
  priceJpy: number;
}

// 代表的サーボのストールトルク（おおよそ・定格電圧）
const SERVOS: Servo[] = [
  { name: 'Dynamixel XL330-M288', stallNm: 0.4, massG: 18, priceJpy: 3500 },
  { name: 'MG996R (6V)', stallNm: 1.0, massG: 55, priceJpy: 600 },
  { name: 'Dynamixel XL430-W250', stallNm: 1.5, massG: 57, priceJpy: 7000 },
  { name: 'Feetech STS3215 (12V)', stallNm: 2.94, massG: 60, priceJpy: 2500 },
  { name: 'Dynamixel XM430-W350', stallNm: 3.5, massG: 82, priceJpy: 25000 },
];

const DESIGN_SAFETY = 1.5; // 最悪姿勢（腕水平）を設計荷重とした上での安全率

function pickServo(requiredNm: number): Servo | null {
  return (
    SERVOS.filter((s) => s.stallNm >= requiredNm).sort((a, b) => a.stallNm - b.stallNm)[0] ?? null
  );
}

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

console.log('');
console.log(
  '注: これは静的保持トルクの一次見積り。動的ピーク（縁越えの瞬間加速）と接触は次段の Rapier で上乗せ評価する。',
);
