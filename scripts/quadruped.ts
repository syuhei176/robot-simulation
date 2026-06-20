/**
 * 四足 3D 動的歩行（Rapier）の実行レポート。
 *   実行: pnpm quadruped
 * 既定歩容で各モーター（torque cap）を当て、前進距離・転倒・トルク飽和を出す。
 */
import { runQuadrupedGait } from '../src/sim3d/quadruped-dynamics.ts';
import { getServo, SELECTABLE_SERVO_IDS, SERVOS } from '../src/sim3d/servos.ts';
import { defaultQuad, defaultPostures, analyzeQuad } from '../src/sim3d/quadruped-static.ts';

const NM_PER_KGCM = 0.0980665;
const SAFETY = 1.5;

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

console.log('=== 四足 3D 動的歩行（Rapier・IKクロール・忠実トルク） ===');
console.log('  形状: 胴24×16×5cm/0.52kg + 4脚(thigh9+shin9cm, 各85g) = 1.2kg / μ=0.9');
console.log(
  '  制御: 足先カートシアン軌道→2リンクIK→PDトルク ±cap ハードclamp＋受動ダンピング、横方向は安定化',
);
console.log('  モーター         | cap N·m | 前進cm | 転倒 | 飽和% | maxτ要求 | 最大傾き° | 成功');

for (const id of SELECTABLE_SERVO_IDS) {
  const servo = getServo(id);
  const replay = await runQuadrupedGait({ motor: { maxTorqueNm: servo.stallNm } }, 0);
  const s = replay.summary;
  const satPct = (s.saturatedSteps / Math.ceil(s.config.duration / s.config.dt)) * 100;
  console.log(
    `  ${id.toUpperCase().padEnd(8)} ${fmt(servo.stallNm / NM_PER_KGCM, 1).padStart(5)}kg·cm | ${fmt(servo.stallNm, 2).padStart(5)} | ${fmt(s.forwardDistanceM * 100, 1).padStart(5)} | ${(s.fell ? `${fmt(s.fellTime ?? 0, 1)}s` : 'なし').padStart(4)} | ${fmt(satPct, 0).padStart(4)} | ${fmt(s.maxDemandNm, 2).padStart(6)} | ${fmt(s.maxTiltDeg, 0).padStart(6)} | ${s.success}`,
  );
}

console.log('');
console.log(
  '注: IKクロール＋忠実トルク。cap が保持・推進に足りないと前進せず失速する＝cap→前進距離が単調（飽和%も強capほど低下）。',
);
console.log(
  '注: 横方向は安定化（脚がpitchのみで横バランス不能のため別機構前提）。重さを上げると弱capから失速する（重さ応答, §6.11）。',
);

// ===== 静的サーボ選定（軽量化スイープ）: 平地歩行の律速 = 踏み出し(3脚支持) の保持トルク =====
// 保持トルクは総質量に線形（payload=mg/3）なので、軽量化はそのまま必要トルクを下げる。
const stepPostures = defaultPostures().filter((p) => p.name.startsWith('踏み出し'));
const tauAt1kg = analyzeQuad(defaultQuad(1.0), stepPostures).worst.peakNm; // τ@1.0kg

console.log('');
console.log(
  '=== 四足 静的サーボ選定: 軽量化スイープ（律速=踏み出し3脚支持, 設計荷重=最悪×1.5） ===',
);
console.log('  総質量 | 踏み出しτ | 設計荷重 | 満たす最安サーボ(価格)');
for (const totalMass of [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]) {
  const worst = analyzeQuad(defaultQuad(totalMass), stepPostures).worst.peakNm;
  const design = worst * SAFETY;
  const cheapest = SERVOS.filter((s) => s.stallNm >= design).sort(
    (a, b) => a.priceJpy - b.priceJpy,
  )[0];
  console.log(
    `  ${fmt(totalMass, 1)}kg | ${fmt(worst / NM_PER_KGCM, 1).padStart(6)}kg·cm | ${fmt(design / NM_PER_KGCM, 1).padStart(6)}kg·cm | ${cheapest ? `${cheapest.name} ¥${cheapest.priceJpy}` : '該当なし'}`,
  );
}

console.log('');
console.log('  各サーボが満たせる上限総質量（設計荷重 ≤ ストール → M ≤ stall / (τ@1kg × 1.5)）:');
for (const id of SELECTABLE_SERVO_IDS) {
  const servo = getServo(id);
  const maxMass = servo.stallNm / (tauAt1kg * SAFETY);
  console.log(
    `    ${servo.name.padEnd(22)} ${fmt(servo.stallNm / NM_PER_KGCM, 1).padStart(4)}kg·cm → 総質量 ≤ ${fmt(maxMass, 2)}kg`,
  );
}
console.log('');
console.log(
  '  注: 静的一次見積り。1.0kg級なら MG996R が正解サイズ。SG90 で歩かせるには総質量≈0.47kg以下が要る。',
);
