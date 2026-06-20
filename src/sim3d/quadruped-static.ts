/**
 * 四足（quadruped）の静的保持トルク。蛇と同じモジュール群を脚に組み替えた前提で、
 * 「脚 = 先端に支持体重を載せた片持ち連鎖」として hip 関節の保持トルクを出す。
 *
 * 蛇の階段保持トルク（chain.ts の cantilever/arc）と同じ物理（重力モーメント）で比較できる。
 * 脚は「絶対角 legAbsAngle で全伸長した直線」とし、足先に支持体重を
 * 「長さ0・質量 payload のリンク」として追加して staticTorques に通す（点荷重を厳密に含む）。
 *
 * 既知の限界:
 *   - 矢状面の1脚モデル。横方向(ロール)の支持や動的な踏み込み衝撃は別。
 *   - 体重シェアは「支持脚数で等分」の近似（立脚=mg/4, 3脚支持=mg/3）。
 *   - 姿勢は代表点（立脚/踏み出し/段差リーチ）の上界見積り。
 *   - 脚は全伸長の直線（膝で内側点へ畳む効果は見ない＝保持トルクの上界寄り）。
 */
import { staticTorques, G } from './chain.ts';

export type Mechanism = 'snake' | 'quad';

export interface QuadConfig {
  totalMass: number; // ロボット総質量 [kg]
  modules: number; // 同一モジュール総数（脚＋胴）
  legCount: number; // 接地脚の本数（通常4）
  modulesPerLeg: number; // 1脚あたりのモジュール数（=リンク数）
  legLength: number; // 1脚の伸長長 hip→足先 [m]
}

export interface QuadPosture {
  name: string;
  legAbsAngleDeg: number; // 脚の絶対角（水平=0, 真下=-90, 上向き=正）
  supportLegs: number; // この姿勢で体重を支える脚数（少ないほど1脚の荷重大）
}

export interface QuadLegResult {
  posture: string;
  peakNm: number; // 1脚の関節保持トルクのピーク [N·m]
  peakJoint: number;
  payloadKg: number; // この姿勢で1脚に載る支持体重 [kg]
}

/** 蛇と公平に比べるための既定の四足構成（同じ総質量・6モジュール級）。 */
export function defaultQuad(totalMass: number, modules = 6): QuadConfig {
  return {
    totalMass,
    modules,
    legCount: 4,
    modulesPerLeg: 1, // 6モジュール=4脚(各1)＋胴2 を想定
    legLength: 0.18, // 18cmの脚（90cm蛇より小型）
  };
}

/** 代表3姿勢（立脚・踏み出し・段差リーチ）。段差リーチは脚を前上方へ伸ばす＝水平寄り。 */
export function defaultPostures(stepRise = 0.18): QuadPosture[] {
  return [
    { name: '立脚(4脚)', legAbsAngleDeg: -80, supportLegs: 4 },
    { name: '踏み出し(3脚支持)', legAbsAngleDeg: -70, supportLegs: 3 },
    // 段差リーチ: 脚を次段へ前上方に伸ばし、ほぼ水平で mg/3 を支える最悪寄りの姿勢
    { name: `段差リーチ(rise${Math.round(stepRise * 100)}cm)`, legAbsAngleDeg: 10, supportLegs: 3 },
  ];
}

/** 1脚・1姿勢の保持トルク。足先に支持体重(payload)を点荷重として載せる。 */
export function analyzeQuadLeg(
  cfg: QuadConfig,
  posture: QuadPosture,
  g: number = G,
): QuadLegResult {
  const legLinks = Math.max(1, cfg.modulesPerLeg);
  const linkLen = cfg.legLength / legLinks;
  const legMass = (cfg.totalMass / cfg.modules) * cfg.modulesPerLeg; // 1脚の自重
  const lengths = new Array<number>(legLinks).fill(linkLen);
  const masses = new Array<number>(legLinks).fill(legMass / legLinks);

  // 直線脚: 全リンクが同じ絶対角を向く（hip から足先まで一直線）
  const angle = (posture.legAbsAngleDeg * Math.PI) / 180;
  const absAngles = new Array<number>(legLinks).fill(angle);

  // 支持体重シェア = 総質量 ÷ 支持脚数（脚自重は別途リンク質量で計上）
  const payloadKg = cfg.totalMass / posture.supportLegs;

  // 足先に「長さ0・質量 payload」のリンクを足して点荷重を厳密に含める
  const ext = staticTorques([...lengths, 0], [...masses, payloadKg], [...absAngles, 0], g);

  return {
    posture: posture.name,
    peakNm: ext.peak,
    peakJoint: ext.peakJoint,
    payloadKg,
  };
}

/** 全姿勢を評価し、最悪（ピーク最大）姿勢を返す。 */
export function analyzeQuad(
  cfg: QuadConfig,
  postures: QuadPosture[],
  g: number = G,
): { legs: QuadLegResult[]; worst: QuadLegResult; jointCount: number } {
  const legs = postures.map((p) => analyzeQuadLeg(cfg, p, g));
  const worst = legs.reduce((a, b) => (b.peakNm > a.peakNm ? b : a));
  const jointCount = cfg.legCount * cfg.modulesPerLeg; // 駆動関節の総数
  return { legs, worst, jointCount };
}
