/**
 * 矢状面（x: 前後・水平, z: 上下）内の多リンク連鎖の静力学。
 *
 * 階段登りでモーターに要求される支配項は「持ち上げた胴体を重力に抗して保持する
 * トルク」。これは歩容コントローラ無しで、与えた姿勢（各リンクの絶対角）から
 * 厳密に計算できる。ここでは:
 *   - reachArc(): 持ち上げ部を「次段へ届く円弧姿勢」に配置する決定論的IK
 *   - staticTorques(): その姿勢で各関節が支える重力モーメント [N·m]
 * を提供する。回転軸は紙面に垂直（y軸）なので問題は 2D に落ちる。
 */

export const G = 9.80665;

/** 円弧到達IKの結果。absAngles[i] は水平からのリンク i の絶対角 [rad]。 */
export interface ReachResult {
  absAngles: number[];
  reachable: boolean;
  /** 実際に到達した先端座標 [x, z]（検証用） */
  tip: [number, number];
}

/** 各関節の保持トルク。tau[j] は関節 j（リンク j の近位端）が支える重力モーメント [N·m]。 */
export interface TorqueResult {
  tau: number[];
  /** 絶対値の最大トルクと、その関節インデックス */
  peak: number;
  peakJoint: number;
}

/** 各リンクの絶対角から先端座標を計算 */
function tipOf(lengths: number[], absAngles: number[]): [number, number] {
  let x = 0;
  let z = 0;
  for (let i = 0; i < lengths.length; i++) {
    x += lengths[i] * Math.cos(absAngles[i]);
    z += lengths[i] * Math.sin(absAngles[i]);
  }
  return [x, z];
}

/**
 * 長さ lengths の連鎖を原点 (0,0) から目標 (targetX, targetZ) まで届かせる円弧姿勢を返す。
 * 円弧長 = Σlengths とし、弦長 c=|target| との比から半角 α を解く（α/sinα は単調増加）。
 * 持ち上げ部は「上に立ててから前へ被せる」低トルク戦略になるよう、開始接線を弦より立てる。
 */
export function reachArc(lengths: number[], targetX: number, targetZ: number): ReachResult {
  const s = lengths.reduce((a, b) => a + b, 0);
  const c = Math.hypot(targetX, targetZ);
  const phiC = Math.atan2(targetZ, targetX);
  const k = lengths.length;

  // 弦が連鎖長を超える → 届かない
  if (c > s || k === 0) {
    const fallback = new Array<number>(k).fill(phiC);
    return { absAngles: fallback, reachable: false, tip: tipOf(lengths, fallback) };
  }

  // s/c = α/sinα を二分法で解く（α ∈ (0, π)）
  const ratio = s / c;
  if (ratio <= 1 + 1e-9) {
    const straight = new Array<number>(k).fill(phiC);
    return { absAngles: straight, reachable: true, tip: tipOf(lengths, straight) };
  }
  let lo = 1e-6;
  let hi = Math.PI - 1e-6;
  for (let it = 0; it < 80; it++) {
    const mid = 0.5 * (lo + hi);
    if (mid / Math.sin(mid) < ratio) lo = mid;
    else hi = mid;
  }
  const alpha = 0.5 * (lo + hi);

  // 弦を中心に ±α 振る。開始を立てる（+α）→ 終端（-α）へ。
  const absAngles: number[] = [];
  for (let i = 0; i < k; i++) {
    absAngles.push(phiC + alpha - (2 * alpha * (i + 0.5)) / k);
  }
  return { absAngles, reachable: true, tip: tipOf(lengths, absAngles) };
}

/**
 * 持ち上げ部（lift-off 点で固定された片持ち連鎖）の各関節保持トルク。
 * 関節 j のトルク = リンク j..k-1 の重力が、リンク j の近位端まわりに作るモーメント。
 * τ_j = -g Σ_{i>=j} m_i (comX_i - proxX_j)。水平片持ちでは j=0 が最大だが、円弧姿勢
 * では lift-off が立つためピークは中間関節へ移る（peakJoint で判定すること）。
 */
export function staticTorques(
  lengths: number[],
  masses: number[],
  absAngles: number[],
  g: number = G,
): TorqueResult {
  const k = lengths.length;
  const proxX = new Array<number>(k);
  const comX = new Array<number>(k);
  let x = 0;
  for (let i = 0; i < k; i++) {
    proxX[i] = x;
    comX[i] = x + 0.5 * lengths[i] * Math.cos(absAngles[i]);
    x += lengths[i] * Math.cos(absAngles[i]);
  }

  const tau = new Array<number>(k);
  for (let j = 0; j < k; j++) {
    let m = 0;
    for (let i = j; i < k; i++) {
      m += -g * masses[i] * (comX[i] - proxX[j]);
    }
    tau[j] = m;
  }

  let peak = 0;
  let peakJoint = 0;
  for (let j = 0; j < k; j++) {
    if (Math.abs(tau[j]) > peak) {
      peak = Math.abs(tau[j]);
      peakJoint = j;
    }
  }
  return { tau, peak, peakJoint };
}

/** 水平片持ち（最悪姿勢・上界）の絶対角 = 全リンク水平 */
export function horizontalCantilever(k: number): number[] {
  return new Array<number>(k).fill(0);
}
