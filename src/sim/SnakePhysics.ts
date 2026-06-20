import type { SimParams } from '../config.ts';

/**
 * マイクロボット連結チェーンの平面ロコモーションモデル（運動学的・抵抗力ベース）。
 *
 * 各リンクの相対関節角 φ_j を「形状」として外部から与え（serpenoid 波）、
 * 摩擦支配（準静的）の力・トルク釣り合いから胴体全体の剛体運動 (ẋ, ẏ, θ̇) を解く。
 * これは Hirose らの蛇型ロボット標準モデルで、力ベースの動的積分と違い
 * カオス的にならず、滑らかで決定論的・前進が振幅/周波数に単調に応答する。
 *
 *  - 異方性摩擦: 体軸方向 cT、横方向 cN（cN >> cT）。比 cN/cT が推進を生む。
 *  - φ_j は setJointTargets() で与え、φ̇ は step() で差分から推定。
 */
export class SnakePhysics {
  readonly links: number;
  readonly nodeCount: number;
  readonly jointCount: number;
  /** レンダリング用ノード位置 [x0,y0, x1,y1, ...]（長さ 2*nodeCount） */
  readonly pos: Float64Array;

  private readonly p: SimParams;
  private readonly L: number;
  private x = 0;
  private y = 0;
  private theta0 = 0;
  private readonly phi: Float64Array; // 関節角（形状）
  private readonly prevPhi: Float64Array;

  // step 内で再利用する一時配列（リンク単位）
  private readonly ux: Float64Array;
  private readonly uy: Float64Array;
  private readonly cx: Float64Array;
  private readonly cy: Float64Array;
  private readonly ax: Float64Array;
  private readonly ay: Float64Array;
  private readonly bx: Float64Array;
  private readonly by: Float64Array;
  private readonly psiDot: Float64Array;
  private lastComVel: [number, number] = [0, 0];

  constructor(params: SimParams) {
    this.p = params;
    this.L = params.restLength;
    this.links = params.links;
    this.nodeCount = params.links + 1;
    this.jointCount = Math.max(0, this.links - 1);
    this.pos = new Float64Array(this.nodeCount * 2);
    this.phi = new Float64Array(this.jointCount);
    this.prevPhi = new Float64Array(this.jointCount);

    const n = this.links;
    this.ux = new Float64Array(n);
    this.uy = new Float64Array(n);
    this.cx = new Float64Array(n);
    this.cy = new Float64Array(n);
    this.ax = new Float64Array(n);
    this.ay = new Float64Array(n);
    this.bx = new Float64Array(n);
    this.by = new Float64Array(n);
    this.psiDot = new Float64Array(n);
    this.reset();
  }

  reset(): void {
    this.x = 0;
    this.y = 0;
    this.theta0 = 0;
    this.phi.fill(0);
    this.prevPhi.fill(0);
    this.lastComVel = [0, 0];
    this.computeGeometry(0, false);
    this.writeNodes();
  }

  setJointTargets(targets: ArrayLike<number>): void {
    for (let j = 0; j < this.jointCount; j++) this.phi[j] = targets[j] ?? 0;
  }

  /** dt 秒進める。形状(φ)の時間変化から胴体の剛体運動を解いて積分する。 */
  step(dt: number): void {
    const comBefore = this.centerOfMass();

    // φ̇（形状速度）→ ψ̇、幾何（û, c, A, B）を構築
    this.computePsiDot(dt);
    this.computeGeometry(this.theta0, true);

    // 摩擦の力・トルク釣り合い: wrench(q) = M q + d = 0 を解く
    const d = this.wrench(0, 0, 0);
    const c0 = this.wrench(1, 0, 0);
    const c1 = this.wrench(0, 1, 0);
    const c2 = this.wrench(0, 0, 1);
    const M = [
      [c0[0] - d[0], c1[0] - d[0], c2[0] - d[0]],
      [c0[1] - d[1], c1[1] - d[1], c2[1] - d[1]],
      [c0[2] - d[2], c1[2] - d[2], c2[2] - d[2]],
    ];
    const q = solve3(M, [-d[0], -d[1], -d[2]]);

    // 剛体運動を積分
    this.x += q[0] * dt;
    this.y += q[1] * dt;
    this.theta0 += q[2] * dt;

    for (let j = 0; j < this.jointCount; j++) this.prevPhi[j] = this.phi[j];

    // 新しい基準姿勢で位置を更新
    this.computeGeometry(this.theta0, false);
    this.writeNodes();

    const comAfter = this.centerOfMass();
    this.lastComVel = [(comAfter[0] - comBefore[0]) / dt, (comAfter[1] - comBefore[1]) / dt];
  }

  /** φ̇ を差分から計算し ψ̇_i = Σ_{k<=i} φ̇_{k-1} を埋める。 */
  private computePsiDot(dt: number): void {
    let acc = 0;
    this.psiDot[0] = 0;
    for (let i = 1; i < this.links; i++) {
      acc += (this.phi[i - 1] - this.prevPhi[i - 1]) / dt;
      this.psiDot[i] = acc;
    }
  }

  /**
   * θ0 と現在の φ から幾何を構築する。
   *  - û_i: リンク向き、c_i: リンク中心
   *  - A_i: θ̇0 係数、B_i: ψ̇ による既知速度成分（includeB のとき）
   * n_i = perp(û_i) = (-uy, ux)。
   */
  private computeGeometry(theta0: number, includeB: boolean): void {
    const n = this.links;
    const L = this.L;

    let theta = theta0;
    for (let i = 0; i < n; i++) {
      if (i > 0) theta += this.phi[i - 1];
      this.ux[i] = Math.cos(theta);
      this.uy[i] = Math.sin(theta);
    }

    this.cx[0] = this.x;
    this.cy[0] = this.y;
    for (let i = 1; i < n; i++) {
      this.cx[i] = this.cx[i - 1] + 0.5 * L * (this.ux[i - 1] + this.ux[i]);
      this.cy[i] = this.cy[i - 1] + 0.5 * L * (this.uy[i - 1] + this.uy[i]);
    }

    this.ax[0] = 0;
    this.ay[0] = 0;
    this.bx[0] = 0;
    this.by[0] = 0;
    for (let i = 1; i < n; i++) {
      const npx0 = -this.uy[i - 1];
      const npy0 = this.ux[i - 1];
      const npx1 = -this.uy[i];
      const npy1 = this.ux[i];
      this.ax[i] = this.ax[i - 1] + 0.5 * L * (npx0 + npx1);
      this.ay[i] = this.ay[i - 1] + 0.5 * L * (npy0 + npy1);
      const sd0 = includeB ? this.psiDot[i - 1] : 0;
      const sd1 = includeB ? this.psiDot[i] : 0;
      this.bx[i] = this.bx[i - 1] + 0.5 * L * (sd0 * npx0 + sd1 * npx1);
      this.by[i] = this.by[i - 1] + 0.5 * L * (sd0 * npy0 + sd1 * npy1);
    }
  }

  /** 与えた剛体速度 q=(vx,vy,w) での全リンク摩擦の総和 [Fx, Fy, Tz]。 */
  private wrench(vx: number, vy: number, w: number): [number, number, number] {
    const { frictionTangential: cT, frictionNormal: cN } = this.p;
    const cRot = (cN * this.L * this.L) / 12;
    const n = this.links;
    let Fx = 0;
    let Fy = 0;
    let Tz = 0;
    for (let i = 0; i < n; i++) {
      const vix = vx + w * this.ax[i] + this.bx[i];
      const viy = vy + w * this.ay[i] + this.by[i];
      const tx = this.ux[i];
      const ty = this.uy[i];
      const nx = -ty;
      const ny = tx;
      const vt = vix * tx + viy * ty;
      const vn = vix * nx + viy * ny;
      const fx = -cT * vt * tx - cN * vn * nx;
      const fy = -cT * vt * ty - cN * vn * ny;
      Fx += fx;
      Fy += fy;
      Tz += this.cx[i] * fy - this.cy[i] * fx;
      Tz += -cRot * (w + this.psiDot[i]);
    }
    return [Fx, Fy, Tz];
  }

  /** リンク中心 c と向き û からノード位置を復元（描画用）。 */
  private writeNodes(): void {
    const L = this.L;
    const n = this.links;
    // node_0 = c_0 - (L/2) û_0
    this.pos[0] = this.cx[0] - 0.5 * L * this.ux[0];
    this.pos[1] = this.cy[0] - 0.5 * L * this.uy[0];
    for (let i = 0; i < n; i++) {
      this.pos[(i + 1) * 2] = this.cx[i] + 0.5 * L * this.ux[i];
      this.pos[(i + 1) * 2 + 1] = this.cy[i] + 0.5 * L * this.uy[i];
    }
  }

  centerOfMass(out: [number, number] = [0, 0]): [number, number] {
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < this.nodeCount; i++) {
      sx += this.pos[i * 2];
      sy += this.pos[i * 2 + 1];
    }
    out[0] = sx / this.nodeCount;
    out[1] = sy / this.nodeCount;
    return out;
  }

  centerOfMassVelocity(out: [number, number] = [0, 0]): [number, number] {
    out[0] = this.lastComVel[0];
    out[1] = this.lastComVel[1];
    return out;
  }

  /** 体の向き（尾→頭の平均接線）の角度 [rad]。 */
  heading(): number {
    let hx = 0;
    let hy = 0;
    for (let i = 0; i < this.nodeCount - 1; i++) {
      hx += this.pos[i * 2] - this.pos[(i + 1) * 2];
      hy += this.pos[i * 2 + 1] - this.pos[(i + 1) * 2 + 1];
    }
    return Math.atan2(hy, hx);
  }
}

/** 3x3 線形方程式 A x = b を Cramer の公式で解く。 */
function solve3(A: number[][], b: number[]): [number, number, number] {
  const det = det3(A);
  if (Math.abs(det) < 1e-12) return [0, 0, 0];
  const ax = det3([
    [b[0], A[0][1], A[0][2]],
    [b[1], A[1][1], A[1][2]],
    [b[2], A[2][1], A[2][2]],
  ]);
  const ay = det3([
    [A[0][0], b[0], A[0][2]],
    [A[1][0], b[1], A[1][2]],
    [A[2][0], b[2], A[2][2]],
  ]);
  const aw = det3([
    [A[0][0], A[0][1], b[0]],
    [A[1][0], A[1][1], b[1]],
    [A[2][0], A[2][1], b[2]],
  ]);
  return [ax / det, ay / det, aw / det];
}

function det3(m: number[][]): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}
