import type { GaitParams } from '../config.ts';

/**
 * Serpenoid 波（中枢パターン生成器, CPG）による手書き歩容。
 * Phase 1 で物理モデルが正しく前進するか検証するためのベースライン制御。
 *
 *   target_j = amplitude * sin(2π f t + j * phaseLag) + turnBias
 */
export class CPGController {
  readonly jointCount: number;
  gait: GaitParams;
  private t = 0;
  private readonly targets: Float64Array;

  constructor(jointCount: number, gait: GaitParams) {
    this.jointCount = jointCount;
    this.gait = gait;
    this.targets = new Float64Array(jointCount);
  }

  reset(): void {
    this.t = 0;
  }

  /** dt 進めて目標関節角を返す。 */
  update(dt: number): Float64Array {
    this.t += dt;
    const { amplitude, frequency, phaseLag, turnBias } = this.gait;
    const w = 2 * Math.PI * frequency;
    for (let j = 0; j < this.jointCount; j++) {
      this.targets[j] = amplitude * Math.sin(w * this.t + j * phaseLag) + turnBias;
    }
    return this.targets;
  }
}
