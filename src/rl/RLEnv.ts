/**
 * 強化学習環境の最小契約。PPO はこのインターフェースだけに依存し、2D 蛇（MicrobotEnv）と
 * 3D 四足（QuadEnv）の両方を同じ学習器で回せるようにする。
 */
export interface StepResult {
  obs: Float32Array;
  reward: number;
  done: boolean;
}

export interface RLEnv {
  readonly obsDim: number;
  readonly actDim: number;
  /** エピソードを初期化して初期観測を返す。 */
  reset(): Float32Array;
  /** raw 行動で 1 制御ステップ進める。 */
  step(action: ArrayLike<number>): StepResult;
  /** ロギング用の進捗指標（前進 x [m] 等）。エピソード前進量の集計に使う。 */
  progressMetric(): number;
}
