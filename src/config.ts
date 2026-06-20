/**
 * チューニング可能なパラメータ群。
 * 物理は XY 平面の 2D で解き、描画時に world(x, z=y) へマップする（y_world は高さ固定）。
 */
export interface SimParams {
  /** リンク（マイクロボット節）の数。ノード数は links + 1 */
  links: number;
  /** ノード間の自然長 [m] */
  restLength: number;
  /** ノード質量 [kg] */
  nodeMass: number;
  /** 1 フレームの時間刻み [s] */
  dt: number;
  /** 1 フレームあたりの物理サブステップ数（安定性のため分割） */
  substeps: number;
  /** 体軸方向（前後）の粘性摩擦係数。小さいほど滑りやすい */
  frictionTangential: number;
  /** 体軸と垂直方向（横）の粘性摩擦係数。大きいほど横滑りしない＝うねりが推進力になる */
  frictionNormal: number;
  /** 関節アクチュエータの剛性（目標角へ戻す強さ） */
  bendStiffness: number;
  /** 関節アクチュエータのダンピング */
  bendDamping: number;
  /** 全体の線形ダンピング（数値安定用にごく弱く） */
  linearDamping: number;
  /** リンク長拘束（PBD）の反復回数 */
  constraintIterations: number;
}

export const DEFAULT_SIM: SimParams = {
  links: 12,
  restLength: 0.5,
  nodeMass: 1,
  dt: 1 / 60,
  substeps: 8,
  frictionTangential: 0.4,
  frictionNormal: 24,
  bendStiffness: 120,
  bendDamping: 8,
  linearDamping: 0.01,
  constraintIterations: 6,
};

/** CPG（serpenoid 波）のパラメータ。Phase 1 のベースライン制御に使う。 */
export interface GaitParams {
  /** 関節振幅 [rad] */
  amplitude: number;
  /** 振動周波数 [Hz] */
  frequency: number;
  /** 隣接関節間の位相差 [rad]（進行波を作る） */
  phaseLag: number;
  /** 旋回用の角度オフセット [rad] */
  turnBias: number;
}

export const DEFAULT_GAIT: GaitParams = {
  amplitude: 0.7,
  frequency: 0.7,
  phaseLag: 0.9,
  turnBias: 0,
};
