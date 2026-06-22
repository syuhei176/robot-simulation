/**
 * TF.js 非依存の方策フォワード（決定論的な平均行動）。
 *
 * ダッシュボードの「ライブ」モードはブラウザ内で `SnakeEnv` をリアルタイムに 1 ステップずつ回し、
 * 各ステップの行動を方策の MLP で求める。学習側（`Policy`）は TF.js に依存するが、ブラウザには
 * TF.js をバンドルしない設計なので、ここでは保存済み重み（`Policy.exportWeights()` 形式）を読み、
 * 純 JS の前進計算で `actMean(obs)` 相当を再現する。
 *
 * 重みは `Policy` の MLP（dense×3: tanh → tanh → linear）を `getWeights()` 順に平坦化したもの:
 *   `policy = [W0, b0, W1, b1, W2, b2]`
 * TF.js の dense kernel は形状 `[in, out]` の row-major、つまり要素 (i,j) は `W[i*out + j]`。
 * よって 1 層は `out_j = b_j + Σ_i in_i · W[i*out + j]`、活性は最終層のみ linear で他は tanh。
 * 価値関数（value）と logStd は平均行動には不要なので無視する。
 */

/** `Policy.exportWeights()` が返す重み（policy 部分のみ使用）。 */
export interface PolicyForwardWeights {
  policy: number[][];
}

interface DenseLayer {
  W: Float64Array; // row-major [inDim, outDim]
  b: Float64Array; // [outDim]
  inDim: number;
  outDim: number;
  tanh: boolean; // 最終層は linear（false）、それ以外は tanh
}

/**
 * 保存済み重みから決定論的な平均行動 `forward(obs) -> action` を組み立てる。
 * 形状は `obsDim`/`actDim` と突き合わせて検証する（学習時と不一致なら例外）。
 */
export function makePolicyForward(
  weights: PolicyForwardWeights,
  obsDim: number,
  actDim: number,
): (obs: ArrayLike<number>) => Float32Array {
  const arrs = weights.policy;
  if (!Array.isArray(arrs) || arrs.length === 0 || arrs.length % 2 !== 0) {
    throw new Error('policy 重みは [W,b] の対で構成される必要があります');
  }
  const nLayers = arrs.length / 2;
  const layers: DenseLayer[] = [];
  let inDim = obsDim;
  for (let l = 0; l < nLayers; l++) {
    const W = arrs[2 * l];
    const b = arrs[2 * l + 1];
    const outDim = b.length;
    if (W.length !== inDim * outDim) {
      throw new Error(`層${l} の重み形状が不一致: ${W.length} != ${inDim}×${outDim}`);
    }
    layers.push({
      W: Float64Array.from(W),
      b: Float64Array.from(b),
      inDim,
      outDim,
      tanh: l < nLayers - 1,
    });
    inDim = outDim;
  }
  if (inDim !== actDim) {
    throw new Error(`最終出力次元が actDim と不一致: ${inDim} != ${actDim}`);
  }

  // 各層の出力バッファを使い回す（毎ステップ呼ばれるので確保を避ける）。
  const outBufs = layers.map((layer) => new Float64Array(layer.outDim));

  return (obs: ArrayLike<number>): Float32Array => {
    let cur: ArrayLike<number> = obs;
    for (let l = 0; l < layers.length; l++) {
      const { W, b, inDim: di, outDim: dj, tanh } = layers[l];
      const out = outBufs[l];
      for (let j = 0; j < dj; j++) {
        let s = b[j];
        for (let i = 0; i < di; i++) s += cur[i] * W[i * dj + j];
        out[j] = tanh ? Math.tanh(s) : s;
      }
      cur = out;
    }
    const result = new Float32Array(actDim);
    for (let j = 0; j < actDim; j++) result[j] = cur[j];
    return result;
  };
}
