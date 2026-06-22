import * as tf from '@tensorflow/tfjs';

const LOG_2PI = Math.log(2 * Math.PI);

export interface ActResult {
  /** サンプルした raw 行動（squash 前） */
  action: Float32Array;
  /** その行動の対数尤度 */
  logProb: number;
  /** 状態価値の推定 */
  value: number;
}

/**
 * 連続行動のガウス方策 + 状態価値関数。いずれも小さな MLP。
 *  - 方策: obs -> 平均 mean(actDim)、分散は状態に依らない学習可能 logStd(actDim)
 *  - 価値: obs -> V(s)
 * PPO の更新では evaluate() を minimize 内で呼び、勾配を流す。
 */
export class Policy {
  readonly obsDim: number;
  readonly actDim: number;
  private readonly policyNet: tf.Sequential;
  private readonly valueNet: tf.Sequential;
  private readonly logStd: tf.Variable;

  constructor(obsDim: number, actDim: number, hidden = 64, logStdInit = -0.5) {
    this.obsDim = obsDim;
    this.actDim = actDim;

    const mlp = (outUnits: number, outActivation: 'linear'): tf.Sequential => {
      const net = tf.sequential();
      net.add(tf.layers.dense({ inputShape: [obsDim], units: hidden, activation: 'tanh' }));
      net.add(tf.layers.dense({ units: hidden, activation: 'tanh' }));
      net.add(tf.layers.dense({ units: outUnits, activation: outActivation }));
      return net;
    };

    this.policyNet = mlp(actDim, 'linear');
    this.valueNet = mlp(1, 'linear');
    this.logStd = tf.variable(tf.fill([actDim], logStdInit), true, 'logStd');
  }

  /** 1 状態から行動をサンプル（学習時のロールアウト収集用）。 */
  act(obs: Float32Array): ActResult {
    return tf.tidy(() => {
      const x = tf.tensor2d(obs, [1, this.obsDim]);
      const mean = this.policyNet.apply(x) as tf.Tensor2D;
      const std = tf.exp(this.logStd);
      const noise = tf.randomNormal([1, this.actDim]);
      const action = mean.add(noise.mul(std)) as tf.Tensor2D;
      const logProb = this.gaussianLogProb(mean, action);
      const value = (this.valueNet.apply(x) as tf.Tensor2D).reshape([1]);
      return {
        action: action.dataSync() as Float32Array,
        logProb: logProb.dataSync()[0],
        value: value.dataSync()[0],
      };
    });
  }

  /** 決定論的な平均行動（学習済み方策の再生用）。 */
  actMean(obs: Float32Array): Float32Array {
    return tf.tidy(() => {
      const x = tf.tensor2d(obs, [1, this.obsDim]);
      const mean = this.policyNet.apply(x) as tf.Tensor2D;
      return mean.dataSync() as Float32Array;
    });
  }

  /** バッチに対する logπ(a|s)、V(s)、エントロピーを返す（minimize 内で使用）。 */
  evaluate(
    obs: tf.Tensor2D,
    act: tf.Tensor2D,
  ): { logProb: tf.Tensor1D; value: tf.Tensor1D; entropy: tf.Scalar } {
    const mean = this.policyNet.apply(obs) as tf.Tensor2D;
    const logProb = this.gaussianLogProb(mean, act);
    const value = (this.valueNet.apply(obs) as tf.Tensor2D).reshape([-1]) as tf.Tensor1D;
    const entropy = this.logStd.sum().add(0.5 * this.actDim * (LOG_2PI + 1)) as tf.Scalar;
    return { logProb, value, entropy };
  }

  /** 対角ガウスの対数尤度（次元方向に総和）。 */
  private gaussianLogProb(mean: tf.Tensor2D, action: tf.Tensor2D): tf.Tensor1D {
    return tf.tidy(() => {
      const logStd = this.logStd;
      const variance = tf.exp(logStd.mul(2));
      const diff = action.sub(mean);
      const term = diff.square().div(variance).add(logStd.mul(2)).add(LOG_2PI);
      return term.sum(1).mul(-0.5) as tf.Tensor1D;
    });
  }

  /** Node 用: 重みを素の配列で取り出す（ファイル保存用。localStorage を使わない）。 */
  exportWeights(): { policy: number[][]; value: number[][]; logStd: number[] } {
    const dump = (net: tf.Sequential): number[][] =>
      net.getWeights().map((w) => Array.from(w.dataSync() as Float32Array));
    return {
      policy: dump(this.policyNet),
      value: dump(this.valueNet),
      logStd: Array.from(this.logStd.dataSync()),
    };
  }

  /** exportWeights の逆。同一構成（obsDim/actDim/hidden）の Policy へ重みを流し込む。 */
  importWeights(data: { policy: number[][]; value: number[][]; logStd: number[] }): void {
    const apply = (net: tf.Sequential, arrs: number[][]): void =>
      net.setWeights(net.getWeights().map((w, i) => tf.tensor(arrs[i], w.shape)));
    apply(this.policyNet, data.policy);
    apply(this.valueNet, data.value);
    this.logStd.assign(tf.tensor1d(data.logStd));
  }

  async save(key: string): Promise<void> {
    await this.policyNet.save(`localstorage://${key}-policy`);
    await this.valueNet.save(`localstorage://${key}-value`);
    localStorage.setItem(`${key}-logstd`, JSON.stringify(Array.from(this.logStd.dataSync())));
  }

  async load(key: string): Promise<boolean> {
    const raw = localStorage.getItem(`${key}-logstd`);
    if (!raw) return false;
    const p = (await tf.loadLayersModel(`localstorage://${key}-policy`)) as tf.Sequential;
    const v = (await tf.loadLayersModel(`localstorage://${key}-value`)) as tf.Sequential;
    this.policyNet.setWeights(p.getWeights());
    this.valueNet.setWeights(v.getWeights());
    this.logStd.assign(tf.tensor1d(JSON.parse(raw) as number[]));
    p.dispose();
    v.dispose();
    return true;
  }

  /** 現在の平均 std（探索量の目安、ログ表示用）。 */
  meanStd(): number {
    return tf.tidy(() => tf.exp(this.logStd).mean().dataSync()[0]);
  }

  /**
   * 探索 std（の対数）を外から設定する。学習を通して std をアニーリング（縮小）し、
   * 決定論的な平均行動に性能を担わせる（mean が good 領域へ寄り、det 評価が確率的性能に追従する）。
   */
  setLogStd(value: number): void {
    this.logStd.assign(tf.fill([this.actDim], value));
  }

  dispose(): void {
    this.policyNet.dispose();
    this.valueNet.dispose();
    this.logStd.dispose();
  }
}
