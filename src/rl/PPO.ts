import * as tf from '@tensorflow/tfjs';
import { Policy } from './Policy.ts';
import type { RLEnv } from './RLEnv.ts';

export interface PPOConfig {
  rolloutSteps: number;
  gamma: number;
  lambda: number;
  clip: number;
  lr: number;
  epochs: number;
  minibatch: number;
  vfCoef: number;
  entCoef: number;
}

export const DEFAULT_PPO: PPOConfig = {
  rolloutSteps: 2048,
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  lr: 3e-4,
  epochs: 4,
  minibatch: 256,
  vfCoef: 0.5,
  entCoef: 0.005,
};

export interface IterationStats {
  iteration: number;
  meanEpisodeReturn: number;
  meanEpisodeForward: number;
  policyLoss: number;
  valueLoss: number;
  std: number;
}

/** Proximal Policy Optimization（クリップ版）。ブラウザで回せる小規模実装。 */
export class PPO {
  readonly policy: Policy;
  private readonly cfg: PPOConfig;
  private readonly optimizer: tf.Optimizer;
  private readonly obsDim: number;
  private readonly actDim: number;
  private iteration = 0;

  // ロールアウトバッファ
  private curObs: Float32Array | null = null;
  private epReturn = 0;
  private epStartX = 0;

  constructor(policy: Policy, cfg: PPOConfig = DEFAULT_PPO) {
    this.policy = policy;
    this.cfg = cfg;
    this.obsDim = policy.obsDim;
    this.actDim = policy.actDim;
    this.optimizer = tf.train.adam(cfg.lr);
  }

  /** 1 イテレーション: ロールアウト収集 → GAE → PPO 更新。 */
  runIteration(env: RLEnv): IterationStats {
    const T = this.cfg.rolloutSteps;
    const obsBuf: number[][] = [];
    const actBuf: number[][] = [];
    const logpBuf: number[] = [];
    const rewBuf: number[] = [];
    const valBuf: number[] = [];
    const doneBuf: number[] = [];

    const episodeReturns: number[] = [];
    const episodeForwards: number[] = [];

    if (!this.curObs) {
      this.curObs = env.reset();
      this.epReturn = 0;
      this.epStartX = env.progressMetric();
    }

    // --- ロールアウト収集 ---
    for (let t = 0; t < T; t++) {
      const obs = this.curObs;
      const { action, logProb, value } = this.policy.act(obs);
      const { obs: nextObs, reward, done } = env.step(action);

      obsBuf.push(Array.from(obs));
      actBuf.push(Array.from(action));
      logpBuf.push(logProb);
      rewBuf.push(reward);
      valBuf.push(value);
      doneBuf.push(done ? 1 : 0);

      this.epReturn += reward;
      this.curObs = nextObs;

      if (done) {
        episodeReturns.push(this.epReturn);
        episodeForwards.push(env.progressMetric() - this.epStartX);
        this.curObs = env.reset();
        this.epReturn = 0;
        this.epStartX = env.progressMetric();
      }
    }

    // 末尾がエピソード途中なら最終状態価値でブートストラップ
    const bootstrap = this.policy.act(this.curObs).value;

    const { advantages, returns } = this.computeGAE(rewBuf, valBuf, doneBuf, bootstrap);

    const { policyLoss, valueLoss } = this.update(obsBuf, actBuf, logpBuf, advantages, returns);

    this.iteration++;
    const mean = (a: number[]): number => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
    return {
      iteration: this.iteration,
      meanEpisodeReturn: mean(episodeReturns),
      meanEpisodeForward: mean(episodeForwards),
      policyLoss,
      valueLoss,
      std: this.policy.meanStd(),
    };
  }

  private computeGAE(
    rew: number[],
    val: number[],
    done: number[],
    bootstrap: number,
  ): { advantages: number[]; returns: number[] } {
    const T = rew.length;
    const advantages = new Array<number>(T).fill(0);
    const returns = new Array<number>(T).fill(0);
    const { gamma, lambda } = this.cfg;
    let gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nextValue = t === T - 1 ? bootstrap : val[t + 1];
      const nonTerminal = 1 - done[t];
      const delta = rew[t] + gamma * nextValue * nonTerminal - val[t];
      gae = delta + gamma * lambda * nonTerminal * gae;
      advantages[t] = gae;
      returns[t] = gae + val[t];
    }
    return { advantages, returns };
  }

  private update(
    obs: number[][],
    act: number[][],
    oldLogp: number[],
    advantages: number[],
    returns: number[],
  ): { policyLoss: number; valueLoss: number } {
    const T = obs.length;

    // アドバンテージ正規化
    const aMean = advantages.reduce((s, x) => s + x, 0) / T;
    const aStd = Math.sqrt(advantages.reduce((s, x) => s + (x - aMean) ** 2, 0) / T) + 1e-8;
    const advNorm = advantages.map((a) => (a - aMean) / aStd);

    const obsT = tf.tensor2d(obs, [T, this.obsDim]);
    const actT = tf.tensor2d(act, [T, this.actDim]);
    const logpT = tf.tensor1d(oldLogp);
    const advT = tf.tensor1d(advNorm);
    const retT = tf.tensor1d(returns);

    const idx = Array.from({ length: T }, (_, i) => i);
    let lastPolicyLoss = 0;
    let lastValueLoss = 0;

    for (let epoch = 0; epoch < this.cfg.epochs; epoch++) {
      shuffle(idx);
      for (let start = 0; start < T; start += this.cfg.minibatch) {
        const mbIdx = idx.slice(start, start + this.cfg.minibatch);
        const idxT = tf.tensor1d(mbIdx, 'int32');

        const oMB = obsT.gather(idxT) as tf.Tensor2D;
        const aMB = actT.gather(idxT) as tf.Tensor2D;
        const lpMB = logpT.gather(idxT) as tf.Tensor1D;
        const advMB = advT.gather(idxT) as tf.Tensor1D;
        const retMB = retT.gather(idxT) as tf.Tensor1D;

        const { plScalar, vlScalar } = this.gradientStep(oMB, aMB, lpMB, advMB, retMB);
        lastPolicyLoss = plScalar;
        lastValueLoss = vlScalar;

        idxT.dispose();
        oMB.dispose();
        aMB.dispose();
        lpMB.dispose();
        advMB.dispose();
        retMB.dispose();
      }
    }

    obsT.dispose();
    actT.dispose();
    logpT.dispose();
    advT.dispose();
    retT.dispose();

    return { policyLoss: lastPolicyLoss, valueLoss: lastValueLoss };
  }

  /** 1 ミニバッチの勾配ステップ。損失値(JS数値)を返す。 */
  private gradientStep(
    obs: tf.Tensor2D,
    act: tf.Tensor2D,
    oldLogp: tf.Tensor1D,
    adv: tf.Tensor1D,
    ret: tf.Tensor1D,
  ): { plScalar: number; vlScalar: number } {
    let pl = 0;
    let vl = 0;
    const cost = this.optimizer.minimize(() => {
      const { logProb, value, entropy } = this.policy.evaluate(obs, act);
      const ratio = tf.exp(logProb.sub(oldLogp));
      const surr1 = ratio.mul(adv);
      const surr2 = ratio.clipByValue(1 - this.cfg.clip, 1 + this.cfg.clip).mul(adv);
      const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1) as tf.Scalar;
      const valueLoss = value.sub(ret).square().mean() as tf.Scalar;
      const entLoss = entropy.mul(-1) as tf.Scalar;
      pl = policyLoss.dataSync()[0];
      vl = valueLoss.dataSync()[0];
      return policyLoss
        .add(valueLoss.mul(this.cfg.vfCoef))
        .add(entLoss.mul(this.cfg.entCoef)) as tf.Scalar;
    }, true);
    cost?.dispose();
    return { plScalar: pl, vlScalar: vl };
  }

  dispose(): void {
    this.optimizer.dispose();
  }
}

function shuffle(a: number[]): void {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
}
