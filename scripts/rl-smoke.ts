/**
 * RL スタック（Env + Policy + PPO）のヘッドレス煙テスト。
 * 小さな設定で数イテレーション回し、例外/NaN が無いこと、報酬が算出されることを確認する。
 *   実行: pnpm rl-smoke
 */
import * as tf from '@tensorflow/tfjs';
import { MicrobotEnv, DEFAULT_ENV } from '../src/env/MicrobotEnv.ts';
import { Policy } from '../src/rl/Policy.ts';
import { PPO, DEFAULT_PPO } from '../src/rl/PPO.ts';

await tf.setBackend('cpu');
await tf.ready();
console.log('tf backend:', tf.getBackend());

const env = new MicrobotEnv({ ...DEFAULT_ENV, episodeSteps: 120 });
const policy = new Policy(env.obsDim, env.actDim, 32);
const ppo = new PPO(policy, {
  ...DEFAULT_PPO,
  rolloutSteps: 512,
  minibatch: 128,
  epochs: 3,
});

const before = tf.memory().numTensors;
for (let i = 0; i < 12; i++) {
  const s = ppo.runIteration(env);
  console.log(
    `iter ${s.iteration}: return=${s.meanEpisodeReturn.toFixed(3)} ` +
      `forward=${s.meanEpisodeForward.toFixed(3)}m ` +
      `pLoss=${s.policyLoss.toFixed(4)} vLoss=${s.valueLoss.toFixed(4)} std=${s.std.toFixed(3)}`,
  );
  if (!Number.isFinite(s.policyLoss) || !Number.isFinite(s.valueLoss)) {
    throw new Error('FAIL: 損失が NaN/Inf になりました');
  }
}
const after = tf.memory().numTensors;
console.log(`tensors: before=${before} after=${after} (leak=${after - before})`);
if (after - before > 50) {
  console.warn(`WARN: テンソルリークの疑い (+${after - before})`);
} else {
  console.log('OK: RL スタックは安定動作しています。');
}
