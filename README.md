# Microbots · RL

ベイマックス（Big Hero 6）のマイクロボットを「連結したマイクロボットの鎖（蛇型）」として再現し、
**深層強化学習（自前PPO）で『速く前進』する歩容をゼロから学習**させるブラウザデモ。
Three.js + TypeScript で、物理エンジンも RL アルゴリズムもすべて自前実装。GPU 不要（学習はブラウザの WebGL バックエンドで動作）。

## 動かす

Node 24 が必要（相対 import を `.ts` 拡張子で解決するため）。

```bash
nvm use 24
pnpm install
pnpm dev          # http://localhost:5173 を開く
```

UI:

- **手動 CPG / 強化学習** … モード切替
- 強化学習モード:
  - **学習開始/停止** … PPO の学習をブラウザ内で実行（報酬曲線がリアルタイムに伸びる）
  - **重みリセット** … 方策ネットを初期化
  - **保存 / 読込** … 学習済み重みを localStorage に保存・復元
  - 3D ビューは「現在の方策」を常時ライブ再生（学習が進むと速くなる）

## しくみ

- **物理** [`src/sim/SnakePhysics.ts`](src/sim/SnakePhysics.ts)
  運動学的・抵抗力ベースモデル（Hirose らの蛇型ロボット標準モデル）。各リンクの相対関節角
  φ を「形状」として与え、異方性摩擦（体軸方向 cT ≪ 横方向 cN）の準静的な力・トルク釣り合いを
  3×3 線形系として解いて胴体の剛体運動を求める。力ベースの動的積分と違いカオス的にならず、
  前進が振幅/周波数に対して滑らか・単調に応答する（＝RL に適した報酬地形）。
- **環境** [`src/env/MicrobotEnv.ts`](src/env/MicrobotEnv.ts)
  Gym 風。行動 a∈R³ を squash して CPG パラメータ [振幅, 周波数, 位相差] に写像し、ローパス
  平滑化して進行波を保つ。報酬 = 前進量 − 進行方向ズレ − エネルギー。直進タスクのため旋回は固定。
- **方策/価値** [`src/rl/Policy.ts`](src/rl/Policy.ts)
  TensorFlow.js の小さな MLP。連続行動のガウス方策（mean は状態依存、logStd は学習可能）＋ V(s)。
- **PPO** [`src/rl/PPO.ts`](src/rl/PPO.ts)
  ロールアウト収集 → GAE → クリップ付き方策更新。すべてブラウザで完結。
- **描画** [`src/render/SceneView.ts`](src/render/SceneView.ts)
  黒キューブ＋青エッジ発光のマイクロボット、影・グリッド・フォグ・COM 追従カメラ。

## 検証用スクリプト（Node 24 で実行）

```bash
pnpm sanity     # 物理: CPG 駆動で前進・数値安定を確認
pnpm rl-smoke   # RL: 数イテレーション回して学習が伸びる/NaN が無いことを確認
```

## 品質チェック

```bash
pnpm quality-check   # lint + format:check + type-check
```
