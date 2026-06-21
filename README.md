# ロボット シミュレーター — モーター選定

小型ロボット（蛇型 / 四足）の **形状を比較し、必要なモーター（サーボ）を具体的に逆算・選定する**
ためのシミュレーター。ハードを作る前に「この形・この重さなら、どのサーボなら動くか／いくらか」を
数値で出すことを目的にする。Three.js + TypeScript、物理は [Rapier](https://rapier.rs/) と自前の静力学、
すべてブラウザ／Node で動く（GPU 不要）。

実サーボのカタログ（SG90 / MG90S / SCS0009 / MG996R / STS3215 ＋ Dynamixel 系）を単一の真実として持ち、
シミュレーション結果から **「このサーボで足りる / 足りない」「最安はどれ」** を直接返す。

## できること

- **静的モーター逆算** — 形状（リンク数・長さ・質量・段の寸法）→ 保持トルク → 安全率込みで満たす最小/最安サーボ。
  機構（蛇 vs 四足）の比較表も出す。`pnpm motor-sizing`
- **動的接触シミュレーション（階段）** — Rapier で剛体リンク＋トルク上限＋摩擦を解き、
  追従不能・滑り・落下・縁越えを実際に起こす。`pnpm stair-dynamics`
- **四足 3D 動的歩行** — IK クロール歩容を「忠実トルク（PD を ±ストールで clamp）」で駆動。
  cap→前進距離が単調・総質量に応答するので、**重さ × サーボ** で歩ける/失速するを判定できる。`pnpm quadruped`
- **統合ダッシュボード** — `index.html` で **機構（蛇/四足）× コース（階段/低い階段/小障害物/平地）× モーター × 歩容パラメータ** を
  1画面で切り替えて挙動を再生。新機構は `src/mech` に `Mechanism` を1つ足すだけで現れる。
- **平面歩容の学習（RL）** — 平面の蛇型で前進歩容を自前 PPO で学習するデモ（`playground.html`）。
  忠実な接触・トルク上限が揃ったので、登攀歩容の学習にも展開できる土台。

## 動かす

Node 24 が必要（相対 import を `.ts` 拡張子で解決するため）。

```bash
nvm use 24
pnpm install
pnpm dev          # http://localhost:5173
```

- `http://localhost:5173/` … 統合ダッシュボード（機構 × コース × モーター × 歩容パラメータ）
  - 蛇＝コース physical attempt（赤: 落下/干渉、オレンジ: トルク超過、黄: 滑り、水色: 支持接触）。コース選択で階段/低い階段/小障害物/平地を切替。
  - 四足＝3D 動的歩行（モーター選択で τ上限を自動設定、歩容スライダーで period/stride/lift 等を調整）。現状は平地のみ。
- `http://localhost:5173/playground.html` … 平面蛇型の歩容学習（手動 CPG / PPO）デモ

## コマンド（Node 24 で実行）

```bash
pnpm motor-sizing   # 形状→モーター逆算＋機構×モーター比較＋軽量化スイープ
pnpm quadruped      # 四足 3D 動的歩行レポート＋静的サーボ選定スイープ
pnpm stair-dynamics # 階段の Rapier 動的プローブ＋完全 physical attempt 診断
pnpm sanity         # 平面物理: CPG 駆動で前進・数値安定を確認
pnpm rl-smoke       # 平面 RL: 数イテレーションで学習が伸びる/NaN なしを確認
pnpm quality-check  # lint + format:check + type-check
```

## 構成

選定の中核（`src/sim3d/`）:

- [`servos.ts`](src/sim3d/servos.ts) … サーボカタログ（単一の真実）。ストール・質量・価格・FB有無・ギア・IF。
- [`chain.ts`](src/sim3d/chain.ts) … 矢状面の多リンク連鎖の静力学（円弧 IK・重力モーメント・片持ち）。
- [`quadruped-static.ts`](src/sim3d/quadruped-static.ts) … 四足の静的保持トルク（立脚/踏み出し/段差リーチ）。
- [`quadruped-dynamics.ts`](src/sim3d/quadruped-dynamics.ts) … 四足 3D 動的歩行（IK クロール＋忠実トルク＋横安定化）。
- [`stair-dynamics.ts`](src/sim3d/stair-dynamics.ts) … 階段の Rapier 動的接触シミュレーション。
- [`stair-kinematic-replay.ts`](src/sim3d/stair-kinematic-replay.ts) / [`stair-feasibility.ts`](src/sim3d/stair-feasibility.ts) / [`stair-physical-attempt.ts`](src/sim3d/stair-physical-attempt.ts)
  … 完全階段登りの目標軌道・実現可能性診断・失敗も動く physical attempt。
- [`StairDynamicsView.ts`](src/render/StairDynamicsView.ts) / [`stair-viewer.ts`](src/stair-viewer.ts) … ビューア。

平面歩容の学習（`src/sim`, `src/env`, `src/rl`, `src/render`）:

- 物理 [`SnakePhysics.ts`](src/sim/SnakePhysics.ts) … 抵抗力ベースの準静的モデル（Hirose 標準モデル）。
  異方性摩擦の力・トルク釣り合いを 3×3 線形系で解き、前進が振幅/周波数に滑らかに応答する。
- 環境 [`MicrobotEnv.ts`](src/env/MicrobotEnv.ts) … Gym 風。行動→CPG パラメータ、報酬＝前進−ズレ−エネルギー。
- 方策/学習 [`Policy.ts`](src/rl/Policy.ts) / [`PPO.ts`](src/rl/PPO.ts) … TensorFlow.js の小さな MLP＋自前 PPO、ブラウザ完結。

## 設計メモ・ハンドオフ

検討の経緯・数値・前提・宿題は [`docs/snake-stair-sizing.md`](docs/snake-stair-sizing.md) に集約している。

- 物理エンジン: **Rapier**（`@dimforge/rapier3d-compat`）。接触で精度不足なら MuJoCo を検討。
- 静的サイズ逆算と動的検証は分離する（両方そろって初めてハード判断できる）。
- 既存の平面 RL（`src/sim`）と 3D 系（`src/sim3d`）は隔離している。
