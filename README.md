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
- **機体スケール連動（小型四足 × 安サーボ）** — 総質量スライダーが**機体スケール** s も兼ねる（密度一定の相似縮小:
  mass ∝ s³, 脚長・歩容 ∝ s, PD ゲイン ∝ s⁵, 横安定化 ∝ s⁴, substeps ∝ 1/s）。これで **150g 級の小型四足が
  SCS0009（実売 ~700円）のトルク上限内**で平地を歩く（要求 ~0.09 N·m ≤ cap 0.226・飽和なし）。大型機体は
  STS3215、小型機体は SCS0009、と同じダッシュボードで端から端まで評価できる。s=1（1.2kg）で従来機体に一致。
- **統合ダッシュボード** — `index.html` で **機構（蛇/四足）× コース（階段/低い階段/小障害物/合成/平地）× モーター × 歩容パラメータ** を
  1画面で切り替えて挙動を再生。新機構は `src/mech` に `Mechanism` を1つ足すだけで現れる。
- **共通Map（合成コース）+ 四足の地形適応歩容** — 1つのコースに「平地→障害物→階段」を直列に並べた
  **合成コース** を `src/sim3d/course.ts` に追加。四足は足先を `terrainTopAt` に沿わせる**地形適応クロール歩容**で
  障害物（2.5cm）と低い階段（3cm×3）を越えて端まで走破する。蛇は輪郭追従の診断用（長い合成地形では座屈する）。
- **オフライン歩容最適化（CMA-ES）** — 歩容パラメータを Mac/Node のヘッドレス Rapier 上で
  CMA-ES 最適化（目的＝前進量−物理破綻ペナルティ。機構が宣言する `replay.score()` で機構非依存）。
  結果（調整済みパラメータ＋改善履歴）を `public/tuned/*.json` に出力し、ダッシュボードで
  **scripted ↔ tuned を切り替えて再生・改善カーブを表示**する。`pnpm optimize-gait`
- **平面歩容の学習（RL）** — 平面の蛇型で前進歩容を自前 PPO で学習するデモ（`playground.html`）。
  忠実な接触・トルク上限が揃ったので、登攀歩容の学習にも展開できる土台。
- **3D 四足の強化学習（残差RL・地形適応）** — `runQuadrupedGait` を 1 制御ステップずつ進められる RL 環境
  `QuadEnv`（`src/env/QuadEnv.ts`）に展開。方策が 8 関節（hip×4 + knee×4）を制御する。既定は **残差RL**＝
  IK クロール歩容に方策の補正を上乗せ（`--base-tuned auto` で CMA-ES の速い tuned 歩容を土台にできる）。
  土台が action=0 でも前進するので決定論方策も最初から歩き、end-to-end が嵌る「ノイズ依存の縮退（平均は静止）」を
  回避できる。観測＝位相クロック＋胴 pitch/速度＋関節角＋**前方地形プレビュー**（合成コースで段差を先読み）、
  報酬＝前進−傾き−行動エネルギー−トルク飽和。学習は Mac オフライン（`Policy`+`PPO`/TF.js）。学習した方策の
  決定論ロールアウトを **frames 記録**して `public/policies/*.replay.json` に保存し、**ダッシュボードの「RL」ボタンで
  TF.js 無しで再生**できる。`pnpm train-quad --base-tuned auto`（end-to-end 比較は `--base-gait false`）
  - 実績: 平地 小型四足×SCS0009 = 決定論 54cm/200step、合成コース 大型四足×STS3215（地形適応）= 障害物＋
    階段3段を登り**踊り場まで全走破 241cm/1800step**（転倒なし）。ダッシュボードの RL ボタンで再生できる。

## 動かす

Node 24 が必要（相対 import を `.ts` 拡張子で解決するため）。

```bash
nvm use 24
pnpm install
pnpm dev          # http://localhost:5173
```

- `http://localhost:5173/` … 統合ダッシュボード（機構 × コース × モーター × 歩容パラメータ）
  - 蛇＝コース physical attempt（赤: 落下/干渉、オレンジ: トルク超過、黄: 滑り、水色: 支持接触）。コース選択で階段/低い階段/小障害物/合成/平地を切替。
  - 四足＝3D 動的歩行（モーター選択で τ上限を自動設定、歩容スライダーで period/stride/lift 等を調整）。
    **地形適応歩容**でコースを選べる（合成コースは障害物＋低い階段を走破。高い段は転倒する）。
    **総質量スライダーが機体スケールを兼ねる**: 既定は 150g の小型四足 × SCS0009 で平地を歩くデモ。
    質量を上げると機体ごと大きくなり要求トルクが増える（合成コース走破は STS3215 + 大型機体 + tuned）。
  - **scripted / tuned / RL 切替** … `pnpm optimize-gait` の結果（`public/tuned/`）がある 機構×コース×モーター で
    **tuned** が、`pnpm train-quad` の RL 方策記録（`public/policies/`）がある組合せで **RL** が有効になる。
    tuned は調整済み歩容＋改善カーブ、RL は学習方策の記録リプレイ（frames）を再生する。
- `http://localhost:5173/playground.html` … 平面蛇型の歩容学習（手動 CPG / PPO）デモ

## コマンド（Node 24 で実行）

```bash
pnpm motor-sizing   # 形状→モーター逆算＋機構×モーター比較＋軽量化スイープ
pnpm quadruped      # 四足 3D 動的歩行レポート＋静的サーボ選定スイープ
pnpm stair-dynamics # 階段の Rapier 動的プローブ＋完全 physical attempt 診断
pnpm optimize-gait  # 歩容を CMA-ES でオフライン最適化 → public/tuned/*.json（--mech/--course/--motor/--gens/--seed）
pnpm train-quad --base-tuned auto  # 3D 四足を残差RL(PPO)で学習 → public/policies/*.json（--iters/--rollout/--base-gait）
```

`optimize-gait` の主なオプション（既定: `--mech quad --course stairs --motor sts3215 --gens 20 --seed 1`）:

```bash
pnpm optimize-gait --mech quad  --course flat --motor scs0009   # 小型四足(既定150g)の歩容を平地で最適化
pnpm optimize-gait --mech quad  --motor sts3215 --mass 1.2      # 大型四足(--massで機体スケール上書き)を階段で最適化
pnpm optimize-gait --mech snake --course lowStairs              # 蛇の歩容（referenceSpeed/clearance）を低い階段で最適化
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
- [`StairDynamicsView.ts`](src/render/StairDynamicsView.ts) … 共有レンダラ（蛇の連鎖・四足・コースを描く）。
- [`course.ts`](src/sim3d/course.ts) … コース（地形）の単一の真実（階段/低い階段/小障害物/平地）。

統合ダッシュボードと歩容最適化:

- [`src/mech/`](src/mech/) … 機構抽象。`Mechanism`（蛇/四足の `run` と `score`）＋ `registry`。新機構はここに1つ足すだけ。
- [`dashboard.ts`](src/dashboard.ts) … 機構 × コース × モーター × 歩容 を1画面で切替＋ scripted/tuned 再生（`index.html`）。
- [`scripts/optimize-gait.ts`](scripts/optimize-gait.ts) … CMA-ES（Jacobi 固有値分解つき・seed 再現可）で歩容をオフライン最適化。
- [`scripts/train-quad.ts`](scripts/train-quad.ts) … 3D 四足の end-to-end RL（PPO）をオフライン学習し方策を `public/policies/` に保存。

歩容の学習（`src/sim`, `src/env`, `src/rl`, `src/render`）:

- 物理 [`SnakePhysics.ts`](src/sim/SnakePhysics.ts) … 抵抗力ベースの準静的モデル（Hirose 標準モデル）。
  異方性摩擦の力・トルク釣り合いを 3×3 線形系で解き、前進が振幅/周波数に滑らかに応答する。
- 環境 [`MicrobotEnv.ts`](src/env/MicrobotEnv.ts)（平面蛇・行動→CPG パラメータ）/ [`QuadEnv.ts`](src/env/QuadEnv.ts)
  （3D 四足・行動→8 関節目標角, `runQuadrupedGait` をステップ化）。どちらも共通契約 [`RLEnv.ts`](src/rl/RLEnv.ts)。
- 方策/学習 [`Policy.ts`](src/rl/Policy.ts) / [`PPO.ts`](src/rl/PPO.ts) … TensorFlow.js の小さな MLP＋自前 PPO。
  ブラウザ（蛇）でも Node オフライン（四足）でも回せる。

## 設計メモ・ハンドオフ

検討の経緯・数値・前提・宿題は [`docs/snake-stair-sizing.md`](docs/snake-stair-sizing.md) に集約している。

- 物理エンジン: **Rapier**（`@dimforge/rapier3d-compat`）。接触で精度不足なら MuJoCo を検討。
- 静的サイズ逆算と動的検証は分離する（両方そろって初めてハード判断できる）。
- 既存の平面 RL（`src/sim`）と 3D 系（`src/sim3d`）は隔離している。
