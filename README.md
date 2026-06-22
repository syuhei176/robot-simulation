# MuJoCo 3D 蛇ロボット シミュレーター

MuJoCo の 3D 蛇（snake3d）を題材に、**手設計の基盤歩容（開ループ）と、実機センサー相当の観測で学んだ
操舵可能な強化学習方策（残差RL）を同じコース上で比較する**シミュレーター。だんだん難しくなる地形（小障害物 →
階段 → 踊り場/壁）を、開ループの基盤歩容は地形に進行方向を蹴られて斜行してしまうのに対し、RL 方策は
**サーボ＋頭IMU だけ**（特権情報なし）の観測で基盤の斜行を**目標方位（ヘディング指令）へ操舵し直しながら**走破する。
観測を実機で取れる信号に限ったので **sim-to-real** が現実的で、目的地を相対方位で与えるので操舵できる
（目標方位=0 が直進に一致）。あわせて、この蛇を実際に作るときの **サーボ選定・材料表（BOM）** も数値で出す。
Three.js + TypeScript、物理は MuJoCo(WASM)。学習は Node オフライン（TensorFlow.js）で、ブラウザは記録済み
リプレイを再生するだけ（TF.js をバンドルしない）。

**🔗 ライブデモ: https://syuhei176.github.io/robot-simulation/**（`main` への push で GitHub Pages へ自動デプロイ）

## できること

- **MuJoCo 3D 蛇（関節構成 × 歩容のキャンバス）** — `src/sim3d/snake3d-dynamics.ts`。関節を軸パターン
  （all-yaw=横うねり / alt-yaw-pitch=yaw/pitch交互 / all-pitch=尺取り）で宣言し、歩容はパラメタ化した
  セルペノイド波（yaw/pitch の振幅・位相・周期・波長）で与える。前進は車輪相当の異方抵抗（接地リンクのみ）で生む。
  地形は実接触の剛体箱（`SnakeTerrainBox`）として置き、蛇は体を押し付けて段を登る。

- **実機センサー相当の観測で操舵できる残差RL** — `runSnake3D` を 1 制御ステップずつ進められる RL 環境
  `SnakeEnv`（`src/env/SnakeEnv.ts`）に展開し、方策が n−1 関節の歩容残差を制御する **残差RL**
  （目標角 = 基盤登坂歩容 + ±maxJointDelta·tanh(action)。action=0 でも障害物＋階段を越える）。
  - **観測は実機で取れる信号だけ（56次元・sim-to-real）**: 関節角／速度／負荷（サーボの present position/velocity/load）
    ＋頭IMU（姿勢・ジャイロ）＋位相クロック＋**目標ヘディング誤差**。シミュ特権情報（前方地形プレビュー・絶対COM位置・
    頭尾から計算する厳密ヘディング）は使わない。頭yaw はうねりで振れるので EMA フィルタで平均方位（≈走行方位）を取り出す。
  - **目的地は「目標ヘディング指令（相対方位）」**: 絶対(x,y)は localization が要り実機困難なので、IMU/コンパスで
    成立する相対方位で条件付け。エピソード毎に目標方位を ±30° ランダム化して操舵を学ぶ。報酬は「指令方向への前進 −
    指令光線からの横ずれ − …」。**基盤歩容は地形に蹴られて ~−40° 斜行する**（progression −55°/challenge −39°）のを、
    RL は指令方向へ**操舵し直して**走破する。
  - **コース汎用**: 毎エピソード地形をランダム化（平地×3／障害物／各種階段／壁つき。`makeCourseBank`）する
    ドメインランダム化で単一方策を学習。平地を厚めに混ぜて「直進時は操舵しない・地形なしでも操舵できる」を学ばせる。
  - **PPO 安定化**: **勾配ノルムクリップ＋KL 早期停止＋探索 std アニーリング**（決定論 mean に性能を移す）＋
    速度比例の横ペナルティ＋上限付き中心線ペナルティ（有界）＋残差幅の抑制で安定収束させた。
  - **実績（MG996R・1800step・決定論・完走。単一方策をコース×方位 {−25°,0°,+25°} で評価）**:

    | コース           | 基盤(0°方位) | RL 走破(0°)     | RL の操舵（−25/0/+25°→達成方位）   |
    | ---------------- | ------------ | --------------- | ---------------------------------- |
    | 平地             | 943cm(−1°)   | 939cm           | −12 / −3 / +4°（0°でほぼ直進）     |
    | 直進チャレンジ   | 555cm(−39°)  | 846cm(**+52%**) | −16 / −10 / −10°（基盤−39°を矯正） |
    | 進行性（終端壁） | 348cm(−55°)  | 383cm           | 壁支配で方位は壁アーティファクト   |

  - **正直なトレードオフ**: 前方地形プレビュー（特権）を捨てたので、**地形依存の方位バイアスを車載の偏った頭IMU
    だけでは解像しきれず、操舵レンジは控えめ**（平地で約16°、challenge で基盤veerの矯正）。走破は基盤を大きく上回り
    （challenge 0° +52%・+25° +123%）、観測は実機相当という核は達成。さらなる精度は **teacher-student 蒸留**
    （特権 teacher → センサーのみ student）が次の一手。

    `pnpm train-snake --course general --episode-steps 1800` で学習、`pnpm verify-policy --stem snake3d-general-mg996r`
    で各コース×方位の再現・操舵・基盤比較を検証できる。

- **統合ダッシュボード（コース選択 × 基盤 vs RL × 操舵）** — `index.html` + `src/dashboard.ts`。コース（平地／進行性／
  直進チャレンジ）・モーターを選び、**基盤歩容（scripted・ライブ実行）** と **RL方策（記録リプレイ再生）** を切り替えて
  比較する。RL モードでは **目標方位スライダー**（−25°〜+25°）が出て、最近傍の録画方位を再生＝操舵を目視できる
  （「目標方位 X° → 実方位 Y°・基盤 → RL（+N%）」を表示）。コースを切り替えても同じ汎用方策が各コース×方位の録画を再生する。

- **材料表（BOM＝買い物リスト）** — `materials.html` + `src/sim3d/bom.ts`。蛇（16リンク=15関節・0.6kg）を実際に
  作るためのサーボ＋制御部品＋フレーム/電源の概算を、サーボカタログ（`servos.ts`）と静的保持トルク（`chain.ts`）から
  導出。必要トルク 0.21 N·m を満たす中で **推奨=SCS0009（位置帰還つきで多関節協調制御に必須＝推奨理由はトルクでなく帰還）**。

## 動かす

Node 24+ が必要（相対 import を `.ts` 拡張子で解決するため）。

```bash
nvm use 24
pnpm install
pnpm dev          # http://localhost:5173
```

- `http://localhost:5173/` … 統合ダッシュボード。コース × モーター × 歩容パラメータを切替。
  - **基盤歩容 / RL方策** ボタンで切替。基盤歩容はスライダーの歩容をライブ実行、RL は学習済み汎用方策の記録リプレイ。
  - 既定は「直進チャレンジ × MG996R」。基盤歩容が ~−39° 斜行 → RL方策ボタンで走破が伸び（基盤 555cm → RL 846cm/+52%）、
    **目標方位スライダー**で左右への操舵を目視できる（平地が最も分かりやすい）。
- `http://localhost:5173/materials.html` … 材料表（サーボ選択で合計が即時更新）。

## コマンド（Node 24+ で実行）

```bash
pnpm train-snake --course general --episode-steps 1800   # 蛇の汎用RL方策を学習（ドメインランダム化）→ public/policies/
pnpm train-snake --course challenge --episode-steps 1800 # 単一コースで学習（比較用。flat / progression / challenge）
pnpm verify-policy --stem snake3d-general-mg996r         # 保存済み方策を再ロードし各コース×方位で操舵・再現・基盤比較を検証
pnpm dev / pnpm build / pnpm preview                     # ダッシュボード（開発 / 本番ビルド / プレビュー）
pnpm quality-check                                       # lint + format:check + type-check
```

`train-snake` の主なオプション: `--iters`（イテレーション数）`--rollout`（PPO ロールアウト長）`--motor`（サーボ id）
`--episode-steps`（1エピソードの制御ステップ）`--lr` `--ent-coef` `--eval-every`。

## 構成

- [`src/sim3d/snake3d-dynamics.ts`](src/sim3d/snake3d-dynamics.ts) … MuJoCo 3D 蛇の物理・歩容・地形（コース生成・ドメインランダム化）。
- [`src/sim3d/mujoco-engine.ts`](src/sim3d/mujoco-engine.ts) … MuJoCo(WASM) のロード。
- [`src/env/SnakeEnv.ts`](src/env/SnakeEnv.ts) … 残差RL 環境（`RLEnv` 契約）。観測=**実機相当**（関節角／速度／負荷＋頭IMU 姿勢/ジャイロ＋位相＋目標ヘディング誤差。特権情報なし）。地形バンクで reset 毎にコースを、毎エピソード目標方位（±30°）をランダム化。
- [`src/rl/`](src/rl/) … 方策 [`Policy.ts`](src/rl/Policy.ts)（ガウス方策＋価値関数 MLP）＋ [`PPO.ts`](src/rl/PPO.ts)（勾配クリップ・KL 早期停止）＋契約 [`RLEnv.ts`](src/rl/RLEnv.ts)。TensorFlow.js（Node オフライン）。
- [`src/mech/`](src/mech/) … ダッシュボード用の機構抽象 [`Mechanism.ts`](src/mech/Mechanism.ts) ＋ [`snake3d.ts`](src/mech/snake3d.ts)（snake3d 実装・基盤歩容の scripted 実行＋RL リプレイ再生）＋ [`registry.ts`](src/mech/registry.ts)。
- [`src/render/StairDynamicsView.ts`](src/render/StairDynamicsView.ts) … Three.js ビュー（蛇カプセル・地形箱・軌跡・距離グリッド、頭追従カメラ）。
- [`src/sim3d/course.ts`](src/sim3d/course.ts) … ダッシュボードのコースカタログ（平地/進行性/直進チャレンジ）。
- [`src/sim3d/{servos,chain,bom}.ts`](src/sim3d/) ＋ [`src/materials.ts`](src/materials.ts) … サーボカタログ・静力学・材料表。
- [`scripts/train-snake.ts`](scripts/train-snake.ts) … 蛇の残差RL（PPO）をオフライン学習し、汎用方策＋各コース×方位の記録リプレイを `public/policies/` に保存。
- [`scripts/verify-policy.ts`](scripts/verify-policy.ts) … 保存済み方策を再ロードし各コース×方位で決定論評価して操舵・基盤と比較。

## 設計メモ

検討の経緯・数値・前提は [`docs/snake-stair-sizing.md`](docs/snake-stair-sizing.md)（過去の四足/多足機構の探索を含む歴史的記録）。

- 物理エンジン: **MuJoCo**（`@mujoco/mujoco` WASM）。蛇は実接触で段に体を押し付けて登る。
- 学習は Node オフライン（TF.js）で実行し、ブラウザは記録済みリプレイ（frames）を再生するだけ＝TF.js をバンドルしない。
- 環境は決定論的（同じ重み・地形なら同結果）。保存方策は `verify-policy` で再現を確認できる。
