# MuJoCo 3D 蛇ロボット シミュレーター

MuJoCo の 3D 蛇（snake3d）を題材に、**手設計の基盤歩容（開ループ）と、コースに汎用な強化学習方策（残差RL）を
同じコース上で比較する**シミュレーター。だんだん難しくなる地形（小障害物 → 階段 → 踊り場/壁）を、開ループの
基盤歩容では地形に進行方向を蹴られて斜行してしまうのに対し、閉ループの RL 方策は横ドリフトを検知して
進行方向へ操舵し直し、**どのコースでも基盤歩容の前進量を上回る**ことを示す。あわせて、この蛇を実際に作るときの
**サーボ選定・材料表（BOM）** も数値で出す。Three.js + TypeScript、物理は MuJoCo(WASM)。学習は Node オフライン
（TensorFlow.js）で、ブラウザは記録済みリプレイを再生するだけ（TF.js をバンドルしない）。

**🔗 ライブデモ: https://syuhei176.github.io/robot-simulation/**（`main` への push で GitHub Pages へ自動デプロイ）

## できること

- **MuJoCo 3D 蛇（関節構成 × 歩容のキャンバス）** — `src/sim3d/snake3d-dynamics.ts`。関節を軸パターン
  （all-yaw=横うねり / alt-yaw-pitch=yaw/pitch交互 / all-pitch=尺取り）で宣言し、歩容はパラメタ化した
  セルペノイド波（yaw/pitch の振幅・位相・周期・波長）で与える。前進は車輪相当の異方抵抗（接地リンクのみ）で生む。
  地形は実接触の剛体箱（`SnakeTerrainBox`）として置き、蛇は体を押し付けて段を登る。

- **コースに汎用な強化学習で「基盤歩容」を実際に上回る（残差RL・直進補正）** — `runSnake3D` を 1 制御ステップ
  ずつ進められる RL 環境 `SnakeEnv`（`src/env/SnakeEnv.ts`）に展開し、方策が n−1 関節の歩容残差を制御する **残差RL**
  （目標角 = 基盤登坂歩容 + ±maxJointDelta·tanh(action)。action=0 でも障害物＋階段を越える）。
  - **基盤の弱点**: 開ループの基盤歩容は地形に進行方向を蹴られ、その後も盲目的に ~−40° へ斜行して前進を浪費する
    （平地では真っ直ぐ進めるのに、コースでは大きく veer する）。閉ループの RL は **横オフセット＋体軸ヘディング観測**で
    斜行を検知し、+x へ操舵し直して浪費していた横移動を前進へ変換する。
  - **コース汎用**: 毎エピソード地形をランダム化（平地／障害物／各種階段／壁つき。`makeCourseBank`）する
    **ドメインランダム化**で単一方策を学習。平地を必ず混ぜることで「ドリフトしていない時は操舵しない」を学ばせ、
    特定コースへの過学習（常時操舵バイアス）を防ぐ。ベスト方策は **コース横断の最小改善率**で選び、どのコースでも
    基盤を下回らないようにする。
  - **PPO 安定化**: 残差RLは報酬の振れ幅が大きく崩れやすいので、**勾配ノルムクリップ＋KL 早期停止＋探索 std
    アニーリング**（決定論 mean に性能を移す）＋上限付き中心線ペナルティ＋残差幅の抑制で安定収束させた。
  - **実績（MG996R・1800step・決定論・完走。単一の汎用方策が全コースで基盤超え）**:

    | コース                           | 基盤歩容 | RL 方策 | 上乗せ                              |
    | -------------------------------- | -------- | ------- | ----------------------------------- |
    | 平地                             | 943cm    | 948cm   | +1%                                 |
    | 進行性（障害物→階段→テーブル壁） | 348cm    | 370cm   | +7%                                 |
    | 直進チャレンジ（壁なし長距離）   | 555cm    | 839cm   | **+51%**（横ドリフト −447cm→−17cm） |

    `pnpm train-snake --course general --episode-steps 1800` で学習、`pnpm verify-policy --stem snake3d-general-mg996r`
    で各コースの再現（基盤超え）を検証できる。

- **統合ダッシュボード（コース選択 × 基盤 vs RL）** — `index.html` + `src/dashboard.ts`。コース（平地／進行性／
  直進チャレンジ）・モーターを選び、**基盤歩容（scripted・ライブ実行）** と **RL方策（記録リプレイ再生）** を切り替えて
  比較する。RL は「基盤 X cm → RL Y cm（+N%）」を表示。コースを切り替えても同じ汎用方策が各コースの録画を再生する。

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
  - 既定は「直進チャレンジ × MG996R」。基盤歩容が斜行 → RL方策ボタンで +x へ操舵し直す様子（基盤 555cm → RL 839cm）が見える。
- `http://localhost:5173/materials.html` … 材料表（サーボ選択で合計が即時更新）。

## コマンド（Node 24+ で実行）

```bash
pnpm train-snake --course general --episode-steps 1800   # 蛇の汎用RL方策を学習（ドメインランダム化）→ public/policies/
pnpm train-snake --course challenge --episode-steps 1800 # 単一コースで学習（比較用。flat / progression / challenge）
pnpm verify-policy --stem snake3d-general-mg996r         # 保存済み方策を再ロードし各コースで「基盤超え」の再現を検証
pnpm dev / pnpm build / pnpm preview                     # ダッシュボード（開発 / 本番ビルド / プレビュー）
pnpm quality-check                                       # lint + format:check + type-check
```

`train-snake` の主なオプション: `--iters`（イテレーション数）`--rollout`（PPO ロールアウト長）`--motor`（サーボ id）
`--episode-steps`（1エピソードの制御ステップ）`--lr` `--ent-coef` `--eval-every`。

## 構成

- [`src/sim3d/snake3d-dynamics.ts`](src/sim3d/snake3d-dynamics.ts) … MuJoCo 3D 蛇の物理・歩容・地形（コース生成・ドメインランダム化）。
- [`src/sim3d/mujoco-engine.ts`](src/sim3d/mujoco-engine.ts) … MuJoCo(WASM) のロード。
- [`src/env/SnakeEnv.ts`](src/env/SnakeEnv.ts) … 残差RL 環境（`RLEnv` 契約）。観測=位相＋COM速度＋頭クリアランス/ピッチ＋横オフセット＋体軸ヘディング＋前方地形プレビュー＋関節角。地形バンクで reset 毎にコースをランダム化。
- [`src/rl/`](src/rl/) … 方策 [`Policy.ts`](src/rl/Policy.ts)（ガウス方策＋価値関数 MLP）＋ [`PPO.ts`](src/rl/PPO.ts)（勾配クリップ・KL 早期停止）＋契約 [`RLEnv.ts`](src/rl/RLEnv.ts)。TensorFlow.js（Node オフライン）。
- [`src/mech/`](src/mech/) … ダッシュボード用の機構抽象 [`Mechanism.ts`](src/mech/Mechanism.ts) ＋ [`snake3d.ts`](src/mech/snake3d.ts)（snake3d 実装・基盤歩容の scripted 実行＋RL リプレイ再生）＋ [`registry.ts`](src/mech/registry.ts)。
- [`src/render/StairDynamicsView.ts`](src/render/StairDynamicsView.ts) … Three.js ビュー（蛇カプセル・地形箱・軌跡・距離グリッド、頭追従カメラ）。
- [`src/sim3d/course.ts`](src/sim3d/course.ts) … ダッシュボードのコースカタログ（平地/進行性/直進チャレンジ）。
- [`src/sim3d/{servos,chain,bom}.ts`](src/sim3d/) ＋ [`src/materials.ts`](src/materials.ts) … サーボカタログ・静力学・材料表。
- [`scripts/train-snake.ts`](scripts/train-snake.ts) … 蛇の残差RL（PPO）をオフライン学習し、汎用方策＋各コースの記録リプレイを `public/policies/` に保存。
- [`scripts/verify-policy.ts`](scripts/verify-policy.ts) … 保存済み方策を再ロードし各コースで決定論評価して基盤と比較。

## 設計メモ

検討の経緯・数値・前提は [`docs/snake-stair-sizing.md`](docs/snake-stair-sizing.md)（過去の四足/多足機構の探索を含む歴史的記録）。

- 物理エンジン: **MuJoCo**（`@mujoco/mujoco` WASM）。蛇は実接触で段に体を押し付けて登る。
- 学習は Node オフライン（TF.js）で実行し、ブラウザは記録済みリプレイ（frames）を再生するだけ＝TF.js をバンドルしない。
- 環境は決定論的（同じ重み・地形なら同結果）。保存方策は `verify-policy` で再現を確認できる。
