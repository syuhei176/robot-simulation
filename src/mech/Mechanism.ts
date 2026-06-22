/**
 * Mechanism — 機構（蛇・四足・将来の新機構）を統合ダッシュボードへ差し込むための抽象。
 *
 * 各機構は「コース＋モーター＋歩容パラメータ → リプレイ」を生成し、共有レンダラ
 * {@link StairDynamicsView} への描画と、統計パネル用の {@link StatRow} を提供する。
 * 新機構は {@link Mechanism} を1つ実装して registry に登録するだけで追加できる
 * （HTML やダッシュボード本体の変更は不要）。
 */
import type { CourseSpec } from '../sim3d/course.ts';
import type { StairDynamicsView } from '../render/StairDynamicsView.ts';

/** ダッシュボードのスライダー1本分の定義（歩容・物理つまみ）。 */
export interface MechParam {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  /** 表示単位（任意）。スライダー右の数値に添える。 */
  unit?: string;
  /**
   * オフライン歩容最適化（`scripts/optimize-gait.ts`）の探索対象に含めるか（既定: true）。
   * 環境・機体設計のつまみ（蛇の床摩擦 μ・四足の総質量）は歩容の制御量ではないので false にし、
   * 最適化中は既定値に固定する。ダッシュボードのスライダーには影響しない（UI は全 param を出す）。
   */
  optimize?: boolean;
}

/**
 * 1回の実行の良さをスカラー化したもの。オフライン最適化（fitness を最大化）と
 * ダッシュボードの改善カーブ表示が同じ評価を共有するための単一の真実。
 */
export interface MechScore {
  /** 最適化が最大化するスカラー目的値（高いほど良い）。前進量から物理的破綻のペナルティを引いたもの。 */
  fitness: number;
  /** 主目的の前進量 [m]（人が読む・改善カーブの縦軸）。 */
  progressM: number;
  /** 物理的に成立したか（転倒/未走破でなく、トルク・摩擦も充足）。 */
  feasible: boolean;
}

/** 統計パネルの1行。 */
export interface StatRow {
  label: string;
  value: string;
  kind?: 'good' | 'warn' | 'bad' | null;
}

/** run() に渡す実行コンテキスト。 */
export interface MechRunCtx {
  course: CourseSpec;
  /** モーター τ 上限 [N·m]（サーボ選択が既定値を入れ、スライダーで微調整可）。 */
  torqueCapNm: number;
  /** モーター表示名（統計表示用）。 */
  motorName: string;
  /** mechanism.params の現在値（key→値）。 */
  params: Record<string, number>;
}

/** 1回の実行結果。描画と統計の供給を担う（生リプレイを内部に閉じ込める）。 */
export interface MechReplay {
  /** リプレイ長 [s]。 */
  duration: number;
  /** メッシュ構築（機構切替・再計算のたびに1回）。 */
  bindView(view: StairDynamicsView): void;
  /** 時刻 t [s] の姿勢を適用。 */
  applyTime(view: StairDynamicsView, t: number): void;
  /** 時刻 t の逐次診断行（再生中に毎フレーム更新）。 */
  liveStats(t: number): StatRow[];
  /** 総合結果行（実行ごとに一定）。 */
  resultStats(): StatRow[];
  /** この実行のスカラー評価（最適化の目的・改善カーブ）。 */
  score(): MechScore;
}

export interface Mechanism {
  id: string;
  name: string;
  subtitle: string;
  /** コース（地形）を反映できるか。false の機構ではコース選択を無効化する。 */
  supportsCourse: boolean;
  /**
   * tuned/RL 成果物を照合するコース名（manifest の course 列）の上書き。コース選択 UI とは独立に
   * 学習を別コースで行う機構（蛇3D は RL を進行性コースで学習）に使う。未指定なら従来どおり。
   */
  rlCourse?: string;
  /** 歩容＋物理のライブ調整つまみ。ダッシュボードがこの配列からスライダーを動的生成する。 */
  params: MechParam[];
  run(ctx: MechRunCtx): Promise<MechReplay>;
}

/** params の既定値マップを得る（ダッシュボードの初期値・リセットに使う）。 */
export function defaultParamValues(mech: Mechanism): Record<string, number> {
  const values: Record<string, number> = {};
  for (const param of mech.params) values[param.key] = param.default;
  return values;
}
