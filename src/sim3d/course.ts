/**
 * Course — MuJoCo 蛇（snake3d）用のコース（地形）カタログ。地形は {@link SnakeTerrainBox} の集合で表す
 *（snake3d-dynamics の単一の真実を流用）。ダッシュボードのコース選択と、scripted（基盤歩容のライブ実行）・
 * RL（汎用方策の各コース録画）が同じコース定義を共有する。
 */
import {
  makeProgressionTerrain,
  makeStraightChallengeTerrain,
  type SnakeTerrainBox,
} from './snake3d-dynamics.ts';

/** コースカタログ（id → 地形を生成する関数）。平地は地形なし＝空配列。 */
export const COURSES = {
  flat: (): SnakeTerrainBox[] => [],
  progression: (): SnakeTerrainBox[] => makeProgressionTerrain(),
  challenge: (): SnakeTerrainBox[] => makeStraightChallengeTerrain(),
} as const;

export type CourseId = keyof typeof COURSES;

/** セレクタ用のコース一覧（id, ラベル）。 */
export const COURSE_OPTIONS: Array<{ id: CourseId; label: string }> = [
  { id: 'flat', label: '平地' },
  { id: 'progression', label: '進行性（障害物→階段→テーブル壁）' },
  { id: 'challenge', label: '直進チャレンジ（壁なし長距離）' },
];
