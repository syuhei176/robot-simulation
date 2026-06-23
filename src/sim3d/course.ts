/**
 * Course — MuJoCo 蛇（snake3d）用のコース（地形）カタログ。地形は {@link SnakeTerrainBox} の集合で表す
 *（snake3d-dynamics の単一の真実を流用）。ダッシュボードのコース選択と、scripted（基盤歩容のライブ実行）・
 * RL（部屋ナビ方策のライブ駆動／録画）が同じコース定義を共有する。
 *
 * コースは「部屋（6×6m 壁囲い）」に一本化し、違いは**障害物の有無**だけ。障害物レイアウトは固定シードで
 * 決定論的に生成する（毎回同じ部屋＝ダッシュボードの見た目とライブ駆動が一致）。室内障害物は乗り越えず
 * 回避する高さ（5-7cm）なので、方策は「目標方位へ操舵しつつ障害物を避けて壁際ゴールへ向かう」を見せる。
 */
import { makeRoomWalls, makeRoomObstacles, type SnakeTerrainBox } from './snake3d-dynamics.ts';

/** 決定論 PRNG（mulberry32）。固定シードで毎回同じ障害物配置を得るために使う。 */
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return function (): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 固定の室内障害物（seed 42 = 3個・y方向に分散・中盤 x≈2.2-3.2・高さ5-7cm の回避物）。 */
const FIXED_ROOM_OBSTACLES: SnakeTerrainBox[] = makeRoomObstacles(mulberry32(42));

/** コースカタログ（id → 地形を生成する関数）。部屋の障害物あり/なしの2択のみ。 */
export const COURSES = {
  'room-obs': (): SnakeTerrainBox[] => [...makeRoomWalls(), ...FIXED_ROOM_OBSTACLES],
  room: (): SnakeTerrainBox[] => makeRoomWalls(),
} as const;

export type CourseId = keyof typeof COURSES;

/** 部屋系コースか（room / room-obs を 'room' と同等に扱う判定の単一ソース）。 */
export function isRoomCourse(course: string): boolean {
  return course === 'room' || course === 'room-obs';
}

/** セレクタ用のコース一覧（id, ラベル）。障害物ありを既定（テスト場所）として先頭に。 */
export const COURSE_OPTIONS: Array<{ id: CourseId; label: string }> = [
  { id: 'room-obs', label: '部屋・障害物あり（6×6m）' },
  { id: 'room', label: '部屋・障害物なし（6×6m）' },
];
