/**
 * Course — 機構に依存しない「コース（地形）」の単一の真実。
 *
 * 地形を矢状面(x-z)の箱primitive群 `boxes` と、地形上面の輪郭ポリライン `profile` で表現する。
 * - `boxes`     … 物理コライダ（{@link buildCourseColliders}）と描画（レンダラ）の真実。
 * - `profile`   … 蛇の参照軌道（輪郭追従）と {@link terrainTopAt}（clearance 診断）の真実。
 * - 横幅(y) は機構側が与える（同じコースを幅の違う機構が共有するため）。
 *
 * 既存の階段は `COURSES.stairs(...)` がバイト一致で再現する（リファクタの後方互換）。
 * 新しいコース（低い階段・小障害物・平地）は同じデータ形で足すだけで全機構が走れる。
 */
import RAPIER, { type Collider, type World } from '@dimforge/rapier3d-compat';

/** 矢状面(x-z)の箱。中心(cx,cz)と半寸法(halfX,halfZ)。横幅(y)は機構側が与える。 */
export interface CourseBox {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
}

export interface CourseSpec {
  id: string;
  name: string;
  /** 物理コライダ＆描画の真実（地面・段・障害物の箱）。 */
  boxes: CourseBox[];
  /** 地形上面の輪郭ポリライン [x, topZ]（x 単調増加）。蛇の参照軌道生成に使う。 */
  profile: Array<[number, number]>;
  /** 目的プラトーが始まる x [m]。歩容の最終整定位置（path 上の落とし所）の決定に使う。 */
  plateauStartX: number;
  /** 完了判定の前進ゴール x [m]（この先まで体が抜ければ走破）。 */
  goalX: number;
  /** 段差を越える代表「持ち上げ高さ」[m]。liftLinks / 静的トルク見積りに使う。 */
  stepRise: number;
  /** 段差を越える代表「前方リーチ」[m]。 */
  stepForward: number;
  /** 既定の床摩擦 μ。 */
  defaultFriction: number;
}

const DEFAULT_HALF_WIDTH_Y = 0.35;

/** 地形上面の高さ [m]。x を覆う箱の上面の最大値（無ければ 0）。 */
export function terrainTopAt(spec: CourseSpec, x: number): number {
  let top = 0;
  for (const box of spec.boxes) {
    if (x >= box.cx - box.halfX - 1e-9 && x <= box.cx + box.halfX + 1e-9) {
      top = Math.max(top, box.cz + box.halfZ);
    }
  }
  return top;
}

/** コースの箱を Rapier コライダとして生成する。横幅(y) は機構側が与える。 */
export function buildCourseColliders(
  world: World,
  spec: CourseSpec,
  friction: number,
  halfWidthY: number = DEFAULT_HALF_WIDTH_Y,
): Collider[] {
  return spec.boxes.map((box) =>
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(box.halfX, halfWidthY, box.halfZ)
        .setTranslation(box.cx, 0, box.cz)
        .setFriction(friction)
        .setRestitution(0),
    ),
  );
}

export interface StairsParams {
  rise: number;
  treadDepth: number;
  stepCount: number;
  forward: number;
}

const DEFAULT_STAIRS: StairsParams = { rise: 0.18, treadDepth: 0.25, stepCount: 3, forward: 0.1 };

// 既存 createStairs / buildTerrainPath と一致させるための固定量。
const STAIR_BASE_HALF_X = 1.5;
const STAIR_BASE_CX = -0.75;
const STAIR_BASE_HALF_Z = 0.02;
const STAIR_LANDING_DEPTH = 1.35; // = morphology.totalLength(0.9) + 0.45（body 非依存に固定）

/** 階段コース。既存の階段ジオメトリ（base 1.5@-0.75 + 段 + landing）をデータ化したもの。 */
function stairsCourse(params: Partial<StairsParams> = {}): CourseSpec {
  const { rise, treadDepth, stepCount, forward } = { ...DEFAULT_STAIRS, ...params };
  const topZ = stepCount * rise;
  const stairEndX = stepCount * treadDepth;

  const boxes: CourseBox[] = [
    {
      cx: STAIR_BASE_CX,
      cz: -STAIR_BASE_HALF_Z,
      halfX: STAIR_BASE_HALF_X,
      halfZ: STAIR_BASE_HALF_Z,
    },
  ];
  for (let i = 0; i < stepCount; i++) {
    const height = (i + 1) * rise;
    boxes.push({
      cx: i * treadDepth + treadDepth / 2,
      cz: height / 2,
      halfX: treadDepth / 2,
      halfZ: height / 2,
    });
  }
  boxes.push({
    cx: stairEndX + STAIR_LANDING_DEPTH / 2,
    cz: topZ / 2,
    halfX: STAIR_LANDING_DEPTH / 2,
    halfZ: topZ / 2,
  });

  // 輪郭: 平地(z=0) → 各段の蹴上げ+踏面 → 上の踊り場。x は単調増加。
  const profile: Array<[number, number]> = [
    [STAIR_BASE_CX - STAIR_BASE_HALF_X, 0],
    [0, 0],
  ];
  for (let step = 0; step < stepCount; step++) {
    const h = (step + 1) * rise;
    profile.push([step * treadDepth, h]); // 蹴上げ
    profile.push([(step + 1) * treadDepth, h]); // 踏面
  }
  profile.push([stairEndX + STAIR_LANDING_DEPTH, topZ]);

  return {
    id: 'stairs',
    name: `階段 ${Math.round(rise * 100)}cm×${stepCount}`,
    boxes,
    profile,
    plateauStartX: (stepCount - 1) * treadDepth, // 最上段の蹴上げ位置（path 落とし所）
    goalX: stairEndX, // 段の終わり（走破判定）
    stepRise: rise,
    stepForward: forward,
    defaultFriction: 0.8,
  };
}

/** 平地コース。四足の既存地面 cuboid(3,3,0.05)@(0,0,-0.05) と一致（halfWidthY=3 で呼ぶ）。 */
function flatCourse(): CourseSpec {
  return {
    id: 'flat',
    name: '平地',
    boxes: [{ cx: 0, cz: -0.05, halfX: 3, halfZ: 0.05 }],
    profile: [
      [-3, 0],
      [3, 0],
    ],
    plateauStartX: 1.0,
    goalX: 1.0,
    stepRise: 0,
    stepForward: 0.05,
    defaultFriction: 0.9,
  };
}

/** 小障害物コース。低い箱を数個ばらまく。蛇は輪郭追従で乗り越える。 */
function bumpsCourse(
  params: { height?: number; count?: number; spacing?: number } = {},
): CourseSpec {
  const height = params.height ?? 0.04;
  const count = params.count ?? 3;
  const spacing = params.spacing ?? 0.3;
  const halfX = 0.05;
  const firstX = 0.2;

  const boxes: CourseBox[] = [
    {
      cx: STAIR_BASE_CX,
      cz: -STAIR_BASE_HALF_Z,
      halfX: STAIR_BASE_HALF_X,
      halfZ: STAIR_BASE_HALF_Z,
    },
  ];
  const profile: Array<[number, number]> = [
    [STAIR_BASE_CX - STAIR_BASE_HALF_X, 0],
    [0, 0],
  ];
  for (let i = 0; i < count; i++) {
    const cx = firstX + i * spacing;
    boxes.push({ cx, cz: height / 2, halfX, halfZ: height / 2 });
    profile.push([cx - halfX, 0], [cx - halfX, height], [cx + halfX, height], [cx + halfX, 0]);
  }
  const lastBumpX = firstX + (count - 1) * spacing;
  // 走破区間の床（base が x≤0.75 までなので、最終障害物の先に体長ぶんの平地を足す）。
  const flatEndX = lastBumpX + 1.3;
  const flatStartX = 0.75;
  boxes.push({
    cx: (flatStartX + flatEndX) / 2,
    cz: -STAIR_BASE_HALF_Z,
    halfX: (flatEndX - flatStartX) / 2,
    halfZ: STAIR_BASE_HALF_Z,
  });
  profile.push([flatEndX, 0]);

  return {
    id: 'bumps',
    name: `小障害物 ${Math.round(height * 100)}cm×${count}`,
    boxes,
    profile,
    plateauStartX: lastBumpX + 0.2, // 最終障害物を越えた先の平地で整定
    goalX: lastBumpX + 0.35, // 全障害物を越えれば走破
    stepRise: height,
    stepForward: 0.06,
    defaultFriction: 0.8,
  };
}

/** コースカタログ。引数で寸法を上書きできる。 */
export const COURSES = {
  stairs: (params?: Partial<StairsParams>): CourseSpec => stairsCourse(params),
  lowStairs: (): CourseSpec => ({
    ...stairsCourse({ rise: 0.08, treadDepth: 0.22, stepCount: 3 }),
    id: 'lowStairs',
    name: '低い階段 8cm×3',
  }),
  bumps: (): CourseSpec => bumpsCourse(),
  flat: (): CourseSpec => flatCourse(),
} as const;

export type CourseId = keyof typeof COURSES;

/** セレクタ等で使うコース一覧（id, ラベル）。ラベルは各コースの name から導出（単一の真実）。 */
export const COURSE_OPTIONS: Array<{ id: CourseId; label: string }> = (
  Object.keys(COURSES) as CourseId[]
).map((id) => ({ id, label: COURSES[id]().name }));
