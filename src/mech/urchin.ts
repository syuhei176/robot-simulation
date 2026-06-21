/**
 * 放射対称「ウニ」機構（Rapier 3D 動的歩行）の Mechanism 実装。
 *
 * multiped（クモ: 行×左右で全脚が矢状面を漕ぐ＝全脚が前進に寄与）に対し、ウニは中心ハブの円環に
 * N 脚を放射配置し、各脚は自分の放射鉛直面で揺動する（{@link runUrchinGait} 参照）。前進(+x)への寄与は
 * cosθ 投影 ＝ 真横の脚は支持専従。1サイクル推力 ∝ N/2 で、同じ脚数なら直進コースでは直列クモに劣る一方、
 * 支持多角形が円形で near-vertical 保持＝弱サーボの静止保持には強い、という形態の差を実機比較できる。
 *
 * 機体スケーリング・PD/慣性・横安定化は四足/多足と同じ scaledBodyOverrides(mass) を流用し、脚長は
 * 絶対値（multiped と同じ思想）で上書きする。
 */
import { runUrchinGait, type UrchinOverrides } from '../sim3d/urchin-dynamics.ts';
import { scaledBodyOverrides } from '../sim3d/quadruped-dynamics.ts';
import type { CourseSpec } from '../sim3d/course.ts';
import { recordedQuadReplay } from './quad.ts';
import type { Mechanism, MechReplay, MechRunCtx } from './Mechanism.ts';

/** 既定の旗艦構成: 八足・脚15cm・250g・リング半径6cm。 */
const DEFAULT_MASS = 0.25;
const DEFAULT_LEG_COUNT = 8;
const DEFAULT_LEG_LEN = 0.15;
const DEFAULT_RING_RADIUS = 0.06;

/** ダッシュボード params から runUrchinGait の overrides を組み立てる（スイープと同一ロジック）。 */
export function buildUrchinOverrides(
  params: Record<string, number>,
  torqueCapNm: number,
  course: CourseSpec,
): UrchinOverrides {
  const mass = params.mass ?? DEFAULT_MASS;
  const legCount = Math.max(4, Math.round(params.legCount ?? DEFAULT_LEG_COUNT));
  const legLen = params.legLen ?? DEFAULT_LEG_LEN;
  const ringRadius = params.ringRadius ?? DEFAULT_RING_RADIUS;
  const urchinGait = (params.gaitMode ?? 0) >= 0.5 ? 'wave' : 'tripod';
  const legPlane = (params.legPlane ?? 0) >= 0.5 ? 'heading' : 'radial';
  const base = scaledBodyOverrides(mass, torqueCapNm);
  const half = legLen / 2;
  // 地形コースは端まで歩く時間を確保（平地は既定 duration）。
  const duration =
    course.stepRise > 0 ? Math.min(40, Math.max(8, (course.goalX + 0.8) / 0.06 + 4)) : undefined;
  return {
    ...base,
    leg: { ...base.leg, thigh: half, shin: half },
    legCount,
    ringRadius,
    urchinGait,
    legPlane,
    course,
    ...(duration !== undefined ? { duration } : {}),
    gait: {
      period: params.period,
      strideM: params.strideM,
      liftM: params.liftM,
      standM: params.standM,
      stanceDuty: params.stanceDuty,
    },
  };
}

export const urchinMechanism: Mechanism = {
  id: 'urchin',
  name: '機構: ウニ（放射対称多脚）',
  subtitle: '円環ハブに放射配置・各脚が放射面で揺動（splay の代償を物理化）',
  supportsCourse: true,
  params: [
    // --- 形態（歩容ではないので最適化中は固定: optimize=false） ---
    {
      key: 'mass',
      label: '総質量(機体スケール)',
      min: 0.15,
      max: 0.6,
      step: 0.05,
      default: DEFAULT_MASS,
      unit: 'kg',
      optimize: false,
    },
    {
      key: 'legCount',
      label: '脚数(放射)',
      min: 4,
      max: 12,
      step: 1,
      default: DEFAULT_LEG_COUNT,
      optimize: false,
    },
    {
      key: 'legLen',
      label: '脚長',
      min: 0.08,
      max: 0.2,
      step: 0.01,
      default: DEFAULT_LEG_LEN,
      unit: 'm',
      optimize: false,
    },
    {
      key: 'ringRadius',
      label: 'リング半径(ハブ)',
      min: 0.03,
      max: 0.1,
      step: 0.005,
      default: DEFAULT_RING_RADIUS,
      unit: 'm',
      optimize: false,
    },
    {
      key: 'gaitMode',
      label: '歩容(0=tripod,1=wave)',
      min: 0,
      max: 1,
      step: 1,
      default: 0,
      optimize: false,
    },
    {
      // radial=本物の splay ウニ（横脚が擦って這う）, heading=操舵ウニ（ヨーDOF相当・クモ並みに歩く）。
      key: 'legPlane',
      label: '揺動面(0=放射,1=操舵)',
      min: 0,
      max: 1,
      step: 1,
      default: 0,
      optimize: false,
    },
    // --- 歩容（最適化対象） ---
    { key: 'period', label: '歩容周期', min: 0.6, max: 1.6, step: 0.05, default: 0.9, unit: 's' },
    {
      key: 'strideM',
      label: 'ストライド',
      min: 0.03,
      max: 0.09,
      step: 0.005,
      default: 0.055,
      unit: 'm',
    },
    {
      key: 'liftM',
      label: '遊脚持ち上げ',
      min: 0.02,
      max: 0.08,
      step: 0.005,
      default: 0.05,
      unit: 'm',
    },
    {
      key: 'standM',
      label: '保持高さ',
      min: 0.06,
      max: 0.16,
      step: 0.005,
      default: 0.108,
      unit: 'm',
    },
    { key: 'stanceDuty', label: '接地比', min: 0.5, max: 0.9, step: 0.05, default: 0.75 },
  ],
  async run(ctx: MechRunCtx): Promise<MechReplay> {
    const replay = await runUrchinGait(
      buildUrchinOverrides(ctx.params, ctx.torqueCapNm, ctx.course),
      60,
    );
    return recordedQuadReplay(replay, ctx.torqueCapNm, ctx.motorName, ctx.course);
  },
};
