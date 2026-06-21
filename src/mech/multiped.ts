/**
 * 多足機構（Rapier 3D 動的歩行）の Mechanism 実装 — 「胴の小さいクモ/ウニ」型。
 *
 * 発見した設計: 合成コース（2.5cm 障害物＋3cm×3 階段, 絶対サイズ）を **安サーボ SCS0009**
 * （ストール 0.226 N·m）で走破できるのは、四足を縮小した機体ではなく
 *   1. 軽い胴（≲300g）— 保持トルクが質量に比例するので cap 内に収まる
 *   2. 障害物より十分長い脚（12〜15cm = 障害物の4〜5倍）— 段差を跨いで上れる
 *   3. 多数の脚（6〜8本）の tripod 歩容 — 常に約半数が接地して静的に安定（段で転ばない）
 * という形態。150g 級の四足は脚が短く（9cm）絶対 2.5cm の障害物に阻まれて 18cm で停滞するのに対し、
 * 軽量・長脚・多脚にすると同じ SCS0009 で踊り場まで全走破する（飽和率 数%・ピーク要求は整定の一過性）。
 *
 * 実装は四足と同じ `runQuadrupedGait`（N脚一般化済み）を使い、脚レイアウトを makeLegs(rows) で
 * 生成し、脚長・胴長だけ上書きする。PD/慣性/横安定化は scaledBodyOverrides(mass) の安定値を流用する。
 */
import {
  runQuadrupedGait,
  scaledBodyOverrides,
  makeLegs,
  type QuadDynOverrides,
} from '../sim3d/quadruped-dynamics.ts';
import type { CourseSpec } from '../sim3d/course.ts';
import { recordedQuadReplay } from './quad.ts';
import type { Mechanism, MechReplay, MechRunCtx } from './Mechanism.ts';

/** 既定の旗艦構成: 六足・脚15cm・250g（合成コースを SCS0009 で走破する）。 */
const DEFAULT_MASS = 0.25;
const DEFAULT_ROWS = 3; // 行数×2 = 脚数（2=四足, 3=六足, 4=八足）
const DEFAULT_LEG_LEN = 0.15;

/** 行数から胴の前後長を決める（行が重ならないよう脚数に比例して伸ばす）。 */
function trunkLengthFor(rows: number): number {
  return rows * 0.08;
}

/** ダッシュボード params から runQuadrupedGait の overrides を組み立てる（プローブと同一ロジック）。 */
export function buildMultipedOverrides(
  params: Record<string, number>,
  torqueCapNm: number,
  course: CourseSpec,
): QuadDynOverrides {
  const mass = params.mass ?? DEFAULT_MASS;
  const rows = Math.max(2, Math.round(params.rows ?? DEFAULT_ROWS));
  const legLen = params.legLen ?? DEFAULT_LEG_LEN;
  const gaitMode = (params.gaitMode ?? 0) >= 0.5 ? 'wave' : 'tripod';
  const base = scaledBodyOverrides(mass, torqueCapNm);
  const half = legLen / 2;
  // 地形コースは端まで歩く時間を確保（平地は既定 duration）。
  const duration =
    course.stepRise > 0 ? Math.min(40, Math.max(8, (course.goalX + 0.8) / 0.06 + 4)) : undefined;
  return {
    ...base,
    trunk: { ...base.trunk, length: trunkLengthFor(rows) },
    leg: { ...base.leg, thigh: half, shin: half },
    legs: makeLegs(rows, gaitMode),
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

export const multipedMechanism: Mechanism = {
  id: 'multiped',
  name: '機構: 多足（クモ/ウニ）',
  subtitle: '軽量・長脚・多脚 tripod（安サーボで段差走破）',
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
      key: 'rows',
      label: '脚の行数(×2=脚数)',
      min: 2,
      max: 4,
      step: 1,
      default: DEFAULT_ROWS,
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
      key: 'gaitMode',
      label: '歩容(0=tripod,1=wave)',
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
    const replay = await runQuadrupedGait(
      buildMultipedOverrides(ctx.params, ctx.torqueCapNm, ctx.course),
      60,
    );
    return recordedQuadReplay(replay, ctx.torqueCapNm, ctx.motorName, ctx.course);
  },
};
