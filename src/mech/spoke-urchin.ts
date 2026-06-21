/**
 * 「トゲ車輪ウニ」機構（Rapier 3D 動的シム）の Mechanism 実装。
 *
 * 放射状の長いスポーク（棘）が回転して「転がり歩き」し、段差の角にスポークを引っかけて梃子＋慣性で登る
 * リムレスホイール／whegs 型。脚で真下に立つ歩行型（urchin/quad）の段差限界≈3cm に対し、**スポーク長に
 * 近い高さ（数倍）まで登れる**（`scripts/spoke-climb.ts` で実測: ∞トルクで ~1.0–1.2×スポーク長、
 * 安サーボ SCS0009 でもトルク律速ながら 4–10cm＝歩行型の数倍）。接地したスポークが脚になり姿勢に依らない。
 *
 * 軸トルク上限＝サーボ τ上限（直結）。targetOmega が「歩容」に相当する唯一の制御量。
 */
import { runSpokeUrchin, type SpokeUrchinOverrides } from '../sim3d/spoke-urchin-dynamics.ts';
import type { CourseSpec } from '../sim3d/course.ts';
import { recordedQuadReplay } from './quad.ts';
import type { Mechanism, MechReplay, MechRunCtx } from './Mechanism.ts';

const DEFAULT_SPOKE_COUNT = 8;
const DEFAULT_SPOKE_LEN = 0.15;
const DEFAULT_MASS = 0.25;
const DEFAULT_WIDTH = 0.05;
const DEFAULT_OMEGA = 6;

export function buildSpokeUrchinOverrides(
  params: Record<string, number>,
  torqueCapNm: number,
  course: CourseSpec,
): SpokeUrchinOverrides {
  const spokeLen = params.spokeLen ?? DEFAULT_SPOKE_LEN;
  // 端まで転がる時間を確保（平地・小段は短め、長いコースは伸ばす）。
  const duration = Math.min(20, Math.max(6, (course.goalX + 0.8) / 0.15 + 3));
  return {
    spokeCount: Math.max(4, Math.round(params.spokeCount ?? DEFAULT_SPOKE_COUNT)),
    spokeLen,
    width: params.width ?? DEFAULT_WIDTH,
    mass: params.mass ?? DEFAULT_MASS,
    targetOmega: params.targetOmega ?? DEFAULT_OMEGA,
    torqueCapNm,
    course,
    duration,
  };
}

export const spokeUrchinMechanism: Mechanism = {
  id: 'spoke-urchin',
  name: '機構: トゲ車輪ウニ',
  subtitle: '放射スポークが回転して段差を登る（リムレスホイール／whegs）',
  supportsCourse: true,
  params: [
    {
      key: 'spokeCount',
      label: 'スポーク数',
      min: 4,
      max: 16,
      step: 1,
      default: DEFAULT_SPOKE_COUNT,
      optimize: false,
    },
    {
      key: 'spokeLen',
      label: 'スポーク長(=登坂力)',
      min: 0.08,
      max: 0.25,
      step: 0.01,
      default: DEFAULT_SPOKE_LEN,
      unit: 'm',
      optimize: false,
    },
    {
      key: 'mass',
      label: '総質量',
      min: 0.15,
      max: 0.6,
      step: 0.05,
      default: DEFAULT_MASS,
      unit: 'kg',
      optimize: false,
    },
    {
      key: 'width',
      label: '車幅(同軸2列)',
      min: 0.03,
      max: 0.1,
      step: 0.005,
      default: DEFAULT_WIDTH,
      unit: 'm',
      optimize: false,
    },
    // 唯一の制御量（歩容相当）。回転を速くすると慣性で高い段を越えやすいが横転リスクも上がる。
    {
      key: 'targetOmega',
      label: '目標回転速度',
      min: 2,
      max: 12,
      step: 0.5,
      default: DEFAULT_OMEGA,
      unit: 'rad/s',
    },
  ],
  async run(ctx: MechRunCtx): Promise<MechReplay> {
    const replay = await runSpokeUrchin(
      buildSpokeUrchinOverrides(ctx.params, ctx.torqueCapNm, ctx.course),
      60,
    );
    return recordedQuadReplay(replay, ctx.torqueCapNm, ctx.motorName, ctx.course);
  },
};
