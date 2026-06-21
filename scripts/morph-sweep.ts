/// <reference types="node" />
/**
 * 形態探索スイープ — 「SCS0009（安サーボ）+ 軽量の足で合成コースを走破できる機構は何か」を答える。
 *
 *   node scripts/morph-sweep.ts
 *
 * 合成コース（2.5cm 障害物＋3cm×3 階段, 障害物は絶対サイズ）を、脚数(rows×2)・脚長を変えた
 * 多足機体で走らせる。各形態を **SCS0009 cap=0.226 と 実質無限 cap=5.0** の2条件で評価し、
 *   - 前進距離（goal 179cm を越えれば走破）
 *   - 飽和ステップ率 / 整定後ピーク要求トルク
 *   - progress(SCS) ≈ progress(∞) か（=安サーボが真に足りるか。ピーク要求が一過性の整定スパイクか）
 * を出す。結論: トルクは軽量機体では律速でなく、**軽い胴 + 障害物より長い脚 + 多脚 tripod**
 * （= 胴の小さいクモ/ウニ）にすると SCS0009 で踊り場まで全走破する。四足を縮小した機体は
 * 脚が短く（相対的に障害物が巨大化して）18cm で停滞する。
 */
import {
  runQuadrupedGait,
  scaledBodyOverrides,
  makeLegs,
  DEFAULT_QUAD_DYN_CONFIG,
  type QuadDynOverrides,
} from '../src/sim3d/quadruped-dynamics.ts';
import { runUrchinGait, type UrchinOverrides } from '../src/sim3d/urchin-dynamics.ts';
import { COURSES, type CourseSpec } from '../src/sim3d/course.ts';
import { getServo } from '../src/sim3d/servos.ts';

const D = DEFAULT_QUAD_DYN_CONFIG;

interface Morph {
  rows: number; // 2=四足, 3=六足, 4=八足
  legLen: number; // 1脚の伸長長 [m]
  mass: number; // PD/慣性スケールの基準質量
  gait: 'tripod' | 'wave';
}

function overrides(m: Morph, cap: number, course: CourseSpec): QuadDynOverrides {
  const base = scaledBodyOverrides(m.mass, cap);
  const half = m.legLen / 2;
  const duration =
    course.stepRise > 0 ? Math.min(40, Math.max(8, (course.goalX + 0.8) / 0.06 + 4)) : D.duration;
  return {
    ...base,
    trunk: { ...base.trunk, length: m.rows * 0.08 },
    leg: { ...base.leg, thigh: half, shin: half },
    legs: makeLegs(m.rows, m.gait),
    course,
    duration,
    gait: {
      period: 0.9,
      strideM: 0.055,
      liftM: 0.05,
      standM: m.legLen * 0.72,
      stanceDuty: m.gait === 'wave' ? 0.83 : 0.75,
    },
  };
}

function totalMassG(m: Morph, cap: number): number {
  const base = scaledBodyOverrides(m.mass, cap);
  return ((base.trunk!.mass ?? 0) + 2 * m.rows * (base.leg!.segMass ?? 0)) * 1000;
}

async function evalOne(
  m: Morph,
  cap: number,
  course: CourseSpec,
): Promise<{ progressM: number; fell: boolean; satFrac: number; warmPeak: number }> {
  const replay = await runQuadrupedGait(overrides(m, cap, course), 20);
  const sm = replay.summary;
  const steps = Math.ceil(sm.config.duration / sm.config.dt);
  let warmPeak = 0;
  for (const f of replay.frames) if (f.t > 0.5) warmPeak = Math.max(warmPeak, f.diag.demandNm);
  return {
    progressM: sm.forwardDistanceM,
    fell: sm.fell,
    satFrac: sm.saturatedSteps / steps,
    warmPeak,
  };
}

/** 放射「ウニ」形態。脚は円環ハブに放射配置（前後軸なし）、各脚は放射面で揺動。 */
interface UrchinMorph {
  legCount: number; // 脚数（放射）
  legLen: number; // 1脚の伸長長 [m]
  ringRadius: number; // ハブ（hip リング）半径 [m]
  mass: number;
  gait: 'tripod' | 'wave';
  legPlane: 'radial' | 'heading'; // radial=本物 splay ウニ, heading=操舵(ヨーDOF相当)
}

function urchinOverrides(u: UrchinMorph, cap: number, course: CourseSpec): UrchinOverrides {
  const base = scaledBodyOverrides(u.mass, cap);
  const half = u.legLen / 2;
  const duration =
    course.stepRise > 0 ? Math.min(40, Math.max(8, (course.goalX + 0.8) / 0.06 + 4)) : D.duration;
  return {
    ...base,
    leg: { ...base.leg, thigh: half, shin: half },
    legCount: u.legCount,
    ringRadius: u.ringRadius,
    urchinGait: u.gait,
    legPlane: u.legPlane,
    course,
    duration,
    gait: {
      period: 0.9,
      strideM: 0.055,
      liftM: 0.05,
      standM: u.legLen * 0.72,
      stanceDuty: u.gait === 'wave' ? 0.83 : 0.75,
    },
  };
}

function urchinMassG(u: UrchinMorph, cap: number): number {
  const base = scaledBodyOverrides(u.mass, cap);
  return ((base.trunk!.mass ?? 0) + u.legCount * (base.leg!.segMass ?? 0)) * 1000;
}

async function evalUrchin(
  u: UrchinMorph,
  cap: number,
  course: CourseSpec,
): Promise<{ progressM: number; fell: boolean; satFrac: number; warmPeak: number }> {
  const replay = await runUrchinGait(urchinOverrides(u, cap, course), 20);
  const sm = replay.summary;
  const steps = Math.ceil(sm.config.duration / sm.config.dt);
  let warmPeak = 0;
  for (const f of replay.frames) if (f.t > 0.5) warmPeak = Math.max(warmPeak, f.diag.demandNm);
  return {
    progressM: sm.forwardDistanceM,
    fell: sm.fell,
    satFrac: sm.saturatedSteps / steps,
    warmPeak,
  };
}

async function urchinRow(
  label: string,
  u: UrchinMorph,
  course: CourseSpec,
  capScs: number,
): Promise<void> {
  const a = await evalUrchin(u, capScs, course); // SCS0009
  const b = await evalUrchin(u, 5.0, course); // 実質無限
  const traversed = a.progressM >= course.goalX && !a.fell;
  const capLimited = b.progressM - a.progressM > 0.1;
  console.log(
    `${label.padEnd(20)} | ${String(u.legCount).padStart(2)}脚 脚${(u.legLen * 100).toFixed(0)}cm ` +
      `${u.legPlane === 'heading' ? '操舵' : '放射'} ${urchinMassG(u, capScs).toFixed(0).padStart(3)}g | ` +
      `SCS ${(a.progressM * 100).toFixed(0).padStart(4)}cm${a.fell ? '転' : ' '} ` +
      `∞ ${(b.progressM * 100).toFixed(0).padStart(4)}cm${b.fell ? '転' : ' '} | ` +
      `飽和${(a.satFrac * 100).toFixed(0).padStart(3)}% peakτ${a.warmPeak.toFixed(2)}(${(a.warmPeak / capScs).toFixed(1)}x) | ` +
      `${traversed ? '走破' : '未 '} ${capLimited ? 'トルク律速' : 'cap十分'}`,
  );
}

async function row(label: string, m: Morph, course: CourseSpec, capScs: number): Promise<void> {
  const a = await evalOne(m, capScs, course); // SCS0009
  const b = await evalOne(m, 5.0, course); // 実質無限
  const traversed = a.progressM >= course.goalX && !a.fell;
  const capLimited = b.progressM - a.progressM > 0.1;
  console.log(
    `${label.padEnd(22)} | ${String(m.rows * 2).padStart(2)}脚 脚${(m.legLen * 100).toFixed(0)}cm ` +
      `${totalMassG(m, capScs).toFixed(0).padStart(3)}g | ` +
      `SCS ${(a.progressM * 100).toFixed(0).padStart(4)}cm${a.fell ? '転' : ' '} ` +
      `∞ ${(b.progressM * 100).toFixed(0).padStart(4)}cm${b.fell ? '転' : ' '} | ` +
      `飽和${(a.satFrac * 100).toFixed(0).padStart(3)}% peakτ${a.warmPeak.toFixed(2)}(${(a.warmPeak / capScs).toFixed(1)}x) | ` +
      `${traversed ? '走破' : '未 '} ${capLimited ? 'トルク律速' : 'cap十分'}`,
  );
}

async function main(): Promise<void> {
  const course = COURSES.combined();
  const capScs = getServo('scs0009').stallNm;
  console.log(`=== 形態スイープ on 合成コース (goal x=${(course.goalX * 100).toFixed(0)}cm) ===`);
  console.log(
    `SCS0009 cap=${capScs.toFixed(3)} N·m vs 実質無限 cap=5.0 N·m / 軽量基準 mass=0.2kg\n`,
  );
  const mass = 0.2;
  // 四足を長脚化しても 4脚は段で不安定。短脚は障害物に阻まれる。六〜八足 tripod + 長脚が走破する。
  await row(
    '四足 短脚(縮小四足相当)',
    { rows: 2, legLen: 0.09, mass, gait: 'tripod' },
    course,
    capScs,
  );
  await row('四足 長脚', { rows: 2, legLen: 0.15, mass, gait: 'tripod' }, course, capScs);
  await row('六足 短脚', { rows: 3, legLen: 0.09, mass, gait: 'tripod' }, course, capScs);
  await row('六足 長脚(本命)', { rows: 3, legLen: 0.15, mass, gait: 'tripod' }, course, capScs);
  await row('八足 長脚', { rows: 4, legLen: 0.15, mass, gait: 'tripod' }, course, capScs);
  await row('六足 長脚 wave', { rows: 3, legLen: 0.15, mass, gait: 'wave' }, course, capScs);

  // 放射「ウニ」形態 — 本物の splay(放射揺動面) は 1自由度脚では進行軸に揃った前後の脚しか足を世界に
  // 固定できず、斜め・横の脚は横滑り(scrub)するため脚数を増やしても這うだけ（全部17cm前後で停滞）。
  // 揺動面を進行方向へ向けた「操舵ウニ」(=各脚にヨー DOF を足した全方向型に相当) は放射配置のまま歩く。
  console.log(`\n=== 本物の放射ウニ（splay）: 脚数を増やしても scrub で這う on 合成コース ===`);
  for (const n of [6, 8, 10, 12]) {
    const r = 0.06 + 0.005 * (n - 6);
    await urchinRow(
      `放射ウニ ${n}脚`,
      { legCount: n, legLen: 0.15, ringRadius: r, mass, gait: 'tripod', legPlane: 'radial' },
      course,
      capScs,
    );
  }
  console.log(
    `\n=== 操舵ウニ（脚にヨーDOF相当 → 揺動面を進行方向へ）: 放射配置のまま歩く on 平地 ===`,
  );
  for (const n of [6, 8]) {
    await urchinRow(
      `操舵ウニ ${n}脚`,
      { legCount: n, legLen: 0.15, ringRadius: 0.06, mass, gait: 'tripod', legPlane: 'heading' },
      COURSES.flat(),
      capScs,
    );
  }

  // モーター・フロンティア: 本命形態を5機種で。安サーボでどこまで足りるか（cap 昇順）。
  console.log(`\n=== モーター比較: 六足 長脚(本命, mass=0.2) on 合成コース ===`);
  const flagship: Morph = { rows: 3, legLen: 0.15, mass, gait: 'tripod' };
  for (const id of ['sg90', 'mg90s', 'scs0009', 'mg996r', 'sts3215']) {
    const servo = getServo(id);
    const r = await evalOne(flagship, servo.stallNm, course);
    const traversed = r.progressM >= course.goalX && !r.fell;
    console.log(
      `${servo.name.padEnd(22)} | cap ${servo.stallNm.toFixed(3)} ¥${String(servo.priceJpy).padStart(4)} ${servo.feedback ? 'FB有' : 'FB無'} | ` +
        `前進 ${(r.progressM * 100).toFixed(0).padStart(4)}cm${r.fell ? '転' : ' '} 飽和${(r.satFrac * 100).toFixed(0).padStart(3)}% ` +
        `peakτ${(r.warmPeak / servo.stallNm).toFixed(1)}x | ${traversed ? '走破' : '未'}`,
    );
  }
}

await main();
