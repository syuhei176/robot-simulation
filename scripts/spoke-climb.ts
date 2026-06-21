/// <reference types="node" />
/**
 * トゲ車輪ウニ「登れる最大段差」スイープ — スポーク長を変え、単段の壁の高さを上げて
 * どこまで登れるかを測る。脚で真下に立つ歩行型(urchin/quad)の段差限界 ≈3cm と対比する。
 *
 *   node scripts/spoke-climb.ts
 *
 * 各 (スポーク長 × 壁高さ) を SCS0009 cap=0.226 と 実質無限 cap=5.0 の2条件で評価し、
 * 壁の上に乗れたか（hub が壁 x を越え、高さが壁ぶん上がったか）を出す。
 */
import { runSpokeUrchin } from '../src/sim3d/spoke-urchin-dynamics.ts';
import type { CourseSpec } from '../src/sim3d/course.ts';
import { getServo } from '../src/sim3d/servos.ts';

const WALL_X = 0.4;

/** 単段の壁コース: z=0 の床 → x=WALL_X から高さ h の台。 */
function wallCourse(h: number): CourseSpec {
  return {
    id: `wall${Math.round(h * 100)}`,
    name: `単段 ${Math.round(h * 100)}cm`,
    boxes: [
      { cx: 0.5, cz: -0.05, halfX: 2.0, halfZ: 0.05 }, // 床(z=0)
      { cx: WALL_X + 0.8, cz: h / 2, halfX: 0.8, halfZ: h / 2 }, // 台(上面 z=h)
    ],
    profile: [
      [-1.5, 0],
      [WALL_X, 0],
      [WALL_X, h],
      [WALL_X + 1.6, h],
    ],
    plateauStartX: WALL_X + 0.3,
    goalX: WALL_X + 0.3,
    stepRise: h,
    stepForward: 0.1,
    defaultFriction: 0.95,
  };
}

interface Res {
  climbed: boolean;
  reach: number; // 到達した最遠 x（startX 基準）[m]
  rose: number; // hub の上昇量 [m]
  fell: boolean;
  satFrac: number;
}

async function evalClimb(spokeLen: number, h: number, cap: number): Promise<Res> {
  const replay = await runSpokeUrchin(
    {
      spokeCount: 8,
      spokeLen,
      mass: 0.25,
      torqueCapNm: cap,
      targetOmega: 6,
      course: wallCourse(h),
      duration: 9,
    },
    30,
  );
  const f = replay.frames;
  const z0 = f.length ? f[0].diag.trunkZ : 0;
  let reach = 0;
  let rose = 0;
  for (const fr of f) {
    reach = Math.max(reach, fr.diag.forwardX);
    rose = Math.max(rose, fr.diag.trunkZ - z0);
  }
  // 登坂成功 = 壁(WALL_X)を十分越えた距離まで到達（台の上を 0.4m 以上進めた）かつ転倒なし。
  const reachWorldX = reach - 0.3; // forwardX = hubX - startX(-0.3) → worldX = reach - 0.3
  const steps = Math.ceil(replay.summary.config.duration / replay.summary.config.dt);
  return {
    climbed: reachWorldX > WALL_X + 0.4 && !replay.summary.fell,
    reach,
    rose,
    fell: replay.summary.fell,
    satFrac: replay.summary.saturatedSteps / steps,
  };
}

async function main(): Promise<void> {
  const capScs = getServo('scs0009').stallNm;
  console.log(`=== トゲ車輪ウニ 登坂スイープ（8スポーク・250g・単段の壁）===`);
  console.log(
    `SCS0009 cap=${capScs.toFixed(3)} N·m vs 実質無限 5.0 N·m / 脚で立つ歩行型の段差限界≈3cm\n`,
  );

  for (const spokeLen of [0.1, 0.15, 0.2]) {
    console.log(`-- スポーク長 ${(spokeLen * 100).toFixed(0)}cm --`);
    let maxClimbScs = 0;
    let maxClimbInf = 0;
    for (const h of [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]) {
      const a = await evalClimb(spokeLen, h, capScs);
      const b = await evalClimb(spokeLen, h, 5.0);
      if (a.climbed) maxClimbScs = h;
      if (b.climbed) maxClimbInf = h;
      console.log(
        `  壁${(h * 100).toFixed(0).padStart(2)}cm | ` +
          `SCS ${a.climbed ? '登◎' : '×  '} reach${(a.reach * 100).toFixed(0).padStart(3)}cm 上昇${(a.rose * 100).toFixed(0).padStart(2)}cm 飽和${(a.satFrac * 100).toFixed(0).padStart(3)}%${a.fell ? ' 転' : ''} | ` +
          `∞ ${b.climbed ? '登◎' : '×  '} reach${(b.reach * 100).toFixed(0).padStart(3)}cm`,
      );
    }
    console.log(
      `  → 登坂上限: SCS0009 ${(maxClimbScs * 100).toFixed(0)}cm / 無限トルク ${(maxClimbInf * 100).toFixed(0)}cm ` +
        `(スポーク長比 SCS ${(maxClimbScs / spokeLen).toFixed(2)} / ∞ ${(maxClimbInf / spokeLen).toFixed(2)})\n`,
    );
  }
}

await main();
