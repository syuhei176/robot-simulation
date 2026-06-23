/// <reference types="node" />
/**
 * 「目標方位へ比例操舵で向き直して進む」基盤旋回プリミティブ（直進・左右・急旋回 ±90° まで対称）。
 *
 * 操舵チャンネル（全 yaw 関節への一定曲げバイアス）は**それ自体はカイラル**で、開ループに最大バイアスを
 * 入れると右(-)は ~75°回るのに左(+)は ~36°で頭打ちになる（うねりの利き手）。だが閉ループの**比例制御**にすると
 * 対称になる:
 *   steerAction = clamp(K · err)   （err = 目標方位 − 実進行方位）
 * 誤差が大きい間は最大バイアスで（カイラルでも）目標側へ回り、誤差が縮むほど操舵が弱まって、左右対称な
 * 「ゆるい操舵ゾーン」を通って滑らかに目標方位へ整定する（ラッチ不要・オーバーシュートしない）。整定後は
 * 残差ゼロの操舵が進路保持も兼ねる。実進行方位は COM の直近 W step 変位の向き（うねりの横揺れを均す）。
 *
 * 壁なしアリーナでの整定方位は −90→−91, −45→−46, 0→−1, +45→+44, +90→+89（平均誤差 0.9°）。
 * RL も観測の「目標ヘディング誤差 sin/cos」から同じ操舵チャンネルを叩くので、この比例操舵は方策が獲得しうる
 * 振る舞いそのもの＝基盤の操舵機構が ±90° まで対称に使えることの実証でもある。
 *
 *   node scripts/turn-to.ts
 */
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getServo } from '../src/sim3d/servos.ts';
import { makeRoomWalls } from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, roomSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { writeRoomTraceSvg } from './room-trace.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');
const DEG = Math.PI / 180;
const TARGETS = [-90, -45, 0, 45, 90] as const;
const headTag = (deg: number): string => (deg === 0 ? 'h0' : deg > 0 ? `hp${deg}` : `hm${-deg}`);
const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;
const wrapPi = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));

// 比例操舵のパラメータ。壁なし掃引で K・scale に広く不感（誤差 ~0.9°）だったので頑健側に取る。
const STEER_SCALE = 0.2; // 操舵バイアスの最大値 [rad]（誤差飽和時の曲率）
const GAIN = 5; // 比例ゲイン（行動/rad）。err≈0.6rad で最大バイアスに飽和
const WIN = 40; // 実進行方位を取る COM 窓（≈うねり1周期弱で横揺れを均す）

function com(bodies: { p: number[] }[]): [number, number] {
  let x = 0;
  let y = 0;
  for (const b of bodies) {
    x += b.p[0];
    y += b.p[1];
  }
  return [x / bodies.length, y / bodies.length];
}

function rebuildManifest(): void {
  const entries = readdirSync(OUT_DIR)
    .filter((f) => f.endsWith('.replay.json'))
    .map((file) => {
      const rec = JSON.parse(readFileSync(join(OUT_DIR, file), 'utf8')) as {
        meta: Record<string, unknown>;
      };
      return { file, ...rec.meta };
    });
  writeFileSync(join(OUT_DIR, 'manifest.json'), JSON.stringify(entries, null, 2) + '\n');
}

async function main(): Promise<void> {
  const servo = getServo('mg996r');
  const cfg = roomSnakeEnvConfig({
    terrain: makeRoomWalls(),
    motor: { stiffness: 3, damping: 0.15, maxTorqueNm: servo.stallNm },
  });
  cfg.sim.yawAmp = 0.52;
  cfg.maxJointDelta = 0; // 純粋にうねり＋比例操舵バイアスだけ（残差は使わない）
  cfg.steerActionScale = STEER_SCALE;
  cfg.episodeSteps = 1300; // 旋回(~400step)＋目標方位へ前進してゴール壁に届く長さ
  delete cfg.goal; // 到達で早期終了せず固定長で回す（操舵の整定を見せる）

  const env = await SnakeEnv.create(cfg);
  const steerIdx = env.actDim - 1; // throttle 無し（actDim16）＝最後の次元が操舵
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('=== 目標方位へ比例操舵で向き直して進む（±90° 対称）===');

  for (const deg of TARGETS) {
    const target = deg * DEG;
    env.enableRecording(2);
    env.reset(undefined, 0);
    const [sx, sy] = env.startCom;
    const path: Array<[number, number]> = [];
    const a = new Float32Array(env.actDim);
    for (let t = 0; t < cfg.episodeSteps; t++) {
      const [cx, cy] = com(env.currentBodies());
      path.push([cx, cy]);
      // 実進行方位＝直近 W step の COM 変位の向き（履歴が貯まるまでは +x=0 とみなす）。
      let travel = 0;
      if (path.length > WIN) {
        const [ox, oy] = path[path.length - 1 - WIN];
        if (Math.hypot(cx - ox, cy - oy) > 1e-3) travel = Math.atan2(cy - oy, cx - ox);
      }
      const err = wrapPi(target - travel);
      a[steerIdx] = Math.max(-3, Math.min(3, GAIN * err)); // 比例操舵（飽和付き）
      env.step(a);
    }
    // 達成方位＝壁に達する前の中盤窓 [500,800] の travel 方向（整定後・壁衝突の汚染前）。
    const i0 = Math.min(500, path.length - 1);
    const i1 = Math.min(800, path.length - 1);
    const achieved =
      (Math.atan2(path[i1][1] - path[i0][1], path[i1][0] - path[i0][0]) * 180) / Math.PI;
    const [cx, cy] = path[path.length - 1];
    console.log(
      `  目標 ${String(deg).padStart(3)}° → 達成方位 ${achieved.toFixed(0).padStart(4)}° ` +
        `（前進 ${((cx - sx) * 100).toFixed(0)}cm 横 ${((cy - sy) * 100).toFixed(0)}cm）`,
    );
    const replay = env.getReplay();
    const meta = {
      mechanism: 'snake3d',
      course: 'room',
      cmdHeadingDeg: deg,
      motor: servo.id,
      motorName: servo.name,
      mass: cfg.sim.totalMass,
      base: 'turn-to',
      policy: 'room',
      reached: false,
      goalDistM: 0,
      achievedDeg: Number(achieved.toFixed(4)),
    };
    writeFileSync(
      join(OUT_DIR, `snake3d-room-${headTag(deg)}-${servo.id}.replay.json`),
      JSON.stringify({ meta, replay }, round5) + '\n',
    );
    writeRoomTraceSvg(
      join(OUT_DIR, `room-trace-${headTag(deg)}-${servo.id}.svg`),
      env.getActiveTerrain(),
      env.startCom,
      deg,
      replay,
      false,
    );
  }
  rebuildManifest();
  env.dispose();
  console.log(
    '\n  書き出し: snake3d-room-h<deg>-mg996r.replay.json（比例操舵で向き直して進む）＋ manifest',
  );
}

await main();
