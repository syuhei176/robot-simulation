/// <reference types="node" />
/**
 * 「目標方位まで円弧で曲がり、揃ったら直進」する閉ループ操舵。
 *
 * 円が描ける（曲率一定で連続旋回）＝必要角度まで弧を進めば任意方位に向ける。本スクリプトはそれを制御化:
 *  - 旋回フェーズ（|方位誤差|大）: ゴール側へ最大の曲げ＋スロットルを絞って弧を締める（= circle.ts の曲率）。
 *  - 直進フェーズ（揃った）: 曲げ0＋スロットル全開で、その方位のまま前進（弱い比例補正で方位を保持）。
 * 各目標方位を録画して room の方位スライダー枠に書き出す（ダッシュボードで「その方位に向き直して進む」が見える）。
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
  cfg.maxJointDelta = 0;
  cfg.steerActionScale = 0.15; // circle.ts と同じ曲率（半径 ≈91cm）
  cfg.enableThrottle = true;
  cfg.episodeSteps = 1300;
  delete cfg.goal;

  const env = await SnakeEnv.create(cfg);
  mkdirSync(OUT_DIR, { recursive: true });
  console.log('=== 目標方位まで弧で曲がって直進（円弧操舵の制御化） ===');

  for (const deg of TARGETS) {
    const target = deg * DEG;
    env.enableRecording(2);
    env.reset(undefined, 0);
    const [sx, sy] = env.startCom;
    const path: Array<[number, number]> = [];
    // 進行方向（COM 速度）でフィードバック。頭 yaw は「向き」だが、うねりの自然 veer で travel≠頭向きなので、
    // 実際に進んでいる方向（直近 W step の COM 変位の向き）を見て、目標に達したら直進にラッチする。
    const W = 50; // ≈うねり1周期。横揺れを均して travel 方向を取り出す窓
    let turning = deg !== 0; // 直進(0°)は最初から直進
    for (let t = 0; t < cfg.episodeSteps; t++) {
      const [cx, cy] = com(env.currentBodies());
      path.push([cx, cy]);
      let travel = 0; // 履歴が貯まるまでは +x（=0）とみなす
      if (path.length > W) {
        const [ox, oy] = path[path.length - 1 - W];
        if (Math.hypot(cx - ox, cy - oy) > 1e-3) travel = Math.atan2(cy - oy, cx - ox);
      }
      const err = wrapPi(target - travel); // >0: もっと左へ
      if (turning && Math.abs(err) < 0.12) turning = false; // 目標方向に達したら直進へラッチ
      const a = new Float32Array(env.actDim);
      if (turning) {
        a[env.actDim - 2] = Math.sign(err) * 3; // 目標側へ最大の曲げ
        a[env.actDim - 1] = -1.3; // throttle ≈0.5（弧を締める）
      } else {
        a[env.actDim - 2] = 0; // 直進（曲げ0）
        a[env.actDim - 1] = 3; // throttle ≈1.0
      }
      env.step(a);
    }
    // 達成方位 = 後半25%（直進区間）の正味進行方向
    const tail = path.slice(Math.floor(path.length * 0.75));
    const [hx, hy] = tail[0];
    const [ex, ey] = tail[tail.length - 1];
    const achieved = (Math.atan2(ey - hy, ex - hx) * 180) / Math.PI;
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
    '\n  書き出し: snake3d-room-h<deg>-mg996r.replay.json（弧で曲がって直進）＋ manifest',
  );
}

await main();
