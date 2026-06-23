/// <reference types="node" />
/**
 * 「前進しながら左右に向ける」最小デモ。
 *
 * 各関節を約30°（yawAmp=0.52）で振るうねりで前進し（異方抵抗が横振りを前進力に変換）、yaw 関節へ一定の曲げ
 * バイアス（steerActionScale=0.06 ぶん）を一律に足して進路を左右へ曲げる。急旋回（90°）と違いゆるい操舵は
 * 左右とも素直に効くゾーン。各方向を録画して「前進量・正味の向き」を測り、room の方位スライダー枠に書き出す。
 *
 *   node scripts/turn-crawl.ts
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
const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;

// スライダー枠 → 一定の曲げ steer 行動（+で左/反時計、-で右）。heading/30 を ±3 にクランプ（±3=最大バイアス0.06）。
const DIRECTIONS = [-90, -45, 0, 45, 90] as const;
const headTag = (deg: number): string => (deg === 0 ? 'h0' : deg > 0 ? `hp${deg}` : `hm${-deg}`);

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
  cfg.sim.yawAmp = 0.52; // 各関節 ≈30° で振る
  cfg.maxJointDelta = 0; // 残差は使わない（純粋にうねり＋一定曲げバイアスだけ）
  cfg.episodeSteps = 700; // 室内で壁に当たる前に前進＋旋回が見える長さ
  delete cfg.goal; // 到達判定で切らず固定長で回す

  const env = await SnakeEnv.create(cfg);
  mkdirSync(OUT_DIR, { recursive: true });

  console.log('=== 前進しながら左右に向ける（うねり≈30° ＋ 一定曲げバイアス） ===');
  for (const deg of DIRECTIONS) {
    env.enableRecording(2);
    env.reset(undefined, 0);
    const [sx, sy] = env.startCom;
    const steer = Math.max(-3, Math.min(3, deg / 30)); // 一定の曲げ（符号=左右）
    const action = new Float32Array(env.actDim);
    action[env.actDim - 1] = steer;
    let cx = sx;
    let cy = sy;
    for (let t = 0; t < cfg.episodeSteps; t++) {
      env.step(action);
      [cx, cy] = com(env.currentBodies());
    }
    const replay = env.getReplay();
    const fwd = cx - sx; // +x 前進量
    const lateral = cy - sy; // 横ずれ（+y=左）
    const headingDeg = (Math.atan2(cy - sy, cx - sx) * 180) / Math.PI; // 正味の進行方位
    console.log(
      `  steer=${steer.toFixed(1).padStart(4)}（${deg > 0 ? '左' : deg < 0 ? '右' : '直進'}）` +
        `前進 ${(fwd * 100).toFixed(0)}cm 横 ${(lateral * 100).toFixed(0).padStart(4)}cm 正味方位 ${headingDeg.toFixed(0).padStart(4)}°`,
    );
    const meta = {
      mechanism: 'snake3d',
      course: 'room',
      cmdHeadingDeg: deg,
      motor: servo.id,
      motorName: servo.name,
      mass: cfg.sim.totalMass,
      base: 'turn-crawl',
      policy: 'room',
      reached: false,
      goalDistM: 0,
      achievedDeg: Number(headingDeg.toFixed(4)),
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
  console.log('\n  書き出し: snake3d-room-h<deg>-mg996r.replay.json（前進＋操舵デモ）＋ manifest');
}

await main();
