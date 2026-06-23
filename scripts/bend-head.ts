/// <reference types="node" />
/**
 * 最小デモ: うねり（クネクネ）なしで、頭側の関節だけを曲げて頭の向きを変える。
 *
 * 歩容（セルペノイド波）を完全に切り（yawAmp=pitchAmp=0）、all-yaw（全関節が水平軸）にして、頭に近い数関節へ
 * 一定の目標角を与えるだけ。前進推進が無いのでその場で頭側が弧を描いて頭が回る＝「ただ頭を曲げる」。
 * room の h0 リプレイ枠に書き出すので、ダッシュボード `?course=room&mode=rl&heading=0` でそのまま見られる。
 *
 *   node scripts/bend-head.ts
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

function headYaw(bodies: { q: number[] }[]): number {
  const [x, y, z, w] = bodies[bodies.length - 1].q;
  return Math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z));
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
  cfg.sim.yawAmp = 0; // うねり OFF（前進推進なし）
  cfg.sim.pitchAmp = 0;
  cfg.sim.pattern = 'all-yaw'; // 全関節が水平軸＝頭を水平に曲げる
  cfg.maxJointDelta = 0.26; // 1 関節あたりの曲げ可動幅 [rad]（前方7関節で ≈90° の弧になる）
  cfg.episodeSteps = 400;
  delete cfg.goal; // 到達判定で早期終了しないように（ただ曲げて保持するだけ）

  const env = await SnakeEnv.create(cfg);
  mkdirSync(OUT_DIR, { recursive: true });
  env.enableRecording(2);
  env.reset(undefined, 0);

  const nJoints = env.actDim - 1; // 最後の1次元は steer（ここでは使わない）
  const BEND_JOINTS = 7; // 頭に近い側（高インデックス）の関節をこれだけ曲げる＝前半分が弧を描く
  const yaw0 = headYaw(env.currentBodies());

  const action = new Float32Array(env.actDim);
  for (let t = 0; t < cfg.episodeSteps; t++) {
    const ramp = Math.min(1, t / 150); // 150 step かけて目標角まで立ち上げ、以降は保持
    action.fill(0);
    for (let k = 0; k < BEND_JOINTS; k++) {
      action[nJoints - 1 - k] = 3 * ramp; // tanh 飽和＝maxJointDelta いっぱいに曲げる
    }
    env.step(action);
  }

  const replay = env.getReplay();
  const yaw1 = headYaw(env.currentBodies());
  console.log(
    `頭の向き: ${((yaw0 * 180) / Math.PI).toFixed(0)}° → ${((yaw1 * 180) / Math.PI).toFixed(0)}° ` +
      `（${BEND_JOINTS}関節 × ${cfg.maxJointDelta}rad ＝ 約${((BEND_JOINTS * cfg.maxJointDelta * 180) / Math.PI).toFixed(0)}° 曲げ）`,
  );

  const meta = {
    mechanism: 'snake3d',
    course: 'room',
    cmdHeadingDeg: 0,
    motor: servo.id,
    motorName: servo.name,
    mass: cfg.sim.totalMass,
    base: 'bend-head-demo',
    policy: 'room',
    reached: false,
    goalDistM: 0,
    achievedDeg: Number(((yaw1 * 180) / Math.PI).toFixed(4)),
  };
  writeFileSync(
    join(OUT_DIR, `snake3d-room-h0-${servo.id}.replay.json`),
    JSON.stringify({ meta, replay }, round5) + '\n',
  );
  writeRoomTraceSvg(
    join(OUT_DIR, `room-trace-h0-${servo.id}.svg`),
    env.getActiveTerrain(),
    env.startCom,
    0,
    replay,
    false,
  );
  rebuildManifest();
  env.dispose();
  console.log('  書き出し: snake3d-room-h0-mg996r.replay.json（頭曲げデモ）＋ manifest');
}

await main();
