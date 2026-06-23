/// <reference types="node" />
/**
 * S18 きっかけ: 「90° 旋回が制御可能な動作として成立する」ことを**手書きの閉ループ専門家**で実証し、
 * ダッシュボードで見られるよう録画する。
 *
 * 失敗の真因は物理ではなく「PPO 特権教師が急旋回を学べない（2/5）→ 蒸留教師が悪い」だった。S17 は教師さえ
 * 良ければ蒸留が通った。そこで privileged な手書きコントローラ（turn-then-drive: ゴール点への方位誤差が大きい間は
 * スロットルを絞って steer でその場旋回、揃ったらスロットル全開で直進）で 90° 旋回を確実にこなし、これを次セッションの
 * DAgger 教師にする。本スクリプトはこの教師を各方位で録画し、room の RL リプレイ枠へ書き出す（ダッシュボードの
 * 「RL方策」＋方位スライダーでそのまま再生＝教師を目視確認できる）。
 *
 *   node scripts/record-teacher.ts
 *
 * 注意: S17 生徒の room リプレイを上書きする（未コミット）。元に戻すには git checkout public/policies/。
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getServo } from '../src/sim3d/servos.ts';
import { makeRoomWalls, roomTargetWall } from '../src/sim3d/snake3d-dynamics.ts';
import { SnakeEnv, roomSnakeEnvConfig } from '../src/env/SnakeEnv.ts';
import { writeRoomTraceSvg } from './room-trace.ts';

const DEG = Math.PI / 180;
const HEADINGS_DEG = [-90, -45, 0, 45, 90] as const;
const REACH_MARGIN = 1.0; // ゴール点に COM がこの距離 [m] 以内で「到達」とみなす（教師デモの評価用）
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public', 'policies');

// 操舵バイアスの符号（実測でこちらがゴール向きに回る）。err>0（ゴールが左/反時計回り側）→ 正バイアスで左へ回す。
const STEER_SIGN = 1;

function headTag(deg: number): string {
  if (deg === 0) return 'h0';
  return deg > 0 ? `hp${deg}` : `hm${-deg}`;
}
const round4 = (x: number): number => Number(x.toFixed(4));
const round5 = (_k: string, v: unknown): unknown =>
  typeof v === 'number' ? Number(v.toFixed(5)) : v;
const wrapPi = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));

/** 先頭リンク（頭）の yaw 方位 [rad]（currentBodies の quat から）。 */
function headYaw(bodies: { q: number[] }[]): number {
  const [x, y, z, w] = bodies[bodies.length - 1].q;
  const fx = 1 - 2 * (y * y + z * z);
  const fy = 2 * (x * y + z * w);
  return Math.atan2(fy, fx);
}
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
  cfg.enableThrottle = true; // スロットル次元を足す（actDim 17）。S17 既定は off のまま（生徒方策は不変）。
  // 歩容は右旋回バイアスのキラリティを持つ: 右旋回(負バイアス)は鋭く効く（θ-90→-70°）が、左旋回(正バイアス)は
  // 自然右veer を打ち消すだけで鋭く曲がらない（0.12 で under-turn / 0.18 で loop）＝左 90° は本歩容の限界。
  cfg.steerActionScale = 0.12;
  cfg.episodeSteps = 1800;
  const env = await SnakeEnv.create(cfg);
  mkdirSync(OUT_DIR, { recursive: true });

  console.log('=== スクリプト教師（turn-then-drive）各方位の到達・達成方位 ===');
  for (const deg of HEADINGS_DEG) {
    const rad = deg * DEG;
    env.enableRecording(2);
    env.reset(undefined, rad);
    const [sx, sy] = env.startCom;
    const goal = roomTargetWall(sx, sy, rad); // θ光線∩壁の交点（向かう先の点）
    // 走行方位の EMA（頭 yaw の振動を均す）。初期 = 開始時の頭 yaw。
    let yawEma = headYaw(env.currentBodies());
    const alpha = 0.15;
    let minDist = Infinity;
    let cx = sx;
    let cy = sy;
    for (let t = 0; t < cfg.episodeSteps; t++) {
      const bodies = env.currentBodies();
      [cx, cy] = com(bodies);
      const yawNow = headYaw(bodies);
      // 単位ベクトル EMA で角度ラップを回避。
      const ec = (1 - alpha) * Math.cos(yawEma) + alpha * Math.cos(yawNow);
      const es = (1 - alpha) * Math.sin(yawEma) + alpha * Math.sin(yawNow);
      yawEma = Math.atan2(es, ec);
      const bearing = Math.atan2(goal.cy - cy, goal.cx - cx);
      const err = wrapPi(bearing - yawEma); // >0: ゴールが反時計回り側（左）
      const dist = Math.hypot(goal.cx - cx, goal.cy - cy);
      minDist = Math.min(minDist, dist);

      // 連続アーク制御: steer はゴールへの方位誤差に比例（離れていれば最大バイアスに飽和）。スロットルは
      // 方位がずれているほど絞って弧を締め（curvature∝bias/(amp·throttle)）、揃ったら全開で前進する。
      const action = new Float32Array(env.actDim);
      action[env.actDim - 2] = STEER_SIGN * Math.max(-3, Math.min(3, err * 6));
      const align = Math.max(0, 1 - Math.abs(err) / 1.0); // 1=揃った / 0=|err|≥1rad
      action[env.actDim - 1] = -1.8 + (3 - -1.8) * align; // throttle ≈0.32(締めて回す)→≈1.0(前進)
      const r = env.step(action);
      if (r.done) break;
    }
    const replay = env.getReplay();
    const netDx = cx - sx;
    const netDy = cy - sy;
    const achievedDeg = (Math.atan2(netDy, netDx) * 180) / Math.PI;
    const reached = minDist < REACH_MARGIN;
    console.log(
      `  [${String(deg).padStart(3)}°→${goal.wall.padEnd(5)}] ${reached ? '到達✓' : '未到達'} ` +
        `最接近 ${(minDist * 100).toFixed(0)}cm 達成方位 ${achievedDeg.toFixed(0)}°`,
    );

    writeRoomTraceSvg(
      join(OUT_DIR, `room-trace-${headTag(deg)}-${servo.id}.svg`),
      env.getActiveTerrain(),
      env.startCom,
      deg,
      replay,
      reached,
    );
    const meta = {
      mechanism: 'snake3d',
      course: 'room',
      cmdHeadingDeg: deg,
      motor: servo.id,
      motorName: servo.name,
      mass: cfg.sim.totalMass,
      base: 'scripted-teacher', // ← S17 生徒(climb-gait)ではなく手書き教師であることを明示
      policy: 'room',
      reached,
      goalDistM: round4(minDist),
      achievedDeg: round4(achievedDeg),
    };
    writeFileSync(
      join(OUT_DIR, `snake3d-room-${headTag(deg)}-${servo.id}.replay.json`),
      JSON.stringify({ meta, replay }, round5) + '\n',
    );
  }

  rebuildManifest();
  env.dispose();
  console.log('\n  書き出し: snake3d-room-h<deg>-mg996r.replay.json（教師）＋ SVG ＋ manifest');
}

await main();
