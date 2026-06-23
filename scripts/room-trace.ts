/// <reference types="node" />
/**
 * 部屋ナビ（S17）の真上(x-y)からのトレースを SVG で書き出す共有ヘルパ（Node・canvas 不要）。
 * 学習(train-snake)・蒸留(distill-snake)の両方が「軌跡がゴールへ寄るか」を目視するために使う。
 */
import { writeFileSync } from 'node:fs';
import {
  ROOM,
  roomTargetWall,
  type SnakeTerrainBox,
  type Snake3DReplay,
} from '../src/sim3d/snake3d-dynamics.ts';

/** フレームの body 位置から COM(x,y) を出す。 */
export function frameCom(bodies: Array<{ p: [number, number, number] }>): [number, number] {
  let x = 0;
  let y = 0;
  for (const b of bodies) {
    x += b.p[0];
    y += b.p[1];
  }
  return [x / bodies.length, y / bodies.length];
}

/** 部屋枠・障害物・スタート・ゴールパッチ・θ光線・COM 軌跡を 1 枚の SVG に描いて書き出す。 */
export function writeRoomTraceSvg(
  path: string,
  terrain: SnakeTerrainBox[],
  start: [number, number],
  thetaDeg: number,
  replay: Snake3DReplay,
  reached: boolean,
): void {
  const scale = 90; // px/m
  const pad = 24;
  const wmeters = ROOM.frontX - ROOM.backX;
  const hmeters = 2 * ROOM.halfY;
  const sw = wmeters * scale + 2 * pad;
  const sh = hmeters * scale + 2 * pad;
  const px = (x: number): number => pad + (x - ROOM.backX) * scale;
  const py = (y: number): number => pad + (ROOM.halfY - y) * scale; // y上→svg下
  const fmt = (v: number): string => v.toFixed(1);

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(sw)}" height="${fmt(sh)}" viewBox="0 0 ${fmt(sw)} ${fmt(sh)}">`,
  );
  parts.push(`<rect x="0" y="0" width="${fmt(sw)}" height="${fmt(sh)}" fill="#0e1118"/>`);
  // 部屋の床
  parts.push(
    `<rect x="${fmt(px(ROOM.backX))}" y="${fmt(py(ROOM.halfY))}" width="${fmt(wmeters * scale)}" height="${fmt(hmeters * scale)}" fill="#161b26" stroke="#2a3344"/>`,
  );
  // 地形箱（壁＝濃色・障害物＝橙）
  for (const b of terrain) {
    const isWall = b.halfZ > 0.1;
    parts.push(
      `<rect x="${fmt(px(b.cx - b.halfX))}" y="${fmt(py(b.cy + b.halfY))}" width="${fmt(2 * b.halfX * scale)}" height="${fmt(2 * b.halfY * scale)}" fill="${isWall ? '#3a4659' : '#d9772e'}"/>`,
    );
  }
  // ゴール（θ光線∩壁）とθ光線
  const goal = roomTargetWall(start[0], start[1], (thetaDeg * Math.PI) / 180);
  parts.push(
    `<line x1="${fmt(px(start[0]))}" y1="${fmt(py(start[1]))}" x2="${fmt(px(goal.cx))}" y2="${fmt(py(goal.cy))}" stroke="#5b6b86" stroke-dasharray="6 5"/>`,
  );
  parts.push(
    `<circle cx="${fmt(px(goal.cx))}" cy="${fmt(py(goal.cy))}" r="10" fill="none" stroke="#e0556b" stroke-width="3"/>`,
  );
  // COM 軌跡
  const pts = replay.frames.map((fr) => {
    const [cx, cy] = frameCom(fr.bodies);
    return `${fmt(px(cx))},${fmt(py(cy))}`;
  });
  parts.push(
    `<polyline points="${pts.join(' ')}" fill="none" stroke="#4ea1ff" stroke-width="2.5"/>`,
  );
  // スタート
  parts.push(`<circle cx="${fmt(px(start[0]))}" cy="${fmt(py(start[1]))}" r="6" fill="#39d98a"/>`);
  // ラベル
  parts.push(
    `<text x="${fmt(pad)}" y="${fmt(sh - 8)}" fill="#c8d2e0" font-family="monospace" font-size="14">θ=${thetaDeg}° → ${goal.wall} ${reached ? '到達✓' : '未到達'}</text>`,
  );
  parts.push('</svg>');
  writeFileSync(path, parts.join('\n') + '\n');
}
