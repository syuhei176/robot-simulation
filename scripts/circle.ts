/// <reference types="node" />
/**
 * 「円を描けるか？」の実証。壁なしの広い床で、一定の曲げバイアス（右）＋うねり＋少し絞ったスロットルで
 * 長く回す。曲率一定なので円弧を描き続け、十分なステップで閉じて円になる。
 * 円が描ける＝任意方位に向けられる（円弧を必要角度まで進んで止めるだけ）。
 *
 *   node scripts/circle.ts
 */
import { SnakeEnv, roomSnakeEnvConfig } from '../src/env/SnakeEnv.ts';

const motor = { stiffness: 3, damping: 0.15, maxTorqueNm: 1.8 };

function com(env: SnakeEnv): [number, number] {
  const b = env.currentBodies();
  let x = 0;
  let y = 0;
  for (const s of b) {
    x += s.p[0];
    y += s.p[1];
  }
  return [x / b.length, y / b.length];
}

async function main(): Promise<void> {
  const cfg = roomSnakeEnvConfig({ terrain: [], motor }); // 壁なし＝広い床（円を遮らない）
  cfg.sim.yawAmp = 0.52; // うねり ≈30°
  cfg.maxJointDelta = 0;
  cfg.steerActionScale = 0.15; // 円を小さく（曲率↑）するため曲げを強め。右（強い方向）に固定するのでキラリティ不問
  cfg.enableThrottle = true;
  cfg.episodeSteps = 4000;
  delete cfg.goal;

  const env = await SnakeEnv.create(cfg);
  env.reset(undefined, 0);
  const action = new Float32Array(env.actDim);
  action[env.actDim - 2] = -3; // 右へ一定の曲げ（最大バイアス）
  action[env.actDim - 1] = -1.3; // throttle ≈0.5（前進を少し絞って円を締める）

  const path: Array<[number, number]> = [];
  let prevHeading = 0;
  let totalTurn = 0; // 累積回転[rad]（360°超えたら一周）
  let [px, py] = env.startCom;
  for (let t = 0; t < cfg.episodeSteps; t++) {
    env.step(action);
    const [cx, cy] = com(env);
    if (t % 25 === 0) path.push([Number(cx.toFixed(3)), Number(cy.toFixed(3))]);
    if (t > 0) {
      const d = Math.hypot(cx - px, cy - py);
      if (d > 1e-4) {
        const h = Math.atan2(cy - py, cx - px);
        let dh = h - prevHeading;
        while (dh > Math.PI) dh -= 2 * Math.PI;
        while (dh < -Math.PI) dh += 2 * Math.PI;
        if (t > 50) totalTurn += dh; // 初期の整定を除く
        prevHeading = h;
      }
    }
    px = cx;
    py = cy;
  }
  // 円の半径推定（軌跡の外接ボックスの平均半径）
  const xs = path.map((p) => p[0]);
  const ys = path.map((p) => p[1]);
  const r = ((Math.max(...xs) - Math.min(...xs)) / 2 + (Math.max(...ys) - Math.min(...ys)) / 2) / 2;
  console.log(
    `累積回転 ${((totalTurn * 180) / Math.PI).toFixed(0)}°（${(Math.abs(totalTurn) / (2 * Math.PI)).toFixed(2)} 周）・推定半径 ≈${(r * 100).toFixed(0)}cm`,
  );
  console.log('軌跡(COM, 25step毎): ' + JSON.stringify(path));
  env.dispose();
}

await main();
