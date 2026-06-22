/**
 * MuJoCo wasm エンジンのローダ（シングルトン）。
 *
 * 蛇（pole-climber）だけ MuJoCo wasm で動かす（他機構は Rapier）。9MB の wasm を一度だけロードして使い回す。
 * single-thread 版（`@mujoco/mujoco`）は SharedArrayBuffer / COOP・COEP ヘッダ不要なので GitHub Pages で動く。
 * ローダは `new URL('mujoco.wasm', import.meta.url)` で wasm を解決するので、Vite が base パス込みで配置する。
 */
import loadMujoco from '@mujoco/mujoco';

/** @mujoco/mujoco の `loadMujoco()` が返す MuJoCo モジュール型。 */
export type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

let modulePromise: Promise<MujocoModule> | null = null;

/** MuJoCo wasm モジュールを取得（初回のみロード、以降はキャッシュ）。 */
export function getMujoco(): Promise<MujocoModule> {
  modulePromise ??= loadMujoco();
  return modulePromise;
}
