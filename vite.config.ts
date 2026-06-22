import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { open: false },
  // MuJoCo wasm（蛇=pole-climber 用）は import.meta.url で .wasm を解決する emscripten モジュール。
  // esbuild の事前バンドルに含めると wasm URL 解決が壊れるため除外する（single-thread 版＝特別ヘッダ不要）。
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
});
