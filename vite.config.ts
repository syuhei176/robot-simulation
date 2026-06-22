import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: './',
  server: { open: false },
  build: {
    // マルチページ: シミュレータ(index) と 材料表/買い物リスト(materials) の2エントリ。
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL('./index.html', import.meta.url)),
        materials: fileURLToPath(new URL('./materials.html', import.meta.url)),
      },
    },
  },
  // MuJoCo wasm（蛇=pole-climber 用）は import.meta.url で .wasm を解決する emscripten モジュール。
  // esbuild の事前バンドルに含めると wasm URL 解決が壊れるため除外する（single-thread 版＝特別ヘッダ不要）。
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
});
