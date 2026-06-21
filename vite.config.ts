import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        playground: 'playground.html',
      },
    },
  },
  server: { open: false },
});
