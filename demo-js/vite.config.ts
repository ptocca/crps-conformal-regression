import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  base: '/crps-conformal-regression/',
  build: {
    outDir: '../docs',
    emptyOutDir: true,
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
});
