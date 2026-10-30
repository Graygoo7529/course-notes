import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
function localWorker(): Plugin {
  let child: ChildProcess | undefined;
  let ready: Promise<string> | undefined;
  return { name: 'local-cartoon-worker', apply: 'serve', configureServer(server) {
    server.middlewares.use('/api/runtime', (req, res) => {
      const expected = new URL(`http://${req.headers.host}`);
      if (!['localhost', '127.0.0.1'].includes(expected.hostname) ||
          (req.headers.origin && req.headers.origin !== expected.origin) || req.method !== 'GET') {
        res.statusCode = 403; res.end(); return;
      }
      if (!ready) {
        child = spawn(process.env.CARTOON_PYTHON ?? 'C:/Anaconda3/envs/d2l/python.exe',
          ['-X', 'utf8', resolve(root, 'python/worker.py'), '--origin', 'http://127.0.0.1:1420', '--origin', 'http://localhost:1420'],
          { cwd: root, windowsHide: true, stdio: ['pipe', 'pipe', 'inherit'] });
        ready = new Promise((ok, fail) => {
          const timeout = setTimeout(() => fail(new Error('Worker start timeout')), 15000);
          const reader = createInterface({ input: child!.stdout! });
          reader.once('line', line => { clearTimeout(timeout); reader.close(); ok(line); });
          child!.once('error', fail);
        });
        child.once('exit', () => { ready = undefined; child = undefined; });
      }
      void ready.then(info => { res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store'); res.end(info); })
        .catch(error => { ready = undefined; child?.kill(); res.statusCode = 503; res.end(JSON.stringify({ error: String(error) })); });
    });
    server.httpServer?.once('close', () => { child?.stdin?.end(); child?.kill(); });
    process.once('exit', () => child?.kill());
  }};
}
export default defineConfig({
  root, plugins: [react(), localWorker()], clearScreen: false,
  server: { host: '127.0.0.1', strictPort: true, port: 1420,
    watch: { ignored: ['**/src-tauri/target/**', '**/models/**', '**/.cache/**'] } },
  build: { outDir: resolve(root, 'dist'), emptyOutDir: true },
});
