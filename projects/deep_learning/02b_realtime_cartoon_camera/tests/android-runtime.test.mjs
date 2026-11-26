import test from 'node:test';
import assert from 'node:assert/strict';
import { AndroidRuntime } from '../apps/desktop/src/android-runtime.ts';

test('Android stop during native initialization releases the matching session', async () => {
  let ready;
  const calls = [];
  globalThis.window = { __TAURI_INTERNALS__: { invoke: (command, args) => {
    calls.push([command, args]);
    return command.endsWith('|initialize') ? new Promise(resolve => { ready = resolve; }) : Promise.resolve();
  }}};
  try {
    const runtime = new AndroidRuntime();
    const connecting = runtime.connect('mobile-old');
    const rejected = assert.rejects(connecting, /Session stopped/);
    let closed = false;
    const closing = runtime.close().then(() => { closed = true; });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(closed, false);
    ready({ backend: 'ONNX Runtime Android CPU', model: 'MobileNetV3-Large', inputSize: 256, device: 'CPU', coldStartMs: 10 });
    await rejected;
    await closing;
    assert.deepEqual(calls, [
      ['plugin:cartoon|initialize', { payload: { sessionId: 'mobile-old' } }],
      ['plugin:cartoon|stop', { payload: { sessionId: 'mobile-old' } }],
    ]);
  } finally { delete globalThis.window; }
});
