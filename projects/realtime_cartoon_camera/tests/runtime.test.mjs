import test from 'node:test';
import assert from 'node:assert/strict';
import { Runtime } from '../apps/desktop/src/runtime.ts';

test('stop during worker launch waits for and releases that worker without opening a socket', async () => {
  let finish;
  const calls = [];
  globalThis.window = { __TAURI_INTERNALS__: { invoke: (command, args) => {
    calls.push([command, args]);
    return command === 'start_worker' ? new Promise(resolve => { finish = resolve; }) : Promise.resolve();
  }}};
  const runtime = new Runtime();
  const connecting = runtime.connect('old', 'openvino');
  const rejected = assert.rejects(connecting, /Session stopped/);
  let closed = false;
  const closing = runtime.close().then(() => { closed = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(closed, false);
  finish({ url: 'ws://127.0.0.1:1', token: 'old-token' });
  await rejected;
  await closing;
  assert.deepEqual(calls, [['start_worker', {}], ['stop_worker', { token: 'old-token' }]]);
  delete globalThis.window;
});
