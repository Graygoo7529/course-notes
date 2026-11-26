import test from 'node:test';
import assert from 'node:assert/strict';
import { LatestFrameScheduler } from '../apps/desktop/src/scheduler.ts';

const flush = () => new Promise(resolve => setImmediate(resolve));
test('only one in-flight frame; busy frames never create a backlog', async () => {
  let finish; let calls = 0; const displayed = [];
  const q = new LatestFrameScheduler('s', 1, meta => { calls++; return new Promise(resolve => finish = () => resolve({meta})); },
    async result => { displayed.push(result.meta.frameId); }, error => { throw error; });
  q.submit({sessionId:'s',frameId:1,configRevision:1},new Blob());
  for(let i=2;i<100;i++) q.submit({sessionId:'s',frameId:i,configRevision:1},new Blob());
  assert.equal(calls,1); assert.equal(q.dropped,98);
  finish(); await flush();
  assert.deepEqual(displayed,[1]); assert.equal(calls,1); assert.equal(q.busy,false);
});
test('pause, config and session discard stale results', async () => {
  let finish; let displayed=0;
  const q=new LatestFrameScheduler('s',1,meta=>new Promise(resolve=>finish=()=>resolve({meta})),async()=>{displayed++;},error=>{throw error;});
  q.submit({sessionId:'s',frameId:1,configRevision:1},new Blob()); q.configure(2); finish(); await flush();
  assert.equal(displayed,0);
  q.submit({sessionId:'old',frameId:2,configRevision:2},new Blob());finish();await flush();assert.equal(displayed,0);
  q.submit({sessionId:'s',frameId:3,configRevision:2},new Blob());const paused=q.pause();finish();await paused;assert.equal(displayed,0);
  q.resume();q.submit({sessionId:'s',frameId:4,configRevision:2},new Blob());finish();await flush();assert.equal(displayed,1);
});
