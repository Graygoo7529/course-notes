import test from 'node:test';
import assert from 'node:assert/strict';
import { createSessionId } from '../apps/desktop/src/session-id.ts';

test('uses the native UUID API with its Crypto receiver when available', () => {
  const source = { randomUUID() { assert.equal(this, source); return 'native-id'; } };
  assert.equal(createSessionId(source), 'native-id');
});

test('WebView without randomUUID generates distinct RFC 4122 v4 IDs from random bytes', () => {
  let value = 0;
  const source = { getRandomValues(bytes) {
    assert.equal(this, source);
    bytes.fill(++value);
    return bytes;
  }};
  const ids = Array.from({ length: 32 }, () => createSessionId(source));
  assert.equal(new Set(ids).size, ids.length);
  for (const id of ids) assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
});

test('without Web Crypto, rapid stop/restart sessions remain distinct even with a frozen clock', () => {
  const original = Date.now;
  Date.now = () => 0;
  try {
    const ids = Array.from({ length: 1000 }, (_, i) => createSessionId(i % 2 ? null : {}));
    assert.equal(new Set(ids).size, ids.length);
    assert.ok(ids.every(id => id.length > 0));
  } finally { Date.now = original; }
});
