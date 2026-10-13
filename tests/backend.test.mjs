import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../evera/api.js', import.meta.url), 'utf8');
const { Backend } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('poll clears its timeout after network failure and never simulates after real connection', async () => {
  const old = { fetch: globalThis.fetch, location: globalThis.location,
    setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
  let cleared = 0;
  globalThis.location = { search: '' };
  globalThis.setTimeout = () => 42;
  globalThis.clearTimeout = (id) => { assert.equal(id, 42); cleared++; };
  try {
    const backend = new Backend(() => ({}), () => {});
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ state: 'idle', running: false }) });
    await backend.poll();
    globalThis.fetch = async () => { throw new Error('offline'); };
    await backend.poll();
    await backend.poll();
    assert.equal(cleared, 3);
    assert.equal(backend.status.state, 'offline');
    assert.equal(backend.status.sim, false);
  } finally {
    Object.assign(globalThis, old);
  }
});
