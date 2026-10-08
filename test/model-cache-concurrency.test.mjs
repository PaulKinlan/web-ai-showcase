import assert from "node:assert/strict";
import test from "node:test";
import { scanCacheInventory, scanCachedFiles, inspectModel } from "../lib/model-cache.js";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function mockCaches({ cacheCount = 3, entriesPerCache = 6, matchDelay = () => 2, failAt = null } = {}) {
  const names = Array.from({ length: cacheCount }, (_, i) => `cache-${i}`);
  const requests = names.map((_, i) => Array.from({ length: entriesPerCache }, (_, j) => ({
    url: `https://hf.co/owner/model/file-${i}-${j}`,
  })));
  let inFlight = 0;
  let peak = 0;
  let openInFlight = 0;
  let peakOpen = 0;
  let matchesStarted = 0;
  const cacheObjects = names.map((_, i) => ({
    keys: async () => requests[i],
    match: async (request) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      matchesStarted++;
      try {
        await delay(matchDelay(request));
        if (request.url === failAt) throw new Error("mock cache match failed");
        return { headers: { get: (key) => key === "content-length" ? String(i * 100 + requests[i].indexOf(request)) : null } };
      } finally { inFlight--; }
    },
  }));
  const caches = {
    keys: async () => names,
    open: async (name) => {
      openInFlight++;
      peakOpen = Math.max(peakOpen, openInFlight);
      try { await delay(1); return cacheObjects[names.indexOf(name)]; }
      finally { openInFlight--; }
    },
  };
  return { caches, names, requests, get peak() { return peak; }, get peakOpen() { return peakOpen; }, get matchesStarted() { return matchesStarted; } };
}

async function withCacheMock(mock, fn) {
  const oldSelf = globalThis.self;
  const oldCaches = globalThis.caches;
  globalThis.self = globalThis;
  globalThis.caches = mock.caches;
  try { return await fn(); }
  finally {
    if (oldSelf === undefined) delete globalThis.self;
    else globalThis.self = oldSelf;
    if (oldCaches === undefined) delete globalThis.caches;
    else globalThis.caches = oldCaches;
  }
}

test("inventory retains cache-major order while bounding and using independent matches", async () => {
  const mock = mockCaches({ matchDelay: (request) => request.url.endsWith("-0") ? 16 : 2 });
  const entries = await withCacheMock(mock, () => scanCacheInventory());
  assert.deepEqual(entries.map((entry) => entry.url), mock.requests.flat().map((req) => req.url));
  assert.deepEqual(entries.map((entry) => entry.cacheName), mock.names.flatMap((name) => Array(6).fill(name)));
  assert.deepEqual(entries.map((entry) => entry.bytes), mock.names.flatMap((_, i) => Array.from({ length: 6 }, (_, j) => i * 100 + j)));
  assert.ok(mock.peak > 1, "independent matches should overlap; the serial baseline is red");
  assert.ok(mock.peak <= 4, "no more than four CacheStorage matches may be in flight");
  assert.equal(mock.peakOpen, 1, "cache opens remain sequential");
});

test("inventory rejects on a failed match rather than returning a partial listing", async () => {
  const mock = mockCaches({ failAt: "https://hf.co/owner/model/file-0-0" });
  await assert.rejects(withCacheMock(mock, () => scanCacheInventory()), /mock cache match failed/);
  await delay(10); // Let already-in-flight operations settle before checking no more were scheduled.
  assert.ok(mock.matchesStarted <= 4, `failure must stop new work (started ${mock.matchesStarted})`);
});

async function withValidationRecord(record, fn) {
  const oldIndexedDB = globalThis.indexedDB;
  globalThis.indexedDB = {
    open: () => {
      const request = {};
      queueMicrotask(() => {
        request.result = {
          transaction: () => {
            const tx = {
              objectStore: () => ({
                get: () => {
                  const getRequest = {};
                  queueMicrotask(() => {
                    getRequest.result = record;
                    getRequest.onsuccess?.();
                    queueMicrotask(() => tx.oncomplete?.());
                  });
                  return getRequest;
                },
              }),
            };
            return tx;
          },
        };
        request.onsuccess?.();
      });
      return request;
    },
  };
  try { return await fn(); }
  finally {
    if (oldIndexedDB === undefined) delete globalThis.indexedDB;
    else globalThis.indexedDB = oldIndexedDB;
  }
}

test("validated model retains expected URL ordering and partial state", async () => {
  const mock = mockCaches({ cacheCount: 2, entriesPerCache: 2 });
  const missing = "https://hf.co/owner/model/evicted";
  const record = { files: [mock.requests[1][1].url, missing, mock.requests[0][0].url] };
  const result = await withCacheMock(mock, () => withValidationRecord(record, () =>
    inspectModel({ key: "cached-model", timeoutMs: 75 })
  ));
  assert.equal(result.state, "partial");
  assert.deepEqual(result.missing, [missing]);
  assert.equal(result.cachedFiles, 2);
});

test("750 ms local cache timeout contract stays optimistic without increasing fan-out", async () => {
  const mock = mockCaches({ cacheCount: 3, entriesPerCache: 2 });
  mock.caches.open = async () => { await delay(80); throw new Error("late storage error"); };
  const record = { files: [mock.requests[0][0].url] };
  const start = performance.now();
  const result = await withCacheMock(mock, () => withValidationRecord(record, () =>
    inspectModel({ key: "cached-model", timeoutMs: 15 })
  ));
  assert.equal(result.state, "current");
  assert.equal(result.cachedFiles, 0);
  assert.ok(performance.now() - start < 65, "whole cache check must settle within its deadline, not per cache");
});

test("cache-file scan remains cache-major and local inspect timeout stays optimistic", async () => {
  const mock = mockCaches({ cacheCount: 3, entriesPerCache: 3 });
  const urls = await withCacheMock(mock, () => scanCachedFiles("owner/model"));
  assert.deepEqual(urls, mock.requests.flat().map((req) => req.url));
  // No IndexedDB: a missing validation record remains absent without a network or CacheStorage probe.
  const result = await withCacheMock(mock, () => inspectModel({ key: "never-seen", timeoutMs: 25 }));
  assert.deepEqual(result, { state: "absent", cachedFiles: 0 });
});
