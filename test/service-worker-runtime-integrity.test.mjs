// Browser-free tests for the third-party runtime integrity check in sw.js (bead web-ai-showcase-mtu).
//
// These drive the SHIPPED sw.js rather than a reimplementation: the source is read from disk and
// evaluated with a stubbed `self`, Cache Storage, `fetch` and crypto. The only transform applied is a
// DATA substitution - the generated RUNTIME_INTEGRITY block is replaced with a test manifest whose
// hashes are computed here, because the shipped hashes are of real jsDelivr bytes that a network-free
// test cannot reproduce. The generated data is separately checked for drift against
// runtime-integrity.json in the last test, so the substitution cannot hide a data problem.
//
// What is being pinned: verification happens on EVERY serve - a cache hit as well as a network
// response before storing - so an entry poisoned after it was stored is still caught. A response that
// does not match its pinned hash is never served and never stored.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const SW_SOURCE = readFileSync(new URL("../sw.js", import.meta.url), "utf8");
const MANIFEST = JSON.parse(readFileSync(new URL("../runtime-integrity.json", import.meta.url), "utf8"));

const ALLOWED = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";
const UNKNOWN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@9.9.9-not-in-manifest";
const MODEL = "https://huggingface.co/some/model.onnx";
const POISON = "export const pipeline = () => { throw new Error('POISONED'); };";
const CLEAN = "export const pipeline = () => 'clean';";

const encoder = new TextEncoder();
async function sha256(text) {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsResponse(body, status = 200) {
  return new Response(body, { status, headers: { "content-type": "text/javascript" } });
}

// Load sw.js with a stubbed worker environment. Returns the captured fetch listener plus the store.
// `foreign` simulates a matching entry living in a cache this worker does NOT own (for example the
// runtime library's own cache, or a shell cache from an older version). Reading is scoped to the shell
// cache, so a foreign entry must not shadow a clean shell copy; the global caches.match sees it, which
// is what makes this discriminating.
async function loadWorker({ stored = new Map(), foreign = new Map(), network }) {
  const store = new Map(stored);
  const listeners = new Map();
  const calls = { fetch: 0, deletes: [] };
  // `match` is exposed on the opened cache as well as globally, because the worker scopes its runtime
  // lookup to the cache it owns; a stub without it fails rather than exercising that.
  const shellCache = {
    put: async (req, res) => { store.set(typeof req === "string" ? req : req.url, res.clone()); },
    delete: async (req) => { const key = typeof req === "string" ? req : req.url; calls.deletes.push(key); return store.delete(key); },
    match: async (req) => { const hit = store.get(typeof req === "string" ? req : req.url); return hit ? hit.clone() : undefined; },
  };
  const cachesStub = {
    open: async () => shellCache,
    keys: async () => [...store.keys(), ...foreign.keys()],
    // Order matters: the foreign cache is consulted first, so a naive caches.match implementation in
    // sw.js would pick up the poisoned foreign copy instead of the clean shell one.
    match: async (req) => {
      const key = typeof req === "string" ? req : req.url;
      const hit = foreign.get(key) ?? store.get(key);
      return hit ? hit.clone() : undefined;
    },
  };
  const selfStub = {
    addEventListener: (type, fn) => listeners.set(type, fn),
    skipWaiting: () => {},
    clients: { claim: () => Promise.resolve() },
    location: { origin: "https://webai.show" },
  };
  const fetchStub = async (input) => {
    calls.fetch++;
    const url = typeof input === "string" ? input : input.url;
    return network(url);
  };

  // Data substitution only: swap the generated manifest for the test one, USING THE PRODUCTION ENTRY
  // SHAPE ({ sha256, bytes }). An earlier version of this file substituted plain hash STRINGS, which
  // let a string-versus-object comparison in sw.js pass every test while every pinned runtime returned
  // 502 in production. The schema test at the bottom pins the shape so this cannot regress.
  const start = SW_SOURCE.indexOf("// >>> runtime-integrity");
  const end = SW_SOURCE.indexOf("// <<< runtime-integrity");
  assert.ok(start !== -1 && end !== -1, "sw.js must keep its runtime-integrity marker block");
  const testManifest = {
    [ALLOWED]: { sha256: await sha256(CLEAN), bytes: encoder.encode(CLEAN).length },
    [MODEL]: { sha256: "unused", bytes: 1 },
  };
  const withTestManifest = SW_SOURCE.slice(0, start) +
    `const RUNTIME_INTEGRITY = ${JSON.stringify(testManifest)};` +
    SW_SOURCE.slice(end);

  globalThis.self = selfStub;
  globalThis.caches = cachesStub;
  globalThis.fetch = fetchStub;
  new Function(withTestManifest)();

  const onFetch = listeners.get("fetch");
  assert.ok(onFetch, "sw.js registered no fetch listener");
  const request = async (url) => {
    let captured;
    onFetch({ request: new Request(url), respondWith: (p) => { captured = p; } });
    return await captured;
  };
  return { request, store, calls };
}

test("clean cached hit is served and the network is not touched", async () => {
  let networkTouched = false;
  const { request, calls } = await loadWorker({
    stored: new Map([[ALLOWED, jsResponse(CLEAN)]]),
    network: () => { networkTouched = true; return Promise.resolve(jsResponse(CLEAN)); },
  });
  const res = await request(ALLOWED);
  assert.equal(await res.text(), CLEAN);
  assert.equal(networkTouched, false, "a verified cache hit must not hit the network");
  assert.equal(calls.fetch, 0);
});

test("poisoned cached hit is rejected, evicted, and replaced with a clean copy", async () => {
  const { request, store, calls } = await loadWorker({
    stored: new Map([[ALLOWED, jsResponse(POISON)]]),
    network: () => Promise.resolve(jsResponse(CLEAN)),
  });
  const res = await request(ALLOWED);
  assert.equal(await res.text(), CLEAN, "poisoned bytes must never be served");
  assert.deepEqual(calls.deletes, [ALLOWED], "the poisoned cache entry must be evicted");
  // Invalidation: the cache now holds the verified copy, so a second request is served from it.
  const second = await request(ALLOWED);
  assert.equal(await second.text(), CLEAN);
  assert.equal(calls.fetch, 1, "the second request should be served from the refreshed cache");
  assert.equal(await store.get(ALLOWED).text(), CLEAN);
});

test("poisoned network response is refused and never stored", async () => {
  const { request, store, calls } = await loadWorker({
    stored: new Map(),
    network: () => Promise.resolve(jsResponse(POISON)),
  });
  const res = await request(ALLOWED);
  assert.equal(res.status, 502, "bytes that do not match the pin must not be served");
  assert.equal(store.has(ALLOWED), false, "unverified bytes must never be stored");
  assert.deepEqual(calls.deletes, []);
});

test("unknown/unpinned library url is fetched but never persisted or served from cache", async () => {
  // Seed the cache as if an earlier version of the worker had stored it; that copy must not be trusted.
  const { request, store, calls } = await loadWorker({
    stored: new Map([[UNKNOWN, jsResponse(POISON)]]),
    network: () => Promise.resolve(jsResponse(CLEAN)),
  });
  const res = await request(UNKNOWN);
  assert.equal(await res.text(), CLEAN, "an unlisted runtime must come from the network, not from cache");
  assert.equal(await store.get(UNKNOWN).text(), POISON, "the cache entry must not be overwritten or trusted");
  assert.equal(calls.fetch, 1);
});

test("offline with a verified cached copy still serves it", async () => {
  const { request } = await loadWorker({
    stored: new Map([[ALLOWED, jsResponse(CLEAN)]]),
    network: () => Promise.reject(new Error("offline")),
  });
  const res = await request(ALLOWED);
  assert.equal(await res.text(), CLEAN, "offline behaviour for allowed assets must be preserved");
});

test("offline with a poisoned cached copy fails closed instead of serving it", async () => {
  const { request, store } = await loadWorker({
    stored: new Map([[ALLOWED, jsResponse(POISON)]]),
    network: () => Promise.reject(new Error("offline")),
  });
  const res = await request(ALLOWED);
  assert.equal(res.status, 502);
  assert.equal(store.has(ALLOWED), false, "the poisoned entry must be evicted even when offline");
});

test("model host behaviour is unchanged: cache-first, never stored by the worker", async () => {
  const { request, store, calls } = await loadWorker({
    stored: new Map([[MODEL, new Response("model-bytes")]]),
    network: () => Promise.resolve(new Response("network-model")),
  });
  const res = await request(MODEL);
  assert.equal(await res.text(), "model-bytes", "model blobs stay cache-first across all caches");
  assert.equal(calls.fetch, 0);
  assert.equal(await store.get(MODEL).text(), "model-bytes", "the worker must not re-store model blobs");
});

// The cache-scoping fix, made discriminating: a poisoned copy in a cache this worker does not own must
// not shadow the clean copy it does own. A caches.match() lookup would read the foreign poison, fail
// verification, evict the GOOD shell entry, and - offline - return 502 even though a verified copy was
// available. Verified to fail when the lookup is reverted to caches.match().
test("a poisoned entry in a foreign cache does not shadow the clean copy in the shell cache", async () => {
  const { request, store } = await loadWorker({
    stored: new Map([[ALLOWED, jsResponse(CLEAN)]]),
    foreign: new Map([[ALLOWED, jsResponse(POISON)]]),
    network: () => Promise.reject(new Error("offline")),
  });
  const res = await request(ALLOWED);
  assert.equal(res.status, 200, "offline, a verified shell copy must still be served");
  assert.equal(await res.text(), CLEAN, "the foreign poisoned copy must not be served");
  assert.equal(await store.get(ALLOWED).text(), CLEAN, "the clean shell entry must not be evicted");
});

test("the manifest embedded in sw.js matches runtime-integrity.json", () => {
  const start = SW_SOURCE.indexOf("// >>> runtime-integrity");
  const end = SW_SOURCE.indexOf("// <<< runtime-integrity");
  const block = SW_SOURCE.slice(start, end);
  const json = block.slice(block.indexOf("{"), block.lastIndexOf("}") + 1);
  assert.deepEqual(JSON.parse(json), MANIFEST.urls, "regenerate with scripts/runtime-integrity.mjs");
});

// This is the test that would have caught the string-versus-object defect: sw.js reads
// `entry.sha256`, so a manifest of plain strings is not merely untidy, it makes every comparison false
// and 502s every pinned runtime. Schema, not values, is what is pinned here.
test("every manifest entry is an object with a string sha256, a byte count and an explicit policy", () => {
  const urls = Object.keys(MANIFEST.urls);
  assert.ok(urls.length > 0, "the manifest must not be empty");
  for (const url of urls) {
    const entry = MANIFEST.urls[url];
    assert.equal(typeof entry, "object", `${url} must map to an object, not a bare hash string`);
    assert.equal(typeof entry.sha256, "string", `${url}.sha256 must be a string`);
    assert.match(entry.sha256, /^[0-9a-f]{64}$/, `${url}.sha256 must be a lowercase sha256 hex digest`);
    assert.equal(typeof entry.bytes, "number", `${url}.bytes must be a number`);
    // Policy is per URL so the cache decision is reviewable per asset rather than implied by the code
    // path. Membership is the policy: anything absent is pass-through and never persisted.
    assert.equal(entry.policy, "verify-then-cache", `${url}.policy must state the caching decision`);
    // Pinned URLs must be immutable and query-free: a query string or a mutable URL means the bytes can
    // change under the hash, which would fail closed rather than protect anything.
    assert.equal(url.includes("?"), false, `${url} must not carry a query string`);
    assert.equal(url.startsWith("https://cdn.jsdelivr.net/npm/"), true, `${url} must be a pinned cdn URL`);
    assert.match(url, /@[0-9]+\.[0-9]+\.[0-9]+/, `${url} must be version-pinned`);
  }
});

// Guards the key-normalisation fix: a request carrying a query string must still be verified rather
// than falling to the unverified unpinned path, because an exact-key lookup would have missed it.
test("a library url with a query string is still verified rather than treated as unpinned", async () => {
  let networkCalls = 0;
  const { request, calls } = await loadWorker({
    stored: new Map(),
    network: () => { networkCalls++; return Promise.resolve(jsResponse(POISON)); },
  });
  const res = await request(`${ALLOWED}?cachebust=1`);
  assert.equal(res.status, 502, "a queried pinned url must not escape verification");
  assert.equal(networkCalls, 1);
  assert.equal(calls.fetch, 1);
});
