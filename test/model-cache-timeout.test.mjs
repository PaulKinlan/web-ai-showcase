import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  inspectModel,
  ModelCacheTimeoutError,
  remoteRevision,
  settleWithin,
  storageEstimate,
} from "../lib/model-cache.js";

const started = performance.now();
await assert.rejects(
  settleWithin(new Promise(() => {}), 25, "Test operation"),
  (error) =>
    error instanceof ModelCacheTimeoutError &&
    error.operation === "Test operation" &&
    error.timeoutMs === 25,
);
assert.ok(performance.now() - started < 500, "timeout must settle promptly");

Object.defineProperty(globalThis, "self", { configurable: true, value: globalThis });
Object.defineProperty(globalThis, "indexedDB", {
  configurable: true,
  value: { open: () => ({}) }, // Deliberately never fires success/error.
});
Object.defineProperty(globalThis, "caches", {
  configurable: true,
  value: { keys: () => new Promise(() => {}) },
});
await assert.rejects(
  inspectModel({ key: "test", modelId: "example/hung-cache", timeoutMs: 25 }),
  (error) =>
    error instanceof ModelCacheTimeoutError && error.operation === "Local validation record check",
);

Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { storage: { estimate: () => new Promise(() => {}) } },
});
assert.equal(
  await storageEstimate(25),
  null,
  "optional storage detail fails open after its timeout",
);

const originalFetch = globalThis.fetch;
globalThis.fetch = (_, { signal }) =>
  new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
assert.equal(
  await remoteRevision(`example/hung-revision-${Date.now()}`, 25),
  null,
  "a hung revision request is aborted and fails open for offline cached use",
);
globalThis.fetch = originalFetch;

const cacheSource = await readFile(new URL("../lib/model-cache.js", import.meta.url), "utf8");
const inspectSource = cacheSource.slice(
  cacheSource.indexOf("export async function inspectModel"),
  cacheSource.indexOf("export async function checkModelUpdate"),
);
assert.doesNotMatch(
  inspectSource,
  /remoteRevision|fetch\(/,
  "the user-facing local inspection must not perform remote revision discovery",
);
assert.doesNotMatch(
  inspectSource,
  /scanCachedFiles/,
  "the local fast path must not enumerate every request in every origin cache",
);

const loader = await readFile(new URL("../lib/model-loader.js", import.meta.url), "utf8");
assert.match(loader, /Local model availability check/);
assert.match(loader, /inspectModel\(\{ key, timeoutMs: 300 \}\)/);
assert.match(loader, /checkModelUpdate/);
assert.match(loader, /Update available\. You can keep using the cached model/);
assert.match(loader, /Retry local check/);
assert.match(loader, /may download/);
assert.match(loader, /WebGPU availability check/);
assert.match(loader, /Model validation record/);
assert.match(
  loader,
  /setState\("ready"\);[\s\S]*recordValidated/,
  "successful runtime load must enable use before background validation persistence",
);
assert.match(loader, /Clear model cache/);
assert.match(loader, /Clear older model version/);
assert.doesNotMatch(
  loader,
  /catch\s*\{\s*info\s*=\s*\{\s*state:\s*["']absent["']/,
  "a cache-check failure must not be misrepresented as a known-absent model",
);

const resumableLoader = await readFile(
  new URL("../lib/resumable-loader.mjs", import.meta.url),
  "utf8",
);
assert.match(resumableLoader, /Retry local check/);
assert.match(resumableLoader, /inspectModel\(\{ key, timeoutMs: 300 \}\)/);
assert.match(resumableLoader, /checkModelUpdate/);
assert.match(resumableLoader, /void ui\.showStorage\(\)/);
assert.match(resumableLoader, /may download/);
assert.match(resumableLoader, /WebGPU availability check/);
assert.doesNotMatch(
  resumableLoader,
  /catch\s*\{\s*info\s*=\s*\{\s*state:\s*["']absent["']/,
  "the resumable loader must not misrepresent cache-check failures either",
);

console.log("model cache timeout tests passed");
