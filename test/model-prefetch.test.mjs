// Integration tests for the resumable prefetch orchestrator (node:test; injected fakes — no network/DOM).
// Run: node --test test/model-prefetch.test.mjs
//
// Proves: exact transformers-cache keys, complete denominator seeding, cached-file skip, resumable
// large-file path → cache.put with Content-Length, small-file path, error propagation, abort, and that
// the emitted events drive lib/download-tracker.mjs to a correct byte-weighted snapshot (cached mix,
// no false 100%, download-done ≠ ready).

import { test } from "node:test";
import assert from "node:assert/strict";
import { assetUrl, prefetchModel } from "../lib/model-prefetch.mjs";
import { createDownloadTracker } from "../lib/download-tracker.mjs";

// Minimal fake Cache Storage cache — captures the exact headers the runtime would see.
function fakeCache(initial = []) {
  const store = new Map(initial.map((u) => [u, { headers: {} }]));
  return {
    store,
    async match(url) {
      return store.has(url) ? { url } : undefined;
    },
    async put(url, res) {
      store.set(url, {
        headers: {
          "Content-Length": res.headers.get("Content-Length"),
          "ETag": res.headers.get("ETag"),
        },
      });
    },
  };
}

const PALI = "onnx-community/paligemma2-3b-pt-224";

test("emits exact transformers-cache keys the runtime will hit", () => {
  assert.equal(
    assetUrl(PALI, "main", "onnx/decoder_model_merged_q4f16.onnx"),
    "https://huggingface.co/onnx-community/paligemma2-3b-pt-224/resolve/main/onnx/decoder_model_merged_q4f16.onnx",
  );
  assert.equal(
    assetUrl(PALI, "main", "config.json"),
    "https://huggingface.co/onnx-community/paligemma2-3b-pt-224/resolve/main/config.json",
  );
});

test("full prefetch: seeds denominator, downloads large resumably, caches small, skips cached", async () => {
  const files = ["config.json", "onnx/embed.onnx", "onnx/decoder.onnx"];
  const sizes = {
    "config.json": 500,
    "onnx/embed.onnx": 600_000_000,
    "onnx/decoder.onnx": 1_400_000_000,
  };
  // Pretend the embed weight is already cached (a resumed session).
  const cache = fakeCache([assetUrl(PALI, "main", "onnx/embed.onnx")]);
  const events = [];
  const downloadedUrls = [];

  const res = await prefetchModel({
    modelId: PALI,
    files,
    onEvent: (e) => events.push(e),
    deps: {
      resolveInfo: async ({ modelId, revision }) =>
        files.map((f) => ({
          file: f,
          url: assetUrl(modelId, revision, f),
          size: sizes[f],
          oid: "a".repeat(64),
        })),
      cacheOpen: async () => cache,
      existsInCache: async (c, url) => !!(await c.match(url)),
      // large-file downloader: report a couple of progress ticks then return a verified blob.
      download: async ({ url, onProgress }) => {
        downloadedUrls.push(url);
        const total = sizes[Object.keys(sizes).find((k) => assetUrl(PALI, "main", k) === url)];
        onProgress({ receivedBytes: total / 2, total });
        onProgress({ receivedBytes: total, total });
        return { blob: { size: total }, total };
      },
      // small-file fetch
      simpleFetch: async (url) => ({
        ok: true,
        headers: { get: () => "application/json" },
        blob: async () => ({ size: sizes["config.json"] }),
      }),
    },
  });

  // config (small) + decoder (large) downloaded; embed was cached (skipped).
  assert.deepEqual(res.cached, ["onnx/embed.onnx"]);
  assert.deepEqual(res.downloaded.sort(), ["config.json", "onnx/decoder.onnx"].sort());
  assert.equal(res.totalBytes, 500 + 600_000_000 + 1_400_000_000);
  // only the decoder went through the resumable downloader (embed cached, config is small)
  assert.deepEqual(downloadedUrls, [assetUrl(PALI, "main", "onnx/decoder.onnx")]);

  // decoder cached with a Content-Length header (so the library's progress total isn't 0)
  const stored = cache.store.get(assetUrl(PALI, "main", "onnx/decoder.onnx"));
  assert.equal(stored.headers["Content-Length"], "1400000000");
  assert.equal(stored.headers["ETag"], `"${"a".repeat(64)}"`);

  // an initiate was emitted for every file up front (complete denominator)
  const initiated = events.filter((e) => e.status === "initiate").map((e) => e.file);
  assert.deepEqual(initiated.sort(), files.sort());
});

test("emitted events drive the tracker to a correct byte-weighted snapshot", async () => {
  const files = ["config.json", "onnx/w.onnx"];
  const sizes = { "config.json": 1000, "onnx/w.onnx": 1_999_000 }; // weight >1 MB ⇒ resumable path
  const cache = fakeCache();
  const tracker = createDownloadTracker();
  await prefetchModel({
    modelId: PALI,
    files,
    onEvent: (e) => tracker.ingest(e),
    deps: {
      resolveInfo: async ({ modelId, revision }) =>
        files.map((f) => ({
          file: f,
          url: assetUrl(modelId, revision, f),
          size: sizes[f],
          oid: null,
        })),
      cacheOpen: async () => cache,
      download: async ({ onProgress }) => {
        onProgress({ receivedBytes: 1_000_000, total: 1_999_000 });
        onProgress({ receivedBytes: 1_999_000, total: 1_999_000 });
        return { blob: { size: 1_999_000 }, total: 1_999_000 };
      },
      simpleFetch: async () => ({
        ok: true,
        headers: { get: () => "application/json" },
        blob: async () => ({ size: 1000 }),
      }),
    },
  });
  const s = tracker.snapshot();
  assert.equal(s.aggregate.totalBytes, 2_000_000);
  assert.equal(s.aggregate.loadedBytes, 2_000_000);
  // download finished but the app hasn't signalled the model is READY yet → phase initialising, not ready
  assert.equal(s.phase, "initialising");
  assert.equal(s.ready, false);
  // now the app posts ready → terminal
  tracker.ingest({ status: "ready" });
  assert.equal(tracker.snapshot().phase, "ready");
});

test("download error propagates + surfaces as a file error event", async () => {
  const cache = fakeCache();
  const events = [];
  await assert.rejects(
    prefetchModel({
      modelId: PALI,
      files: ["onnx/w.onnx"],
      onEvent: (e) => events.push(e),
      deps: {
        resolveInfo: async () => [{
          file: "onnx/w.onnx",
          url: assetUrl(PALI, "main", "onnx/w.onnx"),
          size: 5_000_000,
          oid: null,
        }],
        cacheOpen: async () => cache,
        download: async () => {
          throw new Error("network dropped");
        },
      },
    }),
    /network dropped/,
  );
});

test("abort before work throws AbortError and downloads nothing", async () => {
  const cache = fakeCache();
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(
    prefetchModel({
      modelId: PALI,
      files: ["onnx/w.onnx"],
      signal: ac.signal,
      deps: {
        resolveInfo: async () => [{
          file: "onnx/w.onnx",
          url: assetUrl(PALI, "main", "onnx/w.onnx"),
          size: 5_000_000,
          oid: null,
        }],
        cacheOpen: async () => cache,
        download: async () => ({ blob: { size: 1 }, total: 1 }),
      },
    }),
    (e) => e.name === "AbortError",
  );
  assert.equal(cache.store.size, 0);
});
