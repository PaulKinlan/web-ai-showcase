// Tests for the runtime → reducer adapters (Task 2b · Phase 2). Each adapter is fed a runtime's native
// progress and its emitted events are run through the REAL download-tracker; we assert the snapshot.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createDownloadTracker } from "../lib/download-tracker.mjs";
import { mediapipeAdapter, transformersAdapter, webllmAdapter } from "../lib/download-adapters.mjs";

test("transformersAdapter forwards real per-file byte progress (byte-weighted aggregate)", () => {
  const t = createDownloadTracker();
  const cb = transformersAdapter((e) => t.ingest(e));
  // Two files, one big one small — Transformers.js progress_callback shape.
  cb({ status: "initiate", file: "onnx/model.onnx", name: "model", total: 900 });
  cb({ status: "initiate", file: "config.json", name: "config", total: 100 });
  cb({ status: "progress", file: "onnx/model.onnx", loaded: 450, total: 900 });
  cb({ status: "done", file: "config.json" });
  let snap = t.snapshot();
  // 450 (of model) + 100 (config done) = 550 of 1000 bytes = 0.55 — NOT the mean of per-file % (0.75).
  assert.equal(snap.aggregate.runtimeOwned, false);
  assert.ok(Math.abs(snap.aggregate.ratio - 0.55) < 1e-9, `ratio=${snap.aggregate.ratio}`);
  assert.equal(snap.aggregate.fileCount, 2);
  cb({ status: "progress", file: "onnx/model.onnx", loaded: 900, total: 900 });
  cb({ status: "done", file: "onnx/model.onnx" });
  cb({ status: "ready" });
  snap = t.snapshot();
  assert.equal(snap.ready, true);
  assert.equal(snap.phase, "ready");
  assert.equal(snap.aggregate.ratio, 1);
});

test("transformersAdapter ignores non-download statuses (e.g. generation updates)", () => {
  const seen = [];
  const cb = transformersAdapter((e) => seen.push(e.status));
  cb({ status: "update", output: "token" });
  cb({ status: "start" });
  cb({ status: "progress", file: "f", loaded: 1, total: 2 });
  assert.deepEqual(seen, ["progress"]);
});

test("webllmAdapter surfaces the runtime's overall fraction as a runtime-owned aggregate (no fake bytes)", () => {
  const t = createDownloadTracker();
  const onProgress = webllmAdapter((e) => t.ingest(e), { label: "Llama-3.2" });
  onProgress({ text: "Fetching param cache[1/50]", progress: 0 });
  let snap = t.snapshot();
  assert.equal(snap.phase, "downloading");
  assert.equal(snap.aggregate.runtimeOwned, true);
  assert.equal(snap.aggregate.runtimeLabel, "Llama-3.2");
  assert.equal(snap.aggregate.fileCount, 0); // no fabricated per-file rows
  assert.equal(snap.aggregate.loadedBytes, 0); // no fabricated bytes
  onProgress({ text: "…", progress: 0.42 });
  snap = t.snapshot();
  assert.ok(Math.abs(snap.aggregate.ratio - 0.42) < 1e-9, `ratio=${snap.aggregate.ratio}`);
  assert.equal(snap.aggregate.indeterminate, false);
  // fraction hits 1 but the model isn't ready yet → initialising, NOT ready
  onProgress({ text: "Loading model into memory", progress: 1 });
  snap = t.snapshot();
  assert.equal(snap.phase, "initialising");
  assert.equal(snap.ready, false);
  // the page signals ready once the engine resolves
  t.ingest({ status: "ready" });
  snap = t.snapshot();
  assert.equal(snap.phase, "ready");
  assert.equal(snap.aggregate.ratio, 1);
});

test("mediapipeAdapter reports an honest indeterminate runtime-owned download (no progress exposed)", () => {
  const t = createDownloadTracker();
  const mp = mediapipeAdapter((e) => t.ingest(e), { label: "BlazeFace" });
  mp.begin();
  let snap = t.snapshot();
  assert.equal(snap.phase, "downloading");
  assert.equal(snap.aggregate.runtimeOwned, true);
  assert.equal(snap.aggregate.indeterminate, true); // no fraction → honest "we can't show a %"
  assert.equal(snap.aggregate.ratio, null);
  assert.equal(snap.aggregate.runtimeLabel, "BlazeFace");
  mp.ready();
  snap = t.snapshot();
  assert.equal(snap.phase, "ready");
  assert.equal(snap.ready, true);
});

test("mediapipeAdapter.error surfaces a model error", () => {
  const t = createDownloadTracker();
  const mp = mediapipeAdapter((e) => t.ingest(e));
  mp.begin();
  mp.error("bundle 404");
  const snap = t.snapshot();
  assert.equal(snap.phase, "error");
  assert.match(snap.error, /404/);
});
