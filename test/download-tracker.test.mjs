// Deterministic tests for the multi-file download reducer (node:test, no browser/DOM).
// Run: node --test test/download-tracker.test.mjs
//
// Covers the brief's cases: concurrent interleaved files, late discovery (small manifest 100% then
// weights — must NOT imply the whole model is 100%/ready), cached+downloaded mix, duplicate +
// out-of-order callbacks, unknown sizes, zero-byte/metadata, errors, byte-weighted aggregate (not a
// mean of percentages), honest indeterminate state, and the phase machine incl. download-done ≠ ready.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createDownloadTracker, formatBytes, PHASE_LABEL } from "../lib/download-tracker.mjs";

const P = (status, file, extra = {}) => ({ status, file, name: file, ...extra });

test("byte-weighted aggregate, not a mean of per-file percentages", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "config.json", { total: 100 }));
  t.ingest(P("initiate", "onnx/model.onnx", { total: 900 }));
  // config 100% (100/100), weights 0% (0/900). A naive mean would be 50%. Byte-weighted = 100/1000 = 10%.
  t.ingest(P("progress", "config.json", { loaded: 100, total: 100 }));
  const s = t.ingest(P("progress", "onnx/model.onnx", { loaded: 0, total: 900 }));
  assert.equal(s.aggregate.loadedBytes, 100);
  assert.equal(s.aggregate.totalBytes, 1000);
  assert.ok(Math.abs(s.aggregate.ratio - 0.1) < 1e-9, `ratio=${s.aggregate.ratio}`);
});

test("small manifest reaching 100% never implies the whole model is ready", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "config.json", { total: 50 }));
  t.ingest(P("progress", "config.json", { loaded: 50, total: 50 }));
  let s = t.ingest(P("done", "config.json"));
  // config complete, but no weights yet and not ready → phase must NOT be ready, ratio must not be 1.
  assert.notEqual(s.phase, "ready");
  assert.notEqual(s.aggregate.ratio, 1);
  // Late-discovered big weight joins WITHOUT resetting to 0 and drops the ratio honestly.
  t.ingest(P("initiate", "onnx/model.onnx", { total: 950 }));
  s = t.ingest(P("progress", "onnx/model.onnx", { loaded: 0, total: 950 }));
  assert.equal(s.aggregate.loadedBytes, 50, "no bogus 100→0 reset of already-downloaded bytes");
  assert.ok(s.aggregate.ratio < 0.1);
  assert.equal(s.phase, "downloading");
});

test("download-done is a separate terminal state from model-ready", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "onnx/model.onnx", { total: 100 }));
  t.ingest(P("progress", "onnx/model.onnx", { loaded: 100, total: 100 }));
  let s = t.ingest(P("done", "onnx/model.onnx"));
  assert.equal(s.phase, "initialising", "all files downloaded but model not built yet");
  assert.equal(s.ready, false);
  s = t.ingest({ status: "ready" });
  assert.equal(s.phase, "ready");
  assert.equal(s.ready, true);
});

test("cache-read progress after a file completes does NOT regress it (phase stays initialising)", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "onnx/w.onnx", { total: 100 }));
  t.ingest(P("progress", "onnx/w.onnx", { loaded: 100, total: 100 }));
  t.ingest(P("done", "onnx/w.onnx")); // complete → phase initialising
  // from_pretrained's cache read streams `progress` again for the already-complete weight:
  const s = t.ingest(P("progress", "onnx/w.onnx", { loaded: 100, total: 100 }));
  assert.equal(s.files[0].state, "complete", "stays complete, not bounced to downloading");
  assert.equal(s.phase, "initialising", "phase must not bounce Preparing→Downloading");
});

test("cached + downloaded mix: cache-hit files complete without progress", () => {
  const t = createDownloadTracker();
  // A cached file often emits done (or nothing) with no progress.
  t.ingest(P("initiate", "config.json"));
  t.ingest(P("done", "config.json")); // cache hit — no bytes reported
  t.ingest(P("initiate", "onnx/model.onnx", { total: 200 }));
  const s = t.ingest(P("progress", "onnx/model.onnx", { loaded: 100, total: 200 }));
  const cfg = s.files.find((f) => f.id === "config.json");
  assert.equal(cfg.state, "cached");
  assert.equal(s.aggregate.complete, 1);
  assert.equal(s.aggregate.loadedBytes, 100);
});

test("concurrent interleaved files update independently", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "a", { total: 100 }));
  t.ingest(P("initiate", "b", { total: 100 }));
  t.ingest(P("progress", "a", { loaded: 30, total: 100 }));
  t.ingest(P("progress", "b", { loaded: 70, total: 100 }));
  t.ingest(P("progress", "a", { loaded: 60, total: 100 }));
  const s = t.snapshot();
  assert.equal(s.files.find((f) => f.id === "a").loaded, 60);
  assert.equal(s.files.find((f) => f.id === "b").loaded, 70);
  assert.equal(s.aggregate.loadedBytes, 130);
  assert.equal(s.aggregate.ratio, 0.65);
});

test("duplicate + out-of-order callbacks are idempotent / monotonic", () => {
  const t = createDownloadTracker();
  // progress BEFORE initiate (out of order) creates the file.
  t.ingest(P("progress", "x", { loaded: 500, total: 1000 }));
  // a stale/duplicate lower value must NOT move bytes backwards.
  let s = t.ingest(P("progress", "x", { loaded: 400, total: 1000 }));
  assert.equal(s.files[0].loaded, 500, "monotonic loaded");
  // exact duplicate is a no-op.
  s = t.ingest(P("progress", "x", { loaded: 500, total: 1000 }));
  assert.equal(s.files[0].loaded, 500);
  assert.equal(s.aggregate.fileCount, 1);
});

test("unknown sizes → honest indeterminate aggregate (no fake %)", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "big")); // no total
  const s = t.ingest(P("progress", "big", { loaded: 123 })); // still no total
  assert.equal(s.files[0].total, null);
  assert.equal(s.aggregate.indeterminate, true);
  assert.equal(s.aggregate.ratio, null, "must not fabricate a percentage without a denominator");
  assert.equal(s.aggregate.loadedBytes, 123);
});

test("zero-byte / metadata done event completes cleanly", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "empty", { total: 0 }));
  const s = t.ingest(P("done", "empty"));
  assert.ok(["cached", "complete"].includes(s.files[0].state));
});

test("per-file error is captured and surfaces in counts + phase", () => {
  const t = createDownloadTracker();
  t.ingest(P("initiate", "a", { total: 100 }));
  const s = t.ingest({ status: "error", file: "a", message: "network fail" });
  assert.equal(s.files[0].state, "failed");
  assert.match(s.files[0].error, /network fail/);
  assert.equal(s.aggregate.failed, 1);
});

test("knownFiles seed gives a complete denominator immediately (no discovery flicker)", () => {
  const t = createDownloadTracker({
    knownFiles: [{ id: "config.json", total: 50 }, { id: "onnx/model.onnx", total: 950 }],
  });
  let s = t.snapshot();
  assert.equal(s.aggregate.totalBytes, 1000);
  s = t.ingest(P("progress", "onnx/model.onnx", { loaded: 475, total: 950 }));
  // denominator already complete → ratio is stable/meaningful from the first byte.
  assert.equal(s.aggregate.indeterminate, false);
  assert.ok(Math.abs(s.aggregate.ratio - 0.475) < 1e-9);
});

test("formatBytes is readable + deterministic", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1023), "1023 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(2.9 * 1024 * 1024 * 1024), "2.9 GB");
  assert.equal(formatBytes(null), "—");
});

test("PHASE_LABEL covers every phase the reducer can emit", () => {
  for (
    const p of [
      "checking",
      "discovering",
      "downloading",
      "verifying",
      "initialising",
      "ready",
      "error",
    ]
  ) {
    assert.ok(PHASE_LABEL[p], `missing label for ${p}`);
  }
});

test("paused phase: a paused file (not downloading/verifying) yields phase 'paused'", () => {
  const t = createDownloadTracker();
  t.ingest({ status: "initiate", file: "onnx/a.onnx", total: 900 });
  t.ingest({ status: "progress", file: "onnx/a.onnx", loaded: 300, total: 900 });
  let s = t.ingest({ status: "file-paused", file: "onnx/a.onnx" });
  assert.equal(s.phase, "paused");
  // resuming (a fresh progress) leaves paused
  s = t.ingest({ status: "progress", file: "onnx/a.onnx", loaded: 400, total: 900 });
  assert.equal(s.phase, "downloading");
});
