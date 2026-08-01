import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { composeMask, summarizeConfidence } from "../models/interactive-segmenter/mask-math.js";

const root = new URL("../", import.meta.url);
const read = (path) => readFileSync(new URL(path, root), "utf8");

function catalogue() {
  return JSON.parse(read("models.json")).models;
}

test("catalogue integrates exactly one verified pending family without changing denominator", () => {
  const models = catalogue();
  assert.equal(models.length, 2679);
  assert.deepEqual(
    Object.fromEntries(
      ["built", "pending", "blocked"].map((status) => [
        status,
        models.filter((model) => model.status === status).length,
      ]),
    ),
    { built: 314, pending: 2302, blocked: 63 },
  );
  const model = models.find((entry) => entry.slug === "interactive-segmenter");
  assert.equal(model.status, "built");
  assert.equal(model.hfId, "mediapipe/interactive-segmenter");
  assert.equal(model.task, "mask-generation");
  assert.equal(model.license, "apache-2.0");
});

test("five stable portfolio routes call the genuine shared demo", () => {
  for (
    const route of [
      "index.html",
      "basics/index.html",
      "practical/index.html",
      "wild/index.html",
      "multi-model/index.html",
    ]
  ) {
    const html = read(`models/interactive-segmenter/${route}`);
    assert.match(html, /mountInteractiveDemo/);
    assert.match(html, /public\/styles\.css/);
    assert.match(html, /<main>/);
  }
  assert.match(
    read("models/interactive-segmenter/multi-model/index.html"),
    /MagicTouch[\s\S]*MobileViT/,
  );
});

test("artifact pin and worker integrity constant match", () => {
  const artifact = JSON.parse(read("models/interactive-segmenter/artifact.json"));
  const worker = read("models/interactive-segmenter/worker.js");
  assert.equal(artifact.artifact.bytes, 6_227_884);
  assert.equal(artifact.artifact.sha256.length, 64);
  assert.match(worker, new RegExp(artifact.artifact.sha256));
  assert.match(worker, /crypto\.subtle\.digest\("SHA-256"/);
  assert.match(worker, /modelAssetBuffer/);
  assert.match(worker, /InteractiveSegmenter\.createFromOptions/);
});

test("inference and dense compositing remain worker-first with deterministic disposal", () => {
  const worker = read("models/interactive-segmenter/worker.js");
  const client = read("models/interactive-segmenter/demo.js");
  assert.match(worker, /segmenter\.segment\(bitmap/);
  assert.match(worker, /composeMask/);
  assert.match(worker, /transfer: \[rgba\.buffer\]/);
  assert.match(worker, /segmenter\?\.close/);
  assert.match(client, /WorkerClient/);
  assert.match(client, /pagehide/);
  assert.match(client, /URL\.revokeObjectURL/);
});

test("confidence summaries discriminate target, area and bounds", () => {
  const confidence = new Float32Array([
    0.05,
    0.1,
    0.2,
    0.1,
    0.9,
    0.8,
    0.05,
    0.7,
    0.65,
  ]);
  const summary = summarizeConfidence(confidence, 0.5, { x: 0.5, y: 0.5 }, 3, 3);
  assert.equal(summary.selectedPixels, 4);
  assert.equal(summary.coverage, 4 / 9);
  assert.ok(summary.targetConfidence > 0.89);
  assert.deepEqual(summary.bbox, { minX: 1, minY: 1, maxX: 2, maxY: 2 });
});

test("compositor makes distinct overlay, mask, cutout, and spotlight outputs", () => {
  const source = new Uint8ClampedArray([
    200,
    100,
    50,
    255,
    20,
    40,
    80,
    255,
  ]);
  const confidence = new Float32Array([0.9, 0.1]);
  const outputs = ["overlay", "mask", "cutout", "spotlight"].map((mode) =>
    Array.from(composeMask(source, confidence, { mode, threshold: 0.5, opacity: 0.6 }))
  );
  assert.equal(new Set(outputs.map(JSON.stringify)).size, 4);
  assert.ok(outputs[2][3] > 0);
  assert.equal(outputs[2][7], 0);
  assert.equal(outputs[3][4], outputs[3][5]);
  assert.equal(outputs[3][5], outputs[3][6]);
});
