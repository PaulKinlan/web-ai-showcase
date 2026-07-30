import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const sourceUrl = new URL("../models/all-mpnet-base-v2/mpnet.js", import.meta.url);
const source = (await readFile(sourceUrl, "utf8")).replace(
  /import \{[^}]+\} from "\/web-ai-showcase\/lib\/worker-protocol\.js";/,
  "const WorkerClient = class {}; const SupersededError = class extends Error {};",
);
const { pca2d } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

const vectors = [
  [1, 0, 0],
  [0.9, 0.1, 0],
  [0, 1, 0],
  [0, 0.9, 0.1],
  [0, 0, 1],
  [0.1, 0, 0.9],
].map((vector) => {
  const norm = Math.hypot(...vector);
  return vector.map((value) => value / norm);
});

test("deterministic PCA does not collapse a centered Gram matrix onto its nullspace", () => {
  const first = pca2d(vectors);
  const second = pca2d(vectors);
  assert.deepEqual(first, second);
  assert.ok(first.explained > 0.8, `explained=${first.explained}`);
  assert.ok(first.residual < 1e-6, `residual=${first.residual}`);
  const spreadX = Math.max(...first.points.map((point) => point.x)) -
    Math.min(...first.points.map((point) => point.x));
  const spreadY = Math.max(...first.points.map((point) => point.y)) -
    Math.min(...first.points.map((point) => point.y));
  assert.ok(spreadX > 0.5, `spreadX=${spreadX}`);
  assert.ok(spreadY > 0.5, `spreadY=${spreadY}`);
});
