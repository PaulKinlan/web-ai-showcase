// test/vlm-placeholder-expansion.test.mjs — bead web-ai-showcase-xlt.
//
// Fast, browser-free unit test auditing all vision-language model workers for the
// placeholder-vs-patch-features defect:
// 1. Static audit: every VLM worker separating text tokenization from vision processing
//    must implement the patch-grid placeholder expansion pattern (nanollava-vlm, moondream2-vlm).
// 2. Behavioral verification: the expansion algorithm converts a single <image> placeholder token
//    into floor(image_size / patch_size)^2 copies with matching attention_mask and BigInt-safe numeric comparison.

import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;

// Pure reference implementation of the placeholder expansion pattern from nanollava-vlm and moondream2-vlm
function expandImagePlaceholders(ids, mask, imageTokenId, imageTokenCount) {
  const expandedIds = [];
  const expandedMask = [];
  for (let i = 0; i < ids.length; i++) {
    const idNum = typeof ids[i] === "bigint" ? Number(ids[i]) : ids[i];
    const maskNum = typeof mask[i] === "bigint" ? Number(mask[i]) : mask[i];
    if (idNum === imageTokenId) {
      for (let k = 0; k < imageTokenCount; k++) {
        expandedIds.push(imageTokenId);
        expandedMask.push(1);
      }
    } else {
      expandedIds.push(idNum);
      expandedMask.push(maskNum);
    }
  }
  return {
    expandedIds,
    expandedMask,
  };
}

test("VLM placeholder expansion: single <image> token expands to exact patch grid count (729)", () => {
  // Nanollava and Moondream2 use SigLIP/vision encoders with 14x14 patches:
  // e.g. floor(384 / 14)^2 = 27^2 = 729 patch tokens
  const imageSize = 384;
  const patchSize = 14;
  const imageTokenCount = Math.floor(imageSize / patchSize) ** 2;
  assert.equal(imageTokenCount, 729);

  const imageTokenId = 32000;
  // Simulated prompt: <s> <image> \n What is this? </s>
  const originalIds = [1, 32000, 198, 2044, 318, 428, 30, 2];
  const originalMask = [1, 1, 1, 1, 1, 1, 1, 1];

  const { expandedIds, expandedMask } = expandImagePlaceholders(
    originalIds,
    originalMask,
    imageTokenId,
    imageTokenCount,
  );

  // 8 original tokens - 1 placeholder + 729 expanded = 736 total tokens
  assert.equal(expandedIds.length, originalIds.length - 1 + 729);
  assert.equal(expandedMask.length, expandedIds.length);

  // Assert leading and trailing tokens are preserved
  assert.equal(expandedIds[0], 1);
  assert.equal(expandedIds[729 + 1], 198); // the \n token after 729 image tokens
  assert.equal(expandedIds[expandedIds.length - 1], 2);

  // Assert all expanded tokens have the imageTokenId and mask = 1
  for (let k = 1; k <= 729; k++) {
    assert.equal(expandedIds[k], imageTokenId);
    assert.equal(expandedMask[k], 1);
  }
});

test("VLM placeholder expansion: handles BigInt input_ids from tolist() safely", () => {
  const imageTokenCount = 729;
  const imageTokenId = 32000;

  // tolist() on int64 tensors produces BigInt values in JavaScript
  const bigintIds = [BigInt(1), BigInt(32000), BigInt(198)];
  const bigintMask = [BigInt(1), BigInt(1), BigInt(1)];

  const { expandedIds, expandedMask } = expandImagePlaceholders(
    bigintIds,
    bigintMask,
    imageTokenId,
    imageTokenCount,
  );

  assert.equal(expandedIds.length, 1 + 729 + 1);
  assert.equal(typeof expandedIds[0], "number");
  assert.equal(expandedIds[1], imageTokenId);
});

test("VLM placeholder expansion: sequence without <image> remains unchanged", () => {
  const ids = [1, 2, 3, 4, 5];
  const mask = [1, 1, 1, 1, 1];
  const { expandedIds, expandedMask } = expandImagePlaceholders(ids, mask, 32000, 729);

  assert.deepEqual(expandedIds, ids);
  assert.deepEqual(expandedMask, mask);
});

test("audit models/*/worker.js: separate text-vision VLM workers implement placeholder expansion", () => {
  const modelsDir = join(ROOT, "models");
  const entries = readdirSync(modelsDir, { withFileTypes: true });

  const separateTokenizationWorkers = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const workerPath = join(modelsDir, entry.name, "worker.js");
    let content;
    try {
      content = readFileSync(workerPath, "utf8");
    } catch {
      continue;
    }

    // Identify VLM workers using separate text tokenization and vision processing
    const hasImageToken = content.includes("<image>");
    const hasSeparateProcessor = content.includes("processor(image)");
    const hasLlava = content.includes("LlavaForConditionalGeneration") || content.includes("Moondream1ForConditionalGeneration");

    if (hasImageToken && (hasSeparateProcessor || hasLlava)) {
      separateTokenizationWorkers.push({
        slug: entry.name,
        content,
      });
    }
  }

  // Currently, nanollava-vlm and moondream2-vlm are the built models with separate text/vision tokenization
  assert.ok(separateTokenizationWorkers.length >= 2, "must audit nanollava-vlm and moondream2-vlm");

  for (const { slug, content } of separateTokenizationWorkers) {
    // Assert patch_size and image_size calculation exists
    assert.match(
      content,
      /patch_size/,
      `${slug}/worker.js must reference patch_size to calculate image feature grid`,
    );
    assert.match(
      content,
      /imageTokenCount\s*=\s*Math\.floor/,
      `${slug}/worker.js must calculate imageTokenCount via Math.floor(image_size / patch_size) ** 2`,
    );

    // Assert expansion loop exists
    assert.match(
      content,
      /expandedIds/,
      `${slug}/worker.js must construct expandedIds tensor array`,
    );
    assert.match(
      content,
      /expandedMask/,
      `${slug}/worker.js must construct expandedMask tensor array`,
    );

    // Assert int64 tensor construction
    assert.match(
      content,
      /new\s+Tensor\(\s*["']int64["']/,
      `${slug}/worker.js must build int64 Tensor for expanded input_ids`,
    );

    // Assert BigInt-safe numeric compare
    assert.match(
      content,
      /Number\(/,
      `${slug}/worker.js must use Number(...) compare for BigInt-safe tolist() results`,
    );
  }
});
