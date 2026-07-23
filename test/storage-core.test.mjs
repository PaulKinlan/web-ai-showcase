import assert from "node:assert/strict";
import test from "node:test";
import {
  buildStorageInventory,
  formatBytes,
  inferHuggingFaceModelId,
  matchesModelId,
} from "../storage/storage-core.mjs";

test("formats sizes without pretending unknown values are zero", () => {
  assert.equal(formatBytes(null), "Size unavailable");
  assert.equal(formatBytes(250_000_000), "250.0 MB");
  assert.equal(formatBytes(1_900_000_000), "1.90 GB");
});

test("matches plain and encoded model identifiers", () => {
  assert.equal(
    matchesModelId("https://huggingface.co/Xenova/clip/resolve/main/a.onnx", "Xenova/clip"),
    true,
  );
  assert.equal(matchesModelId("https://cache.test/Xenova%2Fclip/a.onnx", "Xenova/clip"), true);
});

test("infers Hugging Face repositories but ignores library assets", () => {
  assert.equal(
    inferHuggingFaceModelId(
      "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/config.json",
    ),
    "Xenova/clip-vit-base-patch32",
  );
  assert.equal(
    inferHuggingFaceModelId("https://cdn.jsdelivr.net/npm/onnxruntime-web/ort.wasm"),
    null,
  );
});

test("groups validated, unverified, and resumable storage without double counting", () => {
  const known = "https://huggingface.co/Xenova/clip/resolve/main/model.onnx";
  const unknown = "https://huggingface.co/acme/other/resolve/main/model.onnx";
  const inventory = buildStorageInventory({
    records: [{ key: "k", modelId: "Xenova/clip", files: [known], runtime: "transformers.js" }],
    cacheEntries: [
      { cacheName: "transformers-cache", url: known, bytes: 100 },
      { cacheName: "transformers-cache", url: unknown, bytes: 40 },
      { cacheName: "webai-shell", url: "https://site.test/styles.css", bytes: 10 },
    ],
    partials: [{
      url: "https://huggingface.co/acme/large/resolve/main/model.onnx",
      receivedBytes: 25,
    }],
    models: [{ hfId: "Xenova/clip", title: "CLIP", slug: "clip-demo", sizeMB: 88 }],
  });
  assert.equal(inventory.downloaded.length, 2);
  assert.equal(inventory.downloaded[0].title, "CLIP");
  assert.equal(inventory.downloaded[0].verified, true);
  assert.equal(inventory.downloaded[1].verified, false);
  assert.equal(inventory.partials.length, 1);
  assert.equal(inventory.totalKnownBytes, 165);
  assert.deepEqual(inventory.claimedUrls.sort(), [known, unknown].sort());
});
