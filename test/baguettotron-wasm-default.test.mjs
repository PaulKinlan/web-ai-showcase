// web-ai-showcase-5o5: test that baguettotron and declared-WASM families default safely to WASM
// even when navigator.gpu is mock-present (avoiding broken GPU paths like SwiftShader).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { resolveTargetDevice } from "../lib/webai.js";

const DECLARED_WASM_WORKERS = [
  "models/baguettotron/worker.js",
  "models/qwen3-0-6b-wasm/worker.js",
  "models/bloomz-multilingual/worker.js",
  "models/granite-4-350m/worker.js",
  "models/smollm2-135m/worker.js",
];

function extractWorkerDeviceResolver(filePath) {
  const code = readFileSync(filePath, "utf8");
  const scrubbed = code.replace(/import\s+.*?;/g, "const TRANSFORMERS_URL = '';").replace(/export\s+/g, "");
  const context = {
    self: { addEventListener() {}, postMessage() {} },
    navigator: {
      gpu: {
        requestAdapter: async () => ({ features: new Set(), limits: {} }),
      },
    },
    console,
  };
  vm.createContext(context);
  vm.runInContext(scrubbed, context);
  if (typeof context.resolveTargetDevice !== "function") {
    throw new Error(`resolveTargetDevice not found in ${filePath}`);
  }
  return context.resolveTargetDevice;
}

test("resolveTargetDevice defaults to WASM even when mock GPU is present", () => {
  const hasGpu = true;

  // Default with empty options -> wasm
  assert.equal(resolveTargetDevice({}, hasGpu), "wasm");
  assert.equal(resolveTargetDevice(undefined, hasGpu), "wasm");

  // requiresWebGPU: false -> wasm
  assert.equal(resolveTargetDevice({ requiresWebGPU: false }, hasGpu), "wasm");

  // Explicit wasm -> wasm
  assert.equal(resolveTargetDevice({ device: "wasm" }, hasGpu), "wasm");
});

test("resolveTargetDevice honors explicit WebGPU request only when GPU is available", () => {
  // Explicit webgpu request with GPU -> webgpu
  assert.equal(resolveTargetDevice({ device: "webgpu" }, true), "webgpu");

  // Explicit webgpu request without GPU -> wasm
  assert.equal(resolveTargetDevice({ device: "webgpu" }, false), "wasm");

  // requiresWebGPU: true with GPU -> webgpu
  assert.equal(resolveTargetDevice({ requiresWebGPU: true }, true), "webgpu");

  // requiresWebGPU: true without GPU -> wasm fallback
  assert.equal(resolveTargetDevice({ requiresWebGPU: true }, false), "wasm");
});

test("models/baguettotron/worker.js defaults to WASM when navigator.gpu is mock-present", () => {
  const baguettotronResolver = extractWorkerDeviceResolver("models/baguettotron/worker.js");
  const hasGpu = true;

  // Default execution must be wasm
  assert.equal(baguettotronResolver({}, hasGpu), "wasm");
  assert.equal(baguettotronResolver({ requiresWebGPU: false }, hasGpu), "wasm");

  // Explicit webgpu override must be honored
  assert.equal(baguettotronResolver({ device: "webgpu" }, hasGpu), "webgpu");
  assert.equal(baguettotronResolver({ device: "webgpu" }, false), "wasm");
});

test("all declared-WASM worker scripts execute resolveTargetDevice defaulting to wasm", () => {
  for (const workerPath of DECLARED_WASM_WORKERS) {
    const resolver = extractWorkerDeviceResolver(workerPath);
    // Must default to wasm with GPU present
    assert.equal(resolver({}, true), "wasm", `${workerPath} must default to wasm`);
    assert.equal(resolver({ requiresWebGPU: false }, true), "wasm", `${workerPath} must stay wasm with requiresWebGPU: false`);
    // Must allow explicit webgpu override
    assert.equal(resolver({ device: "webgpu" }, true), "webgpu", `${workerPath} must allow explicit webgpu`);
  }
});

test("all declared-WASM worker files pass options to ensureLoaded and gate device choice", () => {
  for (const workerPath of DECLARED_WASM_WORKERS) {
    const content = readFileSync(workerPath, "utf8");
    assert.ok(
      content.includes("resolveTargetDevice"),
      `${workerPath} must use resolveTargetDevice`,
    );
    assert.ok(
      content.includes("ensureLoaded(e.data?.options || e.data)") ||
      content.includes("ensureLoaded(e.data.options || e.data)"),
      `${workerPath} must pass options to ensureLoaded on load message`,
    );
  }
});
