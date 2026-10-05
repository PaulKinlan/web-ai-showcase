// web-ai-showcase-oow: the qwen2-vl route is WebGPU-only and the ORT WebGPU backend bundled with
// the pinned transformers.js (3.7.5) crashes inside the model's own forward pass on some
// GPU/driver/browser builds. The shipped degradation contract: a visitor must NEVER see a raw
// kernel string as the user-facing message — explainRuntimeFailure classifies the measured failure
// classes into a labelled { headline, advice, raw } and returns null for anything it must not
// touch (the loader's own labelled needs-WebGPU state, and unclassified errors the pages keep
// surfacing verbatim). Browser-free unit tests; the page-level rendering is driven for real by
// scripts/validate-qwen2-vl-degradation.mjs.
import test from "node:test";
import assert from "node:assert/strict";
import { explainRuntimeFailure } from "../models/qwen2-vl/qwen2vl.js";

// The exact measured crash signature from the bead (10-cell validator run, dims vary 285–294).
const MEASURED_KERNEL_ERROR =
  '[WebGPU] Kernel "[Concat] /model/layers.0/self_attn/Concat_7" failed. Error: Failed to ' +
  "generate kernel's output[0] with dims [1,2,286,128]. If you are running with pre-allocated " +
  "output, please make sure the output type/dims are correct, the kernel is implemented for the " +
  "given inputs and its parameters are correct. You may try to enable the 'webgpu-graph-capture' " +
  "flag to get more details.";

test("the measured ORT WebGPU Concat kernel crash is classified as webgpu-kernel", () => {
  const info = explainRuntimeFailure(new Error(MEASURED_KERNEL_ERROR));
  assert.equal(info.kind, "webgpu-kernel");
  assert.equal(info.raw, MEASURED_KERNEL_ERROR);
  // The labelled message says what happened and what the visitor can do…
  assert.match(info.headline, /WebGPU backend crashed/);
  assert.match(info.advice, /not something you did/);
  assert.match(info.advice, /update your browser/i);
  assert.match(info.advice, /SmolVLM|Moondream|FastVLM/);
  // …and NEVER leaks the raw kernel text into the user-facing message.
  const userFacing = `${info.headline} ${info.advice}`;
  assert.ok(!userFacing.includes("/model/layers"), "no raw graph path in user-facing text");
  assert.ok(!userFacing.includes("[Concat]"), "no raw kernel name in user-facing text");
  assert.ok(!/dims \[/.test(userFacing), "no raw dims in user-facing text");
});

test("kernel-class variants (any layer/dims, graph-capture suffix) all classify", () => {
  const variants = [
    '[WebGPU] Kernel "[Concat] /model/layers.27/self_attn/Concat_7" failed. Error: Failed to generate kernel\'s output[0] with dims [1,2,294,128].',
    '[WebGPU] Kernel "[MatMul] /model/layers.0/mlp/MatMul" failed.',
    "An error occurred running the WebGPU compute pass. [WebGPU] something failed",
  ];
  for (const v of variants) {
    assert.equal(explainRuntimeFailure(v)?.kind, "webgpu-kernel", `should classify: ${v}`);
  }
});

test("GPU device-lost and out-of-memory get their own labelled degradations", () => {
  const lost = explainRuntimeFailure(new Error("GPU device lost"));
  assert.equal(lost.kind, "device-lost");
  assert.match(lost.advice, /Reload the page/);
  const oom = explainRuntimeFailure(new Error("Out of memory allocating tensor"));
  assert.equal(oom.kind, "out-of-memory");
  assert.match(oom.advice, /smaller on-device vision-language/);
  const wasmOOM = explainRuntimeFailure(
    new Error("Array buffer allocation failed"),
  );
  assert.equal(wasmOOM.kind, "out-of-memory");
});

test("the loader's labelled needs-WebGPU state is left alone (null, not reclassified)", () => {
  assert.equal(explainRuntimeFailure(new Error("needs-webgpu")), null);
});

test("POSITIVE CONTROL: ordinary errors stay unclassified so pages keep surfacing them verbatim", () => {
  const keepRaw = [
    "TypeError: Cannot read properties of undefined (reading 'dims')",
    "Error: Failed to fetch",
    "NetworkError: 404 on onnx/decoder_model_merged_q4f16.onnx_data",
    "empty transcript",
    "",
  ];
  for (const msg of keepRaw) {
    assert.equal(explainRuntimeFailure(new Error(msg)), null, `must not reclassify: ${msg}`);
  }
  // …and a TypeError that merely MENTIONS memory in passing must not be swallowed as OOM.
  assert.equal(explainRuntimeFailure(new Error("reading 'memory' of undefined")), null);
});
