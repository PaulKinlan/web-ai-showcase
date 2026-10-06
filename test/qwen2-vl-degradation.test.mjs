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
import {
  clearRuntimeFailure,
  explainRuntimeFailure,
  showRuntimeFailure,
} from "../models/qwen2-vl/qwen2vl.js";

// The exact measured crash signature from the bead (bead report from earlier hardware, dims vary 285–294; not reproduced on the d6adb38 box — see scripts/validate-qwen2-vl-degradation.mjs header).
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

test("regex precedence: [WebGPU] prefix does not shadow device-lost or out-of-memory", () => {
  // [WebGPU] device lost -> device-lost
  const lost = explainRuntimeFailure(new Error("[WebGPU] device lost"));
  assert.equal(lost?.kind, "device-lost");
  assert.match(lost.headline, /took the GPU away/);

  // [WebGPU] out of memory -> out-of-memory
  const oom = explainRuntimeFailure(new Error("[WebGPU] out of memory"));
  assert.equal(oom?.kind, "out-of-memory");
  assert.match(oom.headline, /ran out of memory/);

  // Real [Concat] ... failed error still -> webgpu-kernel
  const concat = explainRuntimeFailure(
    new Error("[Concat] /model/layers.0/self_attn/Concat_7 failed"),
  );
  assert.equal(concat?.kind, "webgpu-kernel");
  assert.match(concat.headline, /WebGPU backend crashed/);
});

test("bare [WebGPU] prefix without kernel/fail/crash keyword stays unclassified", () => {
  const unclassifiedWebGPU = [
    "[WebGPU] device initialized",
    "[WebGPU] format rgba8unorm selected",
  ];
  for (const msg of unclassifiedWebGPU) {
    assert.equal(explainRuntimeFailure(new Error(msg)), null, `should not classify: ${msg}`);
  }
});

test("OOM_RE is restricted to memory contexts and does not match storage/quota errors", () => {
  const quotaErrors = [
    "Cache storage allocation failed",
    "QuotaExceededError: Storage allocation failed",
    "Disk quota exceeded",
    "Cache allocation failed for model weights",
  ];
  for (const msg of quotaErrors) {
    assert.equal(
      explainRuntimeFailure(new Error(msg)),
      null,
      `storage error must not be OOM: ${msg}`,
    );
  }

  const validOOM = [
    "Out of memory allocating tensor",
    "Array buffer allocation failed",
    "Memory allocation failed",
    "GPU memory allocation failed",
    "CUDA out of memory",
  ];
  for (const msg of validOOM) {
    assert.equal(
      explainRuntimeFailure(new Error(msg))?.kind,
      "out-of-memory",
      `should classify as OOM: ${msg}`,
    );
  }
});

test("showRuntimeFailure and clearRuntimeFailure manage .err-detail DOM lifecycle", () => {
  class MockElement {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.textContent = "";
      this._classes = new Set();
      const self = this;
      this.classList = {
        add(c) {
          self._classes.add(c);
        },
        remove(c) {
          self._classes.delete(c);
        },
        contains(c) {
          return self._classes.has(c);
        },
      };
      this.nextElementSibling = null;
      this.children = [];
    }
    get className() {
      return Array.from(this._classes).join(" ");
    }
    set className(val) {
      this._classes = new Set(String(val).trim().split(/\s+/).filter(Boolean));
    }
    append(...children) {
      this.children.push(...children);
    }
    remove() {
      if (this._prev) {
        this._prev.nextElementSibling = this.nextElementSibling;
        if (this.nextElementSibling) this.nextElementSibling._prev = this._prev;
        this._prev = null;
      }
    }
    insertAdjacentElement(position, el) {
      if (position === "afterend") {
        el.nextElementSibling = this.nextElementSibling;
        if (this.nextElementSibling) this.nextElementSibling._prev = el;
        el._prev = this;
        this.nextElementSibling = el;
      }
    }
  }

  const prevDoc = globalThis.document;
  try {
    globalThis.document = {
      createElement: (tag) => new MockElement(tag),
    };

    const statusEl = new MockElement("p");
    statusEl.textContent = "Generating…";

    // 1. Initial failure inserts .err-detail
    const rendered = showRuntimeFailure(statusEl, new Error(MEASURED_KERNEL_ERROR));
    assert.equal(rendered, true);
    assert.ok(statusEl.classList.contains("err"));
    assert.ok(!statusEl.classList.contains("ok"));
    assert.ok(statusEl.nextElementSibling?.classList.contains("err-detail"));

    // 2. Retry clears .err-detail before running
    clearRuntimeFailure(statusEl);
    assert.equal(statusEl.nextElementSibling, null, "retry must clear stale .err-detail");

    // 3. Second failure inserts fresh .err-detail without stacking
    showRuntimeFailure(statusEl, new Error(MEASURED_KERNEL_ERROR));
    assert.ok(statusEl.nextElementSibling?.classList.contains("err-detail"));
    assert.equal(
      statusEl.nextElementSibling.nextElementSibling,
      null,
      "must not stack .err-detail",
    );

    // 4. Successful completion clears .err-detail
    statusEl.textContent = "Done.";
    statusEl.classList.remove("err");
    statusEl.classList.add("ok");
    clearRuntimeFailure(statusEl);
    assert.equal(statusEl.nextElementSibling, null, "successful completion must clear .err-detail");
  } finally {
    globalThis.document = prevDoc;
  }
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

// Dual-signal precedence (web-ai-showcase-oow review M1): when one string carries BOTH a device-loss
// or OOM signal AND a kernel-failure signal, the more specific class must win. Reordering the
// classifier arms (KERNEL_RE first) must fail these.
test("dual-signal: device-lost wins over a kernel-failure phrase", () => {
  const r = explainRuntimeFailure(
    new Error('[WebGPU] Kernel "[Concat] x" failed because the device lost context'),
  );
  assert.equal(r.kind, "device-lost");
});
test("dual-signal: out-of-memory wins over a kernel-failure phrase", () => {
  const r = explainRuntimeFailure(
    new Error('[WebGPU] Kernel "[Concat] x" failed: out of memory'),
  );
  assert.equal(r.kind, "out-of-memory");
});
test("advice is phase-neutral (does not claim the model loaded)", () => {
  const r = explainRuntimeFailure(new Error("[Concat] /model/x failed"));
  assert.ok(!/model loaded/i.test(r.advice), r.advice);
});
