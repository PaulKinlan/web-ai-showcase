// Deterministic, browser-free tests for the shared loader's load-failure classification.
//
// Bead web-ai-showcase-1ge. The ADR (2i3t) said "session creation failures and missing kernels"; coord
// NARROWED that to the one diagnostically specific ORT missing-kernel signature, because
// lib/model-loader.js backs every demo and "could not create a session" also covers out-of-memory and
// lost devices, which do recover. These tests pin BOTH directions: the narrow class loses its Retry,
// and everything else keeps it.
//
// No browser, no GPU, no model download, no DOM: the module's custom-element side imports need a few
// globals to exist before import, and the helpers under test take their DOM operations as arguments.
import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.HTMLElement ??= class {};
globalThis.customElements ??= { define() {}, get: () => undefined };
globalThis.document ??= { createElement: () => ({}) };

const { classifyLoadFailure, presentLoadFailure } = await import("../lib/model-loader.js");

// The message measured on this box for models/phi-3.5-vision (bead 1ge, verbatim shape).
const MEASURED_KERNEL_FAILURE =
  "Model initialisation failed. Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: " +
  "Failed to find kernel for MemcpyToHost(1) (node:'Memcpy' domain:'com.microsoft' ...)";

test("the measured ORT missing-kernel failure is fatal: state is honest and Retry is withheld", () => {
  const d = classifyLoadFailure(new Error(MEASURED_KERNEL_FAILURE));
  assert.equal(d.kind, "missing-ort-kernel");
  assert.equal(d.retry, false, "re-running an initialisation whose kernel is absent cannot help");
  assert.equal(d.kernel, "MemcpyToHost", "the kernel name is surfaced, not swallowed");
  assert.equal(d.state, "error");
  assert.match(d.copy, /could not initialize on this device\/runtime/);
  assert.match(d.copy, /MemcpyToHost/);
  assert.match(d.copy, /Retrying the same initialisation will not help/);
  // The bounded claim, pinned as a property: this must NOT become a platform-wide verdict, and it must
  // not reuse the "unsupported" wording that the loader reserves for a real capability check.
  assert.doesNotMatch(d.copy, /unsupported/i);
  assert.doesNotMatch(d.copy, /can't run on this device\/browser/i);
});

test("a plain string error is classified the same way as an Error instance", () => {
  assert.equal(classifyLoadFailure(MEASURED_KERNEL_FAILURE).retry, false);
  assert.equal(classifyLoadFailure({ message: MEASURED_KERNEL_FAILURE }).retry, false);
});

test("every other failure class keeps its Retry", () => {
  // Each entry is a failure this repo has actually seen, or a near miss that must not be captured by
  // the narrow matcher. The last two are the traps: the same words in a different sentence.
  const transient = {
    "generic session-creation failure (may be OOM or a lost device)":
      "Model initialisation failed. Can't create a session. ERROR_CODE: 9",
    "out of memory":
      "Model initialisation failed. Can't create a session. ERROR_CODE: 6, ERROR_MESSAGE: out of memory",
    "device lost": "Model initialisation failed. Can't create a session. GPU device lost",
    "fetch/network failure": "Failed to fetch dynamically imported module",
    "timeout": "Model validation record check timed out after 450 ms",
    "empty error": "",
    "null-ish error": null,
    "unrelated error": "Something else went wrong entirely",
    "different phrasing - no kernel found for this graph": "No kernel found for this node",
    "mentions a kernel but not the missing-kernel signature":
      "Failed to compile kernel for Conv, the shader was invalid",
  };
  for (const [label, err] of Object.entries(transient)) {
    const d = classifyLoadFailure(
      err instanceof Error ? err : err === null ? null : new Error(err),
    );
    assert.equal(d.retry, true, `${label} must keep its Retry`);
    assert.equal(d.kind, "transient", `${label} must not be classified fatal`);
    assert.equal(d.state, "error");
  }
});

test("presentLoadFailure routes the decision into the status line and the recovery affordance", () => {
  const calls = [];
  const io = {
    setState: (state, copy) => calls.push(["setState", state, copy]),
    retryButton: () => calls.push(["retryButton"]),
    onError: (err) => calls.push(["onError", err]),
  };

  // Fatal: the status line carries the truthful copy, no Retry is created, and the family's own
  // onError hook still fires (a validator or fallback depends on it).
  const fatal = presentLoadFailure(new Error(MEASURED_KERNEL_FAILURE), io);
  assert.equal(fatal.retry, false);
  assert.deepEqual(calls.map((c) => c[0]), ["setState", "onError"]);
  assert.match(calls[0][2], /could not initialize on this device\/runtime/);
  assert.equal(calls[0][1], "error");

  // Transient: identical shape, but the Retry is created — and AFTER onError, which is the order the
  // base catch used before this refactor. That ordering is pinned on purpose: the first version of
  // presentLoadFailure created the Retry first and this assertion blessed that drift (found in review).
  calls.length = 0;
  const transient = presentLoadFailure(
    new Error("Failed to fetch dynamically imported module"),
    io,
  );
  assert.equal(transient.retry, true);
  assert.deepEqual(calls.map((c) => c[0]), ["setState", "onError", "retryButton"]);
  assert.equal(calls[0][2], "Failed to fetch dynamically imported module");
});
