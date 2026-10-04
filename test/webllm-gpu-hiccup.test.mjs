// test/webllm-gpu-hiccup.test.mjs — tests for the WebLLM GPU buffer-mapping race
// policy, engine rebuild, and the "GPU hiccup — try again" affordance (web-ai-showcase-n81).
//
// Fast, browser-free tests (node:test). Verified by test/suite-stays-browser-free.test.mjs.

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  GPU_RACE,
  isGpuRace,
  renderGpuHiccupAffordance,
  retryPlan,
} from "../lib/webllm-race-policy.mjs";
import { rebuildWorkerEngine } from "../lib/webllm.js";
import { ModelRunStatus } from "../lib/model-run-status.mjs";

// --- DOM mock helper for browser-free testing ---------------------------------------------------
class MockClassList {
  constructor() {
    this._set = new Set();
  }
  add(...classes) {
    for (const c of classes) this._set.add(c);
  }
  remove(...classes) {
    for (const c of classes) this._set.delete(c);
  }
  has(c) {
    return this._set.has(c);
  }
  contains(c) {
    return this._set.has(c);
  }
}

class MockElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = new Map();
    this.classList = new MockClassList();
    this.listeners = new Map();
    this.style = {};
    this.textContent = "";
    this.disabled = false;
    this.type = "";
    this.id = "";
    this.hidden = false;
    this.dataset = {};
  }
  setAttribute(k, v) {
    this.attributes.set(k, String(v));
  }
  getAttribute(k) {
    return this.attributes.get(k) ?? null;
  }
  removeAttribute(k) {
    this.attributes.delete(k);
  }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  remove() {
    if (this.parentElement) {
      const idx = this.parentElement.children.indexOf(this);
      if (idx !== -1) this.parentElement.children.splice(idx, 1);
      this.parentElement = null;
    }
  }
  querySelector(sel) {
    if (sel.startsWith(".")) {
      const cls = sel.slice(1);
      for (const c of this.children) {
        if (c.className?.split(" ").includes(cls) || c.classList?.contains(cls)) return c;
        const sub = c.querySelector(sel);
        if (sub) return sub;
      }
    }
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      for (const c of this.children) {
        if (c.id === id) return c;
        const sub = c.querySelector(sel);
        if (sub) return sub;
      }
    }
    return null;
  }
  addEventListener(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
  }
  async click() {
    const handlers = this.listeners.get("click") || [];
    for (const h of handlers) {
      await h({
        preventDefault: () => {},
        stopPropagation: () => {},
      });
    }
  }
}

// Set up minimal global document if not present in Node
if (typeof globalThis.document === "undefined") {
  globalThis.document = {
    createElement: (tag) => new MockElement(tag),
  };
}

// --- 1. isGpuRace matcher -----------------------------------------------------------------------
test("isGpuRace correctly identifies WebGPU buffer mapping unmapped errors", () => {
  const exact = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );
  assert.equal(isGpuRace(exact), true, "exact match must be true");

  const partial = new Error("Error: Buffer was unmapped before mapping was resolved");
  assert.equal(isGpuRace(partial), true, "partial message match must be true");

  const stringErr =
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved";
  assert.equal(isGpuRace(stringErr), true, "raw string match must be true");

  const caseInsensitive = new Error(
    "failed to execute 'mapasync' on 'gpubuffer': buffer was unmapped before mapping was resolved",
  );
  assert.equal(isGpuRace(caseInsensitive), true, "case-insensitive must match");
});

test("isGpuRace rejects unrelated errors (no false positives / no swallowing)", () => {
  assert.equal(isGpuRace(new Error("Model initialization failed (code 9)")), false);
  assert.equal(isGpuRace(new Error("Out of memory")), false);
  assert.equal(isGpuRace(new Error("SyntaxError: Unexpected token")), false);
  assert.equal(isGpuRace(new Error("TypeError: Failed to fetch")), false);
  assert.equal(isGpuRace(new Error("NetworkError")), false);
  assert.equal(isGpuRace(null), false);
  assert.equal(isGpuRace(undefined), false);
  assert.equal(isGpuRace(""), false);
});

// --- 2. retryPlan decision function -------------------------------------------------------------
test("retryPlan permits retry only for GPU race with zero emitted tokens and bounded attempts", () => {
  const raceErr = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );

  // Attempt 1: permits retry
  const plan1 = retryPlan(raceErr, 1, { attempts: 3, emitted: false });
  assert.deepEqual(plan1, { retry: true, reason: "gpu-race", nextAttempt: 2 });

  // Attempt 2: permits retry
  const plan2 = retryPlan(raceErr, 2, { attempts: 3, emitted: false });
  assert.deepEqual(plan2, { retry: true, reason: "gpu-race", nextAttempt: 3 });

  // Attempt 3: exhausted
  const plan3 = retryPlan(raceErr, 3, { attempts: 3, emitted: false });
  assert.deepEqual(plan3, { retry: false, reason: "attempts-exhausted" });

  // Hard stop on partial output: never retry after tokens already streamed
  const planEmitted = retryPlan(raceErr, 1, { attempts: 3, emitted: true });
  assert.deepEqual(planEmitted, { retry: false, reason: "partial-output-already-streamed" });

  // Non-race error never retried
  const planOther = retryPlan(new Error("Other failure"), 1, { attempts: 3, emitted: false });
  assert.deepEqual(planOther, { retry: false, reason: "not-the-gpu-race" });
});

// --- 3. renderGpuHiccupAffordance DOM behavior --------------------------------------------------
test("renderGpuHiccupAffordance renders button on GPU race when zero deltas streamed", async () => {
  const statusEl = new MockElement("p");
  statusEl.id = "status";
  statusEl.textContent = "Failed.";

  const raceErr = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );

  let retried = false;
  const btn = renderGpuHiccupAffordance({
    error: raceErr,
    deltasStreamed: 0,
    statusEl,
    onRetry: async () => {
      retried = true;
    },
  });

  assert.ok(btn, "button must be created");
  assert.equal(btn.textContent, "GPU hiccup — try again");
  assert.equal(btn.className, "button secondary gpu-hiccup-btn");
  assert.equal(statusEl.children.includes(btn), true, "button must be attached to statusEl");

  // User clicks the button
  await btn.click();
  assert.equal(retried, true, "onRetry must be called on click");
  assert.equal(statusEl.children.includes(btn), false, "button must be removed after successful retry");
});

test("renderGpuHiccupAffordance refuses when tokens were streamed or error is not GPU race", () => {
  const statusEl = new MockElement("p");
  const raceErr = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );

  // Refusal: deltas were streamed (deltasStreamed > 0)
  const btn1 = renderGpuHiccupAffordance({
    error: raceErr,
    deltasStreamed: 5,
    statusEl,
    onRetry: async () => {},
  });
  assert.equal(btn1, null, "must not offer retry after partial text streamed");
  assert.equal(statusEl.children.length, 0);

  // Refusal: unrelated error
  const btn2 = renderGpuHiccupAffordance({
    error: new Error("Model file missing 404"),
    deltasStreamed: 0,
    statusEl,
    onRetry: async () => {},
  });
  assert.equal(btn2, null, "must not offer retry for non-GPU-race errors");
  assert.equal(statusEl.children.length, 0);
});

test("renderGpuHiccupAffordance presents honest final failure if retry fails again", async () => {
  const statusEl = new MockElement("p");

  const raceErr = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );

  let finalFailCalledWith = null;
  const btn = renderGpuHiccupAffordance({
    error: raceErr,
    deltasStreamed: 0,
    statusEl,
    onRetry: async () => {
      throw new Error("Persistent GPU deadlock");
    },
    onFinalFail: (err) => {
      finalFailCalledWith = err;
    },
  });

  assert.ok(btn);
  await btn.click();

  assert.ok(finalFailCalledWith, "onFinalFail must be invoked");
  assert.match(finalFailCalledWith.message, /Persistent GPU deadlock/);
  assert.match(statusEl.textContent, /GPU hiccup recovery failed/);
  assert.equal(statusEl.children.includes(btn), false, "retry button must be removed after final failure");
});

// --- 5. ModelRunStatus component integration ---------------------------------------------------
test("ModelRunStatus.fail renders 'GPU hiccup — try again' button on GPU race", async () => {
  // Test ModelRunStatus in a simulated environment
  let shadowInnerHTML = "";
  const shadowElements = new Map();
  const shadowChildren = [];

  const fakePhase = new MockElement("strong");
  const fakeProgress = new MockElement("progress");
  const fakeDetail = new MockElement("p");
  const fakeLive = new MockElement("p");
  const fakeBox = new MockElement("div");
  fakeBox.className = "box";

  shadowElements.set("#phase", fakePhase);
  shadowElements.set("#progress", fakeProgress);
  shadowElements.set("#detail", fakeDetail);
  shadowElements.set("#live", fakeLive);
  shadowElements.set(".box", fakeBox);

  const shadowRoot = {
    querySelector: (sel) => shadowElements.get(sel) ?? fakeBox.querySelector(sel),
    querySelectorAll: () => [],
    appendChild: (k) => fakeBox.appendChild(k),
  };

  const statusComp = {
    stopTimer: () => {},
    dataset: {},
    shadowRoot,
    $: (sel) => shadowRoot.querySelector(sel),
    fail: ModelRunStatus.prototype.fail,
  };

  const raceErr = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved",
  );

  let rebuildTriggered = false;
  statusComp.fail("Buffer was unmapped", {
    error: raceErr,
    deltasStreamed: 0,
    onRetry: async () => {
      rebuildTriggered = true;
    },
  });

  assert.equal(statusComp.dataset.state, "error");
  assert.equal(fakePhase.textContent, "GPU hiccup");

  const retryBtn = fakeBox.querySelector("#gpu-retry");
  assert.ok(retryBtn, "retry button must be rendered in component box");
  assert.equal(retryBtn.textContent, "GPU hiccup — try again");

  // Click retry button
  await retryBtn.onclick({ preventDefault: () => {} });
  assert.equal(rebuildTriggered, true, "onRetry must be called");
  assert.equal(fakeBox.children.includes(retryBtn), false, "retry button removed after recovery");
});

test("rebuildWorkerEngine terminates old worker and triggers fresh engine load", async () => {
  let terminated = false;
  let loaded = false;

  class FakeWorker {
    terminate() {
      terminated = true;
    }
    addEventListener() {}
  }

  // Mock global Worker
  const originalWorker = globalThis.Worker;
  globalThis.Worker = FakeWorker;

  const fakeEngine = {
    worker: new FakeWorker(),
    ready: true,
    _loadWaiters: [{}],
    _active: {},
    load: async () => {
      loaded = true;
    },
    _onMessage: () => {},
  };

  try {
    await rebuildWorkerEngine(fakeEngine, "/fake/worker.js");
    assert.equal(terminated, true, "old worker must be terminated");
    assert.equal(loaded, true, "engine.load must be invoked on fresh worker");
    assert.equal(fakeEngine.ready, false, "ready flag reset during rebuild");
  } finally {
    globalThis.Worker = originalWorker;
  }
});
