// Deterministic, browser-free probes for the load/forward bounding in
// models/ai-image-detection/worker.js (web-ai-showcase-00qp).
//
// WHY THIS SHAPE: the worker imports "/web-ai-showcase/lib/webai.js" — an absolute deployment path
// that cannot resolve in Node — and its measured defects (a duplicate model load under racing
// messages, and no bound on concurrent forwards) are both worker-side state, so a client-side
// FakeWorker cannot see them. A Node resolver hook maps that ONE specifier to the stub fixture, and
// the shipped worker module is then imported unmodified with a cache-busting query per case, giving
// each case a fresh module (fresh `pipe`, empty queue). Node's test runner runs each file in its own
// process, and the hook only rewrites a specifier nothing else in the suite uses.
import test from "node:test";
import assert from "node:assert/strict";
import { registerHooks } from "node:module";

const STUB = new URL("./fixtures/ai-image-detection-worker-pipeline-stub.mjs", import.meta.url).href;
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "/web-ai-showcase/lib/webai.js") return { url: STUB, shortCircuit: true };
    return next(specifier, context);
  },
});

let caseId = 0;

/** Import a FRESH copy of the shipped worker against `stub` and return a message dispatcher. */
async function loadWorker(stub) {
  globalThis.__aiDetectStub = stub;
  const sent = [];
  const listeners = [];
  globalThis.self = {
    postMessage: (msg) => sent.push(msg),
    addEventListener: (type, fn) => { if (type === "message") listeners.push(fn); },
  };
  const url = new URL(`../models/ai-image-detection/worker.js?case=${++caseId}`, import.meta.url).href;
  await import(url);
  assert.equal(listeners.length, 1, "the worker must register exactly one message listener");
  return { sent, stub, dispatch: (data) => listeners[0]({ data }) };
}

/** A stub pipeline shaped like transformers.js: `model` is callable AND carries `.config`. */
function makeStub({ forwardMs = 0, failUrls = [], loadFailures = 0 } = {}) {
  const metrics = { loads: 0, forwards: 0, inFlight: 0, peakForwards: 0, decodes: 0, order: [] };
  return {
    metrics,
    decode: async (url) => {
      metrics.decodes++;
      return { url };
    },
    load: async (onProgress) => {
      metrics.loads++;
      if (metrics.loads <= loadFailures) throw new Error("stub load failure");
      onProgress?.({ status: "ready" });
      const model = async (inputs) => {
        const url = inputs.image.url;
        metrics.order.push(url);
        metrics.inFlight++;
        metrics.peakForwards = Math.max(metrics.peakForwards, metrics.inFlight);
        metrics.forwards++;
        await new Promise((resolve) => setTimeout(resolve, forwardMs));
        metrics.inFlight--;
        if (failUrls.some((needle) => url.includes(needle))) throw new Error(`stub forward failure for ${url}`);
        return { logits: { data: url.includes("ai") ? [3, -3] : [-3, 3] } };
      };
      model.config = { id2label: { 0: "artificial", 1: "human" } };
      return { device: "wasm", pipe: { processor: async (image) => ({ image }), model } };
    },
  };
}

test("requests racing before the model is loaded share exactly one load", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 1 }));
  // models/ai-image-detection/wild/index.html does Promise.all([classify, classify]) as its normal
  // path, so this is the reachable shape of the defect, not a stress case.
  await Promise.all([
    w.dispatch({ type: "run", id: 1, image: "data:real" }),
    w.dispatch({ type: "gate", id: 2, image: "data:ai", threshold: 0.5 }),
  ]);
  assert.equal(w.stub.metrics.loads, 1, "a racing pair must not start a second model load");
  assert.equal(w.sent.filter((m) => m.type === "ready").length, 1, "exactly one ready message");
  assert.deepEqual(w.sent.filter((m) => m.type === "result").map((m) => m.id), [1]);
  const gate = w.sent.find((m) => m.type === "gate");
  assert.equal(gate.id, 2);
  assert.equal(gate.flagged, true);
  assert.equal(gate.device, "wasm");
});

test("concurrent forwards are bounded to 2 and start in FIFO order", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 5 }));
  await Promise.all(
    [1, 2, 3, 4, 5].map((i) => w.dispatch({ type: "run", id: i, image: `data:real-${i}` })),
  );
  assert.equal(w.stub.metrics.peakForwards, 2, "at most two forwards may be in flight at once");
  assert.deepEqual(
    w.stub.metrics.order,
    ["data:real-1", "data:real-2", "data:real-3", "data:real-4", "data:real-5"],
    "queued work must start in request order",
  );
  assert.equal(w.sent.filter((m) => m.type === "result").length, 5, "every request gets its response");
  assert.equal(w.stub.metrics.loads, 1, "and only one load covers them all");
});

test("the bound spans message types (screen + run + gate share the queue)", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 5 }));
  await Promise.all([
    w.dispatch({ type: "screen", id: 1, images: [{ key: "a", url: "data:ai" }, { key: "b", url: "data:real" }], threshold: 0.5 }),
    w.dispatch({ type: "run", id: 2, image: "data:real" }),
    w.dispatch({ type: "gate", id: 3, image: "data:ai", threshold: 0.5 }),
  ]);
  assert.equal(w.stub.metrics.peakForwards, 2, "the cap is global, not per message type");
  assert.equal(w.sent.filter((m) => m.type === "screen").length, 1);
  assert.equal(w.sent.filter((m) => m.type === "result").length, 1);
  assert.equal(w.sent.filter((m) => m.type === "gate").length, 1);
});

test("a batch keeps input order, keys, flags and posts one response", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 1 }));
  await w.dispatch({
    type: "screen",
    id: 7,
    images: [{ key: "first", url: "data:ai" }, { key: "second", url: "data:real" }, { key: "third", url: "data:ai" }],
    threshold: 0.5,
  });
  const res = w.sent.at(-1);
  assert.equal(res.type, "screen");
  assert.equal(res.id, 7);
  assert.deepEqual(res.items.map((i) => i.key), ["first", "second", "third"], "order is preserved by index");
  assert.deepEqual(res.items.map((i) => i.flagged), [true, false, true], "flag = ai >= threshold");
  for (const item of res.items) {
    assert.ok(Math.abs(item.ai + item.real - 1) < 1e-9, "the two softmax probabilities still sum to 1");
  }
  assert.equal(res.device, "wasm");
  assert.equal(res.items.length, 3);
});

test("a failed forward fails that request closed and does not wedge the queue", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 1, failUrls: ["poison"] }));
  await Promise.all([
    w.dispatch({ type: "run", id: 1, image: "data:poison" }),
    w.dispatch({ type: "run", id: 2, image: "data:real" }),
    w.dispatch({ type: "run", id: 3, image: "data:real-two" }),
  ]);
  const errors = w.sent.filter((m) => m.type === "error");
  assert.equal(errors.length, 1, "one failing forward must not fail the others");
  assert.equal(errors[0].id, 1, "the error is attributed to the request that failed");
  assert.match(errors[0].message, /stub forward failure/);
  assert.deepEqual(
    w.sent.filter((m) => m.type === "result").map((m) => m.id).sort((a, b) => a - b),
    [2, 3],
    "the slot freed by the rejection must let the queued requests through",
  );
  assert.equal(w.stub.metrics.peakForwards, 2);
});

test("a failed load is not memoized, so the next request retries it", async () => {
  const w = await loadWorker(makeStub({ forwardMs: 0, loadFailures: 1 }));
  await w.dispatch({ type: "run", id: 1, image: "data:real" });
  assert.equal(w.sent.filter((m) => m.type === "error").length, 1, "the first request fails closed");
  assert.equal(w.sent.filter((m) => m.type === "ready").length, 0, "a failed load must not announce ready");
  await w.dispatch({ type: "run", id: 2, image: "data:real" });
  assert.equal(w.stub.metrics.loads, 2, "the failed attempt must be cleared so a retry can happen");
  assert.equal(w.sent.filter((m) => m.type === "result").length, 1, "and the retry serves the request");
  assert.equal(w.sent.filter((m) => m.type === "ready").length, 1);
});
