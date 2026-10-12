// The 25 ms deadline of settleWithin (lib/model-cache.js) is a contract about the clock, so it is
// asserted on a virtual clock rather than against wall-clock elapsed time.
//
// WHY NOT IN test/model-cache-timeout.test.mjs: that file awaits real 25 ms timers at module scope,
// and enabling mock timers in the same module is unsafe — node:test starts the test while a top-level
// await is still pending, so module evaluation can resume *inside* the mock-timer window and have its
// `setTimeout` captured by the mock queue, which is then discarded. Measured in the xs9y branch: 1/10
// runs ended in "Detected unsettled top-level await" (a file-level red) even though the deadline
// itself worked. Keeping the virtual-clock assertions here makes both files timing-independent.
//
// See web-ai-showcase-9d4d for the same defect class: `performance.now() - start < 65 ms` around a
// 15 ms timer failed 5/10 runs under full-suite CPU contention on a 2-vCPU slot while the product was
// correct. A wall-clock bound measures machine load; a virtual clock measures the deadline.
import test from "node:test";
import assert from "node:assert/strict";
import { ModelCacheTimeoutError, settleWithin } from "../lib/model-cache.js";

// Microtasks always drain before the event-loop check phase runs a setImmediate, so this needs no
// timer of its own (and nothing here may schedule a real one).
const flushMicrotasks = async () => {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
};

test("settleWithin rejects a still-pending operation at its deadline, and not before", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let settled = false;
  const pending = settleWithin(new Promise(() => {}), 25, "Test operation").catch((error) => {
    settled = true;
    return error;
  });

  await flushMicrotasks();
  assert.equal(settled, false, "a pending operation must not settle before its deadline");

  t.mock.timers.tick(24);
  await flushMicrotasks();
  assert.equal(settled, false, "the deadline must not fire one millisecond early");

  t.mock.timers.tick(1);
  await flushMicrotasks();
  assert.ok(settled, "a still-pending operation must be rejected at exactly its 25 ms deadline");

  const error = await pending;
  assert.ok(error instanceof ModelCacheTimeoutError, "the deadline must reject with ModelCacheTimeoutError");
  assert.equal(error.operation, "Test operation");
  assert.equal(error.timeoutMs, 25);
});

test("settleWithin resolves an operation that finishes before the deadline, and a later deadline cannot undo it", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let release;
  const operation = new Promise((resolve) => { release = resolve; });
  const raced = settleWithin(operation, 25, "Test operation");
  await flushMicrotasks();
  t.mock.timers.tick(10);
  release("done");
  assert.equal(await raced, "done", "an operation that settles in time must win the race");
  // Tick the deadline the operation beat: a late deadline must not turn a settled result into a failure.
  t.mock.timers.tick(15);
  assert.equal(await raced, "done", "the deadline must not override an operation that already settled");
});
