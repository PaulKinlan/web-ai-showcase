// web-ai-showcase-43l: Classification of Chrome's uncatchable view-transition AbortError.
// Pure classification logic and positive controls — browser-free unit tests (verified by
// test/suite-stays-browser-free.test.mjs).
import test from "node:test";
import assert from "node:assert/strict";
import {
  filterConsoleErrors,
  isTransitionSkipAbortError,
} from "../scripts/browser.mjs";

test("isTransitionSkipAbortError matches exact Chrome transition skip variants", () => {
  const exactVariants = [
    "Uncaught (in promise) AbortError: Transition was skipped",
    "AbortError: Transition was skipped",
    "DOMException: Transition was skipped",
    "Uncaught (in promise) DOMException: Transition was skipped",
    { message: "Transition was skipped", name: "AbortError" },
    { description: "AbortError: Transition was skipped" },
    { message: "Transition was skipped", description: "DOMException: Transition was skipped" },
  ];
  for (const v of exactVariants) {
    assert.equal(
      isTransitionSkipAbortError(v),
      true,
      `expected exact transition skip to match: ${JSON.stringify(v)}`,
    );
  }
});

test("POSITIVE CONTROL: isTransitionSkipAbortError rejects real page exceptions and arbitrary aborts", () => {
  const realErrors = [
    "TypeError: Cannot read properties of undefined (reading 'foo')",
    "ReferenceError: myVar is not defined",
    "SyntaxError: Unexpected token <",
    "Error: Model file 404 not found",
    "Error: Worker initialization failed",
    // Legitimate fetch / user aborts must NOT be masked:
    "AbortError: The user aborted a request.",
    "AbortError: Fetch was aborted",
    "DOMException: The operation was aborted.",
    "Error: AbortError: Operation cancelled by user",
    // Incomplete or unrelated matches:
    "Transition was skipped", // Missing AbortError/DOMException type
    "AbortError: Network timeout",
    "DOMException: QuotaExceededError",
    "",
    null,
    undefined,
  ];
  for (const err of realErrors) {
    assert.equal(
      isTransitionSkipAbortError(err),
      false,
      `real error must NOT be masked: ${JSON.stringify(err)}`,
    );
  }
});

test("filterConsoleErrors filters only transition skips and retains all real errors", () => {
  const input = [
    "Uncaught (in promise) AbortError: Transition was skipped",
    "TypeError: Cannot read properties of undefined",
    "DOMException: Transition was skipped",
    "Error: Failed to fetch model weights",
  ];
  const filtered = filterConsoleErrors(input);
  assert.deepEqual(filtered, [
    "TypeError: Cannot read properties of undefined",
    "Error: Failed to fetch model weights",
  ]);
});

test("POSITIVE CONTROL: console-clean assertion fails when a real error accompanies a transition skip", () => {
  const input = [
    "Uncaught (in promise) AbortError: Transition was skipped",
    "ReferenceError: unhandledVar is not defined",
  ];
  const filtered = filterConsoleErrors(input);
  assert.equal(filtered.length === 0, false, "must fail console-clean check when real error is present");
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0], "ReferenceError: unhandledVar is not defined");
});

test("filterConsoleErrors handles empty, non-array, and clean inputs safely", () => {
  assert.deepEqual(filterConsoleErrors([]), []);
  assert.deepEqual(filterConsoleErrors(null), []);
  assert.deepEqual(filterConsoleErrors(undefined), []);
  assert.deepEqual(
    filterConsoleErrors(["Uncaught (in promise) AbortError: Transition was skipped"]),
    [],
  );
});
