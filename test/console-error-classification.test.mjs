// web-ai-showcase-43l: Classification of Chrome's uncatchable view-transition AbortError.
// Pure classification logic and positive controls — browser-free unit tests (verified by
// test/suite-stays-browser-free.test.mjs).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  CLASSIFIED_WARN_THRESHOLD,
  classifiedConsoleNotice,
  consoleSummary,
  filterConsoleErrors,
  isTransitionSkipAbortError,
} from "../scripts/browser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

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

// --- masked-error self-reporting (bead web-ai-showcase-2zh) ------------------------------
// Classification is deliberate, but it must be visible: a run that shows only {errors, network}
// cannot distinguish "clean" from "clean because N errors were classified away".
const skip = (error = "AbortError: Transition was skipped") => ({ type: "transition-skip", error });

test("consoleSummary reports the masked count alongside errors and network", () => {
  const summary = consoleSummary({
    errors: ["TypeError: boom"],
    classifiedErrors: [skip(), skip("DOMException: Transition was skipped")],
    netFailures: ["net::ERR_FAILED"],
  });
  assert.deepEqual(summary, {
    errors: ["TypeError: boom"],
    classified: 2,
    network: ["net::ERR_FAILED"],
  });
});

test("consoleSummary is defensive on partial or missing pages", () => {
  assert.deepEqual(consoleSummary(undefined), { errors: [], classified: 0, network: [] });
  assert.deepEqual(consoleSummary({}), { errors: [], classified: 0, network: [] });
});

test("classifiedConsoleNotice stays silent when nothing was masked", () => {
  assert.equal(
    classifiedConsoleNotice({ errors: [], classifiedErrors: [], netFailures: [] }),
    null,
  );
  assert.equal(classifiedConsoleNotice(undefined), null);
});

test("classifiedConsoleNotice self-reports a few masked skips in the PASS-line shape", () => {
  const line = classifiedConsoleNotice({
    errors: [],
    classifiedErrors: [skip(), skip(), skip()],
    netFailures: [],
  });
  assert.ok(line, "a masked skip must produce a line, never silence");
  assert.ok(line.includes('"errors":[]'), line);
  assert.ok(line.includes('"classified":3'), line);
  assert.ok(line.includes('"network":[]'), line);
  assert.equal(line.startsWith("WARNING"), false, "a few skips are a note, not a storm");
});

test("classifiedConsoleNotice warns above the threshold and names the masked error", () => {
  const storm = [
    skip("AbortError: Transition was skipped"),
    ...Array.from({ length: CLASSIFIED_WARN_THRESHOLD }, () => skip()),
  ];
  const line = classifiedConsoleNotice({ errors: [], classifiedErrors: storm, netFailures: [] });
  assert.ok(line.startsWith("WARNING:"), line);
  assert.ok(line.includes(`"classified":${CLASSIFIED_WARN_THRESHOLD + 1}`), line);
  assert.ok(line.includes(`threshold ${CLASSIFIED_WARN_THRESHOLD}`), line);
  assert.ok(line.includes("first masked: AbortError: Transition was skipped"), line);

  const atThreshold = classifiedConsoleNotice({
    errors: [],
    classifiedErrors: storm.slice(0, CLASSIFIED_WARN_THRESHOLD),
    netFailures: [],
  });
  assert.equal(
    atThreshold.startsWith("WARNING"),
    false,
    "exactly at the threshold is still a note",
  );
});

test("CLASSIFIED_WARN_THRESHOLD is a documented positive integer", () => {
  assert.ok(Number.isInteger(CLASSIFIED_WARN_THRESHOLD), "threshold must be an integer");
  assert.ok(CLASSIFIED_WARN_THRESHOLD > 0, "threshold must be positive");
});

test("openPage is wired to emit the notice (source guard against a helper nobody calls)", () => {
  // The 2zh failure mode was a returned field nothing consumed; openPage needs a live CDP session
  // to exercise, so assert the wiring at the source instead of trusting the export alone.
  const src = readFileSync(join(ROOT, "scripts/browser.mjs"), "utf8");
  const start = src.indexOf("export async function openPage");
  const end = src.indexOf("export async function closePage");
  assert.ok(start >= 0 && end > start, "openPage must exist before closePage");
  const body = src.slice(start, end);
  assert.match(body, /classifiedConsoleNotice\(/);
  assert.match(body, /console\.warn\(/);
});
