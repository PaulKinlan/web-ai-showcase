import assert from "node:assert/strict";
import test from "node:test";
import { formatMemoryBytes } from "../lib/model-memory-diagnostics.mjs";
import { elapsedText } from "../lib/model-run-status.mjs";

test("memory formatting uses binary MiB and preserves unavailable state", () => {
  assert.equal(formatMemoryBytes(null), "unavailable");
  assert.equal(formatMemoryBytes(25 * 1024 * 1024), "25.0 MiB");
  assert.equal(formatMemoryBytes(256 * 1024 * 1024), "256 MiB");
});

test("elapsed inference time remains readable beyond one minute", () => {
  assert.equal(elapsedText(9_900), "9s");
  assert.equal(elapsedText(65_000), "1m 05s");
});
