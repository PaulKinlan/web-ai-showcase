// The audit treats scripts/model-task-vocabulary.mjs as a whitelist. These tests keep it honest: every
// entry must carry a reason, and anything not entered must stay reportable so the audit keeps finding
// genuine drift (web-ai-showcase-4hv).
import test from "node:test";
import assert from "node:assert/strict";
import {
  KNOWN_EQUIVALENT_TASK_PAIRS,
  classifyTaskPair,
} from "../scripts/model-task-vocabulary.mjs";

test("every recorded pair is well formed and carries a real rationale", () => {
  const entries = Object.entries(KNOWN_EQUIVALENT_TASK_PAIRS);
  assert.ok(entries.length >= 14, "expected the recorded pairs from the fb9 sweep");
  for (const [key, rationale] of entries) {
    assert.match(key, /^[a-z0-9-]+ -> [a-z0-9-]+$/, `${key} must be "recorded -> upstream"`);
    const [recorded, upstream] = key.split(" -> ");
    assert.notEqual(recorded, upstream, `${key} is not a vocabulary difference at all`);
    assert.ok(
      String(rationale).length >= 20,
      `${key} needs a one-line reason, got ${JSON.stringify(rationale)}`,
    );
  }
});

test("a recorded pair classifies as equivalent and explains itself", () => {
  const pair = classifyTaskPair("text-to-speech", "text-to-audio");
  assert.equal(pair.equivalent, true);
  assert.match(pair.rationale, /same behaviour/);
  assert.equal(pair.key, "text-to-speech -> text-to-audio");
});

test("the map is a whitelist: an unrecorded mismatch is not equivalent", () => {
  // This is the property the audit's usefulness depends on. If it ever returns true here, the audit has
  // become a rubber stamp.
  for (const [recorded, upstream] of [
    ["text-to-speech", "image-classification"],
    ["fill-mask", "text-to-speech"],
    ["object-detection", "text-ranking"],
    ["feature-extraction", "text-generation"],
  ]) {
    assert.equal(classifyTaskPair(recorded, upstream).equivalent, false, `${recorded} -> ${upstream}`);
  }
});

test("the pairs the catalogue actually contains are all recorded", () => {
  // The 34 taskDrift findings from the fb9 sweep were these 14 pair shapes. If one goes missing the
  // audit starts reporting it again, which is the regression this locks out.
  const observed = [
    ["text-to-speech", "text-to-audio"],
    ["text-classification", "text-ranking"],
    ["feature-extraction", "sentence-similarity"],
    ["sentence-similarity", "feature-extraction"],
    ["audio-feature-extraction", "feature-extraction"],
    ["zero-shot-object-detection", "object-detection"],
    ["image-to-image", "text-to-image"],
    ["image-to-image", "image-to-text"],
    ["zero-shot-audio-classification", "feature-extraction"],
    ["text-classification", "zero-shot-classification"],
    ["text2text-generation", "text-generation"],
    ["zero-shot-image-classification", "feature-extraction"],
    ["image-text-to-text", "text-generation"],
    ["fill-mask", "text-generation"],
  ];
  assert.equal(observed.length, 14, "the observed pair set changed — update the map deliberately");
  for (const [recorded, upstream] of observed) {
    assert.equal(
      classifyTaskPair(recorded, upstream).equivalent,
      true,
      `${recorded} -> ${upstream} must stay recorded or those findings come back`,
    );
  }
});
