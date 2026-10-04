// web-ai-showcase-6b5: pairwise per-call chunk accounting and label honesty.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { checkPairwiseDivergence } from "../scripts/browser.mjs";

const TWO_GEN_VALIDATORS = [
  "scripts/validate-granite-4-350m.mjs",
  "scripts/validate-baguettotron.mjs",
  "scripts/validate-qwen3-0-6b-wasm.mjs",
  "scripts/validate-tinyllama-chat.mjs",
];

test("checkPairwiseDivergence validates single-call generations correctly", () => {
  // Divergent streaming generation
  const res1 = checkPairwiseDivergence([{ chunks: 30, tokens: 42 }]);
  assert.equal(res1.pass, true);
  assert.equal(res1.anyStreamed, true);
  assert.equal(res1.needsDivergence, true);

  // Bug signature: tokens equal chunks on long generation (>= 10 tokens)
  const res2 = checkPairwiseDivergence([{ chunks: 30, tokens: 30 }]);
  assert.equal(res2.pass, false, "equal tokens and chunks on long generation must fail");
  assert.equal(res2.anyStreamed, true);
  assert.equal(res2.needsDivergence, true);

  // Short generation (< 10 tokens): equality allowed
  const res3 = checkPairwiseDivergence([{ chunks: 4, tokens: 4 }]);
  assert.equal(res3.pass, true, "short generation allows equality");
  assert.equal(res3.anyStreamed, true);
  assert.equal(res3.needsDivergence, false);

  // Non-streaming generation: zero chunks, real tokens
  const res4 = checkPairwiseDivergence([{ chunks: 0, tokens: 15 }]);
  assert.equal(res4.pass, true);
  assert.equal(res4.anyStreamed, false);
  assert.equal(res4.needsDivergence, false);

  // Non-streaming generation with zero tokens: failure
  const res5 = checkPairwiseDivergence([{ chunks: 0, tokens: 0 }]);
  assert.equal(res5.pass, false);
});

test("checkPairwiseDivergence validates multi-generation calls pairwise", () => {
  // Both calls diverge -> passes
  const validTwoGen = [
    { chunks: 35, tokens: 45 },
    { chunks: 40, tokens: 52 },
  ];
  const resValid = checkPairwiseDivergence(validTwoGen);
  assert.equal(resValid.pass, true);
  assert.equal(resValid.anyStreamed, true);
  assert.equal(resValid.needsDivergence, true);
  assert.equal(resValid.details.length, 2);
  assert.ok(resValid.details.every((d) => d.pass));

  // Mixed calls where Call 1 diverges but Call 2 does not -> MUST FAIL
  const partialFailure = [
    { chunks: 35, tokens: 45 },
    { chunks: 40, tokens: 40 },
  ];
  const resPartial = checkPairwiseDivergence(partialFailure);
  assert.equal(resPartial.pass, false, "failure on any generation must fail the cell");
  assert.equal(resPartial.details[0].pass, true);
  assert.equal(resPartial.details[1].pass, false);
});

test("checkPairwiseDivergence catches the bead 6b5 aggregate cancellation mutant", () => {
  // The exact failure mode from bead 6b5:
  // Under the bug-restored mutant, Call 1 returns 32 chunks vs 32 IDs.
  // Call 2 returns 33 chunks vs 32 IDs.
  // Sum: 65 chunks vs 64 IDs (65 !== 64 was true, so naive aggregate check passed).
  const mutantRun = [
    { chunks: 32, tokens: 32 },
    { chunks: 33, tokens: 32 },
  ];

  // Old naive aggregate comparison:
  const sumChunks = mutantRun.reduce((a, b) => a + b.chunks, 0); // 65
  const sumTokens = mutantRun.reduce((a, b) => a + b.tokens, 0); // 64
  const oldNaivePassed = sumChunks !== sumTokens; // 65 !== 64 -> true! (BUG MASKED)
  assert.equal(oldNaivePassed, true, "old aggregate comparison falsely passed");

  // Pairwise check:
  const resPairwise = checkPairwiseDivergence(mutantRun);
  assert.equal(resPairwise.pass, false, "pairwise check correctly identifies Call 1 bug");
  assert.equal(resPairwise.details[0].pass, false, "Call 1 failed divergence");
});

test("all four two-generation validators wire pairwise per-call chunk accounting and honest label", () => {
  for (const scriptPath of TWO_GEN_VALIDATORS) {
    const text = readFileSync(scriptPath, "utf8");

    // Must import checkPairwiseDivergence
    assert.ok(
      text.includes("checkPairwiseDivergence"),
      `${scriptPath} must import checkPairwiseDivergence`,
    );

    // Must record per-call chunk accounting in installChunkCounter
    assert.ok(
      text.includes("callChunks += 1") || text.includes("callChunks++"),
      `${scriptPath} must increment callChunks in onToken callback`,
    );
    assert.ok(
      text.includes("window.__calls"),
      `${scriptPath} must track window.__calls`,
    );
    assert.ok(
      text.includes("calls: window.__calls"),
      `${scriptPath} must evaluate calls from window.__calls in evalJSON`,
    );

    // Label honesty: old misleading label must be absent
    assert.ok(
      !text.includes("generated IDs outnumber decoded chunks"),
      `${scriptPath} must not contain stale 'generated IDs outnumber decoded chunks' label`,
    );

    // Label honesty: accurate label must be present
    assert.ok(
      text.includes("resolved ID count differs from the decoded-chunk count"),
      `${scriptPath} must use 'resolved ID count differs from the decoded-chunk count' label`,
    );
  }
});
