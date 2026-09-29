#!/usr/bin/env node
// check-webllm-race-retry — proves the retry policy that lets a WebLLM cell survive the GPU
// buffer-mapping race (web-ai-showcase-w03), including the case that must NOT be retried.
//
// The race is real and intermittent: `Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was
// unmapped before mapping was resolved` rejects a weight load or a first generation roughly half the
// time on this box, in more than one family, and leaves a visitor's page at "Failed." when it is not
// handled. A bounded retry is the response — but a retry with a broad matcher, or one that re-runs a
// call which already streamed text, is worse than no retry at all:
//
//   * a broad matcher SWALLOWS real failures (a broken model, a missing ABI) and hides them behind
//     retries that will never succeed;
//   * retrying after partial output DUPLICATES text the reader has already seen.
//
// This check is the negative test for both, plus the positive case and the attempt bound.
import { isGpuRace, retryPlan } from "../lib/webllm-race-policy.mjs";

let failed = 0;
const check = (what, ok, detail) => {
  if (ok) {
    console.log(`  PASS  ${what}`);
  } else {
    console.error(`  FAIL  ${what}${detail === undefined ? "" : ` — ${JSON.stringify(detail)}`}`);
    failed++;
  }
};

const RACE = new Error(
  "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved.",
);
const OTHER = new Error("Model initialisation failed. Can't create a session. ERROR_CODE: 9, Failed to find kernel for MemcpyToHost(1)");

check("the race is recognised by its exact shape", isGpuRace(RACE) === true, isGpuRace(RACE));
check("a different engine failure is NOT the race", isGpuRace(OTHER) === false, isGpuRace(OTHER));
check("a bare string of the race is recognised", isGpuRace(String(RACE.message)) === true);

const positive = retryPlan(RACE, 1, { attempts: 3 });
check(
  "the race on the first attempt is retryable (positive case)",
  positive.retry === true && positive.nextAttempt === 2,
  positive,
);
check(
  "the race on the LAST attempt is not retryable (attempt bound)",
  retryPlan(RACE, 3, { attempts: 3 }).retry === false,
  retryPlan(RACE, 3, { attempts: 3 }),
);
check(
  "the race after PARTIAL OUTPUT is never retried (no duplicated text)",
  retryPlan(RACE, 1, { attempts: 3, emitted: true }).retry === false,
  retryPlan(RACE, 1, { attempts: 3, emitted: true }),
);
const negative = retryPlan(OTHER, 1, { attempts: 3 });
check(
  "a DIFFERENT engine error fails on the first attempt, un-retried (negative case)",
  negative.retry === false && negative.reason === "not-the-gpu-race",
  negative,
);
check(
  "a null error is not retried",
  retryPlan(null, 1, { attempts: 3 }).retry === false,
  retryPlan(null, 1, { attempts: 3 }),
);

console.log(
  failed === 0
    ? "webllm-race-retry: OK — the retry is narrow, bounded, never duplicates streamed output, and a different engine error still fails first-try"
    : `webllm-race-retry: FAILED — ${failed} check(s) above`,
);
process.exit(failed === 0 ? 0 : 1);
