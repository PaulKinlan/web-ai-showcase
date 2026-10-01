// lib/webllm-race-policy.mjs — WHEN a WebLLM call may be retried, as a pure decision.
//
// The MLC engine's weight-load and prefill paths can lose a WebGPU buffer-mapping race:
//   Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved.
// It rejects the call (a visitor sees "Failed.", a cell records zero deltas) and it is intermittent —
// roughly half the loads on this box, in more than one family — so it is a browser/driver race above
// any one page (web-ai-showcase-w03), and a bounded retry is the honest response.
//
// The decision lives here, not inline in a worker, so the two things that make a retry safe can be
// TESTED rather than asserted:
//   1. the matcher is narrow — only this race is retryable; every other error must propagate on the
//      FIRST attempt (a broad matcher is a swallow, and a swallow hides real failures);
//   2. a call that already streamed output is never retried — re-running it would duplicate text the
//      reader has seen, so partial output is a hard stop even for the race.

/** Narrow: the exact messages this race produces, nothing else. */
export const GPU_RACE = /Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved|Buffer was unmapped before mapping was resolved/i;

export function isGpuRace(err) {
  return GPU_RACE.test(String(err?.message ?? err ?? ""));
}

/**
 * Decide whether one failed attempt may be retried.
 * @param {unknown} err the error the attempt threw
 * @param {number} attempt the attempt that just failed (1-based)
 * @param {{attempts?:number, emitted?:boolean}} [opts] max attempts (default 3), and whether the
 *        attempt already streamed output to the reader
 * @returns {{retry:boolean, reason:string, nextAttempt?:number}}
 */
export function retryPlan(err, attempt, { attempts = 3, emitted = false } = {}) {
  if (!isGpuRace(err)) return { retry: false, reason: "not-the-gpu-race" };
  if (emitted) return { retry: false, reason: "partial-output-already-streamed" };
  if (!Number.isFinite(attempt) || attempt >= attempts) return { retry: false, reason: "attempts-exhausted" };
  return { retry: true, reason: "gpu-race", nextAttempt: attempt + 1 };
}
