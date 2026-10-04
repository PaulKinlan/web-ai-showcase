// lib/webllm-race-policy.mjs — WHEN a WebLLM call may be retried, as a pure decision,
// and the user-facing "GPU hiccup — try again" affordance (web-ai-showcase-n81, web-ai-showcase-w03).
//
// The MLC engine's weight-load and prefill paths can lose a WebGPU buffer-mapping race:
//   Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved.
// It rejects the call (a visitor sees "Failed.", a cell records zero deltas) and it is intermittent —
// roughly half the loads on this box, in more than one family — so it is a browser/driver race above
// any one page.
//
// When a generation rejects with this race before any deltas stream, the page presents a visible
// "GPU hiccup — try again" affordance. Clicking it reloads/rebuilds the engine worker and retries
// the generation once; if it fails again, an honest final failure message is shown.

/** Narrow: the exact messages this race produces, nothing else. */
export const GPU_RACE =
  /Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved|Buffer was unmapped before mapping was resolved/i;

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
  if (!Number.isFinite(attempt) || attempt >= attempts) {
    return { retry: false, reason: "attempts-exhausted" };
  }
  return { retry: true, reason: "gpu-race", nextAttempt: attempt + 1 };
}

/**
 * Renders a visible "GPU hiccup — try again" retry button if the error is a WebGPU buffer race
 * and no tokens have streamed yet (web-ai-showcase-n81).
 *
 * @param {object} opts
 * @param {Error|unknown} opts.error The error caught from the generation
 * @param {number} [opts.deltasStreamed=0] How many deltas were streamed (must be 0 to retry)
 * @param {HTMLElement} [opts.container] Element to attach the button to
 * @param {HTMLElement} [opts.statusEl] Element displaying the status message
 * @param {() => Promise<void>} opts.onRetry Async function to rebuild engine and rerun generation
 * @param {(err: Error|unknown) => void} [opts.onFinalFail] Callback if retry fails again
 * @param {string} [opts.label="GPU hiccup — try again"] Button label
 * @returns {HTMLButtonElement|null} The rendered button or null if not eligible
 */
export function renderGpuHiccupAffordance({
  error,
  deltasStreamed = 0,
  container,
  statusEl,
  onRetry,
  onFinalFail,
  label = "GPU hiccup — try again",
}) {
  if (!isGpuRace(error) || deltasStreamed > 0) return null;
  const target = container || statusEl;
  if (!target) return null;

  // Clean up any existing affordance
  target.querySelector?.(".gpu-hiccup-btn")?.remove();

  const btn = (typeof document !== "undefined" && typeof document.createElement === "function")
    ? document.createElement("button")
    : null;
  if (!btn) return null;

  btn.type = "button";
  btn.className = "button secondary gpu-hiccup-btn";
  btn.textContent = label;
  btn.setAttribute("aria-label", label);
  btn.style.marginLeft = "0.5rem";
  btn.style.display = "inline-block";
  btn.style.verticalAlign = "middle";

  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    btn.disabled = true;
    btn.textContent = "Rebuilding engine…";
    if (statusEl) {
      statusEl.textContent = "Rebuilding GPU engine…";
      statusEl.classList.remove("err");
    }
    try {
      if (typeof onRetry === "function") {
        await onRetry();
      }
      btn.remove();
    } catch (finalErr) {
      btn.remove();
      const failMsg = "GPU hiccup recovery failed: " + (finalErr?.message || finalErr);
      if (statusEl) {
        statusEl.textContent = failMsg;
        statusEl.classList.add("err");
      }
      if (typeof onFinalFail === "function") {
        onFinalFail(finalErr);
      }
    }
  });

  target.appendChild(btn);
  return btn;
}
