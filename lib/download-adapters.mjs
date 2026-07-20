// download-adapters.mjs — runtime → reducer adapters (Task 2b · Phase 2).
//
// The download reducer (lib/download-tracker.mjs) speaks ONE runtime-neutral event vocabulary:
//   { status: "initiate",      file, name, total? }        a required file was discovered
//   { status: "download",      file, name }                its transfer began
//   { status: "progress",      file, name, loaded, total } bytes so far
//   { status: "done",          file, name }                that file finished / cache-hit
//   { status: "file-verifying" | "file-paused", file }     self-download integrity / pause
//   { status: "aggregate", ratio: 0..1|null, label }       runtime-owned overall fraction (no per-file bytes)
//   { status: "phase", phase } | { status: "ready" } | { status: "error", message }
//
// Each runtime family reports progress differently. These adapters translate a runtime's NATIVE callback
// into that vocabulary so the same reducer + UI + <model-download-status> component can present every
// demo consistently — and HONESTLY: a runtime that only exposes an overall fraction (WebLLM) or nothing
// (MediaPipe) is surfaced as a runtime-owned aggregate, never as fabricated per-file byte counts.
//
// One adapter per family from the route inventory (download-routes.json):
//   transformers-pipeline / -from_pretrained / -wrapped → transformersAdapter (real per-file bytes)
//   webllm                                              → webllmAdapter       (runtime-owned fraction)
//   mediapipe                                           → mediapipeAdapter     (runtime-owned, indeterminate)
//   raw-ort / -resumable-prefetch                       → already emit the vocabulary via lib/model-download.js
//                                                         / lib/model-prefetch.mjs (site-controlled)

const KNOWN = new Set([
  "initiate",
  "download",
  "progress",
  "done",
  "file-verifying",
  "file-paused",
]);

/**
 * Transformers.js pipeline() / from_pretrained() progress_callback → reducer events.
 * TJS already emits {status:"initiate"|"download"|"progress"|"done", file, name, loaded, total} per file,
 * plus a terminal {status:"ready"} from pipeline(). We forward the known download statuses (keeping stable
 * `file` identity) and the ready signal, and drop anything else. Use it as the `progress_callback`.
 * @param {(evt:object)=>void} onEvent
 * @returns {(evt:object)=>void} a Transformers.js progress_callback
 */
export function transformersAdapter(onEvent) {
  return (evt) => {
    if (!evt || typeof evt !== "object") return;
    if (KNOWN.has(evt.status)) {
      onEvent({
        status: evt.status,
        file: evt.file ?? evt.name,
        name: evt.name ?? evt.file,
        loaded: evt.loaded,
        total: evt.total,
      });
    } else if (evt.status === "ready") {
      onEvent({ status: "ready" });
    }
    // (other statuses — e.g. "start"/"update" from generation — are not download events; ignore)
  };
}

/**
 * WebLLM initProgressCallback → reducer events. WebLLM reports {text, progress:0..1, timeElapsed} — an
 * OVERALL fraction across shards, with no per-file byte detail exposed to the page. We surface it honestly
 * as a runtime-owned aggregate. Wire it as `createEngine({ model, onProgress: webllmAdapter(onEvent) })`.
 * @param {(evt:object)=>void} onEvent
 * @param {{label?:string}} [opts]
 * @returns {(report:{text?:string,progress?:number})=>void}
 */
export function webllmAdapter(onEvent, { label = "WebLLM model" } = {}) {
  let started = false;
  return (report) => {
    if (!started) {
      started = true;
      onEvent({ status: "phase", phase: "downloading" });
    }
    const progress = typeof report?.progress === "number" ? report.progress : null;
    onEvent({ status: "aggregate", ratio: progress, label });
    // WebLLM does not fire a distinct "ready" — the page posts ready once CreateMLCEngine resolves; we
    // leave that to the caller so "downloaded" (fraction=1) is never conflated with "model ready".
  };
}

/**
 * MediaPipe Tasks has NO download progress callback — FilesetResolver + createFromOptions fetch the
 * .task/.tflite bundle inside the WASM runtime. We surface an honest runtime-owned, indeterminate
 * "downloading" state around that call. The caller drives the lifecycle:
 *   const mp = mediapipeAdapter(onEvent);  mp.begin();  await createFromOptions(...);  mp.ready();
 * @param {(evt:object)=>void} onEvent
 * @param {{label?:string}} [opts]
 */
export function mediapipeAdapter(onEvent, { label = "MediaPipe model bundle" } = {}) {
  return {
    begin() {
      onEvent({ status: "phase", phase: "downloading" });
      onEvent({ status: "aggregate", ratio: null, label }); // indeterminate: no progress exposed
    },
    ready() {
      onEvent({ status: "ready" });
    },
    error(message) {
      onEvent({ status: "error", message: String(message || "failed") });
    },
  };
}

/** Map a route-inventory family → the adapter it should use (for the adoption phase + docs). */
export const FAMILY_ADAPTER = {
  "transformers-pipeline": "transformersAdapter",
  "transformers-from_pretrained": "transformersAdapter",
  "transformers-wrapped": "transformersAdapter",
  "transformers-resumable-prefetch": "model-prefetch.mjs (already emits the vocabulary)",
  "webllm": "webllmAdapter",
  "mediapipe": "mediapipeAdapter",
  "raw-ort": "model-download.js (already emits the vocabulary)",
  "browser-builtin": "capability adapter (no per-file detail — browser owns the download)",
  "non-applicable": null,
};
