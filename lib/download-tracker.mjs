// download-tracker.js — a deterministic multi-file download state machine.
//
// THE problem it solves (Paul): the old loader set one progress bar to the LAST progress_callback's
// single percentage, so a 2.9 GB multi-file model "jumped between unrelated percentages" and you could
// not tell what was downloading, what was cached, what remained, or whether the model was ready.
//
// This is a PURE reducer (no DOM, no network) so it is unit-testable and reusable by any Transformers.js
// multi-file demo. You feed it the raw progress_callback events (and optional lifecycle signals); it
// maintains per-file state keyed by STABLE FILE IDENTITY and derives a byte-weighted aggregate + a calm
// high-level phase. The UI reads snapshots and paints (throttled). It handles: files discovered late,
// cached files, unknown sizes, zero-byte/metadata events, duplicate + out-of-order callbacks, and errors.
//
// Transformers.js v3.7.5 progress_callback event shape (per file):
//   { status: "initiate", file, name }                       // a required file was discovered
//   { status: "download", file, name }                       // its transfer began
//   { status: "progress", file, name, loaded, total, progress } // bytes so far (loaded/total) + %
//   { status: "done", file, name }                            // that file finished (or was cache-hit)
// plus a single terminal { status: "ready" } once the model is built. `total` can be missing/0 for
// small/metadata files. Multiple files download concurrently, so events interleave.

/** @typedef {"checking"|"discovering"|"downloading"|"verifying"|"initialising"|"ready"|"error"} Phase */
/** @typedef {"queued"|"downloading"|"verifying"|"cached"|"complete"|"failed"|"paused"} FileState */

export function createDownloadTracker({ knownFiles } = {}) {
  /** @type {Map<string, {id:string,name:string,state:FileState,loaded:number,total:number|null,error?:string,cached:boolean,seq:number}>} */
  const files = new Map();
  let seq = 0;
  let phase = /** @type {Phase} */ ("checking");
  let ready = false;
  let modelError = null;
  // Once a full denominator is known (e.g. from an HF file tree), later ratio is stable. Until then we
  // only know "total so far", so we present an honest indeterminate/known-so-far state.
  let denominatorComplete = false;

  // Optionally seed the known required files up front (from an HF tree listing) so the denominator is
  // complete immediately and late-discovered files never cause a bogus 100→0 reset.
  if (Array.isArray(knownFiles) && knownFiles.length) {
    for (const f of knownFiles) touch(f.id ?? f.file ?? f.name, f.name ?? f.file, f.total ?? null);
    denominatorComplete = true;
  }

  function idOf(file, name) {
    // `file` is the stable repo-relative path (e.g. "onnx/decoder_model_merged_q4f16.onnx"). Fall back
    // to name; last resort a synthetic id so an event is never dropped.
    return String(file || name || `unknown-${files.size}`);
  }

  function touch(file, name, total) {
    const id = idOf(file, name);
    let f = files.get(id);
    if (!f) {
      f = {
        id,
        name: name || id,
        state: "queued",
        loaded: 0,
        total: null,
        cached: false,
        seq: seq++,
      };
      files.set(id, f);
    }
    if (name && !f.name) f.name = name;
    if (typeof total === "number" && total > 0) f.total = total;
    return f;
  }

  function setPhaseFromState() {
    if (modelError) {
      phase = "error";
      return;
    }
    if (ready) {
      phase = "ready";
      return;
    }
    const list = [...files.values()];
    if (list.length === 0) {
      phase = "checking";
      return;
    }
    const anyDownloading = list.some((f) => f.state === "downloading");
    const anyVerifying = list.some((f) => f.state === "verifying");
    const allTerminal = list.every((f) =>
      f.state === "complete" || f.state === "cached" || f.state === "failed"
    );
    if (anyVerifying) phase = "verifying";
    else if (anyDownloading) phase = "downloading";
    else if (allTerminal) phase = "initialising"; // downloads done; model still being built (not ready yet)
    else phase = "discovering"; // files known/queued but transfers not started
  }

  /** Feed one RAW Transformers.js progress_callback event (or a `{status:"ready"|"error"}` signal). */
  function ingest(evt) {
    if (!evt || typeof evt !== "object") return snapshot();
    const { status } = evt;
    switch (status) {
      case "ready":
        ready = true;
        // Any still-queued file that never reported (pure cache hit) is complete.
        for (const f of files.values()) {
          if (f.state === "queued" || f.state === "downloading") {
            f.state = f.loaded > 0 ? "complete" : "cached";
            if (f.total == null && f.loaded > 0) f.total = f.loaded;
            if (f.total != null) f.loaded = f.total;
          }
        }
        break;
      case "error": {
        const f = evt.file || evt.name ? touch(evt.file, evt.name) : null;
        if (f) {
          f.state = "failed";
          f.error = String(evt.message || evt.error || "failed");
        } else {
          modelError = String(evt.message || evt.error || "failed");
        }
        break;
      }
      case "initiate":
        touch(evt.file, evt.name, evt.total);
        break;
      case "download": {
        const f = touch(evt.file, evt.name, evt.total);
        if (f.state === "queued") f.state = "downloading";
        break;
      }
      case "progress": {
        const f = touch(evt.file, evt.name, evt.total);
        // Don't regress a finished file: from_pretrained's cache reads still stream `progress` for the
        // already-complete/cached weights (3.7.5 hub.js), which would otherwise flip them back to
        // "downloading" and bounce the phase Preparing→Downloading. Keep terminal states terminal.
        if (f.state !== "complete" && f.state !== "cached") f.state = "downloading";
        // Monotonic: never let a late/duplicate event move bytes backwards.
        if (typeof evt.loaded === "number" && evt.loaded >= f.loaded) f.loaded = evt.loaded;
        if (typeof evt.total === "number" && evt.total > 0) f.total = evt.total;
        break;
      }
      case "done": {
        const f = touch(evt.file, evt.name, evt.total);
        // A "done" with no prior progress = a cache hit; with progress = a completed transfer.
        f.state = f.loaded > 0 ? "complete" : "cached";
        if (f.total != null) f.loaded = f.total;
        else if (f.loaded > 0) f.total = f.loaded;
        break;
      }
      // Custom (self-download) lifecycle signals — optional, for the resumable adapter path:
      case "file-verifying": {
        const f = touch(evt.file, evt.name, evt.total);
        f.state = "verifying";
        break;
      }
      case "file-paused": {
        const f = touch(evt.file, evt.name, evt.total);
        if (f.state !== "complete" && f.state !== "cached") f.state = "paused";
        break;
      }
      case "phase":
        if (typeof evt.phase === "string") phase = evt.phase;
        break;
      default:
        break;
    }
    setPhaseFromState();
    return snapshot();
  }

  function snapshot() {
    const list = [...files.values()].sort((a, b) => a.seq - b.seq);
    let loadedBytes = 0;
    let knownTotalBytes = 0;
    let anyUnknownTotal = false;
    let complete = 0;
    let failed = 0;
    for (const f of list) {
      loadedBytes += f.loaded;
      if (f.total != null) knownTotalBytes += f.total;
      else if (f.state !== "cached" && f.state !== "complete") anyUnknownTotal = true;
      if (f.state === "complete" || f.state === "cached") complete++;
      if (f.state === "failed") failed++;
    }
    // Byte-weighted aggregate (NOT a mean of per-file percentages).
    // - `indeterminate`: a downloading file has an unknown size → we genuinely can't compute a %.
    // - `knownSoFar`: the FULL set of files isn't confirmed yet (raw callbacks discover files lazily),
    //   so the denominator may still grow. We may show a ratio of the KNOWN bytes, but we must NEVER
    //   let it read 100% while more files could appear (a config at 100% ≠ the whole model) — so we
    //   cap it just below 1 until the denominator is complete or the model is ready.
    const totalBytes = knownTotalBytes || null;
    const indeterminate = anyUnknownTotal;
    const knownSoFar = !denominatorComplete && !ready;
    let ratio = !indeterminate && totalBytes ? Math.min(1, loadedBytes / totalBytes) : null;
    if (ratio != null && knownSoFar && ratio >= 1) ratio = 0.99; // known set done, whole model not
    return {
      phase,
      ready,
      error: modelError,
      files: list.map((f) => ({
        id: f.id,
        name: f.name,
        state: f.state,
        loaded: f.loaded,
        total: f.total,
        ratio: f.total ? Math.min(1, f.loaded / f.total) : null,
        error: f.error,
      })),
      aggregate: {
        loadedBytes,
        totalBytes,
        knownSoFarBytes: knownTotalBytes,
        indeterminate, // true ⇒ unknown file size ⇒ show "X so far", not a %
        knownSoFar, // true ⇒ denominator may still grow ⇒ label "known so far"
        ratio, // 0..<1 while knownSoFar; 0..1 once complete/ready; null when indeterminate
        fileCount: list.length,
        complete,
        failed,
      },
    };
  }

  return {
    ingest,
    snapshot,
    get phase() {
      return phase;
    },
  };
}

/** Human-readable phase label — calm, high-level, and honest (a config at 100% ≠ model ready). */
export const PHASE_LABEL = {
  checking: "Checking for a local copy…",
  discovering: "Finding the files this model needs…",
  downloading: "Downloading model files…",
  verifying: "Verifying downloaded files…",
  initialising: "Preparing the model to run…",
  ready: "Model ready — running locally on your device.",
  error: "Something went wrong.",
};

/** Format bytes with readable units (deterministic; used in the per-file table + aggregate). */
export function formatBytes(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}
