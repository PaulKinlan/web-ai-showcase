// resumable-loader.mjs — the multi-file, resumable adoption of the shared model-load UX. Used by
// PaliGemma (the proven adoption point) and any Transformers.js demo whose worker prefetches its big
// weights via lib/model-prefetch.mjs and emits download-tracker events as `{type:"dl", evt}`.
//
// It reuses lib/model-cache.js (auto-init a valid cached model; only Download an absent one; "clear
// cached model") and drives lib/download-tracker.mjs + lib/download-ui.mjs for the honest multi-file
// UI, with REAL Pause/Resume/Discard wired to the worker's resumable prefetch (never a fake "Resume").

import { createDownloadTracker } from "./download-tracker.mjs";
import "./model-download-status.mjs"; // registers the <model-download-status> presentation/control element
import "./model-memory-diagnostics.mjs"; // registers visible origin-wide memory snapshots
import { clearModelCache, inspectModel, recordValidated, settleWithin } from "./model-cache.js";

async function adapterAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/**
 * @param {object} o
 * @param {HTMLElement} o.mount
 * @param {{modelId:string, runtime?:string, dtype?:string, sizeMB?:number, requiresWebGPU?:boolean}} o.model
 * @param {{load:Function, pause:Function, resume:Function}} o.engine  the worker-backed engine
 * @param {string[]} [o.resumableUrls]  the big-file HF URLs (for Discard-partials)
 * @param {(instance:any)=>void} [o.onReady]
 * @param {(err:any)=>void} [o.onError]
 */
export function createResumableLoader(
  { mount, model, engine, resumableUrls = [], onReady, onError },
) {
  const key = `${model.runtime || "transformers.js"}::${model.modelId}::${
    model.dtype || "default"
  }`;
  // The reusable <model-download-status> element is the presentation/control boundary; this loader stays
  // the orchestration (reducer + prefetch + cache). PaliGemma is genuinely resumable → can-pause.
  const ui = document.createElement("model-download-status");
  if (model.sizeMB) ui.setAttribute("size-mb", String(model.sizeMB));
  ui.setAttribute("can-pause", "");
  const memory = document.createElement("model-memory-diagnostics");
  mount.replaceChildren(ui, memory);
  let tracker = createDownloadTracker();
  let instance = null;
  let paused = false;

  let loadingActive = false;
  let controlMode = "";

  const feed = (evt) => {
    const snap = tracker.ingest(evt);
    ui.update(snap);
    if (!loadingActive) return;
    // Offer Pause ONLY while the resumable prefetch is actually transferring. Once we reach
    // verifying/initialising (from_pretrained building the model — NOT abortable in 3.7.5), don't show
    // a Pause that would silently no-op; show an honest disabled state instead.
    if (paused) applyControls("paused");
    else if (snap.phase === "verifying" || snap.phase === "initialising") {
      applyControls("preparing");
    } else if (snap.phase === "downloading" || snap.phase === "discovering") {
      applyControls("download");
    }
  };

  function applyControls(mode) {
    if (mode === controlMode) return; // only on transition (no per-event flicker)
    controlMode = mode;
    if (mode === "paused") {
      ui.setActions([
        {
          label: "Resume download",
          onClick: () => {
            paused = false;
            controlMode = "";
            applyControls("download");
            engine.resume(); // continues from persisted partials — a genuine resume
          },
        },
        { label: "Discard partial downloads", className: "secondary", onClick: discardPartials },
      ]);
    } else if (mode === "download") {
      ui.setActions([{
        label: "Pause download",
        className: "secondary",
        onClick: () => engine.pause(),
      }]);
    } else if (mode === "preparing") {
      // Downloads done; the model is building. Pause can't help here — say so honestly.
      ui.setActions([{
        label: "Preparing the model — can't pause",
        className: "secondary",
        disabled: true,
      }]);
    }
  }

  function readyControls() {
    async function clearCachedModel() {
      ui.setActions([]);
      ui.setStorageMessage("Clearing this model's cached files…");
      try {
        await settleWithin(
          clearModelCache(model.modelId, key),
          5000,
          "Clear model cache",
        );
        await discardPartials(true);
        instance = null;
        start(); // back to the Download state, honestly
      } catch (err) {
        feed({ status: "error", message: "The browser did not finish clearing the cached model." });
        ui.setStorageMessage(String(err?.message || err));
        ui.setActions([{
          label: "Retry clear",
          className: "secondary",
          onClick: clearCachedModel,
        }]);
      }
    }
    ui.setActions([{
      label: "Clear cached model",
      className: "secondary",
      onClick: clearCachedModel,
    }]);
  }

  async function discardPartials(silent) {
    try {
      const { clearPartial } = await import("./model-download.js");
      for (const u of resumableUrls) await clearPartial(u);
      if (!silent) {
        ui.setStorageMessage("Partial downloads discarded. A fresh download will start over.");
      }
    } catch { /* best effort */ }
  }

  async function beginLoad(kind) {
    paused = false;
    loadingActive = true;
    controlMode = "";
    tracker = createDownloadTracker();
    feed({ status: "phase", phase: kind === "download" ? "downloading" : "checking" });
    applyControls("download");
    try {
      await memory.capture("Before model load");
      await engine.load({
        onEvent: feed,
        onPaused: () => {
          paused = true;
          applyControls("paused");
        },
      });
      loadingActive = false;
      controlMode = "";
      feed({ status: "ready" });
      try {
        await settleWithin(
          recordValidated({
            key,
            modelId: model.modelId,
            runtime: model.runtime,
            dtype: model.dtype,
          }),
          3000,
          "Model validation record",
        );
      } catch {
        ui.setStorageMessage(
          "Model ready, but the browser did not save the local-cache validation record.",
        );
      }
      instance = engine;
      readyControls();
      onReady?.(engine);
      void memory.capture("Model ready");
    } catch (err) {
      loadingActive = false;
      controlMode = "";
      feed({ status: "error", message: String(err?.message || err) });
      onError?.(err);
      ui.setActions([{ label: "Retry", className: "secondary", onClick: () => beginLoad(kind) }]);
    }
  }

  async function start() {
    // Capability gate — honest, never fake output.
    if (model.requiresWebGPU) {
      let available;
      try {
        available = await settleWithin(
          adapterAvailable(),
          5000,
          "WebGPU availability check",
        );
      } catch (err) {
        feed({ status: "error", message: "The WebGPU capability check took too long." });
        ui.setStorageMessage(String(err?.message || err));
        ui.setActions([{
          label: "Retry capability check",
          className: "secondary",
          onClick: start,
        }]);
        return;
      }
      if (!available) {
        feed({ status: "error", message: "This model needs WebGPU (no GPU adapter here)." });
        ui.setStorageMessage("Enable WebGPU (chrome://gpu) or use a WebGPU-capable browser.");
        onError?.(new Error("needs-webgpu"));
        return;
      }
    }
    await ui.showStorage();
    let info;
    try {
      info = await settleWithin(
        inspectModel({ key, modelId: model.modelId }),
        7000,
        "Local model availability check",
      );
    } catch (err) {
      feed({ status: "error", message: "The local model check could not finish." });
      ui.setStorageMessage(
        `${String(err?.message || err)} Continuing may download missing assets.`,
      );
      ui.setActions([
        { label: "Retry local check", className: "secondary", onClick: start },
        {
          label: `Continue${model.sizeMB ? ` (may download ~${model.sizeMB} MB)` : ""}`,
          onClick: () => beginLoad("download"),
        },
      ]);
      return;
    }
    switch (info.state) {
      case "current":
      case "unverified":
      case "update":
        // Valid current copy on-device → AUTO-initialise (the prefetch sees every file cached and the
        // model builds from cache); no Download click for returning users.
        beginLoad("init");
        break;
      case "partial":
        // Some assets evicted — resume/refetch just the missing bytes.
        beginLoad("download");
        break;
      case "absent":
      default:
        ui.setActions([{
          label: `Download model${model.sizeMB ? " (~" + model.sizeMB + " MB)" : ""}`,
          onClick: () => beginLoad("download"),
        }]);
        break;
    }
  }

  start().catch((err) => {
    feed({ status: "error", message: String(err?.message || err) });
    onError?.(err);
    ui.setActions([{ label: "Retry local check", className: "secondary", onClick: start }]);
  });
  return {
    get instance() {
      return instance;
    },
    refresh: start,
    captureMemory: (label = "Manual snapshot") => memory.capture(label),
  };
}

// This IS the shared auto-init loader architecture (it reuses lib/model-cache.js's inspect/auto-init),
// just the multi-file resumable variant. Export under the canonical name too so pages read as using the
// shared `createModelLoader` auto-init loader (not a hand-rolled Load button).
export { createResumableLoader as createModelLoader };
