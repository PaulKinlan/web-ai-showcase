// Shared, accessible model-loading UX. THE single adoption point for every model page.
//
// Behaviour (Paul's mandate): a valid CURRENT on-device model initialises AUTOMATICALLY — returning
// users never click "Load" for an already-local current version. We only surface:
//   • Download  — model is absent locally (loading would transfer assets)
//   • Re-download — some cached assets were evicted (partial)
//   • Update    — a newer model revision exists than the validated cached one (the cached one still
//                 auto-inits and works; Update is optional)
// plus an honest capability fallback (needs-WebGPU / unsupported) and a retry/recovery path on failure.
// Never falls back to fake output. Status is announced via role="status" aria-live; every control is a
// real, labelled, keyboard-operable button.
//
// Usage:
//   import { createModelLoader } from "/lib/model-loader.js";
//   const loader = createModelLoader({
//     mount: document.getElementById("loader"),
//     model: { modelId, runtime, dtype, sizeMB, requiresWebGPU },
//     load: async (onProgress) => (await loadPipeline({task, model:modelId, dtype, onProgress})).pipe,
//     onReady: (instance) => { /* enable controls, keep instance */ },
//     // Optional — opt into genuine memory release. Terminate the worker / dispose the pipeline /
//     // release the ORT session here. When provided, a "Release from memory" control appears once
//     // ready; it measures origin-wide memory before + after, then drops to a re-initialisable state
//     // (assets stay cached — no re-download). Omit it and the loader behaves exactly as before.
//     dispose: async (instance) => { instance.dispose(); },
//     onDispose: () => { /* disable run controls; onReady re-enables them on reload */ },
//   });

import {
  checkModelUpdate,
  clearModelCache,
  inspectModel,
  recordValidated,
  settleWithin,
  storageEstimate,
} from "./model-cache.js";
import { createDownloadTracker } from "./download-tracker.mjs";
import { transformersAdapter } from "./download-adapters.mjs";
import "./model-download-status.mjs"; // registers <model-download-status>
import "./model-memory-diagnostics.mjs"; // registers <model-memory-diagnostics>

async function adapterAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

const STATE_TEXT = {
  checking: "Checking for a local copy…",
  "check-timeout": "The local model check took too long.",
  initialising: "Initialising the on-device model…",
  ready: "Model ready — running locally on your device.",
  downloading: "Downloading model…",
  "download-required": "This model isn't on your device yet.",
  partial: "Some cached model files were evicted by the browser.",
  update: "A newer version of this model is available.",
  unsupported: "This model can't run on this device/browser.",
  released: "Model released from memory. Its files stay cached on your device.",
  error: "Model initialisation failed.",
};

// ── failure classification (bead web-ai-showcase-1ge; ADR 2i3t as NARROWED by coord 2026-10-12) ──
//
// A load can fail in two materially different ways here:
//   * TRANSIENT — retrying can help: network/fetch failure, timeout, a lost device, out-of-memory, or
//     a generic "could not create a session" (all of which we have seen recover);
//   * DIAGNOSTICALLY FATAL — retrying cannot help, because the runtime reports a KERNEL IT DOES NOT
//     HAVE for this graph. That is a static property of graph + execution provider, so re-running the
//     identical initialisation is a dead end (the failure that trapped phi-3.5-vision for 14+ minutes
//     and timed its validator out twice).
//
// Only the second kind is denied a Retry, and only the ONE narrow signature below recognises it: ORT's
// missing-kernel message. Deliberately NOT matched on "Could not create a session" alone, which the
// ADR's original wording would have covered — that class includes out-of-memory and lost devices,
// which do recover, and this loader backs every demo, so a false positive here would remove recovery
// from demos that need it.
const MISSING_ORT_KERNEL = /Failed to find kernel for\s+([A-Za-z_][A-Za-z0-9_]*)/i;

/**
 * Decide how a load failure must be presented. Pure: no DOM, no globals, so every branch is directly
 * testable without a browser (test/model-loader-failure.test.mjs).
 */
export function classifyLoadFailure(err) {
  const message = String(err?.message || err);
  const missing = MISSING_ORT_KERNEL.exec(message);
  if (!missing) {
    return { kind: "transient", retry: true, kernel: null, state: "error", copy: message };
  }
  return {
    kind: "missing-ort-kernel",
    retry: false,
    kernel: missing[1],
    state: "error",
    // Wording is deliberate and bounded: it states what was OBSERVED about this configuration on this
    // device/runtime. It is NOT a claim that the platform cannot run the model (that needs a real
    // adapter and a run — see bead 1ge), and it does not use the "unsupported" state for that reason.
    copy:
      `This model configuration could not initialize on this device/runtime: the runtime has no ` +
      `${missing[1]} kernel for it. Retrying the same initialisation will not help.`,
  };
}

/**
 * Apply the classification to the status line and the recovery affordance. The DOM operations are
 * injected so the wiring is testable without a browser; the rendered result is what the browser half
 * of the bead verifies.
 */
export function presentLoadFailure(err, { setState, retryButton, onError } = {}) {
  const decision = classifyLoadFailure(err);
  setState(decision.state, decision.copy);
  if (decision.retry) retryButton?.();
  onError?.(err);
  return decision;
}

export function createModelLoader({ mount, model, load, onReady, onError, dispose, onDispose }) {
  // Preserve the historical three-part key for mutable/unpinned families so existing validated
  // caches continue to auto-initialise. Only immutable pinned families extend the identity with the
  // revision; treating an absent revision as a literal `main` segment would strand every legacy
  // validation record and incorrectly ask returning users to download again.
  const baseKey = `${model.runtime || "transformers.js"}::${model.modelId}::${
    model.dtype || "default"
  }`;
  const key = model.revision ? `${baseKey}::${model.revision}` : baseKey;
  const idBase = `ldr-${model.modelId.replace(/[^a-z0-9]+/gi, "-")}`;
  mount.innerHTML = `
    <div class="model-loader panel" data-state="checking">
      <p class="status" role="status" aria-live="polite" id="${idBase}-status">${STATE_TEXT.checking}</p>
      <model-download-status class="loader-progress"${
    model.sizeMB ? ` size-mb="${model.sizeMB}"` : ""
  } hidden></model-download-status>
      <div class="loader-actions"></div>
      <p class="status loader-update" id="${idBase}-update" role="status" aria-live="polite" hidden></p>
      <p class="muted loader-detail" id="${idBase}-detail"></p>
      <model-memory-diagnostics></model-memory-diagnostics>
    </div>`;
  const root = mount.querySelector(".model-loader");
  const statusEl = root.querySelector(".status");
  const dl = root.querySelector("model-download-status"); // the honest multi-file progress panel
  const actions = root.querySelector(".loader-actions");
  const updateEl = root.querySelector(".loader-update");
  const detailEl = root.querySelector(".loader-detail");
  const memory = root.querySelector("model-memory-diagnostics");
  let instance = null;
  let updateInfo = null;
  let detailGeneration = 0;

  // Route the demo's raw onProgress payload into the download-tracker via the right runtime adapter, then
  // paint the <model-download-status> panel. Shape-discriminated so it works across families without the
  // page changing: {status,…} = Transformers.js / MediaPipe file vocabulary; {progress:0..1} = WebLLM's
  // runtime-owned overall fraction. A fresh tracker + adapters are built per load (see runLoad).
  let tracker = null, tAdapter = null, aggregateStarted = false;
  const feed = (evt) => {
    if (tracker) dl.update(tracker.ingest(evt));
  };
  function newLoadSession() {
    tracker = createDownloadTracker();
    tAdapter = transformersAdapter(feed);
    aggregateStarted = false;
    dl.reset();
  }
  const onProgress = (p) => {
    if (!p || typeof p !== "object") return;
    // Per-file event ({status,file/loaded/total}, or any non-"progress" status) → the multi-file adapter
    // (Transformers.js real bytes; MediaPipe initiate/ready).
    const perFile = typeof p.status === "string" &&
      (p.status !== "progress" || p.file != null || p.name != null || p.loaded != null ||
        p.total != null);
    if (perFile) {
      tAdapter(p);
      return;
    }
    // A single OVERALL fraction/percent with NO per-file bytes → an honest runtime-owned aggregate (never
    // fabricated per-file counts). Covers WebLLM {text,progress:0..1} and raw-ORT self-download
    // {status:"progress", progress:0..100} — the old last-callback-wins single percentage, surfaced calmly.
    if (typeof p.progress === "number") {
      if (!aggregateStarted) {
        aggregateStarted = true;
        feed({ status: "phase", phase: "downloading" });
      }
      feed({
        status: "aggregate",
        ratio: p.progress > 1 ? p.progress / 100 : p.progress,
        label: model.modelId,
      });
    }
  };

  function setState(state, extra = "") {
    root.dataset.state = state;
    statusEl.textContent = (STATE_TEXT[state] || state) + (extra ? " " + extra : "");
    statusEl.classList.toggle("err", state === "error" || state === "unsupported");
    statusEl.classList.toggle("ok", state === "ready");
  }
  function button(label, cls, onClick) {
    actions.innerHTML = "";
    const b = document.createElement("button");
    b.textContent = label;
    if (cls) b.className = cls;
    b.addEventListener("click", onClick);
    actions.append(b);
    return b;
  }
  function clearActions() {
    actions.innerHTML = "";
  }
  function detail(state, info) {
    // Paint deterministic model details immediately. Storage estimation is optional, slow on some
    // browsers, and must never hold the availability state machine hostage.
    const generation = ++detailGeneration;
    const size = model.sizeMB ? `~${model.sizeMB} MB` : "size varies";
    const bits = [`${model.modelId}`, `${model.runtime || "transformers.js"}`, size];
    if (Number.isFinite(info?.localCheckMs)) bits.push(`local check ${info.localCheckMs} ms`);
    if (state === "update") {
      bits.push(
        `cached ${info?.cachedRevision?.slice(0, 7)} → new ${info?.remoteRevision?.slice(0, 7)}`,
      );
    }
    const base = bits.join(" · ");
    detailEl.textContent = base;
    void storageEstimate(600).then((est) => {
      if (!est || generation !== detailGeneration) return;
      detailEl.textContent = `${base} · ${(est.usage / 1e6) | 0} MB cached of ${
        (est.quota / 1e6) | 0
      } MB`;
    });
  }

  // Show the download panel (the single live region during a transfer) or the status line (pre/post).
  function showDownloadPanel(on) {
    dl.hidden = !on;
    statusEl.hidden = on;
  }

  async function runLoad(kind) {
    clearActions();
    setState(kind === "download" ? "downloading" : "initialising");
    newLoadSession();
    showDownloadPanel(true); // the <model-download-status> panel announces phase + shows per-file progress
    feed({ status: "phase", phase: kind === "download" ? "downloading" : "checking" });
    try {
      // No automatic lifecycle memory capture: the native API answers only after a GC pass, measured
      // at ~19s on an idle page and ~60s with a model resident in BOTH Chrome 150 and Chromium 152
      // (web-ai-showcase-cgr), so a phase capture can never land inside a budget that does not stall
      // the loader. Measurement is explicit: the visitor clicks "Measure memory".
      instance = await load(onProgress);
      feed({ status: "ready" }); // 100% + "Model ready — running locally on your device."
      // The runtime has successfully created the model: let the user proceed NOW. Cache inventory,
      // validation-record persistence, revision discovery, and memory diagnostics are background work.
      setState("ready");
      renderManageControls();
      onReady?.(instance);
      void settleWithin(
        recordValidated({
          key,
          modelId: model.modelId,
          runtime: model.runtime,
          dtype: model.dtype,
          revision: model.revision,
        }),
        5000,
        "Model validation record",
      ).catch(() => {
        detailEl.textContent +=
          " · Cache validation could not be saved; you may need to prepare the model again next visit.";
      });
      // (Model-ready memory capture removed for the same reason — see the note above the load path.)
    } catch (err) {
      // A load can fail after constructing a worker/model. If it became the live instance, tear it
      // down before offering Retry so retries cannot accumulate workers or stale page state.
      await disposeCurrent("Model load failed");
      showDownloadPanel(false); // surface the specific error on the status line
      // Retry is offered only when the failure is not the narrow, diagnostic missing-kernel class
      // (bead web-ai-showcase-1ge): an initialisation whose kernel is absent cannot succeed on repeat.
      presentLoadFailure(err, {
        setState,
        retryButton: () => button("Retry", "secondary", () => runLoad(kind)),
        onError,
      });
    }
  }

  async function disposeCurrent(reason = "Model disposed") {
    if (!instance) return;
    const current = instance;
    try {
      if (dispose) await dispose(current, reason);
    } finally {
      // Drop the live reference and synchronise route controls even if runtime disposal reports an
      // error. Keeping a half-disposed instance would be less safe than requiring reinitialisation.
      instance = null;
      try {
        onDispose?.();
      } catch { /* route teardown cannot wedge loader lifecycle */ }
    }
  }

  function makeButton(label, cls, onClick) {
    const b = document.createElement("button");
    b.textContent = label;
    if (cls) b.className = cls;
    b.style.marginInlineStart = "0.5rem";
    b.addEventListener("click", onClick);
    return b;
  }

  function clearCacheButton() {
    // A real per-model cache-clear (accessible, confirmable): first disposes the live instance,
    // synchronises route controls, then removes assets. Inference can never continue behind a
    // "download required" state and a later download cannot replace/leak a live worker.
    return makeButton("Clear cached model", "secondary", async () => {
      clearActions();
      setState("checking", "Releasing memory before clearing cached files…");
      try {
        await disposeCurrent("Cached model cleared");
        setState("checking", "Clearing this model's cached files…");
        const n = await settleWithin(
          clearModelCache(model.modelId, key),
          5000,
          "Clear model cache",
        );
        updateInfo = null;
        updateEl.hidden = true;
        showDownloadPanel(false); // back to the pre-download status line + Download button
        setState("download-required", `Cleared ${n} cached file(s).`);
        detail("download-required");
        button(
          `Download model (${model.sizeMB ? "~" + model.sizeMB + " MB" : ""})`,
          "",
          () => runLoad("download"),
        );
        actions.append(clearCacheButton());
      } catch (err) {
        showDownloadPanel(false);
        setState("error", "The browser did not finish clearing the cached model.");
        detailEl.textContent = String(err?.message || err);
        button("Retry clear", "secondary", () => {
          const retry = clearCacheButton();
          retry.click();
        });
      }
    });
  }

  // Genuine memory release: hand the in-memory model back to the demo's dispose(), then drop to a
  // re-initialisable state. Assets stay cached, so re-loading is an init (no re-download). Origin-wide
  // memory is measured before + after (labelled origin-wide by <model-memory-diagnostics>, honest when
  // measureUserAgentSpecificMemory is unavailable off a cross-origin-isolated host).
  async function releaseModel() {
    if (!instance || !dispose) return;
    clearActions();
    showDownloadPanel(false);
    setState("initialising", "Releasing from memory…");
    // Release path no longer samples memory: automatic captures cannot land (see the load path note).
    try {
      await disposeCurrent("Model released from memory");
    } catch (err) {
      // A dispose that throws still drops the reference and synchronises controls in disposeCurrent.
      detailEl.textContent = `Release reported an error: ${String(err?.message || err)}`;
    }
    setState("released");
    await detail("released");
    button("Load model into memory", "", () => runLoad("init"));
    actions.append(clearCacheButton());
    // (After-release capture removed with the rest of the automatic sampling.)
  }

  async function updateModel() {
    clearActions();
    updateEl.hidden = true;
    setState("checking", "Releasing memory before updating…");
    try {
      await disposeCurrent("Model update requested");
      setState("checking", "Clearing the older cached version…");
      await settleWithin(
        clearModelCache(model.modelId, key),
        5000,
        "Clear older model version",
      );
      updateInfo = null;
      await runLoad("download");
    } catch (err) {
      showDownloadPanel(false);
      setState("error", "The browser did not finish clearing the older model version.");
      detailEl.textContent = String(err?.message || err);
      button("Retry update", "secondary", updateModel);
    }
  }

  function renderManageControls() {
    // Once ready, offer Release (frees RAM, keeps cache), cache-clear, and a non-blocking update action.
    clearActions();
    if (dispose) actions.append(makeButton("Release from memory", "secondary", releaseModel));
    actions.append(clearCacheButton());
    if (updateInfo) actions.append(makeButton("Update model", "secondary", updateModel));
  }

  function discoverUpdate(info) {
    // An immutable revision is intentionally fixed: there is no mutable "latest" to advertise.
    // Explicitly background-only for unpinned routes: this request never blocks local availability.
    if (model.revision || !info?.record) return;
    void checkModelUpdate({ modelId: model.modelId, record: info.record }).then((available) => {
      if (!available) return;
      updateInfo = available;
      updateEl.textContent =
        "Update available. You can keep using the cached model or update when convenient.";
      updateEl.hidden = false;
      detail("update", available);
      if (root.dataset.state === "ready") renderManageControls();
    });
  }

  async function start() {
    clearActions();
    showDownloadPanel(false);
    setState("checking");
    // 1) Capability gate for runtimes that truly require WebGPU (e.g. WebLLM) — honest, no fake.
    if (model.requiresWebGPU) {
      let available;
      try {
        available = await settleWithin(
          adapterAvailable(),
          5000,
          "WebGPU availability check",
        );
      } catch (err) {
        setState(
          "check-timeout",
          "The WebGPU check took too long. Retry when the browser is responsive.",
        );
        detailEl.textContent = String(err?.message || err);
        button("Retry capability check", "secondary", start);
        return;
      }
      if (!available) {
        setState(
          "unsupported",
          "It needs WebGPU (no GPU adapter here). Enable it via chrome://gpu / a WebGPU-capable browser.",
        );
        await detail("unsupported");
        onError?.(new Error("needs-webgpu"));
        return;
      }
    }
    // 2) Inspect on-device availability and drive the state machine. The inspection is deliberately
    // bounded: the IDB validation record resolves in ~1 ms and the file probe is a single cache
    // pass that cannot hold the check open (a slow probe degrades optimistically to 'current'), so
    // a validated model clears this step almost immediately.
    let info;
    const localCheckStarted = performance.now();
    try {
      info = await settleWithin(
        inspectModel({ key, timeoutMs: 300 }),
        450,
        "Local model availability check",
      );
    } catch (err) {
      setState("check-timeout", "Retry the check, or continue and verify the model now.");
      detailEl.textContent =
        `${model.modelId} · The browser's local validation store did not answer within 450 ms. You can retry or continue without waiting.`;
      button("Retry local check", "secondary", start);
      actions.append(makeButton(
        `Continue${model.sizeMB ? ` (may download ~${model.sizeMB} MB)` : ""}`,
        "",
        () => runLoad("download"),
      ));
      return;
    }
    info.localCheckMs = Math.max(0, Math.round(performance.now() - localCheckStarted));
    root.dataset.localCheckMs = String(info.localCheckMs);
    detail(info.state, info);
    switch (info.state) {
      case "current":
        // Already local — AUTO-initialise immediately. Check for a newer revision in the background;
        // any update becomes an optional notice/button and never blocks this cached copy.
        discoverUpdate(info);
        await runLoad("init");
        break;
      case "partial":
        setState("partial", `${info.missing?.length ?? 0} file(s) evicted.`);
        button("Re-download missing assets", "", () => runLoad("download"));
        break;
      case "absent":
      default:
        setState("download-required");
        button(
          `Download model${model.sizeMB ? " (~" + model.sizeMB + " MB)" : ""}`,
          "",
          () => runLoad("download"),
        );
        break;
    }
  }

  start().catch((err) => {
    showDownloadPanel(false);
    setState("error", String(err?.message || err));
    onError?.(err);
    button("Retry local check", "secondary", start);
  });
  return {
    get instance() {
      return instance;
    },
    refresh: start,
  };
}

// Every demo page mounts a loader, so registering the service worker here gives EVERY model page
// (and ladder page) the same offline app-shell + cache-first shell serving — previously only a
// subset of pages registered it, so most pages re-fetched their HTML/CSS/JS from the network on
// every visit before the local-model check could even begin.
if (typeof navigator !== "undefined" && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("/web-ai-showcase/sw.js").catch(() => {});
}
