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

export function createModelLoader({ mount, model, load, onReady, onError, dispose, onDispose }) {
  const key = `${model.runtime || "transformers.js"}::${model.modelId}::${
    model.dtype || "default"
  }`;
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
      await memory.capture("Before model load");
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
        }),
        5000,
        "Model validation record",
      ).catch(() => {
        detailEl.textContent +=
          " · Cache validation could not be saved; you may need to prepare the model again next visit.";
      });
      void memory.capture("Model ready");
    } catch (err) {
      showDownloadPanel(false); // surface the specific error + a Retry on the status line
      setState("error", String(err?.message || err));
      onError?.(err);
      button("Retry", "secondary", () => runLoad(kind)); // clear recovery path — never fake output
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
    // A real per-model cache-clear (accessible, confirmable): removes the cached assets entirely,
    // so the next use is an honest Download — distinct from Release, which only frees RAM.
    return makeButton("Clear cached model", "secondary", async () => {
      clearActions();
      setState("checking", "Clearing this model's cached files…");
      try {
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
        button("Retry clear", "secondary", () => clearCacheButton().click());
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
    await memory.capture("Before release");
    try {
      await dispose(instance);
    } catch (err) {
      // A dispose that throws must not leave a half-released zombie: drop the reference anyway and
      // report honestly rather than pretend the model is still healthy.
      detailEl.textContent = `Release reported an error: ${String(err?.message || err)}`;
    }
    instance = null;
    try {
      onDispose?.();
    } catch { /* a demo's control-teardown must not block the release path */ }
    setState("released");
    await detail("released");
    button("Load model into memory", "", () => runLoad("init"));
    actions.append(clearCacheButton());
    await memory.capture("After release");
  }

  async function updateModel() {
    clearActions();
    updateEl.hidden = true;
    setState("checking", "Clearing the older cached version…");
    try {
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
    // Explicitly background-only: this network request never participates in local availability or ready.
    if (!info?.record) return;
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
    // 2) Inspect on-device availability and drive the state machine.
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
    captureMemory: (label = "Manual snapshot") => memory.capture(label),
  };
}
