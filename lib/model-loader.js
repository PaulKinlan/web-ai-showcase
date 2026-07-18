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
//   });

import { clearModelCache, inspectModel, recordValidated, storageEstimate } from "./model-cache.js";

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
  initialising: "Initialising the on-device model…",
  ready: "Model ready — running locally on your device.",
  downloading: "Downloading model…",
  "download-required": "This model isn't on your device yet.",
  partial: "Some cached model files were evicted by the browser.",
  update: "A newer version of this model is available.",
  unsupported: "This model can't run on this device/browser.",
  error: "Model initialisation failed.",
};

export function createModelLoader({ mount, model, load, onReady, onError }) {
  const key = `${model.runtime || "transformers.js"}::${model.modelId}::${
    model.dtype || "default"
  }`;
  const idBase = `ldr-${model.modelId.replace(/[^a-z0-9]+/gi, "-")}`;
  mount.innerHTML = `
    <div class="model-loader panel" data-state="checking">
      <p class="status" role="status" aria-live="polite" id="${idBase}-status">${STATE_TEXT.checking}</p>
      <progress id="${idBase}-prog" hidden max="100" aria-labelledby="${idBase}-status"></progress>
      <div class="loader-actions"></div>
      <p class="muted loader-detail" id="${idBase}-detail"></p>
    </div>`;
  const root = mount.querySelector(".model-loader");
  const statusEl = root.querySelector(".status");
  const progEl = root.querySelector("progress");
  const actions = root.querySelector(".loader-actions");
  const detailEl = root.querySelector(".loader-detail");
  let instance = null;

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
  async function detail(state, info) {
    const size = model.sizeMB ? `~${model.sizeMB} MB` : "size varies";
    const est = await storageEstimate();
    const used = est ? ` · ${(est.usage / 1e6) | 0} MB cached of ${(est.quota / 1e6) | 0} MB` : "";
    const bits = [`${model.modelId}`, `${model.runtime || "transformers.js"}`, size];
    if (state === "update") {
      bits.push(
        `cached ${info?.cachedRevision?.slice(0, 7)} → new ${info?.remoteRevision?.slice(0, 7)}`,
      );
    }
    detailEl.textContent = bits.join(" · ") + used;
  }

  const onProgress = (p) => {
    if (p.status === "progress" && typeof p.progress === "number") {
      progEl.hidden = false;
      progEl.value = p.progress;
      statusEl.textContent = `${STATE_TEXT.downloading} ${p.progress.toFixed(0)}%`;
    }
  };

  async function runLoad(kind) {
    clearActions();
    setState(kind === "download" ? "downloading" : "initialising");
    progEl.hidden = kind !== "download";
    try {
      instance = await load(onProgress);
      progEl.hidden = true;
      await recordValidated({
        key,
        modelId: model.modelId,
        runtime: model.runtime,
        dtype: model.dtype,
      });
      setState("ready");
      addManageControls();
      onReady?.(instance);
    } catch (err) {
      progEl.hidden = true;
      setState("error", String(err?.message || err));
      onError?.(err);
      button("Retry", "secondary", () => runLoad(kind)); // clear recovery path — never fake output
    }
  }

  function addManageControls() {
    // Once ready/present, offer a real per-model cache-clear (accessible, confirmable).
    const b = document.createElement("button");
    b.textContent = "Clear cached model";
    b.className = "secondary";
    b.style.marginInlineStart = "0.5rem";
    b.addEventListener("click", async () => {
      const n = await clearModelCache(model.modelId, key);
      setState("download-required", `Cleared ${n} cached file(s).`);
      detail("download-required");
      button(
        `Download model (${model.sizeMB ? "~" + model.sizeMB + " MB" : ""})`,
        "",
        () => runLoad("download"),
      );
      actions.append(b);
    });
    actions.append(b);
  }

  async function start() {
    setState("checking");
    // 1) Capability gate for runtimes that truly require WebGPU (e.g. WebLLM) — honest, no fake.
    if (model.requiresWebGPU && !(await adapterAvailable())) {
      setState(
        "unsupported",
        "It needs WebGPU (no GPU adapter here). Enable it via chrome://gpu / a WebGPU-capable browser.",
      );
      await detail("unsupported");
      onError?.(new Error("needs-webgpu"));
      return;
    }
    // 2) Inspect on-device availability and drive the state machine.
    let info;
    try {
      info = await inspectModel({ key, modelId: model.modelId });
    } catch {
      info = { state: "absent" };
    }
    await detail(info.state, info);
    switch (info.state) {
      case "current":
      case "unverified":
        // Already local + current (or files present) — AUTO-initialise from cache, no button.
        await runLoad("init");
        break;
      case "update":
        // Cached current version works — auto-init it, and offer an optional Update to the newer rev.
        await runLoad("init");
        button("Update to newer version", "secondary", async () => {
          await clearModelCache(model.modelId, key);
          await runLoad("download");
        });
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

  start();
  return {
    get instance() {
      return instance;
    },
    refresh: start,
  };
}
