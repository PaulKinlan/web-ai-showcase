import { createModelLoader } from "/web-ai-showcase/lib/model-loader.js";

export const FASHION_MODEL = {
  modelId: "patrickjohncyh/fashion-clip",
  runtime: "transformers.js",
  dtype: "fp32",
  revision: "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
  sizeMB: 578,
  requiresWebGPU: false,
};
export const GENERAL_MODEL = {
  modelId: "Xenova/clip-vit-base-patch16",
  runtime: "transformers.js",
  dtype: "q8",
  revision: "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
  onnxBytes: 152040303,
  onnxSha256: "cf5b03d7c03cd78498b0d59a905552b549ae91af4e99ffb985103aa9424d2272",
  sizeMB: 145,
  requiresWebGPU: false,
};

export class FashionEngine {
  constructor(kind = "fashion") {
    this.kind = kind;
    this.worker = new Worker("/web-ai-showcase/models/fashion-clip/worker.js", { type: "module" });
    this.pending = new Map();
    this.waiters = [];
    this.sequence = 0;
    this.ready = false;
    this.onProgress = null;
    this.worker.addEventListener("message", ({ data }) => this.#message(data));
    this.worker.addEventListener(
      "error",
      (event) => this.#fail(new Error(event.message || "Worker failed")),
    );
  }
  #message(message) {
    if (message.type === "progress") this.onProgress?.(message.progress);
    if (message.type === "ready" && message.kind === this.kind) {
      this.ready = true;
      for (const waiter of this.waiters.splice(0)) waiter.resolve(this);
    }
    if (message.type === "result") {
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        pending.resolve(message);
      }
    }
    if (message.type === "error") {
      const error = new Error(message.message);
      if (message.id && this.pending.has(message.id)) {
        this.pending.get(message.id).reject(error);
        this.pending.delete(message.id);
      } else this.#fail(error);
    }
  }
  #fail(error) {
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
  load(onProgress) {
    this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this);
    const promise = new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
    this.worker.postMessage({ type: "load", kind: this.kind });
    return promise;
  }
  classify(image, labels) {
    const id = ++this.sequence;
    const promise = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.worker.postMessage({ type: "classify", id, kind: this.kind, image, labels });
    return promise;
  }
  dispose() {
    this.#fail(new Error("Model released from memory"));
    this.ready = false;
    this.worker.terminate();
  }
}

export function parseLabels(value) {
  return [...new Set(value.split(/[\n,]/).map((label) => label.trim()).filter(Boolean))];
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export function renderRanking(container, result, heading = "Ranking") {
  container.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = heading;
  container.append(title);
  const order = result.labels.map((_, index) => index)
    .sort((a, b) => result.probabilities[b] - result.probabilities[a]);
  for (const index of order) {
    const row = document.createElement("div");
    row.className = "fashion-result";
    const score = result.probabilities[index] * 100;
    const header = document.createElement("div");
    header.className = "fashion-result-head";
    const label = document.createElement("strong");
    label.textContent = result.labels[index];
    const value = document.createElement("span");
    value.textContent = `${score.toFixed(2)}%`;
    header.append(label, value);
    const meter = document.createElement("div");
    meter.className = "fashion-meter";
    meter.setAttribute("role", "meter");
    meter.setAttribute("aria-label", `${result.labels[index]}: ${score.toFixed(2)} percent`);
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", score.toFixed(2));
    const fill = document.createElement("span");
    fill.style.setProperty("--score", `${score}%`);
    meter.append(fill);
    row.append(header, meter);
    container.append(row);
  }
}

export function mountLoader({ mount, kind, onReady, onDispose }) {
  let engine;
  const model = kind === "general" ? GENERAL_MODEL : FASHION_MODEL;
  createModelLoader({
    mount,
    model,
    load: async (onProgress) => {
      engine = new FashionEngine(kind);
      await engine.load(onProgress);
      return engine;
    },
    onReady: () => onReady(engine),
    dispose: (instance) => instance.dispose(),
    onDispose: () => {
      engine = null;
      onDispose?.();
    },
    onError: (error) => {
      const event = new CustomEvent("fashion-loader-error", { detail: error });
      mount.dispatchEvent(event);
    },
  });
}
