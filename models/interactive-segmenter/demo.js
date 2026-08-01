import { createModelLoader } from "/web-ai-showcase/lib/model-loader.js";
import { SupersededError, WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite";
export const MODEL_CACHE_ID = "interactive_segmenter/magic_touch/float32/1";
export const MODEL_REVISION = "1683146867404767";
export const MODEL_SHA256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25";
export const MODEL_SIZE_MB = 5.94;
const CLASSIFIER_REVISION = "a0e5dde3e1b3d9e49894dcaeeb1e046ed01edfc8";

const MODES = {
  overview: { view: "overlay", sample: "portrait", title: "Point at the object you want" },
  basics: { view: "mask", sample: "dog", title: "One point in, one foreground mask out" },
  practical: { view: "cutout", sample: "sneaker", title: "Cut out a product without uploading it" },
  wild: { view: "spotlight", sample: "bicycle", title: "Keep one object in colour" },
  multi: { view: "cutout", sample: "cat", title: "Segment it, then let MobileViT name it" },
};

const SAMPLES = [
  {
    id: "portrait",
    src: "/web-ai-showcase/media/assets/portrait-person.jpg",
    alt: "A portrait of a person",
    point: { x: 0.5, y: 0.5 },
  },
  {
    id: "dog",
    src: "/web-ai-showcase/media/assets/dog-outdoor.jpg",
    alt: "A dog outdoors",
    point: { x: 0.5, y: 0.53 },
  },
  {
    id: "cat",
    src: "/web-ai-showcase/media/assets/obj-cat.jpg",
    alt: "A cat",
    point: { x: 0.5, y: 0.48 },
  },
  {
    id: "sneaker",
    src: "/web-ai-showcase/media/assets/fashion-sneaker.jpg",
    alt: "A sneaker on a plain background",
    point: { x: 0.5, y: 0.52 },
  },
  {
    id: "bicycle",
    src: "/web-ai-showcase/media/assets/bicycle-object.jpg",
    alt: "A bicycle",
    point: { x: 0.5, y: 0.53 },
  },
];

class InteractiveWorker {
  constructor(client, info) {
    this.client = client;
    Object.assign(this, info);
  }
  static async create(onProgress) {
    const client = new WorkerClient({
      url: new URL("./worker.js", import.meta.url),
      name: "magic-touch-interactive-segmenter",
      maxInFlight: 1,
      maxQueue: 1,
      module: false,
    });
    await client.ready;
    const { result } = await client.request("configure", {}, { onProgress });
    return new InteractiveWorker(client, result);
  }
  async segment(source, point, options) {
    const bitmap = await createImageBitmap(source);
    try {
      const { result } = await this.client.request(
        "segment",
        { bitmap, point, ...options },
        { transfer: [bitmap], channel: "mask" },
      );
      return result;
    } catch (error) {
      if (error instanceof SupersededError || error?.name === "AbortError") return null;
      throw error;
    }
  }
  async compose(options) {
    const { result } = await this.client.request("compose", options, { channel: "compose" });
    return result;
  }
  terminate() {
    return this.client.terminate();
  }
}

class ClassifierWorker {
  constructor(client, info) {
    this.client = client;
    Object.assign(this, info);
  }
  static async create(onProgress) {
    const client = new WorkerClient({
      url: new URL("./classifier-worker.js", import.meta.url),
      name: "magic-touch-mobilevit",
      maxInFlight: 1,
      maxQueue: 1,
    });
    await client.ready;
    const { result } = await client.request("configure", {}, { onProgress });
    return new ClassifierWorker(client, result);
  }
  async classify(image) {
    const { result } = await this.client.request("classify", { image, topK: 5 }, {
      channel: "classify",
    });
    return result;
  }
  terminate() {
    return this.client.terminate();
  }
}

function safeLabel(value) {
  return String(value).split(",")[0].replace(/[<>]/g, "");
}

function decodeImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The selected image could not be decoded"));
    image.src = src;
  });
}

function pointFromEvent(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function marker(context, point, width, height) {
  const x = point.x * width;
  const y = point.y * height;
  const radius = Math.max(8, Math.min(width, height) * 0.018);
  context.save();
  context.lineWidth = Math.max(3, radius * 0.28);
  context.strokeStyle = "white";
  context.fillStyle = "#3949ab";
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.beginPath();
  context.moveTo(x - radius * 1.7, y);
  context.lineTo(x + radius * 1.7, y);
  context.moveTo(x, y - radius * 1.7);
  context.lineTo(x, y + radius * 1.7);
  context.stroke();
  context.restore();
}

export const DEMO_CSS = `
.interactive-app { container-type:inline-size; }
.interactive-app code { overflow-wrap:anywhere; }
.sample-list { display:flex; flex-wrap:wrap; gap:.55rem; margin-block:.75rem; }
.sample-button { padding:.15rem; background:var(--bg-raised); color:var(--color); border:2px solid transparent; border-radius:.6rem; }
.sample-button[aria-pressed="true"] { border-color:var(--accent); }
.sample-button img { display:block; inline-size:5.5rem; block-size:4rem; object-fit:cover; border-radius:.35rem; }
.interactive-toolbar { display:flex; flex-wrap:wrap; align-items:end; gap:.7rem; margin-block:.75rem; }
.interactive-toolbar label { display:grid; gap:.2rem; min-inline-size:10rem; color:var(--muted); font-size:.88rem; }
.interactive-toolbar input[type="range"] { inline-size:100%; accent-color:var(--accent); }
.interactive-app input[type="file"] { max-inline-size:100%; }
.view-buttons { display:flex; flex-wrap:wrap; gap:.35rem; }
.view-buttons button[aria-pressed="true"] { background:var(--accent); color:var(--accent-ink); }
.prompt-stage { position:relative; border:1px solid var(--border); border-radius:.75rem; overflow:hidden; background:var(--bg-secondary); }
.prompt-stage canvas { display:block; inline-size:100%; block-size:auto; max-block-size:70dvh; object-fit:contain; cursor:crosshair; touch-action:pan-y; }
.prompt-stage canvas:focus-visible { outline:3px solid var(--accent); outline-offset:-4px; }
.prompt-presets { display:grid; grid-template-columns:repeat(3,minmax(2.75rem,1fr)); gap:.35rem; max-inline-size:18rem; margin-block:.6rem; }
.prompt-presets button { min-block-size:2.75rem; }
.result-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,11rem),1fr)); gap:.65rem; margin-block:.75rem; }
.metric { padding:.65rem; border:1px solid var(--border); border-radius:.55rem; background:var(--bg-raised); }
.metric b { display:block; font-family:var(--font-mono); overflow-wrap:anywhere; }
.inside-table { inline-size:100%; border-collapse:collapse; table-layout:fixed; }
.inside-table :is(th,td) { padding:.45rem; border-block-end:1px solid var(--border); text-align:start; vertical-align:top; overflow-wrap:anywhere; }
.inside-table th { inline-size:min(34%,10rem); }
.inside-table th { color:var(--muted); font-weight:600; }
.model-loader .loader-detail { overflow-wrap:anywhere; }
.classification-list { display:grid; gap:.45rem; margin-block:.75rem; }
.classification-row { display:grid; grid-template-columns:minmax(7rem,1fr) minmax(5rem,2fr) 4rem; gap:.5rem; align-items:center; }
.classification-bar { block-size:.65rem; border-radius:999px; background:var(--bg-secondary); overflow:hidden; }
.classification-bar span { display:block; block-size:100%; background:var(--accent); }
@container (max-width:32rem) { .interactive-toolbar > * { inline-size:100%; } .view-buttons button { flex:1; } .classification-row { grid-template-columns:1fr 4rem; } .classification-bar { grid-column:1 / -1; grid-row:2; } }
@media (pointer:coarse) { .interactive-app :is(button,input) { min-block-size:2.75rem; } .sample-button img { inline-size:5rem; block-size:4rem; } }
@media (forced-colors:active) { .sample-button[aria-pressed="true"] { outline:3px solid Highlight; } }
`;

export async function mountInteractiveDemo(mount, requestedMode) {
  const mode = MODES[requestedMode] ? requestedMode : "overview";
  const config = MODES[mode];
  const style = document.createElement("style");
  style.textContent = DEMO_CSS;
  document.head.append(style);
  mount.classList.add("interactive-app");
  mount.innerHTML = `
    <section class="panel" aria-labelledby="run-heading">
      <h2 id="run-heading">${config.title}</h2>
      <p class="muted">Choose a sample or your own image, then click the object. Keyboard users can move the target with arrow keys and press Enter, or use the nine preset point buttons.</p>
      <div id="model-loader"></div>
      <p class="status" id="status" role="status" aria-live="polite">Choose Download to keep the 5.94 MB pinned model on this device.</p>
      <div class="sample-list" id="samples" aria-label="License-verified sample images"></div>
      <label for="upload">Use your own image</label>
      <input id="upload" type="file" accept="image/*" />
      <div class="interactive-toolbar">
        <div>
          <span class="muted">View</span>
          <div class="view-buttons" role="group" aria-label="Mask view">
            <button type="button" data-view="overlay" aria-pressed="false">Overlay</button>
            <button type="button" data-view="mask" aria-pressed="false">Confidence</button>
            <button type="button" data-view="cutout" aria-pressed="false">Cut-out</button>
            <button type="button" data-view="spotlight" aria-pressed="false">Spotlight</button>
          </div>
        </div>
        <label for="threshold">Foreground threshold <output id="threshold-value" for="threshold">50%</output>
          <input id="threshold" type="range" min="20" max="80" step="5" value="50" />
        </label>
        <label for="opacity">Overlay strength <output id="opacity-value" for="opacity">60%</output>
          <input id="opacity" type="range" min="20" max="90" step="5" value="60" />
        </label>
        <button id="run-center" type="button" disabled>Segment at target</button>
        ${
    mode === "practical"
      ? '<button id="download-cutout" type="button" disabled>Download transparent PNG</button>'
      : ""
  }
      </div>
      <div class="prompt-stage">
        <canvas id="stage" tabindex="0" role="button" aria-describedby="point-help" aria-label="Image prompt: choose an object point"></canvas>
      </div>
      <p id="point-help" class="muted">Target: <output id="point-value">50% across, 50% down</output>. Arrow keys move it by 2%; Enter or Space segments.</p>
      <div class="prompt-presets" id="presets" role="group" aria-label="Preset target points"></div>
      <div class="result-grid" id="metrics" hidden>
        <div class="metric">Foreground coverage<b id="coverage">–</b></div>
        <div class="metric">Mean confidence<b id="mean-confidence">–</b></div>
        <div class="metric">Point confidence<b id="point-confidence">–</b></div>
        <div class="metric">Inference / composite<b id="latency">–</b></div>
      </div>
    </section>
    <section class="panel" aria-labelledby="inside-heading">
      <h2 id="inside-heading">See inside the real output</h2>
      <p class="muted">MagicTouch's raw tensor has background and foreground channels. MediaPipe exposes the selected object's normalized foreground confidence mask plus a binary category mask. These values come from that foreground mask and the selected pixels—not a canned result.</p>
      <table class="inside-table">
        <caption>Current tensor and runtime diagnostics</caption>
        <tbody>
          <tr><th scope="row">Prompt point</th><td id="inside-point">Not run</td></tr>
          <tr><th scope="row">Output tensor</th><td id="inside-shape">Not run</td></tr>
          <tr><th scope="row">Selected pixels</th><td id="inside-pixels">Not run</td></tr>
          <tr><th scope="row">Mask bounds</th><td id="inside-bounds">Not run</td></tr>
          <tr><th scope="row">Delegate</th><td id="inside-delegate">Not loaded</td></tr>
          <tr><th scope="row">Pinned artifact</th><td>generation 1 · SHA-256 <code>${MODEL_SHA256}</code></td></tr>
        </tbody>
      </table>
    </section>
    ${
    mode === "multi"
      ? `<section class="panel" aria-labelledby="classify-heading"><h2 id="classify-heading">Second stage — classify the cut-out</h2><p class="muted">MobileViT q8 runs in its own worker at pinned revision <code>${CLASSIFIER_REVISION}</code>. It sees only the transparent cut-out produced above.</p><div id="classifier-loader"></div><div id="classification" class="classification-list"><p class="muted">Segment an object to classify it.</p></div></section>`
      : ""
  }
  `;

  const $ = (id) => mount.querySelector(`#${id}`);
  let task = null;
  let classifier = null;
  let image = null;
  let point = { x: 0.5, y: 0.5 };
  let view = config.view;
  let result = null;
  let activeSample = null;
  let objectUrl = null;
  let busy = false;

  function options() {
    return {
      mode: view,
      threshold: Number($("threshold").value) / 100,
      opacity: Number($("opacity").value) / 100,
    };
  }

  function setStatus(text, kind) {
    const status = $("status");
    status.textContent = text;
    status.classList.remove("ok", "err");
    if (kind) status.classList.add(kind);
  }

  function updatePointText() {
    $("point-value").textContent = `${Math.round(point.x * 100)}% across, ${
      Math.round(point.y * 100)
    }% down`;
  }

  function drawSource() {
    if (!image) return;
    const canvas = $("stage");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    marker(context, point, canvas.width, canvas.height);
  }

  function drawResult(next) {
    result = next;
    const canvas = $("stage");
    canvas.width = next.width;
    canvas.height = next.height;
    const rgba = new Uint8ClampedArray(next.rgba);
    canvas.getContext("2d").putImageData(new ImageData(rgba, next.width, next.height), 0, 0);
    marker(canvas.getContext("2d"), point, next.width, next.height);
    $("metrics").hidden = false;
    $("coverage").textContent = formatPercent(next.coverage);
    $("mean-confidence").textContent = formatPercent(next.meanSelectedConfidence);
    $("point-confidence").textContent = formatPercent(next.targetConfidence);
    $("latency").textContent = `${next.inferenceMs ?? "–"} / ${next.composeMs} ms`;
    $("inside-point").textContent = `x ${point.x.toFixed(3)}, y ${point.y.toFixed(3)}`;
    $("inside-shape").textContent = next.outputShape || `${next.height}×${next.width}×2`;
    $("inside-pixels").textContent = `${next.selectedPixels.toLocaleString()} (${
      formatPercent(next.coverage)
    })`;
    $("inside-bounds").textContent = next.bbox
      ? `${next.bbox.minX},${next.bbox.minY} → ${next.bbox.maxX},${next.bbox.maxY}`
      : "No pixels above threshold";
    if ($("download-cutout")) {
      $("download-cutout").disabled = view !== "cutout" || !next.selectedPixels;
    }
  }

  async function classifyCutout() {
    if (mode !== "multi" || !classifier || !result || view !== "cutout") return;
    const output = $("classification");
    output.innerHTML = '<p class="muted">MobileViT is classifying the real cut-out…</p>';
    const classified = await classifier.classify($("stage").toDataURL("image/png"));
    output.replaceChildren();
    for (const item of classified.labels) {
      const row = document.createElement("div");
      row.className = "classification-row";
      const label = document.createElement("span");
      label.textContent = safeLabel(item.label);
      const bar = document.createElement("span");
      bar.className = "classification-bar";
      const fill = document.createElement("span");
      fill.style.inlineSize = formatPercent(item.score);
      bar.append(fill);
      const score = document.createElement("span");
      score.textContent = formatPercent(item.score);
      row.append(label, bar, score);
      output.append(row);
    }
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent =
      `MobileViT latency ${classified.latencyMs} ms · input was the MagicTouch cut-out.`;
    output.append(note);
  }

  async function run() {
    if (!task || !image || busy) return;
    busy = true;
    $("run-center").disabled = true;
    setStatus("Running MagicTouch in the worker…");
    try {
      const next = await task.segment(image, point, options());
      if (!next) return;
      drawResult(next);
      setStatus(
        `Selected ${formatPercent(next.coverage)} of the image at ${
          formatPercent(next.targetConfidence)
        } point confidence.`,
        "ok",
      );
      await classifyCutout();
    } catch (error) {
      setStatus(`Segmentation failed: ${error.message}`, "err");
    } finally {
      busy = false;
      $("run-center").disabled = !task || !image;
    }
  }

  async function recompose() {
    if (!task || !result || busy) return;
    try {
      const next = await task.compose(options());
      next.inferenceMs = result.inferenceMs;
      next.outputShape = result.outputShape;
      drawResult(next);
      if (mode === "multi") await classifyCutout();
    } catch (error) {
      setStatus(`Mask rendering failed: ${error.message}`, "err");
    }
  }

  async function useImage(src, sample) {
    image = await decodeImage(src);
    activeSample = sample || null;
    if (sample) point = { ...sample.point };
    result = null;
    $("metrics").hidden = true;
    $("run-center").disabled = !task;
    for (const button of $("samples").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.sample === sample?.id));
    }
    updatePointText();
    drawSource();
    setStatus(
      task
        ? "Model ready — click the object or use Segment at target."
        : "Image ready — download the model when you choose.",
      task ? "ok" : undefined,
    );
  }

  for (const sample of SAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-button";
    button.dataset.sample = sample.id;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Use sample: ${sample.alt}`);
    const thumb = document.createElement("img");
    thumb.src = sample.src;
    thumb.alt = "";
    thumb.width = 88;
    thumb.height = 64;
    thumb.loading = sample.id === config.sample ? "eager" : "lazy";
    button.append(thumb);
    button.addEventListener("click", () => useImage(sample.src, sample));
    $("samples").append(button);
  }

  const presetLabels = [
    "top left",
    "top centre",
    "top right",
    "middle left",
    "centre",
    "middle right",
    "bottom left",
    "bottom centre",
    "bottom right",
  ];
  let presetIndex = 0;
  for (const y of [0.25, 0.5, 0.75]) {
    for (const x of [0.25, 0.5, 0.75]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = presetLabels[presetIndex++];
      button.addEventListener("click", () => {
        point = { x, y };
        updatePointText();
        drawSource();
        void run();
      });
      $("presets").append(button);
    }
  }

  for (const button of mount.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
    button.addEventListener("click", () => {
      view = button.dataset.view;
      for (const peer of mount.querySelectorAll("[data-view]")) {
        peer.setAttribute("aria-pressed", String(peer === button));
      }
      void recompose();
    });
  }

  $("threshold").addEventListener("input", () => {
    $("threshold-value").value = `${$("threshold").value}%`;
  });
  $("threshold").addEventListener("change", recompose);
  $("opacity").addEventListener("input", () => {
    $("opacity-value").value = `${$("opacity").value}%`;
  });
  $("opacity").addEventListener("change", recompose);
  $("run-center").addEventListener("click", run);

  const stage = $("stage");
  stage.addEventListener("pointerup", (event) => {
    point = pointFromEvent(stage, event);
    updatePointText();
    drawSource();
    void run();
  });
  stage.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    let moved = true;
    if (event.key === "ArrowLeft") point.x = Math.max(0, point.x - step);
    else if (event.key === "ArrowRight") point.x = Math.min(1, point.x + step);
    else if (event.key === "ArrowUp") point.y = Math.max(0, point.y - step);
    else if (event.key === "ArrowDown") point.y = Math.min(1, point.y + step);
    else moved = false;
    if (moved) {
      event.preventDefault();
      updatePointText();
      drawSource();
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      void run();
    }
  });

  $("upload").addEventListener("change", async () => {
    const file = $("upload").files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus("Choose an image file.", "err");
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    await useImage(objectUrl, null);
  });

  if ($("download-cutout")) {
    $("download-cutout").addEventListener("click", async () => {
      if (view !== "cutout") return;
      const blob = await new Promise((resolve) => stage.toBlob(resolve, "image/png"));
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "magic-touch-cutout.png";
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    });
  }

  createModelLoader({
    mount: $("model-loader"),
    model: {
      modelId: "interactive_segmenter/magic_touch/float32/1",
      runtime: "mediapipe",
      dtype: "float32",
      sizeMB: MODEL_SIZE_MB,
      revision: MODEL_REVISION,
      requiresWebGPU: false,
    },
    load: (onProgress) => InteractiveWorker.create(onProgress),
    onReady: (instance) => {
      task = instance;
      $("inside-delegate").textContent = `${instance.delegate} · WebGPU adapter ${
        instance.webgpu ? "available" : "not available; CPU fallback active"
      }`;
      $("run-center").disabled = !image;
      setStatus("MagicTouch ready — click an object or use Segment at target.", "ok");
    },
    onError: (error) =>
      setStatus(
        `Model load failed: ${error.message}. Retry will fetch and verify a clean copy.`,
        "err",
      ),
    dispose: (instance) => instance?.terminate?.(),
    onDispose: () => {
      task = null;
      $("run-center").disabled = true;
    },
  });

  if (mode === "multi") {
    createModelLoader({
      mount: $("classifier-loader"),
      model: {
        modelId: "Xenova/mobilevit-small",
        runtime: "transformers.js",
        dtype: "q8",
        sizeMB: 6.1,
        revision: CLASSIFIER_REVISION,
        requiresWebGPU: false,
      },
      load: (onProgress) => ClassifierWorker.create(onProgress),
      onReady: (instance) => {
        classifier = instance;
        if (result) void classifyCutout();
      },
      dispose: (instance) => instance?.terminate?.(),
      onDispose: () => {
        classifier = null;
      },
    });
  }

  const initial = SAMPLES.find((sample) => sample.id === config.sample) || SAMPLES[0];
  await useImage(initial.src, initial);

  globalThis.addEventListener("pagehide", () => {
    task?.terminate?.();
    classifier?.terminate?.();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }, { once: true });
}
