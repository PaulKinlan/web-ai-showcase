// Shared front-end helpers for the Florence-2 pages: the worker handshake, a real WebGPU probe
// (for the honest fallback), canvas overlay drawing for boxes/polygons, and the widget CSS.
// Inference lives in worker.js. Florence-2 is a TASK-PROMPTED model — one model, many vision tasks.

const WORKER_URL = "/web-ai-showcase/models/florence-2-large/worker.js";

// The tasks this UI exposes. `text` = whether the task needs an extra text input (e.g. a phrase to
// segment). `kind` tells the page how to render the parsed output.
export const FLORENCE_TASKS = {
  "<CAPTION>": { label: "Caption", kind: "text" },
  "<DETAILED_CAPTION>": { label: "Detailed caption", kind: "text" },
  "<MORE_DETAILED_CAPTION>": { label: "More detailed caption", kind: "text" },
  "<OD>": { label: "Object detection", kind: "boxes" },
  "<DENSE_REGION_CAPTION>": { label: "Dense region caption", kind: "boxes" },
  "<OCR>": { label: "OCR (read text)", kind: "text" },
  "<OCR_WITH_REGION>": { label: "OCR with regions", kind: "quads" },
  "<REFERRING_EXPRESSION_SEGMENTATION>": { label: "Referring segmentation", kind: "polygons", text: true },
};

export class FlorenceEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null;
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => this._rejectAll(new Error(e.message || "Worker failed to start")));
  }

  _rejectAll(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    if (this._active) {
      this._active.reject(err);
      this._active = null;
    }
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "probe-result":
        for (const w of this._probeWaiters) w.resolve(msg.gpu);
        this._probeWaiters = [];
        break;
      case "ready":
        this.ready = true;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "result":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve(msg);
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
          this._active = null;
        } else {
          this._rejectAll(new Error(msg.message));
        }
        break;
    }
  }

  probeGPU() {
    return new Promise((resolve) => {
      this._probeWaiters.push({ resolve });
      this.worker.postMessage({ type: "probe" });
    });
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve("webgpu");
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Run one task. Resolves with { task, prompt, raw, parsed, imageSize, ms }. */
  run(imageURL, task, { text = "", maxTokens = 512 } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, resolve, reject };
      this.worker.postMessage({ type: "run", id, image: imageURL, task, text, maxTokens });
    });
  }
}

export async function probeWebGPUMain() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter" };
    return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Deterministic, high-contrast colour per label index.
export function labelColor(i) {
  const hues = [12, 200, 145, 275, 45, 330, 95, 235, 175, 310];
  return `hsl(${hues[i % hues.length]} 85% 55%)`;
}

/**
 * Draw an image into a canvas and overlay Florence's parsed output. Returns the scale used so the
 * caller can size things. `parsed` is the object post_process_generation returned for this task.
 */
export function drawOverlay(canvas, img, task, parsed) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const maxW = Math.min(iw, 640);
  const scale = maxW / iw;
  canvas.width = Math.round(iw * scale);
  canvas.height = Math.round(ih * scale);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const data = parsed?.[task] ?? {};
  ctx.lineWidth = 2;
  ctx.font = "600 13px 'SF Mono', ui-monospace, monospace";
  ctx.textBaseline = "top";

  const drawLabel = (text, x, y, color) => {
    if (!text) return;
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = color;
    const ly = Math.max(0, y - 17);
    ctx.fillRect(x, ly, w, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 4, ly + 1);
  };

  const kind = FLORENCE_TASKS[task]?.kind;
  if (kind === "boxes" && Array.isArray(data.bboxes)) {
    data.bboxes.forEach((b, i) => {
      const color = labelColor(i);
      const [x1, y1, x2, y2] = b.map((v, j) => v * scale);
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      drawLabel(data.labels?.[i] ?? "", x1, y1, color);
    });
    return { count: data.bboxes.length, scale };
  }
  if (kind === "quads" && Array.isArray(data.quad_boxes)) {
    data.quad_boxes.forEach((q, i) => {
      const color = labelColor(i);
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (let p = 0; p < q.length; p += 2) {
        const x = q[p] * scale, y = q[p + 1] * scale;
        p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
      drawLabel(data.labels?.[i] ?? "", q[0] * scale, q[1] * scale, color);
    });
    return { count: data.quad_boxes.length, scale };
  }
  if (kind === "polygons" && Array.isArray(data.polygons)) {
    data.polygons.forEach((group, i) => {
      const color = labelColor(i);
      ctx.fillStyle = color.replace("55%)", "55% / 0.35)");
      ctx.strokeStyle = color;
      for (const poly of group) {
        ctx.beginPath();
        for (let p = 0; p < poly.length; p += 2) {
          const x = poly[p] * scale, y = poly[p + 1] * scale;
          p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    });
    return { count: data.polygons.length, scale };
  }
  return { count: 0, scale };
}

export function webgpuFallbackHTML(probe) {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (common in headless Chrome, VMs, or with the GPU blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>Florence-2 needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""}
    Florence-2 is a 230M-parameter vision model in fp16; it needs a GPU to run at a usable speed, so this
    page won't fake a result. To try it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a GPU (desktop or a recent laptop/phone).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>fp16 needs the <code>shader-f16</code> feature (present on most modern GPUs).</li>
    </ul>`;
}

export const FLOR_CSS = `
.vlm-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.vlm-img-col { flex:1 1 300px; max-inline-size:660px; }
.vlm-out-col { flex:1 1 300px; }
.preview-img { max-inline-size:100%; max-block-size:340px; border-radius:8px; display:block; }
.canvas-wrap { position:relative; }
.canvas-wrap canvas { max-inline-size:100%; height:auto; border-radius:8px; border:1px solid var(--border); display:block; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px;
  border:2px solid transparent; cursor:pointer; padding:0; }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.task-grid { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.task-btn { font:inherit; font-size:.78rem; padding:.25rem .6rem; border-radius:8px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.task-btn[aria-pressed=true] { border-color:var(--accent); background:var(--accent); color:var(--accent-ink); }
.answer { min-block-size:3rem; padding:.7rem; border:1px solid var(--border); border-radius:8px;
  background:var(--bg-raised); white-space:pre-wrap; line-height:1.6; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; }
.legend { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; font-size:.78rem; }
.legend .chip { display:inline-flex; align-items:center; gap:.3rem; padding:.1rem .45rem; border-radius:999px;
  border:1px solid var(--border); font-family:var(--font-mono); }
.legend .sw { inline-size:.7rem; block-size:.7rem; border-radius:2px; display:inline-block; }
`;
