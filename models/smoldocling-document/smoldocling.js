// Front-end helpers for the SmolDocling pages: the worker handshake with single-image + DocTags
// streaming, a real WebGPU probe (for the honest fallback), image upload/sample helpers, a small
// DocTags → structure parser (for the "See inside" view), and the widget CSS. All inference lives in
// worker.js (off the main thread, Transformers.js v4 pinned there).
//
// DISTINCT from the Nougat (markdown), Donut (doc-VQA) and TrOCR (line OCR) demos: SmolDocling emits
// DocTags — structured layout markup with per-element bounding boxes, tables (<otsl>) and formulas
// (<formula> as LaTeX). This engine streams those tags and parses them into a legible structure.

const WORKER_URL = "/web-ai-showcase/models/smoldocling-document/worker.js";

export class SmolDoclingEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null; // { id, onToken, onPrompt, onShape, resolve, reject }
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      this._rejectAll(new Error(e.message || "Worker failed to start"));
    });
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
      case "prompt":
        if (this._active && this._active.id === msg.id) {
          this._active.onPrompt?.(msg.template, msg.prompt);
        }
        break;
      case "shape":
        if (this._active && this._active.id === msg.id) {
          this._active.onShape?.({ promptTokens: msg.promptTokens, pixelValues: msg.pixelValues });
        }
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token, msg.t);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({ ms: msg.ms, tokens: msg.tokens, promptTokens: msg.promptTokens });
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

  /**
   * Stream a DocTags conversion of ONE document image. onToken(token, tMs) per token; onPrompt(template)
   * once; onShape({promptTokens, pixelValues}) once. Returns { ms, tokens, promptTokens }.
   */
  convert(imageURL, prompt, { maxTokens = 512, onToken, onPrompt, onShape } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, onShape, resolve, reject };
      this.worker.postMessage({ type: "run", id, image: imageURL, prompt, maxTokens });
    });
  }
}

/** Probe WebGPU on the main thread too (for instant UI gating before we ever load). */
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

/** Fetch a same-origin sample and return a self-contained data URL for the worker. */
export async function urlToDataURL(src) {
  const blob = await (await fetch(src)).blob();
  return fileToDataURL(new File([blob], "sample", { type: blob.type }));
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/**
 * Parse a DocTags string into a legible list of elements: { type, text, box:[x1,y1,x2,y2]|null }.
 * DocTags look like: <section_header_level_1><loc_43><loc_32><loc_231><loc_48>Regional Sales…</section_header_level_1>
 * Tables use <otsl>…</otsl> with <fcel>/<ched> cells; we keep those inline as a "table" element.
 */
export function parseDocTags(doc) {
  const elements = [];
  const clean = doc.replace(/<doctag>/g, "").replace(/<\/doctag>/g, "");
  // Match a block: <tag>(<loc_n>)*content</tag>  — content may itself contain OTSL cell tags.
  const re = /<([a-z0-9_]+)>((?:<loc_\d+>)*)([\s\S]*?)<\/\1>/g;
  let m;
  while ((m = re.exec(clean)) !== null) {
    const type = m[1];
    const locs = (m[2].match(/<loc_(\d+)>/g) || []).map((s) => parseInt(s.replace(/\D/g, ""), 10));
    const box = locs.length >= 4 ? locs.slice(0, 4) : null;
    let text = m[3];
    if (type === "otsl") {
      // Render OTSL cells into rows for readability.
      const rows = text.split(/<nl>/).map((r) =>
        (r.match(/<(?:fcel|ched|rhed|ecel)>([^<]*)/g) || []).map((c) =>
          c.replace(/<[^>]+>/g, "").trim()
        )
      ).filter((r) => r.length);
      elements.push({ type: "table", box, rows });
    } else {
      text = text.replace(/<[^>]+>/g, "").trim();
      elements.push({ type, text, box });
    }
  }
  return elements;
}

/** The labelled needs-WebGPU state — never faked output, always the real reason + how to enable. */
export function webgpuFallbackHTML(probe) {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (common in headless, VMs, or with the GPU blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>SmolDocling needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    A 256M-parameter document VLM over a split page image still needs a GPU to run at a usable speed, so
    this page won't fake a result. To try it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a GPU (desktop or a recent laptop/phone).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
    </ul>`;
}

export const DOCLING_CSS = `
.doc-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.doc-in-col { flex:1 1 300px; min-inline-size:0; max-inline-size:460px; }
.doc-out-col { flex:1 1 320px; min-inline-size:0; }
.doc-preview { max-inline-size:100%; max-block-size:420px; border-radius:8px; border:1px solid var(--border);
  background:var(--bg-raised); display:block; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:72px; block-size:92px; object-fit:cover; object-position:top; border-radius:6px;
  border:2px solid transparent; cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.controls-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.chip { font:inherit; font-size:.78rem; padding:.3rem .6rem; border-radius:999px; min-block-size:32px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.field-label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
/* Structured element list */
.struct-list { display:flex; flex-direction:column; gap:.4rem; margin:.4rem 0; }
.struct-el { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.struct-el .etype { font-family:var(--font-mono); font-size:.68rem; text-transform:uppercase; letter-spacing:.04em;
  color:var(--accent); font-weight:600; }
.struct-el .ebox { font-family:var(--font-mono); font-size:.64rem; color:var(--muted); margin-inline-start:.5rem; }
.struct-el .etext { margin-top:.2rem; line-height:1.45; overflow-wrap:anywhere; }
.struct-table { border-collapse:collapse; margin-top:.3rem; font-size:.78rem; inline-size:100%; }
.struct-table td, .struct-table th { border:1px solid var(--border); padding:.2rem .4rem; text-align:left; }
.struct-table tr:first-child td { background:var(--bg-secondary); font-weight:600; }
/* Raw DocTags stream */
.doctags-raw { font-family:var(--font-mono); font-size:.74rem; white-space:pre-wrap; word-break:break-word;
  line-height:1.5; background:var(--bg-raised); border:1px solid var(--border); border-radius:8px; padding:.7rem;
  max-block-size:22rem; overflow:auto; }
.doctags-raw .dt-tag { color:var(--accent); font-weight:600; }
.doctags-raw .dt-loc { color:var(--muted); }
.doctags-raw .caret { display:inline-block; inline-size:.5rem; background:var(--accent);
  animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .doctags-raw .caret { animation:none; } }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
`;

/** Convert parsed DocTags elements into clean Markdown (headings, paragraphs, tables, formulas). */
export function docTagsToMarkdown(elements) {
  const lines = [];
  for (const el of elements) {
    if (el.type === "table" && el.rows) {
      if (!el.rows.length) continue;
      const [head, ...body] = el.rows;
      lines.push("| " + head.join(" | ") + " |");
      lines.push("| " + head.map(() => "---").join(" | ") + " |");
      for (const r of body) lines.push("| " + r.join(" | ") + " |");
      lines.push("");
    } else if (/section_header|title|page_header/.test(el.type)) {
      const level = /title/.test(el.type) ? "# " : "## ";
      lines.push(level + (el.text || ""));
      lines.push("");
    } else if (el.type === "formula") {
      lines.push("$$" + (el.text || "") + "$$");
      lines.push("");
    } else if (el.type === "caption") {
      lines.push("*" + (el.text || "") + "*");
      lines.push("");
    } else if (/list_item/.test(el.type)) {
      lines.push("- " + (el.text || ""));
    } else if (el.text) {
      lines.push(el.text);
      lines.push("");
    }
  }
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

/** Colourise a DocTags string for the raw stream view (tags in accent, loc coords muted). */
export function highlightDocTags(doc) {
  return escapeHTML(doc)
    .replace(/&lt;loc_(\d+)&gt;/g, '<span class="dt-loc">&lt;loc_$1&gt;</span>')
    .replace(/&lt;(\/?[a-z0-9_]+)&gt;/g, '<span class="dt-tag">&lt;$1&gt;</span>');
}
