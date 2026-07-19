// Front-end helpers shared by every Face-Embedding page. Keeps pages thin: owns the worker handshake
// (transferring ImageBitmaps so nothing is copied), turns files / samples / webcam frames into
// ImageBitmaps, computes cosine similarity between two L2-normalized embeddings, formats the
// same/different SIGNAL, and renders the embedding + similarity visualizations. ALL inference lives in
// worker.js (off the main thread, raw ONNX Runtime Web). Privacy by construction: images, crops, and
// embeddings never leave the device. This is face SIMILARITY, not identity, lookup, or surveillance.

const WORKER_URL = "/web-ai-showcase/models/face-embedding/worker.js";

// Cosine-similarity decision band for FaceNet (VGGFace2, L2-normalized embeddings). These are SIGNAL
// thresholds for a demo, not a security policy: same-person pairs typically land ≥0.5, different-person
// pairs ≤0.25, with a genuinely-uncertain band between. A real system would calibrate on its own data.
export const SAME_THRESHOLD = 0.5;
export const MAYBE_THRESHOLD = 0.32;

export class FaceEmbedEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        msg.embedding = new Float32Array(msg.embedding); // rehydrate transferred buffer
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Embed a face ImageBitmap (transferred → zero-copy). Returns {embedding:Float32Array(128), dims, ms, device}. */
  embed(bitmap) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap }, [bitmap]);
    });
  }
}

/** Cosine similarity of two L2-normalized embeddings == their dot product. */
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Map a cosine similarity to a human, honest SIGNAL (never a hard identity verdict). */
export function verdict(cos, threshold = SAME_THRESHOLD, maybe = MAYBE_THRESHOLD) {
  if (cos >= threshold) return { key: "same", label: "Likely the same person", tone: "good" };
  if (cos >= maybe) return { key: "maybe", label: "Uncertain — borderline", tone: "warn" };
  return { key: "different", label: "Likely different people", tone: "bad" };
}

/** Read a File / Blob / element into an ImageBitmap for the worker. */
export function toBitmap(source) {
  return createImageBitmap(source);
}

/** Fetch a same-origin sample and decode it to an ImageBitmap. */
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

/** Draw a bitmap into a canvas as a centred square preview (matches the model's centre-crop). */
export function drawSquare(canvas, bitmap) {
  const side = Math.min(bitmap.width, bitmap.height);
  const sx = (bitmap.width - side) / 2, sy = (bitmap.height - side) / 2;
  canvas.getContext("2d").drawImage(bitmap, sx, sy, side, side, 0, 0, canvas.width, canvas.height);
}

/**
 * Render a compact 128-D embedding as signed bars (up = positive dim, down = negative), so the abstract
 * vector becomes something you can see and compare. `container` is a flex row; each dim is one bar.
 */
export function renderEmbeddingBars(container, embedding) {
  const max = Math.max(0.001, ...Array.from(embedding, Math.abs));
  container.replaceChildren(
    ...Array.from(embedding, (v) => {
      const cell = document.createElement("span");
      cell.className = "emb-cell";
      const bar = document.createElement("span");
      bar.className = "emb-bar " + (v >= 0 ? "pos" : "neg");
      bar.style.blockSize = (Math.abs(v) / max * 100).toFixed(1) + "%";
      cell.append(bar);
      return cell;
    }),
  );
}

/** A similarity gauge: place a needle on a −0.2 … 1.0 scale with the decision band marked. */
export function renderGauge(el, cos, threshold = SAME_THRESHOLD) {
  const lo = -0.2, hi = 1.0;
  const pct = Math.max(0, Math.min(1, (cos - lo) / (hi - lo))) * 100;
  const tpct = Math.max(0, Math.min(1, (threshold - lo) / (hi - lo))) * 100;
  el.style.setProperty("--needle", pct.toFixed(1) + "%");
  el.style.setProperty("--thresh", tpct.toFixed(1) + "%");
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the face-embedding widgets. Injected once per page. */
export const FACE_EMBED_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .85rem;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.pair-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.face-col { flex: 1 1 240px; min-inline-size: 0; }
.face-col h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.face-frame { position: relative; inline-size: 100%; aspect-ratio: 1 / 1; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.face-canvas, .cam-video { inline-size: 100%; block-size: 100%; object-fit: cover; display: block; }
.cam-video { transform: scaleX(-1); }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 56px; block-size: 56px; object-fit: cover; border-radius: 8px;
  border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.big-verdict { font-size: 1.35rem; font-weight: 600; margin: .3rem 0; display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
.sim-num { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700; }
.tone-good { color: var(--good); } .tone-warn { color: var(--warn); } .tone-bad { color: var(--bad); }
.gauge { position: relative; block-size: 1rem; border-radius: 999px; margin: .8rem 0 .3rem;
  background: linear-gradient(to right, var(--bad) 0%, var(--warn) 40%, var(--good) 70%); border: 1px solid var(--border); }
.gauge::before { content: ""; position: absolute; inset-block: -4px; inline-size: 3px; left: var(--thresh, 60%);
  background: var(--color); border-radius: 2px; opacity: .8; }
.gauge::after { content: ""; position: absolute; inset-block: -5px; inline-size: 4px; left: var(--needle, 0%);
  transform: translateX(-50%); background: var(--accent); border-radius: 2px; box-shadow: 0 0 0 2px var(--background); }
.gauge-scale { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.emb-strip { display: flex; align-items: center; gap: 1px; block-size: 60px; margin: .3rem 0; padding: 0 1px; }
.emb-cell { flex: 1 1 0; block-size: 100%; display: flex; flex-direction: column; justify-content: center; }
.emb-bar { display: block; inline-size: 100%; min-block-size: 1px; border-radius: 1px; }
.emb-bar.pos { background: var(--accent); align-self: flex-end; }
.emb-bar.neg { background: var(--muted); align-self: flex-start; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
.gallery-grid { display: flex; flex-wrap: wrap; gap: .6rem; margin: .6rem 0; }
.gallery-item { inline-size: 92px; text-align: center; }
.gallery-item canvas { inline-size: 92px; block-size: 92px; border-radius: 8px; border: 1px solid var(--border); object-fit: cover; }
.gallery-item .g-sim { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); display: block; margin-top: .2rem; }
.gallery-item.g-top canvas { border-color: var(--accent); border-width: 2px; }
.gallery-item.g-top .g-sim { color: var(--accent); font-weight: 600; }
`;
