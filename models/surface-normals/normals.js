// Front-end helpers shared by every Surface-Normals page. Keeps pages thin: owns the worker handshake
// (transferring ImageBitmaps + the normal field), turns files / samples into ImageBitmaps, draws the
// RGB-encoded normal map, relights the surface from the unit-normal field (Lambertian N·L), and reads
// the normal vector at a picked pixel. ALL inference lives in worker.js (off the main thread, raw ONNX
// Runtime Web). The normal field the worker returns is Float32 CHW [3,H,W] of unit vectors, so every
// page can relight/inspect WITHOUT re-running the (slow) model.

const WORKER_URL = "/web-ai-showcase/models/surface-normals/worker.js";

export class NormalEngine {
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
        msg.normals = new Float32Array(msg.normals); // rehydrate transferred buffer
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

  /** Estimate normals on an ImageBitmap (transferred → zero-copy). Returns
   *  {width,height,normals:Float32(3*w*h),bitmap:ImageBitmap,meanNormal,dims,ms,device}. */
  estimate(bitmap) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap }, [bitmap]);
    });
  }
}

/** Read a File / Blob / element into an ImageBitmap for the worker. */
export function toBitmap(source) {
  return createImageBitmap(source);
}
/** Fetch a same-origin sample and decode it to an ImageBitmap. */
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

/** Blit a worker-composited normal-map ImageBitmap into a canvas (single drawImage). */
export function drawNormalBitmap(canvas, bitmap) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

/**
 * Relight the surface from its unit-normal field with one or more directional lights (Lambertian
 * N·L + ambient). `normals` is Float32 CHW [3,w,h]. `lights` = [{dir:[x,y,z], color:[r,g,b], intensity}].
 * Writes into `canvas` at the native normal-map resolution. Cheap enough (~w*h*lights) to run per
 * animation frame on the main thread; callers coalesce with requestAnimationFrame.
 */
export function relight(canvas, normals, w, h, lights, ambient = 0.12, albedo = [235, 232, 225]) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const px = img.data;
  const plane = w * h;
  // Pre-normalize light directions.
  const L = lights.map((l) => {
    const m = Math.hypot(l.dir[0], l.dir[1], l.dir[2]) || 1;
    return {
      x: l.dir[0] / m,
      y: l.dir[1] / m,
      z: l.dir[2] / m,
      c: l.color || [255, 255, 255],
      i: l.intensity ?? 1,
    };
  });
  for (let i = 0; i < plane; i++) {
    const nx = normals[i], ny = normals[plane + i], nz = normals[2 * plane + i];
    let r = ambient * albedo[0], g = ambient * albedo[1], b = ambient * albedo[2];
    for (const l of L) {
      const d = Math.max(0, nx * l.x + ny * l.y + nz * l.z) * l.i;
      if (d > 0) {
        r += d * (l.c[0] / 255) * albedo[0];
        g += d * (l.c[1] / 255) * albedo[1];
        b += d * (l.c[2] / 255) * albedo[2];
      }
    }
    const o = i * 4;
    px[o] = r > 255 ? 255 : r;
    px[o + 1] = g > 255 ? 255 : g;
    px[o + 2] = b > 255 ? 255 : b;
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Return the unit normal [x,y,z] at fractional coords (u,v) ∈ [0,1] of the normal field. */
export function pickNormal(normals, w, h, u, v) {
  const x = Math.min(w - 1, Math.max(0, Math.round(u * (w - 1))));
  const y = Math.min(h - 1, Math.max(0, Math.round(v * (h - 1))));
  const i = y * w + x;
  const plane = w * h;
  return [normals[i], normals[plane + i], normals[2 * plane + i]];
}

/** Render a "normal sphere" legend: a shaded sphere coloured by its own surface normals, so the
 *  RGB↔XYZ mapping (right=red, up=green, toward-you=blue) is legible at a glance. */
export function renderNormalSphere(canvas, size = 120) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  const px = img.data;
  const r = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = (x - r) / r, ny = -(y - r) / r; // screen up = +y
      const d = nx * nx + ny * ny;
      const o = (y * size + x) * 4;
      if (d > 1) {
        px[o + 3] = 0;
        continue;
      }
      const nz = Math.sqrt(1 - d);
      px[o] = (nx * 0.5 + 0.5) * 255;
      px[o + 1] = (ny * 0.5 + 0.5) * 255;
      px[o + 2] = (nz * 0.5 + 0.5) * 255;
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the surface-normals widgets. Injected once per page. */
export const NORMAL_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: .8rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; font-size: .85rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 62px; block-size: 62px; object-fit: cover; border-radius: 8px; border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 280px; min-inline-size: 0; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas, .preview-img { inline-size: 100%; block-size: auto; max-block-size: 460px; object-fit: contain;
  border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block; }
.viz-canvas.pickable { cursor: crosshair; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.warn-box { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .7rem .9rem; margin: .5rem 0; font-size: .84rem; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.legend-flex { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
.sphere-legend { border-radius: 50%; border: 1px solid var(--border); inline-size: 120px; block-size: 120px; }
.axis-key { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); line-height: 1.7; }
.axis-key .rr { color: #d9534f; } .axis-key .gg { color: #5cb85c; } .axis-key .bb { color: #6c8cff; }
.light-pad { position: relative; inline-size: 150px; block-size: 150px; border-radius: 50%; border: 1px solid var(--border-strong);
  background: radial-gradient(circle at 50% 50%, var(--bg-secondary), var(--bg-raised)); touch-action: none; cursor: pointer; flex: 0 0 auto; }
.light-pad:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.light-dot { position: absolute; inline-size: 16px; block-size: 16px; border-radius: 50%; background: var(--accent); border: 2px solid var(--background); transform: translate(-50%, -50%); pointer-events: none; }
.pick-vec { font-family: var(--font-mono); font-size: .82rem; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
`;
