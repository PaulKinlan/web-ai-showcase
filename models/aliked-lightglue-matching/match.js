// Front-end helpers for the ALIKED + LightGlue matching pages. Keeps each page thin: it owns the worker
// handshake (transferring ImageBitmaps so nothing is copied), turns files / gallery images / canvas
// transforms into ImageBitmaps, and draws the two images with their correspondence lines. ALL inference
// AND preprocessing happen off the main thread in worker.js (raw ONNX Runtime Web). Privacy by
// construction: the images and every correspondence never leave the device.

const WORKER_URL = "/web-ai-showcase/models/aliked-lightglue-matching/worker.js";

export class MatchEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
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

  /** Match two ImageBitmaps (transferred → zero-copy). Returns
   * { pairs:[{a:[x,y],b:[x,y],score}], keypointsA, keypointsB, numMatches, numKeypointsA/B, sizeA,
   *   sizeB, extMs, matchMs, ms, device }. */
  match(bitmapA, bitmapB, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmapA, bitmapB, opts }, [bitmapA, bitmapB]);
    });
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
export function decodeImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("Could not decode that image."));
    im.src = url;
  });
}
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}
export function toBitmap(source) {
  return createImageBitmap(source);
}

// Build a transformed "second view" of an image on a canvas: rotate (deg), scale, translate (fraction
// of size), optional perspective skew. Returns a canvas the demo can turn into an ImageBitmap. Honest:
// the transform is explicit + user-controlled, so matches have known ground truth.
export function transformView(img, { rotate = 0, scale = 1, tx = 0, ty = 0, skew = 0 } = {}) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2 + tx * w, h / 2 + ty * h);
  ctx.rotate(rotate * Math.PI / 180);
  ctx.transform(1, skew, skew, 1, 0, 0);
  ctx.scale(scale, scale);
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
  return c;
}

const MATCH_COLORS = ["#1a7a3a", "#1565c0", "#b06a00", "#6a1b9a", "#b3261e", "#00838f"];

/**
 * Draw the two images side-by-side (or stacked if `stack`) on `canvas` at a common scale, with a line
 * per correspondence. `pairs` = [{a:[x,y], b:[x,y], score}] in each image's native pixel coords.
 * opts: { stack, maxMatches, showKeypoints, keypointsA, keypointsB, lineWidth }.
 */
export function drawMatches(canvas, imgA, imgB, pairs, opts = {}) {
  const stack = opts.stack ?? false;
  const wA = imgA.naturalWidth || imgA.width, hA = imgA.naturalHeight || imgA.height;
  const wB = imgB.naturalWidth || imgB.width, hB = imgB.naturalHeight || imgB.height;
  const ctx = canvas.getContext("2d");
  let ox, oy, dispW, dispH;
  if (stack) {
    dispW = Math.max(wA, wB);
    const sA = dispW / wA, sB = dispW / wB;
    canvas.width = dispW;
    canvas.height = Math.round(hA * sA + hB * sB);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgA, 0, 0, wA * sA, hA * sA);
    ctx.drawImage(imgB, 0, hA * sA, wB * sB, hB * sB);
    ox = { x: 0, y: 0, s: sA };
    oy = { x: 0, y: hA * sA, s: sB };
  } else {
    dispH = Math.max(hA, hB);
    const sA = dispH / hA, sB = dispH / hB;
    canvas.width = Math.round(wA * sA + wB * sB);
    canvas.height = dispH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgA, 0, 0, wA * sA, hA * sA);
    ctx.drawImage(imgB, wA * sA, 0, wB * sB, hB * sB);
    ox = { x: 0, y: 0, s: sA };
    oy = { x: wA * sA, y: 0, s: sB };
  }

  if (opts.showKeypoints) {
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    for (const p of (opts.keypointsA || [])) {
      ctx.beginPath();
      ctx.arc(ox.x + p[0] * ox.s, ox.y + p[1] * ox.s, 1.2, 0, 6.28);
      ctx.fill();
    }
    for (const p of (opts.keypointsB || [])) {
      ctx.beginPath();
      ctx.arc(oy.x + p[0] * oy.s, oy.y + p[1] * oy.s, 1.2, 0, 6.28);
      ctx.fill();
    }
  }

  const lw = opts.lineWidth ?? Math.max(1, canvas.width / 900);
  const r = Math.max(2, canvas.width / 400);
  const list = opts.maxMatches ? pairs.slice(0, opts.maxMatches) : pairs;
  ctx.lineWidth = lw;
  list.forEach((m, i) => {
    const color = MATCH_COLORS[i % MATCH_COLORS.length];
    const ax = ox.x + m.a[0] * ox.s, ay = ox.y + m.a[1] * ox.s;
    const bx = oy.x + m.b[0] * oy.s, by = oy.y + m.b[1] * oy.s;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(ax, ay, r, 0, 6.28);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(bx, by, r, 0, 6.28);
    ctx.fill();
  });
  return { ox, oy };
}

// ── Similarity transform (scale · rotation · translation) from matches, via a closed-form least
// squares fit wrapped in RANSAC. Honest geometry on the model's REAL correspondences — NOT a second
// model. Maps points in image A onto image B: b ≈ s·R·a + t. Returns
// { s, thetaDeg, tx, ty, apply(p), inliers, rms } or null. ──
function fitSimilarity(src, dst) {
  // Umeyama (no reflection). src/dst: arrays of [x,y].
  const n = src.length;
  let mx = 0, my = 0, ux = 0, uy = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0];
    my += src[i][1];
    ux += dst[i][0];
    uy += dst[i][1];
  }
  mx /= n;
  my /= n;
  ux /= n;
  uy /= n;
  let sxx = 0, varS = 0, a = 0, b = 0; // a = Σ(sc·dc), b = cross term for rotation
  let c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - mx, sy = src[i][1] - my;
    const dx = dst[i][0] - ux, dy = dst[i][1] - uy;
    varS += sx * sx + sy * sy;
    c00 += dx * sx;
    c01 += dx * sy;
    c10 += dy * sx;
    c11 += dy * sy;
  }
  // 2x2 covariance H = [[c00,c01],[c10,c11]]; rotation from its SVD; for 2D similarity:
  const theta = Math.atan2(c10 - c01, c00 + c11);
  const cos = Math.cos(theta), sin = Math.sin(theta);
  const s = varS > 1e-9 ? ((c00 + c11) * cos + (c10 - c01) * sin) / varS : 1;
  const tx = ux - s * (cos * mx - sin * my);
  const ty = uy - s * (sin * mx + cos * my);
  const apply = (p) => [s * (cos * p[0] - sin * p[1]) + tx, s * (sin * p[0] + cos * p[1]) + ty];
  return { s, cos, sin, tx, ty, thetaDeg: theta * 180 / Math.PI, apply };
}
export function estimateSimilarity(pairs, { thresh = 6, iters = 300 } = {}) {
  if (pairs.length < 4) return null;
  const src = pairs.map((m) => m.a), dst = pairs.map((m) => m.b);
  let bestInliers = [], bestModel = null;
  for (let it = 0; it < iters; it++) {
    const idx = [];
    while (idx.length < 3) {
      const r = (Math.random() * pairs.length) | 0;
      if (!idx.includes(r)) idx.push(r);
    }
    let m;
    try {
      m = fitSimilarity(idx.map((i) => src[i]), idx.map((i) => dst[i]));
    } catch {
      continue;
    }
    if (!m || !isFinite(m.s)) continue;
    const inliers = [];
    for (let i = 0; i < pairs.length; i++) {
      const q = m.apply(src[i]);
      if (Math.hypot(q[0] - dst[i][0], q[1] - dst[i][1]) < thresh) inliers.push(i);
    }
    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestModel = m;
    }
  }
  if (!bestModel || bestInliers.length < 4) return null;
  const model = fitSimilarity(bestInliers.map((i) => src[i]), bestInliers.map((i) => dst[i]));
  let se = 0;
  for (const i of bestInliers) {
    const q = model.apply(src[i]);
    se += (q[0] - dst[i][0]) ** 2 + (q[1] - dst[i][1]) ** 2;
  }
  model.inliers = bestInliers;
  model.rms = Math.sqrt(se / bestInliers.length);
  return model;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Shared inline styles (design-system tokens only → light/dark for free).
export const MATCH_CSS = `
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; font-size:.85rem; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.pair-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin:.6rem 0; }
@media (max-width:560px){ .pair-grid { grid-template-columns:1fr; } }
.pane { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem; min-inline-size:0; }
.pane h4 { margin:0 0 .4rem; font-size:.82rem; color:var(--muted); }
.pane img, .pane canvas { inline-size:100%; block-size:auto; display:block; border-radius:6px; max-block-size:34vh; object-fit:contain; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.4rem 0; }
.sample-thumb { inline-size:70px; block-size:52px; object-fit:cover; border-radius:6px; border:2px solid transparent;
  cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.canvas-wrap { position:relative; margin-top:.5rem; border-radius:8px; overflow:auto; background:var(--bg-raised); border:1px solid var(--border); }
.match-canvas { display:block; max-inline-size:100%; block-size:auto; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.field-row label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.slider-row { display:flex; align-items:center; gap:.6rem; margin:.5rem 0; flex-wrap:wrap; }
.slider-row input[type=range] { flex:1 1 160px; accent-color:var(--accent); min-inline-size:0; }
.slider-row output { font-family:var(--font-mono); font-size:.82rem; min-inline-size:3.5ch; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem; margin:.6rem 0; }
.metric { border:1px solid var(--border); border-radius:8px; padding:.6rem .7rem; background:var(--bg-raised); }
.metric .k { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.metric .v { font-family:var(--font-mono); font-size:1.15rem; font-weight:600; }
.verdict { border:1px solid var(--border); border-radius:8px; padding:.7rem .9rem; margin:.5rem 0; background:var(--bg-raised); font-size:.9rem; }
.verdict.yes { border-color:var(--ok,#1a7a3a); } .verdict.no { border-color:var(--warn,#b06a00); }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th,.inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border); font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
`;
