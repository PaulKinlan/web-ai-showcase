// HuBERT feature-extraction worker — ALL inference AND the dense heatmap composite run off the main
// thread (invariant 15: measure, don't infer; keep INP low; transfer RGBA, don't clone).
//
// HuBERT is a SELF-SUPERVISED SPEECH representation model — and its pretraining OBJECTIVE is the thing
// that makes it distinct from the other speech models in this showcase:
//   • wav2vec2 (CTC) and Whisper (seq2seq) emit TEXT.
//   • WavLM learns by masked-speech-prediction PLUS a denoising / overlapped-speech objective.
//   • HuBERT learns by MASKED CLUSTER PREDICTION: an offline k-means over acoustic features (MFCCs,
//     then the model's own hidden states in later iterations) assigns every frame a discrete cluster
//     id, and the model is trained to predict those cluster ids for MASKED frames — a BERT-style
//     masked-prediction task, but over self-discovered acoustic units instead of words. That "predict
//     the hidden cluster" objective is why HuBERT's frame embeddings organise speech into phone-like
//     units so cleanly.
//
// The OUTPUT is the same kind of thing WavLM gives: for every ~20 ms audio frame a 768-dim CONTEXTUAL
// EMBEDDING vector (`last_hidden_state`, shape [1, T, 768]) — no letters, no words. We call the model
// DIRECTLY (AutoModel, not a pipeline) so we can return the raw frames for the heatmap and do
// pooling / cosine ourselves.
//
// Model: Xenova/hubert-base-ls960 (facebook/hubert-base-ls960), q8 ONNX, WASM (WebGPU when an adapter
// exists). Transformers.js via the SHARED CDN url from lib/webai.js. HuBERT's model class ("hubert")
// is registered in transformers.js 3.7.5 — no version pin needed.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/hubert-base-ls960";
const SR = 16000;

let model = null;
let processor = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function loadOn(dev) {
  const { AutoProcessor, AutoModel, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  processor = await AutoProcessor.from_pretrained(MODEL, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModel.from_pretrained(MODEL, {
    device: dev,
    dtype: "q8", // → onnx/model_quantized.onnx
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
}

async function ensureLoaded(preferred) {
  if (model) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  try {
    await loadOn(want);
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      model = null;
      processor = null;
      await loadOn("wasm");
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// Run the model on a 16 kHz mono Float32Array → { frames:Float32Array(T*D), T, D }. Frames are the raw
// per-frame contextual embeddings (last_hidden_state), kept flat for cheap pooling.
async function encode(audio) {
  const inputs = await processor(audio);
  const out = await model(inputs);
  const hs = out.last_hidden_state; // Tensor [1, T, D]
  const [, T, D] = hs.dims;
  return { frames: hs.data, T, D };
}

// Mean-pool a flat [T*D] frame matrix over time, then L2-normalise → a single utterance embedding.
function meanPoolL2(frames, T, D, t0 = 0, t1 = T) {
  const v = new Float32Array(D);
  const n = Math.max(1, t1 - t0);
  for (let t = t0; t < t1; t++) {
    const base = t * D;
    for (let d = 0; d < D; d++) v[d] += frames[base + d];
  }
  let norm = 0;
  for (let d = 0; d < D; d++) {
    v[d] /= n;
    norm += v[d] * v[d];
  }
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < D; d++) v[d] /= norm;
  return v;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both already L2-normalised
}

// A perceptual-ish magma ramp (few stops, linear interp) so the heatmap reads in light AND dark.
const RAMP = [
  [12, 8, 38],
  [86, 15, 110],
  [187, 55, 84],
  [249, 142, 8],
  [252, 253, 191],
];
function ramp(x) {
  x = x < 0 ? 0 : x > 1 ? 1 : x;
  const s = x * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(s));
  const f = s - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// Build an RGBA heatmap of the frame matrix IN THE WORKER (invariant 15: dense composite off-thread,
// transfer the buffer back). x = time (frames, capped/averaged), y = embedding dimension (capped).
// Robust per-matrix normalisation (2nd–98th percentile) so a few outliers don't wash it out.
function heatmap(frames, T, D, maxW = 320, maxH = 256) {
  const W = Math.min(T, maxW);
  const H = Math.min(D, maxH);
  const tStep = T / W, dStep = D / H;
  // Downsampled cell values (average-pool over the covered region).
  const cells = new Float32Array(W * H);
  for (let x = 0; x < W; x++) {
    const t0 = Math.floor(x * tStep), t1 = Math.max(t0 + 1, Math.floor((x + 1) * tStep));
    for (let y = 0; y < H; y++) {
      const d0 = Math.floor(y * dStep), d1 = Math.max(d0 + 1, Math.floor((y + 1) * dStep));
      let sum = 0, n = 0;
      for (let t = t0; t < t1; t++) {
        const base = t * D;
        for (let d = d0; d < d1; d++) {
          sum += frames[base + d];
          n++;
        }
      }
      cells[y * W + x] = n ? sum / n : 0;
    }
  }
  // Robust range from a sorted sample.
  const sample = Float32Array.from(cells).sort();
  const lo = sample[Math.floor(sample.length * 0.02)] ?? 0;
  const hi = sample[Math.floor(sample.length * 0.98)] ?? 1;
  const span = hi - lo || 1;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < cells.length; i++) {
    const [r, g, b] = ramp((cells[i] - lo) / span);
    const o = i * 4;
    rgba[o] = r;
    rgba[o + 1] = g;
    rgba[o + 2] = b;
    rgba[o + 3] = 255;
  }
  return { rgba, W, H };
}

// ── Operations ──────────────────────────────────────────────────────────────

async function embed(id, audio, audioSec) {
  await ensureLoaded();
  const t0 = performance.now();
  const { frames, T, D } = await encode(audio);
  const pooled = meanPoolL2(frames, T, D);
  const hm = heatmap(frames, T, D);
  const ms = Math.round(performance.now() - t0);
  const sec = audioSec || audio.length / SR;
  post({
    type: "embed",
    id,
    pooled: Array.from(pooled),
    frames: T,
    dim: D,
    frameMs: T ? (sec / T) * 1000 : 20,
    audioSec: sec,
    ms,
    device,
    heat: { w: hm.W, h: hm.H, rgba: hm.rgba.buffer },
  }, [hm.rgba.buffer]);
}

// Embed several clips → pooled vectors + a pairwise cosine matrix + a 2D projection (PCA via power
// iteration on the covariance of the pooled vectors — all in the worker).
async function similarity(id, clips) {
  await ensureLoaded();
  const t0 = performance.now();
  const vectors = [];
  for (const c of clips) {
    const { frames, T, D } = await encode(c.audio);
    vectors.push(meanPoolL2(frames, T, D));
  }
  const n = vectors.length;
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const c = i === j ? 1 : cosine(vectors[i], vectors[j]);
      matrix[i][j] = c;
      matrix[j][i] = c;
    }
  }
  const coords = pca2(vectors);
  post({
    type: "similarity",
    id,
    names: clips.map((c) => c.name),
    matrix,
    coords,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

// Top-2 principal components of a set of vectors, returned as 2D coordinates (mean-centred, power
// iteration with deflation). Small n, small dim after we only need the projection — cheap.
function pca2(vectors) {
  const n = vectors.length, D = vectors[0].length;
  const mean = new Float32Array(D);
  for (const v of vectors) for (let d = 0; d < D; d++) mean[d] += v[d] / n;
  const X = vectors.map((v) => Float32Array.from(v, (x, d) => x - mean[d]));
  const powerPC = (deflate) => {
    let pc = new Float32Array(D).map(() => Math.random() - 0.5);
    for (let it = 0; it < 60; it++) {
      const acc = new Float32Array(D);
      for (const x of X) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += x[d] * pc[d];
        for (let d = 0; d < D; d++) acc[d] += dot * x[d];
      }
      if (deflate) {
        let dd = 0;
        for (let d = 0; d < D; d++) dd += acc[d] * deflate[d];
        for (let d = 0; d < D; d++) acc[d] -= dd * deflate[d];
      }
      let norm = 0;
      for (let d = 0; d < D; d++) norm += acc[d] * acc[d];
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < D; d++) acc[d] /= norm;
      pc = acc;
    }
    return pc;
  };
  const pc1 = powerPC(null);
  const pc2 = powerPC(pc1);
  return X.map((x) => {
    let a = 0, b = 0;
    for (let d = 0; d < D; d++) {
      a += x[d] * pc1[d];
      b += x[d] * pc2[d];
    }
    return [a, b];
  });
}

// Slide a window over the haystack's frames, cosine each window's pooled embedding against the query's
// pooled embedding → a similarity-over-time curve + the best-matching segment. Real embeddings, real
// windowing — nothing pre-computed.
async function search(id, query, haystack, haySec, windowSec, hopSec) {
  await ensureLoaded();
  const t0 = performance.now();
  const q = await encode(query);
  const qv = meanPoolL2(q.frames, q.T, q.D);
  const h = await encode(haystack);
  const frameSec = haySec / h.T;
  const win = Math.max(1, Math.round((windowSec || 1.0) / frameSec));
  const hop = Math.max(1, Math.round((hopSec || 0.2) / frameSec));
  const times = [], sims = [];
  let best = { score: -Infinity, start: 0, end: 0 };
  for (let s = 0; s + win <= h.T; s += hop) {
    const wv = meanPoolL2(h.frames, h.T, h.D, s, s + win);
    const c = cosine(qv, wv);
    const startSec = s * frameSec;
    const endSec = (s + win) * frameSec;
    times.push(startSec);
    sims.push(c);
    if (c > best.score) best = { score: c, start: startSec, end: endSec };
  }
  post({
    type: "search",
    id,
    times,
    sims,
    best,
    windowSec: win * frameSec,
    frameSec,
    haySec,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.device);
    else if (d.type === "embed") await embed(d.id, d.audio, d.audioSec);
    else if (d.type === "similarity") await similarity(d.id, d.clips);
    else if (d.type === "search") {
      await search(d.id, d.query, d.haystack, d.haySec, d.windowSec, d.hopSec);
    }
  } catch (err) {
    console.error("[hubert worker]", err);
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
