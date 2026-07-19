// CREPE pitch-detection worker — ALL inference off the main thread.
//
// CREPE (Kim et al., ICASSP 2018) is a monophonic pitch tracker: a small CNN that reads a 1024-sample
// window at 16 kHz and outputs a 360-bin pitch "salience" activation spanning C1..B7 in 20-cent steps.
// The fundamental frequency (f0) is the confidence-weighted centre of the activation peak; the peak
// height is the voicing confidence.
//
// transformers.js has NO pitch-estimation task / CREPE class, so — exactly like models/silero-vad —
// we drive inference with onnxruntime-web DIRECTLY, isolated to THIS worker. We do NOT touch the shared
// lib/webai.js (transformers.js only). We DO reuse the shared cache/auto-init layer: the ONNX blob is
// stored in Cache Storage under its real HF URL, so lib/model-cache.js (which scans Cache Storage by
// modelId) detects it and createModelLoader auto-initialises on a return visit — the same auto-init
// contract every other page honours.
//
// Model: niobures/CREPE · crepe_onnx_tiny.onnx (~1.9 MB, fp32). The CREPE weights are by Jong Wook Kim
// et al. (MIT, github.com/marl/crepe); this repo re-hosts an ONNX conversion of the "tiny" model.
//   input  = frames [N, 1024] float32 (each row a per-frame-normalised 1024-sample @16 kHz window)
//   output = probabilities [N, 360] float32 (per-bin pitch salience, ~0..1)

const ORT_VERSION = "1.20.1";
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;
const MODEL_ID = "niobures/CREPE";
const MODEL_URL =
  `https://huggingface.co/${MODEL_ID}/resolve/main/models/onnx/crepe_onnx_tiny.onnx`;
const ONNX_CACHE = "onnxruntime-web-models"; // a real Cache Storage cache model-cache.js can scan
const SR = 16000;
const WIN = 1024; // CREPE input window
const N_BINS = 360;
const BATCH = 128; // frames per session.run — bounds memory; keeps each task short
const MAX_FRAMES = 1600; // cap total frames on a long clip (hop auto-widens) so transfers stay bounded

// CREPE bin→cents mapping (marl/crepe): cents = linspace(0,7180,360) + 1997.3794084376191.
const CENTS = new Float64Array(N_BINS);
for (let i = 0; i < N_BINS; i++) CENTS[i] = (7180 * i) / (N_BINS - 1) + 1997.3794084376191;

let ort = null;
let session = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function fetchModelBytes(onProgress) {
  // Cache-first across Cache Storage so a return visit is offline + button-free (auto-init contract).
  let resp = null;
  try {
    const cache = await caches.open(ONNX_CACHE);
    resp = await cache.match(MODEL_URL);
    if (resp) return await resp.arrayBuffer();
    const net = await fetch(MODEL_URL);
    if (!net.ok) throw new Error(`model fetch ${net.status}`);
    const total = Number(net.headers.get("content-length")) || 0;
    if (net.body && total) {
      const reader = net.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        onProgress?.({ status: "progress", progress: (got / total) * 100, file: "crepe.onnx" });
      }
      const bytes = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) {
        bytes.set(c, off);
        off += c.length;
      }
      const stored = new Response(bytes, {
        headers: { "content-type": "application/octet-stream", "content-length": String(got) },
      });
      await cache.put(MODEL_URL, stored.clone());
      return bytes.buffer;
    }
    const buf = await net.arrayBuffer();
    await cache.put(MODEL_URL, new Response(buf.slice(0)));
    return buf;
  } catch (err) {
    if (resp === null) {
      const net = await fetch(MODEL_URL);
      if (!net.ok) throw new Error(`model fetch ${net.status}`);
      return await net.arrayBuffer();
    }
    throw err;
  }
}

async function ensureLoaded(onProgress) {
  if (session) return;
  if (!ort) {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1; // GitHub Pages has no COOP/COEP ⇒ no SharedArrayBuffer; single-threaded
  }
  const bytes = await fetchModelBytes(onProgress);
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm (onnxruntime-web)" });
}

// Build one per-frame-normalised 1024-sample window centred at `center`, zero-padded at edges.
function windowAt(pcm, center, dst, dstOff) {
  const half = WIN >> 1;
  let mean = 0;
  for (let i = 0; i < WIN; i++) {
    const idx = center - half + i;
    const v = idx >= 0 && idx < pcm.length ? pcm[idx] : 0;
    dst[dstOff + i] = v;
    mean += v;
  }
  mean /= WIN;
  let std = 0;
  for (let i = 0; i < WIN; i++) std += (dst[dstOff + i] - mean) ** 2;
  std = Math.sqrt(std / WIN) || 1;
  for (let i = 0; i < WIN; i++) dst[dstOff + i] = (dst[dstOff + i] - mean) / std;
}

// Confidence-weighted local average of cents around the peak bin → f0 (Hz) + confidence.
function decodeActivation(act, off) {
  let mi = 0, mv = -Infinity;
  for (let i = 0; i < N_BINS; i++) {
    const v = act[off + i];
    if (v > mv) {
      mv = v;
      mi = i;
    }
  }
  const lo = Math.max(0, mi - 4), hi = Math.min(N_BINS - 1, mi + 4);
  let s = 0, w = 0;
  for (let i = lo; i <= hi; i++) {
    s += act[off + i] * CENTS[i];
    w += act[off + i];
  }
  const cents = w ? s / w : CENTS[mi];
  const f0 = 10 * Math.pow(2, cents / 1200);
  return { f0, conf: mv };
}

// Analyse a whole 16 kHz mono clip → per-frame f0 + confidence + the raw activation matrix.
async function runClip(id, pcm, opts = {}) {
  await ensureLoaded();
  const t0 = performance.now();
  let hop = Math.max(32, Math.round((opts.hopMs ?? 16) * SR / 1000));
  let nFrames = Math.max(1, Math.floor(pcm.length / hop));
  if (nFrames > MAX_FRAMES) { // widen hop on long clips so the transfer stays bounded
    hop = Math.ceil(pcm.length / MAX_FRAMES);
    nFrames = Math.max(1, Math.floor(pcm.length / hop));
  }
  const f0 = new Float32Array(nFrames);
  const conf = new Float32Array(nFrames);
  const times = new Float32Array(nFrames);
  const activations = new Float32Array(nFrames * N_BINS); // the pitchgram (see-inside)
  const inName = session.inputNames[0], outName = session.outputNames[0];

  for (let start = 0; start < nFrames; start += BATCH) {
    const count = Math.min(BATCH, nFrames - start);
    const buf = new Float32Array(count * WIN);
    for (let f = 0; f < count; f++) {
      const gi = start + f;
      times[gi] = (gi * hop) / SR;
      windowAt(pcm, gi * hop + (WIN >> 1) - (hop >> 1), buf, f * WIN);
    }
    const t = new ort.Tensor("float32", buf, [count, WIN]);
    const res = await session.run({ [inName]: t });
    const act = res[outName].data;
    for (let f = 0; f < count; f++) {
      const gi = start + f;
      activations.set(act.subarray(f * N_BINS, f * N_BINS + N_BINS), gi * N_BINS);
      const { f0: hz, conf: c } = decodeActivation(act, f * N_BINS);
      f0[gi] = hz;
      conf[gi] = c;
    }
    post({
      type: "progress",
      p: { status: "progress", progress: ((start + count) / nFrames) * 100 },
    });
  }
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    f0,
    conf,
    times,
    activations,
    nFrames,
    nBins: N_BINS,
    hop,
    frameSec: hop / SR,
    durationS: pcm.length / SR,
    ms,
    device: "wasm (onnxruntime-web)",
  }, [f0.buffer, conf.buffer, times.buffer, activations.buffer]);
}

// Live path: run CREPE on the most-recent 1024-sample window of a rolling buffer → latest f0/conf.
async function streamWindow(id, pcm) {
  await ensureLoaded();
  const buf = new Float32Array(WIN);
  const center = Math.max(WIN >> 1, pcm.length - (WIN >> 1));
  windowAt(pcm, center, buf, 0);
  const t = new ort.Tensor("float32", buf, [1, WIN]);
  const res = await session.run({ [session.inputNames[0]]: t });
  const act = res[session.outputNames[0]].data;
  const { f0, conf } = decodeActivation(act, 0);
  // Send back a copy of the activation so the live view can draw the salience curve.
  const actCopy = new Float32Array(N_BINS);
  actCopy.set(act.subarray(0, N_BINS));
  post({ type: "stream", id, f0, conf, activation: actCopy, nBins: N_BINS }, [actCopy.buffer]);
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") {
      await ensureLoaded((p) => post({ type: "progress", p }));
    } else if (d.type === "run") {
      await runClip(d.id, d.pcm, d.opts);
    } else if (d.type === "stream-window") {
      await streamWindow(d.id, d.pcm);
    }
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
