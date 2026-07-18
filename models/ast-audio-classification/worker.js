// AST (Audio Spectrogram Transformer) worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the audio-classification pipeline and returns the top AudioSet labels with scores, the
// real log-mel spectrogram the model reasons over (128 mel bands — AST treats audio as an image), and
// the latency + backend actually used.
//
// Model: Xenova/ast-finetuned-audioset-10-10-0.4593 (task: audio-classification), q8 ONNX, WASM
// (WebGPU when a real adapter exists). We import the SHARED loader from lib/webai.js — no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/ast-finetuned-audioset-10-10-0.4593";
const TASK = "audio-classification";
const SR = 16000;

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: MODEL,
      backend: want,
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      const loaded = await loadPipeline({
        task: TASK,
        model: MODEL,
        backend: "wasm",
        dtype: "q8",
        onProgress: (p) => post({ type: "progress", p }),
      });
      pipe = loaded.pipe;
      device = loaded.device;
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// ---- log-mel spectrogram (the representation AST classifies) ---------------------------------------
// AST is the Audio Spectrogram Transformer: it turns audio into a 128-band log-mel spectrogram and
// treats that 2D image as a Vision-Transformer input. We compute a standard 25 ms / 10 ms log-mel here
// so the "see inside" surface shows the real image the model reasons over.
const NFFT = 512, WIN = 400, HOP = 160, NMEL = 128;

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

let _hann = null, _melFb = null;
function hann() {
  if (_hann) return _hann;
  _hann = new Float32Array(WIN);
  for (let i = 0; i < WIN; i++) _hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1));
  return _hann;
}
// Triangular mel filterbank over [0, SR/2], indexed by rfft bin (0..NFFT/2).
function melFilterbank() {
  if (_melFb) return _melFb;
  const nBins = NFFT / 2 + 1;
  const melMin = hzToMel(0), melMax = hzToMel(SR / 2);
  const pts = new Array(NMEL + 2);
  for (let i = 0; i < pts.length; i++) {
    pts[i] = Math.floor(((NFFT + 1) * melToHz(melMin + ((melMax - melMin) * i) / (NMEL + 1))) / SR);
  }
  _melFb = [];
  for (let m = 0; m < NMEL; m++) {
    const filt = new Float32Array(nBins);
    const l = pts[m], c = pts[m + 1], r = pts[m + 2];
    for (let k = l; k < c; k++) if (c > l) filt[k] = (k - l) / (c - l);
    for (let k = c; k < r; k++) if (r > c) filt[k] = (r - k) / (r - c);
    _melFb.push(filt);
  }
  return _melFb;
}

// Iterative radix-2 FFT (in-place, complex).
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + half] * cr - im[i + k + half] * ci;
        const bi = re[i + k + half] * ci + im[i + k + half] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + half] = ar - br;
        im[i + k + half] = ai - bi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

function logMel(pcm) {
  const w = hann();
  const fb = melFilterbank();
  const maxFrames = Math.min(1024, Math.floor((pcm.length - WIN) / HOP) + 1);
  const frames = Math.max(1, maxFrames);
  const out = new Float32Array(frames * NMEL);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  const nBins = NFFT / 2 + 1;
  const power = new Float64Array(nBins);
  let lo = Infinity, hi = -Infinity;
  for (let f = 0; f < frames; f++) {
    re.fill(0);
    im.fill(0);
    const off = f * HOP;
    for (let i = 0; i < WIN; i++) re[i] = (pcm[off + i] ?? 0) * w[i];
    fft(re, im);
    for (let k = 0; k < nBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < NMEL; m++) {
      const filt = fb[m];
      let e = 0;
      for (let k = 0; k < nBins; k++) e += filt[k] * power[k];
      const v = Math.log(Math.max(e, 1e-10));
      out[f * NMEL + m] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return { data: out, frames, mels: NMEL, min: lo, max: hi };
}

async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const topK = opts?.topK ?? 15;
  const t0 = performance.now();
  const output = await pipe(audio, { top_k: topK });
  const ms = Math.round(performance.now() - t0);
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  const spectrogram = logMel(audio);
  post(
    { type: "result", id, labels, spectrogram, ms, device, durationS: audio.length / SR },
    [spectrogram.data.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.device);
    else if (type === "run") await run(e.data.id, e.data.audio, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
