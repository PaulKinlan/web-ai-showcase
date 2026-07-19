// CLAP zero-shot-audio-classification worker — ALL inference off the main thread so the control UI
// stays responsive. The main thread decodes/records audio into a 48 kHz mono Float32Array and transfers
// it here; the worker embeds the clip and every free-text label into CLAP's ONE shared 512-d space and
// returns, from a single forward pass: the softmax probabilities, the raw scaled logits, the audio↔label
// cosine similarities, the embedding dims, the real 64-band log-mel spectrogram the audio encoder reads,
// and the latency + backend actually used.
//
// Model: Xenova/clap-htsat-unfused (task: zero-shot-audio-classification), q8 ONNX, WASM (WebGPU when a
// real adapter exists). We import the SHARED loader from lib/webai.js and reuse the pipeline's own
// model/processor/tokenizer to reach the low-level tensors — no second download, no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/clap-htsat-unfused";
const TASK = "zero-shot-audio-classification";
const SR = 48000;

let pipe = null;
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

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Cosine similarity between the single audio embedding and each per-label text embedding.
function cosines(audioEmb, textEmb, dim, n) {
  const aN = l2norm(audioEmb);
  const out = [];
  for (let i = 0; i < n; i++) {
    let dot = 0, tN = 0;
    for (let d = 0; d < dim; d++) {
      const t = textEmb[i * dim + d];
      dot += audioEmb[d] * t;
      tN += t * t;
    }
    out.push(dot / (aN * Math.sqrt(tN) || 1));
  }
  return out;
}

// ---- log-mel spectrogram (the representation CLAP's audio encoder classifies) ----------------------
// CLAP's ClapFeatureExtractor: 48 kHz, n_fft=1024, hop=480, 64 mel bands over [50, 14000] Hz. We compute
// the same 64-band log-mel here so the "see inside" surface shows the real image the model reasons over.
const NFFT = 1024, WIN = 1024, HOP = 480, NMEL = 64, FMIN = 50, FMAX = 14000;

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
// Triangular mel filterbank over [FMIN, FMAX], indexed by rfft bin (0..NFFT/2).
function melFilterbank() {
  if (_melFb) return _melFb;
  const nBins = NFFT / 2 + 1;
  const melMin = hzToMel(FMIN), melMax = hzToMel(FMAX);
  const pts = new Array(NMEL + 2);
  for (let i = 0; i < pts.length; i++) {
    const hz = melToHz(melMin + ((melMax - melMin) * i) / (NMEL + 1));
    pts[i] = Math.floor(((NFFT + 1) * hz) / SR);
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
  // Cap the visualised window so the canvas stays a sane width (10 s @ hop 480 ≈ 1000 frames).
  const capSamples = Math.min(pcm.length, SR * 10);
  const maxFrames = Math.min(1000, Math.floor((capSamples - WIN) / HOP) + 1);
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

async function run(id, audio, labels) {
  await ensureLoaded();
  const t0 = performance.now();
  // One forward pass through CLAP's audio + text branches — the same call the pipeline makes, but we
  // keep the intermediate tensors for "See inside".
  const audioInputs = await pipe.processor(audio);
  const textInputs = pipe.tokenizer(labels, { padding: true, truncation: true });
  const output = await pipe.model({ ...textInputs, ...audioInputs });

  const logits = Array.from(output.logits_per_audio.data); // scaled cosine sims (× logit_scale)
  const probs = softmax(logits);

  const audioDims = output.audio_embeds?.dims ?? null;
  const txtDims = output.text_embeds?.dims ?? null;
  let cos = null;
  if (output.audio_embeds && output.text_embeds && audioDims && txtDims) {
    const dim = audioDims[audioDims.length - 1];
    cos = cosines(
      Array.from(output.audio_embeds.data),
      Array.from(output.text_embeds.data),
      dim,
      labels.length,
    );
  } else {
    const scale = Math.exp(pipe.model?.config?.logit_scale_init_value ?? Math.log(100));
    cos = logits.map((l) => l / scale);
  }

  const spectrogram = logMel(audio);
  const ms = Math.round(performance.now() - t0);
  post(
    {
      type: "result",
      id,
      labels,
      probs,
      logits,
      cosines: cos,
      audioDims,
      txtDims,
      spectrogram,
      ms,
      device,
      durationS: audio.length / SR,
    },
    [spectrogram.data.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.device);
    else if (type === "run") await run(e.data.id, e.data.audio, e.data.labels);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
