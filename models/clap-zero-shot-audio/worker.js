// CLAP worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 48 kHz mono Float32Array and transfers it here. The
// worker runs LAION-CLAP (audio tower: HTSAT, text tower: RoBERTa) via the combined ONNX graph and
// returns real per-label probabilities, the raw logits and cosine similarities behind them, the
// 512-d audio/text embeddings, the 64-band log-mel spectrogram CLAP actually reads, and the latency
// + backend used.
//
// Model: Xenova/clap-htsat-unfused (ONNX export of laion/clap-htsat-unfused), q8, WASM (WebGPU when
// a real adapter exists). We call ClapModel exactly the way the zero-shot-audio-classification
// pipeline does internally — tokenize "This is a sound of {label}." sentences, run the combined
// model, softmax(logits_per_audio) — so the scores ARE the pipeline's scores, with the raw
// ingredients exposed. No invented API.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

const MODEL = "Xenova/clap-htsat-unfused";
const SR = 48000; // ClapFeatureExtractor sampling_rate

let model = null, processor = null, tokenizer = null, softmaxFn = null;
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

async function loadOn(want, T) {
  const { ClapModel, AutoProcessor, AutoTokenizer } = T;
  model = await ClapModel.from_pretrained(MODEL, {
    device: want,
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained(MODEL);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  device = want;
}

async function ensureLoaded(preferred) {
  if (model) return;
  const T = await import(TRANSFORMERS_URL);
  T.env.allowLocalModels = false;
  softmaxFn = T.softmax;
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  try {
    await loadOn(want, T);
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      await loadOn("wasm", T);
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// ---- log-mel spectrogram (the image CLAP's audio tower reads) --------------------------------------
// ClapFeatureExtractor: 48 kHz, n_fft 1024, hop 480 (10 ms), 64 mel bands over 50–14000 Hz. We compute
// the same shape here so the "see inside" surface shows the real input representation.
const NFFT = 1024, HOP = 480, NMEL = 64, FMIN = 50, FMAX = 14000;

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

let _hann = null, _melFb = null;
function hann() {
  if (_hann) return _hann;
  _hann = new Float32Array(NFFT);
  for (let i = 0; i < NFFT; i++) _hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1));
  return _hann;
}
function melFilterbank() {
  if (_melFb) return _melFb;
  const nBins = NFFT / 2 + 1;
  const melMin = hzToMel(FMIN), melMax = hzToMel(FMAX);
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
  const frames = Math.max(1, Math.min(1000, Math.floor((pcm.length - NFFT) / HOP) + 1));
  const out = new Float32Array(frames * NMEL);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  const nBins = NFFT / 2 + 1;
  const power = new Float64Array(nBins);
  let lo = Infinity, hi = -Infinity;
  for (let f = 0; f < frames; f++) {
    re.fill(0);
    im.fill(0);
    const off = f * HOP;
    for (let i = 0; i < NFFT; i++) re[i] = (pcm[off + i] ?? 0) * w[i];
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

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/** One combined forward pass: texts × one audio clip. Embeddings come back L2-normalised. */
async function forward(texts, audio) {
  const text_inputs = tokenizer(texts, { padding: true, truncation: true });
  const audio_inputs = await processor(audio);
  return await model({ ...text_inputs, ...audio_inputs });
}

/** Zero-shot classify: exactly the pipeline's mechanism, with the raw ingredients kept. */
async function classify(id, audio, labels, template, wantSpectrogram) {
  const texts = labels.map((l) => (template || "This is a sound of {}.").replace("{}", l));
  const t0 = performance.now();
  const out = await forward(texts, audio);
  const ms = Math.round(performance.now() - t0);
  const logits = Array.from(out.logits_per_audio.data);
  const probs = softmaxFn(logits);
  const audioEmbed = new Float32Array(out.audio_embeds.data);
  const dims = out.text_embeds.dims;
  const textEmbeds = [];
  const cosines = [];
  for (let i = 0; i < dims[0]; i++) {
    const e = new Float32Array(out.text_embeds.data.slice(i * dims[1], (i + 1) * dims[1]));
    textEmbeds.push(e);
    cosines.push(dot(audioEmbed, e));
  }
  const results = labels
    .map((label, i) => ({ label, sentence: texts[i], score: probs[i], logit: logits[i], cosine: cosines[i] }))
    .sort((a, b) => b.score - a.score);
  const msg = {
    type: "result", id, results, ms, device,
    durationS: audio.length / SR,
    audioEmbed,
    textEmbeds,
  };
  const transfer = [audioEmbed.buffer, ...textEmbeds.map((e) => e.buffer)];
  if (wantSpectrogram) {
    msg.spectrogram = logMel(audio);
    transfer.push(msg.spectrogram.data.buffer);
  }
  post(msg, transfer);
}

/** Embed a clip into CLAP's shared 512-d space (a minimal dummy text rides along — the combined
 *  graph needs both inputs; its text output is discarded). */
async function embedAudio(id, audio) {
  const t0 = performance.now();
  const out = await forward([""], audio);
  const audioEmbed = new Float32Array(out.audio_embeds.data);
  post(
    { type: "result", id, audioEmbed, ms: Math.round(performance.now() - t0), device, durationS: audio.length / SR },
    [audioEmbed.buffer],
  );
}

/** Embed free-text queries into the same space (a second of silence rides along as the dummy). */
async function embedTexts(id, texts) {
  const t0 = performance.now();
  const out = await forward(texts, new Float32Array(SR));
  const dims = out.text_embeds.dims;
  const textEmbeds = [];
  for (let i = 0; i < dims[0]; i++) {
    textEmbeds.push(new Float32Array(out.text_embeds.data.slice(i * dims[1], (i + 1) * dims[1])));
  }
  post(
    { type: "result", id, textEmbeds, ms: Math.round(performance.now() - t0), device },
    textEmbeds.map((e) => e.buffer),
  );
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.device);
    else if (d.type === "classify") {
      await ensureLoaded();
      await classify(d.id, d.audio, d.labels, d.template, d.wantSpectrogram !== false);
    } else if (d.type === "embed-audio") {
      await ensureLoaded();
      await embedAudio(d.id, d.audio);
    } else if (d.type === "embed-texts") {
      await ensureLoaded();
      await embedTexts(d.id, d.texts);
    }
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
