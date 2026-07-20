// GTCRN multi-model worker — a real two-model chain, off the main thread:
//   1. GTCRN (raw ONNX Runtime Web) denoises the noisy speech (STFT → streaming GTCRN → iSTFT).
//   2. Whisper (Transformers.js ASR) transcribes BOTH the noisy and the enhanced audio, so you can see
//      denoising improve the transcript (lower word-error-rate against a reference).
// GTCRN: bitsydarel/gtcrn-onnx (~0.5 MB). Whisper: Xenova/whisper-tiny.en (~40 MB). All on-device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const GTCRN_ID = "bitsydarel/gtcrn-onnx";
const GTCRN_URL = `https://huggingface.co/${GTCRN_ID}/resolve/main/gtcrn_simple.onnx`;
const GTCRN_CACHE = "gtcrn-onnx-cache";
const ASR_ID = "Xenova/whisper-tiny.en";
const N = 512, H = 256, F = 257, SR = 16000;

let ort = null, gtcrn = null, mod = null, asr = null, device = "wasm";
const WIN = new Float32Array(N);
for (let i = 0; i < N; i++) WIN[i] = Math.sqrt(0.5 * (1 - Math.cos(2 * Math.PI * i / N)));

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k,
          b = i + k + len / 2,
          vr = re[b] * cwr - im[b] * cwi,
          vi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
  if (inv) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

async function fetchGtcrn() {
  const cache = await caches.open(GTCRN_CACHE);
  let resp = await cache.match(GTCRN_URL);
  if (!resp) {
    const net = await fetch(GTCRN_URL);
    if (!net.ok) throw new Error(`gtcrn fetch (${net.status})`);
    await cache.put(GTCRN_URL, new Response(await net.arrayBuffer()));
    resp = await cache.match(GTCRN_URL);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (gtcrn && asr) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  gtcrn = await ort.InferenceSession.create(await fetchGtcrn(), { executionProviders: ["wasm"] });
  post({ type: "progress", p: { status: "progress", progress: 15 } });
  mod = await import(TRANSFORMERS_URL);
  mod.env.allowLocalModels = false;
  asr = await mod.pipeline("automatic-speech-recognition", ASR_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

function enhance(noisy) {
  const zeros = (d) => new ort.Tensor("float32", new Float32Array(d.reduce((a, b) => a * b, 1)), d);
  let conv = zeros([2, 1, 16, 16, 33]),
    tra = zeros([2, 3, 1, 1, 16]),
    inter = zeros([2, 1, 33, 16]);
  const nframes = Math.max(0, Math.floor((noisy.length - N) / H) + 1);
  const enh = new Float32Array(noisy.length), norm = new Float32Array(noisy.length);
  const re = new Float32Array(N),
    im = new Float32Array(N),
    rr = new Float32Array(N),
    ii = new Float32Array(N),
    mix = new Float32Array(F * 2);
  return (async () => {
    for (let f = 0; f < nframes; f++) {
      const off = f * H;
      for (let i = 0; i < N; i++) {
        re[i] = noisy[off + i] * WIN[i];
        im[i] = 0;
      }
      fft(re, im, false);
      for (let k = 0; k < F; k++) {
        mix[k * 2] = re[k];
        mix[k * 2 + 1] = im[k];
      }
      const r = await gtcrn.run({
        mix: new ort.Tensor("float32", mix, [1, F, 1, 2]),
        conv_cache: conv,
        tra_cache: tra,
        inter_cache: inter,
      });
      conv = r.conv_cache_out;
      tra = r.tra_cache_out;
      inter = r.inter_cache_out;
      const e = r.enh.data;
      rr.fill(0);
      ii.fill(0);
      for (let k = 0; k < F; k++) {
        rr[k] = e[k * 2];
        ii[k] = e[k * 2 + 1];
      }
      for (let k = 1; k < F - 1; k++) {
        rr[N - k] = e[k * 2];
        ii[N - k] = -e[k * 2 + 1];
      }
      fft(rr, ii, true);
      for (let i = 0; i < N; i++) {
        enh[off + i] += rr[i] * WIN[i];
        norm[off + i] += WIN[i] * WIN[i];
      }
    }
    for (let i = 0; i < enh.length; i++) if (norm[i] > 1e-8) enh[i] /= norm[i];
    return enh;
  })();
}

async function run(id, noisy) {
  await ensureLoaded();
  const t0 = performance.now();
  const enh = await enhance(noisy);
  const enhMs = Math.round(performance.now() - t0);
  post({ type: "progress", p: { status: "transcribe", progress: 50 } });
  const noisyTxt = (await asr(noisy.slice())).text.trim();
  post({ type: "progress", p: { status: "transcribe", progress: 80 } });
  const enhTxt = (await asr(enh.slice())).text.trim();
  post({
    type: "result",
    id,
    enhanced: enh,
    noisyTxt,
    enhTxt,
    enhMs,
    ms: Math.round(performance.now() - t0),
    device,
  }, [enh.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.noisy);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
