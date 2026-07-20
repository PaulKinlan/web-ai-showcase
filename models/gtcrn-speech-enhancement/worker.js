// GTCRN speech-enhancement worker — ALL inference AND the STFT/iSTFT DSP off the main thread via raw
// ONNX Runtime Web. GTCRN (Grouped Temporal Convolutional Recurrent Network) is a TINY (~48k params,
// 0.5 MB ONNX) real-time speech denoiser: it takes the complex STFT of noisy speech, frame by frame,
// and predicts the complex STFT of the CLEAN speech, which we invert back to a waveform.
//
// Why raw ORT and not transformers.js: transformers.js has no GTCRN class and no "audio-to-audio"
// pipeline task, so we run the ONNX graph directly with onnxruntime-web and hand-write the DSP a
// pipeline would own: (1) a 512-pt STFT (hop 256, sqrt-Hann window) of the noisy audio, (2) the
// STREAMING GTCRN loop — one frame in, one enhanced frame out, carrying the conv/tra/inter recurrent
// caches across frames, and (3) an iSTFT (overlap-add with the same window, NOLA-normalised) back to
// PCM. Isolated per-worker ORT-web pin (precedent: yolov8-pose / aliked-lightglue): onnxruntime-web is
// pinned HERE only, never in shared lib/webai.js.
//
// Model: bitsydarel/gtcrn-onnx (gtcrn_simple.onnx, fp32, ~0.54 MB). Input "mix" [1,257,1,2] (one STFT
// frame, real/imag) + conv_cache/tra_cache/inter_cache; output "enh" [1,257,1,2] + updated caches.
// Runs at 16 kHz mono. Everything stays on-device: the audio never leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "bitsydarel/gtcrn-onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/gtcrn_simple.onnx`;
const CACHE_NAME = "gtcrn-onnx-cache";
const N = 512, H = 256, F = 257, SR = 16000;

let ort = null, session = null, device = "wasm";
const WIN = new Float32Array(N);
for (let i = 0; i < N; i++) WIN[i] = Math.sqrt(0.5 * (1 - Math.cos(2 * Math.PI * i / N))); // sqrt-Hann

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Iterative radix-2 FFT (in place). inv=true → inverse (divides by N).
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
    const ang = (inv ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const vr = re[b] * cwr - im[b] * cwi, vi = re[b] * cwi + im[b] * cwr;
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

async function fetchModelBytes() {
  const cache = await caches.open(CACHE_NAME);
  let resp = await cache.match(MODEL_URL);
  if (!resp) {
    const net = await fetch(MODEL_URL);
    if (!net.ok || !net.body) throw new Error(`model fetch failed (${net.status})`);
    const total = Number(net.headers.get("content-length")) || 0;
    const reader = net.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) {
        post({ type: "progress", p: { status: "progress", progress: (received / total) * 100 } });
      }
    }
    await cache.put(
      MODEL_URL,
      new Response(new Blob(chunks), { headers: { "content-length": String(received) } }),
    );
    resp = await cache.match(MODEL_URL);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  const bytes = await fetchModelBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  device = "wasm";
  post({ type: "ready", device });
}

function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}

async function enhance(id, noisy) {
  await ensureLoaded();
  const t0 = performance.now();
  const zeros = (d) => new ort.Tensor("float32", new Float32Array(d.reduce((a, b) => a * b, 1)), d);
  let conv = zeros([2, 1, 16, 16, 33]),
    tra = zeros([2, 3, 1, 1, 16]),
    inter = zeros([2, 1, 33, 16]);
  const nframes = Math.max(0, Math.floor((noisy.length - N) / H) + 1);
  const enh = new Float32Array(noisy.length);
  const norm = new Float32Array(noisy.length);
  const re = new Float32Array(N), im = new Float32Array(N);
  const rr = new Float32Array(N), ii = new Float32Array(N);
  const mix = new Float32Array(F * 2);
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
    const r = await session.run({
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
    } // conjugate symmetry
    fft(rr, ii, true);
    for (let i = 0; i < N; i++) {
      enh[off + i] += rr[i] * WIN[i];
      norm[off + i] += WIN[i] * WIN[i];
    }
    if ((f & 31) === 0) {
      post({ type: "progress", p: { status: "enhance", progress: (f / nframes) * 100 } });
    }
  }
  for (let i = 0; i < enh.length; i++) if (norm[i] > 1e-8) enh[i] /= norm[i];
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    enhanced: enh,
    rate: SR,
    frames: nframes,
    inRms: rms(noisy),
    outRms: rms(enh),
    ms,
    device,
  }, [enh.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await enhance(e.data.id, e.data.noisy);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
