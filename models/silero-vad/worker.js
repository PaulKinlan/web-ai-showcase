// Silero VAD worker — ALL inference off the main thread.
//
// This model has NO transformers.js pipeline (transformers.js registers no voice-activity-detection
// task / Silero class). So — per the version-pin / custom-worker precedent in CLAUDE.md invariant 9 —
// we drive inference with onnxruntime-web DIRECTLY, isolated to THIS worker. We do NOT touch the shared
// lib/webai.js (which is transformers.js only). We DO reuse the shared cache/auto-init layer: the model
// blob is stored in Cache Storage under its real HF URL, so lib/model-cache.js (which scans Cache
// Storage by modelId) detects it and createModelLoader auto-initialises on a return visit — exactly the
// same auto-init contract every other page honours.
//
// Model: onnx-community/silero-vad (Silero VAD v5). ~2.2 MB fp32 ONNX. Stateful:
//   inputs  = input [1, 512] float32 (one 512-sample @16 kHz frame) · state [2, 1, 128] float32 (LSTM
//             hidden/cell packed into one tensor in v5) · sr int64 scalar (16000)
//   outputs = output [1, 1] float32 (speech probability for the frame) · stateN [2, 1, 128] (next state)
// We slide 512-sample frames across the clip, carrying `state` forward, to get a per-frame speech
// probability curve, then apply a threshold + hysteresis to derive speech segments.

const ORT_VERSION = "1.20.1";
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;
const MODEL_ID = "onnx-community/silero-vad";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model.onnx`;
const ONNX_CACHE = "onnx-community-models"; // a real Cache Storage cache model-cache.js can scan
const SR = 16000;
const FRAME = 512; // Silero v5 requires exactly 512 samples per frame at 16 kHz
const STATE_SHAPE = [2, 1, 128];

let ort = null;
let session = null;
let streamState = null; // carried LSTM state for the live streaming path

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
    // Absent — download with real progress, then store the blob under its real HF URL.
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
        onProgress?.({ status: "progress", progress: (got / total) * 100, file: "model.onnx" });
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
    // No streamable body — fall back to a plain buffered fetch (still cache it).
    const buf = await net.arrayBuffer();
    await cache.put(MODEL_URL, new Response(buf.slice(0)));
    return buf;
  } catch (err) {
    // Cache Storage unavailable (rare) — fall back to a direct fetch so the demo still runs.
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
    ort.env.wasm.numThreads = 1; // GitHub Pages has no COOP/COEP ⇒ no SharedArrayBuffer; stay single-threaded
  }
  const bytes = await fetchModelBytes(onProgress);
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm (onnxruntime-web)" });
}

function zeroState() {
  return new ort.Tensor("float32", new Float32Array(2 * 1 * 128), STATE_SHAPE);
}

// Run one 512-sample frame → { prob, nextState }.
async function inferFrame(frame, state) {
  const input = new ort.Tensor("float32", frame, [1, FRAME]);
  const sr = new ort.Tensor("int64", new BigInt64Array([BigInt(SR)]), []);
  const out = await session.run({ input, state, sr });
  return { prob: out.output.data[0], nextState: out.stateN };
}

// Slide 512-sample frames across a whole clip → per-frame probabilities.
async function probsForClip(pcm) {
  const nFrames = Math.max(1, Math.floor(pcm.length / FRAME));
  const probs = new Float32Array(nFrames);
  let state = zeroState();
  for (let f = 0; f < nFrames; f++) {
    const frame = pcm.subarray(f * FRAME, f * FRAME + FRAME);
    // subarray of the exact frame length; the model needs exactly 512 samples.
    const buf = frame.length === FRAME ? frame : (() => {
      const p = new Float32Array(FRAME);
      p.set(frame);
      return p;
    })();
    const { prob, nextState } = await inferFrame(buf, state);
    probs[f] = prob;
    state = nextState;
  }
  return probs;
}

// Threshold + hysteresis → speech segments (mirrors the Silero VADIterator defaults).
function segmentsFromProbs(probs, frameSec, opts = {}) {
  const onT = opts.threshold ?? 0.5;
  const offT = opts.negThreshold ?? Math.max(0.15, onT - 0.15);
  const minSpeechMs = opts.minSpeechMs ?? 250;
  const minSilenceMs = opts.minSilenceMs ?? 100;
  const padMs = opts.speechPadMs ?? 30;
  const minSpeech = minSpeechMs / 1000;
  const minSilence = minSilenceMs / 1000;
  const pad = padMs / 1000;

  const segs = [];
  let inSpeech = false;
  let start = 0;
  let silenceRun = 0;
  for (let i = 0; i < probs.length; i++) {
    const t = i * frameSec;
    if (!inSpeech) {
      if (probs[i] >= onT) {
        inSpeech = true;
        start = t;
        silenceRun = 0;
      }
    } else {
      if (probs[i] < offT) {
        silenceRun += frameSec;
        if (silenceRun >= minSilence) {
          const end = t - silenceRun + frameSec;
          if (end - start >= minSpeech) segs.push({ start, end });
          inSpeech = false;
          silenceRun = 0;
        }
      } else {
        silenceRun = 0;
      }
    }
  }
  if (inSpeech) {
    const end = probs.length * frameSec;
    if (end - start >= minSpeech) segs.push({ start, end });
  }
  // Apply symmetric padding and clamp to the clip.
  const dur = probs.length * frameSec;
  return segs.map((s) => ({
    start: Math.max(0, s.start - pad),
    end: Math.min(dur, s.end + pad),
  }));
}

async function runClip(id, pcm, opts) {
  await ensureLoaded();
  const t0 = performance.now();
  const probs = await probsForClip(pcm);
  const ms = Math.round(performance.now() - t0);
  const frameSec = FRAME / SR;
  const segments = segmentsFromProbs(probs, frameSec, opts);
  const speechFrames = probs.reduce((a, p) => a + (p >= (opts?.threshold ?? 0.5) ? 1 : 0), 0);
  post(
    {
      type: "result",
      id,
      probs,
      frameSec,
      segments,
      ms,
      durationS: pcm.length / SR,
      speechRatio: probs.length ? speechFrames / probs.length : 0,
      device: "wasm (onnxruntime-web)",
    },
    [probs.buffer],
  );
}

// Streaming path for the live talk-meter: process a chunk of samples, carrying state across calls.
async function streamReset() {
  await ensureLoaded();
  streamState = zeroState();
}

async function streamChunk(id, pcm) {
  await ensureLoaded();
  if (!streamState) streamState = zeroState();
  const nFrames = Math.floor(pcm.length / FRAME);
  const probs = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const frame = pcm.subarray(f * FRAME, f * FRAME + FRAME);
    const buf = frame.length === FRAME ? frame : (() => {
      const p = new Float32Array(FRAME);
      p.set(frame);
      return p;
    })();
    const { prob, nextState } = await inferFrame(buf, streamState);
    probs[f] = prob;
    streamState = nextState;
  }
  const last = nFrames ? probs[nFrames - 1] : 0;
  post({ type: "stream", id, probs, last, frameSec: FRAME / SR }, [probs.buffer]);
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") {
      await ensureLoaded((p) => post({ type: "progress", p }));
    } else if (d.type === "run") {
      await runClip(d.id, d.pcm, d.opts);
    } else if (d.type === "stream-reset") {
      await streamReset();
      post({ type: "stream-ready", id: d.id });
    } else if (d.type === "stream-chunk") {
      await streamChunk(d.id, d.pcm);
    }
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
