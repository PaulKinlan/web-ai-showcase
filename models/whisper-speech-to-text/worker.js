// Whisper ASR worker — runs ALL inference off the main thread so the UI stays responsive.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the pipeline and returns the transcript plus real word/segment timestamps and a token
// count for a tok/s readout.
//
// Model: onnx-community/whisper-base (task: automatic-speech-recognition), WebGPU q8, WASM fallback.
// We import the SHARED loader from lib/webai.js — no invented API. If WebGPU load fails we honestly
// retry on WASM and report which backend actually ran.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

// whisper-base, exported with cross-attentions so Transformers.js can extract real WORD-level
// timestamps (the plain onnx-community/whisper-base build only supports segment timestamps).
const MODEL = "onnx-community/whisper-base_timestamped";
const TASK = "automatic-speech-recognition";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// `navigator.gpu` existing is not enough — headless / locked-down browsers expose the object but can't
// return an adapter. Actually ask for one so we degrade to WASM honestly instead of failing to load.
async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

// Load, preferring WebGPU when a real adapter is available. If it still throws, fall back to WASM and
// say so — never present one backend as another.
async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
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

// Group flat word chunks into readable segments (~sentence boundaries or ~8-word runs).
function toSegments(words) {
  const segs = [];
  let cur = null;
  for (const w of words) {
    if (!cur) cur = { text: "", start: w.timestamp?.[0] ?? null, end: null, words: [] };
    cur.text += w.text;
    cur.end = w.timestamp?.[1] ?? cur.end;
    cur.words.push(w);
    const endsSentence = /[.!?]["')\]]?\s*$/.test(w.text);
    if (endsSentence || cur.words.length >= 12) {
      cur.text = cur.text.trim();
      segs.push(cur);
      cur = null;
    }
  }
  if (cur) {
    cur.text = cur.text.trim();
    segs.push(cur);
  }
  return segs;
}

async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  // Word-level timestamps give us both the karaoke word stream and (grouped) segments.
  const output = await pipe(audio, {
    return_timestamps: "word",
    chunk_length_s: 30,
    stride_length_s: 5,
  });
  const ms = Math.round(performance.now() - t0);

  const text = (output.text || "").trim();
  const words = (output.chunks || []).map((c) => ({
    text: c.text,
    timestamp: c.timestamp,
  }));
  const segments = toSegments(words);

  // Real token count of the decoded text — a genuine number, not a claim.
  let tokens = null;
  try {
    const enc = pipe.tokenizer(text);
    tokens = enc?.input_ids?.dims
      ? enc.input_ids.dims[enc.input_ids.dims.length - 1]
      : (enc?.input_ids?.size ?? null);
  } catch {
    tokens = null;
  }
  const tokPerSec = tokens && ms ? tokens / (ms / 1000) : null;

  post({
    type: "result",
    id,
    text,
    words,
    segments,
    tokens,
    tokPerSec,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(e.data.device);
    } else if (type === "run") {
      await run(e.data.id, e.data.audio, e.data.opts);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
