// Music transcription worker — turns audio into per-pitch note activations (a piano roll) entirely
// on-device via raw ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js has no music-transcription / audio-to-MIDI task,
// so we run the model's ONNX graph directly with onnxruntime-web. This is the isolated per-worker ORT-web
// escape hatch (like models/raft-optical-flow/worker.js) — onnxruntime-web is pinned HERE only.
//
// Model: Spotify Basic Pitch (nmp). Basic Pitch is published by Spotify under Apache-2.0
// (github.com/spotify/basic-pitch, "A lightweight yet powerful audio-to-MIDI converter"). Apache-2.0
// expressly permits redistribution, so the weights remain Apache-2.0 wherever mirrored; we fetch a
// faithful ONNX conversion and document that provenance. The graph includes Basic Pitch's harmonic-CQT
// frontend, so it takes RAW audio: input [1, 43844, 1] float32 mono at 22050 Hz (one ~1.99 s window) →
// three outputs per window: NOTE frames [1, 172, 88] (sustained per-key activation, MIDI 21–108), ONSET
// [1, 172, 88] (note starts), and a fine CONTOUR [1, 172, 264] (3 bins/semitone). DISTINCT from the built
// monophonic pitch-detection (CREPE): Basic Pitch is POLYPHONIC — many simultaneous notes. Nothing leaves
// the tab.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0, WASM EP, no GPU): a pure 440 Hz
// tone peaks at MIDI 69 (A4) and 261.6 Hz at MIDI 60 (C4); a synthesized C-major chord progression and a
// two-hand duet transcribe to their intended notes with up to 4–5 simultaneous keys active.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "spotify/basic-pitch";
// Apache-2.0 Basic Pitch weights (nmp.onnx), fetched from a mirror; license travels with the model.
const MODEL_URL =
  "https://huggingface.co/shethjenil/Audio2Midi_Models/resolve/main/basicpitch/nmp.onnx";
const CACHE_NAME = "basic-pitch-onnx-cache";

const SR = 22050; // Basic Pitch's native sample rate
const WIN = 43844; // AUDIO_N_SAMPLES — one model window (~1.99 s)
const FRAMES = 172; // annotation frames per window
export const N_KEYS = 88; // MIDI 21..108
export const MIDI_LOW = 21;
export const FRAME_SEC = (WIN / SR) / FRAMES; // ~11.6 ms/frame

let ort = null;
let session = null;
let inName = null;
let noteName = null;
let onsetName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/spotify/basic-pitch") sees
// it → auto-init on a returning visit, honest Download on first visit, and the "clear cached model"
// control all work. Streams download progress. The cache key carries the model id path so the scan matches.
async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${MODEL_ID}/resolve/main/nmp.onnx`; // scan-visible key (model id path)
  const hit = await cache.match(key);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const net = await fetch(url);
  if (!net.ok || !net.body) throw new Error(`fetch failed (${net.status}) for ${url}`);
  const total = Number(net.headers.get("content-length")) || 0;
  const reader = net.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onChunk?.(received, total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  await cache.put(
    key,
    new Response(buf, {
      headers: { "content-length": String(received), "content-type": "application/octet-stream" },
    }),
  );
  return buf;
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const cache = await caches.open(CACHE_NAME);
  const modelBytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  inName = session.inputNames[0];
  // ORT's metadata API doesn't expose dims until a run, so probe once with silence to read output shapes:
  // two outputs are [.,.,88] (note + onset), one is [.,.,264] (contour). Among the 88-wide pair the graph
  // emits the sustained NOTE head first then the sparse ONSET head — validated in headless Chrome, where a
  // 2 s sustained tone lit the first 88-wide output across its whole duration (note) and the second only at
  // the start (onset).
  const probe = await session.run({
    [inName]: new ort.Tensor("float32", new Float32Array(WIN), [1, WIN, 1]),
  });
  const k88 = session.outputNames.filter((n) => probe[n].dims[2] === N_KEYS);
  noteName = k88[0];
  onsetName = k88[1];
  post({ type: "ready", device: "wasm" });
}

// Run one 43844-sample window → { note: Float32Array(FRAMES*88), onset: Float32Array(FRAMES*88) }.
async function runWindow(win) {
  const out = await session.run({ [inName]: new ort.Tensor("float32", win, [1, WIN, 1]) });
  return { note: out[noteName].data, onset: out[onsetName].data };
}

// Transcribe arbitrary-length mono audio (already 22050 Hz) → concatenated piano-roll matrices.
async function transcribe(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  const nWin = Math.max(1, Math.ceil(audio.length / WIN));
  const note = new Float32Array(nWin * FRAMES * N_KEYS);
  const onset = new Float32Array(nWin * FRAMES * N_KEYS);
  for (let w = 0; w < nWin; w++) {
    const win = new Float32Array(WIN);
    const start = w * WIN;
    for (let i = 0; i < WIN && start + i < audio.length; i++) win[i] = audio[start + i];
    const r = await runWindow(win);
    note.set(r.note, w * FRAMES * N_KEYS);
    onset.set(r.onset, w * FRAMES * N_KEYS);
    post({ type: "progress", p: { status: "progress", progress: ((w + 1) / nWin) * 100 } });
  }
  post(
    {
      type: "roll",
      id,
      note,
      onset,
      frames: nWin * FRAMES,
      keys: N_KEYS,
      midiLow: MIDI_LOW,
      frameSec: FRAME_SEC,
      ms: Math.round(performance.now() - t0),
    },
    [note.buffer, onset.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "transcribe") await transcribe(e.data.id, e.data.audio);
  } catch (err) {
    console.error("[basic-pitch worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
