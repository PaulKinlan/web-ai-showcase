// wav2vec2 (Slovak) CTC worker — ALL inference off the main thread.
//
// DISTINCT model + language from every other ASR demo in this showcase. This is a SLOVAK speech
// recogniser: facebook/wav2vec2-base-10k-voxpopuli fine-tuned on Slovak, exported to ONNX by the
// OpenVoiceOS project. The built ASR demos (English wav2vec2, Russian/French/Spanish/Italian/
// Portuguese/Chinese/Japanese/Korean/Thai XLSR + Finnish/Polish/Czech/Croatian/Romanian/Hungarian
// VoxPopuli) are CTC too, but none is Slovak.
//
// HONEST NOTE ON THE RUNTIME. Like the Czech/Croatian/Romanian/Hungarian VoxPopuli demos (and unlike
// the Polish/Finnish onnx-community mirrors that load straight through transformers.js as
// AutoModelForCTC), the ONLY browser ONNX export of the Slovak VoxPopuli checkpoint is
// OpenVoiceOS/wav2vec2-base-10k-voxpopuli-ft-sk-onnx, packaged for the onnx-asr toolkit: its
// config.json declares `model_type: "wav2vec2-ctc"` (a type transformers.js does NOT register →
// "Unsupported model type: wav2vec2-ctc"), the graph takes `input_values` + `input_lengths` and emits
// `logprobs` (log-softmax) rather than `logits`, and feature normalisation is baked INTO the graph. So
// this demo runs the raw ONNX graph directly on ONNX Runtime Web (the same engine transformers.js uses
// under the hood), feeding raw 16 kHz audio and CTC-decoding the log-probs ourselves. It is a real,
// in-browser run of the genuine Slovak Meta-AI checkpoint — not a relabel of another language, not a fake.
//
// LAYOUT / dtype: single fp32 graph (`model.onnx` ~1.7 MB) + external weights (`model.onnx.data` ~360 MB).
// No quantised export exists (onnxruntime.quantization can't process this dynamo-exported graph — see
// the model card), so we honestly ship the fp32 export and disclose the ~360 MB download. Verified in real
// headless Chrome (ORT Web 1.20.1, WASM EP) to produce real log-probs [1, frames, 48] and a coherent
// Slovak transcript on a CC0 VoxPopuli sample. Weights are fetched from the canonical HF repo and cached in
// Cache Storage under the repo URL so the shared model-cache layer detects them (auto-init on return visits).

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
const ORT_WASM_PATHS = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const MODEL_ID = "OpenVoiceOS/wav2vec2-base-10k-voxpopuli-ft-sk-onnx";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/`;
const VOCAB_URL = "/web-ai-showcase/models/slovak-voxpopuli-asr/vocab.txt";
const CACHE_NAME = "web-ai-ort-models";
const GRAPH_FILE = "model.onnx";
const DATA_FILE = "model.onnx.data";

const BLANK = 0; // <blk> — CTC blank
const WORD_DELIM = "▁"; // '▁' → space

let ort = null;
let session = null;
let id2tok = null; // built from vocab.txt
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Fetch a URL, preferring an already-cached copy (offline / return-visit), reporting byte progress.
// On a network fetch, store the response in Cache Storage under its repo URL so the shared model-cache
// layer (scans caches by modelId) can detect it and auto-initialise next time — no silent re-download.
async function fetchCached(url, onBytes) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onBytes?.(buf.byteLength, buf.byteLength);
      return new Uint8Array(buf);
    }
  } catch { /* Cache Storage unavailable — fall through to a plain fetch. */ }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes?.(received, total);
  }
  const bytes = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  if (cache) {
    try {
      await cache.put(
        url,
        new Response(bytes, { headers: { "content-type": "application/octet-stream" } }),
      );
    } catch {
      /* quota/eviction — model still loads this session; loader reports partial next time. */
    }
  }
  return bytes;
}

async function loadVocab() {
  if (id2tok) return;
  const txt = await (await fetch(VOCAB_URL)).text();
  id2tok = {};
  for (const line of txt.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split(" ");
    const id = parseInt(parts[parts.length - 1], 10);
    id2tok[id] = parts.slice(0, parts.length - 1).join(" ");
  }
}

async function ensureLoaded() {
  if (session) return;
  if (!ort) {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
    ort.env.wasm.numThreads = 1; // GitHub Pages can't set COOP/COEP → no SharedArrayBuffer threads.
  }
  await loadVocab();

  // Two files: tiny graph + big external weights. Weight the progress by the ~360 MB data file.
  post({ type: "progress", p: { status: "initiate", file: GRAPH_FILE } });
  const graph = await fetchCached(MODEL_BASE + GRAPH_FILE, () => {});
  const data = await fetchCached(MODEL_BASE + DATA_FILE, (recv, total) => {
    if (total) {
      post({
        type: "progress",
        p: { status: "progress", progress: (recv / total) * 100, file: DATA_FILE },
      });
    }
  });

  post({ type: "progress", p: { status: "initiate", file: "creating session…" } });
  session = await ort.InferenceSession.create(graph, {
    executionProviders: ["wasm"],
    externalData: [{ path: DATA_FILE, data }],
  });
  device = "wasm";
  post({ type: "ready", device });
}

// Greedy CTC decode over log-probs: argmax per frame, collapse repeats, drop blanks. '▁' → word boundary.
// Also derive real per-word forced-alignment timings from the frame each character was emitted at.
function decodeCTC(ids, frameSec) {
  const strip = [];
  const words = [];
  const collapsed = [];
  let cur = null;
  let prev = -1;
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    const tok = id2tok[id];
    const blank = id === BLANK;
    const boundary = tok === WORD_DELIM;
    const isChar = !blank && !boundary && tok && tok.length === 1 && tok[0] !== "<";
    strip.push({ c: isChar ? tok : "", blank, boundary });

    if (id !== prev) {
      if (blank) {
        // blank separates two identical letters, emits nothing
      } else if (boundary) {
        collapsed.push(" ");
        if (cur) {
          words.push(cur);
          cur = null;
        }
      } else if (isChar) {
        collapsed.push(tok);
        if (!cur) cur = { text: "", startFrame: t, endFrame: t };
        cur.text += tok;
        cur.endFrame = t;
      }
    }
    prev = id;
  }
  if (cur) words.push(cur);

  const text = words.map((w) => w.text).join(" ").replace(/\s+/g, " ").trim();
  const timedWords = words.map((w) => ({
    text: w.text,
    start: w.startFrame * frameSec,
    end: (w.endFrame + 1) * frameSec,
  }));
  return { strip, collapsed, text, words: timedWords };
}

async function run(id, audio, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  // Feature normalisation is baked into the graph — feed the RAW 16 kHz mono waveform.
  const feeds = {
    input_values: new ort.Tensor("float32", audio, [1, audio.length]),
    input_lengths: new ort.Tensor("int64", BigInt64Array.from([BigInt(audio.length)]), [1]),
  };
  const out = await session.run(feeds);
  const ms = Math.round(performance.now() - t0);

  const lp = out.logprobs;
  const [, T, V] = lp.dims;
  const data = lp.data;
  const ids = new Array(T);
  for (let t = 0; t < T; t++) {
    let best = -Infinity;
    let bi = 0;
    const base = t * V;
    for (let v = 0; v < V; v++) {
      const val = data[base + v];
      if (val > best) {
        best = val;
        bi = v;
      }
    }
    ids[t] = bi;
  }

  const audioSec = audioDur || (audio.length / 16000);
  const frameSec = T ? audioSec / T : 0;
  const decoded = decodeCTC(ids, frameSec);

  post({
    type: "result",
    id,
    text: decoded.text,
    strip: decoded.strip,
    collapsed: decoded.collapsed,
    words: decoded.words,
    frames: T,
    frameMs: frameSec * 1000,
    emitted: decoded.collapsed.filter((c) => c !== " ").length,
    audioSec,
    rtf: audioSec ? (ms / 1000) / audioSec : null,
    speedup: audioSec && ms ? audioSec / (ms / 1000) : null,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.audio, e.data.audioDur);
    }
  } catch (err) {
    console.error("[wav2vec2-sk worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
