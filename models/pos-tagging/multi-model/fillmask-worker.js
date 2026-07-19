// CamemBERT fill-mask worker — the second model in the POS → fill-mask composition. Off the main thread.
// Model: Xenova/camembert-base (task: fill-mask), WASM, q8. Language-consistent with the French POS
// tagger, so predictions are real, grammatical French words for a masked slot.
//
// The page sends text with a "[MASK]" placeholder; we swap it for the tokenizer's real mask token
// (<mask> for CamemBERT) and return the top-k predictions {token_str, score} with the ▁ marker stripped.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/camembert-base";
let pipe = null;
let device = "wasm";
let maskToken = "<mask>";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "fill-mask",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  maskToken = pipe.tokenizer.mask_token ?? "<mask>";
  post({ type: "ready", device });
}

async function fill(id, text, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const masked = text.replace(/\[MASK\]/g, maskToken);
  const out = await pipe(masked, { top_k: topk ?? 6 });
  // For a single mask, transformers.js returns a flat array of {score, token, token_str, sequence}.
  const preds = (Array.isArray(out[0]) ? out[0] : out).map((p) => ({
    token_str: (p.token_str ?? "").replace(/▁/g, "").trim(),
    score: p.score,
  })).filter((p) => p.token_str);
  post({ type: "result", id, predictions: preds, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await fill(e.data.id, e.data.text, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
