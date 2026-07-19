// Multi-model worker for the formality "detect → paraphrase → re-score" composition. Loads TWO real
// models and chains them off the main thread:
//   1. Deepchecks/roberta_base_formality_ranker_onnx (text-classification) → read the register.
//   2. Felladrin/onnx-chatgpt_paraphraser_on_T5_base (text2text-generation, q8) → reword it N ways.
// Then it re-scores every paraphrase with model 1 and returns the one that best shifts the register
// toward the requested target. A genuine formality-check → paraphrase → formality-check loop, all local.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const FORMALITY_ID = "Deepchecks/roberta_base_formality_ranker_onnx";
const PARA_ID = "Felladrin/onnx-chatgpt_paraphraser_on_T5_base";

let formalPipe = null;
let paraPipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (formalPipe && paraPipe) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  if (!formalPipe) {
    formalPipe = await pipeline("text-classification", FORMALITY_ID, {
      subfolder: "",
      model_file_name: "model_optimized",
      dtype: "fp32",
      progress_callback: (p) => post({ type: "progress", stage: "formality", p }),
    });
  }
  if (!paraPipe) {
    paraPipe = await pipeline("text2text-generation", PARA_ID, {
      device: "wasm",
      dtype: "q8",
      progress_callback: (p) => post({ type: "progress", stage: "paraphrase", p }),
    });
  }
  post({ type: "ready", device });
}

function formalProb(row) {
  const arr = Array.isArray(row) ? row : [row];
  const f = arr.find((r) => r.label === "formal");
  if (f) return f.score;
  const i = arr.find((r) => r.label === "informal");
  return i ? 1 - i.score : 0.5;
}

async function scoreFormal(text) {
  const out = await formalPipe(text, { top_k: 2 });
  return formalProb(out);
}

async function shift(id, text, target, n) {
  await ensureLoaded();
  const t0 = performance.now();
  // Stage 1 — read the original register.
  const origFormal = await scoreFormal(text);
  post({
    type: "detected",
    id,
    formal: origFormal,
    label: origFormal >= 0.5 ? "formal" : "informal",
    target,
    detectMs: Math.round(performance.now() - t0),
    device,
  });
  // Stage 2 — generate N diverse paraphrases (sampling; the input sentence is the prompt, no prefix).
  const t1 = performance.now();
  const variants = [];
  for (let i = 0; i < n; i++) {
    const out = await paraPipe(text, {
      max_new_tokens: 64,
      do_sample: true,
      temperature: 1.1,
      top_k: 50,
      top_p: 0.95,
    });
    const gen = (Array.isArray(out) ? out[0] : out)?.generated_text?.trim() ?? "";
    if (gen && !variants.some((v) => v.text === gen)) variants.push({ text: gen });
    post({ type: "progress-gen", id, done: i + 1, total: n });
  }
  // Stage 3 — re-score every paraphrase's register.
  for (const v of variants) v.formal = await scoreFormal(v.text);
  // Rank by how well each moves toward the target register.
  const wantFormal = target === "formal";
  variants.sort((a, b) => (wantFormal ? b.formal - a.formal : a.formal - b.formal));
  post({
    type: "shifted",
    id,
    origFormal,
    target,
    variants,
    best: variants[0] || null,
    genMs: Math.round(performance.now() - t1),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "shift") await shift(e.data.id, e.data.text, e.data.target, e.data.n || 4);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
