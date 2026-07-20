// Stance-detection worker — target-conditioned stance classification off the main thread.
// Model: Factiverse/factiverse_stance_detection_ort_quantized (task: text-classification), an
// XLM-RoBERTa sequence classifier fine-tuned for fact-checking stance, exported to ONNX (q8), WASM.
//
// This is a DEDICATED stance head (2 classes: SUPPORTS vs REFUTES), NOT a relabelled NLI or sentiment
// model. It answers: given a CLAIM/target and a piece of TEXT, does the text support the claim or
// refute it? The model is target-conditioned — the same text flips stance when the claim changes.
//
// IMPORTANT (verified): the transformers.js `text_pair` path yields a constant/degenerate output for
// this checkpoint (the pair isn't tokenised into the input). The correct, verified formulation is a
// SINGLE string joining claim + evidence with the XLM-R sentence separator "</s></s>". We build that
// here. The ONNX also lives at the repo ROOT (not an onnx/ subfolder), so we pass subfolder:"".
//
// Operations:
//   run   → score one (claim, text) pair → supports + refutes probabilities (real 2-class softmax).
//   batch → score many texts against one claim in a single padded forward pass (for argument mapping).

import { pickDevice } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Factiverse/factiverse_stance_detection_ort_quantized";
const SEP = " </s></s> "; // XLM-RoBERTa pair separator — join claim + text into one sequence.

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Build the verified single-string input the checkpoint actually reads.
function buildInput(target, text) {
  return String(target ?? "").trim() + SEP + String(text ?? "").trim();
}

async function ensureLoaded() {
  if (pipe) return;
  // Direct pipeline import: this repo's ONNX is at the root and needs subfolder:"" + dtype q8, which
  // the shared loadPipeline helper does not expose. Everything else mirrors it (env + device probe).
  const { pipeline, env } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5"
  );
  env.allowLocalModels = false;
  device = await pickDevice("wasm");
  pipe = await pipeline("text-classification", MODEL_ID, {
    device,
    dtype: "q8",
    subfolder: "",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

// Recover the SUPPORTS probability from a top_k:2 result (order can vary; softmax so the pair sums to 1).
function supportsProb(rows) {
  const arr = Array.isArray(rows) ? rows : [rows];
  const s = arr.find((r) => r.label === "SUPPORTS");
  if (s) return s.score;
  const r = arr.find((r) => r.label === "REFUTES");
  return r ? 1 - r.score : 0.5;
}

// A low-confidence softmax (near 50/50) means the text doesn't clearly take a side toward the claim.
// The model has NO dedicated neutral class, so we derive an honest "unclear" band from the margin.
function stanceLabel(supports) {
  if (Math.abs(supports - 0.5) < 0.15) return "unclear";
  return supports >= 0.5 ? "SUPPORTS" : "REFUTES";
}

async function classify(id, target, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(buildInput(target, text), { top_k: 2 });
  const supports = supportsProb(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    target,
    text,
    supports,
    refutes: 1 - supports,
    label: stanceLabel(supports),
    scores: out,
    ms,
    device,
  });
}

async function classifyBatch(id, target, texts) {
  await ensureLoaded();
  if (!texts.length) {
    post({ type: "batch", id, target, texts: [], results: [], ms: 0, device });
    return;
  }
  const t0 = performance.now();
  const inputs = texts.map((t) => buildInput(target, t));
  const out = await pipe(inputs, { top_k: 2 });
  const results = texts.map((_, i) => {
    const supports = supportsProb(out[i]);
    return { supports, refutes: 1 - supports, label: stanceLabel(supports) };
  });
  const ms = Math.round(performance.now() - t0);
  post({ type: "batch", id, target, texts, results, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.target, e.data.text);
    else if (type === "batch") await classifyBatch(e.data.id, e.data.target, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
