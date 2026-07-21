// Educational-quality text scorer worker — inference off the main thread so typing stays smooth.
// Model: onnx-community/fineweb-edu-classifier-ONNX (regression), WASM, q8 (~110 MB). This is HuggingFace's
// FineWeb-Edu classifier — a BERT (Snowflake arctic-embed) encoder with a REGRESSION head that rates a
// passage's EDUCATIONAL VALUE on a 0-5 scale. It's the model HuggingFace used to curate the FineWeb-Edu
// pre-training corpus. DISTINCT from every built text classifier (sentiment/emotion/toxicity/spam/clickbait/
// formality/topic/…): it scores CONTENT QUALITY / educational usefulness, not a category. Apache-2.0.
// Because it's a regression head (num_labels=1, problem_type=regression), we run AutoModel + AutoTokenizer
// directly (the text-classification pipeline would sigmoid the logit) and read the raw score. Nothing leaves
// the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const REPO = "onnx-community/fineweb-edu-classifier-ONNX";

let tokenizer = null;
let model = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const T = await import(TRANSFORMERS_URL);
  tokenizer = await T.AutoTokenizer.from_pretrained(REPO, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await T.AutoModel.from_pretrained(REPO, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

// Score a passage → { raw, score (clamped 0-5), band, ms }. FineWeb-Edu clamps the raw regression output to
// [0,5]; the integer band (0-5) is how the corpus was filtered (>=3 kept as "educational").
const BANDS = [
  "Not educational — promotional, incoherent, or off-topic.",
  "Minimal educational value — some info but mostly non-educational.",
  "Some educational value — touches a topic but is unfocused or shallow.",
  "Good educational value — clear, coherent, on-topic teaching material.",
  "High educational value — well-structured, informative, textbook-like.",
  "Outstanding educational value — thorough, expert, reference-grade.",
];

async function score(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await tokenizer(text, { truncation: true });
  const out = await model(inputs);
  const tensor = out.logits ?? Object.values(out)[0];
  const raw = Number(tensor.data[0]);
  const clamped = Math.max(0, Math.min(5, raw));
  const band = Math.round(clamped);
  post({
    type: "result",
    id,
    raw: +raw.toFixed(2),
    score: +clamped.toFixed(2),
    band,
    bandText: BANDS[band] ?? "",
    ms: Math.round(performance.now() - t0),
    device: "wasm",
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "score") await score(d.id, d.text);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
