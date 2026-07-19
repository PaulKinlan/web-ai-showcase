// Biomedical NER worker — inference off the main thread so the control UI stays responsive.
// Model: onnx-community/biomedical-ner-all-ONNX (task: token-classification), WASM, q8.
// ONNX build of d4data/biomedical-ner-all — a DistilBERT fine-tuned on the MACCROBAT clinical-case
// corpus to tag 40+ BIOMEDICAL entity types (Disease_disorder, Sign_symptom, Medication, Dosage,
// Diagnostic_procedure, Lab_value, Biological_structure, Severity, …) rather than the four generic
// PER/ORG/LOC/MISC classes of newswire NER. That domain label set is the whole point.
//
// The pipeline tags each WordPiece token with a BIO label (B-Disease_disorder / I-Disease_disorder …)
// and a confidence, but does NOT return character offsets. So here we:
//   1. run token-classification to get per-token {entity, score, index, word},
//   2. map each token back to a character span in the ORIGINAL text (cursor-walk indexOf), so a
//      highlight lands on the real substring even with casing/punctuation/sub-words,
//   3. merge B-/I- runs (and ## sub-word pieces) of the same type into whole entities with a pooled
//      score.
// Both the raw per-token tags AND the merged entities are returned — the page shows both.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/biomedical-ner-all-ONNX";
let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "token-classification",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Give each returned token a character span in the ORIGINAL text. The pipeline drops "O" tokens, so we
// walk a monotonic cursor and indexOf the token's surface form (## stripped) from that cursor forward.
function locate(tokens, text) {
  const lower = text.toLowerCase();
  let cursor = 0;
  for (const t of tokens) {
    const surface = (t.word.startsWith("##") ? t.word.slice(2) : t.word).toLowerCase();
    const at = lower.indexOf(surface, cursor);
    if (at >= 0) {
      t.start = at;
      t.end = at + surface.length;
      cursor = t.end;
    } else {
      t.start = t.end = null;
    }
  }
  return tokens;
}

// Merge BIO runs + ## continuations of the same entity type into whole-entity spans.
function merge(tokens, text) {
  const spans = [];
  let cur = null;
  for (const t of tokens) {
    const type = t.entity.slice(2); // strip B-/I-
    const isB = t.entity.startsWith("B-");
    const isCont = t.word.startsWith("##");
    const continues = cur && type === cur.type && (!isB || isCont);
    if (continues) {
      cur.tokens.push(t);
      cur.scores.push(t.score);
      if (t.end != null) cur.end = t.end;
    } else {
      if (cur) spans.push(cur);
      cur = { type, tokens: [t], scores: [t.score], start: t.start, end: t.end };
    }
  }
  if (cur) spans.push(cur);
  return spans.map((s) => ({
    type: s.type,
    start: s.start,
    end: s.end,
    text: s.start != null && s.end != null
      ? text.slice(s.start, s.end)
      : s.tokens.map((t) => t.word.replace(/^##/, "")).join(""),
    score: s.scores.reduce((a, b) => a + b, 0) / s.scores.length,
    tokenCount: s.tokens.length,
  }));
}

async function analyse(text) {
  const raw = await pipe(text);
  const tokens = locate(
    raw.map((t) => ({ entity: t.entity, score: t.score, index: t.index, word: t.word })),
    text,
  );
  const spans = merge(tokens, text);
  return { text, tokens, spans };
}

async function ner(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await analyse(text);
  post({ type: "ner", id, ...r, ms: Math.round(performance.now() - t0), device });
}

async function nerMany(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  for (const text of texts) results.push(await analyse(text));
  post({ type: "nerMany", id, results, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "ner") await ner(e.data.id, e.data.text);
    else if (type === "nerMany") await nerMany(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
