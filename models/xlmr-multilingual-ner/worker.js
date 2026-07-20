// XLM-RoBERTa multilingual NER worker — inference off the main thread so the control UI stays responsive.
// Model: tjruesch/xlm-roberta-base-ner-hrl-onnx (task: token-classification), WASM, q8.
// ONNX export of the canonical Davlan/xlm-roberta-base-ner-hrl.
//
// What makes THIS demo distinct from the multilingual BERT (mBERT) NER page: the backbone is
// XLM-RoBERTa-base — a RoBERTa-family encoder pretrained on 2.5TB of CommonCrawl across 100 languages
// with a 250k SentencePiece BPE vocab (not mBERT's WordPiece Wikipedia vocab). It's fine-tuned on the
// same HRL corpus (PER/ORG/LOC/DATE across ten high-resource languages) but, thanks to XLM-R's stronger
// cross-lingual alignment, it also TRANSFERS to languages it never saw entities in during fine-tuning
// (Turkish, Swedish, Japanese, …). The pipeline tags each SentencePiece token with a BIO label and a
// confidence, but returns NO character offsets. So here we:
//   1. run token-classification to get per-token {entity, score, index, word},
//   2. map each token back to a character span in the ORIGINAL text (cursor-walk indexOf) so the
//      highlight lands on the real substring across Latin / CJK / RTL scripts,
//   3. merge B-/I- runs of the same type into whole entities with a pooled score (pure BIO merge — the
//      right thing for SentencePiece, where sub-word continuations carry no ## marker).
// Both the raw per-token tags AND the merged entities are returned — the page shows both. The raw tags
// visibly expose SentencePiece splitting (e.g. "Schäuble" → Schä · u · ble), the see-inside distinction
// from mBERT's ## WordPiece.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "tjruesch/xlm-roberta-base-ner-hrl-onnx";
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
// walk a monotonic cursor and indexOf the token's surface form from that cursor forward. XLM-R's
// SentencePiece pieces are bare (no ## marker) and contiguous, so a cursor-walk reconstructs whole-word
// and multi-piece spans across Latin, CJK and RTL scripts.
function locate(tokens, text) {
  let cursor = 0;
  for (const t of tokens) {
    const surface = t.word.startsWith("##") ? t.word.slice(2) : t.word;
    const at = text.indexOf(surface, cursor);
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

// Merge BIO runs of the same entity type into whole-entity spans. Pure BIO: a B- always begins a new
// span; an I- of the same type continues. This is exactly right for SentencePiece (no ## continuation
// marker), and cleanly separates adjacent same-type entities (two people back to back each start B-PER).
function merge(tokens, text) {
  const spans = [];
  let cur = null;
  for (const t of tokens) {
    const type = t.entity.slice(2); // strip B-/I-
    const isB = t.entity.startsWith("B-");
    const continues = cur && type === cur.type && !isB;
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
