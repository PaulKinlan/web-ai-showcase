// PII detection & redaction worker — token-classification off the main thread so the editor stays smooth.
//
// Model: phatvo/deberta_finetuned_pii-ONNX (q8) — an ONNX build of lakshyakh93/deberta_finetuned_pii (MIT),
// an mDeBERTa-v3 fine-tune that tags each token with one of 60 personal-data categories (EMAIL, PHONE_NUMBER,
// CREDITCARDNUMBER, IBAN, BITCOINADDRESS, SSN, PASSWORD, IP, STREETADDRESS, FIRSTNAME/LASTNAME, DATE, …).
// deberta-v3 quantises cleanly, so q8 (143 MB) matches fp16/fp32 detection — verified before building.
//
// The pipeline returns one BIO-tagged row per sub-word token with a confidence, but NO character offsets. So,
// exactly like the BERT-NER demo, we:
//   1. run token-classification to get per-token {entity, score, index, word},
//   2. map each token back to a character span in the ORIGINAL text (a monotonic cursor-walk indexOf, so the
//      SentencePiece ▁ / leading-space markers don't matter and highlights land on the real substring),
//   3. merge B-/I- runs of the same category into whole PII spans with a pooled confidence,
//   4. trim stray wrapping punctuation off each span so a highlight/redaction lands on the value itself.
// It was proven correct FIRST in headless Chrome: emails, phone numbers, a full Bitcoin address, a credit-card
// number, an IBAN and a date were each detected and captured as WHOLE spans across varied phrasings (including
// a sentence where a weaker model missed the email entirely) — so redaction masks the value, never half of it.
// Nothing leaves the tab.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "phatvo/deberta_finetuned_pii-ONNX";
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

// Strip the SentencePiece word-start marker (▁), a GPT-style Ġ, or a literal leading space so indexOf matches
// the raw substring in the original text.
const surfaceOf = (word) => String(word ?? "").replace(/^[\s▁Ġ]+/, "");
// A token continues the previous word when it carries NO word-start marker.
const isContinuation = (word) => !/^[\s▁Ġ]/.test(String(word ?? ""));

// Give each returned token a character span in the ORIGINAL text. The pipeline drops "O" tokens, so we walk a
// monotonic cursor and indexOf the token's surface form from that cursor forward.
function locate(tokens, text) {
  let cursor = 0;
  for (const t of tokens) {
    const surface = surfaceOf(t.word);
    if (!surface) {
      t.start = t.end = null;
      continue;
    }
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

// Merge BIO runs of the same category into whole-entity spans.
function merge(tokens, text) {
  const spans = [];
  let cur = null;
  for (const t of tokens) {
    const type = t.entity.replace(/^[BI]-/, "");
    const isB = t.entity.startsWith("B-");
    const continues = cur && type === cur.type && (!isB || isContinuation(t.word));
    if (continues) {
      cur.scores.push(t.score);
      if (t.end != null) cur.end = t.end;
    } else {
      if (cur) spans.push(cur);
      cur = { type, scores: [t.score], start: t.start, end: t.end };
    }
  }
  if (cur) spans.push(cur);
  return spans
    .filter((s) => s.start != null && s.end != null)
    .map((s) => trimSpan(s, text))
    .filter((s) => s.end > s.start)
    .map((s) => ({
      type: s.type,
      start: s.start,
      end: s.end,
      text: text.slice(s.start, s.end),
      score: s.scores.reduce((a, b) => a + b, 0) / s.scores.length,
    }));
}

// A run can bleed onto a leading "(" / quote or a trailing ")" / sentence punctuation. Trim wrapping
// punctuation + whitespace off the ends so the highlight/redaction covers the value itself.
function trimSpan(s, text) {
  let { start, end } = s;
  while (start < end && /[\s([{"'<]/.test(text[start])) start++;
  while (end > start && /[\s)\]}"'>,;:.!?]/.test(text[end - 1])) end--;
  return { ...s, start, end };
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

async function detect(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await analyse(text);
  post({ type: "detect", id, ...r, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "detect") await detect(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
