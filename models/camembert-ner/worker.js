// camembert-ner worker — inference off the main thread so the control UI stays responsive.
// Model: Xenova/camembert-ner (task: token-classification), WASM, q8. A genuine FRENCH named-entity
// recogniser (CamemBERT fine-tuned on the ~170k-sentence WikiNER-fr dataset), labelling PER / ORG /
// LOC / MISC spans. This is the ONNX q8 export of Jean-Baptiste/camembert-ner (MIT).
//
// CamemBERT is a SentencePiece model: a word may be split into several sub-word pieces, and the model's
// label set has ONLY I- tags ({O, I-PER, I-ORG, I-LOC, I-MISC} — no B- tags), so entity boundaries come
// from the token stream alone. The token-classification pipeline tags each sub-token with a label +
// score but returns NO character offsets. So here we:
//   1. run token-classification to get per-sub-token {entity, score, index, word},
//   2. merge sub-word pieces back into whole WORDS (a new word begins at a ▁-prefixed piece), taking the
//      word-initial piece's label and pooling the score,
//   3. map each merged word back to its character span in the ORIGINAL text (cursor-walk) so inline
//      highlighting lands exactly,
//   4. group consecutive same-type entity words into entity SPANS (WikiNER "simple" aggregation:
//      "Los" I-LOC + "Altos" I-LOC → one LOC "Los Altos").
// Raw sub-tokens, merged words AND entity spans are all returned — the page shows each layer.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/camembert-ner";
const SP = "▁"; // SentencePiece word-boundary marker ▁
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

function isSpecial(word) {
  return /^<(s|\/s|unk|pad|mask)>$/i.test(word);
}

const WORD_START = /^[\p{L}\p{N}]/u;
const WORD_END = /[\p{L}\p{N}]$/u;

// Strip the I- prefix → bare entity type (PER/ORG/LOC/MISC), or null for O.
function typeOf(label) {
  return label && label.startsWith("I-") ? label.slice(2) : null;
}

// Merge sub-tokens back into whole words. Transformers.js decodes each sub-token and (for this model)
// STRIPS the SentencePiece ▁ boundary marker, so we can't rely on it alone. Instead we map each token to
// its char span in the ORIGINAL text with a monotonic cursor walk, then treat a token as a CONTINUATION
// of the previous word only when it abuts it with no whitespace AND both sides are word characters (so
// "Al"+"tos" merge, but "Paris"+"." and space-separated words stay apart). A ▁-prefixed token, when
// present, always starts a new word.
function mergeWords(tokens, text) {
  const words = [];
  let cursor = 0;
  for (const t of tokens) {
    if (isSpecial(t.word)) continue;
    const hadMarker = t.word.startsWith(SP);
    const surface = t.word.replace(new RegExp(SP, "g"), "");
    if (!surface) continue;
    const at = text.indexOf(surface, cursor);
    const start = at >= 0 ? at : null;
    const end = at >= 0 ? at + surface.length : null;
    const prev = words[words.length - 1];
    const abuts = prev && start != null && prev.end === start;
    const continues = !hadMarker && abuts && WORD_START.test(surface) &&
      WORD_END.test(prev.surface);
    if (continues) {
      prev.surface += surface;
      prev.scores.push(t.score);
      prev.end = end;
    } else {
      words.push({
        entity: t.entity,
        type: typeOf(t.entity),
        scores: [t.score],
        surface,
        start,
        end,
      });
    }
    if (end != null) cursor = end;
  }
  for (const w of words) {
    w.score = w.scores.reduce((a, b) => a + b, 0) / w.scores.length;
    delete w.scores;
  }
  return words;
}

// Group consecutive same-type entity words into spans (WikiNER "simple" aggregation over the
// I-only tag set): an entity word continues the open span only when it is the very next word in
// sequence and carries the same type — "Los" I-LOC + "Altos" I-LOC → one LOC "Los Altos".
function entitySpans(words, text) {
  const spans = [];
  let prevWord = null;
  for (const w of words) {
    if (w.type) {
      const last = spans[spans.length - 1];
      const prevEntityWord = last && last.words[last.words.length - 1];
      if (last && last.type === w.type && prevEntityWord === prevWord) {
        last.words.push(w);
        if (w.end != null) last.end = w.end;
      } else {
        spans.push({ type: w.type, words: [w], start: w.start, end: w.end });
      }
    }
    prevWord = w;
  }
  return spans.map((s) => ({
    type: s.type,
    text: s.start != null && s.end != null
      ? text.slice(s.start, s.end)
      : s.words.map((w) => w.surface).join(" "),
    start: s.start,
    end: s.end,
    score: s.words.reduce((a, w) => a + w.score, 0) / s.words.length,
  }));
}

async function analyse(text) {
  const raw = await pipe(text);
  const tokens = raw.map((t) => ({
    entity: t.entity,
    score: t.score,
    index: t.index,
    word: t.word,
  }));
  const words = mergeWords(tokens, text);
  const entities = entitySpans(words, text);
  return { text, tokens, words, entities };
}

async function tag(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await analyse(text);
  post({ type: "tag", id, ...r, ms: Math.round(performance.now() - t0), device });
}

async function tagMany(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  for (const text of texts) results.push(await analyse(text));
  post({ type: "tagMany", id, results, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "tag") await tag(e.data.id, e.data.text);
    else if (type === "tagMany") await tagMany(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
