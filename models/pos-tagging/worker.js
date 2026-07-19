// POS-tagging worker — inference off the main thread so the control UI stays responsive.
// Model: Xenova/french-camembert-postag-model (task: token-classification), WASM, q8. A genuine
// part-of-speech tagger over FRENCH text using the French TreeBank (FTB) tag set.
//
// CamemBERT is a SentencePiece model: a word may be split into several sub-word pieces, and the piece
// that STARTS a word carries a leading ▁ (U+2581) boundary marker. The token-classification pipeline
// tags each sub-token with an FTB label + score but returns NO character offsets. So here we:
//   1. run token-classification to get per-sub-token {entity, score, index, word},
//   2. merge sub-word pieces back into whole WORDS (a new word begins at a ▁-prefixed piece), taking the
//      word-initial piece's tag as the word's part of speech and pooling the score,
//   3. map each merged word back to its character span in the ORIGINAL text (cursor-walk) so inline
//      highlighting lands exactly, and fold the FTB tag into a universal group (noun/verb/adj/…).
// Both the raw sub-tokens AND the merged words are returned — the page shows both.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";
import { groupOf } from "/web-ai-showcase/models/pos-tagging/pos.js";

const MODEL_ID = "Xenova/french-camembert-postag-model";
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

// Merge sub-tokens back into whole words. Transformers.js decodes each sub-token and (for this model)
// STRIPS the SentencePiece ▁ boundary marker, so we can't rely on it alone. Instead we map each token to
// its char span in the ORIGINAL text with a monotonic cursor walk, then treat a token as a CONTINUATION
// of the previous word only when it abuts it with no whitespace AND both sides are word characters (so
// "tranquil"+"lement" merge, but "salon"+"." and space-separated words stay apart). A ▁-prefixed token,
// when present, always starts a new word.
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
        tag: t.entity,
        group: groupOf(t.entity),
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

async function analyse(text) {
  const raw = await pipe(text);
  const tokens = raw.map((t) => ({
    entity: t.entity,
    score: t.score,
    index: t.index,
    word: t.word,
  }));
  const words = mergeWords(tokens, text);
  return { text, tokens, words };
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
