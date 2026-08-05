// Stanford de-identifier worker — PHI detection off the main thread, WASM q8.
// Model: onnx-community/stanford-deidentifier-base-ONNX (task: token-classification).
//
// The Stanford de-identifier (StanfordAIMI/stanford-deidentifier-base, MIT) is a PubMedBERT
// encoder fine-tuned on thousands of radiology reports + medical notes (MIDRC + i2b2-style data)
// to detect protected health information (PHI). Unlike classic BIO NER, it uses a FLAT per-token
// label scheme over seven PHI categories: PATIENT, HCW (healthcare worker), HOSPITAL, DATE, ID,
// PHONE, VENDOR — plus O. Adjacent tokens with the same label merge into a span; the model relies
// on run-merging rather than B-/I- prefixes.
//
// The pipeline returns per-token labels but no character offsets, and the tokenizer is
// PubMedBERT-uncased (WordPiece, lowercased). So here we:
//   1. run token-classification with aggregation_strategy "none" to get every non-O token
//      {entity, score, index, word},
//   2. map each token back to a character span in the ORIGINAL text with a case-insensitive
//      cursor-walk (the tokenizer lowercases; indexOf on the lowercased copy preserves offsets),
//   3. merge runs of the same flat label into whole-entity spans, BRIDGING across single-char
//      non-alphanumeric O tokens ("/", "-", ".") so "1/1/2020" and "567-493-1234" stay one span.
// Both raw per-token tags AND merged spans are returned — the pages show both (see-inside shows
// the WordPiece tokenization with its ## continuations and per-token confidences).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

// Loadable browser build: the onnx-community ONNX mirror of StanfordAIMI/stanford-deidentifier-base.
// Declared as a string literal AT the loader call site (repo convention) so the portfolio-acceptance
// gate can statically extract the stage.
let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "token-classification",
    model: "onnx-community/stanford-deidentifier-base-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

const BRIDGE_RE = /^[^A-Za-z0-9]$/; // single non-alphanumeric char: "/", "-", ".", ",", …

// Give each returned token a character span in the ORIGINAL text. The tokenizer lowercases
// (PubMedBERT uncased), so search a lowercased copy — offsets are identical for ASCII. "##" pieces
// continue the previous surface (WordPiece), so strip the prefix and search forward from cursor.
function locate(tokens, text) {
  const lower = text.toLowerCase();
  let cursor = 0;
  for (const t of tokens) {
    const surface = t.word.startsWith("##") ? t.word.slice(2) : t.word;
    if (!surface) {
      t.start = t.end = null;
      continue;
    }
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

// Merge runs of the same flat label into spans. Flat scheme (no B-/I-): consecutive same-label
// tokens merge. Bridge: a span of type T may absorb single non-alphanumeric O tokens between two
// T tokens (e.g. "1/1/2020" → DATE, O, DATE, O, DATE), so phone numbers and dates survive as one
// entity. Only tokens already located in the text are merged.
function merge(tokens, text) {
  const spans = [];
  let cur = null;
  for (const t of tokens) {
    if (t.entity === "O" || t.start == null) {
      // Could be a bridge char between two same-type spans? Only if it is a single non-alnum char
      // and the next token continues the current type — handled below via lookahead-free logic:
      // we instead keep it as a pending bridge and resolve when the next real token arrives.
      if (cur && BRIDGE_RE.test(t.word) && t.start != null) {
        cur.bridge = cur.bridge ?? [];
        cur.bridge.push(t);
      } else {
        if (cur) { spans.push(finalize(cur, text)); cur = null; }
      }
      continue;
    }
    const type = t.entity;
    if (cur && type === cur.type) {
      cur.tokens.push(t);
      cur.end = t.end;
      if (cur.bridge) {
        // absorb the bridge chars into the span now that the type continues
        cur.bridge.forEach((b) => { cur.tokens.push(b); });
        cur.bridge = null;
      }
    } else {
      if (cur) spans.push(finalize(cur, text));
      cur = { type, tokens: [t], start: t.start, end: t.end, bridge: null };
    }
  }
  if (cur) spans.push(finalize(cur, text));

  // collapse token list to the merged [start,end) so bridging chars are included in the span text
  function finalize(c, src) {
    const ordered = [...c.tokens, ...(c.bridge ?? [])].sort((a, b) => a.index - b.index);
    const start = Math.min(...ordered.filter((x) => x.start != null).map((x) => x.start));
    const end = Math.max(...ordered.filter((x) => x.end != null).map((x) => x.end));
    return {
      type: c.type,
      start,
      end,
      text: src.slice(start, end),
      score: ordered.filter((x) => x.score != null).reduce((a, b) => a + b, 0) /
        ordered.filter((x) => x.score != null).length,
      tokenCount: c.tokens.length,
      tokens: ordered.map((t) => ({ word: t.word, entity: t.entity, score: t.score, index: t.index })),
    };
  }
  return spans;
}

const PHI_LABELS = new Set(["PATIENT", "HCW", "HOSPITAL", "DATE", "ID", "PHONE", "VENDOR"]);

async function analyse(text) {
  const raw = await pipe(text, { aggregation_strategy: "none" });
  const tokens = raw
    .map((t) => ({ entity: t.entity, score: t.score, index: t.index, word: t.word }))
    .filter((t) => PHI_LABELS.has(t.entity));
  locate(tokens, text);
  const spans = merge(tokens, text);
  const counts = {};
  for (const s of spans) counts[s.type] = (counts[s.type] ?? 0) + 1;
  return { text, tokens, spans, counts };
}

async function deidentify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await analyse(text);
  post({ type: "result", id, ...r, ms: Math.round(performance.now() - t0), device });
}

async function deidentifyMany(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  for (const text of texts) results.push(await analyse(text));
  post({ type: "many", id, results, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "deidentify") await deidentify(e.data.id, e.data.text);
    else if (type === "deidentifyMany") await deidentifyMany(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
