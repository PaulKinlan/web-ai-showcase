// Family-specific MS MARCO MiniLM cross-encoder reranker worker for the all-mpnet-base-v2 Multi-model
// route. This is a PINNED copy of the reranker (Xenova/ms-marco-MiniLM-L-6-v2, q8, revision
// a09144355adeed5f58c8ed011d209bf8ee5a1fec) so the second stage is immutable rather than loading the
// shared ms-marco-reranker worker's floating "main". The shared built demo is untouched (durable
// contract); this isolated worker exists only to express the pin the shared one cannot.
//
// A CROSS-ENCODER reads the query and a passage TOGETHER ([CLS] query [SEP] passage [SEP]) and emits a
// single relevance logit. We read the RAW logit straight off the model (the pipeline would squash it
// through a sigmoid) so the comparison is honest. Uses worker-protocol.js (cancellation, dispose).

import { serveWorker } from "/web-ai-showcase/lib/worker-protocol.js";
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

const STOP = new Set(
  "the a an of to in on and or is are was were be been for with as at by from this that it its into their his her our your"
    .split(" "),
);
function contentWords(s) {
  return new Set(
    s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((w) =>
      w.length > 1 && !STOP.has(w)
    ),
  );
}
function lexicalOverlap(query, passage) {
  const q = contentWords(query);
  if (q.size === 0) return 0;
  const p = contentWords(passage);
  let hit = 0;
  for (const w of q) if (p.has(w)) hit++;
  return hit / q.size;
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

serveWorker({
  async init() {},
  methods: {
    async load(_payload, { onProgress }) {
      const loaded = await loadPipeline({
        task: "text-classification",
        model: "Xenova/ms-marco-MiniLM-L-6-v2",
        backend: "wasm",
        dtype: "q8",
        revision: "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        onProgress: (p) => onProgress?.(p),
      });
      pipe = loaded.pipe;
      device = loaded.device;
      return { result: { device } };
    },
    // payload: { query, passages }. Returns { results:[{idx,passage,logit,prob,lexical}], ms, device }.
    async rerank({ query, passages }, { signal }) {
      if (!pipe) throw new Error("Reranker not loaded");
      if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const t0 = performance.now();
      const tok = pipe.tokenizer;
      const model = pipe.model;
      const enc = tok(passages.map(() => query), {
        text_pair: passages,
        padding: true,
        truncation: true,
      });
      const out = await model(enc);
      if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const logitsT = out.logits;
      const nCols = logitsT.dims[logitsT.dims.length - 1];
      const flat = Array.from(logitsT.data, Number);
      const results = passages.map((passage, i) => {
        const logit = flat[i * nCols];
        return {
          idx: i,
          passage,
          logit,
          prob: sigmoid(logit),
          lexical: lexicalOverlap(query, passage),
        };
      });
      results.sort((a, b) => b.logit - a.logit);
      const ms = Math.round(performance.now() - t0);
      return { result: { results, ms, device } };
    },
  },
  onDispose() {
    pipe = null;
  },
});
