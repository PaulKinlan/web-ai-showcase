// POC search worker — feasibility evidence for the client-side model explorer (NOT the product).
//
// Proves the CHOSEN stack loads and queries entirely off the main thread on THIS repo's data:
//   • a compact pure-JS inverted index with BM25 ranking over all ~2,500 catalogue entries, and
//   • a brute-force Float32 / Int8 cosine over ~2,500 precomputed vectors (semantic intent).
// It also PROBES the rejected-but-evaluated alternative — OPFS + @sqlite.org/sqlite-wasm FTS5 — to
// confirm whether OPFS SyncAccessHandle actually initialises in a worker on this GitHub-Pages
// deployment (no COOP/COEP), and to time SQLite init/FTS5 for the ARCHITECTURE.md evidence table.
//
// Runs under the repo's real typed protocol (lib/worker-protocol.js) — the same envelope the eventual
// explorer worker will speak — so this doubles as a contract integration test.
//
// modern-web-guidance retained + applied (ids):
//   • break-up-long-tasks — index build yields to the event loop on a 50ms deadline (scheduler.yield
//     fallback to setTimeout) so abort/dispose messages interleave and the worker never wedges.
//   • schedule-tasks-by-priority — build is background-priority conceptually; queries are latest-wins.
//   • identify-inp-causes / interactions-in-complex-layouts — ALL scoring is here, off the main
//     thread, so the page's INP stays input-delay only; the driver verifies ~0 main-thread long tasks.

import { serveWorker, yieldToMain } from "../../lib/worker-protocol.js";

const BASE = new URL("../../", import.meta.url); // repo root (works under /web-ai-showcase/ base path)
const DIM = 384; // gte-small / all-MiniLM-L6-v2 embedding dimension (the repo's built embedders)

// ── deterministic PRNG (mulberry32) — fixed vectors so runs are reproducible ──────────────────────
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hashStr(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// ── tokenizer — lowercase, split on non-alphanumeric, drop 1-char + a tiny stoplist ───────────────
const STOP = new Set(
  "the a an and or of to for with in on is are be this that it as at by from into over your you can"
    .split(
      " ",
    ),
);
function tokenize(text) {
  const out = [];
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 2 || STOP.has(raw)) continue;
    out.push(raw);
  }
  return out;
}
// Tags carry huge noise (dataset:/arxiv:/region:/base_model:/license:) — keep only meaningful ones.
const TAG_DROP = /^(dataset|arxiv|base_model|region|license|doi|co2_eq_emissions):/i;

// ── module state ──────────────────────────────────────────────────────────────────────────────────
let DOCS = []; // [{slug,name,task,modality,license,sizeMB,family,runtime,backend}]
let POSTINGS = null; // Map<term, Int32Array pairs [docId,tf,docId,tf,...]>  (compact)
let DF = null; // Map<term, docFreq>
let DOCLEN = null; // Float64Array of token counts
let AVGDL = 0;
let N = 0;
let VEC_F32 = null; // Float32Array(N*DIM), row-normalised
let VEC_I8 = null; // Int8Array(N*DIM), scale-127 quantised
let indexBytes = 0;

async function fetchJson(path) {
  const r = await fetch(new URL(path, BASE));
  if (!r.ok) throw new Error(`fetch ${path} → ${r.status}`);
  return r.json();
}
async function fetchText(path) {
  const r = await fetch(new URL(path, BASE));
  if (!r.ok) throw new Error(`fetch ${path} → ${r.status}`);
  return r.text();
}

// ── build: join models.json + inventory tags, build inverted index + fixed vectors ────────────────
async function build(_payload, { onProgress }) {
  const t0 = performance.now();

  const [modelsDoc, invText] = await Promise.all([
    fetchJson("models.json"),
    fetchText("inventory/eligible.ndjson").catch(() => ""),
  ]);
  const models = modelsDoc.models || [];
  const tagsById = new Map();
  if (invText) {
    for (const line of invText.split("\n")) {
      if (!line) continue;
      try {
        const rec = JSON.parse(line);
        if (rec.id && Array.isArray(rec.tags)) tagsById.set(rec.id, rec.tags);
      } catch { /* skip malformed line */ }
    }
  }
  const fetchedMs = performance.now() - t0;

  // Build documents + postings in one deterministic pass, yielding on a 50ms deadline.
  const tBuild = performance.now();
  DOCS = new Array(models.length);
  DOCLEN = new Float64Array(models.length);
  const posting = new Map(); // term -> Map<docId,tf>
  DF = new Map();
  let deadline = performance.now() + 50;
  let tokenTotal = 0;

  for (let i = 0; i < models.length; i++) {
    const m = models[i];
    DOCS[i] = {
      slug: m.slug,
      name: m.name,
      task: m.task,
      modality: m.modality,
      license: m.license || "",
      sizeMB: m.sizeMB || 0,
      family: m.family || "",
      runtime: m.runtime || "",
      backend: m.backend || "",
      status: m.status,
    };
    const tags = (tagsById.get(m.hfId) || []).filter((t) => !TAG_DROP.test(t)).join(" ");
    // Field-weighted corpus: repeat high-signal fields so BM25 favours name/task hits.
    const text = [
      m.name,
      m.name,
      m.slug,
      m.task,
      m.task,
      m.modality,
      m.family,
      m.hfId,
      m.blurb,
      m.unlocks,
      m.runtime,
      m.backend,
      m.license,
      tags,
    ].join(" ");
    const toks = tokenize(text);
    DOCLEN[i] = toks.length;
    tokenTotal += toks.length;
    const tf = new Map();
    for (const tok of toks) tf.set(tok, (tf.get(tok) || 0) + 1);
    for (const [term, f] of tf) {
      let pm = posting.get(term);
      if (!pm) posting.set(term, pm = new Map());
      pm.set(i, f);
      DF.set(term, (DF.get(term) || 0) + 1);
    }
    if (performance.now() >= deadline) {
      onProgress({ status: "indexing", progress: (i / models.length) * 100 });
      await yieldToMain();
      deadline = performance.now() + 50;
    }
  }

  // Freeze postings into compact typed arrays (docId,tf interleaved) → smaller + cache-friendly scans.
  POSTINGS = new Map();
  indexBytes = 0;
  for (const [term, pm] of posting) {
    const arr = new Int32Array(pm.size * 2);
    let j = 0;
    for (const [docId, f] of pm) {
      arr[j++] = docId;
      arr[j++] = f;
    }
    POSTINGS.set(term, arr);
    indexBytes += arr.byteLength + term.length * 2 + 16;
  }
  N = models.length;
  AVGDL = tokenTotal / N;
  const buildIndexMs = performance.now() - tBuild;

  // Fixed pseudo-random unit vectors (stand-in for real gte-small/MiniLM embeddings), + int8 quant.
  const tVec = performance.now();
  VEC_F32 = new Float32Array(N * DIM);
  VEC_I8 = new Int8Array(N * DIM);
  for (let i = 0; i < N; i++) {
    const rnd = mulberry32(hashStr(DOCS[i].slug) ^ 0x9e3779b9);
    let norm = 0;
    const off = i * DIM;
    for (let d = 0; d < DIM; d++) {
      const v = rnd() * 2 - 1;
      VEC_F32[off + d] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < DIM; d++) {
      const nv = VEC_F32[off + d] / norm;
      VEC_F32[off + d] = nv;
      VEC_I8[off + d] = Math.max(-127, Math.min(127, Math.round(nv * 127)));
    }
  }
  const buildVecMs = performance.now() - tVec;

  return {
    result: {
      docCount: N,
      terms: POSTINGS.size,
      avgDocLen: +AVGDL.toFixed(1),
      fetchedMs: +fetchedMs.toFixed(1),
      buildIndexMs: +buildIndexMs.toFixed(1),
      buildVecMs: +buildVecMs.toFixed(1),
      totalBuildMs: +(performance.now() - t0).toFixed(1),
      indexBytes,
      vecF32Bytes: VEC_F32.byteLength,
      vecI8Bytes: VEC_I8.byteLength,
      heapUsed: self.performance?.memory?.usedJSHeapSize ?? null,
    },
  };
}

// ── BM25 lexical query ────────────────────────────────────────────────────────────────────────────
const K1 = 1.2, B = 0.75;
function bm25(qTokens, k) {
  const scores = new Map();
  for (const term of qTokens) {
    const arr = POSTINGS.get(term);
    if (!arr) continue;
    const df = DF.get(term) || arr.length / 2;
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
    for (let p = 0; p < arr.length; p += 2) {
      const docId = arr[p], tf = arr[p + 1];
      const dl = DOCLEN[docId];
      const denom = tf + K1 * (1 - B + (B * dl) / AVGDL);
      const s = idf * ((tf * (K1 + 1)) / denom);
      scores.set(docId, (scores.get(docId) || 0) + s);
    }
  }
  return topK(scores, k);
}
function topK(scores, k) {
  const heap = [];
  for (const [id, s] of scores) heap.push([id, s]);
  heap.sort((a, b) => b[1] - a[1]);
  return heap.slice(0, k);
}

function lexical({ q, k = 20 }) {
  const t = performance.now();
  const hits = bm25(tokenize(q), k);
  const ms = performance.now() - t;
  return {
    result: {
      ms: +ms.toFixed(3),
      count: hits.length,
      top: hits.slice(0, 5).map(([id, s]) => ({ slug: DOCS[id].slug, score: +s.toFixed(3) })),
    },
  };
}

// ── brute-force cosine (dot, vectors are unit-norm) ───────────────────────────────────────────────
function queryVec(seed) {
  const rnd = mulberry32(seed >>> 0);
  const v = new Float32Array(DIM);
  let norm = 0;
  for (let d = 0; d < DIM; d++) {
    v[d] = rnd() * 2 - 1;
    norm += v[d] * v[d];
  }
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < DIM; d++) v[d] /= norm;
  return v;
}
function cosineF32(q, k) {
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const off = i * DIM;
    for (let d = 0; d < DIM; d++) s += q[d] * VEC_F32[off + d];
    scores[i] = s;
  }
  return topKArr(scores, k);
}
function cosineI8(qi8, k) {
  const scores = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const off = i * DIM;
    for (let d = 0; d < DIM; d++) s += qi8[d] * VEC_I8[off + d];
    scores[i] = s;
  }
  return topKArr(scores, k);
}
function topKArr(scores, k) {
  const idx = Array.from({ length: scores.length }, (_, i) => i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k).map((i) => [i, scores[i]]);
}

function semantic({ seed = 1, k = 20, quant = "f32" }) {
  const t = performance.now();
  let hits;
  if (quant === "i8") {
    const qf = queryVec(seed);
    const qi8 = new Int8Array(DIM);
    for (let d = 0; d < DIM; d++) qi8[d] = Math.max(-127, Math.min(127, Math.round(qf[d] * 127)));
    hits = cosineI8(qi8, k);
  } else {
    hits = cosineF32(queryVec(seed), k);
  }
  const ms = performance.now() - t;
  return { result: { ms: +ms.toFixed(3), count: hits.length, quant } };
}

// ── bench: run a suite, report p50/p95 ────────────────────────────────────────────────────────────
function pct(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  return +s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))].toFixed(3);
}
async function bench({ queries, iters = 50 } = {}) {
  const qs = queries ||
    [
      "speech to text",
      "background removal",
      "small chat model webgpu",
      "image captioning",
      "translate french",
      "sentence embeddings",
      "object detection",
      "text to speech tiny",
      "code completion",
      "depth estimation apache license",
    ];
  const lex = [], semF = [], semI = [];
  for (let i = 0; i < iters; i++) {
    const q = qs[i % qs.length];
    let t = performance.now();
    bm25(tokenize(q), 20);
    lex.push(performance.now() - t);
    t = performance.now();
    cosineF32(queryVec(i + 1), 20);
    semF.push(performance.now() - t);
    t = performance.now();
    const qf = queryVec(i + 1);
    const qi8 = new Int8Array(DIM);
    for (let d = 0; d < DIM; d++) qi8[d] = Math.round(qf[d] * 127);
    cosineI8(qi8, 20);
    semI.push(performance.now() - t);
    if (i % 8 === 7) await yieldToMain();
  }
  return {
    result: {
      iters,
      lexical: { p50: pct(lex, 50), p95: pct(lex, 95) },
      semanticF32: { p50: pct(semF, 50), p95: pct(semF, 95) },
      semanticI8: { p50: pct(semI, 50), p95: pct(semI, 95) },
    },
  };
}

// ── OPFS SyncAccessHandle probe — does OPFS init in a worker WITHOUT COOP/COEP here? ──────────────
async function probeOPFS() {
  const t = performance.now();
  try {
    if (!navigator.storage?.getDirectory) {
      return { result: { ok: false, reason: "navigator.storage.getDirectory unavailable" } };
    }
    const dir = await navigator.storage.getDirectory();
    const fh = await dir.getFileHandle("poc-probe.bin", { create: true });
    if (!fh.createSyncAccessHandle) {
      return {
        result: { ok: false, reason: "createSyncAccessHandle unavailable (needs worker + OPFS)" },
      };
    }
    const sah = await fh.createSyncAccessHandle();
    const payload = new TextEncoder().encode("opfs-sah-ok");
    sah.write(payload, { at: 0 });
    sah.flush();
    const back = new Uint8Array(payload.length);
    sah.read(back, { at: 0 });
    sah.close();
    await dir.removeEntry("poc-probe.bin").catch(() => {});
    const ok = new TextDecoder().decode(back) === "opfs-sah-ok";
    return {
      result: { ok, ms: +(performance.now() - t).toFixed(1), coi: self.crossOriginIsolated },
    };
  } catch (e) {
    return {
      result: { ok: false, reason: String(e?.message || e), coi: self.crossOriginIsolated },
    };
  }
}

// ── sqlite-wasm + FTS5 probe (the evaluated alternative) — best-effort, network + timeout guarded ──
async function probeSqlite({ version = "3.50.1-build1" } = {}) {
  const t = performance.now();
  const attempt = (async () => {
    const url = `https://cdn.jsdelivr.net/npm/@sqlite.org/sqlite-wasm@${version}/index.mjs`;
    const mod = await import(/* @vite-ignore */ url);
    const sqlite3 = await mod.default();
    const info = {
      version: sqlite3.version?.libVersion,
      hasOpfsPool: !!sqlite3.installOpfsSAHPoolVfs,
    };
    // In-memory DB is enough to confirm the FTS5 module is compiled into the official build.
    const db = new sqlite3.oo1.DB(":memory:");
    db.exec("CREATE VIRTUAL TABLE ft USING fts5(name, blurb);");
    db.exec("INSERT INTO ft(name,blurb) VALUES ('whisper','speech to text model');");
    let matched = 0;
    db.exec({
      sql: "SELECT count(*) c FROM ft WHERE ft MATCH 'speech';",
      rowMode: "object",
      callback: (r) => (matched = r.c),
    });
    db.close();
    return { ...info, fts5: matched === 1, initMs: +(performance.now() - t).toFixed(1) };
  })();
  try {
    const r = await Promise.race([
      attempt,
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout 20s")), 20000)),
    ]);
    return { result: { ok: true, ...r } };
  } catch (e) {
    return {
      result: {
        ok: false,
        reason: String(e?.message || e),
        ms: +(performance.now() - t).toFixed(1),
      },
    };
  }
}

serveWorker({
  methods: { build, lexical, semantic, bench, probeOPFS, probeSqlite },
});
