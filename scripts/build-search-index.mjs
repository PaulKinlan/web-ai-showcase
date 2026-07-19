#!/usr/bin/env node
// Deterministic search-index generator for the client-side model explorer (route: /explore/).
//
// Pure function of the repo's own build artifacts — models.json + inventory/lineage/* + the built
// route pages on disk — emitting `search/index/`:
//   • meta.json          — schemaVersion, sha-256 checksum, row count, byte sizes, embedder identity.
//   • index.json         — per-doc display+filter metadata, the BM25 inverted index (parallel arrays),
//                          the alias / canonical-family expansion map, family→members grouping, and the
//                          facet value lists that drive the filter UI.
//   • vectors.i8.bin      — int8-quantised doc embeddings, N×384 row-major (the ADR's ~0.91 MB default).
//
// Determinism:
//   • The embedder is FIXED (Xenova/all-MiniLM-L6-v2) at a FIXED dtype (q8) and each doc is embedded
//     ONE AT A TIME. Single-string q8 inference is byte-stable on a given ONNX Runtime build, and
//     one-at-a-time matches how the /explore/ worker embeds a query (batch composition changes q8
//     output — see search/ARCHITECTURE.md), so doc and query vectors share one distribution.
//   • checksum = sha256 of the normalised corpus (slug + lexical text + semantic text, sorted by slug).
//     It is the IndexedDB cache key + staleness signal the worker checks on load (BuildPayload.expectChecksum).
//
// Incremental-update path (documented; full rebuild is the cheap default at this scale):
//   The corpus is keyed by `slug`. To patch a data refresh without re-embedding all 2,493 rows:
//     1. Load the previous search/index/index.json + vectors.i8.bin.
//     2. Diff slugs: added / removed / changed (changed = its normalised {lexical,semantic} text moved).
//     3. Re-embed ONLY changed+added slugs (one-at-a-time), drop removed rows, splice vector rows.
//     4. Rebuild the inverted index (35 ms — cheaper than a diff) and recompute the checksum.
//   Run with `--incremental` to reuse cached vectors for unchanged slugs (falls back to full build if
//   no prior artifact exists). A full rebuild is otherwise the default and is byte-identical for
//   identical input.
//
// Usage:
//   npm install @huggingface/transformers@3.7.5   # dev-only; the emitted artifacts are what ships
//   TMPDIR=/path/non-tmpfs node scripts/build-search-index.mjs [--incremental]
//
// This script is ADDITIVE tooling: it reads repo data and writes only under search/index/. It changes
// no route, demo, or model.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const SCHEMA_VERSION = 1;
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const EMBED_DTYPE = "q8";
const EMBED_SIZE_MB = 25; // approximate on-device transfer for the explicit-download UX
const DIM = 384;

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const outDir = `${repoRoot}search/index/`;
const incremental = process.argv.includes("--incremental");

// ── tokenizer — MUST match search/worker.js exactly (same stoplist, same split) ────────────────────
const STOP = new Set(
  "the a an and or of to for with in on is are be this that it as at by from into over your you can"
    .split(" "),
);
function tokenize(text) {
  const out = [];
  for (const raw of String(text).toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 2 || STOP.has(raw)) continue;
    out.push(raw);
  }
  return out;
}
const TAG_DROP = /^(dataset|arxiv|base_model|region|license|doi|co2_eq_emissions):/i;

// ── relationship → canonical/variant/fork classification (honest, from lineage records) ────────────
const REL_KIND = {
  "canonical": { kind: "canonical", label: "Canonical" },
  "quant-variant": { kind: "variant", label: "Quantised variant" },
  "distillation": { kind: "variant", label: "Distillation" },
  "fine-tune": { kind: "fork", label: "Fine-tune" },
  "specialization-distinct": { kind: "specialization", label: "Specialization" },
  "uncertain": { kind: "uncertain", label: "Uncertain lineage" },
};

function readJson(rel) {
  return JSON.parse(readFileSync(`${repoRoot}${rel}`, "utf8"));
}
function readNdjson(rel) {
  if (!existsSync(`${repoRoot}${rel}`)) return [];
  const rows = [];
  for (const line of readFileSync(`${repoRoot}${rel}`, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch { /* skip malformed */ }
  }
  return rows;
}

function percentileRanks(values) {
  // Rank each value into [0,1] by its position among sorted values (ties share the max rank).
  const sorted = [...values].map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const rank = new Array(values.length).fill(0);
  const n = values.length;
  for (let k = 0; k < n; k++) rank[sorted[k][1]] = n <= 1 ? 1 : k / (n - 1);
  return rank;
}

async function main() {
  console.log("building search index → search/index/");
  const models = readJson("models.json").models || [];
  const lineage = new Map(readNdjson("inventory/lineage/records.ndjson").map((r) => [r.id, r]));
  const value = new Map(readNdjson("inventory/lineage/value-records.ndjson").map((r) => [r.id, r]));
  const priorityDoc = readJson("inventory/lineage/priority.json");
  const priority = new Map((priorityDoc.queue || []).map((q) => [q.id, q]));
  const invTags = new Map();
  for (const r of readNdjson("inventory/eligible.ndjson")) {
    if (r.id && Array.isArray(r.tags)) invTags.set(r.id, r.tags);
  }

  // ── 1. per-doc metadata + searchable/semantic text ───────────────────────────────────────────────
  const downloadsArr = models.map((m) => lineage.get(m.hfId)?.downloads ?? 0);
  const dlRank = percentileRanks(downloadsArr);

  const docs = [];
  const lexicalTexts = [];
  const semanticTexts = [];
  for (let i = 0; i < models.length; i++) {
    const m = models[i];
    const ln = lineage.get(m.hfId) || {};
    const vr = value.get(m.hfId);
    const pr = priority.get(m.hfId);
    const rel = ln.relationship || (m.status === "built" ? "canonical" : "uncertain");
    const relKind = REL_KIND[rel] || REL_KIND.uncertain;
    const canonicalFamily = ln.canonicalFamily || m.family || "unknown";

    // Honest quality confidence: prefer the reviewed value-record's overall confidence, else the
    // lineage record's confidence, else "unreviewed". downloads/likes are WEAK context only.
    const confidence = vr?.overallConfidence || ln.confidence || "unreviewed";
    const evalPending = vr?.evalPending || [];
    const tier = pr?.tier || (m.status === "built" ? "built" : null);

    const hasDemo = m.status === "built" && existsSync(`${repoRoot}models/${m.slug}/index.html`);
    const doc = {
      slug: m.slug,
      name: m.name,
      hfId: m.hfId,
      task: m.task,
      modality: m.modality,
      backend: m.backend || "wasm",
      runtime: m.runtime || "transformers.js",
      license: m.license || "",
      sizeMB: m.sizeMB || 0,
      status: m.status, // built | pending | blocked
      family: m.family || "",
      canonicalFamily,
      relationship: rel,
      relKind: relKind.kind, // canonical | variant | fork | specialization | uncertain
      relLabel: relKind.label,
      confidence, // high | medium | low | unreviewed
      evalPending, // dimensions still eval-pending (honest)
      tier, // priority tier: high|medium|low|superseded|blocked|built|null
      tierRationale: pr?.rationale || "",
      canonicalAlternative: pr?.canonicalAlternative || null,
      downloads: ln.downloads ?? null,
      likes: ln.likes ?? null,
      qualityPercentile: +dlRank[i].toFixed(4), // weak popularity signal, 0..1
      blurb: m.blurb || "",
      unlocks: m.unlocks || "",
      blockedReason: m.blockedReason || "",
      requiresWebGPU: !!m.requiresWebGPU,
      demoRoute: hasDemo ? `models/${m.slug}/` : null,
      hfUrl: `https://huggingface.co/${m.hfId}`,
    };
    docs.push(doc);

    // Lexical corpus: field-weighted (name/task repeated) so BM25 favours name/task hits (POC-proven).
    const tags = (invTags.get(m.hfId) || []).filter((t) => !TAG_DROP.test(t)).join(" ");
    lexicalTexts.push(
      [
        m.name,
        m.name,
        m.slug,
        m.task,
        m.task,
        m.modality,
        m.family,
        canonicalFamily,
        m.hfId,
        m.blurb,
        m.unlocks,
        m.runtime,
        m.backend,
        m.license,
        tags,
      ].join(" "),
    );
    // Semantic corpus: a natural-language descriptor (matches NL-intent queries).
    semanticTexts.push(
      `${m.name}. Task: ${m.task}. Modality: ${m.modality}. ${m.blurb} ${m.unlocks} ` +
        `Family: ${canonicalFamily}.`,
    );
  }

  // ── 2. deterministic checksum over the normalised corpus ─────────────────────────────────────────
  const normalised = docs
    .map((d, i) => `${d.slug}${lexicalTexts[i]}${semanticTexts[i]}`)
    .sort()
    .join("\n");
  const checksum = createHash("sha256").update(normalised).digest("hex");
  console.log(`corpus checksum ${checksum.slice(0, 16)}… (${docs.length} rows)`);

  // ── 3. BM25 inverted index (parallel arrays; hydrated as typed arrays in the worker) ─────────────
  const N = docs.length;
  const docLen = new Array(N).fill(0);
  const postingMap = new Map(); // term -> Map<docId, tf>
  const df = new Map();
  let tokenTotal = 0;
  for (let i = 0; i < N; i++) {
    const toks = tokenize(lexicalTexts[i]);
    docLen[i] = toks.length;
    tokenTotal += toks.length;
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    for (const [term, f] of tf) {
      let pm = postingMap.get(term);
      if (!pm) postingMap.set(term, pm = new Map());
      pm.set(i, f);
      df.set(term, (df.get(term) || 0) + 1);
    }
  }
  const terms = [...postingMap.keys()].sort();
  const dfArr = terms.map((t) => df.get(t));
  // postings: for each term, a flat [docId, tf, docId, tf, ...] (docId ascending → deterministic).
  const postings = terms.map((t) => {
    const pm = postingMap.get(t);
    const ids = [...pm.keys()].sort((a, b) => a - b);
    const flat = [];
    for (const id of ids) flat.push(id, pm.get(id));
    return flat;
  });
  const avgdl = tokenTotal / N;
  let indexBytes = 0;
  for (const p of postings) indexBytes += p.length * 4;

  // ── 4. alias / canonical-family expansion + family grouping ──────────────────────────────────────
  const families = {}; // canonicalFamily -> [docIdx...]
  for (let i = 0; i < N; i++) {
    (families[docs[i].canonicalFamily] ||= []).push(i);
  }
  // Hand-curated intent aliases (ADR §5.1) folded into a query before scoring.
  const aliases = {
    whisper: ["automatic-speech-recognition", "speech", "transcription"],
    asr: ["automatic-speech-recognition", "speech"],
    sam: ["segment-anything", "segmentation"],
    llama: ["text-generation", "llm", "chat"],
    qwen: ["text-generation", "llm", "chat"],
    phi: ["text-generation", "llm", "chat"],
    gemma: ["text-generation", "llm", "chat"],
    embedding: ["feature-extraction", "sentence-similarity"],
    embeddings: ["feature-extraction", "sentence-similarity"],
    tts: ["text-to-speech"],
    ocr: ["image-to-text"],
    caption: ["image-to-text", "image-text-to-text"],
    detection: ["object-detection"],
    rembg: ["image-segmentation", "background"],
    vqa: ["visual-question-answering", "image-text-to-text"],
  };

  // ── 5. facet value lists (counts computed live in the worker per query) ──────────────────────────
  const uniq = (f) =>
    [...new Set(docs.map((d) => d[f]).filter((v) => v !== "" && v != null))].sort();
  const facetValues = {
    task: uniq("task"),
    modality: uniq("modality"),
    license: uniq("license"),
    runtime: uniq("runtime"),
    backend: uniq("backend"),
    status: uniq("status"),
    relKind: uniq("relKind"),
    tier: uniq("tier"),
    confidence: uniq("confidence"),
    canonicalFamily: Object.keys(families).sort(),
  };

  // ── 6. embeddings — one at a time (deterministic + query-distribution-matched) ───────────────────
  let vec = null;
  let reused = 0;
  const prevMetaPath = `${outDir}meta.json`;
  const prevIndexPath = `${outDir}index.json`;
  const prevVecPath = `${outDir}vectors.i8.bin`;
  let prevBySlug = null;
  if (incremental && existsSync(prevIndexPath) && existsSync(prevVecPath)) {
    try {
      const pj = JSON.parse(readFileSync(prevIndexPath, "utf8"));
      const pv = new Int8Array(readFileSync(prevVecPath).buffer);
      prevBySlug = new Map();
      for (let i = 0; i < pj.docs.length; i++) {
        prevBySlug.set(pj.docs[i].slug, {
          sem: pj._semHash?.[i],
          row: pv.subarray(i * DIM, (i + 1) * DIM),
        });
      }
      console.log(`incremental: loaded ${prevBySlug.size} prior vectors`);
    } catch (e) {
      console.log(`incremental: prior artifact unusable (${e.message}); full rebuild`);
      prevBySlug = null;
    }
  }

  const semHash = semanticTexts.map((s) => createHash("sha1").update(s).digest("hex").slice(0, 12));
  const { pipeline, env } = await import("@huggingface/transformers");
  env.allowLocalModels = false;
  console.log(`loading embedder ${EMBED_MODEL} (${EMBED_DTYPE})…`);
  const pipe = await pipeline("feature-extraction", EMBED_MODEL, { dtype: EMBED_DTYPE });
  vec = new Int8Array(N * DIM);
  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    const prior = prevBySlug?.get(docs[i].slug);
    if (prior && prior.sem === semHash[i] && prior.row?.length === DIM) {
      vec.set(prior.row, i * DIM);
      reused++;
    } else {
      const out = await pipe(semanticTexts[i], { pooling: "mean", normalize: true });
      const f = out.data; // Float32Array, already L2-normalised
      const off = i * DIM;
      for (let d = 0; d < DIM; d++) {
        vec[off + d] = Math.max(-127, Math.min(127, Math.round(f[d] * 127)));
      }
    }
    if (i % 250 === 249 || i === N - 1) {
      const rate = ((i + 1) / ((Date.now() - t0) / 1000)).toFixed(0);
      console.log(`  embedded ${i + 1}/${N} (${rate}/s, ${reused} reused)`);
    }
  }

  // ── 7. write artifacts ───────────────────────────────────────────────────────────────────────────
  mkdirSync(outDir, { recursive: true });
  const index = {
    schemaVersion: SCHEMA_VERSION,
    checksum,
    docs,
    lexical: { terms, df: dfArr, postings, docLen, avgdl, N },
    aliases,
    families,
    facetValues,
    _semHash: semHash, // internal: enables --incremental reuse; ignored by the worker
  };
  writeFileSync(prevIndexPath, JSON.stringify(index));
  writeFileSync(prevVecPath, Buffer.from(vec.buffer));
  const meta = {
    schemaVersion: SCHEMA_VERSION,
    checksum,
    docCount: N,
    terms: terms.length,
    dim: DIM,
    indexBytes,
    vecBytes: vec.byteLength,
    embedModel: EMBED_MODEL,
    embedDtype: EMBED_DTYPE,
    embedSizeMB: EMBED_SIZE_MB,
    generated: new Date().toISOString().slice(0, 10),
    // Facet value lists live here (small) so the /explore/ main thread never parses the ~3 MB
    // index.json — that fetch/parse stays entirely in the worker (INP: no main-thread long task).
    facetValues,
  };
  writeFileSync(prevMetaPath, JSON.stringify(meta, null, 2));

  console.log(
    `\ndone: ${N} docs · ${terms.length} terms · index ${(indexBytes / 1e6).toFixed(2)} MB ` +
      `(json ${(readFileSync(prevIndexPath).length / 1e6).toFixed(2)} MB) · vectors ` +
      `${(vec.byteLength / 1e6).toFixed(2)} MB · ${reused} vectors reused`,
  );
  console.log(`checksum ${checksum}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
