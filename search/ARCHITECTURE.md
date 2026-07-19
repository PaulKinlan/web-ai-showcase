# Client-side model explorer — search architecture decision record

**Status:** Researched foundation (POC proven). Not the product. No route added yet — the eventual
explorer gets a new stable route `/explore/`. This ADR + the typed contract + the measured POC are
the input to building it.

**Scope:** an additive, fully client-side hybrid search (semantic intent + lexical) over the
web-ai-showcase catalogue — **2,493 entries** in `models.json`, joined to
`inventory/eligible.ndjson` for tags/quality signals. It changes no existing route, adds no demo,
and runs entirely off the main thread on the repo's existing typed worker protocol.

---

## 1. Decision

Build the index and run every query **in a dedicated module worker**, using:

- **Lexical:** a compact **pure-JS inverted index with BM25** ranking, built in-memory from
  `models.json` + inventory tags. ~0.80 MB, rebuilds in ~35 ms.
- **Semantic:** **brute-force cosine over a flat `Float32Array`** of per-model embeddings (384-dim,
  gte-small / all-MiniLM-L6-v2 — the repo's existing built embedders). ~3.65 MB f32, or ~0.91 MB
  int8-quantised. Query time ~1 ms over all 2,493 rows.
- **Hybrid:** normalise BM25 and cosine to [0,1] and blend (`alpha`, default 0.5), after filters.
- **Query embedding:** produced by the embedder **in the same worker**, off the main thread. Lexical
  search plus filters work fully **without** it; the embedder is auto-init-if-cached /
  explicit-download otherwise (invariant 12).
- **Persistence:** IndexedDB for the serialised index/vectors (schema-versioned + checksummed). **No
  SQLite, no OPFS SyncAccessHandle** in the search path.

**Rejected: `@sqlite.org/sqlite-wasm` (OPFS + FTS5 + sqlite-vec), wa-sqlite, sql.js.** They are the
right tools for datasets that don't fit in memory or need SQL. This one fits in memory ~6× over and
rebuilds faster than a single SQLite cold-init. See §4.

---

## 2. Evidence (measured POC)

Measured in headless Chrome via `search/__poc__/drive-bench.mjs`, reusing the repo's own harness
(`scripts/browser.mjs`: base path `/web-ai-showcase/`, fresh profile). Two viewports: **desktop
1280×800 dpr1** and **mobile 360×740 dpr3** (device-metrics + touch emulation). Full raw JSON:
`search/__poc__/results.json`. All work runs under the real `lib/worker-protocol.js` protocol.

| Metric (all in the worker)             | Desktop                                 | Mobile (360×740 dpr3) | Budget / note                      |
| -------------------------------------- | --------------------------------------- | --------------------- | ---------------------------------- |
| Corpus                                 | 2,493 docs · 6,712 terms · avgDocLen 62 | same                  | all entries                        |
| Index build (fetch)                    | 17.2 ms                                 | 15.0 ms               | one-time                           |
| Index build (inverted index)           | 34.7 ms                                 | 34.2 ms               | one-time, off-main-thread          |
| Vector build (f32 + i8 quant)          | 32.5 ms                                 | 34.9 ms               | one-time                           |
| **Total build**                        | **84.4 ms**                             | **84.2 ms**           | one-time; 0 main-thread long tasks |
| Index size (inverted)                  | 0.80 MB                                 | 0.80 MB               | IndexedDB / transfer               |
| Vectors f32 / int8                     | 3.65 MB / 0.91 MB                       | same                  | int8 is the shipping default       |
| **Lexical BM25 query** p50 / p95       | **0.1 / 0.7 ms**                        | **0.1 / 0.7 ms**      | ≪ 8 ms (120 fps)                   |
| **Semantic cosine f32** p50 / p95      | **1.0 / 1.2 ms**                        | **1.0 / 1.3 ms**      | ≪ 8 ms                             |
| **Semantic cosine int8** p50 / p95     | **1.0 / 1.1 ms**                        | **1.0 / 1.2 ms**      | ≪ 8 ms                             |
| Round-trip (main→worker→main)          | 0.9–1.4 ms                              | 0.9–1.2 ms            | incl. postMessage                  |
| **Main-thread long tasks (whole run)** | **0**                                   | **0**                 | the whole point                    |

**Reading against the budgets (invariant 15):** every query is **sub-2 ms** and there are **zero**
main-thread long tasks across build + 180 queries + probes, so the page's INP stays input-delay only
(guidance `identify-inp-causes`, `interactions-in-complex-layouts`). The one-time 84 ms build is
itself in the worker, sliced with a 50 ms-deadline `scheduler.yield()` loop (guidance
`break-up-long-tasks`), so it never blocks paint either.

**Honesty caveat — CPU is not throttled.** The "mobile" run emulates viewport + touch + DPR, **not**
a slow CPU. Applying a conservative 6–10× low-end-device multiplier to the compute: lexical ~1–7 ms,
cosine ~6–13 ms, build ~0.5–0.8 s — all still within budget, and build stays off the main thread
regardless. This is a lower bound of headroom, not a claim of low-end field numbers; real-device RUM
(`web-vitals` `onINP`, guidance `identify-inp-causes`) is the follow-up before shipping.

**Search quality sanity check:** lexical "background removal" → `rmbg-background-removal` (21.7),
`modnet-portrait-matting` (14.5), `modnet`, `birefnet`, `depth-anything`. Field-weighted BM25 (name
and task repeated) already ranks the right family first with no semantic layer.

---

## 3. Why the corpus makes this easy

- Searchable text is **~1.06 MB total** across all 2,493 rows (name, blurb, unlocks, task, modality,
  family, hfId, runtime/backend/license, + filtered inventory tags). Model `card` fields are just
  URLs (avg 59 chars) — ignored.
- Embeddings are **2,493 × 384 × 4 B = 3.65 MB** (f32) or **0.91 MB** (int8). Brute-force cosine is
  2,493 × 384 ≈ **957 k multiply-adds per query** — sub-millisecond in plain JS, measured above. An
  ANN index (HNSW) or a vector DB earns nothing until this is 10–100× bigger.
- The catalogue is a **build artifact**, regenerated by `scripts/discover-models.mjs`. The index is
  a pure function of it → deterministic generation + a content checksum is trivial (§6).

---

## 4. Options evaluated (and why rejected)

Bundle sizes are from the packages' primary distributions; SQLite/OPFS/FTS5 behaviour was **probed
live** in the POC worker (`probeOPFS`, `probeSqlite`), not assumed.

### 4a. Lexical engine

| Option                                     | Bundle                                  | GH-Pages / CSP                                                           | FTS quality                             | Verdict                                                                                              |
| ------------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Pure-JS inverted index + BM25 (chosen)** | ~0 (hand-rolled, in this repo)          | trivial — same-origin JS, no wasm, no headers                            | BM25 + field weighting; measured 0.1 ms | **Chosen**                                                                                           |
| MiniSearch (library)                       | ~7 KB min+gz                            | trivial                                                                  | BM25-ish, fuzzy/prefix built-in         | Viable alt; a dependency for what's ~120 lines. Keep as a drop-in if fuzzy/prefix tuning gets heavy. |
| `@sqlite.org/sqlite-wasm` FTS5 (official)  | ~1 MB wasm (+ jswasm glue)              | wasm is fine to fetch; OPFS needs no COOP/COEP (SAH VFS) — **confirmed** | BM25 via FTS5                           | Rejected — see below                                                                                 |
| wa-sqlite                                  | ~0.8–1 MB wasm                          | same                                                                     | FTS5 if built in                        | Rejected — extra VFS wiring, community build, same non-benefit                                       |
| sql.js                                     | ~1 MB wasm, **no OPFS**, in-memory only | fine                                                                     | FTS5 if built in                        | Rejected — no persistence story, still 1 MB for 2.5 k rows                                           |

**Live SQLite probe (real numbers):** `@sqlite.org/sqlite-wasm@3.50.1` imported from jsDelivr **in
the worker**, `installOpfsSAHPoolVfs` present, **FTS5 compiled in and `MATCH` returned correctly**.
Init cost: **1,910 ms cold** (wasm download + compile) / **33 ms warm**. So SQLite is _deployable_
here — but it costs a ~1 MB wasm transfer and a ~1.9 s first-run init to replace a 0.80 MB structure
that builds in 35 ms and queries in 0.1 ms. No FTS5 feature (BM25, prefix, phrase) is unavailable in
the pure-JS path at this scale. **Rejected on cost-for-zero-benefit, not on capability.**

### 4b. OPFS + persistence

**OPFS SyncAccessHandle works in a worker here without cross-origin isolation** — probed live:
`createSyncAccessHandle` write/read round-tripped in **2–4 ms** with `crossOriginIsolated === false`
(GitHub Pages can't set COOP/COEP; SharedArrayBuffer is unavailable — matches
`lib/media-pipeline.js` notes). So OPFS is a _real_ option. But the chosen structures (0.80 MB
index + 0.91 MB int8 vectors) serialise fine to **IndexedDB**, which needs no worker-only API and is
simpler. OPFS SAH's advantage — synchronous large-file I/O for a DB engine — is moot when there is
no DB engine. **IndexedDB chosen; OPFS documented as viable if the index ever outgrows it.**

### 4c. Vector search

| Option                                             | Verdict                                                                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Brute-force f32/int8 cosine in worker (chosen)** | 1 ms / query over 2,493 rows, measured. int8 halves memory at negligible accuracy cost for top-k ranking.        |
| sqlite-vec                                         | Rejected — pulls in the whole SQLite stack (§4a) for a 1 ms problem; ANN indexing is pointless at 2.5 k vectors. |
| HNSW (hnswlib-wasm etc.)                           | Rejected — ANN trades recall for speed that isn't needed; build/memory overhead for no latency win here.         |

**Re-evaluation trigger:** if the catalogue grows past ~50–100 k rows, or embeddings jump to
768–1024 dim, re-measure — that's where int8 SIMD, a typed-array ANN, or sqlite-vec start to pay.

---

## 5. Hybrid scoring design

Pipeline, all in the worker, per `search` request (latest-wins on channel `"search"`):

1. **Parse + expand.** Tokenise the query; optionally expand via an **alias / canonical-family map**
   (`whisper → automatic-speech-recognition` family; `sam → segment-anything`;
   `llama/qwen/phi →
   text-generation`). Expansion terms are tracked so they can be shown in the
   explanation and highlighted.
2. **Filter** the candidate set by `task / modality / license / runtime / backend / status`, size
   window (`sizeMinMB..sizeMaxMB`), `device` (only models this device can run — probed WebGPU
   adapter + rough memory ceiling), `minQuality` (inventory likes/downloads percentile), and
   `canonicalFamily`. Filtering is a bitset/`Set` intersection over precomputed per-facet posting
   lists — O(matches).
3. **Score** each surviving doc:
   - `bm25` from the inverted index (field-weighted: name/task repeated).
   - `semantic` = cosine(queryEmbedding, docVector) **iff** the embedder is `ready`; else `null`.
   - `combined = alpha · norm(semantic) + (1−alpha) · norm(bm25)` with min-max normalisation over
     the candidate set. `alpha` defaults to 0.5 and is UI-tunable; `mode` can force
     `lexical`/`semantic`.
4. **Explain.** Each hit carries `{ bm25, semantic, matchedTerms, filterHits, aliasExpanded }` →
   drives result highlighting and a "why this matched" detail (contract: `MatchExplanation`).
5. **Page** by `cursor`/`k`; return `total` (post-filter count) and `semanticApplied` so the UI can
   honestly badge "lexical only — download the intent model for semantic search."

**Semantic-optional is a hard requirement**, not a fallback afterthought: FTS + filters must be
fully useful before/without any embedding model, and a missing embedder is an **explicit** download
(invariant 12), never a silent large fetch.

**Shareable URL state.** The entire query is serialisable: `?q=...&task=...&modality=...&size=0-500`
`&license=...&device=this-device&mode=hybrid&alpha=0.5`. `Filters` (see contract) maps 1:1 to search
params, so deep links reproduce a result set and the back button restores it.

---

## 6. Index lifecycle

- **Deterministic generation.** The index is a pure function of `models.json` + `inventory/*`. Build
  in the worker (measured 84 ms) or offline as a prebuilt artifact; both yield byte-identical output
  for identical input.
- **Schema + version + checksum.** Persist
  `{ schemaVersion, checksum = sha256(normalised corpus),
  index, vectors }`. On load,
  `build({ expectChecksum })` returns `source: "persisted-cache"` when the checksum matches (skip
  rebuild) or does a fresh build when the data changed or schema bumped.
- **Incremental update.** Because entries are keyed by `slug`, a data refresh diffs added/changed/
  removed slugs and patches postings + vector rows rather than rebuilding — though at 84 ms a full
  rebuild is cheap enough to be the default until proven otherwise.
- **Embedding model:** cached-and-current → **auto-init** (no button); absent → **explicit
  download** with streamed progress (`EmbedDownloadProgress`), resumable via
  `lib/model-download.js`; WebGPU-only with no adapter → honest `unsupported`, lexical stays live.
- **FTS fallback:** semantic disabled ⇒ `mode` degrades to lexical, `semanticApplied:false` surfaced
  in the UI. The index itself never depends on the embedder.
- **Cancellation / stale-suppression / backpressure / cleanup:** inherited from
  `lib/worker-protocol.js` — every keystroke supersedes the last on channel `"search"` (older
  in-flight queries aborted, late responses dropped), the bounded queue rejects overflow
  deterministically, and `terminate()` disposes the embedder + revokes URLs.

---

## 7. Accessibility & UX constraints (for the build phase)

Guidance retained: **`accessibility`** (combobox/listbox semantics, focus management),
**`identify-inp-causes`** + **`break-up-long-tasks`** + **`schedule-tasks-by-priority`** +
**`interactions-in-complex-layouts`** (all scoring off-main-thread; results list uses
`content-visibility:auto` on rows). The search field is an accessible combobox (`role="combobox"` +
`aria-expanded` + `aria-controls` → `role="listbox"`, arrow-key navigation,
`aria-activedescendant`); a `role="status" aria-live="polite"` region announces result counts and
the embed-model state; hits are keyboard-reachable with ≥44 px targets (mobile+desktop parity).
These are requirements for the explorer, validated by this ADR's guidance consult — not implemented
here.

---

## 8. What this is NOT

No `/explore/` route, no explorer UI, no demo, no change to any existing page — verified by the
route

- conformance gates staying green (this ADR adds no `built` entry). The POC under `search/__poc__/`
  is a bench harness, not a shipped surface. Next step is to build the explorer worker to the
  `search-protocol.ts` contract and wire the accessible combobox UI at `/explore/`.

## Files

- `search/ARCHITECTURE.md` — this ADR.
- `search/search-protocol.ts` — typed worker message contract (layers on `lib/worker-protocol.js`).
- `search/__poc__/search-worker.js` — POC worker: BM25 index + cosine + OPFS/SQLite probes.
- `search/__poc__/bench.html` + `bench.js` — harness page (main-thread long-task observer).
- `search/__poc__/drive-bench.mjs` — headless-Chrome driver (desktop + mobile).
- `search/__poc__/results.json` — raw measured output.

## modern-web-guidance retained (ids)

`break-up-long-tasks`, `schedule-tasks-by-priority`, `identify-inp-causes`,
`interactions-in-complex-layouts`, `accessibility`, `performance`.
