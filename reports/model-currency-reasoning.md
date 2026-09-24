# Model + runtime currency — reasoning evaluation and upgrade matrix

Bead: `web-ai-showcase-fb9` (P1) · lane: `webai-opus` (deep reasoning) · date: 2026-09-24
Partner artifact: `reports/model-currency.{json,md}` + `scripts/audit-model-currency.mjs` (extraction by
`webai-ds-flash`, branch `audit/model-currency-fb9`).

This document is the *evaluation* half: what the extracted signals mean, which are real defects, and the
concrete upgrade matrix + issue breakdown for the fleet. Measurements are recorded honestly, including one
that **refuted my own leading hypothesis**.

## Scope and denominators

Per the denominator-discipline rule, two metrics, neither a completeness claim:

- **Catalogue:** 327 built / 66 blocked / 2299 pending = 2692 catalogued. Built is the audit population.
- **Runtime family across the 327 built** (by what the route source actually imports):

  | family | built routes | pin |
  |---|---:|---|
  | shared `lib/webai.js` (no own CDN string) | 315 | transformers.js **3.7.5** |
  | raw onnxruntime-web (self-managed session) | 43 | 1.20.1 ×8 · 1.21.0 · 1.22.0 ×1 · 1.23.0 ×1 |
  | WebLLM | 12 | **unpinned** |
  | MediaPipe Tasks | 7 | 0.10.18 |
  | transformers.js 4.2.0 escape hatch | 7 | 4.2.0 (per-worker, per policy) |

  (Families overlap: raw-ORT routes still count inside the 327; 3 routes load both transformers.js and
  raw ORT — `bert-base-turkish-cased-ner`, `model2vec-static-embeddings`, `yolo-world`.)

- **Parity backlog:** desktop ok 188/327 · mobile ok 185/327 · needs-review 139/142.

## 1. Runtime currency

| runtime | repo pin | latest published | age of gap | blast radius |
|---|---|---|---|---|
| `@huggingface/transformers` | **3.7.5** (shared) + 4.2.0 (7 routes) | **4.3.0** (2026-09-16) | one major + 3 minors | 315 routes via one constant |
| `onnxruntime-web` (direct) | 1.20.1 / 1.21.0 / 1.22.0 / 1.23.0 | **1.30.0** (2026-09-14) | up to 10 minors | 43 raw-ORT routes |
| `@mlc-ai/web-llm` | **none — floats to latest** | 0.2.85 (2026-09-08) | n/a | 12 routes |
| `@mediapipe/tasks-vision` | 0.10.18 | **1.0.1** (2026-07-31) | major | 7 routes |

Transitive ORT (what transformers.js actually carries), from the npm registry:

```
tjs 3.7.5 -> onnxruntime-web@1.22.0-dev.20250409
tjs 4.0.0 -> onnxruntime-web@1.25.0-dev.20260327 + @huggingface/tokenizers@^0.1.3
tjs 4.2.0 -> onnxruntime-web@1.26.0-dev.20260416 + @huggingface/tokenizers@^0.1.3
tjs 4.3.0 -> onnxruntime-web@1.31.0-dev.20260914 + @huggingface/tokenizers@^0.2.0
```

### What v4 actually changes (not cosmetic)

From the 4.0.0 release notes and the v4 blog post:

- **New WebGPU runtime, rewritten in C++** (native WebGPU EP, not JSEP), tested across ~200 architectures.
  Claimed ~4× on BERT-family embeddings via the `com.microsoft.MultiHeadAttention` contrib op.
- **`ModelRegistry`** — `get_pipeline_files`, `get_file_metadata` (→ real total download size *before*
  fetching), `is_pipeline_cached`, `clear_pipeline_cache`, `get_available_dtypes`.
- **`progress_total` progress event** — end-to-end download progress without manually aggregating per-file
  updates.
- **`env.useWasmCache`** (offline-capable WASM runtime files), **`env.fetch`** (custom fetch), **`env.logLevel`**.
- Tokenizers split into a standalone `@huggingface/tokenizers`; models.js split per-architecture; esbuild
  build (default web bundle 53% smaller).

Three of these map directly onto invariants this repo already maintains by hand, which is the strongest
argument for the upgrade — it is not novelty-chasing:

- `ModelRegistry.is_pipeline_cached` / `get_available_dtypes` ≈ what `lib/model-cache.js` hand-rolls for the
  auto-init policy.
- `progress_total` ≈ what `lib/download-tracker.mjs` + the adapters reconstruct from per-file events.
- `get_file_metadata` would let the download UI state a **real** total size instead of the catalogue's
  hand-recorded `sizeMB`.

### ORT direction of travel

`onnxruntime-web` is deprecating **JSEP and WebGL** in favour of the native WebGPU EP; the default export now
already includes WebGPU, and `onnxruntime-web/webgpu` is the deprecating entry. Our exposure is small and
mostly safe: of 43 raw-ORT routes, **40 request `executionProviders:["wasm"]` only**; just 3 touch WebGPU
(`music-source-separation`, `pitch-detection`, `silero-vad`).

The 8 routes pinned to **1.20.1** are the VoxPopuli ASR family, and the pin appears **deliberate** — their
worker headers record "Verified in real headless Chrome (ORT Web 1.20.1, WASM EP)". There is also a known
upstream regression (ORT #23183) where 1.21.x broke Segment-Anything-class models on WebGPU. Treat 1.20.1 as
evidence-backed until re-measured, not as rot.

### The one urgent currency defect

**`lib/webllm.js` imports `https://esm.run/@mlc-ai/web-llm` with no version.** All 12 WebLLM routes float to
whatever MLC published last. This is the only place in the repo where an upstream release can break published
demos with zero commits here — it defeats the durable-demo contract from outside. It is also the cheapest fix
in this document (one string).

## 2. Quantization trade-offs — a hypothesis I measured and disproved

**Hypothesis.** `lib/webai.js` `pickDevice()` silently degrades `webgpu → wasm`, while `loadPipeline()` passes
the caller's `dtype` through unchanged. AGENTS.md records a measured precedent that fp16 "aborts at execution
on the WASM EP [WebGPU-only fp16 compute]" (jina-embeddings-v3). If that generalises, every route declaring an
fp16-family dtype that can reach the wasm path is a latent runtime abort on a no-WebGPU device — I counted 21
such routes.

**Counter-evidence in-repo.** `models/bart-zero-shot/worker.js` documents the opposite from measurement: it
ships `q4f16` *on WASM* deliberately, because the q8 build is degraded. Both claims cannot be general.

**Measurement.** `scripts/probe-wasm-dtype.mjs` (added here) — headless Chrome via the repo's own harness, a
module worker, real transformers.js 3.7.5, `device:"wasm"` forced, `Xenova/all-MiniLM-L6-v2` (ships q8 + fp16 +
q4f16), running **inference**, not just session creation:

```
q8      RUNS  load=5293ms infer= 75ms  finite=true nonZero=true
fp16    RUNS  load=7567ms infer= 86ms  finite=true nonZero=true
q4f16   RUNS  load=7121ms infer=107ms  finite=true nonZero=true
```

Raw record: `reports/wasm-dtype-probe.json`.

**Conclusion — hypothesis refuted.** fp16-family compute is **not** categorically blocked on the WASM EP in
3.7.5. The jina-v3 abort is export/model-specific, not a general rule, and the AGENTS.md wording should be
read that way. The 21 routes are **not** a systemic defect class, and no fleet work should be spent "fixing"
them. Recording this so the next audit does not re-derive the same wrong theory from the same note.

**What does remain, much smaller:**

- **9 routes already adapt dtype per device** (`device === "webgpu" ? "fp16" : "q8"` — depth-anything family,
  dpt-depth, metric3d-depth, whisper-large-v3-turbo, smollm2-chat …). That is the good pattern; it is a
  performance/size choice, not a correctness workaround.
- **2 routes hardcode `device:"webgpu"` with no WASM path at all** — `florence2-vision`, `florence-2-large`
  (`Florence2ForConditionalGeneration.from_pretrained(..., {dtype:"fp16", device:"webgpu"})`). These need their
  honest needs-WebGPU degradation verified on a no-adapter device. Note this is *correct* API usage —
  `from_pretrained` takes `device`, unlike `loadPipeline` which takes `backend`. I checked every
  `loadPipeline` call site for that confusion: **zero** mis-keyed callers.
- dtype spread over built: q8 183 · fp32 80 · q4f16 18 · q4f16_1 12 · q4 10 · fp16 9 · int8 4 · compound 4 ·
  unrecorded 6.

## 3. Catalogue accuracy — triage of the 59 extracted findings

| kind | n | verdict |
|---|---:|---|
| taskDrift | 34 | **not defects** — Hub `pipeline_tag` vocabulary ≠ transformers.js pipeline task |
| noOnnx | 16 | **not defects** — `hfId` is the upstream base; code loads an ONNX/MLC conversion |
| error (HTTP 401) | 7 | **auditor artifact** — MediaPipe pseudo-ids (`mediapipe/face-landmarker`) aren't HF repos |
| dtypeUnrecorded | 6 | cosmetic — MediaPipe `.task` bundles have no dtype concept |
| gated | 4 | **not defects** — same base-vs-loaded split (WebLLM pulls from the MLC CDN) |
| dtypeMissing | 3 | 2 correct (non-standard repo layout, documented); **1 real** |
| gatedInformational | 1 | already labelled informational |

**taskDrift is a vocabulary mismatch, verified by spot-check:** `bge-sentence-similarity` records
`sentence-similarity` while the code drives `feature-extraction`; `bge-reranker` records `text-classification`
against a Hub `text-ranking` tag. ds-flash's own report already carries a `reading` field saying "same
behaviour". This matters because `models.json.task` is *functionally consumed* — `check-lineage.mjs`
(`slotKey`, `taskCounts`), `check-routes.mjs` (identity `{hfId, task}`), `conformance-lib.mjs` (IO-shape
templates). It is the **identity** task and must stay stable under the durable-demo contract; divergence from a
Hub tag is expected, not drift.

**noOnnx/gated is a missing catalogue field, not a broken model.** Verified examples:

| slug | catalogue `hfId` | what the code actually loads |
|---|---|---|
| mobilevit-small | `apple/mobilevit-small` | `Xenova/mobilevit-small` |
| mobilenetv3-small-100-lamb-in1k | `timm/mobilenetv3_small_100.lamb_in1k` | `onnx-community/mobilenetv3_small_100.lamb_in1k` |
| stanford-deidentifier-base | `StanfordAIMI/stanford-deidentifier-base` | `onnx-community/stanford-deidentifier-base-ONNX` |
| embeddinggemma | `google/embeddinggemma-300m` (gated) | `onnx-community/embeddinggemma-300m-ONNX` |
| llama-3-2-1b-webllm | `meta-llama/Llama-3.2-1B-Instruct` (gated) | `Llama-3.2-1B-Instruct-q4f16_1-MLC` (MLC CDN) |

The catalogue has no field for *the artifact actually fetched*. Adding one turns 20 recurring non-findings into
zero and makes future currency audits meaningful.

## 4. Real defects found

### D1 — `mms-tts-bengali`: metadata mismatch **and a currently-red immutable assertion**

- Catalogue: `dtype: "q8"`, `sizeMB: 109`.
- `models/mms-tts-bengali/worker.js:49` loads `dtype: "fp32"` (repo ships one root `model.onnx`).
- Page text: `fp32` ×4, `q8` ×0.
- Its **immutable** suite asserts `declares-quantisation` → page text must contain `"q8"`.
- `node scripts/conformance.mjs --slug mms-tts-bengali` → **13✓ 1✗**, evidence:
  `page does not mention "q8"`.

The demo is honest (page and code agree on fp32); the **catalogue metadata is wrong**, and the assertion was
derived from that wrong metadata. Per the immutability rule the assertion may not simply be edited: the fix is
`models.json`, and the derived assertion needs an entry in `conformance-migrations.json` (currently `[]`).

> **Correction (2026-09-24).** This report originally called `sizeMB: 109` "the q8 figure" and suspect. **That
> was wrong.** `webai-astra` measured the artifact and I corroborated it: `model.onnx` is 114,314,259 bytes =
> **114.31 MB = 109.02 MiB**. `109` is an accurate *MiB* rendering of the real fp32 graph, mislabeled MB — not
> evidence of a q8 artifact. The dtype was the only real defect here. The unit inconsistency turned out to be
> systemic (below), and is filed separately as `web-ai-showcase-2x0`.

I checked whether this is systemic: across all 327 built routes, **exactly one** has a `declares-quantisation`
string absent from its own `index.html`. Comparing catalogue dtype against the dtype the *primary* surface
loads yields 5 candidates, of which 4 are false positives — `yolov8-pose`, `aliked-lightglue-matching`,
`gtcrn-speech-enhancement` are raw-ORT (no transformers.js dtype option; page and catalogue agree on fp32) and
`ben2-background-removal`'s primary worker is fp16 matching its page; in all four the `q8` came from a
secondary `mm-worker.js` / `classify-worker.js` companion. So D1 is a genuine singleton, not a wave.

### D2 — the conformance gate cannot fail on a failing assertion *(highest leverage)*

`scripts/check-conformance.mjs` reads `reports/conformance/results.json`, pulls `.aggregate`, and **prints**
`fail ${a.fail}` — but never pushes it into `failures[]`. Confirmed end-to-end: with D1 red, the gate prints

```
last run: 141/231 assertions tested — pass 140 · fail 1 · blocked 0 · manual-evidenced 90

PASS — every built demo has a valid immutable suite; no weakened assertions; parity honest.
```

and **exits 0**. The gate validates suite *integrity* (present, schema-valid, not weakened, not orphaned) but
not suite *outcome*. CI runs the same script, so a red assertion cannot block a push — which is precisely how
D1 reached main. Fixing this is worth more than any individual demo fix in this audit.

Care needed: `results.json` is a partial rollup (last run = 10 suites), so the gate must not treat "not in the
last run" as failure. The correct rule is *fail on a recorded `fail`*, plus staleness handling.

### D3 — WebLLM unpinned (§1). ### D4 — no currency gate exists

`grep` across `scripts/check-*.mjs` finds **no** reference to any runtime version. Nothing detects a pin
drifting from upstream, or a new route inventing a fifth ORT version. `scripts/audit-model-currency.mjs`
(ds-flash) already has a `--check` mode diffing against `inventory/model-currency.json` — promoting it to a
gate is mostly wiring.

## 5. Upgrade matrix

Ordered by (value ÷ risk). "Blast radius" = published routes that change behaviour.

| # | change | radius | risk | evidence required before landing |
|---|---|---:|---|---|
| **U1** | Pin `@mlc-ai/web-llm` to `0.2.85` in `lib/webllm.js` | 12 | **low** | drive 2 WebLLM routes on a WebGPU device; confirm streaming + honest needs-WebGPU state |
| **U2** | Make `check-conformance.mjs` fail on a recorded assertion failure | gate | **low** | gate goes red on D1, green after D1 fixed; no false red from partial rollups |
| **U3** | Fix `mms-tts-bengali` metadata + `conformance-migrations.json` entry | 1 | **low** | suite 14/14; verify real `sizeMB`; desktop+mobile matrix |
| **U4** | Add `loadedId` (+`loadedVia`) to catalogue entries; teach the auditor | 0 (metadata) | **low** | 20 findings → 0; `check-routes` identity unchanged |
| **U5** | Record the Hub-tag ↔ pipeline-task vocabulary map | 0 (metadata) | **low** | 34 findings → 0 |
| **U6** | Consolidate raw-ORT pins 1.21.0/1.22.0/1.23.0 → one version; leave 1.20.1 alone | ~35 | **medium** | per-route real inference; the 3 WebGPU raw-ORT routes need explicit re-measurement |
| **U7** | transformers.js **3.7.5 → 4.3.0** on the shared pin | 315 | **high** | see staged plan below |
| **U8** | MediaPipe 0.10.18 → 1.0.1 | 7 | **medium** | re-run `scripts/probe-mpver.mjs` first — the current pin was chosen empirically |
| **U9** | Adopt `ModelRegistry` + `progress_total` in `lib/model-loader.js` | 315 | **medium** | only after U7; replaces hand-rolled cache/progress logic |

### U7 is not a one-shot bump

315 routes inherit one constant, so flipping it is a single-line change with a 315-route blast radius — exactly
the shape of change the durable-demo contract exists to prevent. Staged plan:

1. **Shadow-probe, no source change.** Extend `probe-wasm-dtype.mjs` into a matrix runner: for each of the ~40
   distinct (task, model-id, dtype, device) tuples across the catalogue, load under 3.7.5 **and** 4.3.0 and
   compare real output. Produces a per-architecture compat table before anything ships.
2. **Triage by API surface.** Import census across built routes: `pipeline` 80, `env` 73, `TextStreamer` 50,
   `AutoTokenizer` 34, `AutoProcessor` 29, `AutoModelForMaskedLM` 17, `Tensor` 15, `AutoModelForCTC` 13,
   `AutoModel` 10, `RawImage` 9 … The tokenizers split (`@huggingface/tokenizers`, `^0.1.3` → `^0.2.0`) makes
   the 34 `AutoTokenizer` + 13 `AutoModelForCTC` routes the ones to read first. 161 routes go through the
   shared `loadPipeline`; 99 call `from_pretrained` directly and carry more surface risk.
3. **Per-worker escape hatch first, in reverse.** The repo already has the mechanism (7 routes pin 4.2.0
   locally). Promote a handful of high-signal routes to 4.3.0 locally, gather evidence, *then* move the shared
   pin — with the same hatch available to hold any straggler at 3.7.5.
4. **Never bump the shared pin and the ORT pins in the same change.**

## 6. Candidate beads for the fleet

Sized so a lane can finish one inside its context, each with its own acceptance evidence.

| id | title | P | depends on | acceptance |
|---|---|---|---|---|
| `fb9-a` | Pin `@mlc-ai/web-llm` to 0.2.85 in `lib/webllm.js` | P1 | — | 2 WebLLM routes driven on WebGPU; streaming + needs-WebGPU state; gate suite green |
| `fb9-b` | `check-conformance.mjs` must fail on a recorded assertion failure | P1 | — | red with D1 present, green after `fb9-c`; documented staleness rule; no false red |
| `fb9-c` | Fix `mms-tts-bengali` dtype metadata + migration record | P1 | — | suite 14/14; real `sizeMB` verified; desktop+mobile matrix; `conformance-migrations.json` entry with reason+evidence |
| `fb9-d` | Add `loadedId`/`loadedVia` to the catalogue + auditor | P2 | — | the 20 noOnnx/gated findings drop to 0; `check-routes` identity unchanged |
| `fb9-e` | Record the Hub-tag ↔ pipeline-task vocabulary map | P2 | — | the 34 taskDrift findings drop to 0 |
| `fb9-f` | Promote `audit-model-currency.mjs --check` to a gate in `deno task gate` + CI | P2 | `fb9-d`,`fb9-e` | gate red on an invented 5th ORT pin; green on the current tree |
| `fb9-g` | Verify honest needs-WebGPU degradation on `florence2-vision` + `florence-2-large` | P2 | — | no-adapter device: labelled unsupported state, no blank panel, no faked output |
| `fb9-h` | 3.7.5-vs-4.3.0 shadow compat matrix (probe only, no source change) | P2 | — | per-architecture table over the ~40 distinct tuples; real inference both versions |
| `fb9-i` | Consolidate raw-ORT pins (1.21/1.22/1.23 → one); keep 1.20.1 + document why | P3 | `fb9-h` | real inference per touched route; the 3 WebGPU raw-ORT routes re-measured |
| `fb9-j` | MediaPipe 0.10.18 → 1.0.1 | P3 | re-probe | `probe-mpver.mjs` rerun across candidates; 7 routes driven |
| `fb9-k` | Shared transformers.js pin → 4.3.0, staged | P3 | `fb9-h`,`fb9-i` | staged per §5; per-worker hatch for stragglers |
| `fb9-l` | Adopt `ModelRegistry` + `progress_total` in the shared loader | P3 | `fb9-k` | auto-init policy preserved; adoption gate still 270/270 |

Suggested immediate parallel start: **`fb9-a`, `fb9-b`, `fb9-c`** (independent, low-risk, each small), with
`fb9-h` running as a long background probe since it changes no source.

## 7. What this audit did not establish

Stated explicitly so the next lane does not over-read it:

- The probe used **one** model on **one** device class. It disproves a *universal* fp16-on-WASM block; it does
  not prove every fp16 export works. Per-model evidence still rules.
- `sizeMB` accuracy was not verified *at the time of writing* (HF API rate-limited mid-audit), and my guess
  about `mms-tts-bengali` was **wrong** — see the correction in §4 and the follow-on in §8.
- No route was driven in a browser for *behaviour* in this pass beyond the dtype probe and the
  `mms-tts-bengali` conformance run. The 139/142 parity backlog is untouched.
- The 4.3.0 assessment is from release notes + registry metadata, not from running 4.3.0 against this
  catalogue. That is exactly what `fb9-h` is for.
- **Nothing here audited demo *content* for correctness.** The currency audit checks runtimes, checkpoints and
  metadata. It would not have caught the Tamil-clone residue or the missing rung in §8 — those surfaced only
  because a second lane read the page while fixing something else.

## 8. Found after publication (2026-09-24)

Three further defects surfaced while `webai-astra` executed the `mms-tts-bengali` fix. None were visible to
the currency audit, which is itself the lesson: metadata auditing does not read the page.

**The `sizeMB` unit is undefined repo-wide, and the mms-tts family is split** (`web-ai-showcase-2x0`).
Measured bytes settle it: `mms-tts-ta-ONNX` and `mms-tts-ur-ONNX` are **byte-identical** (114,301,971 B) yet
the catalogue records 109 and 114 respectively. Six routes state a MiB figure labelled "MB". `sizeMB` has no
documented unit in `AGENTS.md`, `CLAUDE.md`, or `schemas/` — that absence is the root cause, so renumbering
without defining the unit will not hold. This is user-facing: `sizeMB` is what tells a visitor how much data
lands on a metered device.

**`mms-tts-bengali` advertises a `multi-model/` rung that does not exist** (`web-ai-showcase-b1s`) — a live
404 from a published overview card, verified a singleton across all 327 built routes. Worse than the broken
link: `_questions.json` records that page as validated evidence ("multi-model chains M2M100 (en→ta) + the
TTS"), which cannot be true of a page that was never built, and names the wrong target language. The family
was cloned from `mms-tts-tamil` without adapting the artifacts; residue survives in the shipped page as
`Bengali (tam)` and "Bengali Nadu". **The demo itself is honest** — `lang="bn"`, 146 Bengali-script characters,
zero Tamil-script, real Bengali checkpoint. The copy and the critique are what lie.

**No gate checks that an advertised ladder link resolves** (`web-ai-showcase-500`). `check-portfolio-acceptance.mjs`
validates one direction only — every *on-disk* rung must appear in `acceptance.json` — so a page can advertise
a rung nobody built and every gate stays green. A sweep found 19 advertised-but-unenumerated rungs across 14
slugs, though only the Bengali one is a genuine 404; the other 18 are record gaps whose directories exist and
must **not** be mass-written into manifests without being run.

That gate gap is the same shape as D2 and `web-ai-showcase-i0h`: **an artifact that looks like enforcement
while the specific thing that broke is unenforced.** Three instances in one audit is a pattern worth naming —
when adding a gate here, check which direction it actually validates.

## 9. Bead mapping

The `fb9-*` labels in §6 were provisional. Filed IDs (`bd list --label fb9` in the repo):

| §6 | bead | P | note |
|---|---|---|---|
| `fb9-b` | `web-ai-showcase-wty` | P1 | implemented, `IN_REVIEW` |
| `fb9-a` | `web-ai-showcase-anr` | P2 | |
| `fb9-c` | `web-ai-showcase-qjp` | P2 | `webai-astra` |
| `fb9-d` | `web-ai-showcase-cqd` | P2 | |
| `fb9-e` | `web-ai-showcase-4hv` | P2 | |
| `fb9-g` | `web-ai-showcase-4f8` | P2 | |
| `fb9-h` | `web-ai-showcase-865` | P2 | |
| `fb9-f` | `web-ai-showcase-bm6` | P2 | blocked on `cqd`+`4hv` |
| `fb9-i` | `web-ai-showcase-62m` | P3 | blocked on `865` |
| `fb9-j` | `web-ai-showcase-7rm` | P3 | |
| `fb9-k` | `web-ai-showcase-9v4` | P3 | blocked on `865` |
| `fb9-l` | `web-ai-showcase-w4n` | P3 | blocked on `9v4` |
| — | `web-ai-showcase-i0h` | P2 | CI runs no node tests; 2 already red on main |
| — | `web-ai-showcase-2x0` | P2 | `sizeMB` unit, §8 |
| — | `web-ai-showcase-9tw` | P2 | migrations honour only `remove`/`weaken` |
| — | `web-ai-showcase-b1s` | P2 | dead rung + clone residue, §8 |
| — | `web-ai-showcase-500` | P2 | no ladder-link gate, §8 |
