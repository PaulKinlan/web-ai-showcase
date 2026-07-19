# Lineage / duplicate / variant / value pass — 2026-07-19 (FIRST evidence-backed pass)

**This is a first pass. NOT complete, NOT "all".** It establishes exact **denominators** and a
**reproducible method** for classifying the eligible model universe by lineage and value, so the
build queue can be prioritized by **unique capability** — without collapsing materially-distinct
specializations. It reviews a bounded sample deeply and classifies the long tail by cheap heuristics
with explicit confidence. Re-run the scripts to refresh; the numbers are refining lower bounds at the
current inventory scan depth.

## What this is / isn't

- **Additive + non-destructive.** Nothing here mutates `models.json`, any route, slug, or demo
  identity. It only reads the catalogue + eligible inventory and writes new files under
  `inventory/lineage/`.
- **Three denominators, kept SEPARATE** (see `denominators.json`):
  - **raw catalogue** = `models.json` entries (built + pending + blocked) — kept intact.
  - **mission baseline** = 635 eligible families @ `--pages 8` — the original mission number.
  - **eligible representatives** = the family-deduped reps in `inventory/eligible.ndjson` at the
    current scan depth (a *refining lower bound*, not a fixed total).
- Exact byte/config duplicates and most quant/format ports already collapse **within `familyKey`**
  in `scripts/inventory.mjs` *before* a representative is emitted. This pass works one level up: it
  classifies the surviving representatives and the lineage *between* them.

## Artifacts

| File | What |
| --- | --- |
| `records.ndjson` | One lineage record per eligible representative. Reviewed rows carry PROVEN evidence; the rest are heuristic with a confidence marker. Schema: `schemas/lineage-record.schema.json`. |
| `value-records.ndjson` | Value/quality records for the 148 **built** models (real evidence) + the reviewed pending sample. Evidence-gated scores. Schema: `schemas/lineage-value.schema.json`. |
| `priority.json` | Pending candidates ranked by unique high-value capability first; every deprioritized/superseded item keeps a rationale + canonical alternative. The long tail stays visible. |
| `denominators.json` | The exact bucket table + reviewed/total + confidence mix + derived family denominators. |
| `reviewed-meta.json` | The real HF metadata fetched for the reviewed sample (evidence: `cardData.base_model`, `config.architectures`, siblings). |

Regenerate: `node scripts/lineage-classify.mjs && node scripts/lineage-value.mjs`. Validate:
`node scripts/check-lineage.mjs` (runs beside `check-routes.mjs` + `check-conformance.mjs`).

## Method

1. **Evidence sources.** For a REVIEWED SAMPLE (built models + a stratified pending sample + gated
   examples, 269 ids, 264 fetched OK) we call `GET /api/models/<id>` and record the *proven* signals:
   `cardData.base_model`, `config.model_type` / `config.architectures`, sibling filenames (ONNX /
   tokenizer / preprocessor presence), and `sha`. For the long tail we reuse the `base_model:*` tags
   already captured in `eligible.ndjson` (761/2435 reps carry them: 490 `quantized`, 254 `finetune`,
   6 `merge`, 14 `adapter`) plus name/`familyKey` heuristics.
2. **Proven vs inferred are SEPARATED** in every record (`evidence.proven[]` vs `evidence.inferred[]`)
   and reflected in `confidence` (high = API-confirmed; medium = declared tag; low = name heuristic).
3. **Relationship classification** (deterministic, most-specific-first): `adapter/merge` →
   `quant-variant` (declared or name token) → `format-port` (re-exporter org / port name marker, when
   NOT a specialization) → `distillation` → `specialization-distinct` (has a base AND a domain/
   language/task signal) → `fine-tune` (has a base, no material shift) → `canonical` (no declared
   base + a real architecture or high usage) → `uncertain`.
4. **Canonical family** = reviewed `config.model_type`/architecture (proven) else `familyKey`.
5. **Value schema** — nine transparent dimensions, each with a score (0–3 or `null`), a label,
   a confidence, and evidence: capability-uniqueness, developer-value, browser-feasibility (size +
   backend static signal; latency/memory `eval-pending`), license-deployability, model-card quality,
   maintenance/provenance confidence, safety/limitations, overlap-with-stronger, showcase-interest.
   **downloads/likes are WEAK supporting context only** and never drive a score.
6. **Priority** — pending candidates are tiered `high / medium / low / superseded / blocked` and
   sorted by (tier, capability-uniqueness, license, feasibility, log-downloads tiebreak).

## Tolerance rules (deliberate, false-positive-safe)

- **Specialization beats port.** If a rep shows a domain/language/task signal it is
  `specialization-distinct` even if its name has an ONNX/port marker. Over-counting a port is benign;
  **collapsing a real specialization is the cardinal error**, so the ambiguous case is resolved in
  favour of keeping the capability visible.
- **No perf claims from metadata.** We NEVER assert two models are performance-equivalent, nor call a
  model "badly trained", from metadata. Where that judgement is needed the field is `eval-pending`.
  A model that looks weak is labelled `insufficient-evidence` / `underperforming (eval-pending)`,
  never "bad".
- **Evidence gates scores.** A numeric dimension score must carry evidence (the gate enforces this).
  Absent evidence ⇒ `null` + `insufficient-evidence` or `eval-pending`.
- **Long tail stays visible.** Deprioritized/superseded candidates are tiered and annotated with a
  rationale + canonical alternative — never deleted.
- **Reviewed vs total is always stated.** `reviewed: 141/2435` of the representatives carry proven
  evidence; the rest are heuristic. (269 ids were fetched; 141 intersect the eligible reps, the
  remainder are built models not present in the deduped rep set.)

## Eval protocol (what `eval-pending` means, and the tolerance for "equivalent")

Latency, memory, and output-quality claims require a **real browser-runtime measurement** — they are
NOT inferable from metadata. When two candidates need comparing on quality:

1. Load both via the shared `lib/model-loader.js` on the SAME device class (one desktop ≈1280×800 +
   WebGPU where required, one narrow mobile ≈360×740 WASM), fresh profile (cache-absent).
2. Run a fixed task probe set, record: cold-load time, p50/p95 inference latency, peak JS heap +
   (if WebGPU) adapter memory, tok/s, and a task-appropriate output sample.
3. **Equivalence tolerance:** two models may be called *practically equivalent for the showcase* only
   if, on the same class, latency is within **±20%**, memory within **±20%**, and the output sample
   is task-valid for both. Anything outside that stays two distinct records. A single run is never
   conclusive — surprising results are treated as pipeline artifacts and re-verified in both runtimes
   (per the measurement-rigor rule) before any equivalence is recorded.

## Critique / question / goal loop (how this pass improves)

- **Critique of this pass:** (a) `familyKey` may over- or under-merge some families (e.g. a longer
  backbone token like `detr-resnet` attributes to the backbone); (b) heuristic specialization
  detection over-flags (safe) and can miss un-named specializations (unsafe — revisit); (c) only
  141/2435 reps have proven lineage — the medium/low-confidence tail is large; (d) `canonical` vs
  `specialization-distinct` for un-based embedders is a judgement call.
- **Open questions** (accumulate here): Which pending `high`-tier embedders are genuinely distinct vs
  near-duplicates under a browser-runtime eval? How many `uncertain` (113) resolve to a known family
  with one more API lookup? Should checkpoint/size variants get their own tier separate from
  `fine-tune`? Do any `format-port`s expose a capability the base can't run in-browser (making the
  port itself the canonical browser rep)?
- **Follow-up goals** (feed the build routine, additive): deepen the reviewed sample toward the
  `high`/`medium` tiers; run the eval protocol on the top embedding candidates before building the
  next embedder; reconcile `uncertain` records; add discovered genuinely-new architectures to
  `scripts/coverage.mjs`'s taxonomy.

**Never claim complete/all.** The eligible universe is unbounded/growing with scan depth; this is a
first evidence-backed frontier, not a finished classification.
