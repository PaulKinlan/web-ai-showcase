# Transformers.js Version Policy: Shared Pin 3.7.5 vs Published 4.3.0

- **Issue**: `web-ai-showcase-djr` (P2)
- **Status**: DECIDED — Adopt Option (b): Staged Upgrade Wave with Documented Re-verification
- **Date**: 2026-09-25
- **Input Evidence**: `reports/transformers-v4-compat-matrix.md` (bead `865`), `reports/model-currency-reasoning.md` (§1, §5, §6), `reports/transformers-v4-tuple-census.json`

---

## 1. Executive Decision

The project decides **Option (b): A staged upgrade wave from transformers.js 3.7.5 to 4.3.0, managed via per-worker escape hatches and verified against real browser inference**.

- **Option (a) (permanently staying on 3.7.5) is REJECTED**: Freezing at 3.7.5 strands 315 built routes on an aging runtime that lacks the native C++ WebGPU execution provider rewrite, the ModelRegistry APIs, progress_total events, and modern model architectures.
- **An unverified big-bang bump of `lib/webai.js` is REJECTED**: 315 of 327 built routes inherit the single `TRANSFORMERS_URL` constant from `lib/webai.js`. Bumping that shared constant without individual verification violates the durable-demo compatibility contract and risks breaking published demos.

---

## 2. Evaluation of the 865 Shadow Compatibility Matrix

The shadow compatibility matrix (bead `web-ai-showcase-865`) evaluated 9 representative tuples across the 91 distinct `(task, dtype, runtime)` groups in the 327 built routes using real browser execution in headless Chrome:

| Tuple | Routes | Split-Exposed | 3.7.5 Status | 4.3.0 Status | Verdict | Findings |
|---|---|---|---|---|---|---|
| `feature-extraction-q8-wasm` | 19 | Yes | RUNS (4625ms load / 60ms infer) | RUNS (1215ms load / 32ms infer) | **same** | Numerical parity (max \|Δ\| = 1.86e-8). 3.8x faster load. |
| `text-classification-q8-wasm` | 1 | Yes | RUNS (8725ms load / 96ms infer) | RUNS (589ms load / 60ms infer) | **same** | Output text and classification labels identical. |
| `token-classification-q8-wasm` | 1 | Yes | RUNS (12632ms load / 86ms infer) | RUNS (694ms load / 73ms infer) | **same** | Output entities and spans identical. |
| `text2text-generation-q8-wasm` | 9 | Yes | RUNS (14870ms load / 99ms infer) | RUNS (2304ms load / 99ms infer) | **same** | Generated text identical. |
| `zero-shot-classification-q8-wasm` | 3 | Yes | RUNS (11936ms load / 37ms infer) | RUNS (1553ms load / 57ms infer) | **same** | Classification labels and probabilities identical. |
| `asr-ctc-q4-wasm` | 4 | Yes | RUNS (26467ms load / 2269ms infer) | RUNS (2535ms load / 1500ms infer) | **same** | CTC transcript identical despite `@huggingface/tokenizers` split. |
| `image-classification-q8-wasm` | 1 | No | RUNS (13835ms load / 1904ms infer) | RUNS (704ms load / 1553ms infer) | **differs (minor)** | Label identical (`sfw`), score Δ = 4.57e-3 due to underlying onnxruntime-web 1.22 -> 1.31 float math. |
| `fill-mask-q8-wasm` | 1 | Yes | RUNS (14046ms load / 306ms infer) | RUNS (2394ms load / 281ms infer) | **differs (structural)** | Output shape differs in token packaging without comparable text; needs handler alignment before upgrading. |
| `text-to-speech-fp32-wasm` | 10 | Yes | FAILS (missing `tokenizer.json`) | FAILS (missing `tokenizer.json`) | **both-fail** | Upstream model packaging issue; route uses custom AutoModel directly, not `pipeline()`. |

### Key Insights from the Matrix
1. **The `@huggingface/tokenizers` split is backwards-compatible for standard pipelines**:
   49 routes use `AutoTokenizer` and 21 use `AutoModelForCTC`. `asr-ctc-q4-wasm`, `token-classification`, and `feature-extraction` all ran cleanly under 4.3.0.
2. **Speed & Load Improvements**:
   WASM execution and model initialization under 4.3.0 show substantial speedups (up to 10x faster load on cached weights).
3. **Identified Risk Areas**:
   - `fill-mask` requires output schema normalization before routes on that task can adopt 4.3.0.
   - Heavy generation models (`text-generation` like Qwen 0.5B / Baguettotron and `image-text-to-text` WebGPU like SmolVLM / Moondream2) were not covered by the initial 9-tuple probe and must be validated before the shared pin moves.

---

## 3. Staged Rollout Policy (Option B Implementation)

The rollout proceeds through five bounded phases:

```
[Phase 1: Local 4.3.0 for existing 4.2.0 routes]
                       │
                       ▼
[Phase 2: Local 4.3.0 staging for proven matrix tuples]
                       │
                       ▼
[Phase 3: Remediation of fill-mask + heavy generation probes]
                       │
                       ▼
[Phase 4: Shared pin promotion (lib/webai.js -> 4.3.0) + reverse escape hatch for stragglers]
                       │
                       ▼
[Phase 5: Blocked-family list audit against 4.3.0 capabilities]
```

### Phase 1: Promote Existing Local 4.2.0 Overrides to 4.3.0
The 7 routes that already use the documented escape hatch for 4.2.0 (`apertus-1-5b`, `ernie-4-5-0-3b`, `gemma-3-270m`, `lfm2`, `qwen2.5-vl`, `sam2-segmentation`, `smoldocling-document`) test newer model architectures. They are promoted to 4.3.0 locally in their workers, verified, and recorded in `scripts/runtime-pin-allowlist.json`.

### Phase 2: Local Staging of High-Signal Proven Routes
High-signal routes from green matrix groups (feature extraction, text classification, CTC speech recognition) are staged on 4.3.0 locally with passing browser acceptance evidence before touching `lib/webai.js`.

### Phase 3: Structural Discrepancy Resolution & Heavy Probes
1. Resolve the `fill-mask` output difference so masked LM routes handle 4.3.0 structures cleanly.
2. Drive headless browser verification for representative `text-generation` and `image-text-to-text` (WebGPU) routes.

### Phase 4: Shared Pin Promotion with Reverse Escape Hatch
Once Phases 1-3 pass all acceptance criteria:
1. Any route with unresolvable regressions under 4.3.0 is pinned locally to `3.7.5` via `allowedLocalOverrides`.
   *(Note for Phase 4 execution: `checkRuntimePins()` enforces the full structural block — `version`, `slugs`, `reason` > 10 chars, `evidence` > 5 chars, `reviewedOn` YYYY-MM-DD date — on every `allowedLocalOverrides` entry, including any reverse pin holding stragglers at 3.7.5).*
2. `lib/webai.js` `TRANSFORMERS_URL` is promoted from `3.7.5` to `4.3.0`.
3. `scripts/runtime-pin-allowlist.json` updates `transformers.shared` to `"4.3.0"`.

### Phase 5: Re-verification of Known Blocked Families
The blocked families in `AGENTS.md` (e.g. `pegasus`, `gliner`, `got-ocr2`, `layoutlmv3`, `canine`, `tapas`, `deplot`, `aimv2`) were audited against 3.7.5 and 4.2.0. A dedicated pass will re-probe whether 4.3.0 adds native class or tokenizer support for any of these families.

---

## 4. Inviolable Guardrails

1. **Separation of Concerns**: Never combine the Transformers.js shared pin upgrade with raw-ORT (`onnxruntime-web`) upgrades. Raw-ORT consolidation (`web-ai-showcase-62m`) remains on its own track.
2. **Durable-Demo Contract**: No route may be broken by an upgrade. If a route fails under 4.3.0, it must be held at 3.7.5 via the local override hatch with documented evidence.
3. **Gate Enforcement**: Every local pin or shared pin change must pass `scripts/audit-model-currency.mjs --check` and `deno task gate`.
