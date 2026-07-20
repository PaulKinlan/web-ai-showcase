# Agent instructions — web-ai-showcase

`CLAUDE.md` is the canonical operator manual; read it before non-trivial edits. This file is the
short version for tools that look for `AGENTS.md`.

## Hard rules

- **Models must really run in-browser.** Real `pipeline()` / WebLLM calls on the visitor's device.
  Never present a canned result as live output. Degrade honestly (labelled needs-WebGPU / too-large
  / unsupported) — never fake success.
- **Errors on the page, not only the console.** Surface every load/inference/WebGPU failure in the
  UI.
- **Inference off the main thread**, stream progress/tokens, keep INP low (workers; break up long
  tasks).
- **Never double-store models in `sw.js`.** transformers.js owns `transformers-cache`; serve model
  files via `caches.match()` across all caches; only cache the app shell + library JS ourselves.
- **Design system only** (`public/styles.css`): warm off-white + indigo, Georgia display, humanist
  sans body, light+dark, WCAG AA. Cross-document view transitions; content-visibility on the
  catalogue.
- **Ground each page in the real model card + API.** Exact pipeline task, I/O shape, special tokens,
  dtype, size, license. Never invent an API surface.
- **Frontend is HTML/CSS/clientside-JS** → run `modern-web-guidance` FIRST for any new UI.
- **Runnable-only catalogue.** Only list models that run in a browser today; tag backend + download
  size.
- **Comprehensive + evidence-backed.** Refresh the inventory (`node scripts/inventory.mjs`) before a
  build wave; work against the real browser-runnable universe (Transformers.js/ONNX, WebLLM/MLC,
  MediaPipe) with family dedup; gated/too-large = blocked but still counted. Evidence in
  `inventory/`.
- **Don't re-attempt known-blocked builds.** Before selecting a family, verify a REAL runnable build
  exists (dynamic-import the exact CDN build; confirm inference returns) — not just that weights/an
  ONNX folder exist. If a family is not browser-runnable today, set the catalogue entry
  `status:"blocked"` with a concrete `blockedReason` (evidence + what would unblock it) so the
  routine stops reselecting an impossible build. **Current known-blocked (re-check only when
  upstream changes): pegasus** (transformers.js registers no `pegasus` model class in any version —
  `Unsupported model
  type: pegasus` even with a full ONNX folder), **gliner** (same —
  `Unsupported model type: gliner`; ONNX export exists but no transformers.js GLiNER
  class/processor), **got-ocr2** (no `got_ocr2` class; `image-text-to-text` pipeline absent in
  3.7.5; stepfun-ai repos safetensors-only), **git** (no `git` class; no `Xenova/git-*` ONNX repo),
  **vilt** (no `vilt` class / no `visual-question-answering` pipeline / no ONNX), **kosmos-2** (no
  `kosmos` class in transformers.js; no browser ONNX), **mask2former** (no `Mask2Former...` class in
  transformers.js 3.7.5/4.2 — only the image processor; no browser ONNX), **internvl** (no
  `internvl` class → `Unsupported model type: internvl`), **arabic-bert / arabic-MLM** (canonical
  Arabic masked-LMs [AraBERT/MARBERT/CAMeLBERT/etc.] are all safetensors-only — 0 Arabic ONNX in the
  HF API; the only loadable Arabic-BERT ONNX are head-less [logits undefined]; don't relabel
  XLM-R/mBERT), **subjectivity/propaganda/humor/text-chunking detection** (dedicated fine-tunes
  exist but ALL are safetensors/pytorch/flair/TF-only — no ONNX export anywhere on HF; architectures
  supported, only export missing; don't relabel sentiment/NLI/NER/POS), **plant-disease** (147-repo
  sweep found zero plant-disease ONNX — canonical MobileNetV2 is .bin-only; only 'plant' ONNX is
  houseplant-SPECIES not disease; don't relabel), **nllb-clip** (no `NLLBCLIPModel` class — custom
  open_clip; Immich ONNX is a bespoke split layout not an HF pipeline; don't relabel
  SigLIP2/Jina-CLIP), **spanbert** (released weights are a bare encoder w/ NO MLM head →
  `logits===undefined`; every ONNX is feature-extraction/QA; same as scibert/electra),
  **mxbai-rerank-v2** (v2 is Qwen2ForCausalLM decoder-reranker; the WASM-q8 export is head-less [no
  lm_head → can't compute 1/0 logit]; don't relabel v1), **boolean-qa/boolq** (runnable BoolQ
  fine-tunes ship no ONNX; the only .onnx exports are head-less encoders [no classification head →
  logits undefined]; don't relabel extractive-QA/NLI), **madlad-400** (smallest checkpoint is 3B;
  the one ONNX export is ~4.94GB INT8 w/ non-standard layout [404s on encoder/decoder_merged] —
  exceeds wasm32 4GB; don't relabel NLLB), **image-orientation** (every candidate is missing
  config/preprocessor [404] OR GATED/access-restricted — unusable from a static site; never fake
  with EXIF/heuristic), **canine** (no `canine` class in transformers.js 3.7.5/4.2 →
  `Unsupported model type: canine`; safetensors-only, only raw ONNX lacks config; don't relabel
  ByT5/subword), **audio-captioning** (no audio-captioning pipeline task in transformers.js + ZERO
  ONNX exports for any audio captioner [whisper-captioners/CNN14/BEATs — all pytorch/espnet w/
  custom generate()]; don't relabel CLAP/ASR), **deplot / pix2struct chart-to-table** (no
  `Pix2Struct` class/processor in 3.7.5 or 4.2 — DePlot/MatCha/ChartQA are all Pix2Struct; ONNX
  exists but no library support; don't relabel a captioner), **monot5** (T5ForConditionalGeneration
  class IS supported but no monoT5 checkpoint ships a transformers.js-loadable ONNX — all
  safetensors-only; don't relabel a cross-encoder), **yi-1.5** / **internlm2.5** / **starcoder2** /
  **codegemma** (distinct LLM families with no browser-runnable path: not in WebLLM
  prebuiltAppConfig AND no transformers.js-loadable ONNX — Yi/StarCoder2 have the llama/gpt_bigcode
  class but no ONNX on the Hub [only ORT-GenAI DirectML exports w/ 404 configs]; InternLM has no
  class in 3.7.5/4.2; CodeGemma has no browser-feasible ONNX. Don't relabel Gemma/Qwen-Coder/other
  LLMs), **nomic-embed-v2-moe** (no ONNX export exists anywhere [~200-repo scan:
  safetensors/GGUF/ExecuTorch only] AND transformers.js registers only dense `nomic_bert`, not the
  MoE expert-routing — don't relabel the dense v1.5), **videomae / video-classification** (no
  videomae class AND `video-classification` is not a supported pipeline task in transformers.js
  3.7.5/4.2; no browser ONNX), **parler-tts** (no `parler_tts` class in 3.7.5/4.2 →
  `Unsupported model type: parler_tts`; custom T5+DAC-codec+description-cross-attn unimplemented; no
  loadable ONNX), **layoutlmv3** (no `layoutlmv3`/`layoutlm` class in 3.7.5/4.2 despite ONNX exports
  existing — the text+bbox+patch token-classification path has no class/processor, same as gliner),
  **aimv2** (no `aimv2`/`Aimv2VisionModel` class in transformers.js 3.7.5 or 4.2; Apple repos
  safetensors/custom-code no ONNX; community ONNX lacks config — don't relabel DINOv2/CLIP),
  **image-quality/aesthetic-assessment** (no honest browser IQA: aesthetic-predictor repos are
  safetensors-only custom classes or configless 1.7GB ONNX; the loadable swin 'quality' ONNX doesn't
  track degradation [sharp≈blur≈noise] and the aesthetic-shadow ONNX is anime-only/rights-unclear —
  measured, blocked not faked), **marigold** (latent-diffusion depth pipeline —
  diffusers/VAE/UNet/DDIM, no transformers.js class; the one ONNX is a 3.46GB SD pipeline; don't
  relabel a regression depth model), **places365 / scene-classification** (no Places365/scene model
  ships a transformers.js-loadable ONNX — only .pt/.tflite/Caffe; don't relabel ImageNet ViT),
  **legal-bert / legal fill-mask** (no legal-domain fill-mask ships a browser ONNX with a real MLM
  head — exports are NSP-only [logits [1,2]] or feature-extraction-only), **tapas** (no `tapas`
  class + no `table-question-answering` pipeline in transformers.js 3.7.5/4.2; no browser ONNX;
  needs its own TapasTokenizer table-encoding + cell-selection/aggregation head),
  **question-generation** (no QG model has a v3-loadable ONNX: the family is safetensors-only, the
  one merged-decoder export is a degenerate quantized build that emits repeated-token garbage, and
  the genuine <hl> QG model uses the pre-v3 separate-decoder layout 3.7.5 can't load), **scibert /
  domain-MLM** (SciBERT scivocab ONNX is feature-extraction only — no MLM head; BioBERT's head is
  broken near-uniform; use a working domain MLM like Bio_ClinicalBERT instead, don't relabel),
  **keyphrase-extraction** (no token-classification keyphrase model has a browser ONNX — only
  seq2seq generators; don't mislabel a generator/NER as keyphrase spans), **electra** (ONNX exports
  are encoder-only; the replaced-token-detection discriminator head is absent), **blip** / **bark**
  (repos gated / no usable ONNX), **italian-sentiment** (every Italian-NATIVE sentiment classifier
  [FEEL-IT/neuraly/osiria] is safetensors/PyTorch-only — 0 Italian-native sentiment ONNX on HF; only
  `language=it` classification ONNX are NER/PII; don't relabel the built multilingual-sentiment),
  **jina-embeddings-v3** (canonical repo ships ONNX but no WASM path: fp32 external-data sidecar fails
  ORT-Web session creation, fp16 aborts at execution on the WASM EP [WebGPU-only fp16 compute], the
  required LoRA `task_id` path never runs; no q8 export; don't relabel the built jina-v2-base-en),
  **dutch-sentiment** (every Dutch-native sentiment fine-tune [RobBERT/BERTje] is safetensors-only — 0
  Dutch-native sentiment ONNX; don't relabel multilingual-sentiment), **indonesian-fill-mask** (every
  canonical Indonesian MLM [indobert p1/p2, cahya, indolem] is safetensors/Flax-only — 0 fill-mask ONNX;
  the only Indonesian ONNX carry the wrong head; don't relabel a multilingual MLM), **korean-bert-fill-mask**
  (canonical Korean MLMs are safetensors-only; the only Korean BERT ONNX are head-less feature-extraction
  [`logits===undefined`] with broken SentencePiece→WordPiece tokenizers; don't relabel a multilingual MLM),
  **wav2vec2-large-xlsr-53-dutch** / **wav2vec2-large-xlsr-53-arabic** (no Dutch/Arabic XLSR-53 ASR ships a
  browser ONNX — full Hub enumeration of every wav2vec2-large-xlsr ONNX [327 candidates] has no NL/AR export;
  the multilingual-56 model transliterates Arabic to Latin and relabeling it as the NL/AR family is forbidden.
  ALSO no XLSR-53 ONNX (blocked): **Greek**, **Polish** (Polish shipped instead as the smaller base-VoxPopuli
  `polish-voxpopuli-asr`, honestly labeled "Not XLSR-53"). XLSR langs with ONNX [zh/ja/ko/th + it/pt/es/fr/ru
  built] remain buildable — these blocks are language-specific, not the whole ASR seam),
  **gte-multilingual-reranker-base** (custom `new`/`NewForSequenceClassification` model_type absent from
  transformers.js 3.7.5 AND 4.2.0 → `Unsupported model type: new`; don't relabel a built reranker/embedder),
  **punctuation-restoration / punctuate-all** (the only browser ONNX is onnx-community/punctuate-all-ONNX, an
  auto-conversion of kredor/punctuate-all, and it is BROKEN: raw-logit inspection in headless Chrome — fp16
  AND q8 — shows near-constant output, every content token argmaxes to class 1 `.` by a ~9-logit margin with
  zero sentence-boundary/question sensitivity, so it "restores" a full stop after every word; not a
  quantization artifact [fp16 preserves the head yet is identically degenerate]; q4/q4f16 are an oversized
  823 MB MatMulNBits export that stalls at session creation. No other punctuation model ships a working
  browser ONNX. Unblock = a faithful ONNX re-export; don't relabel NER/POS as punctuation),
  **mms-tts-turkish** (no Turkish MMS-TTS VITS ships a transformers.js-loadable ONNX — Xenova ships 12
  mms-tts languages but NOT tur [401]; `facebook/mms-tts-tur` is safetensors-only; sweep of all 14
  `mms-tts-tur*` repos found zero `.onnx`. Language-specific — Xenova's other mms-tts languages remain
  buildable; don't relabel another language's MMS-TTS), **mms-tts-korean** (`Xenova/mms-tts-kor` is
  `is_uroman:true` with a 25-symbol Latin-only vocab [no Hangul] — Hangul input → empty `input_ids` →
  VITS aborts on the WASM EP; runs only on pre-romanized Latin [no uroman in transformers.js]; applies
  to every `is_uroman:true` mms-tts checkpoint — check the config first), **mms-tts-kannada**
  (`onnx-community/mms-tts-kan-ONNX` is `is_uroman:false` with a correct native-Kannada vocab and
  non-empty `input_ids`, but the ONNX export is DEFECTIVE — aborts for every input: 3.7.5 WASM aborts
  in the ORT EP for q8/fp32/q4, fp16 fails at session creation, and 4.2.0 reports `Attempting to
  broadcast an axis by a dimension other than 1. 44 by 45` [a hardcoded even seq dim no add_blank
  mms-tts input can match]; no other Kannada MMS-TTS ONNX exists. Sibling Hindi [`Xenova/mms-tts-hin`,
  plain pipeline] + Tamil [`naklitechie/mms-tts-ta-ONNX`, AutoModel + a VitsTokenizer verified
  byte-identical to the real one since that export omits tokenizer.json] are BUILT), **mms-tts-nepali**
  (NO MMS-TTS Nepali ONNX exists — Xenova/onnx-community/naklitechie families have no Nepali member,
  `facebook/mms-tts-npi` is safetensors-only+401; the only Nepali VITS ONNX are Piper/sherpa exports
  [espeak-ng phoneme input, Piper I/O signature] incompatible with the MMS char-tokenizer pipeline —
  don't substitute Piper).
  Never mislabel a substitute as the blocked family.
- **Version-pin escape hatch (isolated).** If a model's class exists only in a transformers.js newer
  than the shared 3.7.5 pin (e.g. SAM2's `Sam2Model` needs 4.2.0), pin the newer version LOCALLY in
  that one model's `worker.js` only — never bump shared `lib/webai.js`. `lib/model-cache.js` is
  version-agnostic so auto-init still works. Precedent: `models/sam2-segmentation/worker.js`.
- **Denominator discipline — report TWO metrics.** PRIMARY: the evidence-backed eligible catalogue
  (`inventory/summary.json` + `models.json`) — built / eligible + built / catalogued (+ pending);
  this is a refining LOWER BOUND (grows with scan depth — state the depth), keep the catalogue +
  summary reconciled. SECONDARY: architecture-family coverage (`node scripts/coverage.mjs`), a
  "kinds of model" view only — a high families ratio does NOT imply comprehensive catalogue
  coverage. NEVER say "all/complete/done" by either metric. Continue selecting pending eligible
  models + missing architecture families, wave after wave.
- **More than a toy per model:** download/progress/cache controls, model-specific params + full
  a11y, inspectable tokens/tensors/timings/memory/backend/quant, capability+fallback diagnostics,
  and an expanding use-case matrix (multiple basics/practical/wild + credible multi-model) — all
  runnable.
- **Auto-initialise on-device models — no unnecessary "Load" button.** Every page uses
  `lib/model-loader.js` `createModelLoader(...)` (backed by `lib/model-cache.js`). A valid current
  local model (browser-native or validated in cache) initialises AUTOMATICALLY with an accessible
  checking→initialising→ready status. Only show Download (absent), Re-download (partial/evicted), or
  Update (newer revision than the validated cached one). Never silently re-download; verify
  integrity before "ready"; failed init → retry/recovery, never fake output; per-model "clear cache"
  control. Re-test states: first visit / current cached / stale-update / partial-corrupt / offline
  cached / eviction / unsupported device. Do NOT hand-roll a bespoke Load button.
- **Durable demo compatibility contract — stable URLs · additive evolution · non-destructive.**
  Every **published** demo's identity is a durable compatibility contract. "Published" means it is
  live to users: a real route/URL + a catalogue entry (here: a `built` demo, and any honestly
  recorded `blocked` entry). The contract covers its **route/URL, slug/ID, the model/feature it
  showcases, its core behavior, controls, use-case intent, and all inbound links** — waves MUST
  preserve these.
  - **Append-only identities.** Published slugs/routes are append-only. NEVER rename, repurpose,
    replace, merge, or delete a published demo because a new wave has a different idea. (`pending`
    placeholders were never published and may be repointed.)
  - **Additive evolution.** A new use case / interaction / composition / presentation, or a
    substantially different demo, is a NEW page with a NEW stable slug — never overwrite an existing
    one to make room. Existing basic/practical/wild demos stay after ambitious ones are added.
  - **In-place fixes only when justified** (demonstrated bug, a11y/runtime/security, factual error,
    compatibility, clear quality win): retain prior identity unless change is necessary, state
    reason + evidence in the commit, regression-test. Default to the SMALLEST patch — never
    regenerate a working page from scratch when a targeted edit suffices.
  - **Read before editing.** Read the existing `models/<slug>/index.html`, its git
    history/rationale, and the route manifest before touching a built page, then make the smallest
    change.
  - **Moves need a tested alias**, recorded in the manifest — never silently break a route.
    **Blocked stays recorded** (never deleted). **Removals/moves/identity-changes are exceptional**
    — each needs a reviewed entry in `migrations.json`
    (`{id, action: "alias"|"move"|"remove"|"identity-change", from, to, reason, evidence, date}`)
    and must pass the gate. Stable ≠ frozen — improve and add freely; just never replace an old demo
    merely to present a new idea.
  - **Run the route regression gate before every push:** `node scripts/check-routes.mjs` (baseline =
- **After building a demo, regenerate lineage value-records** (`node scripts/lineage-value.mjs`) so
  every built demo has a value record — `check-lineage` FAILS otherwise (agents rebase-then-push, so
  a missing record lands on main). It's deterministic; run it before the lineage gate.

  `git show origin/main:models.json` → the published manifest, fallback
  `.route-manifest.baseline.json`). It fails on a lost published id, a `built` route whose
  `models/<slug>/index.html` is gone, a repurposed `{hfId, task}`, a deleted `blocked` record, or an
  unexplained `built`-count drop — passing additive ids, honest new `blocked`, in-place fixes, and
  anything in `migrations.json`. Refresh the manifest with `node scripts/route-manifest.mjs`
  (`--json` / `--write-baseline`). Fold the gate into every build wave.
- **Off-main-thread reference architecture (measure, don't infer).** All inference AND any pre/post
  that could exceed an 8ms frame slice runs off-main-thread; verify by measuring long tasks/INP +
  code paths. Use `lib/worker-protocol.js` (typed/versioned protocol, request ids, transfer not
  clone, AbortSignal cancel, stale-suppression, bounded queue, lifecycle+cleanup; module workers),
  OffscreenCanvas/ImageBitmap + AudioWorklet via `lib/media-pipeline.js` (non-isolated postMessage
  is the GitHub-Pages default; SAB needs COOP/COEP). The systemic hotspot is the dense-output canvas
  composite (segmentation/matting/depth/super-res) — do it in the worker, transfer RGBA/ImageBitmap
  back. Measured baseline: 143/148 @50ms, ~132/148 @8ms (5 MediaPipe are the hard exceptions).
- **Resumable downloads are real (`lib/model-download.js`):** `…/resolve/…` URL + Range/If-Range +
  IndexedDB partials + sha256-vs-git-LFS-oid verify; honest clean restart on eviction/mismatch,
  never a fake resume. Keep the auto-init/explicit-download policy.
- **First-class user input (`lib/capture-ux.js`):** upload/camera/short-video/mic, always
  user-initiated (never auto-request/auto-start); rationale + denied/unavailable/unsupported
  states + stop/retry + duration limit + track/object-URL cleanup + mobile/desktop + bundled
  fallback.
- **Rights-safe media (`media/manifest.json`):** every bundled example records source/creator/
  license/retrieval-date/local-path/dims; optimize; no hotlinking; skip unclear licensing. See
  `/architecture/` and CLAUDE invariant 15.
- Verify with headless Chrome + `Read` the screenshot (no chrome-devtools-mcp in the routine).
- **Critique → immutable conformance → goal lifecycle.** Every built demo has a versioned critique
  (`models/<slug>/_questions.json`) and an IMMUTABLE conformance suite
  (`models/<slug>/conformance.json`, sha256 `suiteHash`). Assertions are DERIVED from real
  metadata + per-task templates and cover: real-local-inference, expected I/O shape + semantic
  sanity, runtime/ backend/model-id/quant, auto-init-on-cached, progress/error/retry/offline, every
  control, no-fake-output, a11y, mobile+desktop, performance/INP, and guidance build-process.
  Immutable = fix the DEMO, never weaken the assertion; record exceptional removals in
  `conformance-migrations.json`. Run `node scripts/conformance.mjs --slug <slug>` (or `--all`) —
  deterministic, headless, download- free — emitting tested/pass/fail/blocked (blocked = honest
  device/feature-unavailable, never a pass). Critique `followUpGoals` feed `goals.json`
  (`node scripts/goals.mjs`) as an ADDITIVE backlog. Gate `node scripts/check-conformance.mjs` runs
  beside the route gate before every push + in CI. Schemas in `schemas/`. New built models get a
  suite via `node scripts/gen-conformance.mjs`.
- Canonical process: **`.agents/skills/web-ai-showcase/SKILL.md`**.

## Mobile + desktop parity — every demo usable on both, or honestly unsupported with recorded evidence

Every existing and future published demo MUST be a usable, polished experience on BOTH mobile and
desktop, unless the underlying platform feature / model / runtime is genuinely unavailable on that
class of device. This sits alongside the durable-demo contract: fix responsiveness in place with
targeted compatibility fixes — never a destructive rewrite, never a new slug to "redo" a demo.

- **Validate a mobile+desktop MATRIX, not just "it loads."** Every autonomous build or fix must
  exercise the demo at, at minimum, one representative **narrow mobile** viewport (≈360×740, touch/
  pointer + DPR≈3) and one **desktop** viewport (≈1280×800, mouse + keyboard), driving every visible
  control and state. Check, on each class: responsive layout with **no unintended horizontal
  overflow or clipped controls/text**; legible font sizes; adequate **tap targets** (≈44px min);
  **focus order
  - visible focus**; dialogs/popovers/menus open, position, dismiss, and trap focus correctly;
    orientation, **dynamic viewport** (dvh/svh, not 100vh traps) and **safe-area** insets where
    relevant; loading / progress / error / **retry** states; **zero console errors**; **no failed
    network requests**; and honest capability handling.
- **Web AI — respect mobile memory/download/storage/backend limits.** Account for constrained
  devices. Do **NOT** auto-download an absent large model just to make a test pass; an
  already-local, current, validated model still auto-initialises per the existing auto-init rule.
  When a device can't run a model, degrade honestly (labelled needs-WebGPU / needs-more-memory /
  too-large-for-this- device) with the requirements — never a blank panel or a faked result.
- **A single-class outcome needs EVIDENCE.** A desktop-only or mobile-only demo is allowed ONLY with
  direct evidence that the API, hardware capability, browser runtime, or model requirement genuinely
  makes the other class unavailable — never because the layout or interaction was left unfinished.
  Then: preserve the stable URL; show a useful, accessible, explicit **unsupported/degraded
  explanation** (requirements + a fallback/alternative where possible); NEVER blank UI, faked
  output, or a hidden/disabled-without-explanation control. Record the **unsupported class +
  evidence** in the catalogue/manifest.
- **Coverage is reported and gated.** Track exact **mobile/desktop tested-vs-total** coverage. A
  build/fix action's completion FAILS when a device class the demo is supposed to support is left
  untested or is broken. The route gate additionally FAILS if any demo is recorded broken on a class
  it claims to support. Apply this to existing demos during audits with targeted compatibility
  fixes, wave by wave — the coverage number is the backlog burn-down, and it never regresses.

**Run the responsive matrix before every push** with `node scripts/responsive-check.mjs` and record
each touched demo's result in the `support` field on its built `models.json` entry (`ok` /
`unsupported`+evidence per class). Automated-only signal marks `needs-review`, not `ok`.

## modern-web-guidance is mandatory for all frontend work

Before ANY HTML, CSS, or client-side JavaScript implementation or modification — new pages AND
targeted fixes — run/consult the **`modern-web-guidance`** skill FIRST for the specific UI/API
topic, then apply its recommendations (or explicitly justify any exception with evidence). This is
required whenever the change involves: layout, responsive mobile+desktop behavior, forms/controls,
dialogs/popovers/menus, loading/progress/error/retry states, animations/transitions, accessibility
interactions, performance / Core Web Vitals, image/model loading + caching, modern CSS, or browser
APIs.

- **Query the SPECIFIC task, not a generic memory.** A past or generic lookup does NOT count. Search
  the actual thing you're building/fixing (e.g. "responsive control panel without horizontal
  overflow", "accessible popover dismissal", "stream progress without INP regressions"), retain the
  relevant recommendation ids + evidence, and apply them — or record a justified exception.
- **Canonical source, no stale fork.** Invoke the canonical skill; if the repo needs a scripted
  call, use the published package (`npx -y modern-web-guidance@latest search "<query>"` /
  `retrieve "<id>"`) rather than copying guide text into the repo. Record the skill **source +
  version / update path** in the repo (so routines stay current) — do NOT vendor a stale copy.
- **Process validation — missing guidance is an INCOMPLETE build/critique, not a pass.** Every
  frontend change must identify which guidance was consulted (ids/queries) and how it was applied or
  why excepted. Record this in the demo's critique artifact (`guidanceConsulted`) and enforce it: a
  frontend change with no identified guidance fails completion. Feed the relevant guidance into the
  critique/questions and the immutable conformance assertions — especially responsive UI, control
  semantics, progressive enhancement, and performance.
- **Use guidance intelligently, not to chase novelty.** Prefer supported, progressive, accessible
  solutions; preserve existing stable URLs + demo identities (durable-demo contract); make targeted
  upgrades, not rewrites. chrome-platform-showcase may intentionally demo EXPERIMENTAL Chrome
  features — but the surrounding shell, fallbacks, and controls still follow current guidance +
  capability detection. web-ai-showcase must account for mobile memory/storage/download/performance
  constraints. gendn must keep reference content readable, resilient, and fast. Audit the shared
  shell/design system first, then apply additive or narrowly-scoped improvements backed by
  mobile+desktop browser evidence.

Source/version + update path recorded in `MODERN_WEB_GUIDANCE.md` (canonical skill + scripted
fallback `npx -y modern-web-guidance@latest`; no vendored copies).

## Per-model page structure

See CLAUDE.md — explainer + at-a-glance + Run-it control UI + See-inside viz + Why-it-matters +
use-case ladder (basics/practical/wild[/multi-model]) + how-the-API-works + references.
