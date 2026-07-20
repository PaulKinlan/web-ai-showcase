# web-ai-showcase — operator manual

> A hand-crafted, interactive showcase for **every AI/ML model you can run locally in a browser**.
> Read this before editing anything. The whole project is maintained largely by a Claude Code
> routine on a cron; the rules below exist so it stays a _pinnacle-of-modern-web_ showcase, not a
> model dump.

## What this project is

The Chrome Platform Showcase, but for **Web AI**. For every browser-runnable model (transformers.js
/ ONNX Runtime Web / WebLLM / WebGPU) we build:

1. **An explainer + control page** — what the model is (digested from its HF model card), how it's
   set up (task, params, quantization, backend), how you interact with it, and a **beautiful,
   interactive control UI that actually runs the model locally in the visitor's browser** — with the
   modality-specific controls a model deserves (image + prompt + special tokens for VLMs, sampling
   controls + token streaming for LLMs, live camera + overlays for vision).
2. **A "see inside the model" surface** — visualize what's happening: attention/saliency maps, token
   probabilities, embedding space, detection boxes, segmentation masks, plus a live backend /
   latency / memory readout. Make the inner workings legible without requiring the visitor to
   understand the maths.
3. **A use-case ladder** (the heart of it — one demo per rung, no ceiling):
   - **Basics** — the fundamental capability, plainly. What this model _does_.
   - **Practical** — concrete use cases a business or developer would actually ship.
   - **Wild** — out-there, novel, fun, engaging: things you could never do before this ran in a
     browser.
   - **Multi-model** (stretch) — connect this model to another (e.g. Whisper → LLM → TTS).
4. **A test harness** — a way to probe the model's behaviour and edges without reading a paper, and
   to debug what it's doing.

The thesis (Paul): people don't know what models actually _do_ or unlock. Ground Web AI in real,
runnable, legible demos — and for each model make the case for _why a business or developer should
be excited_: what it enables, what it unlocks, what it replaces.

- Hosted on **GitHub Pages** (paulkinlan.github.io/web-ai-showcase). Static — no server. Inference,
  model discovery, and caching are all client-side.
- Sibling projects:
  [chrome-platform-showcase](https://github.com/PaulKinlan/chrome-platform-showcase) (web-platform
  demos), [gendn](https://github.com/PaulKinlan/gendn) (MDN-style docs),
  [image-embedding-lab](https://github.com/PaulKinlan/image-embedding-lab) (the direct predecessor —
  its `sw.js`, measurement discipline, and design system are the north star).

## Repo layout

```
index.html            Catalogue — the model grid, rendered from models.json. content-visibility grid.
models.json           The model catalogue: one entry per model (slug, task, modality, backend, size,
                      HF id, runnable?, flags). The routine keeps this fresh from the HF API.
models/<slug>/
  index.html          The explainer + control + "see inside" page for one model.
  <use-case-slug>/index.html   One page per use-case-ladder rung (basics/practical/wild/multi-model).
lib/
  webai.js            Shared: WebGPU detection, transformers.js/WebLLM env config, model-load helper
                      with progress, the "errors on the page" helper. Browser + tooling share it.
  viz.js              Shared visualization helpers (probability bars, heatmaps, overlays) on canvas.
public/styles.css     The design system (see below). Every page links it.
sw.js                 Service worker: offline caching of the app shell + HF model blobs. DO NOT
                      double-store what transformers.js already caches (see sw.js header — hard-won).
scripts/
  discover-models.mjs Node/Deno script: query the HF API for browser-runnable models -> models.json.
.claude/routine-prompt.md   Source of truth for the freshness/build routine (see "The routine").
```

## Critical invariants (each is here to keep quality high)

### 1. Every model page MUST actually run the model in-browser — never fake it

Real `pipeline()` / WebLLM engine calls on the visitor's device. WebGPU when `navigator.gpu` exists,
WASM fallback otherwise, and the page must degrade honestly when neither works or the model is too
large — a labelled "your device can't run this / needs WebGPU / needs N GB" state, never a canned
result presented as live output. Show the real output, the real latency, the real backend used.

### 2. Errors go on the page, not (only) the console

Model loads fail, WebGPU is missing, memory runs out, a fetch 404s. Surface every failure in the UI
with a `.status.err` message the visitor can read. (Carried from image-embedding-lab.)

### 3. Inference off the main thread; keep INP low

Run inference in a Web Worker (transformers.js supports this) so the control UI stays responsive.
Stream tokens/progress incrementally. For any main-thread loop, break it up /
`await scheduler.yield()`. See modern-web-guidance: `break-up-long-tasks`,
`schedule-tasks-by-priority`.

### 4. Do not double-store models in the service worker

transformers.js owns `transformers-cache`; the SW must serve model files via `caches.match()` across
all caches, and only store the app shell + the library JS itself. Double-storing ~GB blobs evicts
the origin. (See the `sw.js` header — this bit image-embedding-lab.)

### 5. Design system only — match the AI-focus look, WCAG AA

Use the CSS variables + classes in `public/styles.css` (warm off-white paper, indigo accent, Georgia
display, humanist-sans body, light+dark). No raw hex for text/bg. Every pair hits WCAG AA. This site
should feel like the rest of Paul's AI demos.

### 6. Cross-document view transitions + content-visibility

The catalogue↔model navigation uses `@view-transition { navigation: auto }` (already in styles.css).
The catalogue grid uses `content-visibility:auto` + `contain-intrinsic-size`. See
modern-web-guidance: `cross-document-transitions`, `defer-rendering-heavy-content`.

### 7. Ground every model page in its real model card + how the API actually works

Read the HF model card and the transformers.js/WebLLM usage before building. Get the exact task,
pipeline id, input/output shape, special tokens, and recommended dtype/quantization. Never invent an
API surface. Explain how to call it in a way that doesn't require understanding the model internals.

### 8. Runnable-only catalogue

Only list models that genuinely run in a browser today: a transformers.js-compatible ONNX build, an
ONNX Runtime Web model, a WebLLM/MLC prebuilt, or a MediaPipe Tasks bundle. Mark each with its
backend

- approximate download size so a visitor knows what they're committing to. If a model needs a flag
  (e.g. WebGPU FP16), say so.

### 9. Comprehensive + evidence-backed — this is NOT a token set

The catalogue works against the **real browser-runnable universe**, not a hand-picked few. Run
`node scripts/inventory.mjs` (see the `web-ai-showcase` skill) to refresh the evidence-backed
inventory across every credible runtime (Transformers.js/ONNX, WebLLM/MLC, MediaPipe/TFLite,
WebGPU-native) with explicit eligibility + **family deduplication** (fine-tunes/quants/sizes of one
architecture collapse to one representative; gated/too-large = **blocked but still counted**). The
evidence lives in `inventory/` (retained every run).

**Verify runnability before selecting; don't re-attempt known-blocked builds.** "Eligible" means a
REAL build runs in the browser today — dynamic-import the exact CDN build and confirm inference
returns, not just that an ONNX folder or weights exist. A family that cannot run gets
`status:"blocked"` + a concrete `blockedReason` (evidence + what would unblock) so the routine stops
reselecting it. **Known-blocked as of 2026-07-20 (re-check only on upstream change): pegasus**
(transformers.js registers no `pegasus` model class in any version →
`Unsupported model type:
pegasus`), **gliner** (same failure — `Unsupported model type: gliner`;
onnx-community ships an ONNX export but no transformers.js class/processor implements GLiNER's
span/entity-prompt pipeline), **got-ocr2** (no `got_ocr2` class in 3.7.5/4.2; the
`image-text-to-text` pipeline task doesn't exist in 3.7.5; stepfun-ai repos are safetensors-only, no
ONNX), **git** (no `git` model class; no `Xenova/git-*` ONNX repo exists), **vilt** (no `vilt`
class, `visual-question-answering` is not a supported pipeline task, `ViltFeatureExtractor`
unimplemented, no ONNX export anywhere), **kosmos-2** (no `kosmos` class in any transformers.js
version; no browser ONNX — repos gated / safetensors-only), **mask2former** (no `Mask2Former...`
class in transformers.js 3.7.5/4.2 — only the image processor; no browser ONNX), **internvl** (no
`internvl` class → `Unsupported model type: internvl`; only third-party ONNX, upstream
safetensors-only), **arabic-bert / arabic-MLM** (canonical Arabic masked-LMs
[AraBERT/MARBERT/CAMeLBERT/etc.] are all safetensors-only — 0 Arabic ONNX in the HF API; the only
loadable Arabic-BERT ONNX are head-less [logits undefined]; don't relabel XLM-R/mBERT),
**subjectivity/propaganda/humor/text-chunking detection** (dedicated fine-tunes exist but ALL are
safetensors/pytorch/flair/TF-only — no ONNX export anywhere on HF; architectures supported, only
export missing; don't relabel sentiment/NLI/NER/POS), **plant-disease** (147-repo sweep found zero
plant-disease ONNX — canonical MobileNetV2 is .bin-only; only 'plant' ONNX is houseplant-SPECIES not
disease; don't relabel), **nllb-clip** (no `NLLBCLIPModel` class — custom open_clip; Immich ONNX is
a bespoke split layout not an HF pipeline; don't relabel SigLIP2/Jina-CLIP), **spanbert** (released
weights are a bare encoder w/ NO MLM head → `logits===undefined`; every ONNX is
feature-extraction/QA; same as scibert/electra), **mxbai-rerank-v2** (v2 is Qwen2ForCausalLM
decoder-reranker; the WASM-q8 export is head-less [no lm_head → can't compute 1/0 logit]; don't
relabel v1), **boolean-qa/boolq** (runnable BoolQ fine-tunes ship no ONNX; the only .onnx exports
are head-less encoders [no classification head → logits undefined]; don't relabel
extractive-QA/NLI), **madlad-400** (smallest checkpoint is 3B; the one ONNX export is ~4.94GB INT8
w/ non-standard layout [404s on encoder/decoder_merged] — exceeds wasm32 4GB; don't relabel NLLB),
**image-orientation** (every candidate is missing config/preprocessor [404] OR
GATED/access-restricted — unusable from a static site; never fake with EXIF/heuristic), **canine**
(no `canine` class in transformers.js 3.7.5/4.2 → `Unsupported model type: canine`;
safetensors-only, only raw ONNX lacks config; don't relabel ByT5/subword), **audio-captioning** (no
audio-captioning pipeline task in transformers.js + ZERO ONNX exports for any audio captioner
[whisper-captioners/CNN14/BEATs — all pytorch/espnet w/ custom generate()]; don't relabel CLAP/ASR),
**deplot / pix2struct chart-to-table** (no `Pix2Struct` class/processor in 3.7.5 or 4.2 —
DePlot/MatCha/ChartQA are all Pix2Struct; ONNX exists but no library support; don't relabel a
captioner), **monot5** (T5ForConditionalGeneration class IS supported but no monoT5 checkpoint ships
a transformers.js-loadable ONNX — all safetensors-only; don't relabel a cross-encoder), **yi-1.5** /
**internlm2.5** / **starcoder2** / **codegemma** (distinct LLM families with no browser-runnable
path: not in WebLLM prebuiltAppConfig AND no transformers.js-loadable ONNX — Yi/StarCoder2 have the
llama/gpt_bigcode class but no ONNX on the Hub [only ORT-GenAI DirectML exports w/ 404 configs];
InternLM has no class in 3.7.5/4.2; CodeGemma has no browser-feasible ONNX. Don't relabel
Gemma/Qwen-Coder/other LLMs), **nomic-embed-v2-moe** (no ONNX export exists anywhere [~200-repo
scan: safetensors/GGUF/ExecuTorch only] AND transformers.js registers only dense `nomic_bert`, not
the MoE expert-routing — don't relabel the dense v1.5), **videomae / video-classification** (no
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
class + no `table-question-answering` pipeline in transformers.js 3.7.5/4.2; no browser ONNX; needs
its own TapasTokenizer table-encoding + cell-selection/aggregation head), **question-generation**
(no QG model has a v3-loadable ONNX: the family is safetensors-only, the one merged-decoder export
is a degenerate quantized build that emits repeated-token garbage, and the genuine <hl> QG model
uses the pre-v3 separate-decoder layout 3.7.5 can't load), **scibert / domain-MLM** (SciBERT
scivocab ONNX is feature-extraction only — no MLM head; BioBERT's head is broken near-uniform; use a
working domain MLM like Bio_ClinicalBERT instead, don't relabel), **keyphrase-extraction** (no
token-classification keyphrase model ships a browser-loadable ONNX — a 151-repo scan found only
seq2seq generators, which can't produce per-token B/I/O spans; don't mislabel a generator or plain
NER as keyphrase span extraction), **electra** (ONNX exports encoder-only; RTD discriminator head
absent), **blip** / **bark** (gated / no usable ONNX), **italian-sentiment** (every Italian-NATIVE
sentiment classifier [FEEL-IT `MilaNLProc/feel-it-italian-sentiment`, `neuraly/bert-base-italian-cased-sentiment`,
`osiria/bert-tweet-italian-uncased-sentiment`] is safetensors/PyTorch-only — 0 Italian-native
sentiment ONNX on HF; the only `language=it` classification ONNX exports are NER/PII, not sentiment;
architecture [UmBERTo/BERT/XLM-R] is supported, only the export is missing; don't relabel the built
multilingual-sentiment), **jina-embeddings-v3** (canonical `jinaai/jina-embeddings-v3` ships ONNX but
no WASM path: fp32 `model.onnx`+`model.onnx_data` external-data sidecar fails at ORT-Web session
creation [`Module.MountedFiles is not available`], fp16 `model_fp16.onnx` creates but aborts at
execution on the WASM EP [fp16 compute is a WebGPU-only kernel path], and the required LoRA `task_id`
path never runs [`Missing the following inputs: task_id`] and still aborts when supplied; no q8/int8
export exists; a third-party Q8 re-export is non-canonical; don't relabel the built jina-v2-base-en),
**dutch-sentiment** (every Dutch-native sentiment/emotion fine-tune [RobBERT/RobBERTje/BERTje] is
safetensors/PyTorch-only — 0 Dutch-native sentiment ONNX on HF; `language=nl` classification ONNX are
multilingual/other-language/non-sentiment; don't relabel multilingual-sentiment), **indonesian-fill-mask**
(every canonical Indonesian MLM [indobenchmark indobert p1/p2, cahya, indolem, flax RoBERTa] is
safetensors/PyTorch/Flax-only — 0 fill-mask ONNX; the only Indonesian ONNX carry the wrong head
[classification/NER/embedding]; don't relabel a multilingual MLM), **korean-bert-fill-mask** (canonical
Korean MLMs [klue bert/roberta, kcbert, bert-kor-base, kobert, KR-BERT] are safetensors-only; the only
Korean BERT ONNX [`Xenova/kobert`, `skt/kobert-base-v1`] are head-less feature-extraction exports
[`out.logits===undefined`] with broken SentencePiece→WordPiece tokenizers [common words → `[UNK]`];
don't relabel a multilingual MLM), **wav2vec2-large-xlsr-53-dutch** / **wav2vec2-large-xlsr-53-arabic**
(no Dutch or Arabic wav2vec2-large-xlsr-53 ASR ships a browser-loadable ONNX — a 135-repo Dutch sweep +
187-repo Arabic sweep + full Hub enumeration of every `wav2vec2-large-xlsr` ONNX [327 candidates: km/th/zh/
fr/de/it/ja/ko/pt/es/en/fi/faroese/icelandic/ru/tr + one multilingual-56] found NO Dutch and NO Arabic
export; the one NL NeMo ONNX declares `model_type: wav2vec2-ctc` → `Unsupported model type`; the
multilingual-56 model runs but transliterates Arabic to Latin and using it as the NL/AR family would be
forbidden relabeling. NOTE: other XLSR languages WITH ONNX mirrors [zh/ja/ko/th/tr/fi …] remain buildable —
this block is language-specific, not the whole XLSR seam), **gte-multilingual-reranker-base** (Alibaba GTE
multilingual reranker uses a custom `new`/`NewForSequenceClassification` model_type absent from transformers.js
3.7.5 AND 4.2.0 → `Unsupported model type: new`; the same `new` type affects gte-multilingual base variants —
don't relabel a built reranker/embedder), **punctuation-restoration / punctuate-all** (the only browser ONNX
is onnx-community/punctuate-all-ONNX, an auto-conversion of kredor/punctuate-all, and it is BROKEN:
raw-logit inspection in headless Chrome [fp16 AND q8] shows near-constant output — every content token
argmaxes to class 1 `.` by a ~9-logit margin with zero sentence-boundary/question sensitivity, so it
predicts a full stop after every word; NOT a quantization artifact [fp16 preserves the head yet is
identically degenerate]; q4/q4f16 are an oversized 823 MB MatMulNBits export that stalls at session
creation; no other punctuation model ships a working browser ONNX — unblock = a faithful ONNX re-export;
don't relabel NER/POS as punctuation), **mms-tts-turkish** (no Turkish MMS-TTS VITS ships a
transformers.js-loadable ONNX — Xenova ships 12 mms-tts languages [incl. kor] but NOT tur
[`Xenova/mms-tts-tur`→401]; onnx-community has only kan/aka; `facebook/mms-tts-tur` is
safetensors/pytorch-only; a sweep of all 14 `mms-tts-tur*` repos found zero `.onnx`; don't relabel
another language's MMS-TTS. NOTE: the block is language-specific — many other Xenova mms-tts
languages [e.g. vie (built), hin, …] remain buildable), **mms-tts-korean** (`Xenova/mms-tts-kor` is
`is_uroman:true` with a 25-symbol LATIN-only vocab [NO Hangul] — feeding Hangul yields empty
`input_ids` and the VITS graph aborts on the WASM EP; it runs ONLY on pre-romanized Latin, and
transformers.js ships no uroman romanizer, so a Latin-input box labeled "Korean" while Hangul crashes
is not honest Hangul Korean TTS; q4 doesn't exist [`Unsupported model type: vits`], fp16 fails at
session creation. Applies to every `is_uroman:true` mms-tts checkpoint — check the config before
selecting), **mms-tts-kannada** (`onnx-community/mms-tts-kan-ONNX` is `is_uroman:false` with a
correct native-Kannada vocab [identical to `facebook/mms-tts-kan`] and NON-empty `input_ids`, but the
ONNX EXPORT is defective and aborts for EVERY input: transformers.js 3.7.5 WASM aborts in the ORT EP
for q8/fp32/q4 [error codes 86122760 / 237829256 / 231992672], fp16 fails at session creation
[RandomNormalLike float16], and 4.2.0 gives the smoking gun — `Attempting to broadcast an axis by a
dimension other than 1. 44 by 45` [a hardcoded even sequence dim that no add_blank mms-tts input
(2N+1, odd) can ever match]. Defective export, not a device limit; no other Kannada MMS-TTS ONNX
exists. NOTE: sibling Hindi (`Xenova/mms-tts-hin`) and Tamil (`naklitechie/mms-tts-ta-ONNX`, loaded
via AutoModel + a VitsTokenizer verified byte-identical to the real one, since that export omits
tokenizer.json) are BUILT). Never mislabel a substitute as the blocked family.

**Version-pin escape hatch (isolated).** A model whose class exists only in a transformers.js newer
than the shared 3.7.5 pin (e.g. SAM2 — `Sam2Model` lands in 4.2.0, absent from 3.7.5) may pin the
newer version LOCALLY in that one model's `worker.js` import — never bump shared `lib/webai.js` or
other pages. `lib/model-cache.js` is version-agnostic (scans Cache Storage by modelId), so
`createModelLoader` auto-init still works. Precedent: `models/sam2-segmentation/worker.js` pins
`@huggingface/transformers@4.2.0`; everything else stays 3.7.5. Verify the pin stays scoped to that
worker.

Cover the full capability range — classification, NER, embeddings/ reranking/search,
summarisation/translation/generation, ASR, audio classification, TTS, image classification,
zero-shot image, detection, segmentation, depth/normal, OCR/doc, captioning/VQA/VLM, background
removal, pose/hand/face landmarks, browser-feasible generation, and LLM chat/tool/RAG via both
Transformers.js and WebLLM — and add categories the inventory surfaces.

### 10. Denominator discipline — never claim "complete" at a toy count

Report **TWO denominators**, both honestly:

1. **PRIMARY — the evidence-backed eligible catalogue.** `node scripts/inventory.mjs` scans the real
   browser-runnable HF universe and records `inventory/summary.json` (eligible families) + merges
   representatives into `models.json` (`built` / `pending`). Report **built / eligible** and **built
   / catalogued (+ pending)**. IMPORTANT: this denominator is a **refining LOWER BOUND, not a fixed
   number** — the deduped eligible-family count grows with scan depth (635 @ `--pages 8` → 754 @ 10
   → 1288 @ 20 → 2355 @ 40 …) because the transformers.js/ONNX long tail is effectively unbounded.
   `inventory/summary.json` is SELF-DOCUMENTING (`scanPages`, `depthCurve`, and `missionBaseline` =
   {scanPages: 8, eligibleFamilies: 635} — the original mission number); the 635→2355 change is ONLY
   the scan depth, not a definition change (same sources + eligibility + dedup) — whenever you
   deepen the scan, say so. Always state the scan depth and treat it as a lower bound; keep
   `models.json` (catalogue) and `inventory/summary.json` (eligible universe) reconciled +
   consistent (same scan depth), and note that the catalogue is a growing subset of the scanned
   eligible universe. Blocked/device-only stay counted; "coming soon" is pending.

2. **SECONDARY — architecture-family coverage.** `node scripts/coverage.mjs` reports built vs a
   bounded curated taxonomy of ~60+ canonical _kinds_ of model (BERT, ViT, Whisper, Llama, DETR,
   SAM, CLIP, MusicGen, MediaPipe, …). This is a useful "have we covered the distinct kinds" view —
   a SECONDARY metric only. **A high architecture-family ratio (e.g. 40/63) does NOT imply
   comprehensive catalogue coverage** and must never be presented as such.

**Never say "all" / "complete" / "done"** by either metric — the eligible universe is
unbounded/growing and new architectures keep surfacing. The mission (a representative demo for every
capability task and every architecture family, while chipping into the eligible catalogue) is a
moving frontier.

### 11. More than a toy per model

Every representative model page needs: download/progress/storage/cache controls; model-specific
parameters with full keyboard/a11y semantics; inspectable
inputs/outputs/tokens/tensors/timings/memory/ backend/quantisation; real capability + fallback
diagnostics; and an **expanding use-case matrix** — _multiple_ basics, _multiple_ practical
(business/developer), _multiple_ ambitious/wild, and credible multi-model compositions — each a real
runnable demo, never a static card or fake output.

### 12. Auto-initialise on-device models — no unnecessary "Load" button

Model loading is a SHARED architecture. **Every page uses `lib/model-loader.js`
`createModelLoader(...)`** (backed by `lib/model-cache.js`):

- If a **valid current** version is already on-device — a browser-native runtime exposes it, OR it's
  already downloaded + validated in the local cache — the page **auto-initialises it and just
  works.** Returning users never click "Load" for an already-local current version.
- Show an accessible **checking → initialising → ready** status while auto-starting (`role="status"`
  `aria-live`).
- Only surface **Download** (model absent — loading would transfer assets), **Re-download** (cache
  partial — assets evicted), or **Update** (live revision newer than the validated cached one; the
  cached version still auto-inits — Update is optional). **Never silently re-download a large
  model.**
- Distinguish **current / stale(update) / partial / evicted / absent**; verify integrity (recorded
  files still present) before "ready".
- Failed auto-init exposes **Retry / re-download / recovery** — never fake output. WebGPU-only
  runtimes gate on a real adapter probe (honest unsupported state). Provide a per-model **"Clear
  cached model"** control + storage usage. The visible control + status MUST match actual state,
  with keyboard + SR semantics.
- Re-test every built page across: **first visit · current cached · stale/update · partial/corrupt ·
  offline cached · eviction · unsupported device.**

### 13. Durable demo compatibility contract — stable URLs · additive evolution · non-destructive

Every **published** demo's identity is a durable compatibility contract. "Published" means it is
live to users: it has a real route/URL and a catalogue entry (for this repo: a `built` demo, and any
`blocked`/unsupported entry that is honestly recorded). A published demo's contract covers its
**route/URL, its slug/ID, the model or platform feature it showcases, its core behavior, its
controls, its use-case intent, and all inbound links.** Routine and agent waves MUST preserve these.

- **Append-only identities.** Published slugs/IDs/routes are append-only. NEVER rename, repurpose,
  replace, merge, or delete an existing published demo because a new wave has a different design
  idea. (Catalogue entries that were never published — e.g. `pending` placeholders with no route —
  are not under contract and may be repointed.)
- **Additive evolution.** A newly discovered use case, interaction concept, model/feature
  composition, presentation approach, or a substantially different demo is added as a NEW page with
  a NEW stable slug + catalogue entry. Do NOT overwrite or repurpose an existing demo to make room.
  Existing basic/practical/wild demos stay available after more ambitious ones are added.
- **In-place fixes only when justified.** Change an existing published demo in place ONLY for a
  demonstrated bug, accessibility/runtime/security issue, factual error, compatibility problem, or
  clear quality improvement. Retain prior behavior/identity unless changing it is necessary; state
  the reason + evidence in the commit message; regression-test the change. Default to the SMALLEST
  patch — never regenerate a working page from scratch when a targeted edit suffices.
- **Moves need a tested alias.** If a URL absolutely must move, keep the old route working via a
  tested permanent redirect/alias recorded in the route manifest. Never silently break a route.
- **Blocked stays recorded.** Unsupported/blocked entries remain honestly recorded (status
  `blocked`), never deleted.
- **Read before editing.** Before editing, read the existing implementation, its history/rationale,
  and the route manifest, then make the smallest change that satisfies the goal.
- **Removals/moves are exceptional.** Any removal, rename, route move, or identity change requires
  an explicit reviewed **migration record** (`MIGRATIONS`/`migrations.json`) and must pass the route
  regression gate. Stable does NOT mean frozen — improve existing demos when justified, and add new
  demos/use cases freely; just never replace an old one merely to present a new idea.

**Gate before every push:** run the route regression gate (`node scripts/check-routes.mjs`). It
compares the previously published manifest against the working tree and fails on any missing
published ID, deleted route, renamed/repurposed slug, changed published identity, or unexplained
concept-count reduction — while allowing additive entries, honest `blocked` records, and in-place
fixes. Exceptional removals/moves must be listed in the migration record with reason + evidence.

**Contract mechanics (this repo).** The published manifest is emitted by
`node scripts/route-manifest.mjs` (`--json` to print, `--write-baseline` to refresh
`.route-manifest.baseline.json`): one `{id, route, identity: {hfId, task}, status, aliases[]}` entry
per `built` or `blocked` model (`pending` placeholders are excluded — they were never published).
The gate `node scripts/check-routes.mjs` derives the baseline from
`git show origin/main:models.json` (falling back to `.route-manifest.baseline.json` offline) and
fails on a lost published id, a `built` route whose `models/<slug>/index.html` is gone, a repurposed
`{hfId, task}`, a deleted `blocked` record, or an unexplained `built`-count drop. Record every
exceptional change in `migrations.json` (git-tracked array of
`{id, action: "alias"|"move"|"remove"|"identity-change", from, to, reason,
evidence, date}`); the
gate consults it to permit the change. **Read before editing:** open the existing
`models/<slug>/index.html`, read its git history/rationale, and check the route manifest before
touching a built page; default to the smallest patch and never regenerate a working page from
scratch when a targeted edit suffices. Every build wave runs the gate before pushing.

> Canonical process: **`.agents/skills/web-ai-showcase/SKILL.md`** (inventory refresh, dedup,
> license/ security review, build requirements, cache versioning/eviction, verification, coverage
> denominators, completion discipline). AGENTS.md has the short rules.

### 14. Critique → immutable conformance → goal lifecycle (per-demo quality contract)

Every built model page runs a WEB-AI-domain lifecycle (analogous to chrome-platform-showcase's, not
a copy): **coverage/build → critique → immutable conformance → validation → goal-setting.** All
ADDITIVE — it never rewrites a page or churns a URL; it sits alongside invariant 13.

- **Critique** (`models/<slug>/_questions.json`, versioned + mutable): a rubric scored across the
  domain dimensions — real-local-inference correctness, expected I/O shape + semantic sanity,
  runtime/backend/model-id/quantisation honesty, download/cache/current-version auto-init behaviour,
  progress/error/retry/offline states, every visible control, no-fake-output, accessibility,
  mobile+desktop matrix, performance/INP — with **REAL retained evidence** (screenshot path, console
  excerpt, latency, output sample), plus `guidanceConsulted` (empty on a frontend critique =
  INCOMPLETE), `openQuestions`, and `followUpGoals`. Schema: `schemas/critique.schema.json`.
- **Immutable conformance** (`models/<slug>/conformance.json`, hashed): per-model assertions DERIVED
  from real `models.json` metadata + per-task templates, covering exactly those categories.
  **Immutable = once committed, an assertion is never deleted, weakened, or regenerated to go green
  — you fix the DEMO.** `suiteHash` = sha256 of the normalized assertions;
  `scripts/check-conformance.mjs` recomputes it and diffs against `origin/main`, failing on any
  removed/weakened assertion without a `conformance-migrations.json` record. Adding assertions
  (coverage grows) is always allowed. Schema: `schemas/conformance.schema.json`. Generate suites for
  new built models with `node scripts/gen-conformance.mjs` (it never overwrites an existing suite).
- **Validation** (`node scripts/conformance.mjs --slug <slug>` / `--all`): a deterministic,
  headless-Chrome-backed, download-free runner emits exact **tested / total · pass / fail /
  blocked** (+ manual-evidenced awaiting an agent verdict). `blocked` = the device/feature is
  GENUINELY unavailable (e.g. a WebGPU-only model on a no-GPU runner showing its honest needs-WebGPU
  state) — explicit, never a pass. It NEVER auto-downloads an absent large model to force a pass; a
  fresh profile keeps every model cache-absent. Rollup: `reports/conformance/index.html` +
  `results.json`.
- **Goal-setting** (`node scripts/goals.mjs` → `goals.json`, schema `schemas/goals.schema.json`):
  critique `followUpGoals` accumulate into an ADDITIVE backlog the routine consumes to pick the next
  NEW demo or targeted in-place fix — never to replace a stable page.
- **Gate:** `node scripts/check-conformance.mjs` runs beside `check-routes.mjs` before every push
  (and in CI). It FAILS on a built demo with no suite, a duplicate/orphan suite id, a malformed
  artifact, a weakened/removed assertion without a migration record, or a touched demo left
  untested/broken on a supported device class — and REPORTS the coverage denominators. Author
  genuine suites by DERIVING from real metadata; never fake, never claim "complete/all" (built-demo
  count is a moving frontier).

### 15. Off-main-thread reference architecture + first-class capture + rights-safe media

web-ai-showcase is a production-grade REFERENCE ARCHITECTURE for running on-device models without
freezing the main thread. All ADDITIVE; preserve stable routes; keep the auto-init/current-version/
explicit-download policy (invariant 12).

- **Off-main-thread by default; MEASURE, don't infer.** All inference AND any pre/post-processing
  that could exceed an **8ms frame slice** runs off the main thread (16ms already misses 60fps).
  Verify by measuring long tasks/INP + inspecting code paths — never assume compliance from "uses a
  library". The systemic hotspot found by measurement is the **dense-output canvas composite**
  (segmentation/ matting/depth/super-res doing `getImageData`→per-pixel→`putImageData` on the main
  thread — fine on a thumbnail, ~40ms at 1080p): do it in the worker and **transfer** an RGBA buffer
  / ImageBitmap back. Measured baseline (bundled samples): **143/148 compliant @50ms bar** (only the
  5 MediaPipe demos —
  `face-landmarker`/`hand-landmarker`/`pose-landmarker`/`gesture-recognizer`/`face-detector` — run
  inference synchronously on the main thread), **~132/148 @8ms bar** with full-res inputs. See
  `scratchpad/worker-compliance-baseline.md`.
- **Shared worker architecture** (`lib/worker-protocol.js`): typed/**versioned** message protocol,
  per-request ids, **transfer** ArrayBuffers/ImageBitmaps (not clone), explicit progress,
  cancellation via AbortSignal, stale-response + supersession suppression, bounded
  queue/backpressure, worker lifecycle states (starting→ready→busy→error→terminated), deterministic
  dispose + object-URL cleanup. Dedicated **module** workers. Image/video preprocessing via
  OffscreenCanvas/ImageBitmap with a measured main-thread fallback; real-time audio via
  **AudioWorklet** posting bounded chunks/ring-buffer to a worker — SharedArrayBuffer needs
  COOP/COEP which **GitHub Pages cannot set**, so the non-isolated postMessage path is the DEFAULT
  (`lib/media-pipeline.js`).
- **Resumable downloads are REAL here (researched + proven), not faked** (`lib/model-download.js`):
  fetch the `…/resolve/…` URL, `Range: bytes=<received>-` + `If-Range:<strong-etag>` (206 append /
  200 or 412 clean restart), IndexedDB partials, `storage.persist()`, and **WebCrypto SHA-256
  verification against the git-LFS oid** before ready. Evicted/mismatched partial → honest clean
  restart, never a fake resume. Evidence: `scratchpad/hf-download-resume-findings.md`.
- **First-class user input** (`lib/capture-ux.js`): upload / camera snapshot / short webcam video /
  mic — **always user-initiated** (never auto-request or auto-start capture). Accessible permission
  rationale before requesting; explicit denied/unavailable/unsupported states; stop/retry; bounded
  duration + countdown; deterministic cleanup of tracks + object URLs; mobile+desktop; a REQUIRED
  bundled fallback example. Camera/mic permission stays user-initiated.
- **Rights-safe media library** (`media/manifest.json`): every bundled example records source URL,
  creator, license (SPDX/explicit), retrieval date, local path, dimensions. Optimize dims/format; NO
  hotlinking; skip anything without a clear license (record the gap). The 2D→3D / removable-object
  family (depth/matting/segmentation/SAM/CLIPSeg/panoptic) gets a broad, varied gallery.
- **The `/architecture/` page** documents all of the above with copyable code. New demos + retrofit
  waves MUST follow this; conformance assertions cover worker-compliance, capture-UX states, and
  bundled-media provenance. Report worker-compliant/tested/total + long-task before/after; never
  claim complete/all until the denominator closes.

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

The skill source/version + update path is recorded in `MODERN_WEB_GUIDANCE.md`: canonical skill
`modern-web-guidance` with the scripted fallback `npx -y modern-web-guidance@latest` — no vendored
guide copies.

## Modern-web references (retrieve before building the relevant surface)

`npx -y modern-web-guidance@latest retrieve <id>` — `cross-document-transitions`,
`defer-rendering-heavy-content`, `break-up-long-tasks`, `schedule-tasks-by-priority`,
`identify-inp-causes`. Frontend is HTML/CSS/clientside-JS: run modern-web-guidance FIRST for any new
UI.

## The per-model page — required structure

`models/<slug>/index.html`:

1. `<link rel="stylesheet" href="/web-ai-showcase/public/styles.css">`, crumbs back to `/`.
2. Eyebrow (task · modality · backend · size), H1 (friendly model name), lede (one-sentence
   what/why).
3. **At a glance** — HF id (linked), task/pipeline, params, quantization/dtype, backend
   (WebGPU/WASM), download size, license, Baseline-of-the-web-APIs-it-needs.
4. **Run it** — the interactive control UI. Load button with real progress; modality-specific
   inputs; real output; latency + backend readout. Off-main-thread.
5. **See inside** — at least one real visualization of the model's working (probabilities,
   attention, embeddings, overlays), with a short "what am I looking at" note.
6. **Why it matters** — what this unlocks for a business/developer; 2-4 concrete "you could build…".
7. **Use cases** — links to the ladder pages (basics/practical/wild[/multi-model]).
8. **How the API works** — the minimal `pipeline()`/WebLLM snippet, explained.
9. **References** — HF model card, the spec/paper, transformers.js docs. Byline footer.

Each use-case page is a full interactive demo of one idea, same shell, crumbs back to the model
page.

## The routine

A Claude Code cron (like the showcase's) keeps the catalogue fresh: discover new browser-runnable
models via the HF API, build the explainer + first use-case pages, one commit per model, headless-
verified. Source of truth: `.claude/routine-prompt.md`; runs on Opus, bootstrap-reads this repo.

## Testing checklist (before pushing)

- `deno fmt --check` (HTML/CSS excluded); `deno check scripts/*.mjs lib/*.js` where typed.
- Serve locally (`python3 -m http.server` or
  `deno run --allow-net --allow-read jsr:@std/http/file-server`) and open each new page.
- Verify in a real browser: the model actually loads and runs (WebGPU path AND the WASM fallback
  path), the "see inside" viz renders, errors surface on the page, light+dark both pass contrast,
  keyboard works, and the SW doesn't double-store (check Application → Cache Storage).
