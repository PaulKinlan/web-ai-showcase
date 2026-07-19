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
  `status:"blocked"` with a concrete `blockedReason` (evidence + what would unblock it) so the routine
  stops reselecting an impossible build. **Current known-blocked (re-check only when upstream changes):
  pegasus** (transformers.js registers no `pegasus` model class in any version — `Unsupported model
  type: pegasus` even with a full ONNX folder), **gliner** (same — `Unsupported model type: gliner`;
  ONNX export exists but no transformers.js GLiNER class/processor), **got-ocr2** (no `got_ocr2`
  class; `image-text-to-text` pipeline absent in 3.7.5; stepfun-ai repos safetensors-only), **git**
  (no `git` class; no `Xenova/git-*` ONNX repo), **vilt** (no `vilt` class / no
  `visual-question-answering` pipeline / no ONNX), **keyphrase-extraction** (no token-classification
  keyphrase model has a browser ONNX — only seq2seq generators; don't mislabel a generator/NER as
  keyphrase spans), **electra** (ONNX exports are encoder-only; the replaced-token-detection
  discriminator head is absent), **blip** / **bark** (repos gated / no usable ONNX). Never mislabel a
  substitute as the blocked family.
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
- Verify with headless Chrome + `Read` the screenshot (no chrome-devtools-mcp in the routine).
- Canonical process: **`.agents/skills/web-ai-showcase/SKILL.md`**.

## Per-model page structure

See CLAUDE.md — explainer + at-a-glance + Run-it control UI + See-inside viz + Why-it-matters +
use-case ladder (basics/practical/wild[/multi-model]) + how-the-API-works + references.
