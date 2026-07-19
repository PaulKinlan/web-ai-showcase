---
name: web-ai-showcase
description: Inventory the browser-runnable Hugging Face model universe, build comprehensive per-model demo pages (explainer + real in-browser inference + diagnostics + a wide use-case matrix), and validate them. Trigger when the user asks to run web-ai-showcase, refresh the model inventory, build/expand model pages, close the coverage gap, or validate the demos.
---

# Web AI Showcase — canonical build & inventory skill

Use in `/home/paulkinlan/web-ai-showcase`. The goal is a _comprehensive_, evidence-backed showcase
of every model you can genuinely run in a browser — not a token set. Read `CLAUDE.md` and
`AGENTS.md` first; this skill is the operational process.

## Modes (infer from the request)

- `inventory` — refresh the evidence-backed model universe + denominator.
- `build` — build the next pending eligible model(s) to the full quality bar.
- `validate` — exercise built pages (real inference, every control, a11y/console/network/visual).
- `coverage` — report built/eligible/pending/blocked with the exact denominator.
- default (`all`): inventory → build a wave → validate → coverage report.

## 1. Inventory refresh (ALWAYS run before a build wave)

`node scripts/inventory.mjs --pages 8` (evidence-only: add `--no-merge`).

- Sources = the real runtime-compatibility catalogues: **Transformers.js**
  (`library=transformers.js`) and **ONNX** (`library=onnx`) via the HF list API; **WebLLM/MLC**
  (`author=mlc-ai`, `-MLC` repos); **MediaPipe Tasks** (curated model-backed
  landmarkers/segmenters). Add credible runtimes as they emerge (ONNX Runtime Web direct,
  WebGPU-native, TFLite Web) — never treat the task list as a cap.
- **Eligibility:** a runtime-compatible exported artifact exists (ONNX for transformers.js/ORT-Web,
  `-MLC` for WebLLM, `.task/.tflite` for MediaPipe), the task is browser-demoable, and the download
  is browser-feasible. **Gated / license-restricted / too-large-for-browser = BLOCKED, but STAY in
  the denominator.** Device-only (needs WebGPU/enough RAM) stays in the denominator too.
- **Family deduplication:** group by architecture/family so millions of fine-tunes/quants/sizes of
  one model collapse to ONE representative (highest downloads) per (task, family). Never pretend
  duplicate fine-tunes are separate demos; never arbitrarily cap the catalogue either.
- Evidence is retained: `inventory/eligible.ndjson` (per-model: id, task, runtime, license, likes,
  downloads, gated, model-card URL) + `inventory/summary.json` (denominator + counts by
  task/runtime).
- Merge adds representatives to `models.json` as `status:"pending"`; it NEVER drops entries or
  downgrades `built`. Preserve exact source model IDs, license, size, quantisation, runtime, and
  card.

## 2. License / security review (before building a model)

- Record the license; for restricted/gated models keep them `blocked` (do not ship weights we
  can't).
- **Verify runnability first; don't churn on known-blocked builds.** Confirm a REAL build runs
  (dynamic-import the exact CDN build; check inference returns) before committing to a family — an
  ONNX folder existing is not enough. If it can't run today, set `status:"blocked"` + a concrete
  `blockedReason` so future waves skip it. Known-blocked (re-check only on upstream change):
  **pegasus** (no `pegasus` model class in transformers.js → `Unsupported model type`), **gliner**
  (same — `Unsupported model type: gliner`; ONNX exists but no transformers.js GLiNER class),
  **got-ocr2** (no `got_ocr2` class; no `image-text-to-text` pipeline in 3.7.5; safetensors-only),
  **git** (no `git` class; no `Xenova/git-*` ONNX), **vilt** (no `vilt` class / no
  `visual-question-answering` pipeline / no ONNX), **electra** (ONNX is encoder-only; no RTD
  discriminator head), **blip** / **bark** (gated / no usable ONNX). Never mislabel a substitute as
  the blocked family.
- Only load weights from the canonical HF repo (or MLC/MediaPipe official). No arbitrary remote
  code.
- Note device/browser requirements (WebGPU-only, RAM, secure context) in the page's at-a-glance.

## 3. Build requirements (per representative model — MORE than a toy)

Read the real **model card** + the runtime usage for the task (exact pipeline/task id, I/O shape,
special tokens/task-prompts, dtype/quantisation, size, license). Never invent an API. Then build
`models/<slug>/index.html` with ALL of:

- **Explainer** of the model + how its API works (no need to understand the maths).
- **Real local inference** — `pipeline()` (Transformers.js, `lib/webai.js`) or MLC engine
  (`lib/webllm.js`) or MediaPipe Tasks — off the main thread, WebGPU with honest fallback.
- **Download / progress / storage / cache controls** — show download size, progress, cached state,
  and a way to clear this model's cache; surface the service-worker cache status.
- **Model-specific parameters** with real controls + keyboard/accessibility semantics (labels,
  focus, roles, live regions) on every control.
- **Inspectable internals** — inputs/outputs/tokens/tensors, timings, tok/s, memory, backend used,
  quantisation. This is the "see inside" surface (probabilities, attention, embeddings, overlays…).
- **Capability + fallback diagnostics** — a real WebGPU-adapter/feature probe; labelled unsupported
  / needs-WebGPU / too-large states with exact enable steps. **Never a faked result.**
- **An expanding use-case matrix** (not one toy per rung): **multiple** Basics, **multiple**
  Practical (business/developer) scenarios, **multiple** ambitious/Wild ideas, and **credible
  Multi-model** compositions — each a real, runnable demo. Prefer adding runnable examples over
  prose.

No static code-cards; no fake output; a page that doesn't actually run the model is a failure.

## 4. Verification (before every push — no exceptions)

Serve (`python3 -m http.server 8080`) and, per page, drive **headless Chrome** and **Read the
screenshot**
(`--headless=new --no-sandbox --virtual-time-budget=25000 --screenshot=... --dump-dom`). Confirm:
the model genuinely loads and produces real output (WASM path) OR shows the honest capability
fallback (WebGPU-only in a no-GPU env); every control works (keyboard included); the "see inside"
viz renders; errors surface on the page; console/network are clean of unhandled errors; light + dark
both pass WCAG AA; the SW does not double-store (Cache Storage). Record a **browser/device matrix**
note (what path ran, what needs a GPU device). Frontend UI → run `modern-web-guidance` first.

## 4b. Model-loading UX — auto-init, shared architecture (MANDATORY)

Every page loads its model through `lib/model-loader.js` `createModelLoader(...)` (backed by
`lib/model-cache.js`). Do NOT hand-roll a bespoke "Load model" button.

- **A valid current on-device model auto-initialises** — a browser-native runtime exposes it, or
  it's already downloaded + validated in the local cache. Returning users never click "Load" for a
  current local version. Show an accessible `checking → initialising → ready` status while
  auto-starting.
- Surface a user action ONLY for: **Download** (absent — would transfer assets), **Re-download**
  (partial — assets evicted), **Update** (live revision newer than the validated cached one; the
  cached version still auto-inits, Update is optional). **Never silently re-download a large
  model.**
- The cache/version layer distinguishes **current / stale(update) / partial / evicted / absent** and
  verifies integrity (recorded files still present) before "ready". Provide a per-model
  **clear-cache** control and surface storage usage.
- Failed auto-init → **Retry / re-download / recovery**, never fake output. WebGPU-only runtimes
  gate on a real adapter probe (honest unsupported state).
- Every control + status reflects actual model/cache/runtime state, with keyboard + screen-reader
  semantics (`role="status"` `aria-live`, labelled buttons, focus).

**State-matrix re-test (every built + retrofitted page):** first visit · current cached (auto-init,
no button) · stale/update (auto-init cached + Update offered) · partial/corrupt cache (Re-download)
· offline cached use · eviction · unsupported device/browser. The visible control MUST match the
state.

## 5. Cache / SW versioning + eviction

`sw.js` owns only the app-shell cache (bump `SW_VERSION` on shell changes); transformers.js owns
`transformers-cache` — never double-store or delete it. Model pages expose a per-model "clear cache"
control. Watch origin quota; document eviction behaviour.

## 6. Coverage + completion discipline

Report **built / eligible / pending / blocked** against the exact `inventory/summary.json`
denominator after every wave, plus the pending model IDs/families. **Never say "all" / "complete" /
"done" unless `built === eligible`** (blocked + device-only remain counted). "Coming soon" cards are
pending, not done. The daily routine keeps selecting pending eligible models, wave after wave, until
the gap closes.

## 7. Git discipline

One model per commit (`git add models/<slug>/ models.json && commit && push`; rebase on race). Keep
`models.json` status accurate every commit. Shared files (`lib/`, `sw.js`, `public/`, inventory)
change deliberately, not inside a per-model build.
