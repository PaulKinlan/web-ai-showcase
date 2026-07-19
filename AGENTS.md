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
- **Denominator discipline.** The coverage denominator is distinct ARCHITECTURE FAMILIES
  (`node scripts/coverage.mjs`), NOT raw repos — the raw HF browser-runnable count is unbounded and
  grows with scan depth, so it's evidence (`inventory/`) not a baseline. Report built
  architecture-families / taxonomy total + pending families. NEVER say "all/complete/done" — new
  architectures keep surfacing and get added to the taxonomy. Continue selecting pending families
  (and per-task representatives), wave after wave.
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
