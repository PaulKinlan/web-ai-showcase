# web-ai-showcase — operator manual

> A hand-crafted, interactive showcase for **every AI/ML model you can run locally in a browser**.
> Read this before editing anything. The whole project is maintained largely by a Claude Code routine
> on a cron; the rules below exist so it stays a *pinnacle-of-modern-web* showcase, not a model dump.

## What this project is

The Chrome Platform Showcase, but for **Web AI**. For every browser-runnable model (transformers.js /
ONNX Runtime Web / WebLLM / WebGPU) we build:

1. **An explainer + control page** — what the model is (digested from its HF model card), how it's
   set up (task, params, quantization, backend), how you interact with it, and a **beautiful,
   interactive control UI that actually runs the model locally in the visitor's browser** — with the
   modality-specific controls a model deserves (image + prompt + special tokens for VLMs, sampling
   controls + token streaming for LLMs, live camera + overlays for vision).
2. **A "see inside the model" surface** — visualize what's happening: attention/saliency maps, token
   probabilities, embedding space, detection boxes, segmentation masks, plus a live backend / latency
   / memory readout. Make the inner workings legible without requiring the visitor to understand the
   maths.
3. **A use-case ladder** (the heart of it — one demo per rung, no ceiling):
   - **Basics** — the fundamental capability, plainly. What this model *does*.
   - **Practical** — concrete use cases a business or developer would actually ship.
   - **Wild** — out-there, novel, fun, engaging: things you could never do before this ran in a browser.
   - **Multi-model** (stretch) — connect this model to another (e.g. Whisper → LLM → TTS).
4. **A test harness** — a way to probe the model's behaviour and edges without reading a paper, and to
   debug what it's doing.

The thesis (Paul): people don't know what models actually *do* or unlock. Ground Web AI in real,
runnable, legible demos — and for each model make the case for *why a business or developer should be
excited*: what it enables, what it unlocks, what it replaces.

- Hosted on **GitHub Pages** (paulkinlan.github.io/web-ai-showcase). Static — no server. Inference,
  model discovery, and caching are all client-side.
- Sibling projects: [chrome-platform-showcase](https://github.com/PaulKinlan/chrome-platform-showcase)
  (web-platform demos), [gendn](https://github.com/PaulKinlan/gendn) (MDN-style docs),
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
Stream tokens/progress incrementally. For any main-thread loop, break it up / `await scheduler.yield()`.
See modern-web-guidance: `break-up-long-tasks`, `schedule-tasks-by-priority`.

### 4. Do not double-store models in the service worker
transformers.js owns `transformers-cache`; the SW must serve model files via `caches.match()` across
all caches, and only store the app shell + the library JS itself. Double-storing ~GB blobs evicts the
origin. (See the `sw.js` header — this bit image-embedding-lab.)

### 5. Design system only — match the AI-focus look, WCAG AA
Use the CSS variables + classes in `public/styles.css` (warm off-white paper, indigo accent, Georgia
display, humanist-sans body, light+dark). No raw hex for text/bg. Every pair hits WCAG AA. This site
should feel like the rest of Paul's AI demos.

### 6. Cross-document view transitions + content-visibility
The catalogue↔model navigation uses `@view-transition { navigation: auto }` (already in styles.css).
The catalogue grid uses `content-visibility:auto` + `contain-intrinsic-size`. See modern-web-guidance:
`cross-document-transitions`, `defer-rendering-heavy-content`.

### 7. Ground every model page in its real model card + how the API actually works
Read the HF model card and the transformers.js/WebLLM usage before building. Get the exact task,
pipeline id, input/output shape, special tokens, and recommended dtype/quantization. Never invent an
API surface. Explain how to call it in a way that doesn't require understanding the model internals.

### 8. Runnable-only catalogue
Only list models that genuinely run in a browser today: a transformers.js-compatible ONNX build, an
ONNX Runtime Web model, or a WebLLM/MLC prebuilt. Mark each with its backend + approximate download
size so a visitor knows what they're committing to. If a model needs a flag (e.g. WebGPU FP16), say so.

## Modern-web references (retrieve before building the relevant surface)
`npx -y modern-web-guidance@latest retrieve <id>` — `cross-document-transitions`,
`defer-rendering-heavy-content`, `break-up-long-tasks`, `schedule-tasks-by-priority`,
`identify-inp-causes`. Frontend is HTML/CSS/clientside-JS: run modern-web-guidance FIRST for any new UI.

## The per-model page — required structure

`models/<slug>/index.html`:
1. `<link rel="stylesheet" href="/web-ai-showcase/public/styles.css">`, crumbs back to `/`.
2. Eyebrow (task · modality · backend · size), H1 (friendly model name), lede (one-sentence what/why).
3. **At a glance** — HF id (linked), task/pipeline, params, quantization/dtype, backend (WebGPU/WASM),
   download size, license, Baseline-of-the-web-APIs-it-needs.
4. **Run it** — the interactive control UI. Load button with real progress; modality-specific inputs;
   real output; latency + backend readout. Off-main-thread.
5. **See inside** — at least one real visualization of the model's working (probabilities, attention,
   embeddings, overlays), with a short "what am I looking at" note.
6. **Why it matters** — what this unlocks for a business/developer; 2-4 concrete "you could build…".
7. **Use cases** — links to the ladder pages (basics/practical/wild[/multi-model]).
8. **How the API works** — the minimal `pipeline()`/WebLLM snippet, explained.
9. **References** — HF model card, the spec/paper, transformers.js docs. Byline footer.

Each use-case page is a full interactive demo of one idea, same shell, crumbs back to the model page.

## The routine

A Claude Code cron (like the showcase's) keeps the catalogue fresh: discover new browser-runnable
models via the HF API, build the explainer + first use-case pages, one commit per model, headless-
verified. Source of truth: `.claude/routine-prompt.md`; runs on Opus, bootstrap-reads this repo.

## Testing checklist (before pushing)
- `deno fmt --check` (HTML/CSS excluded); `deno check scripts/*.mjs lib/*.js` where typed.
- Serve locally (`python3 -m http.server` or `deno run --allow-net --allow-read jsr:@std/http/file-server`)
  and open each new page.
- Verify in a real browser: the model actually loads and runs (WebGPU path AND the WASM fallback path),
  the "see inside" viz renders, errors surface on the page, light+dark both pass contrast, keyboard
  works, and the SW doesn't double-store (check Application → Cache Storage).
