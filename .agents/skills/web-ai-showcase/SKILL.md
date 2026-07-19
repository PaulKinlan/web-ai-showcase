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
  `visual-question-answering` pipeline / no ONNX), **kosmos-2** (no `kosmos` class in
  transformers.js; no browser ONNX), **mask2former** (no `Mask2Former...` class in transformers.js
  3.7.5/4.2 — only the image processor; no browser ONNX), **internvl** (no `internvl` class →
  `Unsupported model type: internvl`), **keyphrase-extraction** (no token-classification keyphrase
  ONNX — only seq2seq generators), **electra** (ONNX is encoder-only; no RTD discriminator head),
  **blip** / **bark** (gated / no usable ONNX). Never mislabel a substitute as the blocked family.
- **Version-pin escape hatch:** a model needing a transformers.js class newer than the shared 3.7.5
  (e.g. SAM2 needs 4.2.0) may pin the newer version LOCALLY in its own `worker.js` only — never bump
  shared `lib/webai.js`. model-cache is version-agnostic so auto-init still works. Precedent:
  `models/sam2-segmentation/worker.js`.
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

## 8. Durable demo compatibility contract — stable URLs · additive evolution · non-destructive

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

**Mechanics.** `node scripts/route-manifest.mjs` emits the published manifest (one
`{id, route, identity: {hfId, task}, status, aliases[]}` per `built`/`blocked` model; `pending`
excluded); `--json` prints it, `--write-baseline` refreshes `.route-manifest.baseline.json`.
`node scripts/check-routes.mjs` derives the baseline from `git show origin/main:models.json`
(offline fallback: the committed baseline) and enforces the failures above. Record every exceptional
change in `migrations.json`
(`{id, action: "alias"|"move"|"remove"|"identity-change", from, to, reason,
evidence, date}`). **In
build/validate modes, read the existing page + its git history + the route manifest before editing a
built demo, prefer the smallest patch, and run the gate before every push.**

## 9. Critique → immutable conformance → goal lifecycle (per built demo)

A WEB-AI-domain analog of chrome-platform-showcase's lifecycle (NOT a copy of its Chrome
assertions). Order per demo: **coverage/build → critique → immutable conformance → validation →
goal-setting.** All ADDITIVE — it never rewrites a page or churns a URL (invariant 13 still holds).
Run critique + conformance before every push.

- **Critique** — `models/<slug>/_questions.json` (versioned, mutable; schema
  `schemas/critique.schema.json`). Rubric dimensions: real-inference, io-shape (expected output
  shape + semantic sanity), runtime-config (backend/model-id/quant), cache-init
  (auto-init-on-cached), states (progress/error/retry/offline), controls (every visible control),
  no-fake-output, accessibility, responsive (mobile+desktop), performance (INP). Each rubric entry
  keeps **REAL retained evidence** (screenshot path / console excerpt / latency / output sample).
  Also `guidanceConsulted[]` — empty on a frontend critique = INCOMPLETE (the gate fails it) —
  `openQuestions`, and `followUpGoals`.
- **Immutable conformance** — `models/<slug>/conformance.json` (schema
  `schemas/conformance.schema.json`). Assertions are DERIVED from real `models.json` metadata + the
  per-task templates in `scripts/conformance-lib.mjs` (task → expected output shape/semantic check;
  runtime/backend/model-id/quant; auto-init-on-cached; no-fake-output; a11y; mobile+desktop
  deviceClass; guidance build-process). **Immutable = never delete/weaken/regenerate an assertion to
  go green — fix the DEMO.** `suiteHash` = sha256 of the normalized assertions; the gate recomputes
  it and diffs against `origin/main`. Adding assertions is always allowed. Generate suites for NEW
  built models with `node scripts/gen-conformance.mjs` (it never overwrites an existing suite; use
  `--rehash
  <slug>` after hand-ADDING an assertion). Record any exceptional removal in
  `conformance-migrations.json`.
- **Validation** — `node scripts/conformance.mjs --slug <slug>` (or `--all`, `--limit N`).
  Deterministic, headless-Chrome-backed (fresh profile ⇒ cache-absent ⇒ **download-free**; it NEVER
  auto-downloads an absent large model to force a pass). Emits **tested / total · pass / fail /
  blocked** (+ manual = manual-evidenced screenshot awaiting an agent verdict). `blocked` = the
  device/feature is GENUINELY unavailable (a WebGPU-only model on a no-GPU runner shows its honest
  needs-WebGPU state) — explicit, never a pass. Rollup: `reports/conformance/index.html` +
  `results.json`.
- **Goal-setting** — `node scripts/goals.mjs` collects critique `followUpGoals` into `goals.json`
  (schema `schemas/goals.schema.json`): an ADDITIVE backlog to pick the next NEW demo or targeted
  in-place fix — never to replace a stable page.
- **Gate** — `node scripts/check-conformance.mjs` runs beside `check-routes.mjs` before every push +
  in CI. FAILS on: a built demo with no suite, a duplicate/orphan suite id, a malformed artifact, a
  weakened/removed assertion without a migration record, or a touched demo left untested/broken on a
  supported device class. REPORTS denominators (suites/built, critique/built, mobile+desktop
  tested/total, pass/fail/blocked). Author genuine suites by DERIVING; never fake; never claim
  "complete/all".

## 10. Mobile + desktop parity — every demo usable on both, or honestly unsupported with recorded evidence

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

## 11. modern-web-guidance is mandatory for all frontend work

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
