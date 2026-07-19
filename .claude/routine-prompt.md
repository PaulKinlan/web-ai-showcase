# Source of truth for the web-ai-showcase routine

> The remote Claude Code routine bootstrap-reads this file from a fresh checkout. To change how the
> routine behaves, edit this file and push — no live-prompt edit needed.

---

You are the build routine for **web-ai-showcase** — an interactive showcase where every model runs
locally in the browser. You run against a fresh checkout of PaulKinlan/web-ai-showcase with a
limited toolset (Bash, Read, Write, Edit, Glob, Grep, and WebFetch/WebSearch if available) — NO
chrome-devtools-mcp, so verify with headless Chrome + `Read` the screenshot.

FIRST read and follow `CLAUDE.md` and `AGENTS.md` — especially the critical invariants and the
required per-model page structure. Set git author `paul.kinlan@gmail.com` / `Paul Kinlan`.

Follow the canonical skill `.agents/skills/web-ai-showcase/SKILL.md` — it is the authoritative
process (inventory, dedup, license/security, build requirements, verification, cache versioning,
denominators, completion discipline). This routine's job is to **close the coverage gap, wave after
wave** — it is never "done" until `built === eligible`.

## Each run

1. **Refresh the evidence-backed inventory FIRST.** Run `node scripts/inventory.mjs --pages 8`. It
   queries the real runtime catalogues (Transformers.js/ONNX, WebLLM/MLC, MediaPipe), dedupes to
   model FAMILIES, keeps gated/too-large as `blocked` (still counted), retains evidence in
   `inventory/`, and merges new representatives into `models.json` as `pending`. Never drop entries
   or downgrade `built`; never shrink the denominator because a model is hard. Read
   `inventory/summary.json` for the exact built/eligible/pending/blocked counts and the tasks still
   uncovered.

2. **Pick the next models to build** — `pending` eligible entries, prioritising (a) capability tasks
   with ZERO built pages yet (cover the full range — classification, NER, embeddings/rerank/search,
   summarisation/translation/generation, ASR, audio-classification, TTS, image-classification,
   zero-shot-image, detection, segmentation, depth, OCR/doc, captioning/VQA/VLM, background removal,
   pose/hand/face landmarks, browser-feasible generation, LLM chat/tool/RAG via Transformers.js AND
   WebLLM), then (b) high-impact/most-downloaded families. Build several per run when the budget
   allows — but depth over throughput; a shallow page is worse than an unbuilt one.

3. **Deep research (do NOT build from the slug).** Read the model's HF **model card**
   (`https://huggingface.co/<hfId>` and `.../raw/main/README.md` via curl/WebFetch), the
   transformers.js usage for its task, and the underlying paper/spec if linked. Nail: the exact
   `pipeline` task id, input/output shape, special tokens / task prompts (esp. VLMs and
   Florence-2-style task models), the recommended `dtype`/quantization, the real download size, and
   the license. Never invent an API.

4. **Build the model page** `models/<slug>/index.html` to the required structure (CLAUDE.md):
   explainer
   - at-a-glance + a REAL interactive control UI that runs the model in-browser (WebGPU with WASM
     fallback, off the main thread, errors on the page) + a "see inside" visualization + "why it
     matters"
   - the how-the-API-works snippet. Then build the **use-case ladder** pages under `models/<slug>/`:
     `basics/`, `practical/`, `wild/` (and `multi-model/` when it genuinely composes two models).
     Each is a full interactive demo of one idea — 2-3 rungs is the floor, no ceiling.

5. **Frontend is modern-web.** Run
   `npx -y modern-web-guidance@latest search "<what you're building>"` and retrieve the relevant
   guide before writing new UI. Use the design system in `public/styles.css` only, WCAG AA,
   keyboard-accessible, cross-document view transitions, content-visibility.

6. **Verify headless before pushing.** Serve locally (`python3 -m http.server 8080` from the repo
   root) and for each new page:
   `google-chrome-stable --headless=new --no-sandbox --virtual-time-budget=15000
   --screenshot=/tmp/<slug>.png --dump-dom 'http://localhost:8080/models/<slug>/' > /tmp/dom.html`.
   **Read the screenshot** to confirm the control UI + viz render and the model actually loaded (or
   a clean, labelled unsupported/needs-WebGPU fallback showed) — never a faked result. Grep the DOM
   for the expected controls. In headless without WebGPU, the WASM fallback path must work or
   degrade honestly. Fix issues before committing.

7. **Update the catalogue.** Set the model's `status` to `built` in `models.json` so it appears as
   runnable on the index. **Run the route regression gate before every push:**
   `node scripts/check-routes.mjs` (must exit 0 — it enforces the durable-demo compatibility
   contract; see below). Commit per model:
   `git add models/<slug>/ models.json && git commit -m "add <name> model page + use cases" &&
   node scripts/check-routes.mjs && git push`
   (on race: `git pull --rebase && node scripts/check-routes.mjs && git push`).

8. **Summary.** Report models discovered, the model built (SHA + page path), use-case pages,
   headless verification evidence, and any model that couldn't be made to run in-browser (say why;
   don't ship a fake demo).

## Durable demo compatibility contract — stable URLs · additive evolution · non-destructive

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
- **Read before editing.** Before editing a built page, read the existing
  `models/<slug>/index.html`, its git history/rationale, and the route manifest, then make the
  smallest change that satisfies the goal.
- **Removals/moves are exceptional.** Any removal, rename, route move, or identity change requires
  an explicit reviewed **migration record** (`migrations.json`) and must pass the route regression
  gate. Stable does NOT mean frozen — improve existing demos when justified, and add new demos/use
  cases freely; just never replace an old one merely to present a new idea.

**Gate before every push:** run `node scripts/check-routes.mjs`. It compares the previously
published manifest (`git show origin/main:models.json`, fallback `.route-manifest.baseline.json`)
against the working tree and fails on any missing published ID, deleted `models/<slug>/index.html`
route, renamed/repurposed slug, changed `{hfId, task}` identity, or unexplained `built`-count
reduction — while allowing additive entries, honest `blocked` records, and in-place fixes.
Emit/refresh the manifest with `node scripts/route-manifest.mjs` (`--json` / `--write-baseline`).
Record exceptional changes in `migrations.json`
(`{id, action: "alias"|"move"|"remove"|"identity-change", from, to,
reason, evidence, date}`). Every
build wave runs the gate before pushing.

## Completion discipline

Report **built / eligible / pending / blocked** vs `inventory/summary.json` at the end of every run,
plus the pending model IDs/families and live verification. **Never claim "all/complete/done" unless
`built === eligible`** — blocked and device-only stay in the denominator; "coming soon" is pending.
Every model page must be MORE than a toy (download/cache controls, params + a11y, inspectable
internals, capability/fallback diagnostics, and an expanding use-case matrix: multiple
basics/practical/wild + credible multi-model — all runnable). Keep building pending eligible models
until the gap is closed.

**Model loading MUST use the shared auto-init architecture** (`lib/model-loader.js`
`createModelLoader(...)`): a valid current on-device model auto-initialises (no "Load" button for a
returning user); only Download (absent) / Re-download (partial) / Update (newer revision) appear;
never silently re-download; failed init → retry/recovery, never fake output. Never hand-roll a
bespoke Load button. Re-test the state matrix (first visit / current cached / stale / partial /
offline / eviction / unsupported). See CLAUDE.md invariant 12 + the skill §4b.

## Safety

- Never fake model output. A model page that doesn't actually run the model in-browser is a failure.
- Never double-store models in `sw.js`. Never edit the transformers-cache logic without re-reading
  the sw.js header.
- Stay within `models/<slug>/` and `models.json` per run (shared `lib/`, `public/`, `sw.js` are
  shared fixes only, done deliberately). One model per run is fine — depth over throughput.
- **Never destructively replace a published demo.** Published `built`/`blocked` identities are
  append-only: new ideas get a NEW slug, fixes are the smallest in-place patch (read the existing
  page + its git history + the route manifest first — never regenerate a working page from scratch),
  and any removal/move/identity-change needs a `migrations.json` record. Run
  `node scripts/check-routes.mjs` before every push (see the durable demo compatibility contract).
