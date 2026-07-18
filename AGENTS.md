# Agent instructions — web-ai-showcase

`CLAUDE.md` is the canonical operator manual; read it before non-trivial edits. This file is the short
version for tools that look for `AGENTS.md`.

## Hard rules

- **Models must really run in-browser.** Real `pipeline()` / WebLLM calls on the visitor's device.
  Never present a canned result as live output. Degrade honestly (labelled needs-WebGPU / too-large /
  unsupported) — never fake success.
- **Errors on the page, not only the console.** Surface every load/inference/WebGPU failure in the UI.
- **Inference off the main thread**, stream progress/tokens, keep INP low (workers; break up long tasks).
- **Never double-store models in `sw.js`.** transformers.js owns `transformers-cache`; serve model
  files via `caches.match()` across all caches; only cache the app shell + library JS ourselves.
- **Design system only** (`public/styles.css`): warm off-white + indigo, Georgia display, humanist
  sans body, light+dark, WCAG AA. Cross-document view transitions; content-visibility on the catalogue.
- **Ground each page in the real model card + API.** Exact pipeline task, I/O shape, special tokens,
  dtype, size, license. Never invent an API surface.
- **Frontend is HTML/CSS/clientside-JS** → run `modern-web-guidance` FIRST for any new UI.
- **Runnable-only catalogue.** Only list models that run in a browser today; tag backend + download size.
- Verify with headless Chrome + `Read` the screenshot (no chrome-devtools-mcp in the routine).

## Per-model page structure
See CLAUDE.md — explainer + at-a-glance + Run-it control UI + See-inside viz + Why-it-matters + use-case
ladder (basics/practical/wild[/multi-model]) + how-the-API-works + references.
