# web-ai-showcase

**Every AI model you can run locally in a browser — with a hands-on control panel, a look inside
what it's doing, and real use cases from the practical to the wild.**

No servers, no API keys, no uploads. Each model loads and runs on your device via
[🤗 Transformers.js](https://github.com/huggingface/transformers.js) (WebGPU, with a WASM fallback),
cached offline by a service worker.

Live: **https://paulkinlan.github.io/web-ai-showcase/** · Cross-origin-isolated deployment:
**https://web-ai-showcase.paulkinlan-ea.deno.net/web-ai-showcase/** · A sibling of
[chrome-platform-showcase](https://github.com/PaulKinlan/chrome-platform-showcase) and
[image-embedding-lab](https://github.com/PaulKinlan/image-embedding-lab).

## Why

People don't know what models actually _do_ or unlock. This grounds Web AI in demos you can run and
see inside — and for each model makes the case for why a developer or business should be excited.
Every model gets:

- **An explainer + control page** — what it is (from its model card), how to use it, and a UI to run
  it live with the controls it deserves.
- **"See inside"** — attention/probabilities/embeddings/overlays + real latency & backend readout.
- **A use-case ladder** — Basics → Practical → Wild → (stretch) Multi-model.

## Develop

Static client-side site. The Deno command runs the same COOP/COEP edge proxy used in production,
backed by the published GitHub Pages files:

```bash
deno task serve   # then open http://localhost:8000/web-ai-showcase/
```

The small Deno edge proxy adds `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` to the GitHub Pages responses, enabling
`crossOriginIsolated`, `SharedArrayBuffer`, and the zero-copy audio ring-buffer path without moving
inference or model data onto a server. GitHub Pages remains the compatibility deployment and uses
transferable `postMessage` because it cannot set those headers. Production browser evidence is
retained in [`reports/deno-isolation-validation.json`](reports/deno-isolation-validation.json): SAB
was shared with a Worker through `Atomics`, the audio pipeline selected `sab-isolated`, and
`performance.measureUserAgentSpecificMemory()` returned a real breakdown.

`models.json` is the catalogue; `models/<slug>/` holds each model's pages; `lib/webai.js` is the
shared model-loading helper; `sw.js` caches the shell + model blobs. See **[CLAUDE.md](CLAUDE.md)**
for the architecture, the required per-model page structure, and the invariants. The catalogue is
kept fresh by a Claude Code routine (`.claude/routine-prompt.md`).

## Model diagnostics

Both shared loaders expose memory snapshots using `measureUserAgentSpecificMemory()` when the host
is cross-origin isolated and the browser supports it. The estimate covers the page and its dedicated
workers, **not model-only RAM, GPU memory, process memory, or cached download size**. Measurement is
**manual-only**: the visitor clicks Measure memory, which races a 90-second bound and keeps the result
when it lands. There is no JS-heap fallback masquerading as total model memory.

The reason sampling is not automatic: the browser answers this API only after a garbage-collection
pass, measured at **~19 seconds with nothing loaded and ~60 seconds with a model resident**, in both
Chrome 150 and Chromium 152 (see `web-ai-showcase-cgr`). No non-blocking page-phase budget can cover
that, so an automatic before/after pair could only ever report "took too long". Explicit requests are
different — the label is the visitor's, so a slow answer is still the answer. A stalled or slow
request never blocks loading or release.

`lib/model-run-status.mjs` displays elapsed time and average generated tokens/second (including
prefill), updates at most four times per second, and announces the final rate accessibly. Call
`token(count, ms)` with **generated token IDs**, not the number of decoded text callbacks. All four
`llama2-c-stories` routes demonstrate this with Transformers.js's `token_callback_function`. Other
families' token counters have not been audited by this change.

```bash
deno task test:diagnostics                      # download-free browser fault/lifecycle checks
node scripts/validate-model-diagnostics.mjs      # also runs the 15 MB model on 4 routes × 2 viewports
```

The real-model check records native memory estimates, DevTools performance metrics, long tasks,
console/network results and screenshots in the printed evidence directory. It uses test-only eager
GC measurement to avoid Chrome's arbitrary GC wait; it does not simulate the reported memory values.
Desktop Chrome mobile emulation is not a physical-phone memory/performance result.

Made by [Paul Kinlan](https://paul.kinlan.me/).
