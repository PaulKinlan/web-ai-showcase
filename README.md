# web-ai-showcase

**Every AI model you can run locally in a browser — with a hands-on control panel, a look inside what
it's doing, and real use cases from the practical to the wild.**

No servers, no API keys, no uploads. Each model loads and runs on your device via
[🤗 Transformers.js](https://github.com/huggingface/transformers.js) (WebGPU, with a WASM fallback),
cached offline by a service worker.

Live: **https://paulkinlan.github.io/web-ai-showcase/** · A sibling of
[chrome-platform-showcase](https://github.com/PaulKinlan/chrome-platform-showcase) and
[image-embedding-lab](https://github.com/PaulKinlan/image-embedding-lab).

## Why

People don't know what models actually *do* or unlock. This grounds Web AI in demos you can run and
see inside — and for each model makes the case for why a developer or business should be excited.
Every model gets:

- **An explainer + control page** — what it is (from its model card), how to use it, and a UI to run
  it live with the controls it deserves.
- **"See inside"** — attention/probabilities/embeddings/overlays + real latency & backend readout.
- **A use-case ladder** — Basics → Practical → Wild → (stretch) Multi-model.

## Develop

Static site — serve the folder and open it:

```bash
python3 -m http.server 8080   # then open http://localhost:8080/
```

`models.json` is the catalogue; `models/<slug>/` holds each model's pages; `lib/webai.js` is the shared
model-loading helper; `sw.js` caches the shell + model blobs. See **[CLAUDE.md](CLAUDE.md)** for the
architecture, the required per-model page structure, and the invariants. The catalogue is kept fresh by
a Claude Code routine (`.claude/routine-prompt.md`).

Made by [Paul Kinlan](https://paul.kinlan.me/).
