# Raw-ORT dual-runtime double-WASM check (web-ai-showcase-62m.3)

**Question.** Three routes import transformers.js *and* a raw `onnxruntime-web` build in one page. Does a
visitor who uses those routes actually pay for two ORT runtimes (two JS bundles *and* two WASM binaries),
or does one path stay lazy?

**Answer. Both runtimes really load — but only when the multi-model flow is actually used.** Every one of
the three multi-model routes fetched two distinct ORT JS runtimes and two distinct ORT WASM binaries on
its desktop pass, each named below. The single-runtime overview routes of the same families fetched only
the raw runtime, so the doubling is specific to the multi-model composition, not incidental.

## Method (reproducible)

`node scripts/measure-raw-ort-dual-runtime.mjs [--fresh] [--json]`

One headless Chrome; for each route it opens the page at desktop and mobile, drives **every**
`.model-loader` to `ready` (session creation is what instantiates a runtime), and attributes every
network request — by URL, encoded bytes, and cache state — to transformers.js, the raw CDN ORT build, or
model weights. Worker and service-worker sessions are auto-attached and their `Network` domains enabled,
because both engines fetch their runtimes from inside a Web Worker. Each family's overview route is
measured as a single-runtime control. `--fresh` wipes the profile for real transferred bytes; without it
the profile persists and repeat runs are cheap.

## What actually loaded

| route (multi-model, desktop) | transformers.js runtime | raw ORT runtime |
| --- | --- | --- |
| bert-base-turkish-cased-ner | `@huggingface/transformers@3.7.5/dist/ort-wasm-simd-threaded.jsep.mjs` + `.jsep.wasm` | `onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs` + `ort-wasm-simd-threaded.jsep.mjs` + `.jsep.wasm` |
| model2vec-static-embeddings | `@huggingface/transformers@3.7.5/dist/ort-wasm-simd-threaded.jsep.mjs` + `.jsep.wasm` | `onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs` + `ort-wasm-simd-threaded.jsep.mjs` + `.jsep.wasm` |
| yolo-world | `@huggingface/transformers@3.7.5/dist/ort-wasm-simd-threaded.jsep.mjs` + `.jsep.wasm` | `onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs` + `ort-wasm-simd-threaded.mjs` + `.wasm` |

Controls: `models/bert-base-turkish-cased-ner/` loaded the raw runtime only (1.22.0 jsep pair);
`models/model2vec-static-embeddings/` and `models/yolo-world/` loaded the raw runtime only. No
transformers.js ORT on any control row.

Raw rows and per-request URLs: `reports/dual-runtime-62m3.json`.

## Payload cost (jsdelivr content-length, exact)

| runtime | JS | ORT glue | WASM | total |
| --- | --- | --- | --- | --- |
| transformers.js 3.7.5 | 852.8 KB | 43.4 KB | 21.09 MB (jsep) | **21.96 MB** |
| raw ORT 1.22.0 `ort.webgpu` | 349.2 KB | 43.6 KB | 21.36 MB (jsep) | **21.74 MB** |
| raw ORT 1.20.1 `ort.webgpu` | 325.7 KB | 45.7 KB | 21.16 MB (jsep) | **21.52 MB** |
| raw ORT 1.21.0 `ort.wasm` | 45.4 KB | 26.0 KB | 12.37 MB (plain) | **12.44 MB** |

So a dual-runtime multi-model route pays roughly **43.5 MB** of runtime before model weights on the
WebGPU-bundle pins (bert, model2vec), or **34.4 MB** once the raw side uses the wasm-only bundle
(yolo-world today, bert after 62m.2). For comparison, the same pages transfer 60 MB (model2vec),
254 MB (yolo-world) and 703 MB (bert) of model weights — the second runtime is real but small next to
the weights, and it is only fetched by visitors who load both stages.

## Why the doubling is not a one-line fix

* The two stages are different model families and the raw sessions exist because those ONNX layouts are
  not transformers.js pipeline-compatible (`models/bert-base-turkish-cased-ner/worker.js` ships a root
  `model.onnx`; `models/manga-ocr/worker.js` documents the same for its decoder). Running them through
  one library would mean re-implementing the pipeline or the raw session against the other runtime.
* The engines live in **separate workers** (transformers.js owns one; each raw route owns another), so
  even where the versions matched they could not share an ORT instance or its WASM heap.
* The loaders are lazy per stage: on a first visit each engine shows its own Download control and the
  page fetches neither runtime until that stage is loaded. The double cost is therefore paid only by the
  visitor who uses the multi-model flow — which by definition needs both.

## Actionable finding: the jsep WASM is the avoidable 9 MB

The raw side of bert (1.22.0) and model2vec (1.20.1) imports an `ort.webgpu` / `ort.all` bundle while
requesting `executionProviders:["wasm"]`. That pulls the **jsep** WASM binary (21.2-21.4 MB) instead of
the plain one (12.4 MB) — about 9 MB of avoidable runtime per route. `web-ai-showcase-62m.2` does this
for bert and `62m.1` for manga-ocr; **model2vec-static-embeddings is the remaining raw route with the
oversized bundle**, and it also still sits on 1.20.1. Proposed follow-up (own bead, since it touches
`models/<slug>/**` and therefore needs a real-inference acceptance run): move model2vec to
`onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs`, verify, and record the acceptance run.

## Caveats

* This artifact comes from a **warm-cache** run: runtime files already in the profile's HTTP cache are
  often not re-requested, so the runtime byte columns read 0 in `reports/dual-runtime-62m3.json`. The
  runtime **file sets** are the evidence; the byte costs above are the content-lengths of exactly those
  files. A `--fresh` run reproduces the cold transfer.
* The artifact's original classifier grouped Hugging Face's Xet CAS hosts (`us.aws.cdn.hf.co`,
  `cas-bridge.xethub-eu.hf.co`) under `other`; the JSON re-buckets those URLs into `model-weights` and
  the committed script carries the fixed classifier.
* Mobile passes reuse the desktop engines and HTTP cache and therefore show thin runtime rows; viewport
  does not change which runtimes the page code imports.
* `model2vec`'s tokenizer-only transformers.js import (overview worker) does not by itself create a
  session; the transformers.js ORT WASM seen on that route's multi-model row belongs to the jina reranker
  stage, which is a real model.
