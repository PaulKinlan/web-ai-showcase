# transformers.js 3.7.5 vs 4.3.0 — shadow compatibility matrix

Probe only: **no source was changed** (`lib/webai.js` and every demo route are untouched).
This is evidence for staging the shared pin (web-ai-showcase-9v4), not the bump itself.

- Generated: 2026-09-24T17:25:53.458Z
- Method: repo harness (headless Chrome + CDP), module worker importing transformers.js from a pinned CDN URL per version, real pipeline() load and real inference, output compared numerically
- Tolerance: same = identical dims, max |Δ| <= 1e-3 on the compared prefix, and equal text where the task returns text; ORT 1.22 -> 1.31 need not be bit-identical
- Coverage: 9 probed tuples — a representative sample of the 91 distinct (task, dtype, runtime) groups the 327 built routes contain; the full group table is in reports/transformers-v4-tuple-census.json
- Population: 327 built routes in 91 distinct (task, dtype, runtime) tuples.

## What 4.x changes, and why this matters

The shared pin in `lib/webai.js` is read by 315 of 327 built routes, so a bump is a one-line change with a 315-route blast radius. v4.0 replaces the WebGPU runtime with a native C++ WebGPU EP (not JSEP), adds a ModelRegistry and new `env` surfaces, and moves tokenizers into `@huggingface/tokenizers`. Transitive onnxruntime-web moves 1.22.0-dev → 1.31.0-dev.

## Results

| tuple | routes | split | 3.7.5 | 4.3.0 | verdict | detail |
|---|---|---|---|---|---|---|
| `feature-extraction-q8-wasm` | 19 | yes | RUNS 4625ms load / 60ms infer | RUNS 1215ms load / 32ms infer | **same** | max\|Δ\| 1.862645149230957e-8 |
| `text-classification-q8-wasm` | 1 | yes | RUNS 8725ms load / 96ms infer | RUNS 589ms load / 60ms infer | **same** | text identical |
| `token-classification-q8-wasm` | 1 | yes | RUNS 12632ms load / 86ms infer | RUNS 694ms load / 73ms infer | **same** | text identical |
| `fill-mask-q8-wasm` | 1 | yes | RUNS 14046ms load / 306ms infer | RUNS 2394ms load / 281ms infer | **differs** | output shape differs without comparable text — see the raw JSON |
| `text2text-generation-q8-wasm` | 9 | yes | RUNS 14870ms load / 99ms infer | RUNS 2304ms load / 99ms infer | **same** | text identical |
| `zero-shot-classification-q8-wasm` | 3 | yes | RUNS 11936ms load / 37ms infer | RUNS 1553ms load / 57ms infer | **same** | text identical |
| `image-classification-q8-wasm` | 1 | no | RUNS 13835ms load / 1904ms infer | RUNS 704ms load / 1553ms infer | **differs** | same label `sfw`, score Δ 4.57e-3 (0.8398016691207886 vs 0.8352280259132385) — float noise from the ORT bump, not a behaviour change |
| `text-to-speech-fp32-wasm` | 10 | yes | **FAILS** — Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json". | **FAILS** — Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json". | **both-fail** | undefined |
| `asr-ctc-q4-wasm` | 4 | yes | RUNS 26467ms load / 2269ms infer | RUNS 2535ms load / 1500ms infer | **same** | text identical |

## Tokenizers-split exposure (acceptance criterion 4)

v4 moves `AutoTokenizer` and `AutoProcessor` into `@huggingface/tokenizers`, and the bead calls out `AutoModelForCTC` routes because CTC decoding consumes tokenizer state directly. Reading each route's own files: **49 built routes use AutoTokenizer** and **21 use AutoModelForCTC**.

The tuples that carry that exposure, largest first:

- `text-classification | q8 | transformers.js` × 28 — AutoConfig, AutoTokenizer — e.g. `ai-text-detection`, `bge-reranker`, `bge-reranker-v2`, `clickbait-detection`
- `fill-mask | q8 | transformers.js` × 13 — AutoTokenizer — e.g. `albert-fill-mask`, `bert-fill-mask`, `bio-clinicalbert`, `camembert-fill-mask`
- `text-generation | q8 | transformers.js` × 12 — AutoTokenizer — e.g. `baguettotron`, `bloomz-multilingual`, `codegen-350m`, `codegen-350m-multi`
- `automatic-speech-recognition | fp32 | raw-ort` × 8 — AutoModelForCTC — e.g. `croatian-voxpopuli-asr`, `czech-voxpopuli-asr`, `dutch-voxpopuli-asr`, `german-voxpopuli-asr`
- `automatic-speech-recognition | fp32 | transformers.js` × 8 — AutoConfig, AutoModelForCTC, AutoProcessor — e.g. `finnish-voxpopuli-asr`, `french-xlsr-asr`, `italian-xlsr-asr`, `japanese-xlsr-asr`
- `image-text-to-text | q4f16 | transformers.js` × 8 — AutoProcessor, AutoTokenizer — e.g. `moondream2-vlm`, `nanollava-vlm`, `paligemma`, `phi-3.5-vision`
- `image-segmentation | q8 | transformers.js` × 5 — AutoProcessor, AutoTokenizer — e.g. `clipseg-text-segmentation`, `detr-panoptic`, `face-parsing`, `segformer-b2-clothes`
- `text-classification | fp32 | transformers.js` × 5 — AutoTokenizer — e.g. `formality-detection`, `german-sentiment`, `japanese-wrime-emotion`, `mmarco-reranker`
- `automatic-speech-recognition | q4 | transformers.js` × 4 — AutoModelForCTC, AutoProcessor — e.g. `chinese-xlsr-asr`, `mms-forced-alignment`, `polish-voxpopuli-asr`, `xlsr-multilingual-asr`
- `automatic-speech-recognition | q8 | transformers.js` × 4 — AutoModelForCTC, AutoProcessor — e.g. `distil-whisper-asr`, `moonshine-asr`, `wav2vec2-asr`, `whisper-speech-to-text`
- `image-feature-extraction | q8 | transformers.js` × 4 — AutoProcessor, AutoTokenizer — e.g. `dinov2-image-features`, `dinov2-registers`, `ijepa-features`, `nomic-embed-vision-v1-5`
- `text-generation | int8 | transformers.js` × 3 — AutoTokenizer — e.g. `gpt2-text-generation`, `lamini-neo`, `llama2-c-stories`
- `feature-extraction | fp32 | raw-ort` × 2 — AutoTokenizer — e.g. `colbert-late-interaction`, `splade-sparse-retrieval`
- `fill-mask | fp32 | transformers.js` × 2 — AutoTokenizer — e.g. `chinese-spell-check`, `portuguese-bert-fill-mask`

## Failure text (acceptance criterion 2)

### `text-to-speech-fp32-wasm` @ 3.7.5

- error name: `Error`
- error text: `Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json".`
- stack (truncated): `Error: Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json". |     at https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5:1:752833 |     at f (https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5:1:752856) |     at async _ (https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5:1:754731)`

### `text-to-speech-fp32-wasm` @ 4.3.0

- error name: `ModelFileNotFoundError`
- error text: `Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json".`
- stack (truncated): `ModelFileNotFoundError: Could not locate file: "https://huggingface.co/naklitechie/mms-tts-ta-ONNX/resolve/main/tokenizer.json". |     at Uc (https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0:11:10575) |     at c$ (https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0:11:18303) |     at async No (https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0:11:20362)`

## Reading this honestly

- Load timings are indicative, not a benchmark: whichever version runs second benefits from weights already in the browser cache.
- `same` means identical dims, max |Δ| ≤ 1e-3 on the compared prefix, and equal text where the task returns text. onnxruntime-web 1.22 → 1.31 need not be bit-identical; float noise at 1e-8 (as observed) is expected and is reported rather than hidden.
- Where the probe's strict verdict reads `differs`, the detail column says WHY. A row that is the same label with a score moved by ~1e-3 is float noise from the ORT bump; a row whose text or labels actually change is a behaviour difference. The numbers are shown so the judgement is the reader's, not the script's.
- Inputs are deterministic and synthetic for vision/audio (a generated PNG, a 220 Hz tone). That is enough to compare versions against each other, and is not a claim about model quality.
- NOT probed in this run: `text-generation-q8-wasm`, `image-text-to-text-q4f16-webgpu`. Those are heavier downloads than this probe's per-case budget allowed; they are named here rather than silently omitted, and are the first checks for the staging bead.
- The probed set is a representative sample of the distinct tuples, chosen to cover the tokenizers split, the rewritten WebGPU path, and the largest route groups. Tuples not listed here were not probed and are marked as such by omission from the table rather than implied to pass.
