// Canonical AutoNLP gibberish detector worker. All tokenization, inference, softmax, and batching
// remain off the main thread. The immutable revision and fp32 artifact are deliberately explicit:
// madhurjindal/autonlp-Gibberish-Detector-492513457@76672dd…/onnx/model.onnx (267,961,863 bytes).
import { serveWorker } from "/web-ai-showcase/lib/worker-protocol.js";
import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "madhurjindal/autonlp-Gibberish-Detector-492513457";
const REVISION = "76672dd7d3575f68ab980705bcec975cc62de71c";
const LABELS = ["clean", "mild gibberish", "noise", "word salad"];
let tokenizer = null;
let model = null;

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

async function infer(text, maxLength, signal) {
  if (!model || !tokenizer) throw new Error("Model not loaded");
  if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  const started = performance.now();
  const inputs = await tokenizer(text, { truncation: true, max_length: maxLength });
  const output = await model(inputs);
  if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  const logits = Array.from(output.logits.data, Number);
  const probabilities = softmax(logits);
  const scores = LABELS.map((label, index) => ({
    label,
    logit: logits[index],
    score: probabilities[index],
  }))
    .sort((a, b) => b.score - a.score);
  const ids = Array.from(inputs.input_ids.data, Number);
  const tokens = ids.map((id) => tokenizer.decode([id], { skip_special_tokens: false }));
  return {
    text,
    scores,
    top: scores[0],
    tokenIds: ids,
    tokens,
    tensorShape: Array.from(output.logits.dims),
    truncated: ids.length >= maxLength,
    maxLength,
    ms: Math.round(performance.now() - started),
    backend: "wasm",
    dtype: "fp32",
    sourceModel: MODEL_ID,
    revision: REVISION,
  };
}

serveWorker({
  async init() {},
  methods: {
    async load(_payload, { onProgress }) {
      const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(
        TRANSFORMERS_URL
      );
      env.allowLocalModels = false;
      tokenizer = await AutoTokenizer.from_pretrained(
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        {
          revision: REVISION,
          progress_callback: (event) => onProgress?.(event),
        },
      );
      model = await AutoModelForSequenceClassification.from_pretrained(
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        {
          revision: REVISION,
          dtype: "fp32",
          device: "wasm",
          progress_callback: (event) => onProgress?.(event),
        },
      );
      return {
        result: { backend: "wasm", dtype: "fp32", sourceModel: MODEL_ID, revision: REVISION },
      };
    },
    async classify({ text, maxLength = 64 }, { signal }) {
      return {
        result: await infer(
          String(text),
          Math.max(8, Math.min(64, Number(maxLength) || 64)),
          signal,
        ),
      };
    },
    async batch({ texts, maxLength = 64 }, { signal, onProgress }) {
      const rows = [];
      for (let index = 0; index < texts.length; index++) {
        rows.push(
          await infer(
            String(texts[index]),
            Math.max(8, Math.min(64, Number(maxLength) || 64)),
            signal,
          ),
        );
        onProgress?.({ status: "inference", completed: index + 1, total: texts.length });
      }
      return { result: { rows } };
    },
  },
  onDispose() {
    tokenizer = null;
    model = null;
  },
});
