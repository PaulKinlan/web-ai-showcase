import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODELS = {
  fashion: {
    id: "patrickjohncyh/fashion-clip",
    revision: "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
    dtype: "fp32",
  },
  general: {
    id: "Xenova/clip-vit-base-patch16",
    revision: "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
    dtype: "q8",
  },
};
let pipe;
let RawImage;
let activeKind;

const post = (message) => self.postMessage(message);

async function ensureLoaded(kind) {
  if (pipe && activeKind === kind) return;
  const spec = MODELS[kind];
  if (!spec) throw new Error(`Unknown model kind: ${kind}`);
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  // Keep each stage literal at its loader call site: the portfolio gate can statically prove that both
  // advertised models really load, while this worker still shares the inference/tensor protocol.
  const loaded = kind === "general"
    ? await loadPipeline({
      task: "zero-shot-image-classification",
      model: "Xenova/clip-vit-base-patch16",
      revision: "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
      backend: "wasm",
      dtype: "q8",
      onProgress: (progress) => post({ type: "progress", progress }),
    })
    : await loadPipeline({
      task: "zero-shot-image-classification",
      model: "patrickjohncyh/fashion-clip",
      revision: "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
      backend: "wasm",
      dtype: "fp32",
      onProgress: (progress) => post({ type: "progress", progress }),
    });
  pipe = loaded.pipe;
  activeKind = kind;
  post({ type: "ready", kind, device: loaded.device });
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function normsAndCosines(image, texts, dimensions, count) {
  let imageNorm = 0;
  for (const value of image) imageNorm += value * value;
  imageNorm = Math.sqrt(imageNorm);
  const similarities = [];
  for (let row = 0; row < count; row++) {
    let dot = 0;
    let textNorm = 0;
    for (let column = 0; column < dimensions; column++) {
      const value = texts[row * dimensions + column];
      dot += image[column] * value;
      textNorm += value * value;
    }
    similarities.push(dot / (imageNorm * Math.sqrt(textNorm) || 1));
  }
  return similarities;
}

async function classify({ id, kind, image, labels }) {
  await ensureLoaded(kind);
  const started = performance.now();
  const raw = await RawImage.read(image);
  const imageInputs = await pipe.processor(raw);
  const textInputs = pipe.tokenizer(labels, { padding: true, truncation: true });
  const output = await pipe.model({ ...textInputs, ...imageInputs });
  const logits = Array.from(output.logits_per_image.data);
  const probabilities = softmax(logits);
  const imageDimensions = output.image_embeds?.dims ?? null;
  const textDimensions = output.text_embeds?.dims ?? null;
  let cosines = logits;
  if (output.image_embeds && output.text_embeds && imageDimensions) {
    const dimension = imageDimensions.at(-1);
    cosines = normsAndCosines(
      output.image_embeds.data,
      output.text_embeds.data,
      dimension,
      labels.length,
    );
  }
  post({
    type: "result",
    id,
    kind,
    labels,
    probabilities,
    logits,
    cosines,
    imageDimensions,
    textDimensions,
    elapsedMs: Math.round(performance.now() - started),
    device: "wasm",
  });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await ensureLoaded(data.kind);
    if (data.type === "classify") await classify(data);
  } catch (error) {
    post({ type: "error", id: data.id, message: String(error?.message ?? error) });
  }
});
