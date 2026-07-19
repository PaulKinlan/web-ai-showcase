// SigLIP 2 zero-shot worker — ALL inference off the main thread so the control UI stays responsive.
// One forward pass returns everything the "Run it" bars and the "See inside" surface need: the raw
// per-label logits, the SIGMOID probability of each label (independent — SigLIP's real read-out), a
// softmax over the same logits (only for contrast), the cosine similarities from the normalized
// embeddings, and an estimate of the learned temperature + bias.
//
// Model: onnx-community/siglip2-base-patch16-224-ONNX (task: zero-shot-image-classification), WASM, q8.
// This is the SigLIP 2 checkpoint (Google, 2025) — a newer training recipe than the original SigLIP,
// notably a MULTILINGUAL Gemma tokenizer, so labels in many languages land in the same space as the
// image. The onnx-community export declares model_type "siglip", so it loads through the SAME
// SiglipModel / zero-shot pipeline that SigLIP v1 uses in transformers.js 3.7.5 — no version pin, no
// invented API. Like the SigLIP v1 demo it keeps the per-pair SIGMOID read-out: each image↔label pair
// is scored on its own, probabilities do NOT sum to 1, and "none of the above" is a valid answer.
//
// We reuse the pipeline's own model/processor/tokenizer to reach the low-level tensors. SigLIP 2's
// Gemma tokenizer reports model_max_length = 1e30, so we MUST pad explicitly to max_length 64 (the
// length SigLIP trained on) — the pipeline's default padding would otherwise try to allocate a 1e30
// array and throw "Invalid array length".

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let RawImage = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const loaded = await loadPipeline({
    task: "zero-shot-image-classification",
    model: "onnx-community/siglip2-base-patch16-224-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Cosine between the single image embedding and each per-label text embedding. SigLIP normalizes both
// embeddings internally before it forms the logits, so the returned embeds are already unit length —
// but we normalize again here defensively so the "See inside" numbers are honest.
function cosines(imageEmb, textEmb, dim, n) {
  const imgN = l2norm(imageEmb) || 1;
  const out = [];
  for (let i = 0; i < n; i++) {
    let dot = 0;
    let tN = 0;
    for (let d = 0; d < dim; d++) {
      const t = textEmb[i * dim + d];
      dot += imageEmb[d] * t;
      tN += t * t;
    }
    out.push(dot / (imgN * (Math.sqrt(tN) || 1)));
  }
  return out;
}

// Recover SigLIP's learned temperature (scale) + bias from logit = scale·cos + bias via a plain
// least-squares fit over the labels. Needs ≥2 distinct cosines; otherwise returns null.
function fitScaleBias(cos, logits) {
  const n = cos.length;
  if (n < 2) return null;
  const mx = cos.reduce((a, b) => a + b, 0) / n;
  const my = logits.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (cos[i] - mx) * (cos[i] - mx);
    sxy += (cos[i] - mx) * (logits[i] - my);
  }
  if (sxx < 1e-9) return null;
  const scale = sxy / sxx;
  const bias = my - scale * mx;
  return { scale, bias };
}

// SigLIP is trained on caption-like text. The zero-shot pipeline wraps each bare label in a hypothesis
// template before tokenizing; we replicate it (default matches the SigLIP convention). An empty
// template means "use the label verbatim" — handy for cross-lingual single words.
const HYPOTHESIS_TEMPLATE = "This is a photo of {}.";
function applyTemplate(labels, template) {
  if (template === "") return labels.slice();
  const t = template || HYPOTHESIS_TEMPLATE;
  return labels.map((l) => t.replace("{}", l));
}

async function run(id, imageURL, labels, template) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const imageInputs = await pipe.processor(image);
  const prompts = applyTemplate(labels, template);
  // SigLIP 2's Gemma tokenizer reports model_max_length = 1e30 — pad explicitly to 64 or it throws.
  const textInputs = pipe.tokenizer(prompts, {
    padding: "max_length",
    max_length: 64,
    truncation: true,
  });
  const output = await pipe.model({ ...textInputs, ...imageInputs });

  const logits = Array.from(output.logits_per_image.data); // scale·cos + bias  (per label, independent)
  const sig = logits.map(sigmoid); // SigLIP's real read-out: independent per-label probability
  const soft = softmax(logits); // shown ONLY to contrast against the softmax CLIP would use

  const imgDims = output.image_embeds?.dims ?? null;
  const txtDims = output.text_embeds?.dims ?? null;
  let cos = null;
  if (output.image_embeds && output.text_embeds && imgDims && txtDims) {
    const dim = imgDims[imgDims.length - 1];
    cos = cosines(
      Array.from(output.image_embeds.data),
      Array.from(output.text_embeds.data),
      dim,
      labels.length,
    );
  }
  const fit = cos ? fitScaleBias(cos, logits) : null;

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    labels,
    prompts,
    logits,
    sigmoid: sig,
    softmax: soft,
    cosines: cos,
    fit, // { scale, bias } or null
    imgDims,
    txtDims,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.labels, e.data.template);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
