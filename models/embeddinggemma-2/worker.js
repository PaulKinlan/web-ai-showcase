// EmbeddingGemma 2 worker — all inference off the main thread, Web Workers (Baseline widely available).
//
// Model: onnx-community/embeddinggemma-2-ONNX (canonical weights google/embeddinggemma-2), q4, WebGPU.
//
// What makes EmbeddingGemma 2 different from every other embedder in this showcase:
//   1. Upstream documents it as MULTIMODAL into one space: Google's model card describes text (incl. code),
//      images, video, and audio all projecting into the same 768-d vector space. This showcase does not
//      exercise multimodal retrieval: this family is text-only by decision, and the worker loads the
//      text encoder only (the 270M backbone). The vision (170M) and audio (300M) ONNX graphs are not loaded.
//   2. Those encoders are SELECTIVELY LOADABLE — dropping vision_config/audio_config before
//      AutoModel.from_pretrained means the text-only page downloads 175 MB, not 473 MB. That budget
//      choice is the point of the multi-model rung, so the worker reports exactly which graphs it loaded.
//   3. Text is INSTRUCTION-PROMPTED with the documented task prefixes (`task: search result | query: …`,
//      `classification`, `clustering`, `sentence similarity`, documents as `title: … | text: …`).
//   4. It is MATRYOSHKA: the 768-d vector truncates to 512 / 256 / 128 and is re-normalized after slicing.
//   5. The model card documents an 8192-token context window (v1 was 2048) while the exported config.json
//      sets different values (262144 / 512); neither figure is verified by any page here (demos use short
//      bounded passages). The soft-token costs (~280 per image, ~140 per video frame) are card figures for
//      encoders this family does not load.
//
// Two measured facts that shape this worker (real headless Chrome, 2026-10-07):
//   • The QUANTIZED exports cannot create a session on the ONNX Runtime Web WASM execution provider:
//     session creation fails with `ERROR_CODE: 9 … GatherBlockQuantized … /model/embed_tokens/Gather_Quant`
//     because the 262k-entry embedding table is block-quantized and the WASM EP has no kernel for it.
//     The fp32 export does load on WASM (61 s for 1085 MB), so the failure is the quantization, not the
//     architecture. This page therefore runs on WebGPU and says so; the WASM path is reported as blocked
//     rather than papered over with a fake fallback.
//   • fp16 is not offered at all: the model card warns that EmbeddingGemma 2's activation range exceeds
//     fp16's dynamic range and the model then returns NaN or silently degraded vectors instead of raising.
//     q4 and q8 are both safe (q8 >= 0.9997 cosine to fp32, q4 >= 0.988 for text).
//
// We ask the graph for its `sentence_embedding` output (mean pooling is inside the ONNX graph, followed by
// the 512->768 projection), so the worker normalizes nothing itself — the graph returns unit vectors and
// cosine similarity is a plain dot product.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

let lib = null;
let model = null;
let tokenizer = null;
let processor = null;
let device = "webgpu";
let dtype = "q4";
let loadedModalities = { vision: false, audio: false };

function post(msg) {
  self.postMessage(msg);
}

/**
 * EmbeddingGemma 2's documented task prefixes (model card "Task Instruction Prefixes").
 * `query` is the asymmetric retrieval side; `document` is the corpus side. The symmetric tasks
 * (classification / clustering / sentence similarity) apply the SAME prefix to everything compared.
 */
const PROMPTS = {
  query: (t) => `task: search result | query: ${t}`,
  document: (t) => `title: none | text: ${t}`,
  question: (t) => `task: question answering | query: ${t}`,
  factcheck: (t) => `task: fact checking | query: ${t}`,
  code: (t) => `task: code retrieval | query: ${t}`,
  classification: (t) => `task: classification | query: ${t}`,
  clustering: (t) => `task: clustering | query: ${t}`,
  similarity: (t) => `task: sentence similarity | query: ${t}`,
  none: (t) => t,
};

/**
 * Turn a raw ORT/Transformers.js failure into something a visitor can act on.
 * The GatherBlockQuantized failure is the one real trap on this family: it looks like a corrupt download
 * or an out-of-memory error but is actually "this execution provider has no kernel for this graph".
 */
function describeLoadError(err, targetDevice) {
  const raw = String(err?.message ?? err);
  if (/GatherBlockQuantized/i.test(raw)) {
    return new Error(
      `This quantized EmbeddingGemma 2 export can't run on the ${targetDevice.toUpperCase()} backend: ` +
        `the graph's block-quantized embedding lookup (GatherBlockQuantized) has no kernel on the ONNX ` +
        `Runtime Web WASM execution provider. WebGPU is required for the q4/q8 exports. (Original: ${raw})`,
    );
  }
  if (/out of memory|OOM|Array buffer allocation failed|RuntimeError: memory/i.test(raw)) {
    return new Error(
      `Ran out of memory while loading the ${targetDevice.toUpperCase()} model. Close other tabs and try again. (Original: ${raw})`,
    );
  }
  if (/no available backend|Failed to get GPU adapter|requestAdapter/i.test(raw)) {
    return new Error(`No usable ${targetDevice.toUpperCase()} backend in this browser. (Original: ${raw})`);
  }
  return err instanceof Error ? err : new Error(raw);
}

async function getLib() {
  if (!lib) {
    lib = await import(TRANSFORMERS_URL);
    lib.env.allowLocalModels = false;
  }
  return lib;
}

/**
 * Load (or reload) the model with exactly the modality encoders requested.
 *
 * Dropping `vision_config` / `audio_config` BEFORE `from_pretrained` is the documented way to load only
 * the text backbone: the loader then never fetches `vision_encoder*.onnx` / `audio_encoder*.onnx`.
 * Switching modalities means re-creating the session, so it is an explicit user action, not a silent cost.
 */
async function ensureLoaded(want = {}) {
  const vision = want.vision === true;
  const audio = want.audio === true;
  const nextDtype = want.dtype || dtype;
  const already =
    model && tokenizer && loadedModalities.vision === vision && loadedModalities.audio === audio && dtype === nextDtype;
  if (already) return;

  const { AutoConfig, AutoModel, AutoTokenizer } = await getLib();
  const previous = model;
  model = null;
  if (previous) {
    try {
      await previous.dispose?.();
    } catch {
      /* the old session is being discarded anyway */
    }
  }

  const config = await AutoConfig.from_pretrained("onnx-community/embeddinggemma-2-ONNX");
  if (!vision) config.vision_config = null;
  if (!audio) config.audio_config = null;

  const perComponent = {};
  perComponent.model = nextDtype;
  if (vision) perComponent.vision_encoder = nextDtype;
  if (audio) perComponent.audio_encoder = nextDtype;

  try {
    model = await AutoModel.from_pretrained("onnx-community/embeddinggemma-2-ONNX", {
      config,
      device,
      dtype: perComponent,
      progress_callback: (p) => post({ type: "progress", p }),
    });
  } catch (err) {
    throw describeLoadError(err, device);
  }
  if (!tokenizer) tokenizer = await AutoTokenizer.from_pretrained("onnx-community/embeddinggemma-2-ONNX");
  dtype = nextDtype;
  loadedModalities = { vision, audio };

  // Report the graphs that actually exist in the session rather than what we asked for.
  post({
    type: "ready",
    device,
    dtype,
    modalities: loadedModalities,
    sessions: Object.keys(model.sessions ?? {}),
  });
}

async function getProcessor() {
  if (!processor) {
    const { AutoProcessor } = await getLib();
    processor = await AutoProcessor.from_pretrained("onnx-community/embeddinggemma-2-ONNX");
  }
  return processor;
}

/** Unit 768-d vectors → plain JS arrays, plus the pre-normalization magnitude for "See inside". */
function unpack(output, n) {
  const tensor = output?.sentence_embedding ?? output?.pooler_output ?? output?.last_hidden_state;
  if (!tensor) {
    throw new Error(`Model returned no embedding output (got: ${Object.keys(output ?? {}).join(", ") || "nothing"})`);
  }
  const dim = tensor.dims[tensor.dims.length - 1];
  const rows = tensor.tolist();
  const embeddings = [];
  const norms = [];
  for (let i = 0; i < n; i++) {
    const vec = Array.isArray(rows[i]) ? rows[i] : rows;
    let sum = 0;
    for (const v of vec) sum += v * v;
    norms.push(Math.sqrt(sum));
    embeddings.push(vec);
  }
  const flat = tensor.data;
  let nan = 0;
  if (flat && flat.length) {
    for (let i = 0; i < flat.length; i += 1) if (!Number.isFinite(flat[i])) nan += 1;
  }
  return { embeddings, norms, dim, nan };
}

async function embedText(id, texts, mode) {
  await ensureLoaded({ vision: loadedModalities.vision, audio: loadedModalities.audio });
  const t0 = performance.now();
  const wrap = PROMPTS[mode] || PROMPTS.none;
  const prompted = texts.map((t) => wrap(t));
  // padding+truncation so a mixed-length batch is one padded tensor (a raw tokenizer call does NOT pad).
  const inputs = tokenizer(prompted, { padding: true, truncation: true });
  const output = await model(inputs);
  const { embeddings, norms, dim, nan } = unpack(output, texts.length);
  let tokenCounts = null;
  try {
    const ids = inputs.input_ids;
    tokenCounts = Array.isArray(ids)
      ? ids.map((r) => (Array.isArray(r) ? r.length : (r.dims?.[0] ?? null)))
      : ids.tolist().map((r) => r.length);
  } catch {
    /* token counts are a nicety, not required */
  }
  post({
    type: "result",
    id,
    kind: "text",
    texts,
    embeddings,
    norms,
    dim,
    tokenCounts,
    nan,
    mode,
    ms: Math.round(performance.now() - t0),
    device,
    dtype,
    modalities: loadedModalities,
  });
}

/**
 * One inference for a whole experiment: each item gets its OWN task prefix and text, and all of them go
 * through the model as a single padded batch. This is how the prefix-comparison page can show six
 * different promptings of the same pair without paying six round-trips to the worker.
 */
/**
 * Batch embed. Each item may carry its own `maxTokens`: the tokenizer lives HERE, so a length limit is
 * applied in the worker rather than by a second tokenizer on the main thread. Items are grouped by their
 * limit so a mixed batch costs one model call per distinct limit instead of one call per item, and the
 * merged result preserves the caller's item order.
 */
async function embedBatch(id, items, label) {
  await ensureLoaded({ vision: loadedModalities.vision, audio: loadedModalities.audio });
  const t0 = performance.now();
  const prompted = items.map((it) => (PROMPTS[it.mode] || PROMPTS.none)(it.text));
  const groups = new Map();
  prompted.forEach((p, i) => {
    const limit = Number.isFinite(items[i]?.maxTokens) ? items[i].maxTokens : null;
    if (!groups.has(limit)) groups.set(limit, []);
    groups.get(limit).push(i);
  });
  const dims = new Array(items.length);
  const tokenCounts = new Array(items.length);
  const norms = new Array(items.length);
  let dim = null;
  let nan = 0;
  for (const [limit, indices] of groups) {
    const inputs = tokenizer(
      indices.map((i) => prompted[i]),
      limit ? { padding: true, truncation: true, max_length: limit } : { padding: true, truncation: true },
    );
    // Real per-item token counts from the file's own attention mask: true length, padding excluded,
    // counted AFTER the prefix was prepended and AFTER any length limit was applied.
    try {
      const mask = inputs.attention_mask.tolist();
      mask.forEach((row, k) => {
        tokenCounts[indices[k]] = row.reduce((sum, v) => sum + (v ? 1 : 0), 0);
      });
    } catch { /* counts are advisory, never invented */ }
    const output = await model(inputs);
    const un = unpack(output, indices.length);
    dim = un.dim;
    nan += un.nan;
    un.embeddings.forEach((d, k) => { dims[indices[k]] = d; });
    un.norms.forEach((n, k) => { norms[indices[k]] = n; });
  }
  post({
    type: "result",
    id,
    kind: "batch",
    label,
    items,
    prompted,
    embeddings: dims,
    norms,
    dim,
    nan,
    tokenCounts,
    ms: Math.round(performance.now() - t0),
    device,
    dtype,
    modalities: loadedModalities,
  });
}

/**
 * Embed media. NOTE: This path is not reachable from any page in this family today: the showcase is
 * text-only by decision, no page sends runMedia, and media inference is unproven on this VM (a software
 * WebGPU adapter). The handler is kept intact to avoid breaking the worker message protocol, but a reader
 * should not mistake this unexercised path for a working feature.
 * Upstream contract: `images` are data/object URLs; one list entry per sample, so N images embedded
 * separately are `[[a],[b]]` while a single sample made of two images is `[a,b]` (documented nesting rule).
 * Video frames arrive as URLs and are decoded with `load_video` (browser decoding, 1 frame/second by
 * default and uniformly subsampled above 32 frames).
 */
async function embedMedia(id, { images = null, videos = null, text = null, maxSoftTokens = null } = {}) {
  await ensureLoaded({ vision: (images?.length ?? 0) > 0 || (videos?.length ?? 0) > 0, audio: false });
  const { load_image, load_video } = await getLib();
  const proc = await getProcessor();
  // Vision token budget: 70 / 140 / 280 (default) / 560 / 1120 soft tokens per image. It is a documented
  // knob that trades image detail for latency and context, so it is worth being able to move at runtime.
  if (maxSoftTokens && proc.image_processor) proc.image_processor.max_soft_tokens = maxSoftTokens;
  const t0 = performance.now();

  const imageSamples = images ? await Promise.all(images.map(async (sample) => {
    const list = Array.isArray(sample) ? sample : [sample];
    return Promise.all(list.map((u) => load_image(u)));
  })) : null;

  const videoSamples = videos ? await Promise.all(videos.map(async (v) => {
    const url = typeof v === "string" ? v : v.url;
    const fps = typeof v === "string" ? 1 : (v.fps ?? 1);
    return await load_video(url, { fps });
  })) : null;

  const inputs = await proc(text ?? null, imageSamples, null, videoSamples);
  const output = await model(inputs);
  const n = (imageSamples?.length ?? 0) + (videoSamples?.length ?? 0) || 1;
  const { embeddings, norms, dim, nan } = unpack(output, n);
  const softTokens = {
    image: inputs.num_soft_tokens_per_image ?? null,
    video: inputs.num_soft_tokens_per_video ?? null,
    frames: inputs.num_frames_per_video ?? null,
  };
  post({
    type: "result",
    id,
    kind: images ? "image" : "video",
    embeddings,
    norms,
    dim,
    nan,
    softTokens,
    maxSoftTokens,
    ms: Math.round(performance.now() - t0),
    device,
    dtype,
    modalities: loadedModalities,
  });
}

/** Reset the processor when the model changes so a stale image-processor config cannot linger. */
async function unload() {
  if (model) {
    try {
      await model.dispose?.();
    } catch {
      /* nothing useful to do */
    }
  }
  model = null;
  tokenizer = null;
  processor = null;
  loadedModalities = { vision: false, audio: false };
}

self.addEventListener("message", async (e) => {
  const data = e.data ?? {};
  try {
    if (data.type === "load") {
      device = data.device === "wasm" ? "wasm" : "webgpu";
      await ensureLoaded({ vision: data.vision === true, audio: data.audio === true, dtype: data.dtype || "q4" });
    } else if (data.type === "run") {
      await embedText(data.id, data.texts, data.mode);
    } else if (data.type === "runBatch") {
      await embedBatch(data.id, data.items, data.label);
    } else if (data.type === "runMedia") {
      await embedMedia(data.id, data);
    } else if (data.type === "unload") {
      await unload();
      post({ type: "unloaded" });
    }
  } catch (err) {
    post({ type: "error", id: data.id, message: String(err?.message ?? err) });
  }
});
