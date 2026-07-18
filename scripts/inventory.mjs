#!/usr/bin/env node
// Evidence-backed inventory of the BROWSER-RUNNABLE Hugging Face model universe.
//
// Queries the real runtime-compatibility catalogues (Transformers.js/ONNX, WebLLM/MLC), applies
// explicit eligibility + family-deduplication rules, and produces:
//   inventory/eligible.ndjson  — one line per eligible model (full evidence: id, task, runtime,
//                                license, likes, downloads, gated, size hint, model-card URL)
//   inventory/summary.json     — exact denominators + counts by task/modality/runtime + the
//                                representative model chosen per (task, family)
//   models.json (merged)       — representatives added as status:"pending" (never drops/downgrades
//                                existing entries; preserves "built").
//
// Run: `node scripts/inventory.mjs [--pages N] [--no-merge]`.
// Rules of the road (see AGENTS.md / SKILL): denominator = eligible families; blocked (gated/too
// large) and device-only stay IN the denominator; we never shrink it because a model is hard.

import { readFile, writeFile, mkdir } from "node:fs/promises";

const MAX_PAGES = Number(argVal("--pages") ?? 8); // 100/page => up to 800 models per source
const NO_MERGE = process.argv.includes("--no-merge");
const HERE = new URL(".", import.meta.url);
const ROOT = new URL("../", HERE);

// pipeline_tag -> {modality, our capability group}. This is the supported task set; the routine adds
// groups discovered here rather than treating it as a cap.
const TASKS = {
  "text-classification": "text",
  "token-classification": "text",
  "feature-extraction": "text",
  "sentence-similarity": "text",
  summarization: "text",
  translation: "text",
  "text2text-generation": "text",
  "text-generation": "text",
  "fill-mask": "text",
  "question-answering": "text",
  "zero-shot-classification": "text",
  "automatic-speech-recognition": "audio",
  "audio-classification": "audio",
  "text-to-speech": "audio",
  "text-to-audio": "audio",
  "image-classification": "vision",
  "object-detection": "vision",
  "image-segmentation": "vision",
  "mask-generation": "vision",
  "depth-estimation": "vision",
  "image-feature-extraction": "vision",
  "image-to-image": "vision",
  "zero-shot-image-classification": "vision-language",
  "image-to-text": "vision-language",
  "image-text-to-text": "vision-language",
  "visual-question-answering": "vision-language",
  "document-question-answering": "vision-language",
};

function argVal(flag) {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

async function hfPage(query, cursor) {
  const url = new URL("https://huggingface.co/api/models");
  for (const [k, v] of Object.entries(query)) url.searchParams.set(k, v);
  if (cursor) url.searchParams.set("cursor", cursor);
  const res = await fetch(url, { headers: { "user-agent": "web-ai-showcase/inventory" } });
  if (!res.ok) throw new Error(`HF ${res.status} ${url}`);
  const total = Number(res.headers.get("x-total-count") ?? 0);
  const link = res.headers.get("link") ?? "";
  const next = /<([^>]+)>;\s*rel="next"/.exec(link)?.[1];
  const nextCursor = next ? new URL(next).searchParams.get("cursor") : null;
  return { rows: await res.json(), total, nextCursor };
}

async function collect(query, runtime) {
  let cursor = null,
    pages = 0,
    total = 0;
  const out = [];
  do {
    const { rows, total: t, nextCursor } = await hfPage(query, cursor);
    total = t || total;
    for (const r of rows) {
      const task = r.pipeline_tag;
      if (!TASKS[task]) continue;
      out.push({
        id: r.id,
        task,
        modality: TASKS[task],
        runtime,
        likes: r.likes ?? 0,
        downloads: r.downloads ?? 0,
        gated: r.gated ?? false,
        tags: r.tags ?? [],
        license: (r.tags ?? []).find((t) => t.startsWith("license:"))?.slice(8) ?? null,
        card: `https://huggingface.co/${r.id}`,
      });
    }
    cursor = nextCursor;
  } while (cursor && ++pages < MAX_PAGES);
  return { models: out, apiTotal: total };
}

// Family key: normalise the model NAME so fine-tunes/quants/sizes of one architecture collapse to one
// family, while genuinely different architectures stay distinct. We keep the org out (many orgs
// re-export the same arch) and strip runtime/quant/size noise.
function familyKey(id) {
  let n = id.split("/").pop().toLowerCase();
  n = n
    .replace(/[-_.](onnx|ort|web|mlc|gguf|ggml)\b/g, "")
    .replace(/[-_.](q4f16|q4|q8|int8|int4|fp16|fp32|bf16|uint8|quantized|8bit|4bit)([-_.]\w+)?/g, "")
    .replace(/[-_.]\d+(\.\d+)?b\b/g, "") // 0.5b, 1b, 7b size markers
    .replace(/[-_.](base|small|tiny|mini|large|xl|xxl|medium|nano|micro)\b/g, "")
    .replace(/[-_.]v?\d+(\.\d+)*\b/g, "") // version tails
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return n || id.toLowerCase();
}

function slugify(id) {
  return id.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 80);
}

// MediaPipe Tasks Web runtime — model-backed landmarkers/segmenters Google ships as .task/.tflite
// bundles that run in the browser via @mediapipe/tasks-*. Not on the HF list API, so curated here
// (each is a real, downloadable, browser-runnable model). Kept in the denominator.
const MEDIAPIPE = [
  { id: "mediapipe/face-landmarker", task: "face-landmark-detection", modality: "vision" },
  { id: "mediapipe/hand-landmarker", task: "hand-landmark-detection", modality: "vision" },
  { id: "mediapipe/pose-landmarker", task: "pose-landmark-detection", modality: "vision" },
  { id: "mediapipe/gesture-recognizer", task: "gesture-recognition", modality: "vision" },
  { id: "mediapipe/image-segmenter", task: "image-segmentation", modality: "vision" },
  { id: "mediapipe/interactive-segmenter", task: "mask-generation", modality: "vision" },
  { id: "mediapipe/object-detector", task: "object-detection", modality: "vision" },
  { id: "mediapipe/face-detector", task: "face-detection", modality: "vision" },
].map((m) => ({
  ...m,
  runtime: "mediapipe",
  likes: 0,
  downloads: 0,
  gated: false,
  tags: [],
  license: "apache-2.0",
  card: "https://ai.google.dev/edge/mediapipe/solutions/tasks",
}));

// WebLLM/MLC prebuilts don't set pipeline_tag — collect -MLC repos and infer the task from the name.
async function collectWebLLM() {
  let cursor = null, pages = 0, total = 0;
  const out = [];
  do {
    const { rows, total: t, nextCursor } = await hfPage(
      { author: "mlc-ai", sort: "downloads", limit: "100", full: "false" },
      cursor,
    );
    total = t || total;
    for (const r of rows) {
      if (!/-MLC$/i.test(r.id)) continue;
      const vlm = /(vlm|vision|llava|phi-3.5-vision|internvl|qwen2?-vl)/i.test(r.id);
      out.push({
        id: r.id,
        task: vlm ? "image-text-to-text" : "text-generation",
        modality: vlm ? "vision-language" : "text",
        runtime: "webllm",
        likes: r.likes ?? 0,
        downloads: r.downloads ?? 0,
        gated: r.gated ?? false,
        tags: r.tags ?? [],
        license: (r.tags ?? []).find((x) => x.startsWith("license:"))?.slice(8) ?? null,
        card: `https://huggingface.co/${r.id}`,
      });
    }
    cursor = nextCursor;
  } while (cursor && ++pages < MAX_PAGES);
  return { models: out, apiTotal: total };
}

async function main() {
  const sources = [
    { query: { library: "transformers.js", sort: "downloads", limit: "100", full: "false" }, runtime: "transformers.js" },
    { query: { library: "onnx", sort: "downloads", limit: "100", full: "false" }, runtime: "onnx" },
  ];

  const all = [];
  const apiTotals = {};
  for (const s of sources) {
    try {
      const { models, apiTotal } = await collect(s.query, s.runtime);
      apiTotals[s.runtime] = apiTotal;
      all.push(...models);
      console.error(`  ${s.runtime}: HF reports ${apiTotal} total; collected ${models.length} with a supported task`);
    } catch (e) {
      console.error(`  ${s.runtime}: FAILED ${e.message}`);
      apiTotals[s.runtime] = `error: ${e.message}`;
    }
  }
  try {
    const { models, apiTotal } = await collectWebLLM();
    apiTotals.webllm = apiTotal;
    all.push(...models);
    console.error(`  webllm(mlc): collected ${models.length} -MLC repos`);
  } catch (e) {
    console.error(`  webllm: FAILED ${e.message}`);
  }
  all.push(...MEDIAPIPE);
  apiTotals.mediapipe = `${MEDIAPIPE.length} curated`;
  for (const m of MEDIAPIPE) TASKS[m.task] ??= m.modality; // register discovered tasks

  // Dedup to families: representative per (task, familyKey) = the most-downloaded eligible model.
  const byFamily = new Map();
  for (const m of all) {
    const key = `${m.task}::${familyKey(m.id)}`;
    const cur = byFamily.get(key);
    if (!cur || m.downloads > cur.downloads) byFamily.set(key, { ...m, familyKey: familyKey(m.id) });
  }
  const reps = [...byFamily.values()].sort((a, b) => b.downloads - a.downloads);

  // Eligibility: gated models are BLOCKED (kept in denominator, not built). Everything else eligible.
  const eligible = reps.filter((m) => !m.gated);
  const blocked = reps.filter((m) => m.gated);

  // Counts by task + modality.
  const byTask = {};
  for (const m of reps) (byTask[m.task] ??= { families: 0, gated: 0 }).families++;
  for (const m of blocked) byTask[m.task].gated++;

  await mkdir(new URL("inventory/", ROOT), { recursive: true });
  await writeFile(
    new URL("inventory/eligible.ndjson", ROOT),
    reps.map((m) => JSON.stringify(m)).join("\n") + "\n",
  );

  // Merge representatives into models.json as pending (preserve existing + built).
  let mergedInto = 0;
  const cat = JSON.parse(await readFile(new URL("models.json", ROOT), "utf8"));
  const existingHf = new Set(cat.models.map((m) => m.hfId));
  const existingSlug = new Set(cat.models.map((m) => m.slug));
  if (!NO_MERGE) {
    for (const m of eligible) {
      if (existingHf.has(m.id)) continue;
      let slug = slugify(m.id);
      while (existingSlug.has(slug)) slug += "-x";
      existingSlug.add(slug);
      existingHf.add(m.id);
      cat.models.push({
        slug,
        name: m.id.split("/").pop(),
        hfId: m.id,
        task: m.task,
        modality: m.modality,
        backend: m.runtime === "webllm" ? "webgpu" : "wasm",
        runtime: m.runtime,
        family: m.familyKey,
        license: m.license,
        sizeMB: null,
        blurb: `${m.task} model — see the model card.`,
        unlocks: "TODO: filled from the model card at build time.",
        status: "pending",
      });
      mergedInto++;
    }
    cat.generated = "auto (inventory.mjs)";
    await writeFile(new URL("models.json", ROOT), JSON.stringify(cat, null, 2) + "\n");
  }

  const builtCount = cat.models.filter((m) => m.status === "built").length;
  const summary = {
    generatedNote: "denominator = eligible families (deduped). blocked (gated) + device-only stay IN the denominator.",
    apiTotalsRaw: apiTotals,
    collectedRepresentativeFamilies: reps.length,
    eligibleFamilies: eligible.length,
    blockedFamilies: blocked.length,
    byTask,
    catalogue: {
      total: cat.models.length,
      built: builtCount,
      pending: cat.models.filter((m) => m.status === "pending").length,
      blocked: cat.models.filter((m) => m.status === "blocked").length,
      addedThisRun: mergedInto,
    },
  };
  await writeFile(new URL("inventory/summary.json", ROOT), JSON.stringify(summary, null, 2) + "\n");

  console.error("\n=== INVENTORY SUMMARY ===");
  console.error(`eligible families: ${eligible.length} (blocked/gated: ${blocked.length})`);
  console.error(`catalogue: ${cat.models.length} total, ${builtCount} built, +${mergedInto} added this run`);
  console.error(`tasks covered: ${Object.keys(byTask).length}`);
  console.error("evidence -> inventory/eligible.ndjson + inventory/summary.json");
}

main().catch((e) => {
  console.error("inventory failed:", e.message);
  process.exit(1);
});
