#!/usr/bin/env node
// Discover browser-runnable models from the Hugging Face API and merge new ones into models.json as
// status:"pending". Browser-runnable ≈ tagged for Transformers.js / ONNX. Never drops existing
// entries; never downgrades a "built" model. Run: `node scripts/discover-models.mjs [--limit N]`.

import { readFile, writeFile } from "node:fs/promises";

const LIMIT = Number(
  process.argv.includes("--limit") ? process.argv[process.argv.indexOf("--limit") + 1] : 60,
);
const HERE = new URL(".", import.meta.url);
const CATALOGUE = new URL("../models.json", HERE);

// pipeline_tag -> our modality bucket + a friendly default. Only tasks that have real browser demos.
const TASK_MODALITY = {
  "image-text-to-text": "vision-language",
  "zero-shot-image-classification": "vision-language",
  "image-to-text": "vision-language",
  "visual-question-answering": "vision-language",
  "depth-estimation": "vision",
  "image-segmentation": "vision",
  "mask-generation": "vision",
  "object-detection": "vision",
  "image-classification": "vision",
  "image-to-image": "vision",
  "automatic-speech-recognition": "audio",
  "text-to-speech": "audio",
  "text-to-audio": "audio",
  "audio-classification": "audio",
  "text-generation": "text",
  "feature-extraction": "text",
  "text-classification": "text",
  "token-classification": "text",
  "translation": "text",
  "summarization": "text",
  "fill-mask": "text",
};

function slugify(id) {
  return id.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 80);
}

async function hf(pathAndQuery) {
  const res = await fetch(`https://huggingface.co/api/${pathAndQuery}`, {
    headers: { "user-agent": "web-ai-showcase/discover" },
  });
  if (!res.ok) throw new Error(`HF API ${res.status} for ${pathAndQuery}`);
  return res.json();
}

async function main() {
  const cat = JSON.parse(await readFile(CATALOGUE, "utf8"));
  const existing = new Set(cat.models.map((m) => m.hfId));
  const found = [];

  // Transformers.js-tagged models, most-liked first, across the tasks we can demo.
  const rows = await hf(`models?library=transformers.js&sort=likes&limit=${LIMIT}&full=false`);
  for (const r of rows) {
    const task = r.pipeline_tag;
    const modality = TASK_MODALITY[task];
    if (!modality || existing.has(r.id)) continue;
    existing.add(r.id);
    found.push({
      slug: slugify(r.id.split("/").pop()),
      name: r.id.split("/").pop(),
      hfId: r.id,
      task,
      modality,
      backend: ["image-text-to-text", "text-generation", "depth-estimation"].includes(task)
        ? "webgpu"
        : "wasm",
      dtype: "q8",
      sizeMB: null,
      blurb: `${task} model — see the model card.`,
      unlocks: "TODO: the routine fills this from the model card.",
      status: "pending",
      likes: r.likes ?? 0,
    });
  }

  if (found.length === 0) {
    console.log("No new browser-runnable models found.");
    return;
  }
  cat.models.push(...found);
  cat.generated = "auto (discover-models.mjs)";
  await writeFile(CATALOGUE, JSON.stringify(cat, null, 2) + "\n");
  console.log(`Added ${found.length} pending model(s):`);
  for (const m of found) console.log(`  ${m.modality.padEnd(16)} ${m.hfId} (${m.likes} likes)`);
}

main().catch((e) => {
  console.error("discover failed:", e.message);
  process.exit(1);
});
