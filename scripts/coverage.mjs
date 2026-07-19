#!/usr/bin/env node
// Coverage against a BOUNDED, meaningful denominator: distinct ARCHITECTURE FAMILIES.
//
// Why: the raw browser-runnable "eligible family" count from inventory.mjs does not converge — it
// grows with scan depth (635 @ --pages 8 → 754 @ 10 → 1288 @ 20 → 2355 @ 40 …) because the HF long
// tail of transformers.js/ONNX re-exports and fine-tunes is effectively unbounded. A showcase's real
// target is not "every repo" but "every KIND of model" — the distinct architecture families. That set
// IS bounded and stable, so it's the honest denominator for coverage.
//
// This reads models.json (built) and classifies each built model into an architecture family; the
// denominator is the curated taxonomy below (the canonical browser-runnable architectures). Raw
// inventory (inventory/eligible.ndjson) is kept as evidence of the universe + to discover NEW
// architectures to add to the taxonomy. Run: `node scripts/coverage.mjs`.

import { readFile } from "node:fs/promises";

// Curated taxonomy of canonical BROWSER-RUNNABLE architecture families. Add to this as the inventory
// surfaces genuinely new architectures (not new fine-tunes). Each: family -> name-substring patterns.
const FAMILIES = {
  // Text — encoder
  bert: ["bert", "-ner", "distilbert", "sst-2", "squad"],
  roberta: ["roberta"],
  deberta: ["deberta", "nli-deberta"],
  albert: ["albert"],
  distilbert: ["distilbert"],
  electra: ["electra"],
  // Text — embeddings / rerank
  "minilm-embed": ["minilm", "all-minilm"],
  bge: ["bge-"],
  gte: ["gte-"],
  e5: ["e5-", "-e5"],
  nomic: ["nomic-embed"],
  "cross-encoder": ["ms-marco", "cross-encoder", "reranker"],
  // Text — generation (decoder)
  llama: ["llama", "tinyllama"],
  qwen: ["qwen"],
  gemma: ["gemma"],
  phi: ["phi-", "phi3", "phi-3", "phi-2"],
  mistral: ["mistral"],
  smollm: ["smollm"],
  gpt2: ["gpt2", "distilgpt2"],
  stablelm: ["stablelm"],
  // Text — seq2seq
  t5: ["t5", "flan-t5"],
  bart: ["bart", "distilbart"],
  pegasus: ["pegasus"],
  "m2m/nllb": ["m2m100", "nllb"],
  marian: ["opus-mt", "marian"],
  // Vision — classify/detect/segment/depth
  vit: ["vit-", "vit_", "vit-base", "beit", "deit"],
  detr: ["detr"],
  yolos: ["yolos"],
  "rt-detr": ["rt-detr", "rtdetr"],
  sam: ["sam", "slimsam", "segment-anything"],
  segformer: ["segformer"],
  "depth-anything": ["depth-anything", "depthanything"],
  dpt: ["dpt-"],
  rmbg: ["rmbg", "briaai"],
  swin: ["swin2sr", "swin"],
  resnet: ["resnet"],
  dinov2: ["dinov2"],
  // Vision-language
  clip: ["clip-", "clip_"],
  siglip: ["siglip"],
  florence: ["florence"],
  smolvlm: ["smolvlm"],
  moondream: ["moondream"],
  llava: ["llava", "nanollava"],
  paligemma: ["paligemma"],
  "qwen-vl": ["qwen2-vl", "qwen-vl", "qwen2.5-vl"],
  owlvit: ["owlvit", "owlv2"],
  "grounding-dino": ["grounding-dino", "groundingdino"],
  trocr: ["trocr"],
  blip: ["blip"],
  "vit-gpt2-caption": ["vit-gpt2"],
  // Audio
  whisper: ["whisper", "distil-whisper", "distil-small", "distil-large"],
  moonshine: ["moonshine"],
  wav2vec2: ["wav2vec2"],
  "ast-audio": ["ast-", "audioset"],
  kokoro: ["kokoro"],
  bark: ["bark"],
  musicgen: ["musicgen"],
  speecht5: ["speecht5"],
  // MediaPipe
  "mediapipe-hand": ["hand-landmark", "hand_landmark"],
  "mediapipe-pose": ["pose-landmark", "pose_landmark"],
  "mediapipe-face-landmark": ["face-landmark", "face_landmark"],
  "mediapipe-gesture": ["gesture"],
  "mediapipe-face-detect": ["face-detect", "face_detect", "blaze"],
};

function classify(id) {
  const s = id.toLowerCase();
  // Pick the MOST SPECIFIC match (longest matching pattern) so e.g. "distilbert"/"deberta"/"clip-vit"
  // aren't swallowed by the generic "bert"/"vit-" patterns.
  let best = null, bestLen = 0;
  for (const [fam, pats] of Object.entries(FAMILIES)) {
    for (const p of pats) {
      if (s.includes(p) && p.length > bestLen) {
        best = fam;
        bestLen = p.length;
      }
    }
  }
  return best;
}

async function main() {
  const cat = JSON.parse(await readFile(new URL("../models.json", import.meta.url), "utf8"));
  const builtByFam = new Map();
  for (const m of cat.models) {
    if (m.status !== "built") continue;
    const fam = classify(m.hfId || m.slug) || `other:${m.task}`;
    (builtByFam.get(fam) || builtByFam.set(fam, []).get(fam)).push(m.slug);
  }
  const taxo = Object.keys(FAMILIES);
  const builtTaxo = taxo.filter((f) => builtByFam.has(f));
  const pending = taxo.filter((f) => !builtByFam.has(f));

  console.log("=== ARCHITECTURE-FAMILY COVERAGE (the bounded, honest denominator) ===");
  console.log(`taxonomy families: ${taxo.length}`);
  console.log(`built (>=1 demo):  ${builtTaxo.length} / ${taxo.length}`);
  console.log(
    `built count total: ${cat.models.filter((m) => m.status === "built").length} model pages`,
  );
  console.log(`\nPENDING architecture families (${pending.length}):`);
  console.log("  " + pending.join(", "));
  const others = [...builtByFam.keys()].filter((f) => f.startsWith("other:"));
  if (others.length) {
    console.log(`\nBuilt demos outside the taxonomy (candidates to add): ${others.join(", ")}`);
  }
  console.log(
    "\nNOTE: the raw browser-runnable repo universe is unbounded by scan depth (evidence:",
  );
  console.log(
    "inventory/eligible.ndjson); this taxonomy is the stable coverage denominator. Never",
  );
  console.log(
    "claim 'all/complete' — new architectures keep appearing and get added to the taxonomy.",
  );
}
main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
