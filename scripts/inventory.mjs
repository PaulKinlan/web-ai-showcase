#!/usr/bin/env node
// Evidence-backed inventory of the BROWSER-RUNNABLE Hugging Face model universe.
//
// Queries the real runtime-compatibility catalogues (Transformers.js/ONNX, WebLLM/MLC), applies
// explicit eligibility + family-deduplication rules, and produces:
//   inventory/eligible.ndjson  — legacy path containing typed verified/candidate/blocked evidence
//   inventory/summary.json     — exact inclusive inventory denominator + typed status counts
//   inventory/collisions.json  — deterministic scan-family and catalogue-merge collision ledger
//   inventory/reviewed-aliases.json — durable reviewed capability-family collision policy (input)
//   models.json (merged)       — unverified representatives added as typed pending candidates;
//                                existing built/blocked entries are never dropped or downgraded.
//
// Run: `node scripts/inventory.mjs [--pages N] [--no-merge]`.
// Rules of the road (see AGENTS.md / SKILL): the inclusive inventory denominator retains verified,
// candidate-unverified, and blocked families. Metadata is discovery evidence, not runtime proof.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import {
  deduplicateFamilies,
  discoveryClassification,
  familyKey,
  findCatalogueCollision,
  isExactCatalogueMatch,
  isRetainedInventoryCandidate,
  slugify,
  validateReviewedAliasPolicy,
} from "./inventory-lib.mjs";

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
  const reviewedAliasPolicy = JSON.parse(
    await readFile(new URL("inventory/reviewed-aliases.json", ROOT), "utf8"),
  );
  const policyFailures = validateReviewedAliasPolicy(reviewedAliasPolicy);
  if (policyFailures.length) {
    throw new Error(`invalid reviewed alias policy: ${policyFailures.join("; ")}`);
  }

  const sources = [
    {
      query: { library: "transformers.js", sort: "downloads", limit: "100", full: "false" },
      runtime: "transformers.js",
    },
    { query: { library: "onnx", sort: "downloads", limit: "100", full: "false" }, runtime: "onnx" },
  ];

  const all = [];
  const apiTotals = {};
  for (const s of sources) {
    try {
      const { models, apiTotal } = await collect(s.query, s.runtime);
      apiTotals[s.runtime] = apiTotal;
      all.push(...models);
      console.error(
        `  ${s.runtime}: HF reports ${apiTotal} total; collected ${models.length} with a supported task`,
      );
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

  // Family-dedup is deterministic. In particular, WebLLM precision ports collapse by their
  // declared base_model:quantized identity, not by the noisy q0/q3/q4 repository spelling.
  const deduped = deduplicateFamilies(all);
  const reps = deduped.representatives.map((model) => ({
    ...model,
    discovery: discoveryClassification(model),
  }));
  const verified = reps.filter((model) => model.discovery.status === "verified-eligible");
  const candidates = reps.filter((model) => model.discovery.status === "candidate-unverified");
  const blocked = reps.filter((model) => model.discovery.status === "blocked");

  const byTask = {};
  for (const model of reps) {
    const counts = byTask[model.task] ??= {
      families: 0,
      verifiedEligible: 0,
      candidateUnverified: 0,
      blocked: 0,
    };
    counts.families++;
    if (model.discovery.status === "verified-eligible") counts.verifiedEligible++;
    else if (model.discovery.status === "candidate-unverified") counts.candidateUnverified++;
    else counts.blocked++;
  }

  await mkdir(new URL("inventory/", ROOT), { recursive: true });
  await writeFile(
    new URL("inventory/eligible.ndjson", ROOT),
    reps.map((model) => JSON.stringify(model)).join("\n") + "\n",
  );

  // Candidates stay pending and explicitly unverified. Do not assign a proven backend/runtime or
  // size until file-level browser-artifact and feasibility verification exists.
  let mergedInto = 0;
  let retainedInventoryCandidates = 0;
  const catalogueCollisions = [];
  const cat = JSON.parse(await readFile(new URL("models.json", ROOT), "utf8"));
  const existingSlug = new Set(cat.models.map((model) => model.slug));
  if (!NO_MERGE) {
    for (const model of candidates) {
      const collision = findCatalogueCollision(model, cat.models, reviewedAliasPolicy);
      if (isRetainedInventoryCandidate(model, collision)) {
        retainedInventoryCandidates++;
        continue;
      }
      if (isExactCatalogueMatch(model, collision)) continue;
      if (collision) {
        catalogueCollisions.push({
          phase: "catalogue-candidate-merge",
          collisionKey: collision.collisionKey,
          kept: collision.model.hfId,
          keptStatus: collision.model.status,
          removed: model.id,
          reason: collision.reason,
        });
        continue;
      }
      let slug = slugify(model.id);
      while (existingSlug.has(slug)) slug += "-x";
      existingSlug.add(slug);
      cat.models.push({
        slug,
        name: model.id.split("/").pop(),
        hfId: model.id,
        task: model.task,
        modality: model.modality,
        family: model.familyKey,
        license: model.license,
        sizeMB: null,
        blurb: `${model.task} candidate — browser artifact and feasible size are unverified.`,
        unlocks: "Pending artifact, size, and runtime verification before build selection.",
        status: "pending",
        candidate: {
          status: "unverified",
          claimedRuntime: model.runtime,
          provenance: model.discovery,
        },
      });
      mergedInto++;
    }
    cat.generated = "auto (inventory.mjs)";
    await writeFile(new URL("models.json", ROOT), JSON.stringify(cat, null, 2) + "\n");
  }

  const collisions = {
    schemaVersion: 1,
    scan: deduped.collisions,
    catalogueMerge: catalogueCollisions.sort((a, b) =>
      a.collisionKey.localeCompare(b.collisionKey) || a.removed.localeCompare(b.removed)
    ),
  };
  await writeFile(
    new URL("inventory/collisions.json", ROOT),
    JSON.stringify(collisions, null, 2) + "\n",
  );

  const builtCount = cat.models.filter((model) => model.status === "built").length;
  const summary = {
    denominatorNote:
      "inventoryFamilies is the family-deduplicated refining lower-bound denominator at this scan " +
      "depth. It includes verified-eligible, candidate-unverified, and blocked families; candidates " +
      "are never discarded merely because artifact/runtime/size proof is absent. rawDiscoveries " +
      "preserves the pre-dedup scan count. Never imply complete/all coverage.",
    scanPages: MAX_PAGES,
    sources: [
      "library=transformers.js",
      "library=onnx",
      "author=mlc-ai (-MLC)",
      "MediaPipe Tasks (curated)",
    ],
    depthCurve: { "8": 635, "10": 754, "20": 1288, "40": 2355 },
    missionBaseline: {
      scanPages: 8,
      eligibleFamilies: 635,
      note: "the original mission denominator; this run scans deeper",
    },
    blockedRule:
      "Blocked and candidate-unverified families stay in inventoryFamilies; never shrink the denominator because proof is incomplete or a model is hard.",
    verificationRule:
      "HF library/tag metadata is a discovery signal, not browser eligibility proof. verified-eligible requires a verified browser artifact/runtime and feasible size.",
    apiTotalsRaw: apiTotals,
    rawDiscoveries: all.length,
    scanFamilyCollisions: deduped.collisions.length,
    duplicateDiscoveries: deduped.duplicateDiscoveries,
    inventoryFamilies: reps.length,
    collectedRepresentativeFamilies: reps.length,
    eligibleFamilies: verified.length,
    verifiedEligibleFamilies: verified.length,
    candidateUnverifiedFamilies: candidates.length,
    blockedFamilies: blocked.length,
    byTask,
    catalogue: {
      total: cat.models.length,
      built: builtCount,
      pending: cat.models.filter((m) => m.status === "pending").length,
      blocked: cat.models.filter((m) => m.status === "blocked").length,
      addedThisRun: mergedInto,
      rawCandidatesAdded: mergedInto + catalogueCollisions.length,
      candidateCollisionsThisRun: catalogueCollisions.length,
    },
    collisionLedger: "inventory/collisions.json",
  };
  await writeFile(new URL("inventory/summary.json", ROOT), JSON.stringify(summary, null, 2) + "\n");

  console.error("\n=== INVENTORY SUMMARY ===");
  console.error(
    `inventory families: ${reps.length} (verified eligible: ${verified.length}, candidate unverified: ${candidates.length}, blocked: ${blocked.length})`,
  );
  console.error(
    `catalogue: ${cat.models.length} total, ${builtCount} built, +${mergedInto} newly added, ${retainedInventoryCandidates} retained inventory candidates`,
  );
  console.error(`tasks covered: ${Object.keys(byTask).length}`);
  console.error("evidence -> inventory/eligible.ndjson + inventory/summary.json");
}

main().catch((e) => {
  console.error("inventory failed:", e.message);
  process.exit(1);
});
