#!/usr/bin/env node
// Task 2b · Phase 1 — the model-download route inventory.
//
// Classifies EVERY built demo route by the runtime/loader family it uses to transfer model assets, who
// controls the byte transfer, and therefore what kind of resume is honestly possible. This is the exact
// denominator the site-wide <model-download-status> component + its adoption gate build on: you cannot
// adopt (or honestly limit) a route you have not classified. Emits download-routes.json.
//
// Evidence-based: it reads each demo's worker.js + page JS + index.html and matches concrete signals
// (imports of the shared loader libs, pipeline()/from_pretrained, CreateMLCEngine, MediaPipe
// FilesetResolver/.task, onnxruntime-web InferenceSession, model-prefetch.mjs, browser built-in AI).
// An unclassifiable route is recorded as family "unknown" — which the gate treats as a failure.
//
// Usage: node scripts/route-download-inventory.mjs            (writes download-routes.json)
//        node scripts/route-download-inventory.mjs --check    (print summary only, non-zero if any unknown)
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const MODELS = "models";
const routes = [];

// signal → family. Ordered by SPECIFICITY (first match wins); each family carries its honest
// byte-control + resume semantics.
const FAMILIES = {
  "transformers-resumable-prefetch": {
    // The site prefetches big weights resumably (Range/206/sha256 → cache) before from_pretrained.
    test: (t) => /model-prefetch\.mjs|prefetchModel\s*\(/.test(t),
    byteControl: "site-controlled",
    resume: "resumable",
    note:
      "site fetches weights resumably (Range/206/sha256) into transformers-cache; from_pretrained then cache-hits",
  },
  "webllm": {
    test: (t) => /lib\/webllm\.js|CreateMLCEngine|@mlc-ai|web-llm/.test(t),
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note: "WebLLM/MLC engine fetches + caches shards; no per-file resume hooks exposed to the page",
  },
  "mediapipe": {
    test: (t) =>
      /lib\/mediapipe\.js|FilesetResolver|tasks-vision|tasks-genai|\.task\b|\.tflite\b/.test(t),
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note:
      "MediaPipe Tasks fetches the .task/.tflite bundle; download managed inside the WASM runtime",
  },
  "raw-ort": {
    test: (t) => /onnxruntime-web|InferenceSession|\bort\.(InferenceSession|env)/.test(t),
    byteControl: "site-controlled",
    // refined below: resumable only if it goes through lib/model-download.js
    resume: "restart-only",
    note:
      "page fetches the .onnx bytes itself and hands them to ORT; restartable unless routed via model-download.js",
  },
  "transformers-wrapped": {
    // Third-party libs (outetts, kokoro-js) that wrap Transformers.js and download through it.
    test: (t) => /outetts@|kokoro-js[@/]|npm\/(outetts|kokoro-js)/.test(t),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "third-party library (outetts / kokoro-js) wraps Transformers.js; it downloads + caches via TJS, so per-file progress depends on the wrapper surfacing it",
  },
  "transformers-from_pretrained": {
    test: (t) =>
      /from_pretrained\s*\(|AutoModel|AutoProcessor|AutoTokenizer|ForConditionalGeneration|\.from_pretrained/
        .test(t),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "Transformers.js fetches each file (no Range); re-running resumes at whole-file granularity from Cache Storage",
  },
  "transformers-pipeline": {
    test: (t) => /lib\/webai\.js|loadPipeline\s*\(|[^a-zA-Z]pipeline\s*\(/.test(t),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note: "Transformers.js pipeline() fetches each file (no Range); whole-file cache resume only",
  },
  "browser-builtin": {
    test: (t) =>
      /window\.(ai|LanguageModel|Summarizer|Translator|Rewriter|Writer)\b|\bLanguageModel\.(create|availability)|ai\.languageModel/
        .test(t),
    byteControl: "browser-owned",
    resume: "runtime-owned",
    note:
      "the browser owns the model + its download state; adapt as a capability signal, never fabricate per-file detail",
  },
};
const FAMILY_ORDER = Object.keys(FAMILIES);

function readAll(dir) {
  // concatenate the demo's own worker + page scripts + index.html (the download entrypoints live here).
  let text = "";
  const files = [];
  const walk = (d) => {
    for (const n of readdirSync(d)) {
      const p = join(d, n);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else if (/\.(js|mjs|html)$/.test(n)) files.push(p);
    }
  };
  walk(dir);
  for (const f of files) text += "\n" + readFileSync(f, "utf8");
  return { text, files };
}

for (const slug of readdirSync(MODELS).sort()) {
  const dir = join(MODELS, slug);
  if (!statSync(dir).isDirectory()) continue;
  if (!existsSync(join(dir, "index.html"))) continue; // built route
  const { text } = readAll(dir);

  // multi-model page = a subpage that loads ≥2 model groups concurrently
  const multiModel = existsSync(join(dir, "multi-model")) || existsSync(join(dir, "multimodel"));

  let family = "unknown";
  for (const f of FAMILY_ORDER) {
    if (FAMILIES[f].test(text)) {
      family = f;
      break;
    }
  }

  // Does the route ADOPT the shared, component-rendering loader? createModelLoader (lib/model-loader.js)
  // and createResumableLoader (lib/resumable-loader.mjs, PaliGemma) both render <model-download-status>
  // and route progress through the adapters (Phase 3/4). Calling one of them IS the adoption.
  const usesResumable = /createResumableLoader\s*\(|resumable-loader/.test(text);
  const usesCentral = /createModelLoader\s*\(/.test(text);
  const adoption = usesResumable ? "resumable-loader" : (usesCentral ? "central-loader" : "bypass");

  // does it download at all? a route with NO loader signal AND no worker is non-applicable.
  const hasWorker = existsSync(join(dir, "worker.js"));
  let byteControl, resume, note, status;
  if (family === "unknown") {
    if (!hasWorker && !/createModelLoader|import\(.*transformers|\.onnx\b/.test(text)) {
      family = "non-applicable";
      byteControl = "none";
      resume = "non-applicable";
      note = "no model-asset download detected (bundled/tiny assets or no inference download)";
      status = "non-applicable";
    } else {
      byteControl = "unknown";
      resume = "unknown";
      note = "loader family could not be determined from the source — needs manual classification";
      status = "unknown";
    }
  } else {
    ({ byteControl, resume, note } = FAMILIES[family]);
    // refine raw-ort: resumable only when the weights go through the site's resumable downloader
    if (family === "raw-ort" && /model-download\.js|downloadModelFile/.test(text)) {
      resume = "resumable";
      note =
        "page fetches the .onnx via lib/model-download.js (Range/206/sha256) — genuinely resumable";
    }
    // Terminal adoption status: adopted (routes progress through the shared component-rendering loader) vs
    // blocked (downloads but bypasses it with a custom loader — a coverage gap the adoption gate fails on).
    status = adoption === "bypass" ? "blocked" : "adopted";
  }

  routes.push({
    slug,
    route: `models/${slug}/`,
    family,
    byteControl,
    resume,
    multiModel,
    hasWorker,
    adoption,
    status,
    note,
  });
}

const tally = (key) => routes.reduce((m, r) => ((m[r[key]] = (m[r[key]] || 0) + 1), m), {});
const inventory = {
  name: "web-ai-showcase model-download route inventory",
  description:
    "Every built demo route classified by download runtime/loader family, byte-transfer control, and honest resume capability. Foundation for the site-wide <model-download-status> component + its fail-closed adoption gate (Task 2b).",
  version: 1,
  generated: "2026-07-20",
  legend: {
    family: "which runtime/loader transfers the model assets",
    byteControl:
      "site-controlled (the page fetches bytes) | runtime-owned (a library fetches) | browser-owned | none",
    resume:
      "resumable (Range/206) | restart-only | runtime-owned | cached-only (whole-file cache) | non-applicable | unknown",
    status:
      "pending-adoption | adopted | honestly-limited | non-applicable | blocked | unknown (unknown FAILS the gate)",
  },
  totals: {
    routes: routes.length,
    downloadingRoutes: routes.filter((r) => r.status !== "non-applicable").length,
    byFamily: tally("family"),
    byResume: tally("resume"),
    byAdoption: tally("adoption"),
    byByteControl: tally("byteControl"),
    byStatus: tally("status"),
    multiModel: routes.filter((r) => r.multiModel).length,
  },
  routes,
};

const outIdx = process.argv.indexOf("--out");
const outPath = outIdx >= 0 ? process.argv[outIdx + 1] : "download-routes.json";
if (!process.argv.includes("--check")) {
  writeFileSync(outPath, JSON.stringify(inventory, null, 2) + "\n");
}
console.error(`routes: ${routes.length} · downloading: ${inventory.totals.downloadingRoutes}`);
console.error("byFamily:", JSON.stringify(inventory.totals.byFamily));
console.error("byResume:", JSON.stringify(inventory.totals.byResume));
const unknown = routes.filter((r) => r.family === "unknown");
console.error(
  `multiModel: ${inventory.totals.multiModel} · UNKNOWN (need manual classification): ${unknown.length}`,
);
if (unknown.length) console.error("  " + unknown.map((r) => r.slug).join(", "));
process.exit(process.argv.includes("--check") && unknown.length ? 1 : 0);
