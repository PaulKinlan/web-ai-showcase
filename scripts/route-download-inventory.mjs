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
//        --out <path>        write the inventory elsewhere
//        --evidence <path>   write diagnostics: per route the matched rule/token/file, and any source
//                           that had to be treated as ambiguous (comments could not be masked safely)
//        --models <dir>      classify a different tree (fixture tests; the gate uses models/)
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import * as acorn from "acorn";
import * as parse5 from "parse5";

const MODELS = "models";
// --models <dir> lets the fixture test drive a synthetic tree; the gate always uses the real one.
const modelsDir = process.argv.includes("--models")
  ? process.argv[process.argv.indexOf("--models") + 1]
  : MODELS;
const routes = [];
const evidence = { routes: {} };

// signal → family. Ordered by SPECIFICITY (first match wins); each family carries its honest
// byte-control + resume semantics.
const FAMILIES = {
  "transformers-resumable-prefetch": {
    // The site prefetches big weights resumably (Range/206/sha256 → cache) before from_pretrained.
    pattern: /model-prefetch\.mjs|prefetchModel\s*\(/,
    byteControl: "site-controlled",
    resume: "resumable",
    note:
      "site fetches weights resumably (Range/206/sha256) into transformers-cache; from_pretrained then cache-hits",
  },
  "webllm": {
    pattern: /lib\/webllm\.js|CreateMLCEngine|@mlc-ai|web-llm/,
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note: "WebLLM/MLC engine fetches + caches shards; no per-file resume hooks exposed to the page",
  },
  "mediapipe": {
    pattern:
      // The bundle-name alternatives are ANCHORED to a quoted filename. A bare /\.task\b/ also matches
      // ordinary JavaScript property access (`${p.task}`, `row.task.id`), which classified any route that
      // reads a `.task` property as MediaPipe. Note readAll() concatenates only .js/.mjs/.html, so a
      // FilesetResolver mention that lives in a route's conformance.json is not visible here either.
      /lib\/mediapipe\.js|FilesetResolver|tasks-vision|tasks-genai|["'][^"']*\.task["']|["'][^"']*\.tflite["']/,
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note:
      "MediaPipe Tasks fetches the .task/.tflite bundle; download managed inside the WASM runtime",
  },
  "raw-ort": {
    pattern: /onnxruntime-web|InferenceSession|\bort\.(InferenceSession|env)/,
    byteControl: "site-controlled",
    // refined below: resumable only if it goes through lib/model-download.js
    resume: "restart-only",
    note:
      "page fetches the .onnx bytes itself and hands them to ORT; restartable unless routed via model-download.js",
  },
  "transformers-wrapped": {
    // Third-party libs (outetts, kokoro-js) that wrap Transformers.js and download through it.
    pattern: /outetts@|kokoro-js[@/]|npm\/(outetts|kokoro-js)/,
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "third-party library (outetts / kokoro-js) wraps Transformers.js; it downloads + caches via TJS, so per-file progress depends on the wrapper surfacing it",
  },
  "transformers-from_pretrained": {
    pattern:
      /from_pretrained\s*\(|AutoModel|AutoProcessor|AutoTokenizer|ForConditionalGeneration|\.from_pretrained/,
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "Transformers.js fetches each file (no Range); re-running resumes at whole-file granularity from Cache Storage",
  },
  "transformers-pipeline": {
    pattern: /lib\/webai\.js|loadPipeline\s*\(|[^a-zA-Z]pipeline\s*\(/,
    byteControl: "runtime-owned",
    resume: "cached-only",
    note: "Transformers.js pipeline() fetches each file (no Range); whole-file cache resume only",
  },
  "browser-builtin": {
    pattern:
      /window\.(ai|LanguageModel|Summarizer|Translator|Rewriter|Writer)\b|\bLanguageModel\.(create|availability)|ai\.languageModel/,
    byteControl: "browser-owned",
    resume: "runtime-owned",
    note:
      "the browser owns the model + its download state; adapt as a capability signal, never fabricate per-file detail",
  },
};
const FAMILY_ORDER = Object.keys(FAMILIES);

// ---------------------------------------------------------------- comment masking
// The signals below are API mentions, so a comment that merely NAMES another runtime's API must not be
// able to change a route's public download classification. Measured (web-ai-showcase-x4ba): a comment
// containing the token "ort.env" flipped ai-image-detection from transformers-pipeline to raw-ort -
// claiming sitely-controlled bytes and restart-only resume for a route that downloads through
// loadPipeline. So comments are masked to spaces before any matching: same length, same offsets,
// every executable token untouched.
//
// Masking is LEXICAL (acorn tokenizer), not parse-based: comment spans are all we need, no AST is
// required, and the corpus lexes cleanly this way (781 model JS/MJS files, 0 lexical failures).
// A source that cannot be lexed safely - an unterminated comment or string - cannot be trusted, so it
// is reported as AMBIGUOUS and fails the gate closed instead of being classified from raw text.
const JS_MIME_ESSENCES = new Set([
  "", "module", "text/javascript", "application/javascript", "text/ecmascript", "application/ecmascript",
  "application/x-javascript", "text/x-javascript", "text/jscript", "text/livescript",
  "text/javascript1.0", "text/javascript1.1", "text/javascript1.2", "text/javascript1.3",
  "text/javascript1.4", "text/javascript1.5",
]);

function maskRanges(source, ranges) {
  if (ranges.length === 0) return source;
  const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
  let out = "";
  let pos = 0;
  for (const [start, end] of sorted) {
    if (start < pos || end <= start) continue; // overlapping or empty span: already masked
    out += source.slice(pos, start) + " ".repeat(end - start);
    pos = end;
  }
  return out + source.slice(pos);
}

/** Comment spans of a JS source, or null when it cannot be lexed safely. */
function jsCommentRanges(source) {
  for (const sourceType of ["module", "script"]) {
    const ranges = [];
    try {
      const tokenizer = acorn.tokenizer(source, {
        ecmaVersion: "latest",
        sourceType,
        allowHashBang: true,
        allowReturnOutsideFunction: true,
        allowAwaitOutsideFunction: true,
        onComment: (block, text, start, end) => ranges.push([start, end]),
      });
      while (tokenizer.getToken().type.label !== "eof") { /* drain: onComment fires as it advances */ }
      return ranges;
    } catch { /* try the other source type, then give up */ }
  }
  return null;
}

/**
 * Comment spans of a CSS stylesheet - the block comment form is the only comment syntax CSS has -
 * or null when it cannot be scanned safely (unterminated comment or string). Strings are skipped so an
 * opener inside one stays data. A <style> block is not executable either way, but a comment in it
 * naming another runtime's API must not be able to classify the route, the same rule as everywhere else.
 */
function cssCommentRanges(source) {
  const ranges = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (ch === "/" && source[i + 1] === "*") {
      const end = source.indexOf("*/", i + 2);
      if (end === -1) return null;
      ranges.push([i, end + 2]);
      i = end + 2;
      continue;
    }
    if (ch === '"' || ch === "'") {
      let j = i + 1;
      while (j < source.length && source[j] !== ch) j += source[j] === "\\" ? 2 : 1;
      if (j >= source.length) return null;
      i = j + 1;
      continue;
    }
    i++;
  }
  return ranges;
}

/** Mask HTML comments and the JS comments inside inline <script> blocks. */
function htmlCommentRanges(source) {
  const document = parse5.parse(source, { sourceCodeLocationInfo: true });
  const ranges = [];
  let ambiguous = null;
  const visit = (node) => {
    if (node.nodeName === "#comment") {
      const loc = node.sourceCodeLocation;
      if (!loc || loc.startOffset === undefined) ambiguous ??= "comment node without a source location";
      else ranges.push([loc.startOffset, loc.endOffset]);
    }
    if (node.tagName === "script") {
      const text = (node.childNodes || []).find((c) => c.nodeName === "#text");
      const loc = text?.sourceCodeLocation;
      if (text?.value && loc && loc.startOffset !== undefined && loc.endOffset !== undefined) {
        const typeAttr = (node.attrs || []).find((a) => a.name.toLowerCase() === "type");
        const essence = (typeAttr?.value ?? "").split(";")[0].trim().toLowerCase();
        if (JS_MIME_ESSENCES.has(essence)) {
          // Tokenize the RAW slice of the source, never text.value: parse5 normalises CRLF to LF in
          // value, so value-relative offsets drift from the source by one per preceding CRLF and a
          // short trailing "// ort.env" comment escapes masking entirely (review finding on x4ba).
          const raw = source.slice(loc.startOffset, loc.endOffset);
          const inner = jsCommentRanges(raw);
          if (inner === null) ambiguous ??= "inline <script> could not be lexed";
          else for (const [start, end] of inner) ranges.push([loc.startOffset + start, loc.startOffset + end]);
        }
      }
    }
    if (node.tagName === "noscript") {
      // Under scripting-enabled parsing (parse5's default, and how these JS demos run) <noscript>
      // content is inert RAWTEXT: parse5 builds a #text node, not a comment node, so nothing in it
      // executes and an "<!-- ort.env -->" written there is literal text. Mask the whole element so
      // no-JS fallback prose cannot classify a route (measured: it used to leak as raw-ort).
      const nsLoc = node.sourceCodeLocation;
      if (nsLoc && nsLoc.startOffset !== undefined && nsLoc.endOffset !== undefined) {
        ranges.push([nsLoc.startOffset, nsLoc.endOffset]);
      } else {
        ambiguous ??= "<noscript> without a source location";
      }
    }
    if (node.tagName === "style") {
      const cssText = (node.childNodes || []).find((c) => c.nodeName === "#text");
      const cssLoc = cssText?.sourceCodeLocation;
      if (cssText?.value && cssLoc && cssLoc.startOffset !== undefined && cssLoc.endOffset !== undefined) {
        // Scan the RAW slice for the same reason as inline scripts: parse5 normalises CRLF in value.
        const inner = cssCommentRanges(source.slice(cssLoc.startOffset, cssLoc.endOffset));
        if (inner === null) ambiguous ??= "<style> could not be scanned for comments";
        else for (const [start, end] of inner) ranges.push([cssLoc.startOffset + start, cssLoc.startOffset + end]);
      }
    }
    for (const child of node.childNodes || []) visit(child);
    // <template> descendants live in the node's `content` fragment, NOT in childNodes (review finding
    // on web-ai-showcase-x4ba): without this, an inert comment inside a template stays in the matchable
    // text and can still flip a route's family - reproduced with
    // <template><!-- ort.env --></template> alongside a real loadPipeline script.
    if (node.content) visit(node.content);
  };
  visit(document);
  return { ranges, ambiguous };
}

function maskSourceComments(source, file) {
  if (/\.(js|mjs)$/.test(file)) {
    const ranges = jsCommentRanges(source);
    if (ranges === null) return { text: null, ambiguous: "not lexable (unterminated comment or string?)" };
    return { text: maskRanges(source, ranges), ambiguous: null };
  }
  const { ranges, ambiguous } = htmlCommentRanges(source);
  return { text: maskRanges(source, ranges), ambiguous };
}

function readAll(dir) {
  // The demo's own worker + page scripts + index.html (the download entrypoints live here), with
  // comments masked. Entries stay per file so a match can name the token AND the file behind it.
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
  const entries = [];
  const ambiguous = [];
  for (const f of files) {
    const masked = maskSourceComments(readFileSync(f, "utf8"), f);
    if (masked.ambiguous) ambiguous.push(`${f}: ${masked.ambiguous}`);
    entries.push({ file: f, text: masked.text ?? "" });
  }
  return { text: entries.map((e) => "\n" + e.text).join(""), files, entries, ambiguous };
}

for (const slug of readdirSync(modelsDir).sort()) {
  const dir = join(modelsDir, slug);
  if (!statSync(dir).isDirectory()) continue;
  if (!existsSync(join(dir, "index.html"))) continue; // built route
  const { text, entries, ambiguous } = readAll(dir);

  // multi-model page = a subpage that loads ≥2 model groups concurrently
  const multiModel = existsSync(join(dir, "multi-model")) || existsSync(join(dir, "multimodel"));

  let family = "unknown";
  let familyMatch = null;
  // First family in FAMILY_ORDER that matches ANY of the demo's files, and the exact token that did it
  // - the token is what the gate prints when a route's committed classification goes stale.
  for (const f of FAMILY_ORDER) {
    for (const entry of entries) {
      const m = FAMILIES[f].pattern.exec(entry.text);
      if (m) {
        family = f;
        familyMatch = { rule: f, token: m[0].trim(), file: entry.file };
        break;
      }
    }
    if (familyMatch) break;
  }

  // Does the route ADOPT the shared, component-rendering loader? createModelLoader (lib/model-loader.js)
  // and createResumableLoader (lib/resumable-loader.mjs, PaliGemma) both render <model-download-status>
  // and route progress through the adapters (Phase 3/4). Calling one of them IS the adoption.
  const resumableMatch = /createResumableLoader\s*\(|resumable-loader/.exec(text);
  const centralMatch = /createModelLoader\s*\(/.exec(text);
  const usesResumable = resumableMatch !== null;
  const usesCentral = centralMatch !== null;
  const adoption = usesResumable ? "resumable-loader" : (usesCentral ? "central-loader" : "bypass");
  const adoptionMatch = resumableMatch ?? centralMatch;

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
    // A MediaPipe page that fetches bytes itself and supplies modelAssetBuffer owns the transfer.
    // Unless it uses the shared resumable downloader, interruption restarts from byte zero.
    if (
      family === "mediapipe" && /modelAssetBuffer/.test(text) &&
      /fetch\s*\(/.test(text)
    ) {
      byteControl = "site-controlled";
      resume = /model-download\.js|downloadModelFile/.test(text) ? "resumable" : "restart-only";
      note = resume === "resumable"
        ? "site fetches the MediaPipe artifact via lib/model-download.js (Range/206/sha256) before supplying modelAssetBuffer"
        : "site fetches and verifies the complete MediaPipe artifact before supplying modelAssetBuffer; interruption restarts from byte zero";
    }
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
  // Diagnostics only (written to the --evidence side file, never into download-routes.json): what
  // matched, where, and whether any source had to be treated as ambiguous.
  evidence.routes[slug] = {
    family: familyMatch,
    adoption: adoptionMatch ? { rule: adoption, token: adoptionMatch[0] } : null,
    ambiguous,
  };
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
const evIdx = process.argv.indexOf("--evidence");
if (evIdx >= 0) writeFileSync(process.argv[evIdx + 1], JSON.stringify(evidence, null, 2) + "\n");
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
