// runtime-pin-parse-census.mjs — measured AST/HTML parse census for the
// runtime-pin currency gate (design/runtime-pin-ast-parser.md, bead
// web-ai-showcase-j6i). READ-ONLY: writes only the census JSON artifact.
//
// Usage: node scripts/runtime-pin-parse-census.mjs [--out inventory/runtime-pin-parse-census.json]
//
// The census is the COVERAGE BUDGET evidence: it proves the pinned parsers
// (acorn/parse5, see scripts/runtime-pin-parser.mjs) cover the current corpus,
// classifies EVERY raw/cooked marker occurrence into an exact executable
// context, and inventories every unsupported context — BEFORE the scanner
// integration (beads j9z/vs2) is coded.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { PIN_SCAN_TARGETS } from "./audit-model-currency.mjs";
import {
  RAW_MARKERS,
  ESCAPED_MARKER_VARIANTS,
  parseJavaScript,
  jsSyntaxSpans,
  classifyJsOffset,
  parseHtmlDocument,
  htmlScriptContexts,
  htmlOffsetInDisplayRegion,
  RuntimePinParseError,
} from "./runtime-pin-parser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const outArg = process.argv.indexOf("--out");
const OUT = outArg > -1 ? process.argv[outArg + 1] : "inventory/runtime-pin-parse-census.json";

function listFiles() {
  // git ls-files keeps the census exactly on tracked files under scan targets.
  const out = execFileSync("git", ["ls-files", ...PIN_SCAN_TARGETS.split(" ").filter(Boolean)], {
    cwd: ROOT,
    encoding: "utf8",
  });
  return out.split("\n").filter(Boolean);
}

function markerOffsets(source) {
  const hits = [];
  const variants = [
    ...RAW_MARKERS.map((m) => ({ needle: m, escaped: false })),
    ...ESCAPED_MARKER_VARIANTS.map((m) => ({ needle: m, escaped: true })),
  ];
  for (const { needle, escaped } of variants) {
    let idx = source.indexOf(needle);
    while (idx !== -1) {
      hits.push({ offset: idx, marker: needle, escaped });
      idx = source.indexOf(needle, idx + 1);
    }
  }
  // Cooked-escape discovery: the raw bytes of the package name with an escaped
  // "@", e.g. `onnxruntime-web\x40` or `onnxruntime-web%40`. These never form
  // a contiguous raw marker, so they must be counted separately (Gemini review
  // requirement: TOTAL marker accounting incl. escaped forms).
  for (const base of ["onnxruntime-web", "@huggingface/transformers", "@huggingface\\/transformers"]) {
    for (const esc of ["\\x40", "\\u0040", "%40"]) {
      const needle = base + esc;
      let idx = source.indexOf(needle);
      while (idx !== -1) {
        hits.push({ offset: idx, marker: needle, escaped: true, cookedEscape: true });
        idx = source.indexOf(needle, idx + 1);
      }
    }
  }
  hits.sort((a, b) => a.offset - b.offset);
  return hits;
}

function isBinary(source) {
  return source.includes("");
}

function censusJs(relPath, source) {
  const file = { path: relPath, kind: "js", markers: [], parse: null };
  const hits = markerOffsets(source);
  if (isBinary(source)) {
    file.binary = true; // legacy binary-NUL fail-closed pass owns this file
  }
  let parsed;
  try {
    parsed = parseJavaScript(source, relPath);
    file.parse = { ok: true, sourceType: parsed.sourceType };
  } catch (err) {
    file.parse = {
      ok: false,
      error: err.message,
      // FAIL CLOSED policy check: a parse failure in a file WITH markers or
      // runtime sinks is a census ERROR, never "inert".
      hasMarkers: hits.length > 0,
      hasRuntimeSink: /import\s*\(|wasmPaths|from\s*["']https?:/.test(source),
    };
    file.markers = hits.map((h) => ({ ...h, context: "unparsed" }));
    return file;
  }
  const spans = jsSyntaxSpans(source, parsed);
  file.markers = hits.map((h) => {
    const cls = classifyJsOffset(spans, h.offset);
    return { ...h, context: cls.kind };
  });
  return file;
}

function censusHtml(relPath, source) {
  const file = { path: relPath, kind: "html", markers: [], scriptContexts: [] };
  const hits = markerOffsets(source);
  const { document } = parseHtmlDocument(source, relPath);
  const contexts = htmlScriptContexts(document);
  file.scriptContexts = contexts.map((c) => ({ kind: c.kind, scriptType: c.scriptType }));
  file.markers = hits.map((h) => {
    // Inline executable script? Re-run the JS classifier on the script body.
    const inline = contexts.find(
      (c) => c.kind === "script-inline" && c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    if (inline) {
      try {
        const body = source.slice(inline.startOffset, inline.endOffset);
        const parsed = parseJavaScript(body, relPath + "#inline-script");
        const spans = jsSyntaxSpans(body, parsed);
        const cls = classifyJsOffset(spans, h.offset - inline.startOffset);
        return { ...h, context: `html-inline-js:${cls.kind}` };
      } catch (err) {
        return { ...h, context: "html-inline-js:unparsed", error: err.message };
      }
    }
    const src = contexts.find(
      (c) => c.kind === "script-src" && c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    if (src) return { ...h, context: "html-script-src" };
    if (htmlOffsetInDisplayRegion(document, h.offset)) {
      return { ...h, context: "html-display(pre/code)" };
    }
    return { ...h, context: "html-text" };
  });
  return file;
}

function censusData(relPath, source) {
  const hits = markerOffsets(source);
  return {
    path: relPath,
    kind: relPath.endsWith(".json") || relPath.endsWith(".ndjson") ? "json" : "other",
    binary: isBinary(source) || undefined,
    markers: hits.map((h) => ({ ...h, context: "non-executable-data" })),
  };
}

const files = listFiles();
const results = { js: [], html: [], data: [] };
let parsedJs = 0;
let failedJs = [];
for (const relPath of files) {
  const source = readFileSync(ROOT + relPath, "utf8");
  if (/\.(mjs|js)$/.test(relPath)) {
    const f = censusJs(relPath, source);
    results.js.push(f);
    if (f.parse.ok) parsedJs++;
    else failedJs.push({ path: relPath, error: f.parse.error, hasMarkers: f.parse.hasMarkers, hasRuntimeSink: f.parse.hasRuntimeSink });
  } else if (relPath.endsWith(".html")) {
    results.html.push(censusHtml(relPath, source));
  } else {
    results.data.push(censusData(relPath, source));
  }
}

const contextCounts = {};
const countMarker = (m) => {
  const key = m.cookedEscape ? `${m.context}|cooked-escape` : m.escaped ? `${m.context}|escaped` : m.context;
  contextCounts[key] = (contextCounts[key] || 0) + 1;
};
for (const group of [results.js, results.html, results.data]) {
  for (const f of group) for (const m of f.markers) countMarker(m);
}

const unsupportedContexts = [];
for (const f of results.html) {
  for (const c of f.scriptContexts) {
    if (c.kind.startsWith("unsupported")) {
      unsupportedContexts.push({ path: f.path, ...c });
    }
  }
}

const census = {
  schemaVersion: 1,
  bead: "web-ai-showcase-j6i",
  generatedBy: "scripts/runtime-pin-parse-census.mjs",
  parsers: {
    acorn: "8.19.0",
    "acorn-walk": "8.3.5",
    parse5: "8.0.1",
  },
  scanTargets: PIN_SCAN_TARGETS,
  totals: {
    filesScanned: files.length,
    jsFiles: results.js.length,
    jsParsedOk: parsedJs,
    jsParseFailures: failedJs.length,
    htmlFiles: results.html.length,
    dataFiles: results.data.length,
    filesWithMarkers:
      results.js.filter((f) => f.markers.length).length +
      results.html.filter((f) => f.markers.length).length +
      results.data.filter((f) => f.markers.length).length,
    markerOccurrences:
      results.js.reduce((n, f) => n + f.markers.length, 0) +
      results.html.reduce((n, f) => n + f.markers.length, 0) +
      results.data.reduce((n, f) => n + f.markers.length, 0),
  },
  markerContextCounts: contextCounts,
  jsParseFailuresDetail: failedJs,
  unsupportedContexts,
  files: [...results.js, ...results.html, ...results.data].filter((f) => f.markers.length > 0),
};

writeFileSync(ROOT + OUT, JSON.stringify(census, null, 2) + "\n");
console.log(`census written to ${OUT}`);
console.log(JSON.stringify(census.totals, null, 2));
console.log("marker contexts:", JSON.stringify(census.markerContextCounts, null, 2));
if (failedJs.length) {
  console.log("JS PARSE FAILURES:", JSON.stringify(failedJs, null, 2));
}
