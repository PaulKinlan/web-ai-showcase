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
//
// FAIL CLOSED (coord-authorized corrections after Codex review 125e1ac8):
//  1. Enumeration is a raw FILESYSTEM walk of PIN_SCAN_TARGETS, exactly the
//     set the gate's `grep -r` reads — gitignored and untracked files are
//     INCLUDED (git status is recorded as information only).
//  2. Every decoded HTML attribute marker occurrence is counted independently
//     — a raw pin in the same attribute never suppresses a second
//     entity-encoded pin.
//  3. An executable inline-script (or scanned JS file) that fails to parse
//     while bearing marker or runtime-sink evidence is FATAL: the census exits
//     non-zero and the artifact is stamped notVerified, never a green record.

import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { PIN_SCAN_TARGETS } from "./audit-model-currency.mjs";
import {
  RAW_MARKERS,
  markerOffsets,
  isBinarySource,
  parseJavaScript,
  jsSyntaxSpans,
  classifyJsOffset,
  parseHtmlDocument,
  htmlScriptContexts,
  htmlOffsetInDisplayRegion,
  RuntimePinParseError,
} from "./runtime-pin-parser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const RUNTIME_SINK_RE = /import\s*\(|wasmPaths|from\s*["']https?:/;

/**
 * Enumerate EXACTLY the files the gate's `grep -r` over PIN_SCAN_TARGETS can
 * read: every regular file under each directory target (recursive, no
 * gitignore filtering) plus each file target. Git tracked/ignored status is
 * recorded per file as INFORMATION (via git check-ignore / ls-files,
 * best-effort) so a divergence between the git view and the filesystem view
 * is visible, never silent.
 */
export function listFiles(rootDir, targets = PIN_SCAN_TARGETS) {
  const files = [];
  const walk = (abs, rel) => {
    for (const entry of readdirSync(abs)) {
      const childAbs = resolve(abs, entry);
      const childRel = rel + entry;
      const st = statSync(childAbs);
      if (st.isDirectory()) walk(childAbs, childRel + "/");
      else if (st.isFile()) files.push(childRel);
    }
  };
  for (const target of targets.split(" ").filter(Boolean)) {
    const abs = resolve(rootDir, target);
    if (!existsSync(abs)) continue;
    if (statSync(abs).isFile()) files.push(target); // bare file target
    else walk(abs, target.endsWith("/") ? target : target + "/");
  }
  const all = [...new Set(files)].sort();

  // Informational git classification (never used to EXCLUDE).
  let tracked = new Set();
  let ignored = new Set();
  try {
    tracked = new Set(
      execFileSync("git", ["ls-files"], { cwd: rootDir, encoding: "utf8" })
        .split("\n")
        .filter(Boolean),
    );
    if (all.length) {
      // check-ignore exits 1 when NOTHING is ignored — that is not an error.
      const res = spawnSync("git", ["check-ignore", "--stdin"], {
        cwd: rootDir,
        encoding: "utf8",
        input: all.join("\n") + "\n",
      });
      if (res.status === 0 || res.status === 1) {
        ignored = new Set((res.stdout || "").split("\n").filter(Boolean));
      }
    }
  } catch {
    // Not a git repo (e.g. unit-test fixture dir): classification unavailable.
  }
  const untracked = all.filter((f) => !tracked.has(f) && !ignored.has(f));
  const ignoredInScope = all.filter((f) => ignored.has(f));
  return {
    all,
    tracked: all.filter((f) => tracked.has(f)),
    untracked,
    ignoredInScope,
  };
}

export function censusJs(relPath, source, fatalErrors) {
  const file = { path: relPath, kind: "js", markers: [], parse: null };
  const hits = markerOffsets(source);
  if (isBinarySource(source)) {
    file.binary = true; // legacy binary-NUL fail-closed pass owns this file
  }
  let parsed;
  try {
    parsed = parseJavaScript(source, relPath);
    file.parse = { ok: true, sourceType: parsed.sourceType };
  } catch (err) {
    const hasMarkers = hits.length > 0;
    const hasRuntimeSink = RUNTIME_SINK_RE.test(source);
    file.parse = { ok: false, error: err.message, hasMarkers, hasRuntimeSink };
    // FAIL CLOSED: a parse failure in a file WITH markers or runtime sinks is
    // a census FATAL error, never "inert".
    if (hasMarkers || hasRuntimeSink) {
      fatalErrors.push({ path: relPath, kind: "js-parse", error: err.message, hasMarkers, hasRuntimeSink });
    }
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

export function censusHtml(relPath, source, fatalErrors) {
  const file = { path: relPath, kind: "html", markers: [], scriptContexts: [] };
  const hits = markerOffsets(source);
  const { document } = parseHtmlDocument(source, relPath);
  const contexts = htmlScriptContexts(document);
  file.scriptContexts = contexts.map((c) => ({
    kind: c.kind,
    scriptType: c.scriptType,
    attrName: c.attrName,
  }));

  // Decoded-value scan: parse5 entity-decodes attribute values. Count EVERY
  // decoded marker occurrence independently: a raw pin in the same attribute
  // explains exactly one decoded occurrence; any further decoded occurrences
  // (entity-encoded pins) are counted, never suppressed.
  const decodedHits = [];
  for (const c of contexts) {
    if (c.kind === "script-src" && typeof c.src === "string") {
      const rawInAttr = hits.filter(
        (h) => c.startOffset <= h.offset && h.offset < c.endOffset && !h.cookedEscape,
      ).length;
      let decodedCount = 0;
      for (const m of RAW_MARKERS) {
        let idx = c.src.indexOf(m);
        while (idx !== -1) {
          decodedCount++;
          idx = c.src.indexOf(m, idx + 1);
        }
      }
      for (let i = rawInAttr; i < decodedCount; i++) {
        decodedHits.push({
          offset: c.startOffset,
          marker: "(decoded attr value)",
          escaped: true,
          decodedAttrValue: true,
          context: "html-script-src(decoded)",
        });
      }
    }
  }

  // Parse EVERY executable inline script that bears marker or runtime-sink
  // evidence — marker-bearing for classification, sink-bearing for
  // validation. A parse failure of either is FATAL (fail closed).
  const inlineParse = new Map(); // context -> parsed spans or error
  for (const c of contexts) {
    if (c.kind !== "script-inline") continue;
    const body = source.slice(c.startOffset, c.endOffset);
    const bodyHasMarker = hits.some(
      (h) => c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    const bodyHasSink = RUNTIME_SINK_RE.test(body);
    if (!bodyHasMarker && !bodyHasSink) continue;
    try {
      const parsed = parseJavaScript(body, relPath + "#inline-script");
      inlineParse.set(c, { spans: jsSyntaxSpans(body, parsed) });
    } catch (err) {
      inlineParse.set(c, { error: err.message });
      fatalErrors.push({
        path: relPath,
        kind: "html-inline-js-parse",
        error: err.message,
        hasMarkers: bodyHasMarker,
        hasRuntimeSink: bodyHasSink,
      });
    }
  }

  file.markers = hits.map((h) => {
    // Inline executable script? Use the parsed body when available.
    const inline = contexts.find(
      (c) => c.kind === "script-inline" && c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    if (inline) {
      const res = inlineParse.get(inline);
      if (res && res.spans) {
        const body = source.slice(inline.startOffset, inline.endOffset);
        const cls = classifyJsOffset(res.spans, h.offset - inline.startOffset);
        return { ...h, context: `html-inline-js:${cls.kind}` };
      }
      return { ...h, context: "html-inline-js:unparsed" };
    }
    const src = contexts.find(
      (c) => c.kind === "script-src" && c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    if (src) return { ...h, context: "html-script-src" };
    // Marker inside a non-executable script body (importmap, JSON, unknown
    // type): link it to that context instead of losing it as html-text.
    const nonexec = contexts.find(
      (c) => c.kind === "script-nonexec" && c.startOffset <= h.offset && h.offset < c.endOffset,
    );
    if (nonexec) return { ...h, context: `html-script-nonexec(${nonexec.scriptType})` };
    if (htmlOffsetInDisplayRegion(document, h.offset)) {
      return { ...h, context: "html-display(pre/code)" };
    }
    return { ...h, context: "html-text" };
  });
  file.markers.push(...decodedHits);
  return file;
}

export function censusData(relPath, source) {
  const hits = markerOffsets(source);
  return {
    path: relPath,
    kind: relPath.endsWith(".json") || relPath.endsWith(".ndjson") ? "json" : "other",
    binary: isBinarySource(source) || undefined,
    markers: hits.map((h) => ({ ...h, context: "non-executable-data" })),
  };
}

export function runCensus(rootDir, outRel = "inventory/runtime-pin-parse-census.json") {
  const fatalErrors = [];
  const { all: files, tracked, untracked, ignoredInScope } = listFiles(rootDir);
  const results = { js: [], html: [], data: [] };
  let parsedJs = 0;
  const failedJs = [];
  for (const relPath of files) {
    const source = readFileSync(resolve(rootDir, relPath), "utf8");
    if (/\.(mjs|js)$/.test(relPath)) {
      const f = censusJs(relPath, source, fatalErrors);
      results.js.push(f);
      if (f.parse.ok) parsedJs++;
      else failedJs.push({ path: relPath, error: f.parse.error, hasMarkers: f.parse.hasMarkers, hasRuntimeSink: f.parse.hasRuntimeSink });
    } else if (relPath.endsWith(".html")) {
      results.html.push(censusHtml(relPath, source, fatalErrors));
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
    schemaVersion: 2,
    bead: "web-ai-showcase-j6i",
    generatedBy: "scripts/runtime-pin-parse-census.mjs",
    notVerified: fatalErrors.length > 0 || undefined,
    parsers: {
      acorn: "8.19.0",
      "acorn-walk": "8.3.5",
      parse5: "8.0.1",
    },
    scanTargets: PIN_SCAN_TARGETS,
    enumeration: "filesystem walk of PIN_SCAN_TARGETS (matches the gate's grep -r; gitignored/untracked files INCLUDED; git status informational only)",
    totals: {
      filesScanned: files.length,
      trackedFiles: tracked.length,
      untrackedFiles: untracked.length,
      untrackedFileList: untracked,
      gitignoredFilesInScope: ignoredInScope.length,
      gitignoredFileList: ignoredInScope,
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
    fatalErrors,
    jsParseFailuresDetail: failedJs,
    unsupportedContexts,
    files: [...results.js, ...results.html, ...results.data].filter((f) => f.markers.length > 0),
  };

  if (outRel) {
    writeFileSync(resolve(rootDir, outRel), JSON.stringify(census, null, 2) + "\n");
  }
  return census;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === resolve(new URL(import.meta.url).pathname);
if (isMain) {
  const outArg = process.argv.indexOf("--out");
  const OUT = outArg > -1 ? process.argv[outArg + 1] : "inventory/runtime-pin-parse-census.json";
  const census = runCensus(ROOT, OUT);
  console.log(`census written to ${OUT}`);
  console.log(JSON.stringify(census.totals, null, 2));
  console.log("marker contexts:", JSON.stringify(census.markerContextCounts, null, 2));
  if (census.jsParseFailuresDetail.length) {
    console.log("JS PARSE FAILURES:", JSON.stringify(census.jsParseFailuresDetail, null, 2));
  }
  if (census.fatalErrors.length) {
    console.error("FATAL (fail-closed):", JSON.stringify(census.fatalErrors, null, 2));
    process.exit(1);
  }
}
