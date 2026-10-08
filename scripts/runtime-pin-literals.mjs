// runtime-pin-literals.mjs — ADDITIVE whole-raw literal runtime-pin check for the
// currency gate (bead web-ai-showcase-j9z; design:
// design/runtime-pin-literal-scanner.md + design/runtime-pin-ast-parser.md).
//
// All JS/HTML syntax knowledge comes from the single parser entry point
// (scripts/runtime-pin-parser.mjs, landed in j6i). This module adds two
// INDEPENDENT nets, so a weakness in one still trips the other:
//
//   1. COMPLETE URL LITERAL judgment: every executable string/template-quasi/
//      script-src/inline-script literal bearing a package marker has its WHOLE
//      RAW specifier sliced from the original source (after `package@` to the
//      next real `/` or the end of the URL token — never truncated, never
//      decoded) and judged exactly against the allowlist (route-scoped for
//      transformers, global set for onnxruntime-web, derived measured-version
//      exemption preserved). Floating (`latest`), suffixed (`1.21.0~`),
//      encoded or nested-template pins all fail NAMING the full candidate.
//
//   2. GOLDEN LEDGER reconciliation: EVERY nonnumeric/escaped/cooked marker
//      occurrence — including ones also judged by net 1 — must match the
//      reviewed ledger (inventory/runtime-pin-marker-ledger.json) exactly:
//      path, full-line SHA-256, marker, context, ordinalOnLine,
//      expectedCountOnLine. Discovery NEVER auto-approves: any addition,
//      change or deletion is drift and fails.
//
//   3. FAIL CLOSED: markers in bare code, unparseable files (with marker/sink
//      evidence), cooked escapes, unsupported HTML contexts (event handlers,
//      javascript: URLs, non-exec script bodies) and NUL-bearing files are
//      errors, never silently skipped.
//
// The legacy numeric grep scan, strict-raw suffix rules, binary-NUL pass,
// derived-inventory exemption and scan floors are UNCHANGED; the literal
// counters emitted here are additive and cannot mask legacy counts.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync, lstatSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  markerOffsets,
  isBinarySource,
  parseJavaScript,
  jsSyntaxSpans,
  classifyJsOffset,
  parseHtmlDocument,
  htmlScriptContexts,
  htmlOffsetInDisplayRegion,
} from "./runtime-pin-parser.mjs";

const LEDGER_PATH = "inventory/runtime-pin-marker-ledger.json";
const CDN_PREFIX = "https://cdn.jsdelivr.net/npm/";
const RUNTIME_SINK_RE = /import\s*\(|wasmPaths|from\s*["']https?:/;

const sha256 = (s) => createHash("sha256").update(s, "utf8").digest("hex");

/**
 * Enumerate exactly what the legacy gate's `grep -r` over PIN_SCAN_TARGETS can
 * read — with ONE deliberate contract point, reviewed under j6i: grep -r does
 * not follow symlinked directories, so this walk uses lstat and SKIPS
 * symlinks (the j6i census's statSync walk follows them). The corpus has zero
 * symlinks under the targets today; introducing one is a reviewed
 * enumeration-contract amendment, never a silent divergence.
 */
export function listGateFiles(rootDir, targets) {
  const files = [];
  const walk = (abs, rel) => {
    for (const entry of readdirSync(abs)) {
      const childAbs = join(abs, entry);
      const childRel = rel + entry;
      const st = lstatSync(childAbs);
      if (st.isSymbolicLink()) continue; // grep -r contract: do not follow
      if (st.isDirectory()) walk(childAbs, childRel + "/");
      else if (st.isFile()) files.push(childRel);
    }
  };
  for (const target of targets.split(" ").filter(Boolean)) {
    const abs = join(rootDir, target);
    if (!existsSync(abs)) continue;
    const st = lstatSync(abs);
    if (st.isSymbolicLink()) continue;
    if (st.isFile()) files.push(target);
    else if (st.isDirectory()) walk(abs, target.endsWith("/") ? target : target + "/");
  }
  return [...new Set(files)].sort();
}

function lineInfo(source, offset) {
  const lineStart = source.lastIndexOf("\n", offset - 1) + 1;
  let lineEnd = source.indexOf("\n", offset);
  if (lineEnd === -1) lineEnd = source.length;
  const line = source.slice(0, lineStart).split("\n").length;
  return { line, lineText: source.slice(lineStart, lineEnd) };
}

/**
 * Extract the whole raw version candidate for a marker inside an executable
 * literal. Returns one of:
 *   { kind: "candidate", candidate }  — complete URL literal; judge exactly
 *   { kind: "dynamic" }               — a `${` expression follows the marker
 *   { kind: "not-url" }               — marker not in a complete CDN URL literal
 * boundaryEnd limits the scan (template quasi end, attr region end, …).
 */
export function extractRawCandidate(source, markerOffset, marker, boundaryEnd) {
  const markerEnd = markerOffset + marker.length;
  // Validate the approved CDN origin immediately before the marker (the marker
  // itself carries the full [scoped] package path).
  const prefixStart = markerOffset - CDN_PREFIX.length;
  if (prefixStart < 0 || source.slice(prefixStart, markerOffset) !== CDN_PREFIX) {
    return { kind: "not-url" };
  }
  let i = markerEnd;
  while (i < boundaryEnd) {
    const ch = source[i];
    if (ch === "/") break; // real path separator ends the candidate
    if (ch === "$" && source[i + 1] === "{") return { kind: "dynamic" };
    if (/[\s"'`<>]/.test(ch)) break; // end of the URL token
    i++;
  }
  if (i === markerEnd) return { kind: "dynamic" }; // empty candidate: `"…@" + v` shape
  return { kind: "candidate", candidate: source.slice(markerEnd, i) };
}

const NUMERIC_PREFIX_RE = /^[0-9]+\.[0-9]+\.[0-9]+/;

// Map the parser's mechanical contexts onto the amended ledger vocabulary.
function ledgerContext(cls, hit) {
  if (hit.cookedEscape) return "cooked-escape";
  switch (cls) {
    case "comment-line": return "line-comment";
    case "comment-block": return "block-comment";
    case "regex": return hit.escaped ? "regex-escaped" : "regex";
    case "string": return hit.escaped ? "string-escaped" : "string";
    case "template-quasi": return "template-dynamic";
    case "non-executable-data": return "json";
    case "html-display(pre/code)": return "html-display";
    default: return cls;
  }
}

/**
 * The additive literal check. `allowlist` is the parsed, structurally
 * validated scripts/runtime-pin-allowlist.json; `isDerivedInventoryHit(file,
 * lineNo, lib, rawCandidate)` is the caller's derived-exemption closure.
 * Returns { errors, counters } — counters are additive, never masking the
 * legacy scannedCounts floors.
 */
export function checkLiteralRuntimePins(rootDir, scanTargets, allowlist, isDerivedInventoryHit) {
  const errors = [];
  const counters = { literalUrlChecked: { onnxruntimeWeb: 0, transformers: 0 }, ledgeredMarkers: 0 };

  // --- load the reviewed golden ledger ---------------------------------------
  let ledger;
  try {
    ledger = JSON.parse(readFileSync(join(rootDir, LEDGER_PATH), "utf8"));
  } catch (e) {
    return { errors: [`runtime marker golden ledger unavailable: ${LEDGER_PATH}: ${e.message}`], counters };
  }
  if (!Array.isArray(ledger.entries) || !Array.isArray(ledger.splitAssemblies)) {
    return { errors: [`runtime marker golden ledger drift: ${LEDGER_PATH} missing entries/splitAssemblies`], counters };
  }

  const allowedOrt = new Set((allowlist.onnxruntimeWeb?.allowedVersions || []).map((v) => v.version));
  const allowedTjsShared = allowlist.transformers?.shared;
  const tjsOverrideMap = new Map();
  for (const o of allowlist.transformers?.allowedLocalOverrides || []) {
    tjsOverrideMap.set(o.version, new Set(o.slugs));
  }

  const libOf = (marker) =>
    marker.startsWith("onnxruntime-web") ? "onnxruntime-web" : "@huggingface/transformers";

  const judgeCandidate = (file, lineNo, lib, candidate) => {
    if (isDerivedInventoryHit(file, String(lineNo), lib, candidate)) return;
    if (lib === "onnxruntime-web") {
      counters.literalUrlChecked.onnxruntimeWeb++;
      if (!allowedOrt.has(candidate)) {
        errors.push(
          `unauthorized onnxruntime-web version "${candidate}" — not in scripts/runtime-pin-allowlist.json`,
        );
      }
    } else {
      counters.literalUrlChecked.transformers++;
      if (candidate === allowedTjsShared) return;
      const allowedSlugs = tjsOverrideMap.get(candidate);
      if (!allowedSlugs) {
        errors.push(
          `unauthorized @huggingface/transformers version "${candidate}" in ${file} — not in scripts/runtime-pin-allowlist.json`,
        );
        return;
      }
      const slugMatch = file.match(/^models\/([^/]+)\//);
      const slug = slugMatch ? slugMatch[1] : null;
      if (!slug || !allowedSlugs.has(slug)) {
        errors.push(
          `unauthorized @huggingface/transformers override "${candidate}" in ${file} — route "${slug || file}" is not authorized in scripts/runtime-pin-allowlist.json`,
        );
      }
    }
  };

  // Net 2: ledger candidates. EVERY nonnumeric/escaped/cooked marker
  // occurrence lands here, whether or not net 1 also judged it. Numeric
  // non-escaped hits belong to the legacy strict-raw grep scan alone.
  const ledgerCandidates = [];
  const ledgerOrLegacy = (file, source, hit, cls) => {
    const after = source.slice(hit.offset + hit.marker.length);
    if (!hit.escaped && !hit.cookedEscape && NUMERIC_PREFIX_RE.test(after)) return;
    const { line, lineText } = lineInfo(source, hit.offset);
    ledgerCandidates.push({
      path: file,
      line,
      ordinalOnLine: null, // assigned per-line below
      expectedCountOnLine: markerOffsets(lineText).length,
      marker: hit.marker,
      context: ledgerContext(cls, hit),
      sha256: sha256(lineText),
      _offset: hit.offset,
    });
  };

  const files = listGateFiles(rootDir, scanTargets);
  for (const file of files) {
    let source;
    try {
      source = readFileSync(join(rootDir, file), "utf8");
    } catch {
      continue; // unreadable (e.g. dangling symlink): grep -r skips it too
    }
    const hits = markerOffsets(source);
    const isJs = /\.(mjs|js)$/.test(file);
    const isHtml = file.endsWith(".html");
    if (hits.length === 0) {
      // Fail-closed parseability proof still applies to sink-bearing JS/HTML
      // even with no marker (design/runtime-pin-ast-parser.md §3).
      if ((isJs || isHtml) && RUNTIME_SINK_RE.test(source)) {
        proveParseable(file, source, isHtml, errors);
      }
      continue;
    }
    if (isBinarySource(source)) {
      errors.push(
        `runtime pin in binary-classified file ${file} (literal pass: NUL-bearing file with raw package marker; grep -I skips these) — fail-closed per web-ai-showcase-5s4`,
      );
      continue;
    }
    if (isJs) classifyJsFile(file, source, hits, errors, judgeCandidate, ledgerOrLegacy, libOf);
    else if (isHtml) classifyHtmlFile(file, source, hits, errors, judgeCandidate, ledgerOrLegacy, libOf);
    else for (const hit of hits) ledgerOrLegacy(file, source, hit, "non-executable-data");
  }

  // Per-line ordinals among ledger-candidate markers on the same line.
  const byLine = new Map();
  for (const c of ledgerCandidates) {
    const key = `${c.path}:${c.line}`;
    if (!byLine.has(key)) byLine.set(key, []);
    byLine.get(key).push(c);
  }
  for (const group of byLine.values()) {
    group.sort((a, b) => a._offset - b._offset);
    group.forEach((c, idx) => {
      c.ordinalOnLine = idx + 1;
      delete c._offset;
    });
  }

  // --- strict ledger reconciliation ------------------------------------------
  const key = (e) =>
    [e.path, e.sha256, e.marker, e.context, e.ordinalOnLine, e.expectedCountOnLine].join("|");
  const want = new Map(ledger.entries.map((e) => [key(e), e]));
  const got = new Map(ledgerCandidates.map((e) => [key(e), e]));
  for (const [k, e] of got) {
    counters.ledgeredMarkers++;
    if (!want.has(k)) {
      errors.push(
        `runtime marker golden ledger drift: unreviewed marker at ${e.path}:${e.line} (${e.context}, ${e.marker}) — discovery never auto-approves; a human-reviewed ledger amendment is required`,
      );
    }
  }
  for (const [k, e] of want) {
    if (!got.has(k)) {
      errors.push(
        `runtime marker golden ledger drift: reviewed ledger entry missing from source: ${e.path} (${e.context}, ${e.marker}, line SHA ${String(e.sha256).slice(0, 12)}…)`,
      );
    }
  }
  // Split-marker assemblies: pinned by exact path + both full-line SHA-256.
  for (const asm of ledger.splitAssemblies) {
    let src;
    try {
      src = readFileSync(join(rootDir, asm.path), "utf8");
    } catch {
      errors.push(`runtime marker golden ledger drift: split-assembly file ${asm.path} unreadable`);
      continue;
    }
    const lines = src.split("\n");
    for (const frag of asm.fragments) {
      if (sha256(lines[frag.line - 1] ?? "") !== frag.sha256) {
        errors.push(
          `runtime marker golden ledger drift: split-marker assembly ${asm.path}:${frag.line} changed — the ${asm.disposition} pair is pinned by exact line fingerprints`,
        );
      }
    }
  }

  return { errors, counters, ledgerCandidates };
}

function proveParseable(file, source, isHtml, errors) {
  try {
    if (isHtml) {
      const { document } = parseHtmlDocument(source, file);
      for (const c of htmlScriptContexts(document)) {
        if (c.kind === "script-inline" && RUNTIME_SINK_RE.test(c.text || "")) {
          parseJavaScript(c.text, `${file}#inline-script`);
        }
      }
    } else {
      parseJavaScript(source, file);
    }
  } catch (e) {
    errors.push(`unparseable executable source with runtime-sink evidence: ${e.message}`);
  }
}

function classifyJsFile(file, source, hits, errors, judgeCandidate, ledgerOrLegacy, libOf) {
  let parsed;
  try {
    parsed = parseJavaScript(source, file);
  } catch (e) {
    errors.push(`unparseable JS with marker evidence: ${e.message}`);
    for (const hit of hits) ledgerOrLegacy(file, source, hit, "unparsed");
    return;
  }
  const spans = jsSyntaxSpans(source, parsed);
  for (const hit of hits) {
    const cls = classifyJsOffset(spans, hit.offset);
    ledgerOrLegacy(file, source, hit, cls.kind); // net 2 first: ledger sees everything nonnumeric
    if (cls.kind === "string" || cls.kind === "template-quasi") {
      const boundaryEnd =
        cls.kind === "template-quasi"
          ? cls.span.quasis.find((q) => q.start <= hit.offset && hit.offset < q.end)?.end ?? cls.span.end
          : cls.span.end;
      const ex = extractRawCandidate(source, hit.offset, hit.marker, boundaryEnd);
      if (ex.kind === "candidate") {
        const { line } = lineInfo(source, hit.offset);
        judgeCandidate(file, line, libOf(hit.marker), ex.candidate);
      }
    } else if (
      cls.kind === "comment-line" ||
      cls.kind === "comment-block" ||
      cls.kind === "regex"
    ) {
      // inert for net 1; ledger (net 2) already reconciled above
    } else {
      // code / template-expression: never a silent pass.
      const { line } = lineInfo(source, hit.offset);
      errors.push(
        `runtime marker in unsupported executable context (${cls.kind}) at ${file}:${line} — fail-closed per design/runtime-pin-ast-parser.md §3`,
      );
    }
  }
}

function classifyHtmlFile(file, source, hits, errors, judgeCandidate, ledgerOrLegacy, libOf) {
  let document, contexts;
  try {
    ({ document } = parseHtmlDocument(source, file));
    contexts = htmlScriptContexts(document);
  } catch (e) {
    errors.push(`unparseable HTML with marker evidence: ${e.message}`);
    for (const hit of hits) ledgerOrLegacy(file, source, hit, "unparsed");
    return;
  }
  // Unsupported executable surfaces fail closed if they bear a marker.
  const markerIn = (c) => hits.some((h) => c.startOffset <= h.offset && h.offset < c.endOffset);
  for (const c of contexts) {
    if ((c.kind.startsWith("unsupported") || c.kind === "script-nonexec") && markerIn(c)) {
      errors.push(
        `runtime marker in unsupported HTML context (${c.kind}${c.attrName ? ":" + c.attrName : ""}) in ${file} — fail-closed per design/runtime-pin-ast-parser.md §3`,
      );
    }
  }
  // Parse inline executable scripts once (marker- or sink-bearing).
  const inlineParsed = new Map();
  for (const c of contexts) {
    if (c.kind !== "script-inline") continue;
    const body = source.slice(c.startOffset, c.endOffset);
    if (!markerIn(c) && !RUNTIME_SINK_RE.test(body)) continue;
    try {
      const p = parseJavaScript(body, `${file}#inline-script`);
      inlineParsed.set(c, jsSyntaxSpans(body, p));
    } catch (e) {
      errors.push(`unparseable executable inline script in ${file}: ${e.message}`);
    }
  }
  for (const hit of hits) {
    const inline = contexts.find(
      (c) => c.kind === "script-inline" && c.startOffset <= hit.offset && hit.offset < c.endOffset,
    );
    if (inline) {
      const spans = inlineParsed.get(inline);
      if (!spans) {
        ledgerOrLegacy(file, source, hit, "html-inline-js:unparsed");
        errors.push(`runtime marker in unparseable inline script in ${file} — fail-closed`);
        continue;
      }
      const rel = hit.offset - inline.startOffset;
      const cls = classifyJsOffset(spans, rel);
      ledgerOrLegacy(file, source, hit, `html-inline-js:${cls.kind}`);
      if (cls.kind === "string" || cls.kind === "template-quasi") {
        const boundaryEnd =
          cls.kind === "template-quasi"
            ? cls.span.quasis.find((q) => q.start <= rel && rel < q.end)?.end ?? cls.span.end
            : cls.span.end;
        const ex = extractRawCandidate(source, hit.offset, hit.marker, inline.startOffset + boundaryEnd);
        if (ex.kind === "candidate") {
          const { line } = lineInfo(source, hit.offset);
          judgeCandidate(file, line, libOf(hit.marker), ex.candidate);
        }
      }
      continue;
    }
    const src = contexts.find(
      (c) => c.kind === "script-src" && c.startOffset <= hit.offset && hit.offset < c.endOffset,
    );
    if (src) {
      ledgerOrLegacy(file, source, hit, "html-script-src");
      const ex = extractRawCandidate(source, hit.offset, hit.marker, src.endOffset);
      if (ex.kind === "candidate") {
        const { line } = lineInfo(source, hit.offset);
        judgeCandidate(file, line, libOf(hit.marker), ex.candidate);
      }
      continue;
    }
    if (htmlOffsetInDisplayRegion(document, hit.offset)) {
      ledgerOrLegacy(file, source, hit, "html-display(pre/code)");
      continue;
    }
    ledgerOrLegacy(file, source, hit, "html-text");
  }
}
