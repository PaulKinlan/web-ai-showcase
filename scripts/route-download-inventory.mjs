#!/usr/bin/env node
// Task 2b · Phase 1 — the model-download route inventory.
//
// Classifies EVERY built demo route by the runtime/loader family it uses to transfer model assets, who
// controls the byte transfer, and therefore what kind of resume is honestly possible. This is the exact
// denominator the site-wide <model-download-status> component + its adoption gate build on: you cannot
// adopt (or honestly limit) a route you have not classified. Emits download-routes.json.
//
// EXECUTABLE-CONTEXT SCANNER (bead web-ai-showcase-6rgd, ADR web-ai-showcase-kc7p): classification
// signals are extracted ONLY from executable code, never from inert text:
//   · parse5 parses each demo HTML and yields <script> bodies from executable contexts only —
//     anything inside <template> (inert, e.g. j0pm's ort.env snippet) or <noscript> is skipped, and
//     <style> is not a script; <script type="importmap"> bodies are parsed as declarative JSON config,
//     not executed code.
//   · every executable script (inline or .js/.mjs file) is parsed with acorn; acorn-walk collects
//     semantic signals — ImportDeclaration sources, CallExpression callees (identifier + member
//     paths), MemberExpression dotted paths with all prefixes (ort.env preserved as an object/property
//     relation, punctuation intact), string Literals, and Identifiers. CRLF is native to the parser.
//   · FAIL CLOSED: a script that does not parse (or an importmap that is not valid JSON) marks the
//     whole route "unknown" — never a guessed classification.
//
// Usage: node scripts/route-download-inventory.mjs            (writes download-routes.json)
//        node scripts/route-download-inventory.mjs --check    (print summary only, non-zero if any unknown)
//        node scripts/route-download-inventory.mjs --models <dir>  (fixture root for tests; default "models")
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as acorn from "acorn";
import * as walkAst from "acorn-walk";
import * as parse5 from "parse5";

const modelsIdx = process.argv.indexOf("--models");
const MODELS = modelsIdx >= 0 ? process.argv[modelsIdx + 1] : "models";
const routes = [];

// ── executable-source extraction ─────────────────────────────────────────────────────────────────────

/** Executable <script> types. Anything else (importmap handled separately, JSON templates, data-…) is inert. */
const EXECUTABLE_SCRIPT_TYPES = new Set(["", "module", "application/javascript", "text/javascript"]);

/**
 * Collect executable script bodies + importmap texts from a parse5 document.
 * <template> content (inert by definition — j0pm's executable-LOOKING snippet inside a template is
 * documentation, not code) and <noscript> are never descended into.
 */
function extractHtmlScripts(node, out, { inInert = false } = {}) {
  if (!node) return;
  const name = node.nodeName;
  const inert = inInert || name === "template" || name === "noscript";
  if (!inert && name === "script") {
    const type = node.attrs?.find((a) => a.name === "type")?.value ?? "";
    const body = node.childNodes?.find((c) => c.nodeName === "#text")?.value;
    if (body !== undefined) {
      if (type === "importmap") out.importmaps.push(body);
      else if (EXECUTABLE_SCRIPT_TYPES.has(type)) out.scripts.push(body);
    }
  }
  if (node.content) extractHtmlScripts(node.content, out, { inInert: inert });
  for (const child of node.childNodes ?? []) extractHtmlScripts(child, out, { inInert: inert });
}

/** One demo's executable sources: JS files verbatim, HTML scripts + importmaps from executable contexts. */
function readExecutableSources(dir) {
  const out = { scripts: [], importmaps: [], files: 0 };
  const walk = (d) => {
    for (const n of readdirSync(d).sort()) {
      const p = join(d, n);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else if (/\.(js|mjs)$/.test(n)) { out.scripts.push(readFileSync(p, "utf8")); out.files++; }
      else if (/\.html$/.test(n)) {
        extractHtmlScripts(parse5.parse(readFileSync(p, "utf8")), out);
        out.files++;
      }
    }
  };
  walk(dir);
  return out;
}

// ── AST signal collection ────────────────────────────────────────────────────────────────────────────

/** Dotted path of a member chain ("ort.env.wasm"), with computed string-literal segments included. */
function memberPath(node) {
  if (node.type !== "MemberExpression") return null;
  const prop = node.computed
    ? (node.property.type === "Literal" && typeof node.property.value === "string" ? node.property.value : null)
    : (node.property.type === "Identifier" ? node.property.name : null);
  const base = memberPath(node.object) ??
    (node.object.type === "Identifier" ? node.object.name : null);
  if (base === null || prop === null) return null;
  return `${base}.${prop}`;
}

/** Every dotted prefix of a member chain: ort.env.wasm.numThreads → [ort.env, ort.env.wasm, …]. */
function memberPrefixes(path) {
  const parts = path.split(".");
  const out = [];
  for (let i = 2; i <= parts.length; i++) out.push(parts.slice(0, i).join("."));
  return out;
}

/**
 * Parse one executable script and collect its semantic signals.
 * @returns {boolean} false when the script FAILS TO PARSE (route must fail closed).
 */
function collectScriptSignals(code, signals) {
  let ast;
  try {
    ast = acorn.parse(code, {
      ecmaVersion: "latest",
      sourceType: "module",
      allowHashBang: true,
      // Demo pages may embed code the browser accepts but strict ES modules reject; ambiguity still
      // fails closed at the parse level per the ADR — we never guess a classification.
    });
  } catch {
    return false;
  }
  const rememberMember = (node) => {
    const path = memberPath(node);
    if (path) for (const p of memberPrefixes(path)) signals.members.add(p);
  };
  walkAst.full(ast, (node) => {
    switch (node.type) {
      case "ImportDeclaration":
      case "ExportNamedDeclaration":
      case "ExportAllDeclaration":
        if (node.source?.value !== undefined) signals.imports.add(node.source.value);
        break;
      case "CallExpression": {
        const c = node.callee;
        if (c.type === "Identifier") signals.calls.add(c.name);
        else if (c.type === "MemberExpression") {
          const path = memberPath(c);
          if (path) signals.memberCalls.add(path);
        } else if (c.type === "Import") {
          const arg = node.arguments[0];
          if (arg?.type === "Literal" && typeof arg.value === "string") signals.imports.add(arg.value);
        }
        break;
      }
      case "MemberExpression":
        rememberMember(node);
        break;
      case "Property":
        // acorn-walk's full() does not visit non-computed Property KEYS as Identifier nodes (a key is a
        // config name, not a value reference) — but `modelAssetBuffer:` IS the executable signal the
        // mediapipe refine keys on, so collect non-computed identifier keys explicitly.
        if (!node.computed && node.key?.type === "Identifier") signals.idents.add(node.key.name);
        break;
      case "PropertyDefinition":
      case "MethodDefinition":
        if (!node.computed && node.key?.type === "Identifier") signals.idents.add(node.key.name);
        break;
      case "Literal":
        if (typeof node.value === "string") signals.strings.add(node.value);
        break;
      case "Identifier":
        signals.idents.add(node.name);
        break;
    }
  });
  return true;
}

/** Signals for one demo route from all its executable sources. parseFailed ⇒ fail closed downstream. */
function routeSignals(dir) {
  const signals = {
    imports: new Set(), strings: new Set(), calls: new Set(),
    memberCalls: new Set(), members: new Set(), idents: new Set(),
    parseFailed: false,
  };
  const sources = readExecutableSources(dir);
  for (const code of sources.scripts) {
    if (!collectScriptSignals(code, signals)) signals.parseFailed = true;
  }
  for (const map of sources.importmaps) {
    // importmap is declarative config: its string values/keys are visible signals, but it is not
    // executed code. Invalid JSON ⇒ fail closed (ambiguous config, not a guessable one).
    try {
      const json = JSON.parse(map);
      const visit = (v) => {
        if (typeof v === "string") { signals.strings.add(v); signals.imports.add(v); }
        else if (Array.isArray(v)) v.forEach(visit);
        else if (v && typeof v === "object") { for (const [k, val] of Object.entries(v)) { signals.strings.add(k); visit(val); } }
      };
      visit(json);
    } catch {
      signals.parseFailed = true;
    }
  }
  return signals;
}

// ── classification (signal conditions replacing the raw regexes) ────────────────────────────────────

const strOrImport = (s, re) => [...s.strings, ...s.imports].some((v) => re.test(v));
const call = (s, name) => s.calls.has(name) || [...s.memberCalls].some((p) => p === name || p.endsWith(`.${name}`));
const ident = (s, name) => s.idents.has(name);

// signal conditions → family. Ordered by SPECIFICITY (first match wins); each family carries its honest
// byte-control + resume semantics. Conditions are the AST-walk equivalents of the pre-6rgd regexes,
// matched one-for-one (imports/strings, call names incl. member-call tails, member-path prefixes,
// identifiers, quoted filenames) so the committed inventory is reproducible field-for-field.
const FAMILIES = {
  "transformers-resumable-prefetch": {
    // The site prefetches big weights resumably (Range/206/sha256 → cache) before from_pretrained.
    test: (s) => strOrImport(s, /model-prefetch\.mjs/) || call(s, "prefetchModel"),
    byteControl: "site-controlled",
    resume: "resumable",
    note:
      "site fetches weights resumably (Range/206/sha256) into transformers-cache; from_pretrained then cache-hits",
  },
  "webllm": {
    test: (s) => strOrImport(s, /lib\/webllm\.js|@mlc-ai|web-llm/) || ident(s, "CreateMLCEngine"),
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note: "WebLLM/MLC engine fetches + caches shards; no per-file resume hooks exposed to the page",
  },
  "mediapipe": {
    test: (s) =>
      // .task/.tflite are matched as QUOTED FILENAMES (string literals ending in the extension) — a
      // bare property access like row.task is an Identifier/Member signal, never a filename string.
      strOrImport(s, /lib\/mediapipe\.js|tasks-vision|tasks-genai/) || ident(s, "FilesetResolver") ||
      [...s.strings].some((v) => v.endsWith(".task") || v.endsWith(".tflite")),
    byteControl: "runtime-owned",
    resume: "runtime-owned",
    note:
      "MediaPipe Tasks fetches the .task/.tflite bundle; download managed inside the WASM runtime",
  },
  "raw-ort": {
    // ort.env / ort.InferenceSession as an object/property RELATION (member-path prefix), not text.
    test: (s) =>
      strOrImport(s, /onnxruntime-web/) || ident(s, "InferenceSession") ||
      s.members.has("ort.env") || s.members.has("ort.InferenceSession"),
    byteControl: "site-controlled",
    // refined below: resumable only if it goes through lib/model-download.js
    resume: "restart-only",
    note:
      "page fetches the .onnx bytes itself and hands them to ORT; restartable unless routed via model-download.js",
  },
  "transformers-wrapped": {
    // Third-party libs (outetts, kokoro-js) that wrap Transformers.js and download through it.
    test: (s) => strOrImport(s, /outetts@|kokoro-js[@/]|npm\/(outetts|kokoro-js)/),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "third-party library (outetts / kokoro-js) wraps Transformers.js; it downloads + caches via TJS, so per-file progress depends on the wrapper surfacing it",
  },
  "transformers-from_pretrained": {
    test: (s) =>
      call(s, "from_pretrained") || [...s.members].some((p) => p.endsWith(".from_pretrained")) ||
      ["AutoModel", "AutoProcessor", "AutoTokenizer", "ForConditionalGeneration"].some((n) => ident(s, n)),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note:
      "Transformers.js fetches each file (no Range); re-running resumes at whole-file granularity from Cache Storage",
  },
  "transformers-pipeline": {
    test: (s) => strOrImport(s, /lib\/webai\.js/) || call(s, "loadPipeline") || call(s, "pipeline"),
    byteControl: "runtime-owned",
    resume: "cached-only",
    note: "Transformers.js pipeline() fetches each file (no Range); whole-file cache resume only",
  },
  "browser-builtin": {
    test: (s) =>
      ["window.ai", "window.LanguageModel", "window.Summarizer", "window.Translator", "window.Rewriter",
        "window.Writer", "LanguageModel.create", "LanguageModel.availability", "ai.languageModel"]
        .some((m) => s.members.has(m)),
    byteControl: "browser-owned",
    resume: "runtime-owned",
    note:
      "the browser owns the model + its download state; adapt as a capability signal, never fabricate per-file detail",
  },
};
const FAMILY_ORDER = Object.keys(FAMILIES);

for (const slug of readdirSync(MODELS).sort()) {
  const dir = join(MODELS, slug);
  if (!statSync(dir).isDirectory()) continue;
  if (!existsSync(join(dir, "index.html"))) continue; // built route
  const s = routeSignals(dir);

  // multi-model page = a subpage that loads ≥2 model groups concurrently
  const multiModel = existsSync(join(dir, "multi-model")) || existsSync(join(dir, "multimodel"));

  // FAIL CLOSED (ADR #4): any executable source that did not parse makes the route unclassifiable.
  let family = "unknown";
  if (!s.parseFailed) {
    for (const f of FAMILY_ORDER) {
      if (FAMILIES[f].test(s)) {
        family = f;
        break;
      }
    }
  }

  // Does the route ADOPT the shared, component-rendering loader? createModelLoader (lib/model-loader.js)
  // and createResumableLoader (lib/resumable-loader.mjs, PaliGemma) both render <model-download-status>
  // and route progress through the adapters (Phase 3/4). Calling one of them IS the adoption.
  const usesResumable = call(s, "createResumableLoader") || strOrImport(s, /resumable-loader/);
  const usesCentral = call(s, "createModelLoader");
  const adoption = usesResumable ? "resumable-loader" : (usesCentral ? "central-loader" : "bypass");

  // does it download at all? a route with NO loader signal AND no worker is non-applicable.
  const hasWorker = existsSync(join(dir, "worker.js"));
  const dynamicTransformersImport = [...s.imports].some((v) => v.includes("transformers"));
  const onnxFilename = [...s.strings].some((v) => /\.onnx\b/.test(v));
  let byteControl, resume, note, status;
  if (family === "unknown") {
    if (s.parseFailed) {
      byteControl = "unknown";
      resume = "unknown";
      note = "executable source failed to parse — inventory fails closed rather than guess (see bead 6rgd)";
      status = "unknown";
    } else if (!hasWorker && !usesCentral && !dynamicTransformersImport && !onnxFilename) {
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
      family === "mediapipe" && (ident(s, "modelAssetBuffer") || s.members.has("modelAssetBuffer")) &&
      call(s, "fetch")
    ) {
      byteControl = "site-controlled";
      resume = strOrImport(s, /model-download\.js/) || call(s, "downloadModelFile") ? "resumable" : "restart-only";
      note = resume === "resumable"
        ? "site fetches the MediaPipe artifact via lib/model-download.js (Range/206/sha256) before supplying modelAssetBuffer"
        : "site fetches and verifies the complete MediaPipe artifact before supplying modelAssetBuffer; interruption restarts from byte zero";
    }
    // refine raw-ort: resumable only when the weights go through the site's resumable downloader
    if (family === "raw-ort" && (strOrImport(s, /model-download\.js/) || call(s, "downloadModelFile"))) {
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
  generated: "2026-10-12",
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
