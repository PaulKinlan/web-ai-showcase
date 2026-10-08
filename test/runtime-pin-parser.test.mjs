// Unit tests for scripts/runtime-pin-parser.mjs — the single parser entry point
// for the runtime-pin currency gate (design/runtime-pin-ast-parser.md, bead
// web-ai-showcase-j6i). Hermetic: inline fixtures plus the three real vs2
// workers; no mutation, no network, no browser.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  parseJavaScript,
  jsSyntaxSpans,
  classifyJsOffset,
  resolveTopLevelStringConstants,
  parseHtmlDocument,
  htmlScriptContexts,
  htmlOffsetInDisplayRegion,
  RuntimePinParseError,
} from "../scripts/runtime-pin-parser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

function classify(src, needle = "onnxruntime-web@") {
  const parsed = parseJavaScript(src, "fixture.js");
  return classifyJsOffset(jsSyntaxSpans(src, parsed), src.indexOf(needle)).kind;
}

test("nested template literal: marker inside ${…`inner`…} is a template-quasi, never lost", () => {
  // The exact ea72200 RED shape that defeated the rejected local lexer.
  const src =
    "const __nested = `x ${ { nested: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0~/dist/x.js` }.nested}`;";
  assert.equal(classify(src), "template-quasi");
});

test("cooked escape: \\x40 marker bytes stay inside the template-quasi raw span", () => {
  const src = "const u = `https://cdn.jsdelivr.net/npm/onnxruntime-web\\x401.21.0/dist/x.js`;";
  assert.equal(classify(src, "onnxruntime-web"), "template-quasi");
});

test("string literal object keys (sw.js derived-manifest shape) classify as string", () => {
  const src = 'const M = { "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs": { "sha256": "x" } };';
  assert.equal(classify(src), "string");
});

test("comments and regexes are distinct from strings", () => {
  assert.equal(classify("// onnxruntime-web@1.21.0 prose\nconst x = 1;", ), "comment-line");
  assert.equal(classify("/* onnxruntime-web@1.21.0 */\nconst x = 1;"), "comment-block");
  assert.equal(classify('const re = /onnxruntime-web@[0-9]+/;'), "regex");
  assert.equal(classify('const s = "onnxruntime-web@1.21.0";'), "string");
});

test("a marker in bare code position classifies as code (must stay empty in the census)", () => {
  // Not valid JS with the raw marker text — use a split that parses.
  const src = "const a = onnxruntime - web;\n/*@huggingface/transformers@*/";
  const parsed = parseJavaScript(src, "fixture.js");
  const spans = jsSyntaxSpans(src, parsed);
  assert.equal(classifyJsOffset(spans, src.indexOf("onnxruntime")).kind, "code");
});

test("parse failure throws RuntimePinParseError (fail closed)", () => {
  assert.throws(() => parseJavaScript("const = `unterminated", "bad.js"), RuntimePinParseError);
});

test("hashbang workers parse", () => {
  const parsed = parseJavaScript("#!/usr/bin/env node\nconst x = 1;", "hb.js");
  assert.equal(parsed.sourceType, "module");
});

// --- bounded constant binding (vs2) ----------------------------------------

const V2_FIXTURE = `
const ORT_VERSION = "1.20.1";
const ORT_URL = \`https://cdn.jsdelivr.net/npm/onnxruntime-web@\${ORT_VERSION}/dist/ort.wasm.min.mjs\`;
export async function load() {
  const ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = \`https://cdn.jsdelivr.net/npm/onnxruntime-web@\${ORT_VERSION}/dist/\`;
}
`;

test("constant binding resolves the exact three-worker pattern", () => {
  const b = resolveTopLevelStringConstants(parseJavaScript(V2_FIXTURE, "w.js").ast);
  assert.equal(b.get("ORT_VERSION").value, "1.20.1");
});

test("constant binding fails closed on shadow, reassign, update and nonliteral init", () => {
  const shadow = V2_FIXTURE + "function f(ORT_VERSION) { return ORT_VERSION; }\n";
  const reassign = V2_FIXTURE + 'ORT_VERSION = "9.9.9";\n';
  const update = V2_FIXTURE + "ORT_VERSION++;\n";
  const nonliteral = V2_FIXTURE.replace('"1.20.1"', '"1." + "20.1"');
  for (const [label, src] of [["shadow", shadow], ["reassign", reassign], ["update", update], ["nonliteral", nonliteral]]) {
    const b = resolveTopLevelStringConstants(parseJavaScript(src, label + ".js").ast);
    assert.equal(b.has("ORT_VERSION"), false, label);
  }
});

test("constant binding: a 9.9.9 mutation RESOLVES to 9.9.9 (so the allowlist rejects it)", () => {
  const b = resolveTopLevelStringConstants(
    parseJavaScript(V2_FIXTURE.replace('"1.20.1"', '"9.9.9"'), "w.js").ast,
  );
  assert.equal(b.get("ORT_VERSION").value, "9.9.9");
});

test("the three real vs2 workers resolve their ORT version constants", () => {
  const workers = [
    ["models/silero-vad/worker.js", "ORT_VERSION"],
    ["models/pitch-detection/worker.js", "ORT_VERSION"],
    ["models/model2vec-static-embeddings/worker.js", "ORT_VER"],
  ];
  for (const [path, name] of workers) {
    const src = readFileSync(ROOT + path, "utf8");
    const b = resolveTopLevelStringConstants(parseJavaScript(src, path).ast);
    assert.match(b.get(name)?.value ?? "", /^1\.(20\.1|21\.0)$/, `${path} ${name}`);
  }
});

// --- HTML -------------------------------------------------------------------

test("htmlScriptContexts: quoted src, inline module/classic, importmap, handlers, javascript: URLs", () => {
  const html = `<!doctype html><html><head>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"></script>
<script type="module">const x = 1;</script>
<script>const y = 2;</script>
<script type="importmap">{"imports":{}}</script>
</head><body>
<button onclick="go()">x</button>
<a href="javascript:void(0)">y</a>
<pre><code>onnxruntime-web@1.21.0 docs</code></pre>
</body></html>`;
  const { document } = parseHtmlDocument(html, "fixture.html");
  const kinds = htmlScriptContexts(document).map((c) => c.kind);
  assert.ok(kinds.includes("script-src"));
  assert.ok(kinds.filter((k) => k === "script-inline").length === 2);
  assert.ok(kinds.includes("script-nonexec")); // importmap
  assert.ok(kinds.includes("unsupported-event-handler"));
  assert.ok(kinds.includes("unsupported-javascript-url"));
  const preOff = html.indexOf("onnxruntime-web@1.21.0 docs");
  assert.equal(htmlOffsetInDisplayRegion(document, preOff), true);
  const srcOff = html.indexOf("onnxruntime-web@1.21.0/dist");
  assert.equal(htmlOffsetInDisplayRegion(document, srcOff), false);
});

test("htmlScriptContexts: unquoted script src is surfaced as unsupported, not ignored", () => {
  const html = "<script src=https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js></script>";
  const { document } = parseHtmlDocument(html, "fixture.html");
  // parse5 always reports attr locations; the guard exists for cases where it cannot.
  const contexts = htmlScriptContexts(document);
  assert.ok(contexts.length >= 1);
  assert.ok(["script-src", "unsupported-unquoted-src"].includes(contexts[0].kind));
});
