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
  markerOffsets,
  isBinarySource,
  RuntimePinParseError,
} from "../scripts/runtime-pin-parser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

function classify(src, needle = "onnxruntime-web@") {
  const offset = src.indexOf(needle);
  assert.notEqual(offset, -1, "fixture must actually contain the needle (else indexOf=-1 false-greens as 'code')");
  const parsed = parseJavaScript(src, "fixture.js");
  return classifyJsOffset(jsSyntaxSpans(src, parsed), offset).kind;
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

test("class bodies are covered: methods, getters, fields, static fields, arrow initializers", () => {
  // Regression for the acorn-walk base override: visiting Literal keys must not
  // drop node.value, or every string inside a class body misclassifies as code.
  assert.equal(classify('class A { m() { return "onnxruntime-web@1.21.0"; } }'), "string");
  assert.equal(classify('class A { get m() { return "onnxruntime-web@1.21.0"; } }'), "string");
  assert.equal(classify('class A { f = "onnxruntime-web@1.21.0"; }'), "string");
  assert.equal(classify('class A { static URL = "https://x/onnxruntime-web@1.21.0/d.js"; }'), "string");
  assert.equal(classify('class A { f = () => "onnxruntime-web@1.21.0"; }'), "string");
  assert.equal(classify('class A { static { const s = "onnxruntime-web@1.21.0"; } }'), "string");
  assert.equal(classify('const o = { m() { return "onnxruntime-web@1.21.0"; } };'), "string");
  assert.equal(classify('class A { m() { class B { g() { return "onnxruntime-web@1.21.0"; } } return B; } }'), "string");
});

test("import/export/import-expression sources and directives classify as string", () => {
  assert.equal(classify('import "onnxruntime-web@1.21.0";'), "string");
  assert.equal(classify('export { x } from "onnxruntime-web@1.21.0";'.replace("export { x }", "export {}"), ), "string");
  assert.equal(classify('const m = import("onnxruntime-web@1.21.0");'), "string");
  assert.equal(classify('"use strict"; const s = "onnxruntime-web@1.21.0";'), "string");
});

test("a bare Identifier in code position classifies as code", () => {
  // A contiguous raw marker can never appear here (it would not parse —
  // fail-closed via RuntimePinParseError); this pins the classifier's fallback.
  const src = "const a = onnxruntime - web;";
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
  const blockShadow = V2_FIXTURE + '{ const ORT_VERSION = "9.9.9"; }\n';
  const ifShadow = V2_FIXTURE + 'if (globalThis.x) { const ORT_VERSION = "9.9.9"; }\n';
  const catchShadow = V2_FIXTURE + 'try { f(); } catch (ORT_VERSION) { g(ORT_VERSION); }\n';
  const forOfShadow = V2_FIXTURE + 'for (const ORT_VERSION of []) { g(ORT_VERSION); }\n';
  const forInWrite = V2_FIXTURE + 'for (ORT_VERSION in {}) { g(ORT_VERSION); }\n';
  const objParamShadow = V2_FIXTURE + 'function f({ORT_VERSION}) { return ORT_VERSION; }\n';
  const arrParamShadow = V2_FIXTURE + 'function f([ORT_VERSION]) { return ORT_VERSION; }\n';
  const blockFnShadow = V2_FIXTURE + 'if (globalThis.x) { function ORT_VERSION() {} }\n';
  const concatSink = V2_FIXTURE.replace("@${ORT_VERSION}/dist/ort.wasm.min.mjs", "@${ORT_VERSION + \"-evil\"}/dist/ort.wasm.min.mjs");
  const callSink = V2_FIXTURE.replace("@${ORT_VERSION}/dist/ort.wasm.min.mjs", "@${String(ORT_VERSION)}/dist/ort.wasm.min.mjs");
  const cases = {
    shadow, reassign, update, nonliteral, blockShadow, ifShadow, catchShadow,
    forOfShadow, forInWrite, objParamShadow, arrParamShadow, blockFnShadow,
    concatSink, callSink,
  };
  for (const [label, src] of Object.entries(cases)) {
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

test("htmlScriptContexts: <template> content is walked (parse5 node.content)", () => {
  const html =
    '<template><script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"></script></template>' +
    '<template><script>const z = 3;</script></template>';
  const { document } = parseHtmlDocument(html, "fixture.html");
  const kinds = htmlScriptContexts(document).map((c) => c.kind);
  assert.ok(kinds.includes("script-src"), "template script src must be visible");
  assert.ok(kinds.includes("script-inline"), "template inline script must be visible");
});

test("htmlScriptContexts: SVG script href is a src context", () => {
  const html =
    '<svg><script href="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"></script></svg>';
  const { document } = parseHtmlDocument(html, "fixture.html");
  const srcs = htmlScriptContexts(document).filter((c) => c.kind === "script-src");
  assert.equal(srcs.length, 1);
  assert.match(srcs[0].src, /onnxruntime-web@1\.21\.0/);
});

test("htmlScriptContexts: MIME essence match — parameters do not make a JS script non-exec", () => {
  const html = '<script type="text/javascript; charset=utf-8">const x = 1;</script>';
  const { document } = parseHtmlDocument(html, "fixture.html");
  const kinds = htmlScriptContexts(document).map((c) => c.kind);
  assert.deepEqual(kinds, ["script-inline"]);
});

test("javascript: URL with embedded TAB/LF (spec-valid obfuscation) is surfaced", () => {
  const html = '<a href="java\nscript:alert(1)">x</a><a href="java\tscript:alert(1)">y</a>';
  const { document } = parseHtmlDocument(html, "fixture.html");
  const hits = htmlScriptContexts(document).filter((c) => c.kind === "unsupported-javascript-url");
  assert.equal(hits.length, 2);
});

test("unquoted script src gets a located script-src context (parse5 always locates attrs)", () => {
  const html = "<script src=https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js></script>";
  const { document } = parseHtmlDocument(html, "fixture.html");
  const contexts = htmlScriptContexts(document);
  assert.equal(contexts.length, 1);
  assert.equal(contexts[0].kind, "script-src");
  assert.ok(contexts[0].startOffset < contexts[0].endOffset);
});

test("entity-encoded marker in script src is visible in the DECODED value", () => {
  // Raw bytes never form a contiguous marker; parse5 decodes the entity, so the
  // census's decoded-attr scan (not the raw byte search) is what catches this.
  const html = '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web&#64;1.21.0/dist/ort.min.js"></script>';
  assert.equal(html.includes("onnxruntime-web@"), false, "raw marker absent by construction");
  const { document } = parseHtmlDocument(html, "fixture.html");
  const srcs = htmlScriptContexts(document).filter((c) => c.kind === "script-src");
  assert.equal(srcs.length, 1);
  assert.ok(srcs[0].src.includes("onnxruntime-web@1.21.0"), "decoded value bears the marker");
});

// --- total marker discovery -------------------------------------------------

test("isBinarySource: NUL byte only", () => {
  assert.equal(isBinarySource("const x = 1;"), false);
  assert.equal(isBinarySource("const x = 1;\u0000"), true);
  assert.equal(isBinarySource(""), false);
});

test("markerOffsets: raw, regex-escaped and every cooked/entity form is discovered", () => {
  const forms = [
    ["onnxruntime-web@1.21.0", false],
    ["@huggingface/transformers@3.7.5", false],
    ["@huggingface\\/transformers@3.7.5", true], // regex-literal escaped slash
    ["onnxruntime-web\\x401.21.0", true],
    ["onnxruntime-web\\u00401.21.0", true],
    ["onnxruntime-web\\u{40}1.21.0", true],
    ["onnxruntime-web\\1001.21.0", true],
    ["onnxruntime-web%401.21.0", true],
    ["onnxruntime-web&#64;1.21.0", true],
    ["onnxruntime-web&#x40;1.21.0", true],
    ["onnxruntime-web&commat;1.21.0", true],
  ];
  for (const [text, escaped] of forms) {
    const hits = markerOffsets(`const s = "${text}";`);
    assert.ok(hits.length >= 1, `no hit for ${text}`);
    assert.equal(hits[0].escaped, escaped, text);
  }
  // Non-marker lookalikes must NOT hit.
  assert.equal(markerOffsets('const s = "onnxruntime-web is great";').length, 0);
});
