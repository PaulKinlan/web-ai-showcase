// Deterministic DOM test for the nsfw wild tile builder (web-ai-showcase-giw).
//
// No browser (jdz blocks Chrome): the REAL buildTileFigure function is extracted from the demo's
// inline script (balanced-brace source slice, so the test can never drift from the shipped code)
// and driven against a minimal element mock — the same globalThis.document stubbing pattern as
// test/qwen2-vl-degradation.test.mjs. Severity calibrated as SELF-XSS: the filename is the
// uploader's own, so the threat is markup in the filename executing in the uploader's own page.
//
// Covers: (1) malicious filenames render as inert text — no elements/attributes are born from the
// label; (2) ordinary filenames keep the exact pre-fix DOM shape (focus/appearance preservation:
// same tags, classes, alt text, verdict span, "not screened" fallback).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const HTML = readFileSync(
  fileURLToPath(new URL("../models/nsfw-image-detection/wild/index.html", import.meta.url)),
  "utf8",
);

/** Extract one top-level function by name from the inline script (balanced braces). */
function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found in demo source`);
  let depth = 0, i = source.indexOf("{", start);
  for (let j = i; j < source.length; j++) {
    if (source[j] === "{") depth++;
    else if (source[j] === "}") { depth--; if (depth === 0) return source.slice(start, j + 1); }
  }
  assert.fail(`unbalanced braces extracting ${name}`);
}

class MockElement {
  constructor(tag) { this.tag = tag; this.children = []; this.className = ""; this.src = ""; this.alt = ""; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get textContent() { return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join(""); }
  set textContent(v) { this.children = [String(v)]; }
}

function buildWith(decl) {
  const prev = globalThis.document;
  globalThis.document = { createElement: (tag) => new MockElement(tag) };
  try {
    return new Function("document", `"use strict"; ${decl}; return buildTileFigure;`)(globalThis.document);
  } finally { globalThis.document = prev; }
}

const buildTileFigure = buildWith(extractFunction(HTML, "buildTileFigure"));

/** Recursively collect every element reachable from a node. */
function elements(node, out = []) {
  for (const c of node.children ?? []) {
    if (typeof c !== "string") { out.push(c); elements(c, out); }
  }
  return out;
}

test("giw repro: a filename containing markup renders as inert text, not elements", () => {
  const evil = `<img src=x onerror=globalThis.__pwned=1>.png`;
  const fig = buildTileFigure({ key: "user0", label: evil, url: "data:image/png;base64,AAAA" }, undefined);
  // The figure has exactly the two intended children: img + figcaption — nothing born from the label.
  assert.equal(fig.children.length, 2);
  const [img, caption] = fig.children;
  assert.equal(img.tag, "img");
  assert.equal(caption.tag, "figcaption");
  // alt/src receive the raw strings as VALUES (attribute assignment never reinterprets markup).
  assert.equal(img.alt, evil);
  assert.equal(img.src, "data:image/png;base64,AAAA");
  // The label appears as a TEXT child of figcaption, verbatim, and no nested element was created.
  assert.equal(caption.children[0], evil);
  const all = elements(fig);
  assert.deepEqual(all.map((e) => e.tag), ["img", "figcaption", "br"], "no extra elements from the filename");
  assert.ok(!all.some((e) => "onerror" in e || e.tag === "script"), "no handler/script born from the filename");
  // And the exact self-XSS payload string is present as text, i.e. displayed, not executed.
  assert.match(caption.textContent, /onerror=globalThis\.__pwned=1/);
});

test("ordinary filename with a safe verdict keeps the pre-fix DOM shape and classes", () => {
  const fig = buildTileFigure(
    { key: "user0", label: "beach-photo.png", url: "data:image/png;base64,BBBB" },
    { user0: { safe: false, nsfw: 0.876 } },
  );
  assert.equal(fig.tag, "figure");
  assert.equal(fig.className, "withheld"); // non-safe verdict -> withheld class, as before
  const [img, caption] = fig.children;
  assert.equal(img.alt, "beach-photo.png");
  assert.deepEqual(
    caption.children.map((c) => (typeof c === "string" ? "text" : c.tag)),
    ["text", "br", "span"],
    "caption shape: label text, <br>, verdict span",
  );
  const span = caption.children[2];
  assert.equal(span.className, "cap-verdict withheld");
  assert.equal(span.textContent, "nsfw 88%");
});

test("safe verdict keeps .safe class; unscreened tile keeps the plain \"not screened\" fallback", () => {
  const safe = buildTileFigure({ key: "t0", label: "solid blue", url: "data:,0" }, { t0: { safe: true, nsfw: 0.01 } });
  assert.equal(safe.className, "safe");
  assert.equal(safe.children[1].children[2].textContent, "nsfw 1%");
  const plain = buildTileFigure({ key: "absent", label: "noise.png", url: "data:,1" }, { other: { safe: true, nsfw: 0 } });
  assert.equal(plain.className, "");
  assert.equal(plain.children[1].textContent, "noise.pngnot screened");
});

test("static complement: no innerHTML/insertAdjacentHTML/document.write sink remains in the demo", () => {
  // Strip comments first: the fix's guidance-citation comment legitimately NAMES the sinks it
  // removed (capture-ux-sinks.test.mjs uses the same stripComments approach).
  const script = HTML
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
  for (const sink of ["innerHTML", "insertAdjacentHTML", "document.write", "outerHTML"]) {
    assert.ok(!script.includes(sink), `dangerous sink ${sink} still present in the demo script`);
  }
});

test("the extracted builder contains no string-concatenation fallback (regression guard for giw)", () => {
  const decl = extractFunction(HTML, "buildTileFigure");
  assert.ok(!decl.includes("innerHTML"), "buildTileFigure must not use innerHTML");
  assert.ok(decl.includes("textContent"), "verdict text must go through textContent");
  assert.ok(decl.includes("append"), "structure must be built with append");
});
