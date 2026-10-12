// Deterministic tests for explore/explore.js's highlight() term-regex cache (web-ai-showcase-spc).
//
// WHY: highlight() runs twice per rendered result row (model name + blurb, see renderResults in
// explore/explore.js) and every hit in a render carries the SAME matched-term values, so it used to
// rebuild the identical RegExp per call: measured 40 compilations for a 20-hit render and 200 across
// five typed searches, every one of them the same pattern. The fix memoises on the full ordered term
// list. These tests pin the contract that matters, none of which needs a browser: the compile COUNT
// (via a counted RegExp constructor), the invalidation key, the escaping/matching semantics that must
// not change, and the shared-global-regex state contract.
//
// SEAM: explore/explore.js reads the DOM at module scope (document.getElementById while building `els`),
// so it cannot be imported in Node without a DOM shim, and the change was deliberately kept in-file
// rather than moved to a new module. The contiguous `esc` ... `highlight` region is therefore extracted
// from the SHIPPED source text and evaluated here, so the assertions run against the shipped logic
// rather than a copy. If that region is renamed or moved, extraction throws and this file fails loudly
// instead of silently covering nothing.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = readFileSync(join(ROOT, "explore", "explore.js"), "utf8");

/** End index (exclusive) of a named function declaration, by brace matching. */
function endOfFunction(src, name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} must exist in explore/explore.js`);
  let depth = 0;
  for (let i = src.indexOf("{", start); i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}" && --depth === 0) return i + 1;
  }
  throw new Error(`unbalanced braces after ${name}`);
}

/** The shipped region: esc, the cache, the keyed builder and highlight. */
const REGION = SOURCE.slice(SOURCE.indexOf("function esc("), endOfFunction(SOURCE, "highlight"));

/**
 * Evaluate the shipped region with a COUNTED RegExp constructor, injected as the `RegExp` binding so
 * only this instance is affected. Regex literals (esc's /[&<>"]/g) are not constructor calls, so the
 * count is exactly the term-regex compilation this bead is about.
 */
function makeHarness() {
  const state = { compiles: 0 };
  const Counting = new Proxy(RegExp, {
    construct(target, args) {
      state.compiles++;
      return Reflect.construct(target, args, target);
    },
    apply(target, thisArg, args) {
      state.compiles++;
      return Reflect.apply(target, thisArg, args);
    },
  });
  const factory = new Function("RegExp", `${REGION}\nreturn { esc, highlight };`);
  const { esc, highlight } = factory(Counting);
  return { esc, highlight, compiles: () => state.compiles };
}

const rows = (n) =>
  Array.from({ length: n }, (_, i) => ({
    name: `model-${i}-wasm-quantized`,
    blurb: "tensor kernels, webgpu paths and cpu fallbacks",
  }));

/** One render: renderResults calls highlight twice per hit - name always, blurb when present. */
function render(h, hits, terms) {
  for (const r of hits) {
    h.highlight(r.name, terms);
    h.highlight(r.blurb, terms);
  }
}

test("a 20-hit render compiles the term regex once, not once per highlighted call", () => {
  const h = makeHarness();
  render(h, rows(20), ["wasm", "webgpu", "tensor"]);
  // 40 highlight() calls (20 names + 20 blurbs). Before the fix this was 40 compilations.
  assert.equal(h.compiles(), 1);
});

test("five typed searches compile five times, not 200", () => {
  const h = makeHarness();
  for (const q of ["w", "wa", "was", "wasm", "wasm i"]) render(h, rows(20), [q]);
  assert.equal(h.compiles(), 5);
});

test("invalidation keys on the full ordered term list", () => {
  const h = makeHarness();
  const terms = ["wasm", "webgpu"];
  h.highlight("x", terms);
  assert.equal(h.compiles(), 1);

  h.highlight("x", [...terms]); // same values in a FRESH array: every hit gets its own array
  assert.equal(h.compiles(), 1, "equal values in a new array must hit the cache");

  h.highlight("x", ["wasm", "tensor"]); // same LENGTH, different values
  assert.equal(h.compiles(), 2, "a same-length list of different terms must not reuse the entry");

  h.highlight("x", ["webgpu", "wasm"]); // same values, reordered
  assert.equal(h.compiles(), 3, "the key is the ordered list, so a reordered list is a new entry");

  h.highlight("x", ["wasm", "wasm", "webgpu"]); // duplicate value
  assert.equal(h.compiles(), 4, "a duplicated term is a different list");

  h.highlight("x", []);
  assert.equal(h.compiles(), 4, "no terms means nothing to compile");
});

test("a term containing the separator-like NUL cannot impersonate two separate terms", () => {
  const h = makeHarness();
  h.highlight("x", ["a\u0000b"]);
  h.highlight("x", ["a", "b"]);
  assert.equal(h.compiles(), 2, "the key must escape term values, not join them on a guessable separator");
});

test("matching and escaping semantics are unchanged", () => {
  const h = makeHarness();
  assert.equal(h.highlight("axb", ["a.b"]), "axb", "a dot in a term stays literal");
  assert.equal(h.highlight("a.b", ["a.b"]), "<mark>a.b</mark>");
  assert.equal(h.highlight("a+b", ["a+b"]), "<mark>a+b</mark>");
  assert.equal(h.highlight("f(x)", ["(x)"]), "f<mark>(x)</mark>");
  assert.equal(h.highlight("a\\b", ["a\\b"]), "<mark>a\\b</mark>");
  assert.equal(h.highlight("WASM", ["wasm"]), "<mark>WASM</mark>", "matching stays case-insensitive");
  assert.equal(h.highlight("<wasm>", ["wasm"]), "&lt;<mark>wasm</mark>&gt;", "text is escaped around the mark");
  assert.equal(h.highlight("webgpu or wasm", ["webgpu", "wasm"]), "<mark>webgpu</mark> or <mark>wasm</mark>");
  assert.equal(h.highlight("nothing here", ["absent"]), "nothing here");
  assert.equal(h.highlight("wasm", []), "wasm");
});

test("the pre-existing word-boundary limitation is pinned, not changed", () => {
  const h = makeHarness();
  // The \b(...) wrapper means a term starting with a non-word character cannot match at a boundary.
  // This is pre-existing behaviour and out of scope for the caching change; it is pinned so a future
  // change to it is a deliberate decision rather than an accident.
  assert.equal(h.highlight(".task file", [".task"]), ".task file");
  assert.equal(h.highlight("[i]", ["[i]"]), "[i]");
});

test("the shared global regex stays stateless across calls (lastIndex contract)", () => {
  const h = makeHarness();
  const terms = ["wasm"];
  const first = h.highlight("wasm and wasm", terms);
  const second = h.highlight("wasm and wasm", terms);
  assert.equal(first, "<mark>wasm</mark> and <mark>wasm</mark>");
  assert.equal(second, first, "reusing the cached global regex must not skip a match on the next call");
  assert.equal(h.compiles(), 1, "and both calls used the one compiled regex");
});

test("the retained manual escape carries the Baseline TODO marker", () => {
  // The manual escape is kept deliberately: RegExp.escape() is Baseline Newly available (feature id
  // regexp-escape), below the project's Baseline target, so the marker must stay next to the escape.
  assert.match(REGION, /TODO\(baseline\/regexp-escape\):/);
});
