// Deterministic tests for explore/explore.js's highlight() term-regex cache (web-ai-showcase-spc).
//
// WHY: highlight() runs twice per rendered result row (model name + blurb, see renderResults), so it
// used to rebuild the RegExp per call: measured 40 compilations for a 20-hit render and 200 across five
// distinct searches, every one of them a pattern that had just been built.
//
// The counts are WORKLOAD-CONDITIONAL, and these tests pin each case rather than a render-wide number,
// because the term list varies PER HIT: the worker records, per document, only the query terms that
// document matched (matchedByDoc in search/worker.js), so hits in one render can carry different lists.
//   20-hit render, all hits one list .......... 40 -> 1
//   20-hit render, two alternating lists ...... 40 -> 2
//   20-hit render, three interleaved lists .... 40 -> 3
//   20-hit render, 16 distinct lists (== cap) . 40 -> 16
//   20-hit render, 18 distinct lists (> cap) .. 40 -> 20   (documented: ~one compile per hit)
//   five distinct searches .................... 200 -> 5
//
// The cap is 16 because the number of distinct lists one render can carry is bounded by the non-empty
// subsets of the query's tokenized terms: 15 for the largest query worth caching (four terms), plus one
// spare. Beyond it the cache still degrades gracefully - one eviction-and-recompile per hit, never
// unbounded - and that limit is asserted here so it stays visible.
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

/** One render: renderResults calls highlight twice per hit - name always, blurb when present. */
function render(h, perHitTerms) {
  perHitTerms.forEach((terms, i) => {
    h.highlight(`model-${i}-wasm-quantized`, terms);
    h.highlight("tensor kernels, webgpu paths and cpu fallbacks", terms);
  });
}
const cycles = (n) => (count) => Array.from({ length: count }, (_, i) => [`list${i % n}a`, `list${i % n}b`]);
const termList = (n) => Array.from({ length: n }, (_, i) => `term${i % 4}-${i}`);

test("a 20-hit render where every hit carries the same term list compiles once, not once per call", () => {
  const h = makeHarness();
  render(h, Array.from({ length: 20 }, () => ["wasm", "webgpu", "tensor"]));
  // 40 highlight() calls (20 names + 20 blurbs). Before the fix this was 40 compilations.
  assert.equal(h.compiles(), 1);
});

test("a render whose hits carry different term lists compiles once per distinct list, not per hit", () => {
  // The real shape: matchedTerms comes from per-document postings, so hits can alternate between lists.
  const alternating = makeHarness();
  render(alternating, Array.from({ length: 20 }, (_, i) => (i % 2 ? ["webgpu"] : ["wasm"])));
  assert.equal(alternating.compiles(), 2, "two alternating lists must cost two compiles, not twenty");

  const interleaved = makeHarness();
  render(interleaved, Array.from({ length: 20 }, (_, i) => [["wasm"], ["webgpu"], ["wasm", "webgpu"]][i % 3]));
  assert.equal(interleaved.compiles(), 3, "three interleaved lists must cost three compiles");
});

test("five distinct searches compile five times, not 200", () => {
  // Realistic tokenized terms: the worker lowercases, splits on non-alphanumerics and drops tokens
  // shorter than 2 characters and stop-words (search/worker.js tokenize), so these are the shapes the
  // highlighter actually receives. (A one-character query tokenizes to no terms, so it compiles nothing.)
  const searches = [
    ["quantized"],
    ["quantized", "wasm"],
    ["wasm", "webgpu"],
    ["quantized", "wasm", "webgpu"],
    ["tensor", "webgpu"],
  ];
  const h = makeHarness();
  for (const terms of searches) render(h, Array.from({ length: 20 }, () => terms));
  assert.equal(h.compiles(), searches.length);
});

test("the cache is bounded: it holds the cap, and beyond the cap effort degrades to ~one compile per hit", () => {
  const atCap = makeHarness();
  render(atCap, cycles(16)(20));
  assert.equal(atCap.compiles(), 16, "16 distinct lists fit the cap, so each compiles once");

  const overCap = makeHarness();
  render(overCap, cycles(18)(20));
  // Documented degradation, not a guarantee: a working set larger than the cap evicts and recompiles.
  // This is still half the pre-fix cost (40), but it is deliberately asserted so the limit is visible.
  assert.equal(overCap.compiles(), 20);
});

test("eviction is least-recently-used: a cache HIT refreshes recency and the LRU entry is the one dropped", () => {
  // The discriminating shape matters: touching a list that was ALREADY EVICTED re-inserts it at the
  // tail under any insertion-ordered policy, so a FIFO cache would answer the same way and the test
  // would not bite (found by review, bead web-ai-showcase-4v5v). The only sequence that separates LRU
  // from FIFO is a HIT on a still-cached entry before the eviction, which refreshes recency under LRU
  // and does nothing under FIFO.
  const h = makeHarness();
  for (let i = 0; i < 16; i++) h.highlight("name", [`list${i}a`, `list${i}b`]);
  assert.equal(h.compiles(), 16, "16 distinct lists fill the cap exactly");

  h.highlight("name", ["list0a", "list0b"]); // HIT on a cached entry: no compile, but recency moves
  assert.equal(h.compiles(), 16, "a cache hit must not recompile");

  h.highlight("name", ["list16a", "list16b"]); // 17th distinct list: exactly one entry must go
  assert.equal(h.compiles(), 17);

  // Under LRU the hit above saved list0 and list1 is the least recently used, so list0 is still cached
  // and list1 has to recompile. A FIFO cache evicts list0 here, and a clear-on-overflow cache would
  // have dropped everything, so both fail the next assertion.
  h.highlight("name", ["list0a", "list0b"]);
  assert.equal(h.compiles(), 17, "the list refreshed by a hit must survive the eviction");
  h.highlight("name", ["list1a", "list1b"]);
  assert.equal(h.compiles(), 18, "the least recently used list must be the one that was dropped");
});

test("an evicted list is recompiled, and the most recent one is not", () => {
  const h = makeHarness();
  for (let i = 0; i < 17; i++) h.highlight("name", [`list${i}a`, `list${i}b`]);
  assert.equal(h.compiles(), 17, "17 distinct lists: the 17th insert evicts exactly one entry");

  h.highlight("name", ["list0a", "list0b"]); // evicted by the seventeenth insert -> must recompile
  assert.equal(h.compiles(), 18);
  h.highlight("name", ["list16a", "list16b"]); // most recent -> still cached
  assert.equal(h.compiles(), 18);
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

test("the cached regex is only ever used with replace, and the escape keeps its Baseline marker", () => {
  // String.prototype.replace resets lastIndex, which is what makes sharing a global regex safe. A call
  // to test()/exec() on the cached object would be stateful and is banned by the comment above it; that
  // ban is not observable through replace(), so it is pinned statically here.
  assert.doesNotMatch(REGION, /\.(test|exec)\s*\(/, "the cached regex must not be used with test()/exec()");
  // RegExp.escape() is Baseline Newly available (feature id regexp-escape), below the project's target,
  // so the manual escape stays and must keep its marker.
  assert.match(REGION, /TODO\(baseline\/regexp-escape\):/);
});
