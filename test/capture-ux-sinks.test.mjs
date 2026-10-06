// Static sink-closure proof for lib/capture-ux.js (web-ai-showcase-zc7).
//
// The threat-model/vuln-discovery audit found h() exposing an unescaped `html:` attribute — an
// innerHTML sink that any future caller could turn into DOM-XSS. The fix removed the branch and
// replaced it with an `icon:` option that (a) allowlist-asserts the markup is one of the module's
// own static SVG factory outputs and (b) parses it with DOMParser("image/svg+xml"), never an HTML
// sink. The one dynamic call site (the dropzone accept string) was converted to DOM construction
// with textContent.
//
// These assertions run on the SOURCE (browser-free suite convention): strip comments first so
// prose that mentions a sink ("never innerHTML") can never mask or fake a real one.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const SRC = fileURLToPath(new URL("../lib/capture-ux.js", import.meta.url));
const HARNESS = fileURLToPath(new URL("../lib/__capture-selftest__/index.html", import.meta.url));

/** Remove block comments and whole-line/trailing-whitespace-preceded line comments.
 *  Conservative on purpose: a `//` is only a comment start at line start or after whitespace,
 *  so URL-ish `://` and `${base}//x` template content can never mask code that follows
 *  (web-ai-showcase-zc7 reviewer P2 — the old pattern could mask a same-line sink). */
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "))
    .replace(/(^|[ \t])\/\/[^\n]*/g, "$1");
}

const code = stripComments(readFileSync(SRC, "utf8"));

test("capture-ux.js contains no HTML sink in code (innerHTML family + HTML parsing)", () => {
  for (const sink of [
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "insertAdjacentElement",
    "document.write",
    "createContextualFragment",
    '"text/html"',
    "srcdoc",
  ]) {
    assert.ok(!code.includes(sink), `lib/capture-ux.js still contains the HTML sink: ${sink}`);
  }
  // DOMParser is only ever used for the icon path, and only as image/svg+xml.
  for (const m of code.matchAll(/parseFromString\([^)]*\)/g)) {
    assert.match(m[0], /"image\/svg\+xml"/, `unexpected DOMParser mime: ${m[0]}`);
  }
});

test("h() offers no `html` attribute branch", () => {
  assert.ok(!/k === "html"/.test(code), 'h() still recognises a "html" attribute');
  assert.ok(!/"html"\s*:/.test(code), "a caller still passes html: to h()");
});

test("icon: is allowlist-asserted and DOMParser-parsed (never HTML)", () => {
  assert.ok(/k === "icon"/.test(code), "h() must keep the dedicated icon: attribute");
  assert.ok(code.includes("TRUSTED_ICON_MARKUP"), "the icon allowlist set must exist");
  assert.ok(
    /new DOMParser\(\)\.parseFromString\(\s*svg,\s*"image\/svg\+xml"\s*\)/.test(code),
    "icons must be parsed with DOMParser image/svg+xml",
  );
  assert.ok(
    /throw new TypeError\([^)]*module-local static SVG/.test(code),
    "trustedIconNode must throw a TypeError on non-allowlisted markup",
  );
  // Fail closed on namespace: XML parsing does NOT infer the SVG namespace from a bare <svg>,
  // so the factories must declare xmlns and the parse result must be namespace-checked
  // (web-ai-showcase-zc7 reviewer P0/P2 — a null-namespace root renders as an invisible inert
  // element, silently stripping every icon).
  assert.ok(
    code.includes('<svg xmlns="http://www.w3.org/2000/svg"'),
    "the _svg() template must declare the SVG xmlns",
  );
  assert.ok(
    /namespaceURI !== "http:\/\/www\.w3\.org\/2000\/svg"/.test(code),
    "trustedIconNode must reject a non-SVG-namespace parse result",
  );
  // The allowlist is built from the module's own icon factories only.
  const setBody = code.match(/TRUSTED_ICON_MARKUP = new Set\(\[([\s\S]*?)\]\)/);
  assert.ok(setBody, "TRUSTED_ICON_MARKUP must be a Set literal");
  const entries = setBody[1].match(/icon[A-Z]\w*\(\)/g) || [];
  assert.ok(entries.length >= 9, `allowlist should cover every icon factory, got ${entries.length}`);
  for (const e of entries) {
    assert.ok(new RegExp(`const ${e.slice(0, -2)} =`).test(code), `${e} must be a module-local factory`);
  }
});

test("dropzone accept string is built from textContent, not interpolated markup", () => {
  const dz = code.match(/cap__dz-text[\s\S]{0,300}/);
  assert.ok(dz, "dropzone text span must exist");
  assert.ok(/cap__dz-accept",\s*text:/.test(dz[0]), "the accept chip must be set via text:");
  // `${accept}` inside a text: template is safe (textContent); what must never come back is
  // accept interpolated into a MARKUP template.
  assert.ok(!dz[0].includes("<span") && !dz[0].includes("<strong>Choose"),
    "the dropzone text must be built from DOM nodes, not a markup template");
});

test("self-test harness drives the sink-closure proof headlessly — and has no sink of its own", () => {
  const harness = readFileSync(HARNESS, "utf8");
  assert.ok(harness.includes("zc7-accept-is-text-not-markup"), "hostile-accept probe must be recorded");
  assert.ok(harness.includes("zc7-icons-render-via-allowlist"), "icon allowlist probe must be recorded");
  assert.ok(harness.includes('onerror="window.__zc7xss=1"'), "probe must use an executable-markup canary");
  // The harness itself must not interpolate test details as markup (reviewer P1: a raw hostile
  // detail string re-injected the payload into the results table and falsified the evidence).
  assert.ok(!stripComments(harness).includes("innerHTML"), "the harness must not use innerHTML");
  assert.ok(!harness.includes("dz-accept text=${JSON.stringify"), "probe details must not echo raw hostile markup");
  // Both probes must also be recorded on the throw path.
  const catchBlock = harness.match(/\} catch \(err\) \{([\s\S]*?)\} finally \{[\s\S]*?cap\?\.destroy/);
  assert.ok(catchBlock, "probe catch/finally structure must exist");
  assert.ok(
    catchBlock[1].includes("zc7-accept-is-text-not-markup") && catchBlock[1].includes("zc7-icons-render-via-allowlist"),
    "the catch path must record BOTH zc7 probes as failed",
  );
});
