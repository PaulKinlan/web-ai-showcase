// Static, browser-free guards for the example-gallery selection ring (bead web-ai-showcase-kxa).
//
// WHAT THIS PROVES
//  · the injected selected-state ring is the two-tone ::after overlay, and the OLD inset accent
//    outline that measured ~1.01:1 over dark thumbnails is gone (the regression guard for kxa);
//  · the component's injected rules and the page stylesheet's equivalent rule sets DECLARE THE SAME
//    THINGS, so the two copies cannot silently diverge - that divergence is exactly how kxa happened
//    (yf2 fixed the stylesheet only, leaving the component's copy behind);
//  · keyboard focus keeps its own outside ring, and is declared after selection because both
//    selectors are specificity (0,2,0) and source order decides;
//  · TOKEN MATH: the WCAG non-text contrast ratio of the token pair the ring relies on, per scheme.
//
// WHAT THIS DOES NOT PROVE (stated so nobody reads more into a green run)
//  · nothing about rendered pixels. There is no browser in this suite (browser/visual acceptance is
//    blocked on the jdz containment bead), so this is NOT visual proof, NOT thumbnail-overlap proof
//    and NOT forced-colors proof. The ~1.01:1 image-overlap measurement needs real thumbnails in a
//    browser and is deliberately not reproduced here.
//  · the token maths is arithmetic on the design tokens, not a measurement of anything as rendered.
//
// Both assertions below are deliberately independent: STRUCTURE pins the expected values (a change to
// border width, band colour or offsets fails even if both files change together), while PARITY pins
// agreement (a change to one file only fails). Neither alone would catch every mutation.
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const COMPONENT = "lib/example-gallery.js";
const STYLESHEET = "public/styles.css";

/** The CSS the component injects (the template literal inside injectStyle()), verbatim. */
function injectedCss() {
  const src = readFileSync(COMPONENT, "utf8");
  const marker = "const css = `";
  const start = src.indexOf(marker);
  assert.ok(start >= 0, `${COMPONENT}: injectStyle() must build its css template literal`);
  const from = start + marker.length;
  const end = src.indexOf("`;", from);
  assert.ok(end > from, `${COMPONENT}: the injected css template literal must be terminated`);
  return src.slice(from, end);
}

/** Whitespace-normalised, sorted declaration list of one rule block, by exact selector. */
function declarations(css, selector) {
  const at = css.indexOf(`${selector} {`);
  assert.ok(at >= 0, `rule not found: ${selector}`);
  const open = css.indexOf("{", at);
  const close = css.indexOf("}", open);
  assert.ok(close > open, `unterminated rule: ${selector}`);
  return css
    .slice(open + 1, close)
    .split(";")
    // Canonicalise so cosmetic spacing cannot pass or fail the comparison: one space after a colon,
    // one after a comma, single spaces elsewhere. Both files are compared through this same lens.
    .map((d) =>
      d
        .replace(/\s*:\s*/g, ": ")
        .replace(/\s*,\s*/g, ", ")
        .replace(/\s+/g, " ")
        .trim(),
    )
    .filter(Boolean)
    .sort();
}

/** CSS with block comments removed, so prose about a rule cannot satisfy or trip an assertion. */
function stripComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

const componentCss = stripComments(injectedCss());
const stylesheet = stripComments(readFileSync(STYLESHEET, "utf8"));

test("the injected selected ring is the two-tone overlay, and the old inset accent outline is gone", () => {
  // The old construction: a -3px inset accent outline drawn straight over the thumbnail image.
  assert.doesNotMatch(
    componentCss,
    /outline\s*:\s*3px solid var\(--accent\)\s*;\s*outline-offset\s*:\s*-3px/,
    "the injected sheet must not reintroduce the 1.01:1 inset accent outline (kxa)",
  );
  assert.doesNotMatch(componentCss, /-3px/, "no -3px geometry may remain in the injected sheet");

  // Selection now neutralises the outline and draws the two-tone overlay instead.
  assert.deepEqual(declarations(componentCss, '.exg-item[aria-pressed="true"]'), [
    "outline: none",
    "position: relative",
  ]);
  const ring = declarations(componentCss, '.exg-item[aria-pressed="true"]::after');
  assert.ok(ring.includes("content: \"\""), "the ring must be generated content");
  assert.ok(ring.includes("position: absolute"), "the ring must be overlaid, not in flow");
  assert.ok(ring.includes("inset: 0"), "the ring must cover the item box");
  assert.ok(ring.includes("border-radius: inherit"), "the ring must follow the item's radius");
  assert.ok(ring.includes("pointer-events: none"), "the ring must not intercept clicks");
  // The accent band is a BORDER: borders are painted in forced-colors mode, box-shadow is not.
  assert.ok(ring.includes("border: 3px solid var(--accent)"), "accent band must be a 3px border");
  assert.ok(
    ring.includes("box-shadow: inset 0 0 0 2px var(--bg-raised)"),
    "the inner band must be the 2px --bg-raised inset shadow, so the accent never borders the image",
  );
});

test("the component's injected rules and the page stylesheet's agree (the two copies cannot silently diverge)", () => {
  assert.deepEqual(
    declarations(componentCss, '.exg-item[aria-pressed="true"]'),
    declarations(stylesheet, '.exg .exg-item[aria-pressed="true"]'),
    "selected-state declarations drifted between the component and public/styles.css",
  );
  assert.deepEqual(
    declarations(componentCss, '.exg-item[aria-pressed="true"]::after'),
    declarations(stylesheet, '.exg .exg-item[aria-pressed="true"]::after'),
    "the ::after ring declarations drifted between the component and public/styles.css",
  );
  assert.deepEqual(
    declarations(componentCss, ".exg-item:focus-visible"),
    declarations(stylesheet, '.exg .exg-item[aria-pressed="true"]:focus-visible'),
    "focus-ring declarations drifted between the component and public/styles.css",
  );
});

test("focus keeps its own ring and is declared after selection (equal specificity, so order decides)", () => {
  const focus = declarations(componentCss, ".exg-item:focus-visible");
  assert.deepEqual(focus, ["outline-offset: 2px", "outline: 3px solid var(--accent)"]);
  const selectedAt = componentCss.indexOf('.exg-item[aria-pressed="true"] {');
  const focusAt = componentCss.indexOf(".exg-item:focus-visible {");
  assert.ok(selectedAt >= 0 && focusAt >= 0, "both rules must exist in the injected sheet");
  assert.ok(
    selectedAt < focusAt,
    "the focus rule must come AFTER the selected rule: both are (0,2,0), so a selected+focused item " +
      "would otherwise lose its outside ring to `outline: none`",
  );
});

// ── TOKEN MATH (arithmetic on design tokens, NOT rendered-pixel or forced-colors proof) ─────────────

const channel = (c) => {
  const s = c / 255;
  return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};

function hexColour(value, token) {
  const m = /^#([0-9a-f]{6})$/i.exec((value || "").trim());
  assert.ok(
    m,
    `${token} must be a literal 6-digit hex for this token maths; got ${JSON.stringify(value)}. ` +
      "If the token legitimately became a function (color-mix/var/oklch), this test cannot compute a " +
      "ratio and must be re-designed rather than silently skipped.",
  );
  return m[1];
}

function luminance(hex) {
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(a, b) {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** Token value inside a `:root { … }` block, searched from `from`. */
function token(css, name, from = 0) {
  const rootAt = css.indexOf(":root", from);
  assert.ok(rootAt >= 0, `no :root block found at or after offset ${from}`);
  const open = css.indexOf("{", rootAt);
  const close = css.indexOf("}", open);
  const block = css.slice(open + 1, close);
  const m = new RegExp(`${name}\\s*:\\s*([^;]+);`).exec(block);
  assert.ok(m, `${name} not found in the :root block at offset ${from}`);
  return m[1];
}

test("TOKEN MATH: the ring's token pair clears WCAG non-text contrast (3:1) in both schemes", () => {
  const darkAt = stylesheet.indexOf("@media (prefers-color-scheme: dark)");
  assert.ok(darkAt >= 0, `${STYLESHEET} must keep a dark-scheme block`);

  const schemes = [
    { name: "light", from: 0 },
    { name: "dark", from: darkAt },
  ];
  for (const { name, from } of schemes) {
    const accent = hexColour(token(stylesheet, "--accent", from), "--accent");
    const raised = hexColour(token(stylesheet, "--bg-raised", from), "--bg-raised");
    const ratio = contrast(accent, raised);
    assert.ok(
      ratio >= 3,
      `${name}: --accent #${accent} on --bg-raised #${raised} is ${ratio.toFixed(2)}:1, below the 3:1 ` +
        "non-text minimum the selection ring needs (the accent band borders this band, never the image)",
    );
  }
});
