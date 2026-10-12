// Box reservation for the Qwen2-VL sample strip (bead web-ai-showcase-0rr).
//
// Why this test exists: the component CSS lives in `QWEN_CSS` inside models/qwen2-vl/qwen2vl.js and is
// appended to <head> at RUNTIME by a deferred module. Until it lands, `public/styles.css` has no rules
// for these classes at all, so the sample images paint at intrinsic size (up to 620x440) and the
// two-column `.vlm-grid` sits stacked full width — then both reflow when the module loads. Each page
// therefore carries a render-blocking `<style id="qwen2vl-sample-box">` copy of the GEOMETRY rules, and
// this test is the thing that keeps that copy honest:
//   * every declaration in the page copy must exist, identically, in QWEN_CSS (so the copy cannot
//     drift into a second source of truth with different numbers);
//   * the thumbs must declare the CSS box they are rendered in (76x56) with `object-fit:cover` in both
//     places, because the intrinsic aspect ratios differ by design (1.333-3.892) and cover CROPS them;
//   * each preview must declare the intrinsic size of ITS OWN initial `src` asset, read from the image
//     header here rather than trusted from a table, so swapping the sample image without updating the
//     attribute fails.
// This is a static guarantee only: it does NOT measure CLS and it cannot (that needs a real browser at
// desktop and mobile widths, still gated by bead jdz). No browser, no network, no writes.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// Properties that move a box (and therefore a page) when they change. The page copy mirrors GEOMETRY
// only - colour, focus and transition styling stay in QWEN_CSS on purpose, where they cannot shift
// anything - so the drift check below is closure over THIS set rather than equality of every
// declaration (which would force the copy to duplicate colours as well).
const BOX_PREFIXES = [
  "display",
  "position",
  "inset",
  "top",
  "right",
  "bottom",
  "left",
  "flex",
  "order",
  "gap",
  "row-gap",
  "column-gap",
  "margin",
  "padding",
  "border",
  "inline-size",
  "block-size",
  "width",
  "height",
  "min",
  "max",
  "aspect-ratio",
  "object-fit",
  "box-sizing",
  "overflow",
  "align",
  "justify",
  "place",
  "grid",
];
const NOT_BOX = new Set([
  "border-radius",
  "border-color",
  "border-top-color",
  "border-right-color",
  "border-bottom-color",
  "border-left-color",
  "border-image",
]);
function isBoxProp(prop) {
  if (NOT_BOX.has(prop)) return false;
  return BOX_PREFIXES.some((p) => prop === p || prop.startsWith(`${p}-`));
}
const QWEN_DIR = "models/qwen2-vl";
const PAGES = ["index.html", "basics/index.html", "practical/index.html"];
const read = (rel) => readFileSync(join(ROOT, rel), "utf8");

/** selector -> { prop: value }, whitespace collapsed and lowercased so formatting cannot fail a match. */
function cssRules(text) {
  const rules = new Map();
  for (const m of text.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const selector = m[1].replace(/\s+/g, " ").trim();
    if (!selector || selector.startsWith("@")) continue;
    const decls = new Map();
    for (const decl of m[2].split(";")) {
      const i = decl.indexOf(":");
      if (i < 0) continue;
      decls.set(
        decl.slice(0, i).trim().toLowerCase(),
        decl.slice(i + 1).replace(/\s+/g, " ").trim().toLowerCase(),
      );
    }
    if (decls.size) rules.set(selector, decls);
  }
  return rules;
}

/** QWEN_CSS is a template literal in the page's own module — the single source these pages mirror. */
function qwenCssRules() {
  const src = read(join(QWEN_DIR, "qwen2vl.js"));
  const m = src.match(/export const QWEN_CSS = `([\s\S]*?)`;/);
  assert.ok(m, "QWEN_CSS template literal must still be findable in models/qwen2-vl/qwen2vl.js");
  return cssRules(m[1]);
}

function boxStyle(page) {
  const src = read(join(QWEN_DIR, page));
  const m = src.match(/<style id="qwen2vl-sample-box">([\s\S]*?)<\/style>/);
  assert.ok(m, `${page} must carry the render-blocking <style id="qwen2vl-sample-box"> block`);
  return { src, rules: cssRules(m[1]) };
}

/** Every <img> in document order, with the attributes this test cares about. */
function imgs(src) {
  return [...src.matchAll(/<img\b[^>]*>/g)].map((m) => {
    const tag = m[0];
    const attr = (name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1];
    return {
      tag,
      class: attr("class") ?? "",
      src: attr("src") ?? "",
      width: attr("width"),
      height: attr("height"),
    };
  });
}

/** Intrinsic dimensions straight from the file header: PNG IHDR, or the JPEG SOFn segment. */
function intrinsicSize(rel) {
  const b = readFileSync(join(ROOT, rel));
  if (b.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
  }
  const SOF = new Set([
    0xc0,
    0xc1,
    0xc2,
    0xc3,
    0xc5,
    0xc6,
    0xc7,
    0xc9,
    0xca,
    0xcb,
    0xcd,
    0xce,
    0xcf,
  ]);
  let i = 2;
  while (i < b.length - 1) {
    if (b[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = b[i + 1];
    if (SOF.has(marker)) return { width: b.readUInt16BE(i + 7), height: b.readUInt16BE(i + 5) };
    if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;
      continue;
    }
    i += 2 + b.readUInt16BE(i + 2);
  }
  throw new Error(`no image dimensions found in ${rel}`);
}

test("every page copy of the box rules exists identically in QWEN_CSS", () => {
  const source = qwenCssRules();
  for (const page of PAGES) {
    const { rules } = boxStyle(page);
    assert.ok(rules.size > 0, `${page}: the reservation block must not be empty`);
    for (const [selector, decls] of rules) {
      const src = source.get(selector);
      assert.ok(
        src,
        `${page}: "${selector}" is not a QWEN_CSS selector — the mirrored rules must stay a subset`,
      );
      for (const [prop, value] of decls) {
        assert.equal(
          src.get(prop),
          value,
          `${page}: ${selector}{${prop}: ${value}} does not match QWEN_CSS (${src.get(prop)}) — ` +
            "fix models/qwen2-vl/qwen2vl.js and this copy together, or the box shifts twice",
        );
      }
      // The OTHER direction, for geometry: a box-affecting declaration added to QWEN_CSS and not mirrored
      // here shifts the box the moment the deferred sheet arrives. Review found this gap: the check below
      // did not exist, so `margin-bottom` added to QWEN_CSS's .sample-thumb slipped through with every
      // test still green. Raw declaration equality would close it too, at the cost of forcing the copy to
      // duplicate colour/focus rules as well; closure over the properties that move boxes is the intent.
      for (const [prop, value] of src) {
        if (!isBoxProp(prop)) continue;
        assert.equal(
          decls.get(prop),
          value,
          `${page}: QWEN_CSS ${selector}{${prop}: ${value}} is a box-affecting declaration this page does ` +
            "not mirror — mirror it in the head block, or the box shifts when the deferred CSS arrives",
        );
      }
    }
  }
});

test("the geometry that reserves the media box is present in every page", () => {
  // The columns, strip and image boxes are what move when the deferred CSS lands; colours and focus
  // styling deliberately stay in QWEN_CSS only.
  const required = {
    ".vlm-grid": ["display", "flex-wrap", "gap"],
    ".vlm-img-col": ["flex", "max-inline-size"],
    ".vlm-out-col": ["flex"],
    ".preview-img": ["max-inline-size", "max-block-size", "display"],
    ".sample-strip": ["display", "flex-wrap", "gap"],
    ".sample-thumb": ["inline-size", "block-size", "object-fit", "padding", "border"],
    // Found by review: .dropzone was mirrored in the block but absent from this schema, so deleting it
    // from a page left every test green while the dropzone - which sits between the strip and the
    // preview in the same column - lost its border and padding before the deferred CSS arrived.
    ".dropzone": ["border", "padding"],
  };
  for (const page of PAGES) {
    const { rules } = boxStyle(page);
    for (const [selector, props] of Object.entries(required)) {
      const decls = rules.get(selector);
      assert.ok(decls, `${page}: ${selector} must be reserved before the deferred CSS arrives`);
      for (const prop of props) {
        assert.ok(decls.has(prop), `${page}: ${selector} must declare ${prop}`);
      }
    }
  }
});

test("thumbs declare the box they are rendered in, with the cover crop preserved", () => {
  for (const page of PAGES) {
    const { src, rules } = boxStyle(page);
    const thumbs = imgs(src).filter((i) => i.class.split(/\s+/).includes("sample-thumb"));
    assert.ok(thumbs.length >= 3, `${page}: expected the sample thumbnails`);
    const thumb = rules.get(".sample-thumb");
    const cssWidth = Number.parseFloat(thumb.get("inline-size"));
    const cssHeight = Number.parseFloat(thumb.get("block-size"));
    // border-box is global (public/styles.css: "* { box-sizing: border-box }"), so the CSS box IS the
    // rendered box and the attributes must equal it, border included.
    assert.equal(
      cssWidth,
      76,
      `${page}: .sample-thumb inline-size must stay 76px to match the attributes`,
    );
    assert.equal(
      cssHeight,
      56,
      `${page}: .sample-thumb block-size must stay 56px to match the attributes`,
    );
    assert.equal(
      thumb.get("object-fit"),
      "cover",
      `${page}: the thumbs crop with object-fit:cover`,
    );
    let cropped = 0;
    assert.equal(
      qwenCssRules().get(".sample-thumb").get("object-fit"),
      "cover",
      "QWEN_CSS must keep the same crop so the two copies cannot disagree about distortion",
    );
    for (const t of thumbs) {
      assert.equal(t.width, "76", `${page}: ${t.src} must declare width="76" (its reserved box)`);
      assert.equal(t.height, "56", `${page}: ${t.src} must declare height="56" (its reserved box)`);
      // Reserving the CSS box rather than the intrinsic size is only correct because the thumbs CROP:
      // the assets run from 362x93 to 620x440, so at least one of them cannot share the 76:56 ratio.
      // If this ever becomes untrue the reserved box is wrong for some page and cover is hiding it.
      const asset = join(QWEN_DIR, t.src.replace(/^\.\.\//, ""));
      const { width, height } = intrinsicSize(asset);
      if (width * cssHeight === height * cssWidth) continue;
      cropped += 1;
    }
    assert.ok(
      cropped > 0,
      `${page}: cover must be doing real work here (no thumb shares the reserved 76:56 ratio)`,
    );
  }
});

test("each preview reserves its own initial asset's aspect ratio under the CSS cap", () => {
  for (const page of PAGES) {
    const { src, rules } = boxStyle(page);
    const previews = imgs(src).filter((i) => i.class.split(/\s+/).includes("preview-img"));
    assert.equal(previews.length, 1, `${page}: exactly one preview image`);
    const [preview] = previews;
    assert.ok(preview.src, `${page}: the preview must keep its initial src`);
    // Resolve the page-relative src against the page's own directory.
    const assetRel = join(QWEN_DIR, preview.src.replace(/^\.\.\//, ""));
    const { width, height } = intrinsicSize(assetRel);
    assert.equal(
      Number(preview.width),
      width,
      `${page}: preview width must be ${width} (${preview.src})`,
    );
    assert.equal(
      Number(preview.height),
      height,
      `${page}: preview height must be ${height} (${preview.src})`,
    );
    // Cross-multiplication states "same aspect ratio" without floating point: the attribute pair is a
    // ratio hint, and the rendered box is capped by the CSS in the same block.
    assert.equal(
      preview.width * height,
      preview.height * width,
      `${page}: preview attrs must preserve the asset ratio`,
    );
    const css = rules.get(".preview-img");
    assert.equal(
      css.get("max-inline-size"),
      "100%",
      `${page}: the preview is capped by its column`,
    );
    assert.equal(
      css.get("max-block-size"),
      "340px",
      `${page}: the preview cap must match QWEN_CSS`,
    );
    assert.equal(qwenCssRules().get(".preview-img").get("max-block-size"), "340px");
  }
});

test("no sample image in these pages is left without dimensions", () => {
  for (const page of PAGES) {
    const { src } = boxStyle(page);
    const media = imgs(src).filter((i) =>
      i.class.split(/\s+/).some((c) => c === "sample-thumb" || c === "preview-img")
    );
    assert.ok(media.length >= 4, `${page}: expected thumbs plus a preview`);
    for (const img of media) {
      assert.ok(
        img.width && img.height,
        `${page}: ${img.src} is missing width/height, which is the whole defect`,
      );
      assert.ok(
        Number(img.width) > 0 && Number(img.height) > 0,
        `${page}: ${img.src} must be positive`,
      );
    }
  }
});
