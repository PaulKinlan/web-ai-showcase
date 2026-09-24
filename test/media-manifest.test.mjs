// Deterministic tests for media/manifest.json — the rights-safe media library record.
//
// Why this file exists (bead web-ai-showcase-1d6): the manifest sat on main with unresolved git
// conflict markers. It did not parse, lib/example-gallery.js fetched it, called r.json(), and
// rendered "Example gallery unavailable (...)" on 12 published routes — with ZERO console errors,
// because the gallery fails closed politely. Every gate stayed green and the node suite passed at
// 135/135, because nothing read this file. A corrupt manifest must never be able to land silently
// again.
//
// No browser, no network: the manifest, the schema and the filesystem are all in-repo.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

const raw = readFileSync("media/manifest.json", "utf8");
const schema = JSON.parse(readFileSync("media/manifest.schema.json", "utf8"));

// Parse once, but NEVER at module scope unguarded: a corrupt manifest would throw during import
// and abort the whole file, so the conflict-marker sweep below — the check that actually explains
// the breakage — would never run, and the operator would get a bare SyntaxError stack instead of a
// named failure. Capture the error and let each test report it properly.
let parsed = null;
let parseError = null;
try {
  parsed = JSON.parse(raw);
} catch (e) {
  parseError = e;
}

/** The parsed manifest, or a clear named failure if it does not parse. */
function doc() {
  assert.equal(
    parseError,
    null,
    `media/manifest.json does not parse: ${parseError && parseError.message}`,
  );
  return parsed;
}

// ── The failure that shipped ───────────────────────────────────────────────────────────────────

test("media/manifest.json parses as JSON", () => {
  // The whole bug in one assertion.
  assert.doesNotThrow(() => JSON.parse(raw), "media/manifest.json must be valid JSON");
});

test("no tracked text file contains git conflict markers", () => {
  // Repo-wide, not just the manifest: the marker class is what escaped review, and the sweep is
  // cheap. Uses git ls-files so untracked scratch files and node_modules are out of scope.
  const files = execSync("git ls-files", { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 })
    .split("\n")
    .filter((f) => /\.(json|mjs|js|ts|md|html|css|yml|yaml)$/.test(f));
  const offenders = [];
  for (const f of files) {
    let text;
    try {
      text = readFileSync(f, "utf8");
    } catch {
      continue;
    }
    // A conflict marker is only a marker at the start of a line.
    if (/^(<{7} |={7}$|>{7} )/m.test(text)) offenders.push(f);
  }
  assert.deepEqual(offenders, [], `unresolved conflict markers in: ${offenders.join(", ")}`);
});

// ── Schema conformance ─────────────────────────────────────────────────────────────────────────

test("top-level required fields are present", () => {
  const manifest = doc();
  for (const k of schema.required) assert.ok(k in manifest, `missing top-level "${k}"`);
  assert.ok(Array.isArray(manifest.assets) && manifest.assets.length > 0);
});

test("every image asset carries the required provenance fields", () => {
  const manifest = doc();
  const required = schema.properties.assets.items.required;
  const idPattern = new RegExp(schema.properties.assets.items.properties.id.pattern);
  for (const a of manifest.assets) {
    for (const k of required) assert.ok(k in a, `asset "${a.id}" missing "${k}"`);
    assert.match(a.id, idPattern, `asset id "${a.id}" is not kebab-case`);
    assert.ok(a.formats.jpg, `asset "${a.id}" has no jpg fallback`);
    // Provenance is the point of this file, so a blank license or creator is a failure, not a gap.
    assert.ok(a.license && a.license.length, `asset "${a.id}" has an empty license`);
    assert.ok(a.creator && a.creator.length, `asset "${a.id}" has no creator`);
    // Traceability uses the SAME rule as scripts/check-image-provenance.mjs:89 — a source URL, or
    // a documented composite. faces-crowd is a montage of the individually-licensed face-*
    // portraits, each of which carries its own sourceUrl, so it has no single upstream URL of its
    // own. Encoding the gate's real rule rather than a stricter one that the repo would have to
    // violate.
    const composite = a.id === "faces-crowd";
    assert.ok(
      (a.sourceUrl && a.sourceUrl.length) || composite,
      `asset "${a.id}" has no sourceUrl and is not a documented composite`,
    );
    if (composite) {
      assert.ok(a.attribution && a.attribution.length, `composite "${a.id}" has no attribution`);
    }
  }
});

test("asset ids are unique", () => {
  const ids = doc().assets.map((a) => a.id);
  const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
  assert.deepEqual([...new Set(dupes)], [], `duplicate asset ids: ${dupes.join(", ")}`);
});

test("count matches the real number of assets", () => {
  // The conflict left two competing counts (38 vs 37). Tie it to reality so it cannot drift again.
  const manifest = doc();
  assert.equal(
    manifest.count,
    manifest.assets.length,
    `count says ${manifest.count} but assets[] has ${manifest.assets.length}`,
  );
});

// ── The records must describe files that actually ship ─────────────────────────────────────────

test("every declared asset format file exists on disk", () => {
  const missing = [];
  for (const a of doc().assets) {
    for (const [fmt, f] of Object.entries(a.formats || {})) {
      if (!existsSync(`media/${f.path}`)) missing.push(`${a.id}.${fmt} -> media/${f.path}`);
    }
  }
  assert.deepEqual(missing, [], `manifest references missing files: ${missing.join(", ")}`);
});

test("audioAssets (non-image samples) are well-formed and their files exist", () => {
  // Audio samples ship beside their demo rather than under media/assets/, so they carry
  // localPath + dims instead of the avif/webp/jpg triple and cannot live in assets[].
  const required = schema.properties.audioAssets.items.required;
  for (const a of doc().audioAssets ?? []) {
    for (const k of required) assert.ok(k in a, `audio asset "${a.id}" missing "${k}"`);
    assert.ok(existsSync(a.localPath), `audio asset "${a.id}" -> ${a.localPath} does not exist`);
    assert.ok(!("formats" in a), `audio asset "${a.id}" should not declare image formats`);
  }
});

test("no id collides between assets and audioAssets", () => {
  const manifest = doc();
  const imageIds = new Set(manifest.assets.map((a) => a.id));
  for (const a of manifest.audioAssets ?? []) {
    assert.ok(!imageIds.has(a.id), `id "${a.id}" appears in both assets[] and audioAssets[]`);
  }
});

// ── The consumer contract ──────────────────────────────────────────────────────────────────────

test("the gallery's filter still yields assets (the shape lib/example-gallery.js relies on)", () => {
  // createExampleGallery keeps only assets with formats.jpg || formats.webp. If a future edit
  // reshapes the manifest, this fails here instead of rendering an empty gallery on 12 routes.
  const manifest = doc();
  const usable = manifest.assets.filter((a) => a && a.formats && (a.formats.jpg || a.formats.webp));
  assert.ok(usable.length > 0, "no asset survives the example-gallery filter");
  assert.equal(
    usable.length,
    manifest.assets.length,
    "some assets would be silently dropped by the gallery filter",
  );
});
