// One-off: optimize every source asset in media/assets/ to modern formats
// (avif + webp + jpg fallback), then emit the git-tracked media/manifest.json
// with full per-asset provenance. Sources come from the two fetch helpers.
// Usage: node scripts/build-media-manifest.mjs
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";

const MEDIA = new URL("../media/", import.meta.url);
const ASSETS = new URL("assets/", MEDIA);
const MAXW = 1600;
const today = new Date().toISOString().slice(0, 10);
const sh = (cmd, args) => execFileSync(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
const dims = (f) => {
  const out = sh("magick", ["identify", "-format", "%w %h", f]).toString().trim().split(/\s+/);
  return { width: +out[0], height: +out[1] };
};
const kb = (f) => Math.round(statSync(f).size / 1024);

const commons = JSON.parse(readFileSync(new URL("_commons-fetch.json", MEDIA))).results;
const shots = JSON.parse(readFileSync(new URL("_screenshots-fetch.json", MEDIA))).results;

function optimize(slug, srcFile) {
  const src = new URL(srcFile, ASSETS).pathname;
  const jpg = new URL(`${slug}.jpg`, ASSETS).pathname;
  const webp = new URL(`${slug}.webp`, ASSETS).pathname;
  const avif = new URL(`${slug}.avif`, ASSETS).pathname;
  // Downscale to <=MAXW (never upscale), strip metadata, sRGB.
  const resize = ["-strip", "-colorspace", "sRGB", "-resize", `${MAXW}x${MAXW}>`];
  sh("magick", [src, ...resize, "-quality", "82", jpg]);
  sh("magick", [src, ...resize, "-quality", "80", "-define", "webp:method=6", webp]);
  sh("magick", [src, ...resize, "-quality", "52", avif]);
  rmSync(src); // drop the un-optimized source; the optimized set is what ships
  const d = dims(jpg);
  return {
    formats: {
      avif: { path: `assets/${slug}.avif`, bytesKB: kb(avif) },
      webp: { path: `assets/${slug}.webp`, bytesKB: kb(webp) },
      jpg: { path: `assets/${slug}.jpg`, bytesKB: kb(jpg) },
    },
    width: d.width,
    height: d.height,
  };
}

const assets = [];

for (const r of commons) {
  const opt = optimize(r.slug, r.srcFile);
  assets.push({
    id: r.slug,
    description: r.description || r.note,
    familyRelevance: r.note,
    source: "Wikimedia Commons",
    sourceUrl: r.filePage,
    creator: r.author,
    license: r.license,
    licenseName: r.licenseShort,
    licenseUrl: r.licenseUrl,
    attribution: `${r.author} — ${
      r.title.replace(/^File:/, "")
    } — via Wikimedia Commons — ${r.licenseShort}`,
    retrieved: today,
    ...opt,
    original: {
      width: r.originalWidth,
      height: r.originalHeight,
      note: "rendered from the licensed Commons original",
    },
  });
}

for (const s of shots) {
  const opt = optimize(s.slug, s.srcFile);
  assets.push({
    id: s.slug,
    description: s.desc,
    familyRelevance: s.note,
    source: "Headless Chrome screenshot",
    sourceUrl: s.url,
    creator: s.owner,
    license: s.license,
    licenseName: s.license,
    licenseUrl: s.licenseUrl || "",
    attribution: `Screenshot of ${s.url} — owner: ${s.owner} — ${s.license}`,
    retrieved: s.retrieved,
    ...opt,
    capture: { tool: "google-chrome-stable --headless=new", viewport: s.viewport },
  });
}

// Hand-curated arrays this script cannot derive from media/assets/: `audioAssets` (rights-cleared
// non-image samples that ship beside their demo, e.g. models/<slug>/sample.wav) and `skipped` (the
// recorded license gaps). The manifest object is rebuilt from scratch on every run, so anything not
// carried across here is DELETED — silently, because nothing downstream reads these two. Carry them
// over from the committed manifest rather than dropping provenance for a shipped file.
const prevPath = new URL("manifest.json", MEDIA);
let prev = {};
if (existsSync(prevPath)) {
  try {
    prev = JSON.parse(readFileSync(prevPath, "utf8"));
  } catch (e) {
    // Refuse to regenerate over an unreadable manifest: that is how a hand-curated record gets
    // erased without anyone noticing (see bead web-ai-showcase-1d6 — unresolved conflict markers
    // made this file unparseable and broke 12 galleries for weeks with zero console errors).
    throw new Error(
      `media/manifest.json exists but does not parse (${e.message}). Fix it before regenerating — ` +
        `otherwise audioAssets/skipped would be silently dropped.`,
    );
  }
}

const manifest = {
  $schema: "./manifest.schema.json",
  name: "web-ai-showcase rights-safe media library",
  description:
    "Curated, clearly-licensed test images for the 2D->3D / removable-object model family " +
    "(depth, matting, segmentation, SAM, CLIPSeg, panoptic). Every asset is stored locally " +
    "(no hotlinking), optimized to avif/webp/jpg, and carries full provenance. See README.md.",
  provenanceRules: [
    "Every asset records: source URL, creator, license (SPDX or explicit), retrieval date, local path(s), dimensions, one-line description.",
    "Only clearly-licensed assets are bundled. If a license is unclear, the asset is skipped and the gap is noted.",
    "Assets are stored LOCALLY under media/assets/ — demos must never hotlink the original source.",
    "CC-BY / CC-BY-SA assets must be shown with the recorded attribution when displayed.",
  ],
  generated: today,
  count: assets.length,
  assets,
  ...(prev.audioAssets ? { audioAssets: prev.audioAssets } : {}),
  ...(prev.skipped ? { skipped: prev.skipped } : {}),
};

writeFileSync(new URL("manifest.json", MEDIA), JSON.stringify(manifest, null, 2) + "\n");
// tidy the intermediate fetch logs
for (const f of ["_commons-fetch.json", "_screenshots-fetch.json"]) {
  const p = new URL(f, MEDIA);
  if (existsSync(p)) rmSync(p);
}
process.stderr.write(`manifest.json written — ${assets.length} assets\n`);
