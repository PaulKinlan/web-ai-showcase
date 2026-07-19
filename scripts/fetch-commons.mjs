// One-off curation helper (not a gate). Searches Wikimedia Commons for varied,
// clearly-licensed source images for the 2D->3D / removable-object demo family,
// verifies license + author via the API, pulls a web-sized rendering of the
// licensed original, and prints a provenance record per accepted asset.
// Usage: node scripts/fetch-commons.mjs
import { mkdirSync, writeFileSync } from "node:fs";

const UA =
  "web-ai-showcase-media-curation/1.0 (https://github.com/PaulKinlan/web-ai-showcase; paul.kinlan@gmail.com)";
const API = "https://commons.wikimedia.org/w/api.php";
const OUT = new URL("../media/assets/", import.meta.url);
mkdirSync(OUT, { recursive: true });

// SPDX mapping for the licenses we accept. Anything else is skipped (no bundling
// without a clear, redistributable license).
const SPDX = {
  "cc0": "CC0-1.0",
  "pd": "Public-Domain",
  "cc-by-2.0": "CC-BY-2.0",
  "cc-by-2.5": "CC-BY-2.5",
  "cc-by-3.0": "CC-BY-3.0",
  "cc-by-4.0": "CC-BY-4.0",
  "cc-by-sa-2.0": "CC-BY-SA-2.0",
  "cc-by-sa-2.5": "CC-BY-SA-2.5",
  "cc-by-sa-3.0": "CC-BY-SA-3.0",
  "cc-by-sa-4.0": "CC-BY-SA-4.0",
};
function toSpdx(licenseField, shortName) {
  if (!licenseField) {
    const s = (shortName || "").toLowerCase();
    if (s.includes("public domain") || s === "pd") return "Public-Domain";
    if (s.includes("cc0")) return "CC0-1.0";
    return null;
  }
  const k = licenseField.toLowerCase().trim();
  if (SPDX[k]) return SPDX[k];
  if (k.startsWith("pd") || k.includes("public")) return "Public-Domain";
  if (k.startsWith("cc0")) return "CC0-1.0";
  return null;
}
const stripHtml = (s) => (s || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

async function api(params) {
  const u = new URL(API);
  for (const [k, v] of Object.entries({ format: "json", origin: "*", ...params })) {
    u.searchParams.set(k, v);
  }
  const r = await fetch(u, { headers: { "User-Agent": UA } });
  if (!r.ok) throw new Error(`API ${r.status} for ${JSON.stringify(params)}`);
  return r.json();
}

// slug -> {query, desc(one-line), family-relevance}. Deliberately varied.
const WANT = [
  {
    slug: "portrait-person",
    q: "portrait woman face studio",
    note: "human portrait — matting / person segmentation",
  },
  {
    slug: "city-street",
    q: "city street scene pedestrians cars",
    note: "busy street — panoptic segmentation / monocular depth",
  },
  {
    slug: "fruit-still-life",
    q: "fresh fruit still life",
    note: "objects on a surface — instance segmentation / object removal",
  },
  {
    slug: "dog-outdoor",
    q: "dog portrait grass",
    note: "single animal — foreground/background matting",
  },
  {
    slug: "mountain-landscape",
    q: "mountain valley landscape",
    note: "deep scene — monocular depth estimation",
  },
  {
    slug: "wood-texture",
    q: "wood plank texture surface",
    note: "flat repeating texture — a hard case for depth/segmentation",
  },
  {
    slug: "handwritten-document",
    q: "handwritten letter manuscript page",
    note: "text-heavy document — OCR / document layout",
  },
  {
    slug: "statistics-chart",
    q: "bar chart statistics diagram",
    note: "synthetic chart — a non-photographic edge case",
  },
  {
    slug: "night-city",
    q: "city skyline night lights",
    note: "low-light night scene — challenging illumination",
  },
  {
    slug: "backlit-silhouette",
    q: "silhouette sunset backlit person",
    note: "backlit silhouette — hard hair/edge matting",
  },
  {
    slug: "glass-transparency",
    q: "drinking glass water transparent",
    note: "transparent object — a known matting/segmentation failure case",
  },
  {
    slug: "market-crowd",
    q: "market crowd people stalls",
    note: "many overlapping people/objects — panoptic + detection",
  },
  {
    slug: "bicycle-object",
    q: "bicycle leaning wall",
    note: "thin-structured object — SAM point/box prompting",
  },
  {
    slug: "houseplant",
    q: "potted plant indoor windowsill",
    note: "foreground plant — CLIPSeg text-prompted segmentation",
  },
];

const MAXW = 1600; // web-sized rendering width
const results = [];
const skipped = [];

for (const item of WANT) {
  try {
    // generator=search over File namespace, pull imageinfo + extmetadata for candidates.
    const data = await api({
      action: "query",
      generator: "search",
      gsrsearch: `filetype:bitmap ${item.q}`,
      gsrnamespace: "6",
      gsrlimit: "12",
      prop: "imageinfo",
      iiprop: "url|size|extmetadata|mime",
      iiurlwidth: String(MAXW),
    });
    const pages = Object.values(data?.query?.pages || {});
    let chosen = null;
    for (const p of pages) {
      const ii = p.imageinfo?.[0];
      if (!ii) continue;
      if (!/^image\/(jpeg|png)$/.test(ii.mime || "")) continue;
      if ((ii.width || 0) < 900) continue;
      const em = ii.extmetadata || {};
      const spdx = toSpdx(em.License?.value, em.LicenseShortName?.value);
      if (!spdx) continue;
      const author = stripHtml(em.Artist?.value) || "Unknown";
      if (/^unknown$/i.test(author) && spdx !== "Public-Domain" && spdx !== "CC0-1.0") continue;
      chosen = { p, ii, em, spdx, author };
      break;
    }
    if (!chosen) {
      skipped.push({
        slug: item.slug,
        reason: "no candidate with an accepted, attributable license in top results",
      });
      continue;
    }
    const { p, ii, em, spdx, author } = chosen;
    const thumbUrl = ii.thumburl || ii.url;
    const buf = Buffer.from(
      await (await fetch(thumbUrl, { headers: { "User-Agent": UA } })).arrayBuffer(),
    );
    const ext = (ii.mime === "image/png") ? "png" : "jpg";
    const srcPath = new URL(`${item.slug}.src.${ext}`, OUT);
    writeFileSync(srcPath, buf);
    results.push({
      slug: item.slug,
      note: item.note,
      srcFile: `${item.slug}.src.${ext}`,
      filePage: `https://commons.wikimedia.org/wiki/${
        encodeURIComponent(p.title.replace(/ /g, "_"))
      }`,
      title: p.title,
      author,
      license: spdx,
      licenseShort: stripHtml(em.LicenseShortName?.value) || spdx,
      licenseUrl: em.LicenseUrl?.value || "",
      description: stripHtml(em.ImageDescription?.value).slice(0, 200) || item.note,
      credit: stripHtml(em.Credit?.value).slice(0, 160) || "",
      renderedWidth: ii.thumbwidth || ii.width,
      renderedHeight: ii.thumbheight || ii.height,
      originalWidth: ii.width,
      originalHeight: ii.height,
      descriptionUrl: ii.descriptionurl || "",
    });
    process.stderr.write(`OK  ${item.slug}  <- ${p.title}  [${spdx}]  by ${author.slice(0, 40)}\n`);
  } catch (e) {
    skipped.push({ slug: item.slug, reason: String(e.message || e) });
    process.stderr.write(`ERR ${item.slug}: ${e.message}\n`);
  }
}

writeFileSync(
  new URL("../media/_commons-fetch.json", import.meta.url),
  JSON.stringify({ results, skipped }, null, 2),
);
process.stderr.write(`\nDONE  accepted=${results.length}  skipped=${skipped.length}\n`);
