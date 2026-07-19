// One-off curation helper: capture web-page screenshots via headless Chrome for the
// media library (varied web UI as segmentation / OCR / document-layout test inputs).
// Records provenance (source URL, owner, license, retrieval date) per shot.
// Usage: node scripts/fetch-screenshots.mjs
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";

const CHROME = ["/usr/bin/google-chrome-stable", "/usr/bin/google-chrome", "/usr/bin/chromium"]
  .find(existsSync);
if (!CHROME) throw new Error("no chrome binary");
const OUT = new URL("../media/assets/", import.meta.url);
mkdirSync(OUT, { recursive: true });

// Owner/license per source. Paul's own sites: he holds the rights (bundled with
// permission — his own project). Wikipedia article body text is CC BY-SA 4.0
// (site chrome / logos remain the Wikimedia Foundation's marks).
const SHOTS = [
  {
    slug: "screenshot-paul-home",
    url: "https://paul.kinlan.me/",
    owner: "Paul Kinlan",
    license: "Proprietary-OwnerPermission",
    note: "personal blog homepage — web-UI layout / text region segmentation",
    desc: "Screenshot of paul.kinlan.me homepage (site owner: Paul Kinlan).",
  },
  {
    slug: "screenshot-paul-post",
    url:
      "https://paul.kinlan.me/2025-05-02-from-pytorch-to-browser-creating-a-web-friendly-ai-model/",
    owner: "Paul Kinlan",
    license: "Proprietary-OwnerPermission",
    note: "text- and code-heavy article page — OCR / document layout",
    desc: "Screenshot of a paul.kinlan.me blog post (site owner: Paul Kinlan).",
  },
  {
    slug: "screenshot-wikipedia-article",
    url: "https://en.wikipedia.org/wiki/Depth_map",
    owner: "Wikipedia contributors",
    license: "CC-BY-SA-4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    note: "encyclopedia article — dense text + figures for OCR / layout / segmentation",
    desc:
      "Screenshot of the English Wikipedia 'Depth map' article (article text CC BY-SA 4.0; logos are WMF marks).",
  },
];

const today = new Date().toISOString().slice(0, 10);
const results = [];
for (const s of SHOTS) {
  const png = new URL(`${s.slug}.src.png`, OUT);
  try {
    execFileSync(CHROME, [
      "--headless=new",
      "--no-sandbox",
      "--hide-scrollbars",
      "--force-color-profile=srgb",
      "--window-size=1280,900",
      "--virtual-time-budget=9000",
      `--screenshot=${png.pathname}`,
      s.url,
    ], { stdio: "ignore", timeout: 60000 });
    if (!existsSync(png) || statSync(png).size < 1000) throw new Error("empty screenshot");
    results.push({ ...s, srcFile: `${s.slug}.src.png`, retrieved: today, viewport: "1280x900" });
    process.stderr.write(`OK  ${s.slug}  <- ${s.url}  (${statSync(png).size} bytes)\n`);
  } catch (e) {
    process.stderr.write(`ERR ${s.slug}: ${e.message}\n`);
  }
}
writeFileSync(
  new URL("../media/_screenshots-fetch.json", import.meta.url),
  JSON.stringify({ results }, null, 2),
);
process.stderr.write(`\nDONE screenshots=${results.length}\n`);
