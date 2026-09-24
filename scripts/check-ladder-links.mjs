#!/usr/bin/env node
// Ladder-link gate — fail closed when a published demo advertises a link that does not resolve.
//
// Why this exists (web-ai-showcase-500): check-portfolio-acceptance.mjs validates ONE direction —
// every ON-DISK ladder route must be enumerated in that family's acceptance.json rungs. Nothing
// validated the other direction, so a page could advertise a rung that was never built.
// mms-tts-bengali shipped an overview card linking to multi-model/ with no such directory: a
// published 404 with every gate green. Same blind-spot class as wty (the conformance gate printed
// a failure count and exited 0) and i0h (CI ran no node tests): an artifact that looks like
// enforcement while the specific thing that broke is unenforced.
//
// What it enforces: for every built slug, every <a href> inside the family's own pages that
// resolves INSIDE models/<slug>/ must exist. A directory link must contain index.html; a file link
// must be a file. Out of scope by construction (never a failure): external URLs, data:/mailto:/
// tel:, site-absolute paths (/web-ai-showcase/…), links that resolve outside the family (e.g. a
// sibling family via ../), and hrefs containing template syntax ({{…}}, ${…}, <%), which cannot be
// resolved statically.
//
// Advisory, never written (criterion 4 of the bead): ladder rungs a family advertises but has not
// enumerated in its acceptance.json. This is a coverage/record gap, NOT a 404, and deliberately
// NOT auto-inserted into the manifests — writing rungs nobody ran would manufacture acceptance
// records, the exact failure mode the manifests exist to prevent. check-portfolio-acceptance
// remains the gate that enforces manifests for new and touched families.
//
// Modes:
//   node scripts/check-ladder-links.mjs            → enforce (exit 1 on any FAIL)
//   node scripts/check-ladder-links.mjs --root <dir> → run against a fixture tree (tests only)
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, normalize, relative, sep } from "node:path";

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const ROOT = normalize(flag("--root", new URL("..", import.meta.url).pathname));

/** Ladder rungs a demo may advertise, in the repo's own vocabulary. */
const RUNGS = new Set(["basics", "practical", "wild", "multi-model"]);

const loadCatalogue = (text) => {
  const j = JSON.parse(text);
  return Array.isArray(j) ? j : j.models ?? [];
};
const cataloguePath = join(ROOT, "models.json");
if (!existsSync(cataloguePath)) {
  console.error(`FAIL  models.json is missing at ${cataloguePath}`);
  process.exit(1);
}
const built = loadCatalogue(readFileSync(cataloguePath, "utf8"))
  .filter((e) => e.status === "built")
  .map((e) => e.slug)
  .sort();

const failures = [];
const advisories = []; // { slug, rung } — advertised but not enumerated
let pagesScanned = 0;
const inScopeLinks = []; // { slug, page, href, target }
const skipped = { externalOrSiteAbsolute: 0, templated: 0, outsideFamily: 0 };

/** Unresolvable statically: templates and non-path hrefs. */
const isTemplated = (href) => /\{\{|\$\{|<%|@@/.test(href);
const isNonPath = (href) =>
  href === "" || href.startsWith("#") || href.startsWith("/") ||
  /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(href);

const stripTarget = (href) => {
  const hash = href.indexOf("#");
  const query = href.indexOf("?");
  let end = href.length;
  if (hash >= 0) end = Math.min(end, hash);
  if (query >= 0) end = Math.min(end, query);
  return href.slice(0, end);
};

/** index.html files in the family tree = the pages a visitor can land on. */
const familyPages = (dir) => {
  const pages = [];
  const walk = (d) => {
    for (const ent of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (ent.name === "index.html") pages.push(p);
    }
  };
  walk(dir);
  return pages;
};

const rungOf = (href) => {
  const clean = stripTarget(href).replace(/^\.\//, "").replace(/\/+$/, "");
  return RUNGS.has(clean) ? clean : null;
};

/**
 * Rungs a family has enumerated in its acceptance.json.
 *
 * Returns null when the family has no manifest at all: pre-baseline legacy families are
 * grandfathered by check-portfolio-acceptance and must not become advisory noise here.
 */
const declaredRungs = (slug) => {
  const manifest = join(ROOT, "models", slug, "acceptance.json");
  if (!existsSync(manifest)) return null;
  try {
    const parsed = JSON.parse(readFileSync(manifest, "utf8"));
    const rungs = Array.isArray(parsed.rungs) ? parsed.rungs : [];
    const declared = new Set();
    for (const r of rungs) {
      const target = stripTarget(String(r.route ?? "")).replace(/^\/+/, "").replace(/\/+$/, "");
      const last = target.split("/").pop() ?? "";
      if (RUNGS.has(last)) declared.add(last);
    }
    return declared;
  } catch (e) {
    failures.push(`${slug}: acceptance.json does not parse — ${e.message}`);
    return null;
  }
};

for (const slug of built) {
  const dir = join(ROOT, "models", slug);
  if (!existsSync(dir)) {
    // check-routes.mjs owns "built route with no directory/index.html"; do not double-report.
    continue;
  }
  const declared = declaredRungs(slug);
  const advertised = new Set();

  for (const page of familyPages(dir)) {
    pagesScanned++;
    const html = readFileSync(page, "utf8");
    const pageRel = relative(ROOT, page).split(sep).join("/");
    for (const m of html.matchAll(/\bhref\s*=\s*("([^"]*)"|'([^']*)')/gi)) {
      const href = (m[2] ?? m[3] ?? "").trim();
      if (href === "") continue;
      if (isTemplated(href)) {
        skipped.templated++;
        continue;
      }
      if (isNonPath(href)) {
        skipped.externalOrSiteAbsolute++;
        continue;
      }
      const target = stripTarget(href);
      if (target === "" || target === ".") continue; // in-page or self

      const resolved = normalize(join(join(ROOT, "models", slug), decodeURIComponent(target)));
      const familyDir = join(ROOT, "models", slug);
      if (resolved !== familyDir && !resolved.startsWith(familyDir + sep)) {
        skipped.outsideFamily++; // e.g. ../sibling-family/ — another route owns that link
        continue;
      }

      inScopeLinks.push({ slug, page: pageRel, href, target: resolved });

      const rung = rungOf(href);
      if (rung) advertised.add(rung);

      // Classify before probing index.html: a directory with no index.html must read as that, not
      // as "missing" (probing it inside the same try would throw first and lose the reason).
      let exists = false;
      let kind = "missing";
      try {
        const st = statSync(resolved);
        if (st.isDirectory()) {
          exists = existsSync(join(resolved, "index.html"));
          if (!exists) kind = "directory without index.html";
        } else if (st.isFile()) {
          exists = true;
        }
      } catch {
        exists = false;
      }
      if (!exists) {
        failures.push(`${slug}: ${pageRel} advertises href="${href}" — ${kind}`);
      }
    }
  }

  for (const rung of advertised) {
    // Only families carrying a manifest can have a record gap; legacy families are out of scope.
    if (declared && !declared.has(rung)) advisories.push({ slug, rung });
  }
}

for (const f of failures) console.log(`FAIL  ${f}`);

if (advisories.length) {
  advisories.sort((a, b) => a.slug.localeCompare(b.slug) || a.rung.localeCompare(b.rung));
  console.log(
    `\nADVISORY  ${advisories.length} advertised-but-unenumerated rung(s): published rungs absent ` +
      "from the family's acceptance.json rungs. These are record gaps, not 404s — reported only, " +
      "never written into a manifest (an unrun rung must not gain an acceptance record).",
  );
  for (const a of advisories) console.log(`  ${a.slug}: ${a.rung}`);
}

console.log(
  `\nladder-links: ${pagesScanned} built page(s) scanned · ${inScopeLinks.length} in-family link(s) ` +
    `checked · ${skipped.outsideFamily} link(s) to other routes, ${skipped.externalOrSiteAbsolute} ` +
    `external/site-absolute, ${skipped.templated} templated (out of scope) · ` +
    `${advisories.length} advisory rung(s) · ${failures.length} failure(s)`,
);
if (!failures.length) {
  console.log("PASS  every advertised in-family link on a built route resolves");
}
process.exit(failures.length ? 1 : 0);
