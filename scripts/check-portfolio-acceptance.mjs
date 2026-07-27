#!/usr/bin/env node
// Portfolio acceptance gate (integrator-owned, ratchet from be427da "Require route-complete
// Web AI acceptance"). The model-level check-conformance gate cannot see a family's ROUTE
// completeness: the Manga OCR incident shipped a green family validator that never exercised the
// ladder routes. This gate fails closed on any NEW or TOUCHED built family since the baseline
// unless:
//   1. models/<slug>/acceptance.json declares its committed validator, the full route matrix
//      (overview WITH inside:true + every on-disk ladder route) at desktop AND mobile, and every
//      advertised model stage;
//   2. the committed validator source actually opens every enumerated route, drives DESKTOP and
//      MOBILE viewports via setViewport, and names every advertised stage (full HF id or its
//      model-name part);
//   3. a committed run record (models/<slug>/acceptance-run.json) proves the validator's latest
//      passing run against the family's current commit, with a passing desktop+mobile result per
//      enumerated route.
// Legacy families published at the baseline are out of scope — nothing retroactively marks the
// 294 passed. A family touched after the baseline (any programme) enters scope. The six existing
// gates run first; this runs among them via `deno task gate`.
//
// Modes:
//   node scripts/check-portfolio-acceptance.mjs                 → enforce (exit 1 on any FAIL)
//   node scripts/check-portfolio-acceptance.mjs --write-baseline → regenerate the fallback
//                                                                  baseline file from BASELINE_SHA
// Test hooks (fixtures only, never set in CI): --root <dir> --baseline <sha>
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_BASELINE = "be427dacc1ff3580d39bec3de24a9e78933d1306";
const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const ROOT = flag("--root", new URL("..", import.meta.url).pathname);
const BASELINE_SHA = flag("--baseline", DEFAULT_BASELINE);
const WRITE_BASELINE = argv.includes("--write-baseline");

const git = (args, allowFail = false) => {
  try {
    return execFileSync("git", args, {
      cwd: ROOT,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    }).trim();
  } catch (e) {
    if (allowFail) return "";
    throw e;
  }
};

const loadCatalogue = (text) => {
  const j = JSON.parse(text);
  return Array.isArray(j) ? j : j.models ?? [];
};
const builtSlugs = (entries) =>
  new Set(entries.filter((e) => e.status === "built").map((e) => e.slug));

const headCatalogue = loadCatalogue(readFileSync(join(ROOT, "models.json"), "utf8"));
const builtNow = builtSlugs(headCatalogue);

// Baseline = families published at the ratchet commit. Prefer git (authoritative); fall back to
// the committed snapshot so fresh clones without full history still run the gate.
let baseline;
const baselineText = git(["show", `${BASELINE_SHA}:models.json`], true);
if (baselineText) {
  baseline = builtSlugs(loadCatalogue(baselineText));
} else {
  const snap = JSON.parse(
    readFileSync(join(ROOT, ".portfolio-acceptance-baseline.json"), "utf8"),
  );
  baseline = new Set(snap.slugs);
}

if (WRITE_BASELINE) {
  writeFileSync(
    join(ROOT, ".portfolio-acceptance-baseline.json"),
    JSON.stringify(
      {
        baseline: BASELINE_SHA,
        generatedAt: new Date().toISOString(),
        legacyFamilies: baseline.size,
        slugs: [...baseline].sort(),
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`baseline written: ${baseline.size} legacy families at ${BASELINE_SHA.slice(0, 7)}`);
  process.exit(0);
}

const HEAD = git(["rev-parse", "HEAD"]);
const familyPaths = (slug, validator) =>
  [`models/${slug}`, validator ?? `scripts/validate-${slug}.mjs`].filter(Boolean);

// In scope: built families that are NEW since the baseline, or legacy families TOUCHED after it
// (committed diff OR uncommitted working-tree changes to the family or its validator).
const newSlugs = [...builtNow].filter((s) => !baseline.has(s));
const touchedSlugs = [...baseline].filter((s) => {
  if (!builtNow.has(s)) return false;
  const diff = git(
    ["diff", "--name-only", BASELINE_SHA, HEAD, "--", `models/${s}`, `scripts/validate-${s}.mjs`],
    true,
  );
  const dirty = git(
    ["status", "--porcelain", "--", `models/${s}`, `scripts/validate-${s}.mjs`],
    true,
  );
  return diff.length > 0 || dirty.length > 0;
});
const inScope = [...new Set([...newSlugs, ...touchedSlugs])].sort();

// Advertised model stages = model ids the family's code actually loads. Targeted contexts only
// (loader/pipeline call sites), never free-text URL matching.
const STAGE_PATTERNS = [
  /\bmodelId\s*:\s*["'`]([A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+)["'`]/g,
  /\bmodel\s*:\s*["'`]([A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+)["'`]/g,
  /\bpipeline\(\s*["'`][^"'`]*["'`]\s*,\s*["'`]([A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+)["'`]/g,
  /\bfrom_?pretrained\(\s*["'`]([A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+)["'`]/gi,
];
const extractStages = (dir) => {
  const found = new Set();
  const walk = (d) => {
    for (const ent of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (/\.(js|mjs|html)$/.test(ent.name)) {
        const src = readFileSync(p, "utf8");
        for (const re of STAGE_PATTERNS) {
          re.lastIndex = 0;
          let m;
          while ((m = re.exec(src))) found.add(m[1]);
        }
      }
    }
  };
  walk(dir);
  return [...found].sort();
};

const errors = [];
const checkSlug = (slug) => {
  const dir = join(ROOT, "models", slug);
  const manifestRel = `models/${slug}/acceptance.json`;
  const startLen = errors.length;
  if (!existsSync(join(dir, "acceptance.json"))) {
    errors.push(
      `${slug}: missing ${manifestRel} — new/touched families must declare their route-complete acceptance matrix (validator, rungs x viewports, stages, run record)`,
    );
    return;
  }
  let mf;
  try {
    mf = JSON.parse(readFileSync(join(dir, "acceptance.json"), "utf8"));
  } catch (e) {
    errors.push(`${slug}: ${manifestRel} does not parse — ${e.message}`);
    return;
  }
  if (mf.slug && mf.slug !== slug) {
    errors.push(`${slug}: manifest slug "${mf.slug}" does not match the family slug`);
  }
  if (!mf.validator || !existsSync(join(ROOT, mf.validator))) {
    errors.push(`${slug}: committed validator "${mf.validator ?? "(unset)"}" not found`);
    return;
  }
  const vsrc = readFileSync(join(ROOT, mf.validator), "utf8");

  // On-disk routes are the source of truth: overview + every subdir with an index.html.
  const diskRoutes = [
    `models/${slug}/`,
    ...readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && existsSync(join(dir, d.name, "index.html")))
      .map((d) => `models/${slug}/${d.name}/`),
  ].sort();
  const rungs = Array.isArray(mf.rungs) ? mf.rungs : [];
  const mfRoutes = rungs.map((r) => r.route);
  for (const r of diskRoutes) {
    if (!mfRoutes.includes(r)) {
      errors.push(`${slug}: on-disk route ${r} is not enumerated in acceptance.json rungs`);
    }
  }
  for (const r of mfRoutes) {
    if (!diskRoutes.includes(r)) {
      errors.push(`${slug}: acceptance.json enumerates ${r}, which does not exist on disk`);
    }
  }
  const overview = rungs.find((r) => r.route === `models/${slug}/`);
  if (!overview) {
    errors.push(`${slug}: rungs must include the overview route models/${slug}/`);
  } else if (overview.inside !== true) {
    errors.push(
      `${slug}: overview rung must declare "inside": true — the see-inside surface is exercised on the overview route`,
    );
  }
  for (const r of rungs) {
    const vp = Array.isArray(r.viewports) ? r.viewports : [];
    for (const need of ["desktop", "mobile"]) {
      if (!vp.includes(need)) {
        errors.push(`${slug}: rung ${r.route} does not declare the ${need} viewport`);
      }
    }
  }

  // The validator must open every enumerated route (not just mention the family).
  for (const r of diskRoutes) {
    if (r === `models/${slug}/`) {
      // Sub-route strings contain the overview prefix, so remove them before testing for an
      // independent overview open.
      const stripped = diskRoutes
        .filter((x) => x !== r)
        .reduce((s, x) => s.split(x).join(""), vsrc);
      if (!stripped.includes(`models/${slug}/`)) {
        errors.push(`${slug}: validator never opens the overview route models/${slug}/`);
      }
    } else if (!vsrc.includes(r) && !vsrc.includes(`/${r}`)) {
      errors.push(`${slug}: validator never opens route ${r}`);
    }
  }
  if (!/\bMOBILE\b/.test(vsrc)) {
    errors.push(`${slug}: validator never drives the MOBILE viewport (scripts/browser.mjs)`);
  }
  if (!/\bDESKTOP\b/.test(vsrc) || !/\bsetViewport\b/.test(vsrc)) {
    errors.push(
      `${slug}: validator must exercise viewports via setViewport(DESKTOP|MOBILE) from scripts/browser.mjs`,
    );
  }

  // Stages: every model id the family loads must be declared, and the validator must name each
  // declared stage (full HF id or its model-name part, case-insensitive).
  const stages = Array.isArray(mf.stages) ? mf.stages : [];
  for (const s of extractStages(dir)) {
    if (!stages.includes(s)) {
      errors.push(`${slug}: advertised model stage "${s}" is missing from acceptance.json stages`);
    }
  }
  const vsrcLower = vsrc.toLowerCase();
  for (const s of stages) {
    const name = String(s).split("/").pop().toLowerCase();
    if (!vsrcLower.includes(String(s).toLowerCase()) && !vsrcLower.includes(name)) {
      errors.push(`${slug}: validator never names advertised stage "${s}"`);
    }
  }

  // Run record: committed proof the validator passed against the family's CURRENT commit.
  const rrRel = mf.runRecord || `models/${slug}/acceptance-run.json`;
  if (!existsSync(join(ROOT, rrRel))) {
    errors.push(
      `${slug}: missing run record ${rrRel} — commit the validator's latest passing run`,
    );
  } else {
    let rr;
    try {
      rr = JSON.parse(readFileSync(join(ROOT, rrRel), "utf8"));
    } catch (e) {
      errors.push(`${slug}: run record ${rrRel} does not parse — ${e.message}`);
    }
    if (rr) {
      if (rr.exitCode !== 0) {
        errors.push(`${slug}: run record exitCode is ${rr.exitCode}, not 0`);
      }
      const dirty = git(["status", "--porcelain", "--", ...familyPaths(slug, mf.validator)], true);
      // Staleness is measured against the family's real code/routes/validator, EXCLUDING the
      // acceptance meta files — committing the run record itself must not move the target.
      const latest = git(
        [
          "log",
          "-n1",
          "--format=%H",
          "HEAD",
          "--",
          ...familyPaths(slug, mf.validator),
          `:(exclude)${manifestRel}`,
          `:(exclude)${rrRel}`,
        ],
        true,
      );
      if (dirty) {
        errors.push(
          `${slug}: uncommitted family/validator changes — the run record cannot be current; commit, re-run the validator, refresh the record`,
        );
      } else if (latest && rr.commit !== latest) {
        errors.push(
          `${slug}: run record commit ${
            String(rr.commit).slice(0, 7)
          } is stale vs latest family commit ${
            latest.slice(0, 7)
          } — re-run the validator and refresh the record`,
        );
      }
      const results = Array.isArray(rr.results) ? rr.results : [];
      for (const r of rungs) {
        for (const vp of ["desktop", "mobile"]) {
          const hit = results.find(
            (x) => x.route === r.route && x.viewport === vp && x.pass === true,
          );
          if (!hit) {
            errors.push(`${slug}: run record lacks a passing ${vp} result for ${r.route}`);
          }
        }
      }
    }
  }
  return errors.length === startLen;
};

console.log(
  `portfolio-acceptance: baseline ${
    BASELINE_SHA.slice(0, 7)
  } — ${baseline.size} legacy families out of scope (never retroactively passed)`,
);
if (inScope.length === 0) {
  console.log("PASS  no new or touched built families since the baseline");
}
for (const slug of inScope) {
  if (checkSlug(slug)) {
    const kind = newSlugs.includes(slug) ? "new" : "touched";
    console.log(
      `PASS  ${slug} (${kind}): route-complete manifest + validator + current run record`,
    );
  }
}
for (const e of errors) console.log(`FAIL  ${e}`);
console.log(
  `portfolio-acceptance: ${inScope.length} in scope (${newSlugs.length} new, ${touchedSlugs.length} touched) · ${errors.length} failure(s)`,
);
process.exit(errors.length ? 1 : 0);
