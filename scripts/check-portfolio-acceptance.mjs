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

// --- Stage-literal hardening (review finding: non-literal model ids evade extraction) --------
// Stage ids are extracted from loader call-site string LITERALS, so a non-literal model
// argument (const, variable, expression — e.g. `pipeline("task", M)` or `modelId: cfg.model`)
// loads an undeclared, unnamed model straight past the gate. In-scope families must therefore
// declare the model id as a string literal AT the call site (the repo convention across the 294
// legacy families), and every declared stage must be referenced in the family sources.
// Detection is offset-preserving and regex-based like the rest of this gate: HTML prose,
// comments, and string CONTENTS are blanked first so literal call sites, comments, and
// non-loader identifiers never trip it.

const familySourceFiles = (dir) => {
  const files = [];
  const walk = (d) => {
    for (const ent of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (/\.(js|mjs|html)$/.test(ent.name)) {
        files.push({
          rel: p.startsWith(ROOT) ? p.slice(ROOT.length).replace(/^\//, "") : p,
          html: ent.name.endsWith(".html"),
          src: readFileSync(p, "utf8"),
        });
      }
    }
  };
  walk(dir);
  return files;
};

// HTML: blank everything except <script> bodies (prose can legitimately contain "model: ...").
const blankNonScript = (src) => {
  const keep = new Uint8Array(src.length);
  const re = /<script\b[^>]*>([\s\S]*?)<\/script\s*>/gi;
  let m;
  while ((m = re.exec(src))) {
    if (m[1].length === 0) continue;
    const start = m.index + m[0].indexOf(m[1]);
    for (let i = start; i < start + m[1].length; i++) keep[i] = 1;
  }
  return src.split("").map((ch, i) => (keep[i] || ch === "\n" ? ch : " ")).join("");
};

// Blank comments and string/template CONTENTS (the quotes stay, so a literal argument is still
// recognisable as a literal). Newlines and offsets are preserved for snippet reporting.
const blankStringsAndComments = (src) => {
  const out = src.split("");
  const n = src.length;
  let i = 0;
  while (i < n) {
    const ch = src[i];
    if (ch === "/" && src[i + 1] === "/") {
      while (i < n && src[i] !== "\n") out[i++] = " ";
    } else if (ch === "/" && src[i + 1] === "*") {
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) {
        if (src[i] !== "\n") out[i] = " ";
        i++;
      }
      for (let k = 0; k < 2 && i < n; k++, i++) if (src[i] !== "\n") out[i] = " ";
    } else if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      i++;
      while (i < n && src[i] !== quote) {
        if (quote !== "`" && src[i] === "\n") break; // unterminated single-line string
        if (src[i] !== "\n") out[i] = " ";
        i++;
        if (src[i - 1] === "\\" && i < n) { // escaped char is part of the string
          if (src[i] !== "\n") out[i] = " ";
          i++;
        }
      }
      i++; // keep the closing quote
    } else {
      i++;
    }
  }
  return out.join("");
};

// Non-literal model arguments at loader call sites. Matched against the blanked scan source, so
// a string literal at the argument position (quotes preserved above) never matches.
const NON_LITERAL_STAGE_PATTERNS = [
  // pipeline("<task>", <model>) with a non-literal model argument.
  /\bpipeline\s*\(\s*["'`][^"'`]*["'`]\s*,\s*(?!["'`)\s])/g,
  // modelId: <value> / model: <value> with a non-literal value. Whitespace must be excluded
  // (regex backtracking would otherwise let the lookahead succeed on the space before a
  // literal), null/undefined/true/false are state initialisers (not model ids), and `{` opens
  // a nested config object whose inner keys are scanned in their own right.
  /(?<![\w$-])modelId\s*:\s*(?![\s{"'`]|null\b|undefined\b|true\b|false\b)/g,
  /(?<![\w$-])model\s*:\s*(?![\s{"'`]|null\b|undefined\b|true\b|false\b)/g,
  // <Class>.from_pretrained(<model>) / fromPretrained(<model>) with a non-literal first arg.
  /\bfrom_?pretrained\s*\(\s*(?!["'`)\s])/gi,
];

const snippetAt = (src, idx) => {
  const s = src.slice(idx, idx + 160).replace(/\s+/g, " ").trim();
  return s.length > 80 ? s.slice(0, 77) + "..." : s;
};

const findNonLiteralStageLoads = (files) => {
  const hits = [];
  for (const f of files) {
    const scan = blankStringsAndComments(f.html ? blankNonScript(f.src) : f.src);
    for (const re of NON_LITERAL_STAGE_PATTERNS) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(scan))) {
        hits.push({ rel: f.rel, snippet: snippetAt(f.src, m.index) });
        if (re.lastIndex === m.index) re.lastIndex++; // guard against zero-width loops
      }
    }
  }
  return hits;
};

const errors = [];
const checkSlug = (slug) => {
  const dir = join(ROOT, "models", slug);
  const manifestRel = `models/${slug}/acceptance.json`;
  const startLen = errors.length;

  // Non-literal loader call sites evade stage extraction entirely — fail closed on them no
  // matter what acceptance.json declares.
  const srcFiles = familySourceFiles(dir);
  for (const h of findNonLiteralStageLoads(srcFiles)) {
    errors.push(
      `${slug}: non-literal model id at loader call site in ${h.rel} — "${h.snippet}" — ` +
        "declare the model id as a string literal at the call site (repo convention across the " +
        "294 families) so stages are statically extractable",
    );
  }

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
  // A declared stage that is never referenced in the family sources proves nothing is loaded
  // under that id — fake stages could otherwise pad the manifest while something else loads.
  // The id may legitimately appear only in a worker, so search every family file (raw text).
  const familyText = srcFiles.map((f) => f.src).join("\n").toLowerCase();
  for (const s of stages) {
    const name = String(s).split("/").pop().toLowerCase();
    if (!familyText.includes(String(s).toLowerCase()) && !familyText.includes(name)) {
      errors.push(
        `${slug}: declared stage "${s}" is never referenced in the family sources — ` +
          "remove it or load it for real",
      );
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
