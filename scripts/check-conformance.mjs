#!/usr/bin/env node
// Conformance + parity gate — the sibling of scripts/check-routes.mjs. Run in the SAME pre-push step
// and in CI. Enforces the immutable-conformance contract, the modern-web-guidance mandate, and the
// mobile+desktop parity invariant. Exit 1 on any violation; always REPORTS the coverage denominators.
//
// FAILS (exit 1) when:
//   1. a built model has NO conformance suite (missing).
//   2. a conformance.json exists for an id that is not a built model (unknown/orphan), or two suites
//      share an id (duplicate).
//   3. any artifact is malformed (schema/validateSuite/validateCritique), or a suiteHash doesn't match
//      its assertions.
//   4. a suite present on origin/main lost or CHANGED an assertion (normalized text changed /
//      removed) without a record in conformance-migrations.json — immutable means fix the demo,
//      never weaken. The record's action must be one of remove|weaken|correct (see
//      CONFORMANCE_MIGRATION_ACTIONS in conformance-lib.mjs): a factual correction of an assertion
//      derived from wrong metadata is not a weakening and must not be mislabelled as one.
//   5. a demo the action TOUCHED (its page HTML/JS changed vs origin/main) has a support class left
//      "untested"/"broken" — a touched demo must be validated on both classes.
//   6. any support class regressed non-monotonically: a class that was "ok" on origin/main is now
//      untested/needs-review/broken/removed without a migration record.
//   7. any support class is explicitly "broken" (a recorded breakage that must be fixed, not shipped).
//   8. reports/conformance/results.json RECORDS a failing assertion (state:"fail"). Rules 1-7 prove
//      each suite is PRESENT and INTACT; they never proved the demo PASSES it. Until this rule
//      existed the gate printed the `fail` count and exited 0 anyway, so a red assertion could —
//      and did — land on main (mms-tts-bengali, bead web-ai-showcase-qjp).
//   9. that evidence record is internally inconsistent: the stored `aggregate` disagrees with the
//      per-run tallies, or a run's own counters disagree with its own results array. Rule 8 reads
//      the assertion STATES and rule 9 reads the COUNTERS, so neither a stale summary nor a
//      hand-edited tally can hide a failure from both. A missing or malformed results.json also
//      fails — it is a tracked artifact, and its absence must never read as "nothing failed".
//
// PASSES: many demos still "untested"/"needs-review" (that is the backlog burn-down, not a failure);
// additive new suites/assertions; honest new blocked/unsupported records; a suite with NO run record
// (results.json is a merge-by-slug rollup, so a targeted `--slug` run legitimately leaves every
// other suite's record untouched — partial coverage is the backlog, not a regression); assertions in
// `manual` (needs an agent verdict) or `blocked` (honest device/feature-unavailable) state.
//
// Usage: node scripts/check-conformance.mjs   (belongs beside check-routes.mjs before every push + CI)

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  builtModels,
  computeSuiteHash,
  CONFORMANCE_MIGRATION_ACTIONS,
  evaluateRecordedOutcome,
  loadCatalogue,
  migratedAssertion,
  normalizeAssertion,
  repoRoot,
  validateConformanceMigrations,
  validateCritique,
  validateSuite,
} from "./conformance-lib.mjs";

function gitShow(ref) {
  try {
    return execFileSync("git", ["show", ref], {
      cwd: repoRoot,
      maxBuffer: 1 << 30,
      stdio: ["ignore", "pipe", "ignore"],
    }).toString();
  } catch {
    return null;
  }
}

function changedFilesVsOriginMain() {
  try {
    const out = execFileSync("git", ["diff", "--name-only", "origin/main", "--", "models/"], {
      cwd: repoRoot,
      maxBuffer: 1 << 30,
      stdio: ["ignore", "pipe", "ignore"],
    }).toString();
    return out.split("\n").filter(Boolean);
  } catch {
    return [];
  }
}

// Which built demos had their PAGE (HTML/JS) touched vs origin/main — the conformance/critique
// sidecars do NOT count as touching the demo implementation.
function touchedDemos() {
  const touched = new Set();
  for (const f of changedFilesVsOriginMain()) {
    const m = f.match(/^models\/([^/]+)\/(.+)$/);
    if (!m) continue;
    const [, slug, rest] = m;
    if (rest.endsWith("conformance.json") || rest.endsWith("_questions.json")) continue;
    if (/\.(html|js|mjs)$/.test(rest)) touched.add(slug);
  }
  return touched;
}

function loadConfMigrations() {
  const p = join(repoRoot, "conformance-migrations.json");
  if (!existsSync(p)) return { migrations: [], errors: [] };
  const arr = JSON.parse(readFileSync(p, "utf8"));
  // Structure is enforced here, not only documented in schemas/: an unargued or mislabelled record
  // is the audit-trail hole the immutability rule exists to prevent (web-ai-showcase-9tw).
  return { migrations: Array.isArray(arr) ? arr : [], errors: validateConformanceMigrations(arr) };
}

// ── Rules 8 + 9: the recorded OUTCOME is part of the gate ───────────────────────────────────────
// Thin IO wrapper over evaluateRecordedOutcome() in conformance-lib.mjs. The decision logic is a
// pure function there so every branch (failing assertion, desynced per-run counter, drifted
// aggregate, partial rollup, stale record) is unit-tested in test/conformance-outcome.test.mjs
// rather than only exercised by hand. Reading the file is the only thing that belongs here.
function recordedOutcome(suiteAssertionCounts, builtCount) {
  const p = join(repoRoot, "reports", "conformance", "results.json");

  if (!existsSync(p)) {
    return {
      lines: [],
      failures: [
        "MISSING EVIDENCE: reports/conformance/results.json is absent, but it is a tracked " +
        'artifact — a deleted outcome record must not read as "nothing failed". Regenerate it ' +
        "with \`node scripts/conformance.mjs --all\`.",
      ],
    };
  }

  let doc;
  try {
    doc = JSON.parse(readFileSync(p, "utf8"));
  } catch (e) {
    return {
      lines: [],
      failures: [
        `MALFORMED: reports/conformance/results.json is not valid JSON (${e.message})`,
      ],
    };
  }

  return evaluateRecordedOutcome(doc, { suiteAssertionCounts, builtCount });
}

function main() {
  const failures = [];
  const catalogue = loadCatalogue();
  const built = builtModels(catalogue);
  const builtSlugs = new Set(built.map((m) => m.slug));
  const { migrations, errors: migrationErrors } = loadConfMigrations();
  // A malformed or under-argued migration record fails the gate outright: if the audit trail is
  // broken, every "it's recorded" claim downstream is worthless.
  for (const e of migrationErrors) failures.push(`migration record invalid — ${e}`);

  // Enumerate on-disk suites + critiques.
  const suiteFiles = [];
  const critiqueFiles = [];
  for (const dir of readdirSync(join(repoRoot, "models"))) {
    const cp = join(repoRoot, "models", dir, "conformance.json");
    const qp = join(repoRoot, "models", dir, "_questions.json");
    if (existsSync(cp)) suiteFiles.push({ slug: dir, path: cp });
    if (existsSync(qp)) critiqueFiles.push({ slug: dir, path: qp });
  }

  // 1. every built model has a suite.
  const withSuite = new Set(suiteFiles.map((s) => s.slug));
  for (const m of built) {
    if (!withSuite.has(m.slug)) {
      failures.push(
        `MISSING SUITE: built model "${m.slug}" has no models/${m.slug}/conformance.json`,
      );
    }
  }

  // 2 + 3. parse, validate, dedup ids, orphan detection.
  const seenIds = new Map();
  const suites = [];
  for (const { slug, path } of suiteFiles) {
    let suite;
    try {
      suite = JSON.parse(readFileSync(path, "utf8"));
    } catch (e) {
      failures.push(`MALFORMED: ${path} is not valid JSON (${e.message})`);
      continue;
    }
    suites.push({ slug, suite });
    for (const err of validateSuite(suite)) failures.push(`SCHEMA ${slug}: ${err}`);
    if (!builtSlugs.has(suite.id)) {
      failures.push(`ORPHAN SUITE: "${suite.id}" (models/${slug}/) maps to no built model`);
    }
    if (seenIds.has(suite.id)) {
      failures.push(`DUPLICATE SUITE ID: "${suite.id}" in ${slug} and ${seenIds.get(suite.id)}`);
    }
    seenIds.set(suite.id, slug);
  }

  // 4. immutability vs origin/main — no removed/weakened assertion without a migration record.
  for (const { slug, suite } of suites) {
    const raw = gitShow(`origin/main:models/${slug}/conformance.json`);
    if (!raw) continue; // new suite — additive, nothing to weaken.
    let base;
    try {
      base = JSON.parse(raw);
    } catch {
      continue;
    }
    const curById = new Map(suite.assertions.map((a) => [a.id, normalizeAssertion(a)]));
    for (const ba of base.assertions) {
      const bn = normalizeAssertion(ba);
      const cn = curById.get(ba.id);
      if (!cn) {
        if (!migratedAssertion(migrations, suite.id, ba.id)) {
          failures.push(
            `WEAKENED (${slug}): assertion "${ba.id}" was REMOVED without a conformance-migrations.json record. ` +
              `Immutable — fix the demo, never delete the assertion. Record it with action ${
                CONFORMANCE_MIGRATION_ACTIONS.join("|")
              }.`,
          );
        }
        continue;
      }
      if (
        JSON.stringify(bn) !== JSON.stringify(cn) && !migratedAssertion(migrations, suite.id, ba.id)
      ) {
        failures.push(
          `WEAKENED (${slug}): assertion "${ba.id}" CHANGED vs origin/main without a migration record. ` +
            `Adding assertions is allowed; changing one needs a conformance-migrations.json record with ` +
            `action ${
              CONFORMANCE_MIGRATION_ACTIONS.join("|")
            } (use "correct" when the old assertion was factually wrong).`,
        );
      }
    }
  }

  // 3b. critiques well-formed.
  for (const { slug, path } of critiqueFiles) {
    let c;
    try {
      c = JSON.parse(readFileSync(path, "utf8"));
    } catch (e) {
      failures.push(`MALFORMED CRITIQUE: ${path} (${e.message})`);
      continue;
    }
    for (const err of validateCritique(c)) failures.push(`CRITIQUE ${slug}: ${err}`);
  }

  // 3c. goals.json well-formed (if present).
  const goalsPath = join(repoRoot, "goals.json");
  if (existsSync(goalsPath)) {
    try {
      const g = JSON.parse(readFileSync(goalsPath, "utf8"));
      if (g.schemaVersion !== 1) failures.push("goals.json: schemaVersion must be 1");
      if (!Array.isArray(g.goals)) failures.push("goals.json: goals must be an array");
    } catch (e) {
      failures.push(`MALFORMED: goals.json (${e.message})`);
    }
  }

  // 5/6/7. support / parity checks.
  const baseCatRaw = gitShow("origin/main:models.json");
  const baseSupport = new Map();
  if (baseCatRaw) {
    try {
      const bc = JSON.parse(baseCatRaw);
      for (const m of (Array.isArray(bc) ? bc : bc.models)) {
        if (m.support) baseSupport.set(m.slug, m.support);
      }
    } catch { /* ignore */ }
  }
  const touched = touchedDemos();
  const migSlugs = new Set(migrations.filter((m) => m.slug).map((m) => m.slug));
  for (const m of built) {
    const sup = m.support || { desktop: "untested", mobile: "untested" };
    for (const cls of ["desktop", "mobile"]) {
      const cur = sup[cls] || "untested";
      // 7 — explicit breakage.
      if (cur === "broken") {
        failures.push(
          `BROKEN: "${m.slug}" support.${cls} is "broken" — fix the demo before shipping.`,
        );
      }
      // 5 — touched demo left untested/broken on a class.
      if (touched.has(m.slug) && (cur === "untested" || cur === "broken")) {
        failures.push(
          `UNTESTED TOUCH: "${m.slug}" was touched but support.${cls} is "${cur}" — validate the mobile+desktop matrix for a touched demo.`,
        );
      }
      // 6 — monotonicity: ok must not silently regress.
      const prev = baseSupport.get(m.slug)?.[cls];
      if (prev === "ok" && cur !== "ok" && cur !== "unsupported" && !migSlugs.has(m.slug)) {
        failures.push(
          `SUPPORT REGRESSION: "${m.slug}" support.${cls} was "ok" on origin/main, now "${cur}" — coverage must be monotonic (record a migration if intentional).`,
        );
      }
    }
  }

  // ── Coverage report (always printed) ──
  const critiqueSlugs = new Set(critiqueFiles.map((c) => c.slug));
  const dOk = built.filter((m) => m.support?.desktop === "ok").length;
  const mOk = built.filter((m) => m.support?.mobile === "ok").length;
  const dReview = built.filter((m) => m.support?.desktop === "needs-review").length;
  const mReview = built.filter((m) => m.support?.mobile === "needs-review").length;

  console.log("=== CONFORMANCE + PARITY GATE ===");
  console.log(
    `conformance suites: ${withSuite.size}/${built.length} built demos` +
      `   critique: ${critiqueSlugs.size}/${built.length} built demos`,
  );
  console.log(
    `mobile+desktop parity: desktop ok ${dOk}/${built.length} (needs-review ${dReview}) · ` +
      `mobile ok ${mOk}/${built.length} (needs-review ${mReview})  [untested = backlog]`,
  );
  const outcome = recordedOutcome(
    new Map(suites.map(({ suite }) => [suite.id, (suite.assertions || []).length])),
    built.length,
  );
  for (const line of outcome.lines) console.log(line);
  for (const f of outcome.failures) failures.push(f);

  if (failures.length) {
    console.error(`\nFAIL — ${failures.length} conformance/parity violation(s):`);
    for (const f of failures) console.error("  ✗ " + f);
    console.error(
      "\nImmutable conformance: fix the DEMO, never weaken/delete an assertion. Record exceptional " +
        "assertion removals in conformance-migrations.json. Validate touched demos on mobile+desktop.",
    );
    process.exit(1);
  }
  console.log(
    "\nPASS — every built demo has a valid immutable suite; no weakened assertions; parity honest.",
  );
}

main();
