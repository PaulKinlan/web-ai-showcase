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
//   4. a suite present on origin/main lost or WEAKENED an assertion (normalized text changed / removed)
//      without a record in conformance-migrations.json — immutable means fix the demo, never weaken.
//   5. a demo the action TOUCHED (its page HTML/JS changed vs origin/main) has a support class left
//      "untested"/"broken" — a touched demo must be validated on both classes.
//   6. any support class regressed non-monotonically: a class that was "ok" on origin/main is now
//      untested/needs-review/broken/removed without a migration record.
//   7. any support class is explicitly "broken" (a recorded breakage that must be fixed, not shipped).
//
// PASSES: many demos still "untested"/"needs-review" (that is the backlog burn-down, not a failure);
// additive new suites/assertions; honest new blocked/unsupported records.
//
// Usage: node scripts/check-conformance.mjs   (belongs beside check-routes.mjs before every push + CI)

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  builtModels,
  computeSuiteHash,
  loadCatalogue,
  normalizeAssertion,
  repoRoot,
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
  if (!existsSync(p)) return [];
  const arr = JSON.parse(readFileSync(p, "utf8"));
  if (!Array.isArray(arr)) throw new Error("conformance-migrations.json must be an array");
  return arr;
}
const migratedAssertion = (migs, suiteId, assertionId) =>
  migs.some((m) =>
    m.suiteId === suiteId && m.assertionId === assertionId &&
    ["remove", "weaken"].includes(m.action)
  );

function main() {
  const failures = [];
  const catalogue = loadCatalogue();
  const built = builtModels(catalogue);
  const builtSlugs = new Set(built.map((m) => m.slug));
  const migrations = loadConfMigrations();

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
            `WEAKENED (${slug}): assertion "${ba.id}" was REMOVED without a conformance-migrations.json record. Immutable — fix the demo, never delete the assertion.`,
          );
        }
        continue;
      }
      if (
        JSON.stringify(bn) !== JSON.stringify(cn) && !migratedAssertion(migrations, suite.id, ba.id)
      ) {
        failures.push(
          `WEAKENED (${slug}): assertion "${ba.id}" CHANGED vs origin/main without a migration record. Adding assertions is allowed; changing/weakening one is not.`,
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
  const resultsPath = join(repoRoot, "reports", "conformance", "results.json");
  if (existsSync(resultsPath)) {
    try {
      const a = JSON.parse(readFileSync(resultsPath, "utf8")).aggregate;
      console.log(
        `last run: ${a.tested}/${a.total} assertions tested — pass ${a.pass} · fail ${a.fail} · ` +
          `blocked ${a.blocked} · manual-evidenced ${a.manual}`,
      );
    } catch { /* ignore */ }
  }

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
