#!/usr/bin/env node
// Route regression gate — enforces the durable-demo compatibility contract (CLAUDE.md invariant 13).
//
// Compares the PREVIOUSLY PUBLISHED manifest (baseline) against the working tree (current) and FAILS
// (exit 1) on any destructive change to a published demo's identity. Additive work, honest new
// `blocked` records, and in-place fixes (same id + identity + live route) PASS. Exceptional
// removals/moves/identity-changes are permitted ONLY when recorded in migrations.json.
//
// Baseline (can't drift — derived from git):
//   git show origin/main:models.json  → buildManifest()   [preferred]
//   fallback: .route-manifest.baseline.json (already a normalized manifest) when git is unavailable.
// Current:
//   working-tree models.json → buildManifest(), plus on-disk models/<slug>/index.html existence for
//   `built` route resolution.
//
// FAIL (exit 1) when, baseline → current:
//   1. a baseline published (built OR blocked) id is missing from current (deleted/renamed), or
//   2. a baseline `built` route's models/<slug>/index.html no longer exists, or
//   3. a `built` id's identity {hfId, task} changed (repurposed), or
//   4. a baseline `blocked` id was deleted (must stay recorded) [special case of 1], or
//   5. the published `built` count dropped and the drop is not covered by migration records.
// PASS (exit 0): additive new ids, honest new `blocked`, in-place fixes (same id+identity+live route),
//   and anything listed in migrations.json.
//
// Usage: node scripts/check-routes.mjs   (run before every push; belongs in CI).

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildManifest } from "./route-manifest.mjs";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const MIGRATION_ACTIONS = new Set(["alias", "move", "remove", "identity-change"]);

function loadBaseline() {
  // Preferred: the manifest derived from origin/main's models.json, so the baseline can't drift.
  try {
    const raw = execFileSync("git", ["show", "origin/main:models.json"], {
      cwd: repoRoot,
      maxBuffer: 1 << 30,
      stdio: ["ignore", "pipe", "ignore"],
    }).toString();
    return { manifest: buildManifest(JSON.parse(raw)), source: "origin/main:models.json" };
  } catch {
    // Fallback: the committed snapshot (already a normalized manifest) for offline runs.
    const snap = new URL("../.route-manifest.baseline.json", import.meta.url);
    if (existsSync(fileURLToPath(snap))) {
      const arr = JSON.parse(readFileSync(fileURLToPath(snap), "utf8"));
      return { manifest: arr, source: ".route-manifest.baseline.json (git unavailable)" };
    }
    throw new Error(
      "No baseline available: git show origin/main:models.json failed and " +
        ".route-manifest.baseline.json is missing.",
    );
  }
}

function loadCurrent() {
  const cat = JSON.parse(
    readFileSync(fileURLToPath(new URL("../models.json", import.meta.url)), "utf8"),
  );
  return buildManifest(cat);
}

function loadMigrations() {
  const url = new URL("../migrations.json", import.meta.url);
  if (!existsSync(fileURLToPath(url))) return [];
  const arr = JSON.parse(readFileSync(fileURLToPath(url), "utf8"));
  if (!Array.isArray(arr)) throw new Error("migrations.json must be an array");
  for (const m of arr) {
    if (!m.id || !MIGRATION_ACTIONS.has(m.action)) {
      throw new Error(
        `migrations.json entry needs {id, action in ${[...MIGRATION_ACTIONS].join("|")}}: ` +
          JSON.stringify(m),
      );
    }
  }
  return arr;
}

// Does a migration record excuse a change to this baseline id?
function migratedBy(migrations, id, actions) {
  return migrations.some((m) => m.id === id && actions.includes(m.action));
}

function routeExists(entry) {
  return existsSync(`${repoRoot}${entry.route}index.html`);
}

function main() {
  const { manifest: baseline, source } = loadBaseline();
  const current = loadCurrent();
  const migrations = loadMigrations();

  const curById = new Map(current.map((e) => [e.id, e]));
  const baseById = new Map(baseline.map((e) => [e.id, e]));

  const failures = [];
  const migratedOut = []; // baseline ids legitimately removed/moved via migrations.json

  for (const b of baseline) {
    const c = curById.get(b.id);

    // 1 / 4 — baseline published id missing from current (deleted or renamed).
    if (!c) {
      if (migratedBy(migrations, b.id, ["remove", "move", "alias"])) {
        migratedOut.push(b.id);
      } else {
        failures.push(
          `MISSING: baseline ${b.status} id "${b.id}" is gone from the current catalogue ` +
            "(deleted/renamed). Published identities are append-only — record a migration or restore it.",
        );
      }
      continue;
    }

    if (b.status === "built") {
      // 5 (per-id half) — a built demo demoted out of `built` without a migration.
      if (
        c.status !== "built" && !migratedBy(migrations, b.id, ["remove", "move", "identity-change"])
      ) {
        failures.push(
          `DEMOTED: baseline built id "${b.id}" is now "${c.status}". A published built demo may not ` +
            "be un-published without a migration record.",
        );
      }
      // 2 — the built route's page file no longer exists.
      if (
        c.status === "built" && !routeExists(c) && !migratedBy(migrations, b.id, ["move", "alias"])
      ) {
        failures.push(
          `ROUTE GONE: baseline built route "${c.route}index.html" no longer exists on disk. ` +
            "Keep the route live or record a move/alias migration.",
        );
      }
      // 3 — identity repurposed (same slug now showcases a different model/task).
      if (
        c.status === "built" &&
        (c.identity.hfId !== b.identity.hfId || c.identity.task !== b.identity.task) &&
        !migratedBy(migrations, b.id, ["identity-change"])
      ) {
        failures.push(
          `IDENTITY CHANGED: built id "${b.id}" was ${JSON.stringify(b.identity)} and is now ` +
            `${JSON.stringify(c.identity)} (repurposed). Add a NEW slug instead, or record an ` +
            "identity-change migration.",
        );
      }
    }

    if (b.status === "blocked" && c.status !== "blocked" && c.status !== "built") {
      // A blocked record must stay recorded (blocked→built is an allowed promotion; anything else that
      // is not still published is a silent deletion of the honest record).
      if (!migratedBy(migrations, b.id, ["remove", "move"])) {
        failures.push(
          `BLOCKED RECORD LOST: baseline blocked id "${b.id}" is now "${c.status}" and no longer ` +
            "published. Blocked entries must stay honestly recorded.",
        );
      }
    }
  }

  // 5 (aggregate) — published built count dropped beyond what migrations explain.
  const baseBuilt = baseline.filter((e) => e.status === "built").length;
  const curBuilt = current.filter((e) => e.status === "built").length;
  const allowedDrop = new Set(
    migrations
      .filter((m) =>
        ["remove", "move", "identity-change"].includes(m.action) &&
        baseById.get(m.id)?.status === "built"
      )
      .map((m) => m.id),
  ).size;
  if (baseBuilt - curBuilt > allowedDrop) {
    failures.push(
      `BUILT COUNT DROP: baseline had ${baseBuilt} built demos, current has ${curBuilt} ` +
        `(drop ${baseBuilt - curBuilt}); only ${allowedDrop} covered by migrations.json.`,
    );
  }

  // Reporting.
  const added = current.filter((e) => !baseById.has(e.id));
  const inPlace = current.filter((e) => {
    const b = baseById.get(e.id);
    return (
      b &&
      b.status === "built" &&
      e.status === "built" &&
      (b.identity.hfId !== e.identity.hfId || b.identity.task !== e.identity.task) === false
    );
  });

  console.log("=== ROUTE REGRESSION GATE (durable-demo compatibility contract) ===");
  console.log(`baseline source: ${source}`);
  console.log(
    `${baseline.length} baseline / ${current.length} current / ` +
      `+${added.length} added / ~${inPlace.length} in-place / ` +
      `${migratedOut.length} removed(migrated)${
        migratedOut.length ? " [" + migratedOut.join(", ") + "]" : ""
      }`,
  );

  if (failures.length) {
    console.error(`\nFAIL — ${failures.length} contract violation(s):`);
    for (const f of failures) console.error("  ✗ " + f);
    console.error(
      "\nThe durable-demo contract forbids renaming/repurposing/deleting a published demo. " +
        "Add new slugs for new ideas; fix in place with the same id+identity+route; record any " +
        "exceptional move/removal in migrations.json.",
    );
    process.exit(1);
  }

  console.log("\nPASS — all published demo identities preserved (additive-only).");
}

main();
