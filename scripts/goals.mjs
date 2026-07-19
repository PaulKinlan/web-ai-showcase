#!/usr/bin/env node
// questions → goals. Collects every critique's `followUpGoals` (models/<slug>/_questions.json) into
// the repo-level goals.json backlog the routine consumes to pick the next ADDITIVE demo or targeted
// in-place fix — never to replace a stable published page.
//
// Merge-safe: existing goal `status` (open/in-progress/done) is preserved across regenerations; new
// goals are appended; a goal whose critique dropped it stays (marked so) rather than vanishing.
//
// Usage:
//   node scripts/goals.mjs            # rebuild goals.json from all critiques
//   node scripts/goals.mjs --print    # print the open backlog, don't write

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { repoRoot } from "./conformance-lib.mjs";

const args = process.argv.slice(2);
const NOW = process.env.CONFORMANCE_GENERATED_AT || "2026-07-19T00:00:00Z";
const goalsPath = join(repoRoot, "goals.json");

function goalId(demo, goal) {
  return demo + "-" + createHash("sha256").update(goal).digest("hex").slice(0, 8);
}

function collectFromCritiques() {
  const goals = [];
  for (const dir of readdirSync(join(repoRoot, "models"))) {
    const qp = join(repoRoot, "models", dir, "_questions.json");
    if (!existsSync(qp)) continue;
    let c;
    try {
      c = JSON.parse(readFileSync(qp, "utf8"));
    } catch {
      continue;
    }
    for (const g of c.followUpGoals || []) {
      goals.push({
        id: goalId(c.id || dir, g.goal),
        demo: c.id || dir,
        goal: g.goal,
        kind: g.kind || "targeted-fix",
        priority: g.priority || "medium",
        status: "open",
        source: `critique rev ${c.revision ?? 1}`,
      });
    }
  }
  return goals;
}

function main() {
  const existing = existsSync(goalsPath)
    ? JSON.parse(readFileSync(goalsPath, "utf8")).goals || []
    : [];
  const prevStatus = new Map(existing.map((g) => [g.id, g.status]));
  const collected = collectFromCritiques();
  const byId = new Map();
  // Preserve prior status on rebuild; new goals default to open.
  for (const g of collected) {
    if (prevStatus.has(g.id)) g.status = prevStatus.get(g.id);
    byId.set(g.id, g);
  }
  // Keep any prior goal whose critique no longer emits it (don't silently lose backlog items).
  for (const g of existing) {
    if (!byId.has(g.id)) byId.set(g.id, g);
  }
  const goals = [...byId.values()].sort((a, b) =>
    a.demo < b.demo ? -1 : a.demo > b.demo ? 1 : (a.id < b.id ? -1 : 1)
  );

  if (args.includes("--print")) {
    const open = goals.filter((g) => g.status === "open");
    console.log(`goals: ${goals.length} total · ${open.length} open`);
    for (const g of open) console.log(`  [${g.kind}/${g.priority}] ${g.demo}: ${g.goal}`);
    return;
  }
  writeFileSync(
    goalsPath,
    JSON.stringify({ schemaVersion: 1, generatedAt: NOW, goals }, null, 2) + "\n",
  );
  const open = goals.filter((g) => g.status === "open").length;
  console.log(
    `goals.json: ${goals.length} goals (${open} open) from ${collectFromCritiques().length} critique follow-ups.`,
  );
}

main();
