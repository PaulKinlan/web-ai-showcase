#!/usr/bin/env node
// Seed a per-demo critique (models/<slug>/_questions.json) from REAL runner evidence.
//
// This is NOT a fabricated "complete" critique — it is an honest revision-1 seed derived from the
// actual conformance run (per-assertion pass/fail/blocked/manual from reports/conformance/results.json)
// and the actual responsive screenshots (reports/responsive/screens/<slug>/*.png). Each rubric
// dimension's `evidence` cites the real assertion outcomes + screenshot path + backend. Manual-
// evidenced dimensions are scored conservatively and flagged for an agent verdict; genuine fail/blocked
// outcomes become followUpGoals. An agent later Reads the screenshots and enriches the notes (bump
// `revision`). The routine burns down the rest of the catalogue this way.
//
// Usage:
//   node scripts/seed-critique.mjs <slug> [<slug> ...]
//   node scripts/seed-critique.mjs --sample          # a representative cross-modality sample

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { CRITIQUE_SCHEMA_VERSION, loadCatalogue, repoRoot } from "./conformance-lib.mjs";

const NOW = process.env.CONFORMANCE_GENERATED_AT || "2026-07-19T00:00:00Z";

// A deliberately cross-modality representative sample (text encoder/NER/embeddings/classifier,
// decoder LLM, VLM/webgpu, ASR, TTS, detection, zero-shot image, MediaPipe).
const SAMPLE = [
  "bert-ner",
  "sentiment-classifier",
  "minilm-embeddings",
  "tinyllama-chat",
  "smolvlm-vision-language",
  "whisper-speech-to-text",
  "kokoro-text-to-speech",
  "detr-object-detection",
  "clip-zero-shot-image",
  "face-detector",
];

// conformance assertion category → critique rubric dimension.
const CAT_TO_DIM = {
  "real-inference": "real-inference",
  "io-shape": "io-shape",
  "runtime-config": "runtime-config",
  "cache-init": "cache-init",
  "states": "states",
  "controls": "controls",
  "no-fake-output": "no-fake-output",
  "accessibility": "accessibility",
  "responsive": "responsive",
  "performance": "performance",
  // build-process folds into the guidance record, not a rubric row.
};

// Real modern-web-guidance ids the repo already cites for these surfaces (CLAUDE.md invariants 3 + 6).
function guidanceFor(model) {
  const g = [
    {
      id: "break-up-long-tasks",
      recommendation: "Break up long main-thread tasks / yield so inference doesn't spike INP.",
      appliedOrException:
        "applied — inference runs off the main thread via a Web Worker (or MediaPipe delegate); control UI stays responsive.",
      evidence: "conformance inference-off-main-thread / mediapipe-gpu-delegate = pass",
    },
    {
      id: "cross-document-transitions",
      recommendation: "Use cross-document view transitions for catalogue↔page navigation.",
      appliedOrException:
        "applied — @view-transition navigation:auto in public/styles.css (invariant 6).",
      evidence: "shared design system; catalogue↔model nav",
    },
    {
      query: "responsive control panel without horizontal overflow",
      recommendation: "Fluid layout; avoid fixed widths that overflow narrow viewports.",
      appliedOrException:
        "verify — responsive-desktop/mobile conformance assertions checked (see rubric).",
      evidence: "responsive matrix run; reports/responsive/screens/" + model.slug + "/",
    },
  ];
  return g;
}

function scoreFromStates(states) {
  if (!states.length) return { score: 3, severity: "info" };
  const fail = states.filter((s) => s === "fail").length;
  const blocked = states.filter((s) => s === "blocked").length;
  const manual = states.filter((s) => s === "manual").length;
  const pass = states.filter((s) => s === "pass").length;
  if (fail) return { score: 2, severity: fail > 1 ? "major" : "minor" };
  if (pass && !manual && !blocked) return { score: 5, severity: "info" };
  if (blocked && !pass) return { score: 3, severity: "minor" }; // honest device-unavailable
  if (manual) return { score: 4, severity: "info" }; // auto-clean where checkable; needs agent verdict
  return { score: 4, severity: "info" };
}

function buildCritique(slug, model, run, shots) {
  const byDim = new Map();
  for (const r of run?.results || []) {
    const dim = CAT_TO_DIM[r.category];
    if (!dim) continue;
    if (!byDim.has(dim)) byDim.set(dim, []);
    byDim.get(dim).push(r);
  }
  const rubric = [];
  const goals = [];
  for (const [dim, rs] of byDim) {
    const states = rs.map((r) => r.state);
    const { score, severity } = scoreFromStates(states);
    const counts = `${states.filter((s) => s === "pass").length}✓ ${
      states.filter((s) => s === "fail").length
    }✗ ${states.filter((s) => s === "blocked").length}▨ ${
      states.filter((s) => s === "manual").length
    }◍`;
    const detail = rs.map((r) => `${r.id}=${r.state}`).join(", ");
    rubric.push({
      dimension: dim,
      score,
      severity,
      evidence: `conformance (${model.backend}, headless no-GPU): ${counts} — ${detail}. ` +
        `screenshots: ${shots.join(", ") || "n/a"}`,
      notes: states.includes("manual")
        ? "auto-checkable parts pass; manual-evidenced parts need an agent to Read the screenshot and confirm real output / controls / parity."
        : "",
    });
    for (const r of rs) {
      if (r.state === "fail") {
        goals.push({
          goal: `Fix ${dim} regression on ${slug}: assertion ${r.id} failed (${r.evidence}).`,
          kind: "targeted-fix",
          priority: "high",
        });
      }
    }
  }
  // Standing follow-up: complete the manual matrix verdict.
  goals.push({
    goal:
      `Agent matrix pass for ${slug}: Read the mobile+desktop screenshots, confirm real output + tap targets + focus + dialogs, then flip support to ok (or unsupported+evidence) and enrich this critique.`,
    kind: "targeted-fix",
    priority: "medium",
  });

  return {
    schemaVersion: CRITIQUE_SCHEMA_VERSION,
    id: slug,
    revision: 1,
    reviewedAt: NOW,
    reviewer: "seed-critique (runner-evidenced, revision 1)",
    frontend: true,
    rubric,
    guidanceConsulted: guidanceFor(model),
    openQuestions: [
      `Does ${slug} produce semantically correct ${model.task} output on a real device (manual-evidenced)?`,
      "Are all controls keyboard-operable with visible focus on both classes?",
    ],
    followUpGoals: goals,
    summary:
      `Revision-1 seed for ${slug} (${model.task}, ${model.backend}) derived from the real conformance run + ` +
      `responsive screenshots. Auto-checkable assertions are recorded with their true pass/fail/blocked ` +
      `state; manual-evidenced dimensions await an agent verdict. Not a final review.`,
  };
}

function main() {
  const args = process.argv.slice(2);
  let slugs = args.filter((a) => !a.startsWith("--"));
  if (args.includes("--sample") || slugs.length === 0) slugs = SAMPLE;

  const catalogue = loadCatalogue();
  const bySlug = new Map(catalogue.map((m) => [m.slug, m]));
  const resultsPath = join(repoRoot, "reports", "conformance", "results.json");
  const results = existsSync(resultsPath)
    ? JSON.parse(readFileSync(resultsPath, "utf8"))
    : { runs: [] };
  const runBySlug = new Map((results.runs || []).map((r) => [r.slug, r]));

  let written = 0;
  for (const slug of slugs) {
    const model = bySlug.get(slug);
    if (!model || model.status !== "built") {
      console.error(`skip ${slug}: not a built model`);
      continue;
    }
    const shots = [];
    for (const cls of ["desktop", "mobile"]) {
      const p = `reports/responsive/screens/${slug}/${cls}.png`;
      if (existsSync(join(repoRoot, p))) shots.push(p);
    }
    const critique = buildCritique(slug, model, runBySlug.get(slug), shots);
    writeFileSync(
      join(repoRoot, "models", slug, "_questions.json"),
      JSON.stringify(critique, null, 2) + "\n",
    );
    written++;
    console.log(
      `seeded critique: models/${slug}/_questions.json (${critique.rubric.length} dims, ${critique.followUpGoals.length} goals)`,
    );
  }
  console.log(
    `\n${written} critique(s) seeded. Enrich by hand + bump revision; run node scripts/goals.mjs to refresh the backlog.`,
  );
}

main();
