#!/usr/bin/env node
// Value/quality assessment (SCHEMA + SEEDED records) and a prioritized build queue over the eligible
// universe. ADDITIVE + deterministic. Reads inventory/lineage/records.ndjson (lineage), models.json
// (built/pending/blocked), inventory/lineage/reviewed-meta.json (proven metadata). Writes:
//   inventory/lineage/value-records.ndjson — value records for the 148 BUILT models (real evidence)
//                                            + the reviewed pending sample. Scores are EVIDENCE-GATED:
//                                            null/"insufficient-evidence"/"eval-pending" where we
//                                            lack direct evidence. downloads/likes = WEAK context only.
//   inventory/lineage/priority.json        — pending candidates ranked by unique high-value capability
//                                            first; every deprioritized/superseded item keeps its
//                                            rationale + canonical alternative. Long tail stays VISIBLE.
//
// NEVER asserts two models perf-equivalent, nor "badly trained", from metadata. Where a real
// browser-runtime eval is required (latency/memory/quality), the field is marked "eval-pending" and
// the eval protocol lives in inventory/lineage/README.md. NOT complete/all — a first pass.
//
// Run: `node scripts/lineage-value.mjs`.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const p = (rel) => ROOT + rel;

const records = readFileSync(p("inventory/lineage/records.ndjson"), "utf8").trim().split("\n").map((
  l,
) => JSON.parse(l));
const cat = JSON.parse(readFileSync(p("models.json"), "utf8")).models;
const reviewed = existsSync(p("inventory/lineage/reviewed-meta.json"))
  ? JSON.parse(readFileSync(p("inventory/lineage/reviewed-meta.json"), "utf8"))
  : {};

const recByHf = new Map(records.map((r) => [r.id, r]));
const built = cat.filter((m) => m.status === "built");
const builtTasks = new Set(built.map((m) => m.task));
const builtFamilies = new Set(built.map((m) => (m.family || "").toLowerCase()).filter(Boolean));

// capability-slot rarity across the eligible reps (fewer reps in a slot = rarer capability)
const slotCount = new Map();
for (const r of records) {
  const slot = `${r.task}::${r.canonicalFamily}::${r.specialization.join(",")}`;
  slotCount.set(slot, (slotCount.get(slot) || 0) + 1);
}
// how many built demos already cover a task (overlap signal)
const builtPerTask = new Map();
for (const m of built) builtPerTask.set(m.task, (builtPerTask.get(m.task) || 0) + 1);

const PERMISSIVE = new Set([
  "apache-2.0",
  "mit",
  "bsd",
  "bsd-3-clause",
  "bsd-2-clause",
  "unlicense",
  "cc0-1.0",
]);
const CC_OK = new Set(["cc-by-4.0", "cc-by-sa-4.0", "cc-by-3.0"]);
const RESTRICTED = new Set([
  "cc-by-nc-4.0",
  "cc-by-nc-sa-4.0",
  "cc-by-nc-nd-4.0",
  "gemma",
  "llama2",
  "llama3",
  "llama3.1",
  "llama3.2",
  "openrail",
  "creativeml-openrail-m",
  "other",
]);

function licenseScore(license) {
  if (!license) return { score: null, label: "unknown", evidence: ["no license tag"] };
  if (PERMISSIVE.has(license)) {
    return { score: 3, label: "permissive", evidence: [`license=${license}`] };
  }
  if (CC_OK.has(license)) {
    return { score: 2, label: "attribution", evidence: [`license=${license}`] };
  }
  if (RESTRICTED.has(license)) {
    return {
      score: 1,
      label: "restricted",
      evidence: [`license=${license} (non-commercial/gated/custom)`],
    };
  }
  return { score: null, label: "review-needed", evidence: [`license=${license}`] };
}

// Browser feasibility: SIZE + backend give a real static signal; latency/memory need a runtime eval.
function feasibility(model, rec) {
  const ev = [];
  let score = null;
  const sizeMB = model && model.sizeMB;
  const backend = (model && model.backend) || (rec && rec.runtime === "webllm" ? "webgpu" : "wasm");
  if (typeof sizeMB === "number") {
    ev.push(`sizeMB=${sizeMB}`);
    if (sizeMB <= 120) score = 3;
    else if (sizeMB <= 500) score = 2;
    else if (sizeMB <= 1500) score = 1;
    else score = 0;
  }
  ev.push(`backend=${backend}`);
  if (backend === "webgpu") ev.push("needs a real WebGPU adapter (honest fallback required)");
  return {
    score,
    label: score === null ? "size-unknown" : ["heavy", "large", "moderate", "light"][score],
    backend,
    evidence: ev,
    latencyMemory: "eval-pending", // requires a browser-runtime measurement (see README eval protocol)
  };
}

// Capability uniqueness: rarer slot + task not yet built = more unique/high-value to the showcase.
function capabilityUniqueness(rec, isBuilt) {
  const slot = `${rec.task}::${rec.canonicalFamily}::${rec.specialization.join(",")}`;
  const n = slotCount.get(slot) || 1;
  const taskBuilt = builtPerTask.get(rec.task) || 0;
  const ev = [`slot=${slot}`, `reps-in-slot=${n}`, `built-demos-for-task=${taskBuilt}`];
  let score;
  if (
    ["format-port", "quant-variant", "distillation", "fork-no-change"].includes(rec.relationship)
  ) {
    score = 0; // a port/quant/distil re-presents an existing capability
    ev.push(`relationship=${rec.relationship} → re-presents an existing capability`);
  } else if (rec.relationship === "specialization-distinct") {
    score = rec.specialization.length ? 2 : 1;
    ev.push("materially-distinct domain/language/task specialization");
  } else if (rec.relationship === "canonical") {
    score = taskBuilt === 0 ? 3 : n <= 3 ? 2 : 1;
  } else {
    score = 1;
  }
  return {
    score,
    label: ["redundant", "incremental", "distinct", "novel-capability"][score] ?? "distinct",
    evidence: ev,
  };
}

// Overlap with a stronger/existing BUILT model (higher = MORE overlap = lower marginal value).
function overlap(rec, isBuilt) {
  const famBuilt = builtFamilies.has((rec.canonicalFamily || "").toLowerCase());
  const taskBuilt = builtPerTask.get(rec.task) || 0;
  const ev = [`built-demos-for-task=${taskBuilt}`, `family-already-built=${famBuilt}`];
  if (isBuilt) return { level: "self-built", evidence: ["this model is already a built demo"] };
  if (
    famBuilt &&
    ["format-port", "quant-variant", "distillation", "fine-tune"].includes(rec.relationship)
  ) {
    return { level: "high", evidence: [...ev, `${rec.relationship} of an already-built family`] };
  }
  if (taskBuilt >= 3 && rec.relationship === "canonical") {
    return { level: "medium", evidence: [...ev, "task already has several built demos"] };
  }
  return { level: "low", evidence: ev };
}

// Model-card / provenance quality — only asserted from REAL reviewed metadata; else insufficient.
function cardQuality(rec) {
  const rev = reviewed[rec.id];
  if (!rev || rev.error) {
    return {
      score: null,
      label: "insufficient-evidence",
      confidence: "low",
      evidence: ["not in reviewed sample — no card fetched"],
    };
  }
  const ev = [];
  let pts = 0;
  if (rev.config && rev.config.architectures) {
    pts++;
    ev.push("config.architectures present");
  }
  if (rev.hasTokenizer) {
    pts++;
    ev.push("tokenizer present");
  }
  if (rev.base_model) {
    pts++;
    ev.push("base_model declared");
  }
  if (rev.license) {
    pts++;
    ev.push("license declared");
  }
  const score = Math.min(3, pts);
  return {
    score,
    label: ["minimal", "sparse", "adequate", "well-documented"][score],
    confidence: "medium",
    evidence: ev.length ? ev : ["reviewed but card fields sparse"],
  };
}

function buildValueRecord(rec, model, isBuilt) {
  const lic = licenseScore(rec.license);
  const feas = feasibility(model, rec);
  const uniq = capabilityUniqueness(rec, isBuilt);
  const ov = overlap(rec, isBuilt);
  const card = cardQuality(rec);
  const rev = reviewed[rec.id];
  const dims = {
    capabilityUniqueness: { ...uniq, confidence: rec.reviewed ? "medium" : "low" },
    developerValue: {
      score: null,
      label: "eval-pending",
      note:
        "developer value = does a real use-case matrix run in-browser; assessed at build/validation time.",
      evidence: [],
    },
    browserFeasibility: {
      ...feas,
      confidence: typeof (model && model.sizeMB) === "number" ? "medium" : "low",
    },
    licenseDeployability: { ...lic, confidence: rec.license ? "high" : "low" },
    modelCardQuality: card,
    maintenanceProvenance: {
      lineageConfidence: rec.confidence,
      relationship: rec.relationship,
      lastModified: rev && !rev.error ? rev.lastModified : null,
      evidence: rec.evidence.proven.length
        ? rec.evidence.proven
        : rec.evidence.inferred.slice(0, 3),
      weakContext: {
        downloads: rec.downloads,
        likes: rec.likes,
        note: "downloads/likes are WEAK supporting context only",
      },
    },
    safetyLimitations: {
      label: /medical|clinical|legal|finance|toxic|pii/.test(rec.specialization.join(","))
        ? "domain-sensitive — needs explicit not-professional-advice + limitations disclosure"
        : "insufficient-evidence",
      confidence: "low",
      evidence: rec.specialization.filter((s) =>
        /medical|clinical|legal|finance|toxic|pii/.test(s)
      ),
    },
    overlapWithStronger: { ...ov },
    showcaseInterest: {
      score: uniq.score >= 2 ? uniq.score : null,
      label: uniq.score >= 2 ? "candidate-of-interest" : "eval-pending",
      confidence: "low",
      evidence: ["derived from capability uniqueness; qualitative — refine at build time"],
    },
  };
  return {
    id: rec.id,
    task: rec.task,
    canonicalFamily: rec.canonicalFamily,
    relationship: rec.relationship,
    specialization: rec.specialization,
    catalogueStatus: isBuilt ? "built" : rec.catalogueStatus,
    reviewed: rec.reviewed,
    evalPending: ["browserFeasibility.latencyMemory", "developerValue"].concat(
      dims.showcaseInterest.label === "eval-pending" ? ["showcaseInterest"] : [],
    ),
    dimensions: dims,
    overallConfidence: rec.reviewed ? "medium" : "low",
    sourceRepo: rec.sourceRepo,
  };
}

// ── Value records: 148 BUILT (real evidence) + reviewed pending sample ──
const valueRecords = [];
const seen = new Set();
for (const m of built) {
  // built models may not be in the eligible reps; synthesize a minimal rec if absent.
  const rec = recByHf.get(m.hfId) || {
    id: m.hfId || m.slug,
    task: m.task,
    modality: m.modality,
    runtime: m.runtime || (m.backend === "webgpu" ? "webllm" : "transformers.js"),
    license: m.license || null,
    downloads: 0,
    likes: 0,
    gated: false,
    reviewed: !!(reviewed[m.hfId] && !reviewed[m.hfId].error),
    canonicalFamily: (m.family || m.task).toLowerCase(),
    base_model: null,
    baseRoot: null,
    relationship: "canonical",
    specialization: [],
    catalogueStatus: "built",
    sourceRepo: m.hfId ? `https://huggingface.co/${m.hfId}` : null,
    evidence: {
      proven: [],
      inferred: ["built demo — value evidence from models.json + validation"],
    },
    confidence: "medium",
  };
  valueRecords.push(buildValueRecord(rec, m, true));
  seen.add(rec.id);
}
// reviewed pending sample
for (const rec of records) {
  if (seen.has(rec.id)) continue;
  if (!rec.reviewed) continue;
  if (rec.catalogueStatus === "built") continue;
  valueRecords.push(buildValueRecord(rec, cat.find((m) => m.hfId === rec.id), false));
  seen.add(rec.id);
}

writeFileSync(
  p("inventory/lineage/value-records.ndjson"),
  valueRecords.map((r) => JSON.stringify(r)).join("\n") + "\n",
);

// ── Priority queue over PENDING eligible candidates ──
const builtHf = new Set(built.map((m) => m.hfId));
const pending = records.filter((r) => !builtHf.has(r.id));
const scored = pending.map((rec) => {
  const model = cat.find((m) => m.hfId === rec.id);
  const uniq = capabilityUniqueness(rec, false).score;
  const lic = licenseScore(rec.license).score;
  const feas = feasibility(model, rec).score;
  const ov = overlap(rec, false).level;
  const taskBuilt = builtPerTask.get(rec.task) || 0;
  const deployable = lic === 3 || lic === 2;
  let tier, rationale, canonicalAlternative = rec.baseRoot || null;
  if (rec.gated) {
    tier = "blocked";
    rationale = "gated/license-restricted — stays counted in the denominator, not buildable as-is.";
  } else if (["format-port", "quant-variant", "distillation"].includes(rec.relationship)) {
    tier = "superseded";
    rationale =
      `${rec.relationship} — re-presents an existing capability; build the canonical/base instead.`;
  } else if (ov === "high") {
    tier = "low";
    rationale = "high overlap with an already-built family/task — low marginal capability.";
  } else if (uniq === 3 && deployable && (feas === null || feas >= 1)) {
    // canonical model of a task with NO built demo yet — the strongest build-first signal.
    tier = "high";
    rationale =
      "canonical model for a task with NO built demo yet, deployable license, browser-feasible.";
  } else if (uniq >= 2 && deployable && taskBuilt <= 1 && (feas === null || feas >= 1)) {
    // materially-distinct specialization in a barely-covered task.
    tier = "high";
    rationale =
      "materially-distinct specialization in a barely-covered task, deployable license, feasible.";
  } else if (uniq >= 1 && lic !== 1) {
    tier = "medium";
    rationale = taskBuilt >= 2
      ? "distinct but the task already has built demos — build after the high tier."
      : "incremental capability or unverified feasibility/license — worth building after high.";
  } else {
    tier = "low";
    rationale =
      "redundant, restricted-license, or low-confidence — kept VISIBLE in the long tail, not deleted.";
  }
  // composite sort key: tier weight, then uniqueness, license, feasibility, downloads (weak tiebreak)
  const tierW = { high: 4, medium: 3, low: 2, superseded: 1, blocked: 0 }[tier];
  const sortKey = tierW * 1e12 + (uniq ?? 0) * 1e9 + (lic ?? 0) * 1e6 + ((feas ?? 1) + 1) * 1e3 +
    Math.min(999, Math.log10((rec.downloads || 0) + 1) * 100);
  return {
    id: rec.id,
    task: rec.task,
    canonicalFamily: rec.canonicalFamily,
    relationship: rec.relationship,
    specialization: rec.specialization,
    tier,
    rationale,
    canonicalAlternative: tier === "superseded" || tier === "low" ? canonicalAlternative : null,
    scores: { capabilityUniqueness: uniq, license: lic, feasibility: feas, overlap: ov },
    confidence: rec.confidence,
    reviewed: rec.reviewed,
    sortKey,
  };
});
scored.sort((a, b) => b.sortKey - a.sortKey);
for (const s of scored) delete s.sortKey;

const tierCounts = {};
for (const s of scored) tierCounts[s.tier] = (tierCounts[s.tier] || 0) + 1;

const priority = {
  generatedAt: new Date().toISOString().slice(0, 10),
  disclaimer:
    "FIRST evidence-backed prioritization pass — NOT complete/all. Ranks PENDING eligible candidates " +
    "by unique high-value capability first. The long tail is KEPT VISIBLE (never deleted). Scores are " +
    "evidence-gated; latency/memory/quality are eval-pending (browser-runtime eval required — protocol " +
    "in README). downloads/likes are weak context only.",
  denominatorsRef: "inventory/lineage/denominators.json",
  pendingTotal: scored.length,
  tierCounts,
  tierMeaning: {
    high: "distinct/novel capability + deployable license + browser-feasible — build first.",
    medium: "incremental capability or unverified feasibility/license — build after high.",
    low: "redundant/restricted/low-confidence — visible long tail, low marginal value.",
    superseded:
      "format-port/quant/distillation of an existing capability — build the canonical instead.",
    blocked: "gated/restricted — stays counted, not buildable as-is.",
  },
  queue: scored,
};
writeFileSync(p("inventory/lineage/priority.json"), JSON.stringify(priority, null, 2) + "\n");

console.error("=== VALUE + PRIORITY ===");
console.error(
  `value records: ${valueRecords.length} (built ${built.length} + reviewed pending ${
    valueRecords.length - built.length
  })`,
);
console.error(`priority queue: ${scored.length} pending — tiers ${JSON.stringify(tierCounts)}`);
console.error("→ inventory/lineage/value-records.ndjson + priority.json");
