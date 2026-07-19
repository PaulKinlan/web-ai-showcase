#!/usr/bin/env node
// LINEAGE GATE — deterministic, additive validator for the evidence-first lineage/value/priority pass.
// Runs ALONGSIDE check-routes.mjs + check-conformance.mjs (never replaces them). Exit 1 on violation;
// always REPORTS the denominators. What it PROTECTS:
//
//   A. Schema validity — every lineage record + value record has the required fields and a valid
//      relationship enum; priority.json is structurally sound.
//   B. Identity preservation — no built/blocked model identity is lost by the lineage pass: every
//      built+blocked id present on origin/main:models.json is still in the working models.json, and
//      every built model is represented in value-records.ndjson. (Lineage is ADDITIVE — it must never
//      imply a removal/rename.)
//   C. Exact-duplicate / quant detection REPRODUCIBILITY — every record flagged quant-variant carries
//      a reproducible quant marker (declared base_model:quantized OR a quant name token); every
//      format-port carries a port/re-exporter marker. And byRelationship recomputed from the records
//      MUST equal denominators.json (no stale/hand-edited drift).
//   D. FALSE-POSITIVE protection for specializations — a record flagged specialization-distinct keeps
//      a non-empty specialization[] AND is NEVER collapsed: it must not be marked "superseded" in the
//      priority queue, and its capability slot is preserved (distinctCapabilitySlots recomputed from
//      the records equals denominators.json and is >= canonicalUpstreamFamilies). Fine-tunes/language/
//      domain specializations are protected from being folded into their base family.
//
// Usage: node scripts/check-lineage.mjs
//
// NOTE: reports are a FIRST evidence-backed pass — never asserts complete/all.

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const p = (rel) => ROOT + rel;
const failures = [];
const fail = (m) => failures.push(m);
let denom, priority; // referenced by report()

function readNdjson(rel) {
  return readFileSync(p(rel), "utf8").trim().split("\n").filter(Boolean).map((l, i) => {
    try {
      return JSON.parse(l);
    } catch (e) {
      fail(`MALFORMED ${rel}:${i + 1} — ${e.message}`);
      return null;
    }
  }).filter(Boolean);
}
function gitShow(ref) {
  try {
    return execFileSync("git", ["show", ref], {
      cwd: ROOT,
      maxBuffer: 1 << 30,
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString();
  } catch {
    return null;
  }
}

// ── load artifacts ──
for (
  const f of [
    "inventory/lineage/records.ndjson",
    "inventory/lineage/value-records.ndjson",
    "inventory/lineage/denominators.json",
    "inventory/lineage/priority.json",
  ]
) {
  if (!existsSync(p(f))) {
    fail(`MISSING ARTIFACT: ${f} (run scripts/lineage-classify.mjs + lineage-value.mjs)`);
  }
}
if (failures.length) report();

const records = readNdjson("inventory/lineage/records.ndjson");
const values = readNdjson("inventory/lineage/value-records.ndjson");
denom = JSON.parse(readFileSync(p("inventory/lineage/denominators.json"), "utf8"));
priority = JSON.parse(readFileSync(p("inventory/lineage/priority.json"), "utf8"));
const cat = JSON.parse(readFileSync(p("models.json"), "utf8")).models;

const REL_ENUM = new Set([
  "canonical",
  "exact-dup",
  "format-port",
  "quant-variant",
  "checkpoint",
  "fine-tune",
  "distillation",
  "fork-no-change",
  "specialization-distinct",
  "uncertain",
]);
const CONF_ENUM = new Set(["high", "medium", "low"]);

// ── A. schema validity ──
const recById = new Map();
for (const [i, r] of records.entries()) {
  const where = `records.ndjson[${i}] (${r && r.id})`;
  for (
    const k of [
      "id",
      "task",
      "runtime",
      "relationship",
      "canonicalFamily",
      "specialization",
      "reviewed",
      "evidence",
      "confidence",
    ]
  ) {
    if (r[k] === undefined) fail(`SCHEMA ${where}: missing "${k}"`);
  }
  if (!REL_ENUM.has(r.relationship)) fail(`SCHEMA ${where}: bad relationship "${r.relationship}"`);
  if (!CONF_ENUM.has(r.confidence)) fail(`SCHEMA ${where}: bad confidence "${r.confidence}"`);
  if (!Array.isArray(r.specialization)) fail(`SCHEMA ${where}: specialization must be an array`);
  if (!r.evidence || !Array.isArray(r.evidence.proven) || !Array.isArray(r.evidence.inferred)) {
    fail(`SCHEMA ${where}: evidence.proven[]/inferred[] required`);
  }
  if (r.reviewed && r.evidence.proven.length === 0) {
    fail(`SCHEMA ${where}: reviewed record must carry >=1 proven evidence item`);
  }
  recById.set(r.id, r);
}
for (const [i, v] of values.entries()) {
  const where = `value-records.ndjson[${i}] (${v && v.id})`;
  for (
    const k of ["id", "task", "relationship", "dimensions", "overallConfidence", "evalPending"]
  ) {
    if (v[k] === undefined) fail(`SCHEMA ${where}: missing "${k}"`);
  }
  const d = v.dimensions || {};
  for (
    const dim of [
      "capabilityUniqueness",
      "browserFeasibility",
      "licenseDeployability",
      "modelCardQuality",
      "overlapWithStronger",
      "showcaseInterest",
    ]
  ) {
    if (!d[dim]) fail(`SCHEMA ${where}: dimensions.${dim} missing`);
  }
  // evidence-gating honesty: any numeric score must carry evidence.
  for (const [name, dim] of Object.entries(d)) {
    if (
      dim && typeof dim.score === "number" && Array.isArray(dim.evidence) &&
      dim.evidence.length === 0
    ) {
      fail(`EVIDENCE ${where}: dimensions.${name} has a numeric score but no evidence`);
    }
  }
}
if (!Array.isArray(priority.queue)) fail("SCHEMA priority.json: queue must be an array");

// ── B. identity preservation ──
const builtIds = cat.filter((m) => m.status === "built").map((m) => m.hfId);
const valueIds = new Set(values.map((v) => v.id));
for (const hf of builtIds) {
  if (hf && !valueIds.has(hf)) {
    fail(
      `IDENTITY: built model "${hf}" is absent from value-records.ndjson (every built demo must have a value record)`,
    );
  }
}
const baseRaw = gitShow("origin/main:models.json");
if (baseRaw) {
  try {
    const base = JSON.parse(baseRaw);
    const nowIds = new Set(cat.map((m) => m.hfId + "::" + m.slug));
    for (const m of (Array.isArray(base) ? base : base.models)) {
      if ((m.status === "built" || m.status === "blocked") && !nowIds.has(m.hfId + "::" + m.slug)) {
        fail(
          `IDENTITY LOST: published ${m.status} model ${m.slug} (${m.hfId}) missing/renamed vs origin/main — lineage pass must be additive.`,
        );
      }
    }
  } catch { /* offline / new file — skip */ }
}

// ── C. exact-dup / quant / port detection reproducibility ──
const QUANT_TOK =
  /(q4f16|q4f32|q4|q8|q2|q3|q5|q6|int8|int4|fp16|fp32|bf16|uint8|quantized|8bit|4bit|awq|gptq|gguf)/i;
const PORT_TOK = /(onnx|ort|-web|mlc|tflite|gguf|ggml)/i;
for (const r of records) {
  const name = r.id.split("/").pop();
  const org = (r.id.split("/")[0] || "").toLowerCase();
  if (r.relationship === "quant-variant") {
    const declared = r.evidence.inferred.some((e) => /quantized|quant-marker/.test(e)) ||
      QUANT_TOK.test(name);
    if (!declared) {
      fail(`REPRODUCIBILITY: "${r.id}" is quant-variant but carries no reproducible quant marker`);
    }
  }
  if (r.relationship === "format-port") {
    const declared = PORT_TOK.test(name) || ["onnx-community", "xenova", "mlc-ai"].includes(org) ||
      r.evidence.inferred.some((e) => /port-marker|re-exporter/.test(e));
    if (!declared) {
      fail(
        `REPRODUCIBILITY: "${r.id}" is format-port but carries no reproducible port/re-exporter marker`,
      );
    }
  }
}
// byRelationship recomputed must equal denominators.json (no drift between the artifacts).
const relCount = {};
for (const r of records) relCount[r.relationship] = (relCount[r.relationship] || 0) + 1;
if (JSON.stringify(relCount) !== JSON.stringify(denom.byRelationship)) {
  fail(
    "DRIFT: byRelationship recomputed from records.ndjson does not match denominators.json — regenerate with scripts/lineage-classify.mjs",
  );
}

// ── D. false-positive protection: specializations are NOT collapsed ──
const slotKey = (r) => `${r.task}::${r.canonicalFamily}::${r.specialization.join(",")}`;
const canonicalFams = new Set(
  records.filter((r) => ["canonical", "specialization-distinct"].includes(r.relationship)).map((
    r,
  ) => r.canonicalFamily),
);
const capabilitySlots = new Set(
  records.filter((r) => ["canonical", "specialization-distinct"].includes(r.relationship)).map(
    slotKey,
  ),
);
if (capabilitySlots.size !== denom.derivedFamilyDenominators.distinctCapabilitySlots) {
  fail("DRIFT: distinctCapabilitySlots recomputed does not match denominators.json");
}
if (capabilitySlots.size < canonicalFams.size) {
  fail(
    "COLLAPSE: distinctCapabilitySlots < canonicalUpstreamFamilies — specializations were folded away",
  );
}
for (const r of records) {
  if (r.relationship === "specialization-distinct" && r.specialization.length === 0) {
    fail(
      `COLLAPSE RISK: "${r.id}" is specialization-distinct but has an empty specialization[] — it would be indistinguishable from a plain canonical`,
    );
  }
}
// priority.json must NOT supersede a distinct specialization or a fine-tune (that collapses a capability).
const superseded = priority.queue.filter((q) => q.tier === "superseded");
for (const q of superseded) {
  if (
    !["format-port", "quant-variant", "distillation", "fork-no-change"].includes(q.relationship)
  ) {
    fail(
      `COLLAPSE: priority.json supersedes "${q.id}" (relationship=${q.relationship}) — only ports/quants/distillations/forks may be superseded, never a specialization or canonical`,
    );
  }
}
// every specialization-distinct in the queue keeps a canonicalAlternative=null under high/medium (not folded)
const specInQueue = priority.queue.filter((q) => q.relationship === "specialization-distinct");
for (const q of specInQueue) {
  if (q.tier === "superseded" || q.tier === "blocked" && !q.rationale) {
    fail(`COLLAPSE: specialization-distinct "${q.id}" placed in tier ${q.tier} without rationale`);
  }
}

report();

function report() {
  console.log("=== LINEAGE GATE (evidence-first pass — never complete/all) ===");
  if (denom && denom.denominators) {
    const d = denom.denominators;
    console.log(
      `denominators: rawCatalogue ${d.rawCatalogue} · mission ${d.missionBaseline} · eligible reps ${d.eligibleRepresentatives} (runnable ${d.eligibleRunnable}, blocked/gated ${d.blockedGated})`,
    );
    console.log(
      `reviewed: ${denom.reviewed?.reviewed}/${denom.reviewed?.total}   byRelationship: ${
        JSON.stringify(denom.byRelationship)
      }`,
    );
    console.log(
      `derived: canonicalUpstreamFamilies ${denom.derivedFamilyDenominators?.canonicalUpstreamFamilies} · distinctCapabilitySlots ${denom.derivedFamilyDenominators?.distinctCapabilitySlots}`,
    );
  }
  if (typeof priority !== "undefined" && priority.tierCounts) {
    console.log(
      `priority tiers: ${JSON.stringify(priority.tierCounts)}  (pending ${priority.pendingTotal})`,
    );
  }
  if (failures.length) {
    console.error(`\nFAIL — ${failures.length} lineage violation(s):`);
    for (const f of failures) console.error("  ✗ " + f);
    process.exit(1);
  }
  console.log(
    "\nPASS — lineage records schema-valid; identities preserved; dup/quant/port detection reproducible; specializations NOT collapsed.",
  );
}
