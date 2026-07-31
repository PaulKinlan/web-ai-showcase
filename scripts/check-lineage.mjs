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
import {
  findCatalogueCollision,
  isExactCatalogueMatch,
  validateInventorySummary,
  validateReviewedAliasPolicy,
} from "./inventory-lib.mjs";

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
    "inventory/eligible.ndjson",
    "inventory/summary.json",
    "inventory/collisions.json",
    "inventory/reviewed-aliases.json",
  ]
) {
  if (!existsSync(p(f))) {
    fail(`MISSING ARTIFACT: ${f} (run scripts/lineage-classify.mjs + lineage-value.mjs)`);
  }
}
if (failures.length) report();

const records = readNdjson("inventory/lineage/records.ndjson");
const values = readNdjson("inventory/lineage/value-records.ndjson");
const inventory = readNdjson("inventory/eligible.ndjson");
denom = JSON.parse(readFileSync(p("inventory/lineage/denominators.json"), "utf8"));
priority = JSON.parse(readFileSync(p("inventory/lineage/priority.json"), "utf8"));
const summary = JSON.parse(readFileSync(p("inventory/summary.json"), "utf8"));
const collisionLedger = JSON.parse(readFileSync(p("inventory/collisions.json"), "utf8"));
const reviewedAliasPolicy = JSON.parse(
  readFileSync(p("inventory/reviewed-aliases.json"), "utf8"),
);
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
      "runtimeVerification",
      "discoveryStatus",
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
for (const policyFailure of validateReviewedAliasPolicy(reviewedAliasPolicy)) {
  fail(`SCHEMA reviewed-aliases.json: ${policyFailure}`);
}

// ── A2. current inventory/summary/catalogue reconciliation ──
const discoveryStatuses = new Set(["verified-eligible", "candidate-unverified", "blocked"]);
for (const [i, model] of inventory.entries()) {
  const where = `eligible.ndjson[${i}] (${model.id})`;
  if (!model.discovery || !discoveryStatuses.has(model.discovery.status)) {
    fail(`SCHEMA ${where}: typed discovery.status is required`);
    continue;
  }
  if (
    !model.discovery.runtime || !model.discovery.browserArtifact || !model.discovery.feasibleSize
  ) {
    fail(`SCHEMA ${where}: typed runtime/artifact/size provenance is required`);
  }
  if (model.discovery.status === "candidate-unverified") {
    if (
      model.discovery.runtime?.verification !== "source-claimed" ||
      model.discovery.browserArtifact?.verification !== "unverified" ||
      model.discovery.feasibleSize?.verification !== "unverified"
    ) {
      fail(
        `HONESTY ${where}: candidate must keep runtime source-claimed and artifact/size unverified`,
      );
    }
  }
}
const inventoryStatusCounts = { "verified-eligible": 0, "candidate-unverified": 0, blocked: 0 };
for (const model of inventory) inventoryStatusCounts[model.discovery.status]++;
for (const drift of validateInventorySummary(summary, inventory, cat)) {
  fail(`DRIFT: ${drift}`);
}
const taskCounts = {};
for (const model of inventory) {
  const counts = taskCounts[model.task] ??= {
    families: 0,
    verifiedEligible: 0,
    candidateUnverified: 0,
    blocked: 0,
  };
  counts.families++;
  if (model.discovery.status === "verified-eligible") counts.verifiedEligible++;
  else if (model.discovery.status === "candidate-unverified") counts.candidateUnverified++;
  else counts.blocked++;
}
if (JSON.stringify(taskCounts) !== JSON.stringify(summary.byTask)) {
  fail("DRIFT: byTask recomputed from current inventory does not match inventory/summary.json");
}
const lineageIds = new Set(records.map((record) => record.id));
const inventoryIds = new Set(inventory.map((model) => model.id));
if (
  lineageIds.size !== inventoryIds.size ||
  [...inventoryIds].some((id) => !lineageIds.has(id))
) {
  fail("DRIFT: lineage records do not represent the exact current inventory snapshot");
}
const currentDenom = denom.denominators || {};
if (
  currentDenom.rawCatalogue !== cat.length ||
  currentDenom.inventoryRepresentatives !== inventory.length ||
  currentDenom.verifiedEligible !== inventoryStatusCounts["verified-eligible"] ||
  currentDenom.candidateUnverified !== inventoryStatusCounts["candidate-unverified"] ||
  currentDenom.blocked !== inventoryStatusCounts.blocked
) {
  fail("DRIFT: lineage denominators do not match current models.json + inventory snapshot");
}
if (
  summary.catalogue.rawCandidatesAdded !==
    summary.catalogue.addedThisRun + summary.catalogue.candidateCollisionsThisRun
) {
  fail("DRIFT: raw catalogue candidates must reconcile to additions + catalogue collisions");
}
if (!Array.isArray(collisionLedger.scan) || !Array.isArray(collisionLedger.catalogueMerge)) {
  fail("SCHEMA collisions.json: scan[] and catalogueMerge[] are required");
} else {
  if (
    collisionLedger.scan.length !== summary.scanFamilyCollisions ||
    collisionLedger.catalogueMerge.length !== summary.catalogue.candidateCollisionsThisRun
  ) {
    fail("DRIFT: collision ledger counts do not match inventory/summary.json");
  }
  const currentHf = new Set(cat.map((model) => model.hfId));
  for (const collision of collisionLedger.scan) {
    if (!inventoryIds.has(collision.kept) || inventoryIds.has(collision.removed)) {
      fail(
        `COLLISION LEDGER: scan collision ${collision.removed} -> ${collision.kept} is not reflected in inventory`,
      );
    }
  }
  for (const collision of collisionLedger.catalogueMerge) {
    if (!currentHf.has(collision.kept) || currentHf.has(collision.removed)) {
      fail(
        `COLLISION LEDGER: catalogue collision ${collision.removed} -> ${collision.kept} is not reflected in models.json`,
      );
    }
  }

  // Re-run the exact generator policy against the committed inventory/catalogue. This rejects a
  // hand-patched ledger, a stale reviewed policy, or generator drift that would append an alias on
  // the next refresh. Exact self-matches are already-appended candidates, not merge collisions.
  const generatedCatalogueMerge = [];
  for (const candidate of inventory) {
    if (candidate.discovery.status !== "candidate-unverified") continue;
    try {
      const collision = findCatalogueCollision(candidate, cat, reviewedAliasPolicy);
      if (!collision || isExactCatalogueMatch(candidate, collision)) continue;
      generatedCatalogueMerge.push({
        phase: "catalogue-candidate-merge",
        collisionKey: collision.collisionKey,
        kept: collision.model.hfId,
        keptStatus: collision.model.status,
        removed: candidate.id,
        reason: collision.reason,
      });
    } catch (error) {
      fail(`COLLISION POLICY: ${error.message}`);
    }
  }
  const collisionSort = (a, b) =>
    a.collisionKey.localeCompare(b.collisionKey) || a.removed.localeCompare(b.removed);
  generatedCatalogueMerge.sort(collisionSort);
  const committedCatalogueMerge = [...collisionLedger.catalogueMerge].sort(collisionSort);
  if (JSON.stringify(generatedCatalogueMerge) !== JSON.stringify(committedCatalogueMerge)) {
    fail(
      "DRIFT: catalogue collision ledger does not equal decisions generated from reviewed-aliases.json plus inventory collision rules",
    );
  }

  const committedByRemoved = new Map(
    collisionLedger.catalogueMerge.map((collision) => [collision.removed, collision]),
  );
  for (const alias of reviewedAliasPolicy.aliases || []) {
    const candidate = inventory.find((model) =>
      model.id === alias.candidateHfId && model.task === alias.task
    );
    if (!candidate) continue; // Retain reviewed policy even when a later scan no longer observes it.
    try {
      const generated = findCatalogueCollision(candidate, cat, reviewedAliasPolicy);
      const committed = committedByRemoved.get(alias.candidateHfId);
      if (
        generated?.collisionKey !== alias.collisionKey ||
        generated?.model.hfId !== alias.keptHfId ||
        generated?.reason !== alias.reason ||
        committed?.collisionKey !== alias.collisionKey ||
        committed?.kept !== alias.keptHfId ||
        committed?.reason !== alias.reason
      ) {
        fail(
          `DRIFT: reviewed alias policy/generator/ledger diverge for ${alias.candidateHfId}`,
        );
      }
    } catch (error) {
      fail(`COLLISION POLICY: ${error.message}`);
    }
  }
}

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
    const baseModels = Array.isArray(base) ? base : base.models;
    const nowIds = new Set(cat.map((m) => m.hfId + "::" + m.slug));
    for (const m of baseModels) {
      if ((m.status === "built" || m.status === "blocked") && !nowIds.has(m.hfId + "::" + m.slug)) {
        fail(
          `IDENTITY LOST: published ${m.status} model ${m.slug} (${m.hfId}) missing/renamed vs origin/main — lineage pass must be additive.`,
        );
      }
    }
    const baseHf = new Set(baseModels.map((model) => model.hfId));
    const addedModels = cat.filter((item) => !baseHf.has(item.hfId));
    if (addedModels.length !== summary.catalogue.addedThisRun) {
      fail(
        `DRIFT: summary addedThisRun=${summary.catalogue.addedThisRun} but models.json adds ${addedModels.length} vs origin/main`,
      );
    }
    for (const model of addedModels) {
      if (
        model.status !== "pending" || model.backend !== undefined || model.runtime !== undefined ||
        model.candidate?.status !== "unverified" || !model.candidate?.claimedRuntime ||
        model.candidate?.provenance?.browserArtifact?.verification !== "unverified" ||
        model.candidate?.provenance?.feasibleSize?.verification !== "unverified"
      ) {
        fail(
          `HONESTY: new catalogue candidate ${model.hfId} must be typed unverified without proven backend/runtime`,
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
      `denominators: rawCatalogue ${d.rawCatalogue} · mission ${d.missionBaseline} · inventory reps ${d.inventoryRepresentatives} (verified ${d.verifiedEligible}, candidate-unverified ${d.candidateUnverified}, blocked ${d.blocked})`,
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
