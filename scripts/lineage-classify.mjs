#!/usr/bin/env node
// EVIDENCE-FIRST lineage / duplicate / variant / value classifier over the eligible universe.
//
// Establishes exact DENOMINATORS so the build queue can be prioritized by UNIQUE capability without
// collapsing materially-distinct specializations. Deterministic + additive: reads
//   inventory/eligible.ndjson            — the evidence-backed eligible representatives (family-deduped)
//   models.json                          — catalogue (built/pending/blocked status)
//   inventory/lineage/reviewed-meta.json — REAL HF metadata for a REVIEWED SAMPLE (proven evidence:
//                                          cardData.base_model, config.architectures, siblings)
// and writes (ADDITIVE, never mutates the inputs):
//   inventory/lineage/records.ndjson     — one lineage record per representative (reviewed=proven,
//                                          else heuristic w/ confidence)
//   inventory/lineage/denominators.json  — the exact bucket table + reviewed/total + confidence mix
//
// Separation of evidence is explicit: `evidence.proven[]` = API-confirmed (cardData.base_model /
// config.architectures / siblings / sha) for the reviewed sample; `evidence.inferred[]` =
// name/tag/familyKey heuristics for the long tail. NEVER claims complete/all — a first evidence pass.
//
// Run: `node scripts/lineage-classify.mjs`.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const p = (rel) => ROOT + rel;

// ── Inputs ──
const elig = readFileSync(p("inventory/eligible.ndjson"), "utf8").trim().split("\n").map((l) =>
  JSON.parse(l)
);
const cat = JSON.parse(readFileSync(p("models.json"), "utf8")).models;
const reviewed = existsSync(p("inventory/lineage/reviewed-meta.json"))
  ? JSON.parse(readFileSync(p("inventory/lineage/reviewed-meta.json"), "utf8"))
  : {};

const statusByHf = new Map(cat.map((m) => [m.hfId, m.status]));
const builtHf = new Set(cat.filter((m) => m.status === "built").map((m) => m.hfId));
const blockedHf = new Set(cat.filter((m) => m.status === "blocked").map((m) => m.hfId));

// ── Re-exporter orgs / port + quant markers (format-port + quant heuristics) ──
const REEXPORT_ORGS = new Set(["onnx-community", "xenova", "mlc-ai"]);
const PORT_NAME = /(-|_|\.)(onnx|ort|web|mlc|tflite|gguf|ggml)\b/i;
const QUANT_NAME =
  /(-|_|\.)(q4f16|q4f32|q4|q8|q2|q3|q5|q6|int8|int4|fp16|fp32|bf16|uint8|quantized|8bit|4bit|awq|gptq|gguf)\b/i;
const DISTIL_NAME = /(distil|-tiny-distilled|distilled)/i;

// Domain / task specialization signals — mark a rep as materially DISTINCT so it is NOT collapsed
// into its base family for prioritization. Boundary-safe regexes (avoid substring traps like "code"
// in "encoder" or "ner" in "generation"). Embedding-family names (e5/gte/bge/…) are intentionally
// EXCLUDED — canonicalFamily already keeps those distinct. (False-positive-safe: over-flagging keeps
// a capability visible; the validator protects flagged specializations from collapse.)
const DOMAIN_RE = [
  ["domain:finance", /financ|fin[-_]?bert|fingpt/],
  ["domain:legal", /legal|law[-_]/],
  ["domain:medical", /clinical|medical|biomed|bio[-_]|biobert|clinicalbert|pubmed|radiolog/],
  ["domain:scientific", /scibert|scientific|patent/],
  [
    "domain:code",
    /codebert|codegen|codet5|starcoder|[-_]code\b|code[-_]|text2sql|[-_]sql\b|sql[-_]/,
  ],
  ["domain:chemistry", /chemistry|chem[-_]|molecul|protein/],
  ["task:toxicity", /toxic|hate[-_]?speech|offensive|abusive/],
  ["task:spam", /[-_]?spam\b|phishing/],
  ["task:emotion", /emotion|speech[-_]emotion/],
  ["task:sentiment", /sentiment/],
  ["task:formality", /formality/],
  ["task:grammar", /grammar|grammatical/],
  ["task:ner", /named[-_]entity|[-_]ner[-_]|[-_]ner$|^ner[-_]|[-_]pos[-_]tag/],
  ["task:pii", /[-_]pii\b|[-_]pii[-_]|redact|anonymi/],
  ["task:irony", /sarcasm|irony|ironic/],
  ["domain:social", /twitter|tweet|reddit/],
  ["domain:commerce", /invoice|receipt|resume|ecommerce|e-commerce|product[-_]review/],
  ["task:intent", /[-_]intent[-_]|intent[-_]class|topic[-_]class/],
  ["task:langid", /language[-_]detect|lang[-_]?id|langid/],
];
// Language codes that, when the model is tuned FOR them, signal a distinct language specialization.
const NONEN_LANG = new Set([
  "zh",
  "ja",
  "ko",
  "fr",
  "de",
  "es",
  "it",
  "pt",
  "ru",
  "ar",
  "hi",
  "nl",
  "pl",
  "tr",
  "vi",
  "th",
  "id",
  "fa",
  "he",
  "uk",
  "cs",
  "ro",
  "sv",
  "fi",
  "da",
  "no",
  "hu",
  "el",
  "bn",
  "ta",
  "te",
  "ur",
  "ml",
  "mr",
  "multilingual",
]);

function orgOf(id) {
  return id.includes("/") ? id.split("/")[0].toLowerCase() : "";
}
function nameOf(id) {
  return id.split("/").pop();
}

// Base-model tag → { relKind, baseId }[] (from eligible.ndjson tags: base_model:<kind>:<id> or base_model:<id>)
function baseTags(m) {
  const out = [];
  for (const t of m.tags || []) {
    if (!t.startsWith("base_model:")) continue;
    const rest = t.slice("base_model:".length);
    const kinds = ["quantized", "finetune", "adapter", "merge"];
    const firstSeg = rest.split(":")[0];
    if (kinds.includes(firstSeg)) {
      out.push({ relKind: firstSeg, baseId: rest.slice(firstSeg.length + 1) });
    } else out.push({ relKind: "bare", baseId: rest });
  }
  return out;
}

// Detect a language / domain specialization from id + tags + reviewed language.
function specializationSignal(m, rev) {
  const s = m.id.toLowerCase();
  const signals = [];
  for (const [label, re] of DOMAIN_RE) if (re.test(s)) signals.push(label);
  // language: reviewed cardData.language, else language: tags
  let langs = [];
  if (rev && rev.language) langs = Array.isArray(rev.language) ? rev.language : [rev.language];
  for (const t of m.tags || []) if (/^[a-z]{2,3}$/.test(t) && NONEN_LANG.has(t)) langs.push(t);
  const nonEn = [
    ...new Set(langs.map((l) => String(l).toLowerCase()).filter((l) => NONEN_LANG.has(l))),
  ];
  if (nonEn.length > 3 || nonEn.includes("multilingual")) signals.push(`lang:multilingual`);
  else if (nonEn.length) signals.push(`lang:${nonEn.sort().join("+")}`);
  return signals;
}

// Canonical family: reviewed config.model_type (proven) → coverage-style arch pattern → familyKey.
function canonicalFamily(m, rev) {
  if (rev && rev.config && rev.config.model_type) return rev.config.model_type.toLowerCase();
  if (rev && rev.config && Array.isArray(rev.config.architectures) && rev.config.architectures[0]) {
    return rev.config.architectures[0].toLowerCase();
  }
  return m.familyKey || m.id.toLowerCase();
}

let reviewedCount = 0;
const records = [];

for (const m of elig) {
  const rev = reviewed[m.id] && !reviewed[m.id].error ? reviewed[m.id] : null;
  if (rev) reviewedCount++;
  const org = orgOf(m.id);
  const name = nameOf(m.id);
  const bts = baseTags(m);
  const revBase = rev && rev.base_model
    ? (Array.isArray(rev.base_model) ? rev.base_model : [rev.base_model])
    : [];
  const proven = [];
  const inferred = [];

  // ── proven evidence (reviewed sample only) ──
  if (rev) {
    if (revBase.length) proven.push(`cardData.base_model=${revBase.join(",")}`);
    if (rev.config && rev.config.architectures) {
      proven.push(`config.architectures=${rev.config.architectures.join(",")}`);
    }
    if (rev.config && rev.config.model_type) {
      proven.push(`config.model_type=${rev.config.model_type}`);
    }
    if (rev.hasOnnx) proven.push("siblings:onnx-present");
    if (rev.hasTokenizer) proven.push("siblings:tokenizer-present");
    if (rev.sha) proven.push(`sha=${rev.sha.slice(0, 12)}`);
  }

  // ── inferred evidence (heuristics) ──
  const quantTag = bts.find((b) => b.relKind === "quantized");
  const ftTag = bts.find((b) => b.relKind === "finetune");
  const bareTag = bts.find((b) => b.relKind === "bare");
  const adapterTag = bts.find((b) => b.relKind === "adapter" || b.relKind === "merge");
  const bestBaseId = (revBase[0]) ||
    (quantTag && quantTag.baseId) || (ftTag && ftTag.baseId) || (bareTag && bareTag.baseId) ||
    (adapterTag && adapterTag.baseId) || null;
  if (bts.length) {
    inferred.push(
      `tags.base_model=[${bts.map((b) => b.relKind + ":" + b.baseId).join(" | ").slice(0, 160)}]`,
    );
  }

  const spec = specializationSignal(m, rev);
  if (spec.length) inferred.push(`specialization=${spec.join(",")}`);

  const isReexporter = REEXPORT_ORGS.has(org);
  const portName = PORT_NAME.test(name);
  const quantName = QUANT_NAME.test(name);
  const distilName = DISTIL_NAME.test(name);
  if (portName) inferred.push(`name:port-marker`);
  if (quantName) inferred.push(`name:quant-marker`);
  if (isReexporter) inferred.push(`org:re-exporter(${org})`);

  // ── relationship decision (deterministic, most-specific-first) ──
  // Whether the rep shares its base's architecture (proven arch OR declared base with matching family root)
  let relationship, confidence;

  const hasBase = revBase.length > 0 || bts.length > 0;
  const isSpecialization = spec.length > 0;

  if (adapterTag) {
    relationship = "fine-tune"; // LoRA/adapter/merge → derivative
    inferred.push(`rel:${adapterTag.relKind}`);
    confidence = "medium";
  } else if (quantTag || (quantName && hasBase)) {
    relationship = "quant-variant";
    confidence = quantTag ? (rev ? "high" : "medium") : "low";
  } else if ((isReexporter || portName) && hasBase && !isSpecialization) {
    relationship = "format-port";
    confidence = rev && revBase.length ? "high" : "medium";
  } else if (distilName) {
    relationship = "distillation";
    confidence = "medium";
  } else if (hasBase && isSpecialization) {
    relationship = "specialization-distinct";
    confidence = rev && revBase.length ? "high" : "medium";
  } else if (ftTag || bareTag || revBase.length) {
    // derived, no domain/lang shift detected → treat as fine-tune/checkpoint of the base family
    relationship = "fine-tune";
    confidence = rev && revBase.length ? "high" : "medium";
  } else if (isSpecialization) {
    // its own upstream, but a domain/language-specific canonical (e.g. a multilingual embedder)
    relationship = "specialization-distinct";
    confidence = "low";
  } else {
    // no base declared. If reviewed with a real architecture, it's a canonical upstream; else uncertain.
    if (rev && rev.config && (rev.config.architectures || rev.config.model_type)) {
      relationship = "canonical";
      confidence = "high";
    } else if (m.downloads > 50000) {
      relationship = "canonical"; // high-usage upstream w/o declared base — canonical by convention
      confidence = "low";
    } else {
      relationship = "uncertain";
      confidence = "low";
    }
  }

  // Resolve canonical base root for lineage clustering (the upstream identity, if known).
  const baseRoot = bestBaseId ? bestBaseId.split("/").pop().toLowerCase() : null;

  const status = statusByHf.get(m.id) ||
    (m.gated ? "blocked-gated" : "pending-or-uncatalogued");

  records.push({
    id: m.id,
    task: m.task,
    modality: m.modality,
    runtime: m.runtime,
    license: m.license,
    downloads: m.downloads,
    likes: m.likes,
    gated: m.gated,
    reviewed: !!rev,
    canonicalFamily: canonicalFamily(m, rev),
    base_model: bestBaseId,
    baseRoot,
    relationship,
    specialization: spec,
    catalogueStatus: status,
    revision: rev && rev.sha ? rev.sha.slice(0, 12) : null,
    sourceRepo: `https://huggingface.co/${m.id}`,
    evidence: { proven, inferred },
    confidence,
  });
}

// ── Denominator table ──
const byRel = {};
const byRelConf = {};
for (const r of records) {
  byRel[r.relationship] = (byRel[r.relationship] || 0) + 1;
  const k = `${r.relationship}/${r.confidence}`;
  byRelConf[k] = (byRelConf[k] || 0) + 1;
}
const byStatus = {};
for (const r of records) byStatus[r.catalogueStatus] = (byStatus[r.catalogueStatus] || 0) + 1;

// Canonical upstream/base families = distinct canonicalFamily among reps NOT themselves a
// port/quant/fine-tune of another (i.e. relationship canonical or specialization-distinct).
const canonicalFamilies = new Set(
  records.filter((r) => ["canonical", "specialization-distinct"].includes(r.relationship))
    .map((r) => r.canonicalFamily),
);
// Distinct capability slots = (task + canonicalFamily) for canonical/specialization + each distinct
// specialization; ports/quants/fine-tunes-of-same collapse to their base capability.
const capabilitySlots = new Set(
  records.filter((r) => ["canonical", "specialization-distinct"].includes(r.relationship)).map((
    r,
  ) => `${r.task}::${r.canonicalFamily}::${r.specialization.join(",")}`),
);

const denominators = {
  generatedAt: new Date().toISOString().slice(0, 10),
  disclaimer:
    "FIRST evidence-backed pass — NOT complete/all. Reviewed sample has PROVEN metadata (HF API: " +
    "cardData.base_model, config.architectures, siblings); the long tail is classified by cheap " +
    "heuristics (tags/name/familyKey) with confidence markers. Denominators are refining lower " +
    "bounds at the current scan depth, kept SEPARATE from the raw catalogue and mission denominators.",
  denominators: {
    rawCatalogue: cat.length,
    rawCatalogueNote:
      "models.json — the full catalogue denominator (built+pending+blocked); KEPT INTACT.",
    missionBaseline: 635,
    missionBaselineNote:
      "eligibleFamilies @ --pages 8 (original mission denominator); KEPT SEPARATE.",
    eligibleRepresentatives: records.length,
    eligibleRepresentativesNote:
      "family-deduped representatives in inventory/eligible.ndjson at the current scan depth (a " +
      "refining lower bound; exact byte/config dups + quant/format ports already collapse WITHIN " +
      "familyKey before this layer).",
    eligibleRunnable: records.filter((r) => !r.gated).length,
    blockedGated: records.filter((r) => r.gated).length,
  },
  reviewed: { reviewed: reviewedCount, total: records.length },
  byRelationship: byRel,
  byRelationshipConfidence: byRelConf,
  byCatalogueStatus: byStatus,
  derivedFamilyDenominators: {
    canonicalUpstreamFamilies: canonicalFamilies.size,
    distinctCapabilitySlots: capabilitySlots.size,
    note:
      "canonicalUpstreamFamilies = distinct canonicalFamily among canonical + specialization-distinct " +
      "reps (ports/quants/fine-tunes fold into their base). distinctCapabilitySlots additionally keeps " +
      "each materially-distinct domain/language specialization as its own slot — these are NOT collapsed.",
  },
  bucketGlossary: {
    canonical:
      "upstream/base family — no declared derivation (or a domain/lang upstream); the head of a lineage.",
    "exact-dup":
      "proven identical bytes/config to another rep (near-zero here: familyKey already collapses exact dups pre-representative).",
    "format-port":
      "runtime re-export (ONNX/ORT/MLC/TFLite) of an upstream — same capability, portability only.",
    "quant-variant":
      "quantization/precision variant (declared base_model:quantized or quant name marker).",
    checkpoint:
      "size/checkpoint variant of a family (folds under familyKey; surfaced when a distinct rep).",
    "fine-tune": "derivative training on the same task/domain with no material capability shift.",
    distillation: "distilled student of a larger teacher.",
    "fork-no-change": "mirror/copy with no material capability change.",
    "specialization-distinct":
      "shares a base but delivers a materially-distinct domain/language/task capability — NOT collapsed.",
    uncertain: "insufficient metadata to classify confidently.",
  },
};

writeFileSync(
  p("inventory/lineage/records.ndjson"),
  records.map((r) => JSON.stringify(r)).join("\n") + "\n",
);
writeFileSync(
  p("inventory/lineage/denominators.json"),
  JSON.stringify(denominators, null, 2) + "\n",
);

console.error("=== LINEAGE CLASSIFY ===");
console.error(`records: ${records.length}  reviewed: ${reviewedCount}`);
console.error("byRelationship:", JSON.stringify(byRel));
console.error(
  "canonical upstream families:",
  canonicalFamilies.size,
  " capability slots:",
  capabilitySlots.size,
);
console.error("→ inventory/lineage/records.ndjson + denominators.json");
