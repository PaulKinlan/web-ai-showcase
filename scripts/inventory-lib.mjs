const QUANTIZED_BASE_PREFIX = "base_model:quantized:";

export function declaredQuantizedBase(model) {
  const tag = (model.tags || []).find((value) => value.startsWith(QUANTIZED_BASE_PREFIX));
  return tag ? tag.slice(QUANTIZED_BASE_PREFIX.length) : null;
}

// Normalize display/repository noise while preserving capability-bearing words such as instruct,
// chat, python and language names. WebLLM precision spellings include q0/q3 and f16/f32 suffixes.
export function familyKey(id) {
  let name = id.split("/").pop().toLowerCase();
  name = name
    .replace(/[-_.](onnx|ort|web|mlc|gguf|ggml)\b/g, "")
    .replace(
      /[-_.](q[0-8](?:f(?:16|32)|bf16)?(?:[-_.]?\d+)?|int8|int4|fp16|fp32|bf16|uint8|quantized|8bit|4bit|awq|gptq)(?=$|[-_.])/g,
      "",
    )
    .replace(/[-_.]\d+(?:\.\d+)?b\b/g, "")
    .replace(/[-_.](base|small|tiny|mini|large|xl|xxl|medium|nano|micro)\b/g, "")
    .replace(/[-_.]v?\d+(?:\.\d+)*\b/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return name || id.toLowerCase();
}

export function familyIdentity(model) {
  const declaredBase = declaredQuantizedBase(model);
  if (model.runtime === "webllm" && declaredBase) {
    return `${model.task}::declared-base::${declaredBase.toLowerCase()}`;
  }
  return `${model.task}::family::${model.familyKey || familyKey(model.id)}`;
}

function preferredRepresentative(a, b) {
  if (a.downloads !== b.downloads) return a.downloads > b.downloads ? a : b;
  return a.id.localeCompare(b.id) <= 0 ? a : b;
}

export function deduplicateFamilies(models) {
  const representatives = new Map();
  const collisions = [];
  for (const model of models) {
    const declaredBase = declaredQuantizedBase(model);
    const normalized = {
      ...model,
      familyKey: model.familyKey ||
        familyKey(model.runtime === "webllm" && declaredBase ? declaredBase : model.id),
    };
    const key = familyIdentity(normalized);
    const current = representatives.get(key);
    if (!current) {
      representatives.set(key, normalized);
      continue;
    }
    const kept = preferredRepresentative(current, normalized);
    const removed = kept === current ? normalized : current;
    representatives.set(key, kept);
    collisions.push({
      phase: "scan-family-dedup",
      collisionKey: key,
      kept: kept.id,
      removed: removed.id,
      reason: key.includes("::declared-base::")
        ? "same declared quantized base model; precision/quantization port collapsed"
        : "same task and normalized family",
    });
  }
  return {
    representatives: [...representatives.values()].sort((a, b) =>
      b.downloads - a.downloads || a.id.localeCompare(b.id)
    ),
    collisions: collisions.map((collision) => ({
      ...collision,
      kept: representatives.get(collision.collisionKey).id,
    })).sort((a, b) =>
      a.collisionKey.localeCompare(b.collisionKey) || a.removed.localeCompare(b.removed)
    ),
  };
}

export function discoveryClassification(model) {
  if (model.gated) {
    return {
      status: "blocked",
      reason: "access-gated",
      runtime: { value: model.runtime, verification: "source-claimed" },
      browserArtifact: { verification: "unverified" },
      feasibleSize: { verification: "unverified", sizeMB: null },
    };
  }
  if (model.runtime === "mediapipe") {
    return {
      status: "verified-eligible",
      reason: "curated browser Task bundle",
      runtime: { value: "mediapipe", verification: "verified-curated" },
      browserArtifact: { verification: "verified-curated", kind: ".task/.tflite" },
      feasibleSize: { verification: "verified-curated", sizeMB: null },
    };
  }
  const artifactTag = (model.tags || []).find((tag) => tag === "onnx" || tag === "transformers.js");
  const mlcRepo = model.runtime === "webllm" && /-MLC$/i.test(model.id);
  return {
    status: "candidate-unverified",
    reason: "runtime artifact and browser-feasible size require file-level verification",
    runtime: { value: model.runtime, verification: "source-claimed" },
    browserArtifact: {
      verification: "unverified",
      sourceSignal: artifactTag || (mlcRepo ? "repository-name:-MLC" : null),
    },
    feasibleSize: { verification: "unverified", sizeMB: null },
  };
}

function catalogueFamilyKeys(model) {
  const values = [model.hfId, model.family].filter(Boolean).map(familyKey);
  return new Set(values.map((value) => `${model.task}::${value}`));
}

function candidateCollisionKeys(model) {
  const keys = new Set([`${model.task}::${model.familyKey || familyKey(model.id)}`]);
  const base = declaredQuantizedBase(model);
  if (base) keys.add(`${model.task}::${familyKey(base)}`);
  return keys;
}

function policyAliases(policy) {
  if (Array.isArray(policy)) return policy;
  return policy?.aliases ?? [];
}

export function validateReviewedAliasPolicy(policy) {
  const failures = [];
  if (policy?.schemaVersion !== 1) failures.push("schemaVersion must be 1");
  if (!Array.isArray(policy?.aliases)) {
    failures.push("aliases must be an array");
    return failures;
  }
  const collisionKeys = new Set();
  const candidates = new Set();
  for (const [index, alias] of policy.aliases.entries()) {
    const where = `aliases[${index}]`;
    for (const key of ["collisionKey", "task", "candidateHfId", "keptHfId", "reason"]) {
      if (typeof alias?.[key] !== "string" || !alias[key]) {
        failures.push(`${where}.${key} must be a non-empty string`);
      }
    }
    if (alias?.collisionKey && !alias.collisionKey.startsWith("reviewed-family::")) {
      failures.push(`${where}.collisionKey must start with reviewed-family::`);
    }
    if (collisionKeys.has(alias?.collisionKey)) {
      failures.push(`${where}.collisionKey duplicates ${alias.collisionKey}`);
    }
    collisionKeys.add(alias?.collisionKey);
    const candidateKey = `${alias?.task}::${alias?.candidateHfId}`;
    if (candidates.has(candidateKey)) {
      failures.push(`${where} duplicates candidate ${candidateKey}`);
    }
    candidates.add(candidateKey);
    if (alias?.candidateHfId && alias.candidateHfId === alias.keptHfId) {
      failures.push(`${where} must map distinct candidate and kept ids`);
    }
  }
  return failures;
}

export function findCatalogueCollision(candidate, catalogue, reviewedAliasPolicy = []) {
  // Reviewed capability-family aliases are durable decisions rather than heuristics. Apply them
  // before exact-id matching so a previously appended alias cannot mask policy drift.
  const reviewed = policyAliases(reviewedAliasPolicy).filter((alias) =>
    alias.candidateHfId === candidate.id && alias.task === candidate.task
  );
  if (reviewed.length > 1) {
    throw new Error(`ambiguous reviewed alias policy for ${candidate.task}::${candidate.id}`);
  }
  if (reviewed.length === 1) {
    const alias = reviewed[0];
    const kept = catalogue.find((model) => model.hfId === alias.keptHfId);
    if (!kept) {
      throw new Error(
        `reviewed alias ${alias.collisionKey} kept target is missing: ${alias.keptHfId}`,
      );
    }
    return {
      model: kept,
      collisionKey: alias.collisionKey,
      reason: alias.reason,
    };
  }

  const exact = catalogue.find((model) => model.hfId === candidate.id);
  if (exact) return { model: exact, collisionKey: `hfId::${candidate.id}`, reason: "exact-hf-id" };

  // Quantized ports must not create another pending identity when their declared base/family is
  // already built, blocked, or pending. Other non-quantized candidates remain distinct unless an
  // explicit reviewed alias above supplies the richer lineage decision.
  if (!declaredQuantizedBase(candidate)) return null;
  const candidateKeys = candidateCollisionKeys(candidate);
  const matches = [];
  for (const model of catalogue) {
    for (const key of catalogueFamilyKeys(model)) {
      if (candidateKeys.has(key)) matches.push({ model, collisionKey: key });
    }
  }
  const statusRank = { built: 0, blocked: 1, pending: 2 };
  matches.sort((a, b) =>
    (statusRank[a.model.status] ?? 3) - (statusRank[b.model.status] ?? 3) ||
    a.model.hfId.localeCompare(b.model.hfId)
  );
  return matches.length ? { ...matches[0], reason: "declared-quantized-base-family" } : null;
}

export function isExactCatalogueMatch(candidate, collision) {
  return collision?.reason === "exact-hf-id" && collision.model.hfId === candidate.id;
}

export function isRetainedInventoryCandidate(candidate, collision) {
  return isExactCatalogueMatch(candidate, collision) &&
    collision.model.candidate?.status === "unverified";
}

export function mergeCandidates(catalogue, representatives, reviewedAliasPolicy = []) {
  const added = [];
  const collisions = [];
  for (const candidate of representatives) {
    if (candidate.discovery.status !== "candidate-unverified") continue;
    const collision = findCatalogueCollision(candidate, catalogue, reviewedAliasPolicy);
    if (isExactCatalogueMatch(candidate, collision)) continue;
    if (collision) {
      collisions.push({
        phase: "catalogue-candidate-merge",
        collisionKey: collision.collisionKey,
        kept: collision.model.hfId,
        keptStatus: collision.model.status,
        removed: candidate.id,
        reason: collision.reason,
      });
      continue;
    }
    added.push(candidate);
  }
  collisions.sort((a, b) =>
    a.collisionKey.localeCompare(b.collisionKey) || a.removed.localeCompare(b.removed)
  );
  return { added, collisions };
}

export function slugify(id) {
  return id.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")
    .slice(0, 80);
}

export function validateInventorySummary(summary, inventory, catalogue) {
  const failures = [];
  const statusCounts = { "verified-eligible": 0, "candidate-unverified": 0, blocked: 0 };
  for (const model of inventory) {
    if (statusCounts[model.discovery?.status] === undefined) {
      failures.push(`unknown discovery status for ${model.id}`);
    } else statusCounts[model.discovery.status]++;
  }
  if (summary.rawDiscoveries !== summary.inventoryFamilies + summary.scanFamilyCollisions) {
    failures.push("rawDiscoveries does not reconcile with representatives plus scan collisions");
  }
  const expected = {
    inventoryFamilies: inventory.length,
    collectedRepresentativeFamilies: inventory.length,
    eligibleFamilies: statusCounts["verified-eligible"],
    verifiedEligibleFamilies: statusCounts["verified-eligible"],
    candidateUnverifiedFamilies: statusCounts["candidate-unverified"],
    blockedFamilies: statusCounts.blocked,
  };
  for (const [key, value] of Object.entries(expected)) {
    if (summary[key] !== value) {
      failures.push(`${key}=${summary[key]} but current inventory=${value}`);
    }
  }
  const catalogueExpected = {
    total: catalogue.length,
    built: catalogue.filter((model) => model.status === "built").length,
    pending: catalogue.filter((model) => model.status === "pending").length,
    blocked: catalogue.filter((model) => model.status === "blocked").length,
  };
  for (const [key, value] of Object.entries(catalogueExpected)) {
    if (summary.catalogue?.[key] !== value) {
      failures.push(`catalogue.${key}=${summary.catalogue?.[key]} but models.json=${value}`);
    }
  }
  return failures;
}
