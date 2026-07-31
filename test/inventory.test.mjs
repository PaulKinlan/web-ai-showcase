import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  declaredQuantizedBase,
  deduplicateFamilies,
  discoveryClassification,
  familyKey,
  findCatalogueCollision,
  mergeCandidates,
  validateInventorySummary,
  validateReviewedAliasPolicy,
} from "../scripts/inventory-lib.mjs";

const ROOT = new URL("../", import.meta.url);

function webllm(id, base, downloads) {
  return {
    id,
    task: "text-generation",
    modality: "text",
    runtime: "webllm",
    downloads,
    likes: 0,
    gated: false,
    tags: [`base_model:quantized:${base}`],
  };
}

test("WebLLM q0/q3/q4 precision ports collapse by declared base model", () => {
  const models = [
    webllm("mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC", "meta-llama/Llama-3.2-1B-Instruct", 10),
    webllm("mlc-ai/Llama-3.2-1B-Instruct-q3f16_1-MLC", "meta-llama/Llama-3.2-1B-Instruct", 20),
    webllm("mlc-ai/Llama-3.2-1B-Instruct-q4f32_1-MLC", "meta-llama/Llama-3.2-1B-Instruct", 30),
  ];
  const result = deduplicateFamilies(models);
  assert.deepEqual(result.representatives.map((model) => model.id), [models[2].id]);
  assert.equal(result.collisions.length, 2);
  assert.ok(result.collisions.every((item) => item.collisionKey.includes("declared-base")));
});

test("representative selection is deterministic when downloads tie", () => {
  const models = [
    webllm("mlc-ai/Z-q4f16_1-MLC", "owner/base", 10),
    webllm("mlc-ai/A-q0f32-MLC", "owner/base", 10),
  ];
  assert.equal(deduplicateFamilies(models).representatives[0].id, "mlc-ai/A-q0f32-MLC");
  assert.equal(deduplicateFamilies(models.reverse()).representatives[0].id, "mlc-ai/A-q0f32-MLC");
});

test("quantized candidate merge rejects built and pending base-family collisions", () => {
  const candidate = webllm(
    "mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    1,
  );
  candidate.familyKey = familyKey("deepseek-ai/DeepSeek-R1-Distill-Llama-8B");
  const built = [{
    hfId: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    task: "text-generation",
    family: "deepseek-r1-distill-llama-webllm",
    status: "built",
  }];
  assert.equal(findCatalogueCollision(candidate, built)?.model.status, "built");

  const pending = [{
    hfId: "mlc-ai/QwQ-32B-q4f16_1-MLC",
    task: "text-generation",
    family: "qwq",
    status: "pending",
  }];
  const qwq = webllm("mlc-ai/QwQ-32B-q4f16_0-MLC", "Qwen/QwQ-32B", 1);
  qwq.familyKey = familyKey("Qwen/QwQ-32B");
  assert.equal(findCatalogueCollision(qwq, pending)?.model.status, "pending");
});

test("all 21 reviewed family aliases are durable merge policy", async () => {
  const policy = JSON.parse(
    await readFile(new URL("inventory/reviewed-aliases.json", ROOT), "utf8"),
  );
  const inventory = (await readFile(new URL("inventory/eligible.ndjson", ROOT), "utf8"))
    .trim().split("\n").map(JSON.parse);
  const catalogue = JSON.parse(
    await readFile(new URL("models.json", ROOT), "utf8"),
  ).models;

  assert.deepEqual(validateReviewedAliasPolicy(policy), []);
  assert.equal(policy.aliases.length, 21);
  const reviewedCandidates = policy.aliases.map((alias) => {
    const candidate = inventory.find((model) =>
      model.id === alias.candidateHfId && model.task === alias.task
    );
    assert.ok(candidate, `inventory contains reviewed candidate ${alias.candidateHfId}`);
    const collision = findCatalogueCollision(candidate, catalogue, policy);
    assert.equal(collision?.collisionKey, alias.collisionKey);
    assert.equal(collision?.model.hfId, alias.keptHfId);
    assert.equal(collision?.reason, alias.reason);
    return candidate;
  });

  const nonQuantAliases = reviewedCandidates.filter((candidate) =>
    !declaredQuantizedBase(candidate)
  );
  assert.equal(nonQuantAliases.length, 19);
  for (
    const required of [
      "albert/albert-base-v2",
      "sentence-transformers/all-mpnet-base-v2",
      "facebook/bart-large-mnli",
    ]
  ) {
    assert.ok(nonQuantAliases.some((candidate) => candidate.id === required));
  }
  const merged = mergeCandidates(catalogue, nonQuantAliases, policy);
  assert.deepEqual(merged.added, [], "reviewed non-quant aliases cannot be re-appended");
  assert.equal(merged.collisions.length, 19);
  assert.ok(
    merged.collisions.every((collision) => collision.collisionKey.startsWith("reviewed-family::")),
  );

  const regenerated = mergeCandidates(catalogue, inventory, policy);
  const committedLedger = JSON.parse(
    await readFile(new URL("inventory/collisions.json", ROOT), "utf8"),
  ).catalogueMerge;
  assert.deepEqual(regenerated.added, [], "equivalent refresh cannot append any current candidate");
  assert.deepEqual(regenerated.collisions, committedLedger);
  assert.equal(
    regenerated.collisions.filter((collision) =>
      collision.collisionKey.startsWith("reviewed-family::")
    ).length,
    21,
  );
});

test("reviewed alias policy fails closed when its kept target is absent", () => {
  const policy = {
    schemaVersion: 1,
    aliases: [{
      collisionKey: "reviewed-family::albert",
      task: "fill-mask",
      candidateHfId: "albert/albert-base-v2",
      keptHfId: "Xenova/albert-base-v2",
      reason: "reviewed existing capability family",
    }],
  };
  assert.deepEqual(validateReviewedAliasPolicy(policy), []);
  assert.throws(
    () =>
      findCatalogueCollision(
        { id: "albert/albert-base-v2", task: "fill-mask", tags: [] },
        [],
        policy,
      ),
    /kept target is missing/,
  );
});

test("metadata and tags alone remain candidate-unverified", () => {
  const metadataOnly = {
    id: "Qwen/Qwen3-ASR-0.6B",
    runtime: "transformers.js",
    gated: false,
    tags: ["transformers.js", "safetensors"],
  };
  assert.deepEqual(discoveryClassification(metadataOnly), {
    status: "candidate-unverified",
    reason: "runtime artifact and browser-feasible size require file-level verification",
    runtime: { value: "transformers.js", verification: "source-claimed" },
    browserArtifact: { verification: "unverified", sourceSignal: "transformers.js" },
    feasibleSize: { verification: "unverified", sizeMB: null },
  });
});

test("lineage summary reconciliation detects stale inventory and catalogue denominators", () => {
  const inventory = [{ id: "candidate", discovery: { status: "candidate-unverified" } }];
  const catalogue = [{ hfId: "candidate", status: "pending" }];
  const current = {
    rawDiscoveries: 1,
    scanFamilyCollisions: 0,
    inventoryFamilies: 1,
    collectedRepresentativeFamilies: 1,
    eligibleFamilies: 0,
    verifiedEligibleFamilies: 0,
    candidateUnverifiedFamilies: 1,
    blockedFamilies: 0,
    catalogue: { total: 1, built: 0, pending: 1, blocked: 0 },
  };
  assert.deepEqual(validateInventorySummary(current, inventory, catalogue), []);
  assert.match(
    validateInventorySummary({ ...current, inventoryFamilies: 2435 }, inventory, catalogue).join(
      "\n",
    ),
    /inventoryFamilies=2435/,
  );
  assert.match(
    validateInventorySummary(
      { ...current, catalogue: { ...current.catalogue, total: 2495 } },
      inventory,
      catalogue,
    ).join("\n"),
    /catalogue\.total=2495/,
  );
});

test("curated MediaPipe entries are the only scan-time verified eligible type", () => {
  const result = discoveryClassification({
    id: "mediapipe/face-landmarker",
    runtime: "mediapipe",
    gated: false,
    tags: [],
  });
  assert.equal(result.status, "verified-eligible");
  assert.equal(result.runtime.verification, "verified-curated");
});
