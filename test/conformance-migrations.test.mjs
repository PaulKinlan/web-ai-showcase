// Deterministic tests for the conformance-migration vocabulary (web-ai-showcase-9tw).
//
// The gap these cover: check-conformance.mjs honoured only remove|weaken, so a FACTUAL CORRECTION
// of an assertion derived from wrong metadata (the mms-tts-bengali declares-quantisation case) could
// only be recorded by mislabelling it "weaken" — the opposite of what happened, polluting the audit
// trail the immutability rule protects. The label is the trail, so intent has to be expressible and
// the vocabulary must not drift between the gate, the validator, and the published schema.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  CONFORMANCE_MIGRATION_ACTIONS,
  migratedAssertion,
  validateConformanceMigrations,
} from "../scripts/conformance-lib.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const SCHEMA_PATH = join(root, "schemas", "conformance-migration.schema.json");
const DATA_PATH = join(root, "conformance-migrations.json");

const record = (over = {}) => ({
  suiteId: "fixture-suite",
  assertionId: "declares-quantisation",
  action: "correct",
  reason: "The assertion was derived from wrong catalogue metadata; the demo always loaded fp32.",
  evidence: "models/fixture-suite/worker.js uses dtype fp32; reports run record shows fp32.",
  date: "2026-09-24",
  ...over,
});

test("vocabulary is exactly remove | weaken | correct", () => {
  assert.deepEqual(CONFORMANCE_MIGRATION_ACTIONS, ["remove", "weaken", "correct"]);
});

test("a 'correct' record is honoured (the case that could not be recorded before)", () => {
  assert.equal(migratedAssertion([record()], "fixture-suite", "declares-quantisation"), true);
  assert.equal(migratedAssertion([record()], "other-suite", "declares-quantisation"), false);
  assert.equal(migratedAssertion([record()], "fixture-suite", "other-assertion"), false);
});

test("remove and weaken stay honoured; an unknown action is ignored", () => {
  for (const action of ["remove", "weaken", "correct"]) {
    assert.equal(
      migratedAssertion([record({ action })], "fixture-suite", "declares-quantisation"),
      true,
      `${action} must be honoured`,
    );
  }
  for (const action of ["fix", "identity-change", "weakened", "CORRECT", ""]) {
    assert.equal(
      migratedAssertion([record({ action })], "fixture-suite", "declares-quantisation"),
      false,
      `${JSON.stringify(action)} must NOT be honoured`,
    );
  }
});

test("a correction or weakening must be argued: reason and evidence are enforced", () => {
  assert.deepEqual(validateConformanceMigrations([record()]), []);
  assert.deepEqual(validateConformanceMigrations([record({ action: "remove" })]), []);

  const noReason = validateConformanceMigrations([record({ reason: "too short" })]);
  assert.ok(
    noReason.some((e) => /reason must be a substantive sentence/.test(e)),
    noReason.join("; "),
  );

  for (const action of ["correct", "weaken"]) {
    const noEvidence = validateConformanceMigrations([record({ action, evidence: "" })]);
    assert.ok(
      noEvidence.some((e) => new RegExp(`action "${action}" needs evidence`).test(e)),
      `expected an evidence error for ${action}: ${noEvidence.join("; ")}`,
    );
  }
});

test("structural rules: action, date, hashes, duplicates, non-array", () => {
  assert.deepEqual(validateConformanceMigrations([]), []);
  assert.deepEqual(validateConformanceMigrations(undefined), [
    "conformance-migrations.json must be an array",
  ]);
  assert.ok(
    validateConformanceMigrations([record({ action: "fix" })])
      .some((e) => /not one of remove\|weaken\|correct/.test(e)),
  );
  assert.ok(
    validateConformanceMigrations([record({ date: "24/09/2026" })])
      .some((e) => /date must be YYYY-MM-DD/.test(e)),
  );
  assert.ok(
    validateConformanceMigrations([record({ fromSuiteHash: "sha256:short" })])
      .some((e) => /fromSuiteHash must be sha256/.test(e)),
  );
  assert.ok(
    validateConformanceMigrations([record({ suiteId: "" })])
      .some((e) => /missing suiteId/.test(e)),
  );
  const dup = validateConformanceMigrations([record(), record()]);
  assert.ok(dup.some((e) => /duplicate correct record/.test(e)), dup.join("; "));
  assert.ok(
    validateConformanceMigrations([record({ toTest: 42 })])
      .some((e) => /toTest must be a string/.test(e)),
  );
});

test("FAIL-CLOSED: the published schema and the code vocabulary cannot drift", () => {
  const schema = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));
  assert.deepEqual(
    schema.items.properties.action.enum,
    CONFORMANCE_MIGRATION_ACTIONS,
    "schemas/conformance-migration.schema.json action enum must mirror CONFORMANCE_MIGRATION_ACTIONS",
  );
  for (const required of ["suiteId", "assertionId", "action", "reason", "date"]) {
    assert.ok(schema.items.required.includes(required), `schema must require ${required}`);
  }
  // The schema documents the evidence requirement in prose; keep the two in step.
  assert.ok(/correct/.test(schema.items.properties.evidence.description));
  assert.ok(/correct/.test(schema.description));
});

test("the committed conformance-migrations.json satisfies its own schema", () => {
  const data = JSON.parse(readFileSync(DATA_PATH, "utf8"));
  const errs = validateConformanceMigrations(data);
  assert.deepEqual(errs, [], errs.join("; "));
  // The bengali correction is the record that motivated the vocabulary: it must be labelled
  // "correct", never "weaken" (a weakening would misreport what happened).
  const bengali = data.find((r) =>
    r.suiteId === "mms-tts-bengali" && r.assertionId === "declares-quantisation"
  );
  assert.ok(bengali, "expected the mms-tts-bengali declares-quantisation record");
  assert.equal(bengali.action, "correct");
  assert.ok(bengali.evidence.length > 20, "a correction must cite evidence");
  assert.equal(migratedAssertion(data, bengali.suiteId, bengali.assertionId), true);
});
