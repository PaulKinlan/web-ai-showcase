import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { execSync } from "node:child_process";

const ROOT = new URL("..", import.meta.url).pathname;
const ALLOWLIST_PATH = join(ROOT, "scripts/runtime-pin-allowlist.json");
const PIN_SCAN_TARGETS = "models/ lib/ public/ scripts/ search/ models.json sw.js";

test("runtime-pin-allowlist.json exists, parses, and has required structure", () => {
  assert.ok(existsSync(ALLOWLIST_PATH), "missing scripts/runtime-pin-allowlist.json");
  const data = JSON.parse(readFileSync(ALLOWLIST_PATH, "utf8"));
  assert.ok(data.transformers?.shared, "transformers.shared must be defined");
  assert.ok(
    Array.isArray(data.transformers?.allowedLocalOverrides),
    "allowedLocalOverrides must be an array",
  );
  assert.ok(
    Array.isArray(data.onnxruntimeWeb?.allowedVersions),
    "onnxruntimeWeb.allowedVersions must be an array",
  );
  assert.ok(data.webLlm?.shared, "webLlm.shared must be defined");
  assert.ok(data.mediapipe?.shared, "mediapipe.shared must be defined");

  for (const o of data.transformers.allowedLocalOverrides) {
    assert.match(
      o.version,
      /^[0-9]+\.[0-9]+\.[0-9]+$/,
      `override version ${o.version} must be semver`,
    );
    assert.ok(
      Array.isArray(o.slugs) && o.slugs.length > 0,
      "override entry needs non-empty slugs array",
    );
    assert.ok(String(o.reason).length > 10, "override entry needs a documented reason");
    assert.ok(String(o.evidence).length > 5, "override entry needs documented evidence");
    assert.match(o.reviewedOn, /^\d{4}-\d{2}-\d{2}$/, "reviewedOn must be YYYY-MM-DD");
  }

  for (const v of data.onnxruntimeWeb.allowedVersions) {
    assert.match(v.version, /^[0-9]+\.[0-9]+\.[0-9]+$/, `version ${v.version} must be semver`);
    assert.ok(String(v.reason).length > 10, `entry ${v.version} needs a documented reason`);
    assert.ok(String(v.evidence).length > 5, `entry ${v.version} needs documented evidence`);
    assert.match(v.reviewedOn, /^\d{4}-\d{2}-\d{2}$/, "reviewedOn must be YYYY-MM-DD");
  }
});

test("all onnxruntime-web versions in the repository are in the allowlist", () => {
  const allowlist = JSON.parse(readFileSync(ALLOWLIST_PATH, "utf8"));
  const allowed = new Set(allowlist.onnxruntimeWeb.allowedVersions.map((x) => x.version));

  const raw = execSync(
    `grep -rhoE 'onnxruntime-web@[0-9]+\\.[0-9]+\\.[0-9]+' ${PIN_SCAN_TARGETS} 2>/dev/null || true`,
    { cwd: ROOT, encoding: "utf8" },
  );

  const found = new Set();
  for (const line of raw.split("\n")) {
    const v = line.split("@").pop()?.trim();
    if (v) found.add(v);
  }

  assert.ok(found.size > 0, "expected to find onnxruntime-web references in repo");
  for (const v of found) {
    assert.ok(allowed.has(v), `unauthorized onnxruntime-web version "${v}" found in repository`);
  }
});

test("all @huggingface/transformers versions in the repository are in the allowlist", () => {
  const allowlist = JSON.parse(readFileSync(ALLOWLIST_PATH, "utf8"));
  const allowed = new Set([
    allowlist.transformers.shared,
    ...allowlist.transformers.allowedLocalOverrides.map((x) => x.version),
  ]);

  const raw = execSync(
    `grep -rhoE '@huggingface/transformers@[0-9]+\\.[0-9]+\\.[0-9]+' ${PIN_SCAN_TARGETS} 2>/dev/null || true`,
    { cwd: ROOT, encoding: "utf8" },
  );

  const found = new Set();
  for (const line of raw.split("\n")) {
    const v = line.split("@").pop()?.trim();
    if (v) found.add(v);
  }

  assert.ok(found.size > 0, "expected to find @huggingface/transformers references in repo");
  for (const v of found) {
    assert.ok(
      allowed.has(v),
      `unauthorized @huggingface/transformers version "${v}" found in repository`,
    );
  }
});

test("scripts/audit-model-currency.mjs --check succeeds on the clean tree", () => {
  const out = execSync("node scripts/audit-model-currency.mjs --check", {
    cwd: ROOT,
    encoding: "utf8",
  });
  assert.match(
    out,
    /PASS — \d+ built routes covered; evidence matches catalogue; runtime pins authorized\./,
  );
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized pin in scripts/", () => {
  const p = join(ROOT, "scripts/__mutant_test_pin.mjs");
  writeFileSync(p, "// mutant pin: onnxruntime-web@1.99.0\n", "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when unauthorized pin exists in scripts/",
    );
  } finally {
    if (existsSync(p)) rmSync(p);
  }
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized pin in search/", () => {
  const p = join(ROOT, "search/__mutant_test_pin.js");
  writeFileSync(p, "// mutant pin: @huggingface/transformers@9.9.9\n", "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when unauthorized pin exists in search/",
    );
  } finally {
    if (existsSync(p)) rmSync(p);
  }
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized route using allowed override", () => {
  const p = join(ROOT, "models/depth-anything/worker.js");
  const orig = readFileSync(p, "utf8");
  writeFileSync(p, orig + "\n// mutant: @huggingface/transformers@4.2.0\n", "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when unlisted route uses 4.2.0",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches allowlist entry missing reason", () => {
  const p = join(ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  delete data.transformers.allowedLocalOverrides[0].reason;
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when allowlist entry is missing reason",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});
