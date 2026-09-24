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

  for (const v of data.onnxruntimeWeb.allowedVersions) {
    assert.match(v.version, /^[0-9]+\.[0-9]+\.[0-9]+$/, `version ${v.version} must be semver`);
    assert.ok(String(v.reason).length > 10, `entry ${v.version} needs a documented reason`);
    assert.ok(String(v.evidence).length > 5, `entry ${v.version} needs documented evidence`);
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
