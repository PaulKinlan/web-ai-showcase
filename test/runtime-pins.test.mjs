import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { execSync } from "node:child_process";
import {
  ALLOWLIST_PATH,
  checkRuntimePins,
  MIN_EVIDENCE_LENGTH,
  MIN_REASON_LENGTH,
  PIN_SCAN_TARGETS,
  REVIEWED_ON_DATE_RE,
  SEMVER_VERSION_RE,
} from "../scripts/audit-model-currency.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

test("runtime-pin-allowlist.json exists, parses, and has required structure", () => {
  assert.ok(existsSync(ALLOWLIST_PATH), "missing scripts/runtime-pin-allowlist.json");
  const data = JSON.parse(readFileSync(ALLOWLIST_PATH, "utf8"));
  assert.ok(data.transformers?.shared, "transformers.shared must be defined");
  assert.match(
    data.transformers.shared,
    SEMVER_VERSION_RE,
    "transformers.shared must be semver",
  );
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
      SEMVER_VERSION_RE,
      `override version ${o.version} must be semver`,
    );
    assert.ok(
      Array.isArray(o.slugs) && o.slugs.length > 0,
      "override entry needs non-empty slugs array",
    );
    for (const slug of o.slugs) {
      assert.ok(
        existsSync(join(ROOT, "models", slug)),
        `override slug "${slug}" must name an existing models/ directory`,
      );
    }
    assert.ok(
      String(o.reason).trim().length > MIN_REASON_LENGTH,
      `override entry needs a documented reason (> ${MIN_REASON_LENGTH} chars)`,
    );
    assert.ok(
      String(o.evidence).trim().length > MIN_EVIDENCE_LENGTH,
      `override entry needs documented evidence (> ${MIN_EVIDENCE_LENGTH} chars)`,
    );
    assert.match(o.reviewedOn, REVIEWED_ON_DATE_RE, "reviewedOn must be YYYY-MM-DD");
  }

  for (const v of data.onnxruntimeWeb.allowedVersions) {
    assert.match(v.version, SEMVER_VERSION_RE, `version ${v.version} must be semver`);
    assert.ok(
      String(v.reason).trim().length > MIN_REASON_LENGTH,
      `entry ${v.version} needs a documented reason (> ${MIN_REASON_LENGTH} chars)`,
    );
    assert.ok(
      String(v.evidence).trim().length > MIN_EVIDENCE_LENGTH,
      `entry ${v.version} needs documented evidence (> ${MIN_EVIDENCE_LENGTH} chars)`,
    );
    assert.match(v.reviewedOn, REVIEWED_ON_DATE_RE, "reviewedOn must be YYYY-MM-DD");
  }

  assert.deepEqual(checkRuntimePins(), []);
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

test("MUTANT PROOF: checkRuntimePins catches stub reason (<= MIN_REASON_LENGTH chars)", () => {
  const p = join(ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].reason = "too short";
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when reason is <= MIN_REASON_LENGTH chars",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches invalid reviewedOn date", () => {
  const p = join(ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].reviewedOn = "soon";
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when reviewedOn is not YYYY-MM-DD",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches duplicate version in allowedLocalOverrides", () => {
  const p = join(ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  const dup = JSON.parse(JSON.stringify(data.transformers.allowedLocalOverrides[0]));
  data.transformers.allowedLocalOverrides.push(dup);
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when duplicate version exists in allowedLocalOverrides",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches nonexistent model directory in override slugs", () => {
  const p = join(ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].slugs.push("nonexistent-model-slug-xyz");
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    assert.throws(
      () => execSync("node scripts/audit-model-currency.mjs --check", { cwd: ROOT, stdio: "pipe" }),
      /Command failed/,
      "expected --check to fail when nonexistent model slug is listed",
    );
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("operator manuals (AGENTS.md, CLAUDE.md, SKILL.md) reference transformers-version-policy and avoid absolute freeze phrasing", () => {
  const files = [
    join(ROOT, "AGENTS.md"),
    join(ROOT, "CLAUDE.md"),
    join(ROOT, ".agents/skills/web-ai-showcase/SKILL.md"),
  ];
  for (const f of files) {
    assert.ok(existsSync(f), `missing documentation file: ${f}`);
    const text = readFileSync(f, "utf8");
    assert.ok(
      text.includes("reports/transformers-version-policy.md"),
      `${f} must reference reports/transformers-version-policy.md`,
    );
    assert.match(
      text,
      /(without\s+full\s+staging|staged\s+rollout)/i,
      `${f} must contain qualifier phrase ("without full staging" or "staged rollout")`,
    );
    assert.doesNotMatch(
      text,
      /never bump shared (`lib\/webai\.js`|lib\/webai\.js)(?!\s+without\s+full\s+staging)/,
      `${f} must not contain absolute freeze phrasing without "without full staging"`,
    );
  }
});


