// Adversarial tests for the runtime-pin currency gate.
//
// These tests MUTATE repository files (the allowlist, model workers, probe files) and run the gate, so
// they must never touch the live checkout: `node --test` runs test FILES in parallel, and a test that
// rewrote scripts/runtime-pin-allowlist.json / models/* in place raced with the other pin tests and, on
// a SIGKILL or timeout, left those tracked files corrupted forever (the finally never ran). Every
// mutation below happens inside an ISOLATED COPY materialised from `git archive HEAD`, exactly like
// test/runtime-integrity-exemption.test.mjs (the helper is shared via test/lib/isolated-repo.mjs so the
// two tests can never diverge again).
//
// The audit script itself is the file under test, so the working-tree copy is overlaid onto the
// isolated copy by the shared helper test/lib/isolated-repo.mjs before it is imported/run. In CI,
// where the test and the script land in the same commit, that overlay is a no-op; during local
// development it makes the mutations below be judged by the CURRENT source rather than the last
// committed snapshot.

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { materializeIsolatedRepo } from "./lib/isolated-repo.mjs";

// The isolated copy the mutations are applied to. Materialised once in `before`, removed in `after`.
let COPY_ROOT = null;
// The copy's audit module, dynamically imported so checkRuntimePins()/findPinsInBinaryFiles() resolve
// their own ROOT against the isolated copy (never the live checkout).
let audit = null;

// Measured on the clean tree (fleet/mtu): checkRuntimePins() scans 407 @huggingface/transformers pins
// and 152 onnxruntime-web pins (559 total) across models/, lib/, public/, scripts/, search/,
// models.json, sw.js and runtime-integrity.json. Each floor is the non-vacuity guard for ITS OWN
// pattern: a pattern that matches nothing would return 0 and pass vacuously, so the test asserts each
// scan saw a healthy positive number of pins. Per-pattern floors are required because a single total
// floor of 300 sat BELOW the transformers-only count of 407 — a fully-broken onnxruntime-web scan (0)
// plus a healthy transformers scan (407) still cleared 300, so the break was invisible. The chosen
// floors sit well above 0 ("matches nothing") while leaving headroom for normal route growth/shrinkage.
const TJS_SCAN_FLOOR = 350; // measured 407 transformers.js pins
const ORT_SCAN_FLOOR = 120; // measured 152 onnxruntime-web pins

before(async () => {
  COPY_ROOT = materializeIsolatedRepo();
  // The working-tree audit script (the file under test) is overlaid onto the committed snapshot by the
  // shared helper test/lib/isolated-repo.mjs, so this import resolves to the CURRENT source.
  audit = await import(pathToFileURL(join(COPY_ROOT, "scripts/audit-model-currency.mjs")).href);
});

after(() => {
  if (COPY_ROOT) rmSync(COPY_ROOT, { recursive: true, force: true });
});

/** Runs the currency gate inside the isolated copy. Returns its exit status plus stdout/stderr. */
function runGate() {
  try {
    const stdout = execFileSync(
      "node",
      [join(COPY_ROOT, "scripts/audit-model-currency.mjs"), "--check"],
      { cwd: COPY_ROOT, stdio: "pipe", encoding: "utf8" },
    );
    return { status: 0, stdout, stderr: "" };
  } catch (error) {
    return {
      status: typeof error.status === "number" ? error.status : 1,
      stdout: String(error.stdout ?? ""),
      stderr: String(error.stderr ?? ""),
    };
  }
}

/** Mutate `relativePath` inside the copy, run the gate, assert it failed naming `needle`, restore. */
function expectVersionFailure(relativePath, mutate, needle, message) {
  const path = join(COPY_ROOT, relativePath);
  const original = readFileSync(path, "utf8");
  try {
    writeFileSync(path, mutate(original));
    const result = runGate();
    assert.equal(result.status, 1, message);
    assert.ok(result.stderr.includes(needle), `gate error must name ${needle}: ${result.stderr}`);
  } finally {
    writeFileSync(path, original, "utf8");
  }
  assert.equal(readFileSync(path, "utf8"), original, `${relativePath} must be restored byte-exactly`);
}

// --- binary-classified files (bead web-ai-showcase-5s4) ---------------------------------
// `grep -I` skips a file the moment it contains a NUL byte, so a pin written into an ordinary
// .js file with one stray control byte would never reach the version/route scans.
function writeBinaryProbe(relPath, text) {
  writeFileSync(
    join(COPY_ROOT, relPath),
    Buffer.concat([Buffer.from(text, "utf8"), Buffer.from([0x00]), Buffer.from("\n")]),
  );
  return join(COPY_ROOT, relPath);
}

test("runtime-pin-allowlist.json exists, parses, and has required structure", () => {
  assert.ok(existsSync(audit.ALLOWLIST_PATH), "missing scripts/runtime-pin-allowlist.json");
  const data = JSON.parse(readFileSync(audit.ALLOWLIST_PATH, "utf8"));
  assert.ok(data.transformers?.shared, "transformers.shared must be defined");
  assert.match(
    data.transformers.shared,
    audit.SEMVER_VERSION_RE,
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
      audit.SEMVER_VERSION_RE,
      `override version ${o.version} must be semver`,
    );
    assert.ok(
      Array.isArray(o.slugs) && o.slugs.length > 0,
      "override entry needs non-empty slugs array",
    );
    for (const slug of o.slugs) {
      assert.ok(
        existsSync(join(COPY_ROOT, "models", slug)),
        `override slug "${slug}" must name an existing models/ directory`,
      );
    }
    assert.ok(
      String(o.reason).trim().length > audit.MIN_REASON_LENGTH,
      `override entry needs a documented reason (> ${audit.MIN_REASON_LENGTH} chars)`,
    );
    assert.ok(
      String(o.evidence).trim().length > audit.MIN_EVIDENCE_LENGTH,
      `override entry needs documented evidence (> ${audit.MIN_EVIDENCE_LENGTH} chars)`,
    );
    assert.match(o.reviewedOn, audit.REVIEWED_ON_DATE_RE, "reviewedOn must be YYYY-MM-DD");
  }

  for (const v of data.onnxruntimeWeb.allowedVersions) {
    assert.match(v.version, audit.SEMVER_VERSION_RE, `version ${v.version} must be semver`);
    assert.ok(
      String(v.reason).trim().length > audit.MIN_REASON_LENGTH,
      `entry ${v.version} needs a documented reason (> ${audit.MIN_REASON_LENGTH} chars)`,
    );
    assert.ok(
      String(v.evidence).trim().length > audit.MIN_EVIDENCE_LENGTH,
      `entry ${v.version} needs documented evidence (> ${audit.MIN_EVIDENCE_LENGTH} chars)`,
    );
    assert.match(v.reviewedOn, audit.REVIEWED_ON_DATE_RE, "reviewedOn must be YYYY-MM-DD");
  }
});

test("every runtime pin in the repository is authorized (single-sourced via checkRuntimePins)", () => {
  // These used to be two independent grep scans here that duplicated the gate's own patterns and
  // (unlike the gate) had no derived-inventory exemption, so the generated integrity inventory's
  // onnxruntime-web 1.22.0 / dev-suffixed and transformers 3.1.2 literals read as unallowlisted route
  // pins. The scan now lives in exactly one place — checkRuntimePins() — which honours the exemption
  // and returns [] only when every pin (route pins AND the derived inventory) is authorised. An
  // unauthorised version introduced anywhere, including a model route, still makes this assertion fail.
  const result = audit.checkRuntimePins();
  assert.equal(
    result.length,
    0,
    `checkRuntimePins must report no unauthorized runtime pins: ${JSON.stringify([...result])}`,
  );
  // NON-VACUITY (P1): a pattern that matched nothing would return [] and pass the line above silently.
  // scannedCounts carries the per-pattern breakdown; assert each pattern saw a healthy count so a
  // broken scan for ONE runtime cannot hide behind the other runtime's healthy total (a single total
  // floor of 300 was below the transformers-only count of 407, so a broken onnxruntime-web scan passed).
  assert.ok(
    Number.isInteger(result.scannedCounts?.transformers) &&
      result.scannedCounts.transformers >= TJS_SCAN_FLOOR,
    `checkRuntimePins must scan a healthy number of @huggingface/transformers pins (got ${result.scannedCounts?.transformers}, need >= ${TJS_SCAN_FLOOR}); zero matches would mean the transformers scan is broken, not clean`,
  );
  assert.ok(
    Number.isInteger(result.scannedCounts?.onnxruntimeWeb) &&
      result.scannedCounts.onnxruntimeWeb >= ORT_SCAN_FLOOR,
    `checkRuntimePins must scan a healthy number of onnxruntime-web pins (got ${result.scannedCounts?.onnxruntimeWeb}, need >= ${ORT_SCAN_FLOOR}); zero matches would mean the onnxruntime-web scan is broken, not clean`,
  );
});

test("unit: findPinsInBinaryFiles finds a pin grep -I would skip (NUL byte)", () => {
  const p = writeBinaryProbe("scripts/__binary_pin_probe.mjs", "// probe: onnxruntime-web@1.99.0");
  try {
    const hit = audit.findPinsInBinaryFiles().find((h) => h.file === "scripts/__binary_pin_probe.mjs");
    assert.ok(hit, "NUL-byte probe must be reported by the binary pass");
    assert.deepEqual(
      hit.hits,
      [{ label: "onnxruntime-web", versions: ["1.99.0"] }],
      "the error must name the pattern and the version found",
    );
  } finally {
    rmSync(p, { force: true });
  }
});

test("unit: findPinsInBinaryFiles is empty on the clean tree", () => {
  assert.deepEqual(audit.findPinsInBinaryFiles(), []);
});

test("unit: findPinsInBinaryFiles does not report text files (no false positives)", () => {
  const p = join(COPY_ROOT, "scripts/__text_pin_probe.mjs");
  writeFileSync(p, "// probe: onnxruntime-web@1.99.0\n", "utf8");
  try {
    assert.equal(
      audit.findPinsInBinaryFiles().some((h) => h.file === "scripts/__text_pin_probe.mjs"),
      false,
      "text-file pins belong to the version/route scans, not the binary pass",
    );
  } finally {
    rmSync(p, { force: true });
  }
});

test("scripts/audit-model-currency.mjs --check succeeds on the clean tree", () => {
  const result = runGate();
  assert.equal(result.status, 0, result.stderr);
  assert.match(
    result.stdout,
    /PASS — \d+ built routes covered; evidence matches catalogue; runtime pins authorized\./,
  );
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized pin in scripts/", () => {
  const p = join(COPY_ROOT, "scripts/__mutant_test_pin.mjs");
  writeFileSync(p, "// mutant pin: onnxruntime-web@1.99.0\n", "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when unauthorized pin exists in scripts/");
    assert.ok(result.stderr.includes("1.99.0"), result.stderr);
  } finally {
    if (existsSync(p)) rmSync(p);
  }
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized pin in search/", () => {
  const p = join(COPY_ROOT, "search/__mutant_test_pin.js");
  writeFileSync(p, "// mutant pin: @huggingface/transformers@9.9.9\n", "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when unauthorized pin exists in search/");
    assert.ok(result.stderr.includes("9.9.9"), result.stderr);
  } finally {
    if (existsSync(p)) rmSync(p);
  }
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized pin in a binary-classified file under an authorized slug", () => {
  // Exact repro from bead web-ai-showcase-5s4: a NUL-byte .js file under an AUTHORIZED slug
  // (gemma-3-270m, allowlisted for transformers 4.2.0) carrying an unauthorized ort version.
  // Measured pre-fix: the version/route scans skipped it and --check exited 0.
  const p = writeBinaryProbe(
    "models/gemma-3-270m/__review_binary_probe.js",
    "// mutant pin: onnxruntime-web@1.99.0",
  );
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when a pin hides in a binary-classified file");
    assert.match(result.stderr, /binary-classified file/);
    assert.match(result.stderr, /1\.99\.0/);
  } finally {
    rmSync(p, { force: true });
  }
});

test("MUTANT PROOF: an allowlisted version inside a binary-classified file still fails (never a skip)", () => {
  // 4.3.0 is allowlisted for all-distilroberta-v1 and 3.7.5 is shared, but a version string
  // inside opaque bytes carries no reviewable context, so the fail-closed pass refuses it.
  const p = writeBinaryProbe(
    "scripts/__binary_pin_probe.mjs",
    "// mutant: @huggingface/transformers@4.3.0",
  );
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail on a pin in a binary-classified file");
    assert.match(result.stderr, /binary-classified file/);
    assert.match(result.stderr, /4\.3\.0/);
  } finally {
    rmSync(p, { force: true });
  }
});

test("MUTANT PROOF: checkRuntimePins catches unauthorized route using allowed override", () => {
  const p = join(COPY_ROOT, "models/depth-anything/worker.js");
  const orig = readFileSync(p, "utf8");
  writeFileSync(p, orig + "\n// mutant: @huggingface/transformers@4.2.0\n", "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when unlisted route uses 4.2.0");
    assert.ok(result.stderr.includes("4.2.0"), result.stderr);
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches allowlist entry missing reason", () => {
  const p = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  delete data.transformers.allowedLocalOverrides[0].reason;
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when allowlist entry is missing reason");
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches stub reason (<= MIN_REASON_LENGTH chars)", () => {
  const p = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].reason = "too short";
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when reason is <= MIN_REASON_LENGTH chars");
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches invalid reviewedOn date", () => {
  const p = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].reviewedOn = "soon";
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when reviewedOn is not YYYY-MM-DD");
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches duplicate version in allowedLocalOverrides", () => {
  const p = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  const dup = JSON.parse(JSON.stringify(data.transformers.allowedLocalOverrides[0]));
  data.transformers.allowedLocalOverrides.push(dup);
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when duplicate version exists in allowedLocalOverrides");
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

test("MUTANT PROOF: checkRuntimePins catches nonexistent model directory in override slugs", () => {
  const p = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const orig = readFileSync(p, "utf8");
  const data = JSON.parse(orig);
  data.transformers.allowedLocalOverrides[0].slugs.push("nonexistent-model-slug-xyz");
  writeFileSync(p, JSON.stringify(data, null, 2), "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 1, "expected --check to fail when nonexistent model slug is listed");
  } finally {
    writeFileSync(p, orig, "utf8");
  }
});

// --- P0 falsification: version-token boundary (web-ai-showcase-mtu) ---------------------
// The old VERSION_TOKEN had an optional -suffix group only, so grep -oE truncated each of these to the
// allowed base and the gate passed. The full-token capture must judge the WHOLE string and fail.
const TRANSFORMERS_SUFFIXES = ["+build1", ".evil", "_evil", "evil", "-"];
const ORT_SUFFIXES = ["+malicious", ".evil", "_evil", "evil", "-"];

for (const suffix of TRANSFORMERS_SUFFIXES) {
  test(`MUTANT PROOF: transformers pin "@huggingface/transformers@3.7.5${suffix}" in a model route fails`, () => {
    const full = `3.7.5${suffix}`;
    expectVersionFailure(
      "models/animegan-cartoonization/worker.js",
      (source) => `${source}\n// mutant: @huggingface/transformers@${full}\n`,
      full,
      `expected --check to fail and name the full offending transformers string for suffix "${suffix}"`,
    );
  });
}

for (const suffix of ORT_SUFFIXES) {
  test(`MUTANT PROOF: ort pin "onnxruntime-web@1.21.0${suffix}" in a model route fails`, () => {
    const full = `1.21.0${suffix}`;
    expectVersionFailure(
      "models/animegan-cartoonization/worker.js",
      (source) => `${source}\n// mutant: onnxruntime-web@${full}\n`,
      full,
      `expected --check to fail and name the full offending onnxruntime-web string for suffix "${suffix}"`,
    );
  });
}

// --- P2: trailing full stop is prose, not a version suffix (web-ai-showcase-mtu) -----------------
// A legitimate pin followed by a sentence-ending period must NOT be read as an unauthorised suffix.
// `3.7.5.` / `3.7.5..` / `3.7.5...` all normalise to `3.7.5` (ALL trailing dots are stripped); a
// trailing '-', '+', '_' or a real suffix (`.evil`, `+build1`, `evil`) is NOT stripped and still
// fails (covered by the suffix loops above). The same normalisation is applied to the ORT scan.
const TRAILING_DOT_PINS = [
  ["@huggingface/transformers@3.7.5.", "transformers, one trailing full stop"],
  ["@huggingface/transformers@3.7.5..", "transformers, two trailing full stops"],
  ["@huggingface/transformers@3.7.5...", "transformers, three trailing full stops"],
  ["onnxruntime-web@1.21.0.", "onnxruntime-web, one trailing full stop"],
  ["onnxruntime-web@1.21.0..", "onnxruntime-web, two trailing full stops"],
  ["onnxruntime-web@1.21.0...", "onnxruntime-web, three trailing full stops"],
];

for (const [pin, label] of TRAILING_DOT_PINS) {
  test(`false-red regression: "${pin}" (${label}) must not be rejected`, () => {
    const p = join(COPY_ROOT, "models/animegan-cartoonization/worker.js");
    const orig = readFileSync(p, "utf8");
    writeFileSync(p, `${orig}\n// prose: ${pin}\n`, "utf8");
    try {
      const result = runGate();
      assert.equal(
        result.status,
        0,
        `expected --check to pass for the prose pin "${pin}": ${result.stderr}`,
      );
    } finally {
      writeFileSync(p, orig, "utf8");
    }
  });
}

// --- P0 regression (web-ai-showcase-mtu): trailing dot before a URL path must stay RAW ------------
// stripTrailingDots is prose normalisation: `3.7.5.` in a comment is punctuation and must normalise to
// `3.7.5`. But in a REAL pinned URL the trailing dot is part of the version string — `1.21.0.` before
// `/dist/` is a different (invalid) version than `1.21.0` — so normalising it would turn a
// previously-failing pin into an allowed one (a FALSE GREEN, worse than the false red the normalisation
// fixed). The gate discriminates by the character immediately after the match: `/` means a URL path, so
// the token is validated RAW. These tests inject a URL-shaped pin (trailing dot followed by `/dist/...`)
// into a model route and assert the gate FAILS naming the dotted token. They go RED if anyone re-widens
// the normalisation to strip trailing dots from URL-path tokens.
test("MUTANT PROOF: onnxruntime-web URL pin with a trailing dot before /dist/ still fails", () => {
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => `${source}\nconst BAD = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0./dist/ort.wasm.min.mjs";\n`,
    "1.21.0.",
    "a URL-pathed onnxruntime-web pin with a trailing dot must not be normalised to the allowed base",
  );
});

test("MUTANT PROOF: transformers URL pin with a trailing dot before /dist/ still fails", () => {
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => `${source}\nconst BAD = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5./dist/transformers.min.js";\n`,
    "3.7.5.",
    "a URL-pathed transformers pin with a trailing dot must not be normalised to the allowed shared base",
  );
});

test("unit: normaliseCapturedVersion keeps a trailing dot RAW when followed by a URL path", () => {
  // `/` immediately after the match means the token is part of a URL path, so no dot is stripped.
  assert.equal(audit.normaliseCapturedVersion("1.21.0.", "/"), "1.21.0.");
  assert.equal(audit.normaliseCapturedVersion("1.21.0..", "/"), "1.21.0..");
  assert.equal(audit.normaliseCapturedVersion("3.7.5.", "/"), "3.7.5.");
  // Any other follower is prose punctuation: strip trailing dots exactly like the plain helper.
  assert.equal(audit.normaliseCapturedVersion("3.7.5.", " "), "3.7.5");
  assert.equal(audit.normaliseCapturedVersion("3.7.5.", '"'), "3.7.5");
  assert.equal(audit.normaliseCapturedVersion("3.7.5.", ","), "3.7.5");
  assert.equal(audit.normaliseCapturedVersion("1.21.0.", ""), "1.21.0"); // end of line
  assert.equal(audit.normaliseCapturedVersion("3.7.5.", undefined), "3.7.5");
});

test("MUTANT PROOF: a suffixed pin written into the exempt generator still fails", () => {
  // scripts/runtime-integrity.mjs is exempt only for measuredVersions; a +build suffix is not measured.
  expectVersionFailure(
    "scripts/runtime-integrity.mjs",
    (source) => source.replace(
      "const ORT_ASSETS = [",
      'const ORT_ASSETS = [\n  "onnxruntime-web@1.21.0+malicious/dist/ort.wasm.min.mjs",',
    ),
    "1.21.0+malicious",
    "the generator must not hide a suffixed (non-measured) pin",
  );
});

test("MUTANT PROOF: a suffixed pin written into runtime-integrity.json still fails", () => {
  expectVersionFailure(
    "runtime-integrity.json",
    (source) => {
      const data = JSON.parse(source);
      data.urls["https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0+malicious/dist/ort.wasm.min.mjs"] = {
        sha256: "0".repeat(64),
        bytes: 1,
        policy: "verify-then-cache",
      };
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    "1.21.0+malicious",
    "the generated data file must not authorise a suffixed (non-measured) pin",
  );
});

test("operator manuals (AGENTS.md, CLAUDE.md, SKILL.md) reference transformers-version-policy and avoid absolute freeze phrasing", () => {
  const files = [
    join(COPY_ROOT, "AGENTS.md"),
    join(COPY_ROOT, "CLAUDE.md"),
    join(COPY_ROOT, ".agents/skills/web-ai-showcase/SKILL.md"),
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
