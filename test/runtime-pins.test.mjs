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

/** Append an executable URL string literal to a model worker, run the gate, assert the verdict. */
function expectExecutableUrlVerdict(url, { status, needle }, message) {
  const p = join(COPY_ROOT, "models/animegan-cartoonization/worker.js");
  const orig = readFileSync(p, "utf8");
  writeFileSync(p, `${orig}\nconst __probe_url = ${JSON.stringify(url)};\n`, "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, status, message);
    if (needle) {
      assert.ok(result.stderr.includes(needle), `gate error must name "${needle}": ${result.stderr}`);
    }
  } finally {
    writeFileSync(p, orig, "utf8");
  }
  assert.equal(readFileSync(p, "utf8"), orig, "models/animegan-cartoonization/worker.js must be restored byte-exactly");
}

/** Run the same grep the scanner runs (single-sourced pattern + targets) over the isolated copy.
 *  execFileSync argv (no shell) — mirrors the gate's grepScan; exit 1 = no matches. */
function grepScan(pattern) {
  try {
    return execFileSync(
      "grep",
      ["-I", "-rnE", pattern, ...audit.PIN_SCAN_TARGETS.trim().split(/\s+/)],
      { cwd: COPY_ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
    );
  } catch (e) {
    if (e.status === 1 && e.signal == null) return "";
    throw e;
  }
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

test("POSITIVE DISCOVERY: the scan actually finds the known committed fixture URLs", () => {
  // Counts prove SOMETHING matched; this proves the RIGHT fixtures matched. If the pattern were
  // rewritten to match nothing, the count-floor assertion above fails; if it were rewritten to match a
  // different (wrong) set of lines, this assertion fails. Neither is a vacuous green.
  const tjs = grepScan(audit.TJS_PIN_PATTERN);
  assert.match(
    tjs,
    /lib\/webai\.js:\d+:.*@huggingface\/transformers@3\.7\.5/,
    "the shared transformers pin in lib/webai.js must be found by the scan",
  );
  const ort = grepScan(audit.ORT_PIN_PATTERN);
  assert.match(
    ort,
    /models\/nafnet-image-deblurring\/worker\.js:\d+:.*onnxruntime-web@1\.21\.0\/dist\/ort\.wasm\.min\.mjs/,
    "the onnxruntime-web 1.21.0 URL in the nafnet worker must be found by the scan",
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

test("unit: findPinsInBinaryFiles shares the whole-candidate discovery (query suffix not truncated)", () => {
  // The binary pass must use the SAME package-marker discovery as the text scans, so a query-suffixed
  // specifier inside a binary-classified file is captured in FULL (never truncated to its allowed base).
  const p = writeBinaryProbe("scripts/__binary_query_pin_probe.mjs", "// probe: onnxruntime-web@1.99.0?x=1");
  try {
    const hit = audit.findPinsInBinaryFiles().find((h) => h.file === "scripts/__binary_query_pin_probe.mjs");
    assert.ok(hit, "query-suffixed NUL-byte probe must be reported by the binary pass");
    assert.deepEqual(
      hit.hits,
      [{ label: "onnxruntime-web", versions: ["1.99.0?x=1"] }],
      "the binary pass must name the whole raw candidate, not a truncated prefix",
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

// --- STRICT RAW specifier matrix (web-ai-showcase-mtu) -------------------------------------------
// The gate now compares the WHOLE RAW candidate after the package marker against the allowlist — never
// stripped, never normalised, never decoded, never context-heuristicked. Each row below is a PERMANENT
// regression: a real executable URL string literal (both runtimes, independently) whose verdict is
// pinned. GREEN means the exact authorised version followed by a REAL `/` path is accepted; RED means
// any dot/suffix/query/hash/percent-encoding that changes the candidate is rejected and NAMED IN FULL.
const STRICT_RUNTIMES = [
  { label: "@huggingface/transformers", base: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5", version: "3.7.5" },
  { label: "onnxruntime-web", base: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0", version: "1.21.0" },
];
// suffix = appended to the authorised base URL; captured() = the whole raw candidate the gate must name
// on RED (a REAL `/` terminates the candidate; the dot / suffix / query / hash / percent-encoding is
// part of it and therefore fails).
const STRICT_RAW_SUFFIX_CASES = [
  { suffix: "/dist/x.js", verdict: "green", name: "authorised version + real path" },
  { suffix: "./dist/x.js", verdict: "red", captured: (v) => `${v}.`, name: "dot before path is part of the URL" },
  { suffix: ".", verdict: "red", captured: (v) => `${v}.`, name: "trailing dot at end of URL" },
  { suffix: ".?x=1", verdict: "red", captured: (v) => `${v}.?x=1`, name: "trailing dot then query" },
  { suffix: ".#x", verdict: "red", captured: (v) => `${v}.#x`, name: "trailing dot then hash" },
  { suffix: "?x=1", verdict: "red", captured: (v) => `${v}?x=1`, name: "query-suffixed (unreviewed)" },
  { suffix: "#x", verdict: "red", captured: (v) => `${v}#x`, name: "hash-suffixed (unreviewed)" },
  { suffix: "%2Fdist/x.js", verdict: "red", captured: (v) => `${v}%2Fdist`, name: "percent-encoded separator" },
  { suffix: ".evil/dist/x.js", verdict: "red", captured: (v) => `${v}.evil`, name: ".evil suffix" },
  { suffix: "+build1/dist/x.js", verdict: "red", captured: (v) => `${v}+build1`, name: "+build1 suffix" },
  { suffix: "-/dist/x.js", verdict: "red", captured: (v) => `${v}-`, name: "bare - suffix" },
  { suffix: "", verdict: "green", name: "exact authorised version, no dot" },
];

for (const rt of STRICT_RUNTIMES) {
  for (const c of STRICT_RAW_SUFFIX_CASES) {
    test(`STRICT RAW: ${rt.label} — ${c.name} → ${c.verdict.toUpperCase()}`, () => {
      const url = `${rt.base}${c.suffix}`;
      const needle = c.verdict === "red" ? c.captured(rt.version) : null;
      expectExecutableUrlVerdict(
        url,
        { status: c.verdict === "red" ? 1 : 0, needle },
        `${rt.label} URL "${url}" must be ${c.verdict}${needle ? ` (naming "${needle}")` : ""}`,
      );
    });
  }
}

// --- Floating specifiers must never bypass the exact-version allowlist ---------------------------
// Each probe is a CDN package URL in an isolated model worker. A scan requiring X.Y.Z misses these.
const FLOATING_RUNTIME_CASES = [
  { pkg: "@huggingface/transformers", specifiers: ["latest", "next", "3", "3.7", "^3.7.5", "v3.7.5", "3.x"] },
  { pkg: "onnxruntime-web", specifiers: ["latest", "1"] },
];
for (const { pkg, specifiers } of FLOATING_RUNTIME_CASES) {
  for (const specifier of specifiers) {
    test(`FLOATING: ${pkg}@${specifier} is rejected and named`, () => {
      expectExecutableUrlVerdict(
        `https://cdn.jsdelivr.net/npm/${pkg}@${specifier}/dist/x.js`,
        { status: 1, needle: specifier },
        `${pkg}@${specifier} must not bypass the runtime-pin allowlist`,
      );
    });
  }
}
for (const { pkg, version } of [
  { pkg: "@huggingface/transformers", version: "3.7.5" },
  { pkg: "onnxruntime-web", version: "1.21.0" },
]) {
  test(`FLOATING control: ${pkg}@${version} remains authorized`, () => {
    expectExecutableUrlVerdict(
      `https://cdn.jsdelivr.net/npm/${pkg}@${version}/dist/x.js`,
      { status: 0 },
      `${pkg}@${version} exact authorized pin must remain green`,
    );
  });
}

// --- Reviewed literal URL and marker-ledger boundary proofs ----------------------------------------
for (const suffix of ["^", "~", "|", "+", ".", "_", "%2Fdist", "?x=1", "#x", "!", "@evil", ";evil", ":evil", ",evil", "&evil"]) {
  test(`LITERAL BOUNDARY: raw Transformers suffix ${suffix} is never truncated`, () => {
    expectExecutableUrlVerdict(
      `https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5${suffix}/dist/x.js`,
      { status: 1, needle: `3.7.5${suffix}` },
      `the complete raw suffix ${suffix} must fail, not the authorized prefix pass`,
    );
  });
}

test("LITERAL BOUNDARY: two URLs on one line reject the later floating pin", () => {
  const p = join(COPY_ROOT, "models/animegan-cartoonization/worker.js");
  const original = readFileSync(p, "utf8");
  try {
    writeFileSync(p, `${original}\nconst __urls = ["https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/x.js", "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/x.js"];\n`);
    const result = runGate();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /onnxruntime-web version "latest"/);
    assert.match(result.stderr, /runtime marker golden ledger drift/);
  } finally { writeFileSync(p, original, "utf8"); }
});

test("LITERAL BOUNDARY: floating script src is rejected; exact src remains authorized", () => {
  expectVersionFailure(
    "models/animegan-cartoonization/index.html",
    (source) => `${source}\n<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/x.js"></script>\n`,
    'onnxruntime-web version "latest"',
    "HTML executable quoted src must fail on the whole floating specifier",
  );
  const p = join(COPY_ROOT, "models/animegan-cartoonization/index.html");
  const original = readFileSync(p, "utf8");
  try {
    writeFileSync(p, `${original}\n<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/x.js"></script>\n`);
    assert.equal(runGate().status, 0, "exact authorized HTML script src stays green");
  } finally { writeFileSync(p, original, "utf8"); }
});

test("LITERAL BOUNDARY: a literal URL nested inside interpolation still rejects the full suffix", () => {
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => source + '\nconst __nested = `x ${ { nested: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0~/dist/x.js` }.nested}`;\n',
    'onnxruntime-web version "1.21.0~"',
    "a nested but complete runtime URL literal must not hide behind template syntax",
  );
});

test("LITERAL LEDGER: floating marker added to JSON evidence fails census", () => {
  expectVersionFailure(
    "models/yolo11-detection/_questions.json",
    (source) => `${source}\n"onnxruntime-web@latest"\n`,
    "runtime marker golden ledger drift",
    "non-executable JSON is still census-controlled, not blanket exempt",
  );
});

test("LITERAL LEDGER: a changed reviewed dynamic expression invalidates its exact fingerprint", () => {
  expectVersionFailure(
    "models/silero-vad/worker.js",
    (source) => source.replace("onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs", "onnxruntime-web@${ORT_VERSION}/dist/ort.min.mjs"),
    "runtime marker golden ledger drift",
    "a changed vs2 exception requires independent ledger review",
  );
});

test("LITERAL LEDGER: floating pin in a NUL-bearing file fails closed", () => {
  const p = writeBinaryProbe("scripts/__floating_binary_probe.mjs", "// probe: onnxruntime-web@latest");
  try {
    const result = runGate();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /runtime pin in binary-classified file scripts\/__floating_binary_probe\.mjs/);
  } finally { rmSync(p, { force: true }); }
});

// --- DELIBERATE prose false reds ------------------------------------------------------------------
// Sentence punctuation is part of the candidate and FAILS. A prose comment ending a sentence right after
// a pinned version is an ACCEPTED false red — the prose author rewords. There is deliberately no
// punctuation exception and no comment detector (both would re-open the truncation hole).
const PROSE_FALSE_RED_CASES = [
  ["// prose: pinned to @huggingface/transformers@3.7.5.", "3.7.5."],
  ["// prose: pinned to onnxruntime-web@1.21.0.", "1.21.0."],
];

for (const [line, needle] of PROSE_FALSE_RED_CASES) {
  test(`STRICT RAW: comment "${line.slice(3).trim()}" is a deliberate false red`, () => {
    expectVersionFailure(
      "models/animegan-cartoonization/worker.js",
      (source) => `${source}\n${line}\n`,
      needle,
      "prose sentence punctuation must FAIL (accepted false red), never be silently stripped",
    );
  });
}

// --- derived sites: a dotted version is never covered by the measured-version exemption -----------
const DERIVED_DOTTED_CASES = [
  {
    name: "sw.js generated block (onnxruntime-web)",
    path: "sw.js",
    mutate: (s) => s.replace("onnxruntime-web@1.21.0/dist/ort.min.mjs", "onnxruntime-web@1.21.0./dist/ort.min.mjs"),
    needle: "1.21.0.",
  },
  {
    name: "sw.js generated block (transformers)",
    path: "sw.js",
    mutate: (s) => s.replace('@huggingface/transformers@3.7.5"', '@huggingface/transformers@3.7.5."'),
    needle: "3.7.5.",
  },
  {
    name: "runtime-integrity.json (onnxruntime-web)",
    path: "runtime-integrity.json",
    mutate: (s) => {
      const data = JSON.parse(s);
      data.urls["https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0./dist/ort.wasm.min.mjs"] = {
        sha256: "0".repeat(64),
        bytes: 1,
        policy: "verify-then-cache",
      };
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    needle: "1.21.0.",
  },
  {
    name: "runtime-integrity.json (transformers)",
    path: "runtime-integrity.json",
    mutate: (s) => {
      const data = JSON.parse(s);
      data.urls["https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5."] = {
        sha256: "0".repeat(64),
        bytes: 1,
        policy: "verify-then-cache",
      };
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    needle: "3.7.5.",
  },
  {
    name: "scripts/runtime-integrity.mjs (onnxruntime-web)",
    path: "scripts/runtime-integrity.mjs",
    mutate: (s) => s.replace("const ORT_ASSETS = [", 'const ORT_ASSETS = [\n  "onnxruntime-web@1.21.0./dist/ort.wasm.min.mjs",'),
    needle: "1.21.0.",
  },
  {
    name: "scripts/runtime-integrity.mjs (transformers)",
    path: "scripts/runtime-integrity.mjs",
    mutate: (s) => s.replace('"@huggingface/transformers@3.1.2/+esm"', '"@huggingface/transformers@3.1.2/+esm",\n  "@huggingface/transformers@3.7.5."'),
    needle: "3.7.5.",
  },
];

for (const c of DERIVED_DOTTED_CASES) {
  test(`STRICT RAW: dotted version in ${c.name} still fails`, () => {
    expectVersionFailure(
      c.path,
      c.mutate,
      c.needle,
      `a dotted version in ${c.name} must not be normalised into a measured/allowed base`,
    );
  });
}

// --- TWO references on ONE executable line, second unauthorised ---------------------------------
// matchAll must enumerate EVERY match; the gate must name the SECOND (unauthorised) pin, not stop at the
// first (authorised) one.
for (const rt of [
  { label: "@huggingface/transformers", good: "3.7.5", bad: "9.9.9" },
  { label: "onnxruntime-web", good: "1.21.0", bad: "9.9.9" },
]) {
  test(`STRICT RAW: two ${rt.label} pins on one executable line name the second`, () => {
    const url1 = `https://cdn.jsdelivr.net/npm/${rt.label}@${rt.good}`;
    const url2 = `https://cdn.jsdelivr.net/npm/${rt.label}@${rt.bad}`;
    expectVersionFailure(
      "models/animegan-cartoonization/worker.js",
      (source) => `${source}\nconst __probe = [${JSON.stringify(url1)}, ${JSON.stringify(url2)}];\n`,
      rt.bad,
      `the second (unauthorised) ${rt.label} pin on one executable line must be named`,
    );
  });
}

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

// --- numeric-suffix seam closure (bead web-ai-showcase-cdt) ------------------------------
// VERSION_TOKEN's break set is now ONLY `/`, a literal space/TAB byte, quotes, backtick, `<`, `>`.
// Every other byte after a numeric pin — ^ ~ | ! @ ; : , & \ $ ${ } — is part of the judged
// candidate, in EVERY context the legacy scans see (comments, JSON prose, regexes, bare specifiers),
// not just executable literals. Reviewed prose lines are grandfathered by exact line-SHA fingerprints
// (inventory/runtime-pin-prose-fingerprints.json) with occurrence reconciliation.
// Gate runs are ~6s each, so mutations that assert independent errors are BATCHED into one run.

/** Apply several mutations at once, run the gate once, assert it fails naming EVERY needle, restore. */
function expectBatchedFailures(mutations, needles, message) {
  const originals = mutations.map(({ path: rel }) => [rel, readFileSync(join(COPY_ROOT, rel), "utf8")]);
  try {
    for (const { path: rel, mutate } of mutations) {
      writeFileSync(join(COPY_ROOT, rel), mutate(readFileSync(join(COPY_ROOT, rel), "utf8")));
    }
    const result = runGate();
    assert.equal(result.status, 1, message);
    for (const needle of needles) {
      assert.ok(result.stderr.includes(needle), `gate error must name ${needle}: ${result.stderr}`);
    }
  } finally {
    for (const [rel, text] of originals) writeFileSync(join(COPY_ROOT, rel), text, "utf8");
  }
  for (const [rel, text] of originals) {
    assert.equal(readFileSync(join(COPY_ROOT, rel), "utf8"), text, `${rel} must be restored byte-exactly`);
  }
}

test("cdt: numeric pin + non-semver suffix in comments/JSON/regex/prose fails naming the whole candidate", () => {
  const suffixes = ["^evil", "~", "|2.0.0", "!x", ";evil", ":evil", "&evil", "\\evil", "$evil", "${x}"];
  const commentBlock = suffixes
    .map((s) => `// review note: pinned to onnxruntime-web@1.21.0${s} here`)
    .join("\n");
  expectBatchedFailures(
    [
      {
        path: "models/animegan-cartoonization/worker.js",
        mutate: (c) =>
          `${c}\n${commentBlock}\n// new prose: uses @huggingface/transformers@3.7.5, per notes\nconst __probe_re = /onnxruntime-web@1.21.0;rx/;\n`,
      },
      {
        path: "models/animegan-cartoonization/_questions.json",
        mutate: (c) =>
          c.replace(`"schemaVersion": 1`, `"schemaVersion": 1, "probe": "uses @huggingface/transformers@3.7.5|4.0.0 range"`),
      },
    ],
    [
      ...suffixes.map((s) => `1.21.0${s}`),
      "3.7.5,", // new prose line: mtu false-red policy still applies outside the fingerprint set
      "1.21.0;rx", // regex literal
      "3.7.5|4.0.0", // JSON evidence prose
    ],
    "every suffixed numeric pin must be judged whole and rejected, in every context",
  );
});

test("cdt: exact pin followed by a TAB byte stays green through the real GREP path", () => {
  const p = join(COPY_ROOT, "models/animegan-cartoonization/worker.js");
  const orig = readFileSync(p, "utf8");
  writeFileSync(p, `${orig}\n// pin: onnxruntime-web@1.21.0\t(tab-separated exact pin)\n`, "utf8");
  try {
    const result = runGate();
    assert.equal(result.status, 0, `an exact pin followed by a TAB must stay green: ${result.stderr}`);
  } finally {
    writeFileSync(p, orig, "utf8");
  }
  // And the engines agree: the runtime token (imported from the gate) breaks on TAB in JS too.
  const m = "onnxruntime-web@1.21.0\tnext".match(new RegExp(`onnxruntime-web@(${audit.VERSION_TOKEN})`));
  assert.equal(m?.[1], "1.21.0", "TAB must terminate the candidate in the JS engine as well");
});

test("cdt: VERSION_TOKEN carries a literal TAB byte and no engine-divergent constructs", () => {
  assert.ok(audit.VERSION_TOKEN.includes("\t"), "runtime token must contain a literal TAB byte");
  assert.ok(!audit.VERSION_TOKEN.includes("[:space:]"), "[:space:] is not valid in a JS RegExp");
  assert.ok(!audit.VERSION_TOKEN.includes("\\s"), "\\s is literal characters to POSIX grep");
  const breakClass = audit.VERSION_TOKEN.slice(audit.VERSION_TOKEN.indexOf("[^"));
  assert.ok(!breakClass.includes("\\"),
    "the break class must NOT exclude backslash (1.21.0\\evil must stay whole)");
  assert.ok(!breakClass.includes("$"),
    "the break class must NOT exclude $ (1.21.0$evil and 1.21.0${x} must stay whole)");
});

test("cdt: clean tree fingerprints reconcile exactly (gate green, 19 reviewed entries)", () => {
  const fps = JSON.parse(readFileSync(join(COPY_ROOT, "inventory/runtime-pin-prose-fingerprints.json"), "utf8"));
  assert.equal(fps.fingerprints.length, 19, "the reviewed set is exactly the 19 known prose lines");
  const result = runGate();
  assert.equal(result.status, 0, `clean tree must pass with fingerprints: ${result.stderr}`);
});

test("cdt: fingerprint tampering fails closed (edit / duplicate / remove / ledger independence)", () => {
  // Editing a fingerprinted line breaks its SHA.
  expectVersionFailure(
    "models/yolo11-detection/worker.js",
    (c) => c.replace("onnxruntime-web@1.21.0),", "onnxruntime-web@1.21.0) ;"),
    "prose fingerprint drift",
    "editing a fingerprinted line must break the SHA and trip drift reconciliation",
  );
  // Duplicating it exceeds the reviewed occurrence count: the extra occurrence is NOT suppressed
  // (only the first match consumes the fingerprint), so it surfaces as an unauthorized candidate.
  expectVersionFailure(
    "models/yolov10-detection/worker.js",
    (c) => `${c}${c.split("\n").find((l) => l.includes("onnxruntime-web@1.21.0),"))}\n`,
    `unauthorized onnxruntime-web version "1.21.0),"`,
    "a duplicated fingerprinted line must fail on the extra occurrence",
  );
  // Removing the fingerprinted candidate leaves the entry unmatched.
  expectVersionFailure(
    "models/surface-normals/_questions.json",
    (c) => c.replace("onnxruntime-web@1.21.0)", "onnxruntime-web@1.21.0 "),
    "prose fingerprint drift",
    "removing a fingerprinted candidate must trip drift reconciliation",
  );
  // Fingerprint suppression never masks the golden-ledger drift net.
  expectVersionFailure(
    "inventory/runtime-pin-marker-ledger.json",
    (c) => c.replace(`"sha256"`, `"sha256x"`, 1),
    "runtime marker golden ledger drift",
    "ledger drift must still fail while prose fingerprints suppress their own lines",
  );
});

test("cdt: the gate's grep scans use execFileSync argv (no shell interpolation)", () => {
  const src = readFileSync(join(COPY_ROOT, "scripts/audit-model-currency.mjs"), "utf8");
  assert.ok(src.includes('execFileSync("grep"'), "the gate must invoke grep via argv");
  assert.ok(!/execSync\(\s*`grep/.test(src), "no shell-interpolated grep call may remain in the gate");
});
