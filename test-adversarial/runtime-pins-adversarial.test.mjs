// ADVERSARIAL runtime-pin currency tests — the scheduled heavy tier (bead web-ai-showcase-ehf).
//
// Every test here MUTATES the isolated repository copy (materialised from `git archive HEAD` by
// test/lib/isolated-repo.mjs — never the live checkout) and runs the REAL currency gate to prove
// it CATCHES the bad pin. Each proof is a full repo-wide gate run (~6s measured), and this file
// holds ~60+ of them: measured standalone at ~7 minutes (bead ehf), which is far too heavy for the
// default per-commit suite but exactly right as a separately scheduled tier.
//
// Layout (mirrors test-browser/): this file lives OUTSIDE test/ so the default
// `node --test "test/**/*.test.mjs"` suite — run by CI on every push/PR and by the fleet gate —
// stays fast, while the clean-tree/core protections remain in test/runtime-pins.test.mjs and the
// static audit (`scripts/audit-model-currency.mjs --check`) still runs in EVERY gate. This tier is
// run by `deno task test:adversarial` and by the scheduled CI job in .github/workflows/gate.yml
// (daily cron + manual workflow_dispatch), so its result is durable without taxing every push.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { materializeIsolatedRepo } from "../test/lib/isolated-repo.mjs";
import { makeGateHelpers } from "../test/lib/runtime-pins-gate.mjs";

// The isolated copy the mutations are applied to. Materialised once in `before`, removed in `after`.
let COPY_ROOT = null;
// The copy's audit module, dynamically imported so checkRuntimePins()/findPinsInBinaryFiles() resolve
// their own ROOT against the isolated copy (never the live checkout).
let audit = null;
// Shared, single-sourced gate helpers bound to the copy (see test/lib/runtime-pins-gate.mjs).
let runGate, writeBinaryProbe, expectVersionFailure, expectExecutableUrlVerdict, expectBatchedFailures;

before(async () => {
  COPY_ROOT = materializeIsolatedRepo();
  // The working-tree audit script (the file under test) is overlaid onto the committed snapshot by
  // the shared helper test/lib/isolated-repo.mjs, so this import resolves to the CURRENT source.
  audit = await import(pathToFileURL(join(COPY_ROOT, "scripts/audit-model-currency.mjs")).href);
  ({
    runGate,
    writeBinaryProbe,
    expectVersionFailure,
    expectExecutableUrlVerdict,
    expectBatchedFailures,
  } = makeGateHelpers({ copyRoot: COPY_ROOT, audit }));
});

after(() => {
  if (COPY_ROOT) rmSync(COPY_ROOT, { recursive: true, force: true });
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
  // web-ai-showcase-vs2: the 6 worker rows were REMOVED (constant binding now judges those
  // specifiers whole). The remaining dynamic-expression row is the parameter-bound probe template;
  // changing its text must still invalidate its exact fingerprint.
  expectVersionFailure(
    "scripts/probe-tjs-compat-matrix.mjs",
    (source) => source.replace(
      "@huggingface/transformers@${version}",
      "@huggingface/transformers@${version}/dist/transformers.min.js",
    ),
    "runtime marker golden ledger drift",
    "a changed reviewed dynamic expression requires independent ledger review",
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

// --- numeric-suffix seam closure (bead web-ai-showcase-cdt) ------------------------------
// VERSION_TOKEN's break set is now ONLY `/`, a literal space/TAB byte, quotes, backtick, `<`, `>`.
// Every other byte after a numeric pin — ^ ~ | ! @ ; : , & \ $ ${ } — is part of the judged
// candidate, in EVERY context the legacy scans see (comments, JSON prose, regexes, bare specifiers),
// not just executable literals. Reviewed prose lines are grandfathered by exact line-SHA fingerprints
// (inventory/runtime-pin-prose-fingerprints.json) with occurrence reconciliation.
// Gate runs are ~6s each, so mutations that assert independent errors are BATCHED into one run.

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

// --- constant-bound dynamic CDN URL templates (bead web-ai-showcase-vs2) -------------------
// The three raw-ORT workers pin the runtime via a top-level const (ORT_VERSION / ORT_VER) and a
// template URL (`…/onnxruntime-web@${NAME}/dist/…`, twice per worker: import + wasmPaths). Before
// vs2 the gate saw only the ledgered template TEXT — mutating the constant to 9.9.9 passed. Now
// untagged, escape-free templates whose ${} expressions all resolve to top-level string constants
// are judged on the reconstructed specifier, and the 6 worker ledger rows are removed, so an
// unbound site is an unreviewed marker → RED. (The clean-tree floors for vs2 live in
// test/runtime-pins.test.mjs; these are the falsification proofs.)

test("vs2: the original false-green is closed — each worker constant mutated to 9.9.9 fails naming the bound value", () => {
  const workers = [
    ["models/silero-vad/worker.js", "ORT_VERSION"],
    ["models/pitch-detection/worker.js", "ORT_VERSION"],
    ["models/model2vec-static-embeddings/worker.js", "ORT_VER"],
  ];
  for (const [rel, name] of workers) {
    expectVersionFailure(
      rel,
      (c) => c.replace(new RegExp(`const ${name} = "[^"]+"`), `const ${name} = "9.9.9"`),
      `unauthorized onnxruntime-web version "9.9.9" (bound from const ${name} at ${rel}:`,
      `mutating ${name} to 9.9.9 in ${rel} must fail naming the bound value and its origin`,
    );
  }
});

test("vs2: binding failures fail closed (rename / shadow / let / computed / tagged)", () => {
  // Rename the constant's DECLARATION only: the template's ${ORT_VERSION} no longer resolves.
  expectVersionFailure(
    "models/silero-vad/worker.js",
    (c) => c.replace(`const ORT_VERSION = "1.20.1"`, `const ORT_V2 = "1.20.1"`),
    "runtime marker golden ledger drift",
    "an unresolvable ${} must leave the marker unjudged → ledger drift RED",
  );
  // Shadow the constant inside a function scope around usage.
  expectVersionFailure(
    "models/pitch-detection/worker.js",
    (c) => c.replace(`const ORT_VERSION = "1.20.1";`, `let ORT_VERSION = "1.20.1";`),
    "runtime marker golden ledger drift",
    "a let binding must not resolve (only const string literals bind)",
  );
  // Computed expression is not a bare Identifier.
  expectVersionFailure(
    "models/model2vec-static-embeddings/worker.js",
    (c) => c.replace("onnxruntime-web@${ORT_VER}/dist/ort.wasm.min.mjs", 'onnxruntime-web@${ORT_VER + ""}/dist/ort.wasm.min.mjs'),
    "runtime marker golden ledger drift",
    "a computed ${} expression must fail closed to the ledger",
  );
  // Tagged template: the cooked quasi text is not necessarily the runtime string.
  expectVersionFailure(
    "models/silero-vad/worker.js",
    (c) => `const __tag = (s, v) => s.raw[0] + "9.9.9" + s.raw[1];\n${c}`.replace(
      "const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;",
      "const ORT_URL = __tag`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;",
    ),
    "runtime marker golden ledger drift",
    "a tagged template must be unresolvable → ledger RED (the tag could rewrite the version)",
  );
});
