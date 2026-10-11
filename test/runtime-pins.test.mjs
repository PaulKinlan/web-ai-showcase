// CORE runtime-pin currency tests — the default-suite (fast) tier (bead web-ai-showcase-ehf).
//
// This file is the in-gate half of the runtime-pin protection. It proves, WITHOUT mutating
// anything, that the committed tree is authorized and internally consistent:
//   - the allowlist is structurally valid (semver, documented reason/evidence, real slugs),
//   - checkRuntimePins() reports zero unauthorized pins on the real tree AND each scan pattern
//     is non-vacuous (per-runtime floors) and actually finds the known fixture URLs,
//   - the binary (NUL-byte) pass works, captures whole candidates, and is empty on the clean tree,
//   - the clean tree passes `--check` end-to-end; prose fingerprints regenerate byte-identically
//     and reconcile; the constant-bound floor holds; the marker ledger reconciles.
//
// The ADVERSARIAL half — every test that MUTATES the isolated copy to prove the gate CATCHES a
// bad pin (MUTANT PROOF / STRICT RAW / FLOATING / LITERAL BOUNDARY / LITERAL LEDGER / cdt
// tampering / vs2 falsification; ~60+ full gate runs, ~7 minutes measured, bead ehf) — moved to
// test-adversarial/runtime-pins-adversarial.test.mjs, OUTSIDE the default `node --test
// "test/**/*.test.mjs"` glob (the same layout test-browser/ uses to stay out of the browser-free
// suite). Run it with `deno task test:adversarial`; CI runs it as a scheduled job
// (.github/workflows/gate.yml) so its result is durable without taxing every push.
//
// The STATIC runtime-pin audit itself is unchanged and still runs in every gate (deno task gate,
// the fleet CHECK_CMD, and CI) — this split moves only where the MUTATION proofs live.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { materializeIsolatedRepo } from "./lib/isolated-repo.mjs";
import { makeGateHelpers } from "./lib/runtime-pins-gate.mjs";

// The isolated copy the binary-probe tests write into. Materialised once in `before`, removed in `after`.
let COPY_ROOT = null;
// The copy's audit module, dynamically imported so checkRuntimePins()/findPinsInBinaryFiles() resolve
// their own ROOT against the isolated copy (never the live checkout).
let audit = null;
// Shared, single-sourced gate helpers bound to the copy (see test/lib/runtime-pins-gate.mjs).
let runGate, grepScan, writeBinaryProbe;

before(async () => {
  COPY_ROOT = materializeIsolatedRepo();
  // The working-tree audit script (the file under test) is overlaid onto the committed snapshot by
  // the shared helper test/lib/isolated-repo.mjs, so this import resolves to the CURRENT source.
  audit = await import(pathToFileURL(join(COPY_ROOT, "scripts/audit-model-currency.mjs")).href);
  ({ runGate, grepScan, writeBinaryProbe } = makeGateHelpers({ copyRoot: COPY_ROOT, audit }));
});

after(() => {
  if (COPY_ROOT) rmSync(COPY_ROOT, { recursive: true, force: true });
});

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

test("cdt: the gate's grep scans use execFileSync argv (no shell interpolation)", () => {
  const src = readFileSync(join(COPY_ROOT, "scripts/audit-model-currency.mjs"), "utf8");
  assert.ok(src.includes('execFileSync("grep"'), "the gate must invoke grep via argv");
  assert.ok(!/execSync\(\s*`grep/.test(src), "no shell-interpolated grep call may remain in the gate");
});

test("cdt: committed prose fingerprints are in sync with a fresh regeneration (--check)", () => {
  // The fingerprint file is a review artifact: it must be byte-identical to what the committed
  // generator produces from the tree, so nobody can hand-edit a suppression in. Runs in the
  // isolated copy so the working-tree generator + fingerprints overlay is what gets checked.
  const result = (() => {
    try {
      execFileSync("node", [join(COPY_ROOT, "scripts/generate-prose-fingerprints.mjs"), "--check"], {
        cwd: COPY_ROOT,
        stdio: "pipe",
        encoding: "utf8",
      });
      return { status: 0, stderr: "" };
    } catch (e) {
      return { status: e.status ?? 1, stderr: String(e.stderr ?? "") };
    }
  })();
  assert.equal(result.status, 0, `fingerprint freshness check must pass: ${result.stderr}`);
});

// --- constant-bound dynamic CDN URL templates (bead web-ai-showcase-vs2) -------------------
// The three raw-ORT workers pin the runtime via a top-level const (ORT_VERSION / ORT_VER) and a
// template URL (`…/onnxruntime-web@${NAME}/dist/…`, twice per worker: import + wasmPaths). Before
// vs2 the gate saw only the ledgered template TEXT — mutating the constant to 9.9.9 passed. Now
// untagged, escape-free templates whose ${} expressions all resolve to top-level string constants
// are judged on the reconstructed specifier, and the 6 worker ledger rows are removed, so an
// unbound site is an unreviewed marker → RED. (The MUTATION halves of vs2 live in the adversarial
// tier; these are the clean-tree floors that stay in the default suite.)

test("vs2: clean tree — exactly 6 bound sites (floor), ledger reconciles, probe-tjs parameter row stays green", async () => {
  const result = runGate();
  assert.equal(result.status, 0, `clean tree must pass: ${result.stderr}`);
  // constantBound is an additive floor, never a mask for the legacy/literal counts.
  const allowlist = JSON.parse(readFileSync(join(COPY_ROOT, "scripts/runtime-pin-allowlist.json"), "utf8"));
  const { checkLiteralRuntimePins } = await import(pathToFileURL(join(COPY_ROOT, "scripts/runtime-pin-literals.mjs")).href);
  const derived = () => false; // floor probe only: we count, we do not exempt
  const r = checkLiteralRuntimePins(COPY_ROOT + "/", "models/silero-vad/ models/pitch-detection/ models/model2vec-static-embeddings/", allowlist, derived);
  assert.ok(r.counters.constantBound >= 6, `constantBound floor: expected >= 6, got ${r.counters.constantBound}`);
});

test("vs2: binding does not over-reach — tooling regex templates and parameter templates stay ledgered", () => {
  // The probe-tjs matrix template's usage must remain ledger-visible (row intact) and the gate
  // stays green because the row still matches.
  const result = runGate();
  assert.equal(result.status, 0, `probe-tjs parameter template must stay ledgered: ${result.stderr}`);
  const ledger = JSON.parse(readFileSync(join(COPY_ROOT, "inventory/runtime-pin-marker-ledger.json"), "utf8"));
  assert.equal(ledger.entries.length, 21, "ledger holds 21 rows after the 6 worker rows were removed");
  const probe = ledger.entries.find((e) => e.path === "scripts/probe-tjs-compat-matrix.mjs");
  assert.ok(probe && probe.disposition === "vs2-dynamic", "the parameter-bound probe row must remain");
  const workerRows = ledger.entries.filter((e) => e.path?.startsWith("models/silero-vad/") || e.path?.startsWith("models/pitch-detection/") || e.path?.startsWith("models/model2vec-static-embeddings/"));
  assert.equal(workerRows.length, 0, "no worker template rows remain in the ledger");
});
