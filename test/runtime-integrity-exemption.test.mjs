// Adversarial tests for the derived-inventory exemption in the currency gate.
//
// The exemption lets the GENERATED integrity inventory (sw.js's embedded manifest block, the JSON it
// is generated from, and the generator) hold runtime URL literals without them being read as
// unauthorised route pins. That is a deliberate hole in a security/currency gate, so these tests exist
// to prove the hole is the size we claim and no larger.
//
// A previous revision of this exemption was BLOCKED because it was file-scoped: a pin placed inside the
// generated block, or a hand-written pin in the generator's arrays, was silently exempted. Rather than
// record that finding once in a review comment, each case below re-instates the attack and asserts the
// gate still fails. If someone widens the exemption again, these tests fail.
//
// Each case mutates a file, runs the gate, and asserts the gate's OWN error text names the offending
// version (not merely a non-zero exit). Every mutation happens inside an ISOLATED COPY materialised
// from `git archive HEAD`, never in the live checkout: `node --test` runs test FILES in parallel, and a
// test that rewrote sw.js / the allowlist / models/* in place raced with the other pin tests and, on a
// SIGKILL or timeout, left those tracked files corrupted forever (the finally never ran).

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, execSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// The isolated copy the mutations are applied to. Materialised once in `before`, removed in `after`.
let COPY_ROOT = null;

/** Shell-quote a value for interpolation into a single shell pipeline. */
const shq = (s) => `'${String(s).replaceAll("'", "'\\''")}'`;

/**
 * Materialise a clean, immutable copy of the repository from the committed HEAD. HEAD (unlike the live
 * working tree, which other test FILES mutate in parallel) is a stable tree, so two mutating tests can
 * never capture one another's half-written files. Running `--check` from this copy exercises the audit's
 * `import.meta.url`-relative root resolution (everything resolves against the copy, not this checkout).
 */
function materializeIsolatedRepo() {
  const dir = mkdtempSync(join(tmpdir(), "web-ai-currency-exemption-"));
  execSync(`git archive HEAD | tar -x -C ${shq(dir)}`, {
    cwd: ROOT,
    stdio: "pipe",
    maxBuffer: 64 * 1024 * 1024,
  });
  assert.ok(
    existsSync(join(dir, "scripts/audit-model-currency.mjs")),
    "the isolated copy must contain the audit script",
  );
  return dir;
}

before(() => {
  COPY_ROOT = materializeIsolatedRepo();
});

after(() => {
  if (COPY_ROOT) rmSync(COPY_ROOT, { recursive: true, force: true });
});

/** Runs the currency gate inside the isolated copy. Returns its exit status AND stderr. */
function runGate() {
  try {
    execFileSync("node", [join(COPY_ROOT, "scripts/audit-model-currency.mjs"), "--check"], {
      cwd: COPY_ROOT,
      stdio: "pipe",
    });
    return { status: 0, stderr: "" };
  } catch (error) {
    return {
      status: typeof error.status === "number" ? error.status : 1,
      stderr: String(error.stderr ?? ""),
    };
  }
}

/** Asserts the gate failed AND its error text names the offending version, then restores the file. */
function expectVersionFailure(relativePath, mutate, versionPattern, message) {
  const path = join(COPY_ROOT, relativePath);
  const original = readFileSync(path, "utf8");
  try {
    writeFileSync(path, mutate(original));
    const result = runGate();
    assert.equal(result.status, 1, message);
    assert.match(result.stderr, versionPattern, `gate error must mention ${versionPattern}: ${result.stderr}`);
  } finally {
    writeFileSync(path, original);
  }
  assert.equal(readFileSync(path, "utf8"), original, `${relativePath} must be restored byte-exactly`);
}

const insertBeforeClosingMarker = (source, line) => {
  const lines = source.split("\n");
  const index = lines.findIndex((l) => l.includes("<<< runtime-integrity"));
  assert.ok(index > 0, "sw.js must contain the closing runtime-integrity marker");
  lines.splice(index, 0, line);
  return lines.join("\n");
};

const insertAfterOpeningMarker = (source, line) => {
  const lines = source.split("\n");
  const index = lines.findIndex((l) => l.includes(">>> runtime-integrity (generated"));
  assert.ok(index > 0, "sw.js must contain the opening runtime-integrity marker");
  lines.splice(index + 1, 0, line);
  return lines.join("\n");
};

const ORT_PIN = 'importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@9.9.9/dist/ort.min.js");';

test("baseline: the gate passes with no mutation", () => {
  assert.equal(runGate().status, 0, "the unmutated tree must pass; the cases below are only meaningful against a green baseline");
});

test("an unauthorised pin in sw.js OUTSIDE the generated block still fails", () => {
  expectVersionFailure(
    "sw.js",
    (source) => `${source}\n${ORT_PIN}\n`,
    /9\.9\.9/,
    "a pin appended after the generated block must not be exempt",
  );
});

test("an unauthorised pin in sw.js INSIDE the generated block still fails", () => {
  // This is the case that defeated the previous file-scoped revision.
  expectVersionFailure("sw.js", (source) => insertBeforeClosingMarker(source, `const smuggled = "https://cdn.jsdelivr.net/npm/onnxruntime-web@9.9.9/dist/ort.wasm.min.mjs";`), /9\.9\.9/, "a pin smuggled into the generated block must not be exempt");
});

test("an unauthorised pin ON the closing marker line still fails", () => {
  expectVersionFailure(
    "sw.js",
    (source) =>
      source
        .split("\n")
        .map((line) => (line.includes("<<< runtime-integrity") ? `const m = "onnxruntime-web@9.9.9"; ${line}` : line))
        .join("\n"),
    /9\.9\.9/,
    "a pin riding on a marker line must not be exempt",
  );
});

test("a duplicated generated marker fails loudly instead of widening the exemption", () => {
  const path = join(COPY_ROOT, "sw.js");
  const original = readFileSync(path, "utf8");
  try {
    writeFileSync(
      path,
      (() => {
        const lines = original.split("\n");
        const index = lines.findIndex((line) => line.includes(">>> runtime-integrity (generated"));
        assert.ok(index > 0, "sw.js must contain the opening runtime-integrity marker");
        lines.splice(index, 0, "// >>> runtime-integrity (generated by scripts/runtime-integrity.mjs)");
        return lines.join("\n");
      })(),
    );
    const result = runGate();
    assert.equal(result.status, 1, "duplicated markers must disable the exemption, not extend it");
    assert.match(result.stderr, /marker/, `gate error must mention the marker problem: ${result.stderr}`);
  } finally {
    writeFileSync(path, original);
  }
});

test("an unauthorised pin in a MODEL ROUTE still fails", () => {
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => `${source}\n${ORT_PIN}\n`,
    /9\.9\.9/,
    "no model route may ever be exempt from the currency gate",
  );
});

test("an unauthorised version in the GENERATOR's arrays still fails", () => {
  // scripts/runtime-integrity.mjs is NOT generated - it holds hand-written arrays - so exempting it
  // wholesale (as the first revision did) hid real pin decisions.
  expectVersionFailure(
    "scripts/runtime-integrity.mjs",
    (source) => source.replace("const ORT_ASSETS = [", 'const ORT_ASSETS = [\n  "onnxruntime-web@9.9.9/dist/ort.wasm.min.mjs",'),
    /9\.9\.9/,
    "the generator must not be a place to hide an unauthorised pin",
  );
});

test("an unauthorised version written into runtime-integrity.json still fails", () => {
  expectVersionFailure(
    "runtime-integrity.json",
    (source) => {
      const data = JSON.parse(source);
      data.urls["https://cdn.jsdelivr.net/npm/onnxruntime-web@9.9.9/dist/ort.wasm.min.mjs"] = {
        sha256: "0".repeat(64),
        bytes: 1,
        policy: "verify-then-cache",
      };
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    /9\.9\.9/,
    "the generated data file must not be a place to authorise a new pin",
  );
});

test("adding an arbitrary file to derivedInventory.files is rejected", () => {
  expectVersionFailure(
    "scripts/runtime-pin-allowlist.json",
    (source) => {
      const data = JSON.parse(source);
      data.derivedInventory.files.push("models/animegan-cartoonization/worker.js");
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    /derivedInventory\.files/,
    "the exempt file set must be a fixed allowlist, not data anyone can extend",
  );
});

test("COORD'S EXACT BYPASS: adding a route file to derivedInventory.files AND injecting a pin into it", () => {
  // The precise mutation to close: exempt a route file via the data, then hide a forbidden pin inside it.
  // If the exempt set is only "currently exact", this succeeds and the audit PASSES. It must not.
  // A UNIQUE directory, so the test can never collide with - or delete - anything that already exists.
  // All of it happens inside the isolated copy, never the live checkout.
  const uniqueName = `__exemption-test-${process.pid}-${Date.now()}`;
  const routeRel = `models/${uniqueName}/index.html`;
  const routeAbs = join(COPY_ROOT, routeRel);
  const routeDir = join(COPY_ROOT, "models", uniqueName);
  const allowPath = join(COPY_ROOT, "scripts/runtime-pin-allowlist.json");
  const allowOriginal = readFileSync(allowPath, "utf8");
  assert.equal(existsSync(routeDir), false, "the unique temp directory must not already exist");
  try {
    mkdirSync(routeDir, { recursive: true });
    writeFileSync(routeAbs, `<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@9.9.9/dist/ort.min.js"></script>\n`);
    const data = JSON.parse(allowOriginal);
    data.derivedInventory.files.push(routeRel);
    writeFileSync(allowPath, `${JSON.stringify(data, null, 2)}\n`);
    const result = runGate();
    assert.equal(result.status, 1, "exempting a route file via derivedInventory.files must be rejected outright");
    assert.match(result.stderr, /derivedInventory\.files/, `gate error must name the rejected file: ${result.stderr}`);
  } finally {
    writeFileSync(allowPath, allowOriginal);
    try {
      // Remove ONLY the directory this test created, by its unique name.
      rmSync(routeDir, { recursive: true, force: true });
    } catch {
      // best effort cleanup
    }
  }
  assert.equal(readFileSync(allowPath, "utf8"), allowOriginal, "allowlist must be restored byte-exactly");
  assert.equal(existsSync(routeAbs), false, "the temporary route file must be removed");
});

test("the exempt set cannot be silently narrowed either", () => {
  expectVersionFailure(
    "scripts/runtime-pin-allowlist.json",
    (source) => {
      const data = JSON.parse(source);
      data.derivedInventory.files = ["sw.js"];
      return `${JSON.stringify(data, null, 2)}\n`;
    },
    /EXACTLY/,
    "the exempt set must equal the intended immutable list exactly",
  );
});

test("an INVERTED marker pair fails loudly", () => {
  const path = join(COPY_ROOT, "sw.js");
  const original = readFileSync(path, "utf8");
  try {
    writeFileSync(
      path,
      (() => {
        const lines = original.split("\n");
        const open = lines.findIndex((l) => l.includes(">>> runtime-integrity (generated"));
        const close = lines.findIndex((l) => l.includes("<<< runtime-integrity"));
        const tmp = lines[open];
        lines[open] = lines[close];
        lines[close] = tmp;
        return lines.join("\n");
      })(),
    );
    const result = runGate();
    assert.equal(result.status, 1, "an inverted marker pair must disable the exemption and error");
    assert.match(result.stderr, /ORDERED|marker/, `gate error must mention the marker problem: ${result.stderr}`);
  } finally {
    writeFileSync(path, original);
  }
});

test("the same-line bypass: a SECOND pin on an already-exempt line is still examined", () => {
  // The transformers scan used to read only the FIRST pin per line, so an unapproved version appended to
  // a line that already carried an exempt one was never examined at all and the gate passed.
  expectVersionFailure(
    "scripts/runtime-integrity.mjs",
    (source) =>
      source.replace(
        '"@huggingface/transformers@3.1.2/+esm"',
        '"@huggingface/transformers@3.1.2/+esm", "@huggingface/transformers@9.9.9/+esm"',
      ),
    /9\.9\.9/,
    "a second, unapproved pin on the same line must not ride along",
  );
});

test("same-line bypass (a): a NON-exempt model route with two pins on one line", () => {
  // The first pin (3.7.5, the shared version) is authorised, the second (9.9.9) is not. A scan that read
  // only the first pin per line would pass; grep -o must split them so the second is judged independently.
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => `${source}\n// @huggingface/transformers@3.7.5 @huggingface/transformers@9.9.9\n`,
    /9\.9\.9/,
    "an unapproved transformers pin on a line with an approved one must still fail",
  );
});

test("same-line bypass (b): the onnxruntime-web scan with two pins on one line", () => {
  // 1.21.0 is allowlisted for raw-ORT routes, 9.9.9 is not. The -o split must expose the second pin.
  expectVersionFailure(
    "models/animegan-cartoonization/worker.js",
    (source) => `${source}\n// onnxruntime-web@1.21.0 onnxruntime-web@9.9.9\n`,
    /9\.9\.9/,
    "an unapproved onnxruntime-web pin on a line with an approved one must still fail",
  );
});

test("same-line bypass (c): two pins on one line inside the sw.js generated block", () => {
  // 3.7.5 is a measured version and so is exempt INSIDE the generated block; 9.9.9 is not measured, so it
  // must still fail even though it shares a line with an exempt pin. The version-scoped exemption must not
  // let a line-scoped first-match swallow the unapproved second pin.
  expectVersionFailure(
    "sw.js",
    (source) => insertAfterOpeningMarker(source, "// @huggingface/transformers@3.7.5 @huggingface/transformers@9.9.9"),
    /9\.9\.9/,
    "a second, unapproved pin inside the generated block must not be exempt",
  );
});
