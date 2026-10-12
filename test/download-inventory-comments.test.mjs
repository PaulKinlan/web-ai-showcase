// Focused, deterministic tests for the route-download-inventory classifier (web-ai-showcase-x4ba).
//
// WHY: the classifier used to match runtime-family API names in the demo's raw source text, COMMENTS
// INCLUDED, with first-match-wins over ordered regexes. A comment naming another runtime's API could
// therefore change a route's public download declaration: measured in the wild, a comment containing
// "ort.env" flipped ai-image-detection from transformers-pipeline to raw-ort — claiming site-controlled
// bytes and restart-only resume for a route that downloads through loadPipeline. These tests pin the
// three properties that fix requires: comments cannot influence the family, executable tokens still can,
// and a source that cannot be lexed fails the gate closed instead of being classified from raw text.
//
// The fixture tree under test/fixtures/download-inventory-comments/models/ is copied into an isolated
// temp root together with the WORKING-TREE gate scripts, so the real classifier and the real checker run
// end-to-end over synthetic demos. No browser, no network, no model weights, and the repository's own
// models/ tree and download-routes.json are never touched.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE = join(ROOT, "test", "fixtures", "download-inventory-comments", "models");
const ALL_SLUGS = ["comment-names-ort", "comment-names-mediapipe", "code-ort", "string-traps", "html-comments", "ambiguous"];

/** Isolated repo root: the chosen fixture demos, the working-tree gate scripts, a node_modules link. */
function makeRoot(slugs) {
  const dir = mkdtempSync(join(tmpdir(), "web-ai-x4ba-"));
  mkdirSync(join(dir, "scripts"), { recursive: true });
  for (const slug of slugs) cpSync(join(FIXTURE, slug), join(dir, "models", slug), { recursive: true });
  for (const f of ["route-download-inventory.mjs", "check-download-inventory.mjs"]) {
    copyFileSync(join(ROOT, "scripts", f), join(dir, "scripts", f));
  }
  symlinkSync(join(ROOT, "node_modules"), join(dir, "node_modules"), "dir");
  return dir;
}

/** Run the classifier over the isolated tree and return its inventory + diagnostics. */
function classify(dir) {
  const out = join(dir, "download-routes.json");
  const evidencePath = join(dir, "evidence.json");
  execFileSync(
    "node",
    ["scripts/route-download-inventory.mjs", "--out", out, "--evidence", evidencePath],
    { cwd: dir, stdio: "pipe" },
  );
  return {
    inventory: JSON.parse(readFileSync(out, "utf8")),
    evidence: JSON.parse(readFileSync(evidencePath, "utf8")),
  };
}

/** Run the fail-closed gate in the isolated tree; never throws, so the message can be asserted. */
function check(dir) {
  try {
    execFileSync("node", ["scripts/check-download-inventory.mjs"], { cwd: dir, stdio: "pipe" });
    return { code: 0, message: "" };
  } catch (error) {
    return { code: error.status, message: `${error.stdout ?? ""}${error.stderr ?? ""}` };
  }
}

const byFamily = (inventory) => Object.fromEntries(inventory.routes.map((r) => [r.slug, r.family]));

test("a comment naming another runtime's API cannot change a route's download family", (t) => {
  const dir = makeRoot(ALL_SLUGS);
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const { inventory, evidence } = classify(dir);
  assert.deepEqual(byFamily(inventory), {
    // raw-ort is tested BEFORE transformers-pipeline, so this fixture reproduced the false flip.
    "comment-names-ort": "transformers-pipeline",
    // transformers-resumable-prefetch is the FIRST family in FAMILY_ORDER and mediapipe is named too,
    // both only in a comment: the earliest family must not be winnable from a comment.
    "comment-names-mediapipe": "webllm",
    // the true ORT route must stay raw-ort - comments ignored, executable tokens still matched.
    "code-ort": "raw-ort",
    // comment openers inside strings/regexes/templates are data and must survive masking.
    "string-traps": "transformers-pipeline",
    // both the HTML comment and the inline <script>'s JS comment are ignored.
    "html-comments": "transformers-pipeline",
    ambiguous: "transformers-pipeline",
  });
  // The diagnostic names the exact token and file that produced each family.
  assert.equal(evidence.routes["comment-names-ort"].family.token, "lib/webai.js");
  assert.equal(evidence.routes["comment-names-ort"].family.file, "models/comment-names-ort/worker.js");
  assert.equal(evidence.routes["comment-names-mediapipe"].family.rule, "webllm");
  assert.equal(evidence.routes["comment-names-mediapipe"].family.file, "models/comment-names-mediapipe/detector.js");
  assert.match(evidence.routes["comment-names-mediapipe"].family.token, /CreateMLCEngine|lib\/webllm\.js/);
  assert.equal(evidence.routes["code-ort"].family.rule, "raw-ort");
  assert.match(evidence.routes["code-ort"].family.token, /onnxruntime-web|InferenceSession/);
  // The unlexable fixture is reported, and it did not contribute to the family decision.
  assert.equal(evidence.routes.ambiguous.ambiguous.length, 1);
  assert.match(evidence.routes.ambiguous.ambiguous[0], /broken\.js/);
  assert.equal(evidence.routes.ambiguous.family.file, "models/ambiguous/index.html");
  assert.deepEqual(evidence.routes["html-comments"].ambiguous, []);
});

test("the gate passes when the committed inventory matches the source", (t) => {
  const dir = makeRoot(["comment-names-ort", "code-ort", "string-traps"]);
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  classify(dir); // writes download-routes.json, exactly as a maintainer would
  assert.deepEqual(check(dir), { code: 0, message: "" });
});

test("a stale record fails naming the matched token and file, not just 'regenerate'", (t) => {
  const dir = makeRoot(["comment-names-ort"]);
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  classify(dir);
  // Simulate the record the pre-fix classifier would have written for this route.
  const path = join(dir, "download-routes.json");
  const inv = JSON.parse(readFileSync(path, "utf8"));
  inv.routes[0].family = "raw-ort";
  inv.routes[0].byteControl = "site-controlled";
  inv.routes[0].resume = "restart-only";
  writeFileSync(path, JSON.stringify(inv, null, 2) + "\n");
  const { code, message } = check(dir);
  assert.equal(code, 1);
  assert.match(message, /STALE CLASSIFICATION: comment-names-ort\.family/);
  assert.match(message, /matched 'lib\/webai\.js' in models\/comment-names-ort\/worker\.js/);
  assert.match(message, /regenerate ONLY if the new classification is true/);
});

test("an unlexable source fails the gate closed, and regeneration is not the advice", (t) => {
  const dir = makeRoot(["ambiguous"]);
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const { inventory } = classify(dir);
  // Classification itself is unaffected (the signal comes from index.html) - the source is still refused.
  assert.equal(inventory.routes[0].family, "transformers-pipeline");
  const { code, message } = check(dir);
  assert.equal(code, 1);
  assert.match(message, /AMBIGUOUS SOURCE: ambiguous/);
  assert.match(message, /broken\.js/);
  assert.match(message, /regenerating will not help/);
  assert.doesNotMatch(message, /Regenerate: node scripts\/route-download-inventory\.mjs/);
});
