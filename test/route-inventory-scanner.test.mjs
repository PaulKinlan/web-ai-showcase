// Deterministic fixture tests for the executable-context route scanner (web-ai-showcase-6rgd, ADR kc7p).
//
// Drives scripts/route-download-inventory.mjs against synthetic demo dirs (mkdtemp) — no browser, no
// corpus dependency. Covers the three mandated regressions plus the fail-closed rule:
//   · j0pm: executable-LOOKING code inside an inert <template> must never classify (raw-ort stays out).
//   · CRLF line endings inside inline scripts must not break extraction or classification.
//   · executable inline ORT (ort.env member relation, punctuation preserved) classifies raw-ort.
//   · <pre><code> display samples (the 8-route corpus finding) are inert documentation, never signals.
//   · a script that fails to parse fails CLOSED (unknown), never a guessed family.
//   · importmap is declarative config: visible as string/import signals, not executed code.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync, spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

/** Build a synthetic demo dir and classify it; returns the route record. */
function classifyRoute(files) {
  const root = mkdtempSync(join(tmpdir(), "dr-fix-"));
  try {
    const dir = join(root, "demo"); // the scanner treats each subdirectory of --models as a route slug
    mkdirSync(dir, { recursive: true });
    for (const [name, content] of Object.entries(files)) writeFileSync(join(dir, name), content);
    const out = join(root, "out.json");
    execSync(`node scripts/route-download-inventory.mjs --models ${JSON.stringify(root)} --out ${JSON.stringify(out)}`, {
      stdio: ["ignore", "ignore", "pipe"],
    });
    return JSON.parse(readFileSync(out, "utf8")).routes[0];
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("j0pm regression: inert <template> code never classifies; executable module script does", () => {
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<template><script>ort.env.wasm.numThreads=2</script></template>
<script type="module">import {loadPipeline} from "lib/webai.js";
loadPipeline()</script>
</body></html>`,
  });
  // The inert template's ort.env must NOT classify raw-ort; the executable import wins.
  assert.equal(r.family, "transformers-pipeline");
  assert.equal(r.adoption, "bypass"); // loadPipeline is not createModelLoader
  assert.equal(r.status, "blocked");
});

test("executable inline ORT classifies raw-ort with the ort.env member relation intact", () => {
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<script>ort.env.wasm.numThreads = 2; ort.env.logLevel = "info";</script>
</body></html>`,
  });
  assert.equal(r.family, "raw-ort");
  assert.equal(r.byteControl, "site-controlled");
  assert.equal(r.resume, "restart-only");
});

test("CRLF line endings in inline scripts do not break extraction or classification", () => {
  const r = classifyRoute({
    "index.html": [
      "<!doctype html><html><body>",
      "<script type=\"module\">",
      "import { loadPipeline } from \"lib/webai.js\";",
      "loadPipeline({ model: \"x\" });",
      "</script>",
      "</body></html>",
    ].join("\r\n"),
  });
  assert.equal(r.family, "transformers-pipeline");
});

test("<pre><code> display samples are inert documentation, never classification signals", () => {
  // The exact corpus false-positive shape (bge-reranker et al.): the API docs show
  // AutoTokenizer.from_pretrained, while the executable code uses the shared loader.
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<section><h2>How the API works</h2>
<pre><code>import { AutoTokenizer } from "@huggingface/transformers";
const tokenizer = await AutoTokenizer.from_pretrained(model_id);</code></pre>
<script type="module">import { loadPipeline } from "/lib/webai.js"; loadPipeline();</script>
</body></html>`,
    "worker.js": `import { loadPipeline } from "/lib/webai.js";\nexport const boot = () => loadPipeline();\n`,
  });
  assert.equal(r.family, "transformers-pipeline");
});

test("malformed script FAILS CLOSED as unknown (never a guessed family)", () => {
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<script type="module">import { loadPipeline} from "lib/webai.js"; this is ((( not valid javascript</script>
</body></html>`,
  });
  assert.equal(r.family, "unknown");
  assert.equal(r.status, "unknown");
  assert.match(r.note, /fail/i);
});

test("mediapipe modelAssetBuffer refine works from a Property key + fetch call", () => {
  // interactive-segmenter shape: worker fetches bytes and supplies modelAssetBuffer (site-controlled).
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body><script type="module">
import { FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision";
</script></body></html>`,
    "worker.js":
      `const response = await fetch(MODEL_URL, { cache: "no-store" });\n` +
      `const modelBytes = new Uint8Array(await response.arrayBuffer());\n` +
      `const options = { baseOptions: { modelAssetBuffer: new Uint8Array(modelBytes), delegate: "CPU" } };\n`,
  });
  assert.equal(r.family, "mediapipe");
  assert.equal(r.byteControl, "site-controlled");
  assert.equal(r.resume, "restart-only");
});

test("importmap is declarative config: keys/values are signals, invalid JSON fails closed", () => {
  const ok = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<script type="importmap">{ "imports": { "@mlc-ai/web-llm": "https://esm.run/@mlc-ai/web-llm" } }</script>
<script type="module">import "@mlc-ai/web-llm";</script>
</body></html>`,
  });
  assert.equal(ok.family, "webllm");
  const broken = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<script type="importmap">{ "imports": { not json</script>
<script type="module">console.log("hi");</script>
</body></html>`,
  });
  assert.equal(broken.family, "unknown");
});

test("noscript content is inert; non-executable script types are skipped", () => {
  const r = classifyRoute({
    "index.html":
      `<!doctype html><html><body>
<noscript><script>ort.env.wasm.numThreads=2</script></noscript>
<script type="application/json">{"model": "AutoModel.from_pretrained"}</script>
<script type="module">import { loadPipeline } from "lib/webai.js"; loadPipeline();</script>
</body></html>`,
  });
  assert.equal(r.family, "transformers-pipeline");
});

// ── OPTIONAL diagnostic provenance (bead web-ai-showcase-iem2) ───────────────────────────────────────
// Two halves are pinned here: the diagnostic names a REAL token and the file that carries it (so an
// unexpected classification is actionable), and the public route record gains no field at all (the
// gate reads only download-routes.json). The decision path is untouched — the probes are a separate
// table, and the precedence test below shows the probes cannot reorder the families.

/** Like classifyRoute, but also captures the --evidence provenance and the errors channel. */
function classifyWithEvidence(files) {
  const root = mkdtempSync(join(tmpdir(), "dr-ev-"));
  try {
    const dir = join(root, "demo");
    for (const [name, content] of Object.entries(files)) {
      mkdirSync(dirname(join(dir, name)), { recursive: true });
      writeFileSync(join(dir, name), content);
    }
    const out = join(root, "out.json");
    const evidencePath = join(root, "evidence.json");
    const res = spawnSync(
      "node",
      ["scripts/route-download-inventory.mjs", "--models", root, "--out", out, "--evidence", evidencePath],
      { encoding: "utf8" },
    );
    return {
      route: JSON.parse(readFileSync(out, "utf8")).routes[0],
      evidence: JSON.parse(readFileSync(evidencePath, "utf8")),
      stderr: res.stderr ?? "",
      status: res.status,
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("provenance names the matched token and the file it came from (import string)", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body><script type="module">import { loadPipeline } from "lib/webai.js"; loadPipeline();</script></body></html>`,
    "worker.js": `export const unrelated = "worker";\n`,
  });
  const p = evidence.routes[route.slug];
  assert.equal(route.family, "transformers-pipeline");
  assert.equal(p.rule, "transformers-pipeline");
  assert.equal(p.token, "lib/webai.js");
  // An import specifier is a string Literal, so `strings` already contains it: the kind says so
  // rather than pretending imports and strings can be told apart (reviewer finding on ab5a5f38).
  assert.equal(p.kind, "string-or-import");
  // The token lives in the inline <script>, so naming worker.js here would be a false lead.
  assert.match(p.file, /\.html$/);
});

test("one file contributing several matching script blocks is listed once, not once per block", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body>` +
      `<script type="module">import { loadPipeline } from "lib/webai.js";</script>` +
      `<script type="module">import { loadPipeline } from "lib/webai.js";</script>` +
      `</body></html>`,
  });
  const p = evidence.routes[route.slug];
  assert.equal(route.family, "transformers-pipeline");
  assert.deepEqual(p.files, [...new Set(p.files)], "files must list distinct sources");
  assert.equal(p.files.length, 1);
  assert.match(p.file, /index\.html$/);
});

test("provenance identifies a member-relation match (ort.env) in the inline script", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html": `<!doctype html><html><body><script>ort.env.wasm.numThreads = 2;</script></body></html>`,
  });
  const p = evidence.routes[route.slug];
  assert.equal(route.family, "raw-ort");
  assert.equal(p.token, "ort.env");
  assert.equal(p.kind, "member");
  assert.match(p.file, /index\.html$/);
});

test("provenance attributes a signal to the executable file, never to inert template code", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body><template><script>ort.env.wasm.numThreads=2</script></template></body></html>`,
    "detector.js": `import { CreateMLCEngine } from "@mlc-ai/web-llm";\n`,
  });
  const p = evidence.routes[route.slug];
  assert.equal(route.family, "webllm");
  assert.match(p.token, /@mlc-ai\/web-llm|lib\/webllm\.js/);
  assert.match(p.file, /detector\.js$/);
  assert.ok(
    !p.files.some((f) => f.endsWith("index.html")),
    "the inert template's ort.env must not be named as the source of this family",
  );
});

test("provenance cannot reorder families: a route matching two resolves as FAMILY_ORDER dictates", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body><script type="module">import { loadPipeline } from "lib/webai.js"; ort.env.wasm.numThreads = 2; loadPipeline();</script></body></html>`,
  });
  // raw-ort precedes transformers-pipeline, and the evidence must name the family that actually won.
  assert.equal(route.family, "raw-ort");
  assert.equal(evidence.routes[route.slug].rule, "raw-ort");
});

test("a source that fails to parse is NAMED in evidence and on the errors channel (fail closed)", () => {
  const { route, evidence, stderr } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body><script type="module">import { loadPipeline } from "lib/webai.js";</script></body></html>`,
    "broken.js": `const x = 1; /* unterminated\n`,
  });
  assert.equal(route.family, "unknown"); // still fails closed, as before
  assert.equal(route.status, "unknown");
  assert.match(stderr, /AMBIGUOUS SOURCE: .*broken\.js/); // the operator is told WHICH file
  assert.equal(evidence.unparseable.length, 1);
  assert.match(evidence.unparseable[0].file, /broken\.js$/);
  assert.match(evidence.unparseable[0].error, /comment/i);
});

test("provenance stays out of the public inventory: a route record keeps exactly its 10 fields", () => {
  const { route, evidence } = classifyWithEvidence({
    "index.html":
      `<!doctype html><html><body><script type="module">import { loadPipeline } from "lib/webai.js"; loadPipeline();</script></body></html>`,
  });
  assert.deepEqual(Object.keys(route).sort(), [
    "adoption", "byteControl", "family", "hasWorker", "multiModel", "note", "resume", "route", "slug", "status",
  ]);
  // ...while the diagnostic still carries what the record deliberately does not.
  assert.equal(evidence.routes[route.slug].token, "lib/webai.js");
});

test("--evidence with no path fails loudly instead of crashing or writing a file named after a flag", () => {
  const root = mkdtempSync(join(tmpdir(), "dr-cli-"));
  try {
    mkdirSync(join(root, "demo"), { recursive: true });
    writeFileSync(join(root, "demo", "index.html"), `<!doctype html><html><body></body></html>`);
    // Absolute script path + `cwd: root`: a *relative* write (e.g. a file named after a flag) therefore
    // lands in `root`, which is where the assertion below looks. Without this the assertion checked a
    // directory the script never wrote to and passed vacuously (reviewer P3 on b020ebb5).
    const script = join(process.cwd(), "scripts", "route-download-inventory.mjs");
    const run = (args) => spawnSync("node", [script, ...args], { cwd: root, encoding: "utf8" });
    for (const args of [["--models", root, "--evidence"], ["--models", root, "--evidence", "--check"]]) {
      const res = run(args);
      assert.equal(res.status, 1, `expected a loud failure for ${args.join(" ")}`);
      assert.match(res.stderr, /--evidence requires a target path/);
      assert.doesNotMatch(res.stderr, /ERR_INVALID_ARG_TYPE/); // never the raw crash
    }
    assert.ok(!existsSync(join(root, "--check")), "must not create a file named after a flag");
    // The fail-closed contract for --check is unchanged: an unparseable source still exits non-zero.
    writeFileSync(join(root, "demo", "broken.js"), `const x = 1; /* unterminated\n`);
    const check = run(["--models", root, "--check"]);
    assert.equal(check.status, 1);
    assert.match(check.stderr, /AMBIGUOUS SOURCE: .*broken\.js/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
