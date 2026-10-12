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
import { execSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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
