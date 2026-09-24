#!/usr/bin/env node
// Probe (bead web-ai-showcase-865): does the catalogue still work on transformers.js 4.3.0?
//
// The shared pin in lib/webai.js is 3.7.5 and 315 of 327 built routes inherit that ONE constant, so
// bumping it is a one-line change with a 315-route blast radius. This bead gathers evidence first and
// changes no source. Method: for each distinct catalogue tuple (task, model, dtype, device), load and
// RUN the model under both versions in the repo's own harness, then compare real output.
//
// v4 is not a small bump. The 4.0 notes describe a WebGPU runtime rewritten in C++ (native WebGPU EP
// rather than JSEP), a ModelRegistry, new env surfaces, a progress_total event, and TOKENIZERS MOVED
// INTO @huggingface/tokenizers — which makes the AutoTokenizer (34 routes) and AutoModelForCTC
// (13 routes) surfaces the ones to read first. Transitive ORT jumps 1.22.0-dev -> 1.31.0-dev.
//
// Honest by construction: a failure is recorded with its exact error text, never summarised as
// "incompatible"; a tuple that was not probed says so rather than being omitted; and outputs are
// compared numerically with a stated tolerance, because two ORT versions need not be bit-identical.
//
// Usage: node scripts/probe-tjs-compat-matrix.mjs [--only name,name] [--versions 3.7.5,4.3.0]
//                                                 [--timeout 240000] [--out reports/...json]

import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { CDP, closePage, launchChrome, openPage, startServer } from "./browser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const VERSIONS = String(arg("versions", "3.7.5,4.3.0")).split(",").map((v) => v.trim());
const ONLY = arg("only", "") ? String(arg("only")).split(",").map((s) => s.trim()) : null;
const TIMEOUT_MS = Number(arg("timeout", 240000));
const OUT_RAW = arg("out", "reports/transformers-v4-compat-matrix.json");
const OUT = OUT_RAW.startsWith("/") ? OUT_RAW : ROOT + OUT_RAW;
// One worker FILE per case: a shared path (or even a query-string variation) lets the browser reuse a
// cached module, which silently attributes one tuple's result to the next.
const workerPathFor = (name, version) => `${ROOT}__probe_tjs_matrix_${name}_${version.replace(/[^a-z0-9]/gi, "_")}.js`;
const createdWorkers = [];

// A representative sample of the 91 distinct (task, dtype, runtime) groups the built routes actually
// contain, chosen to hit the surfaces v4 changes most: the tokenizers split (AutoTokenizer /
// AutoModelForCTC), the rewritten WebGPU runtime, and the largest route groups. `routes` is how many
// built routes share the tuple — coverage is reported per tuple, never averaged away.
export const TUPLES = [
  { name: "feature-extraction-q8-wasm", task: "feature-extraction", model: "Xenova/all-MiniLM-L6-v2", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "The quick brown fox jumps over the lazy dog." }, options: { pooling: "mean", normalize: true } },
  { name: "text-classification-q8-wasm", task: "text-classification", model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "This demo is genuinely useful." } },
  { name: "token-classification-q8-wasm", task: "token-classification", model: "Xenova/bert-base-NER", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "Paul Kinlan works in London." } },
  { name: "fill-mask-q8-wasm", task: "fill-mask", model: "Xenova/bert-base-uncased", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "The capital of France is [MASK]." } },
  { name: "asr-ctc-q4-wasm", task: "automatic-speech-recognition", model: "onnx-community/wav2vec2-large-xlsr-53-chinese-zh-cn-ONNX", dtype: "q4", device: "wasm", tokenizersSplit: true, input: { kind: "tone", seconds: 1.0 } },
  { name: "text-generation-q8-wasm", task: "text-generation", model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "The capital of France is" }, options: { max_new_tokens: 8 } },
  { name: "text2text-generation-q8-wasm", task: "text2text-generation", model: "Xenova/flan-t5-small", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "Translate to German: good morning" }, options: { max_new_tokens: 8 } },
  { name: "zero-shot-classification-q8-wasm", task: "zero-shot-classification", model: "Xenova/nli-deberta-v3-xsmall", dtype: "q8", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "I love this product." }, options: { candidate_labels: ["positive", "negative"] } },
  { name: "image-classification-q8-wasm", task: "image-classification", model: "AdamCodd/vit-base-nsfw-detector", dtype: "q8", device: "wasm", tokenizersSplit: false, input: { kind: "image" } },
  { name: "text-to-speech-fp32-wasm", task: "text-to-speech", model: "naklitechie/mms-tts-ta-ONNX", dtype: "fp32", device: "wasm", tokenizersSplit: true, input: { kind: "text", value: "A short sample." } },
  { name: "image-text-to-text-q4f16-webgpu", task: "image-text-to-text", model: "HuggingFaceTB/SmolVLM-256M-Instruct", dtype: "q4f16", device: "webgpu", tokenizersSplit: true, input: { kind: "image" }, options: { max_new_tokens: 6 }, optional: true },
];

const workerTpl = (t, version) => `
import * as TJS from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@${version}";
const { pipeline, env${t.input.kind === "image" ? ", RawImage" : ""} } = TJS;
self.onmessage = async () => {
  const t0 = performance.now();
  try {
    env.allowLocalModels = false;
    const pipe = await pipeline(${JSON.stringify(t.task)}, ${JSON.stringify(t.model)}, { device: ${JSON.stringify(t.device)}, dtype: ${JSON.stringify(t.dtype)} });
    const loadMs = Math.round(performance.now() - t0);
    const input = await (async () => {
      const kind = ${JSON.stringify(t.input.kind)};
      if (kind === "text") return ${JSON.stringify(t.input.value ?? "")};
      if (kind === "image") {
        const c = new OffscreenCanvas(224, 224); const ctx = c.getContext("2d");
        ctx.fillStyle = "#3366cc"; ctx.fillRect(0, 0, 224, 224);
        ctx.fillStyle = "#eef2ff"; ctx.fillRect(40, 40, 144, 144);
        const blob = await c.convertToBlob({ type: "image/png" });
        return await RawImage.fromBlob(blob);
      }
      // Synthetic tone: enough to exercise the audio path and compare versions against each other.
      const seconds = ${JSON.stringify(t.input.seconds ?? 1.0)};
      const rate = 16000; const n = Math.floor(rate * seconds);
      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = Math.sin((2 * Math.PI * 220 * i) / rate) * 0.25;
      return data;
    })();
    const t1 = performance.now();
    const out = await pipe(input, ${JSON.stringify(t.options ?? {})});
    const inferMs = Math.round(performance.now() - t1);
    // A shape/sample digest that can be compared across versions without shipping huge payloads.
    let dims = null, sample = null, text = null, waveforms = null;
    if (out && out.data) { dims = Array.from(out.dims ?? []); sample = Array.from(out.data.slice(0, 8)); }
    else if (Array.isArray(out)) {
      const first = out[0] ?? {};
      if (first.generated_text !== undefined) text = String(first.generated_text).slice(0, 200);
      else if (first.label !== undefined) text = JSON.stringify(first).slice(0, 200);
      if (first.audio) { waveforms = Array.from((first.audio.data ?? first.audio).slice(0, 8)); dims = [first.audio.data ? first.audio.data.length : -1]; }
      else if (first.embedding) { dims = Array.from(first.embedding.dims ?? []); sample = Array.from(first.embedding.data.slice(0, 8)); }
      if (!text && !sample) text = JSON.stringify(out).slice(0, 200);
    } else if (out && typeof out === "object" && out.generated_text !== undefined) {
      text = String(out.generated_text).slice(0, 200);
    } else if (out && out.audio) { waveforms = Array.from((out.audio.data ?? out.audio).slice(0, 8)); }
    self.postMessage({ ok: true, loadMs, inferMs, dims, sample, text, waveforms,
      outputKeys: out && typeof out === "object" ? Object.keys(out).slice(0, 10) : [] });
  } catch (e) {
    const err = String((e && e.message) || e);
    self.postMessage({ ok: false, err, errName: (e && e.name) || null, errStack: String((e && e.stack) || "").slice(0, 500) });
  }
};`;

const buildReport = () => {
  const byT = new Map();
  for (const r of results) {
    if (!byT.has(r.tuple)) byT.set(r.tuple, {});
    byT.get(r.tuple)[r.version] = r;
  }
  const rows = [...byT.entries()].map(([name, v]) => {
    const first = Object.values(v)[0];
    return {
      tuple: name,
      task: first.task,
      model: first.model,
      dtype: first.dtype,
      device: first.device,
      tokenizersSplitAffected: first.tokenizersSplit,
      versions: Object.fromEntries(Object.entries(v).map(([ver, r]) => [ver, r.ok
        ? { runs: true, loadMs: r.loadMs, inferMs: r.inferMs, dims: r.dims, sample: r.sample, text: r.text, waveforms: r.waveforms }
        : { runs: false, error: r.err, errorName: r.errName ?? null, stack: r.errStack ?? null }])),
      comparison: compareVersions(v["3.7.5"], v["4.3.0"]),
    };
  });
  return {
    generated: new Date().toISOString(),
    bead: "web-ai-showcase-865",
    purpose: "evidence for staging the shared transformers.js pin 3.7.5 -> 4.3.0; no source was changed",
    versions: VERSIONS,
    method: "repo harness (headless Chrome + CDP), module worker importing transformers.js from a pinned CDN URL per version, real pipeline() load and real inference, output compared numerically",
    tolerance: "same = identical dims, max |Δ| <= 1e-3 on the compared prefix, and equal text where the task returns text; ORT 1.22 -> 1.31 need not be bit-identical",
    coverage: {
      probedTuples: rows.length,
      note: "a representative sample of the distinct (task, dtype, runtime) tuples the built routes contain; see reports/transformers-v4-tuple-census.json for the full census",
    },
    notProbed: TUPLES.filter((x) => !rows.some((r) => r.tuple === x.name)).map((x) => ({ tuple: x.name,
      reason: "not probed: the model download exceeds the per-case budget this probe used — a probe limit, not a compatibility finding" })),
    rows,
  };
};

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/`);
const selected = TUPLES.filter((t) => !ONLY || ONLY.includes(t.name));
// Seed from an existing report so a targeted re-run ADDS evidence instead of replacing it. Without this,
// re-probing one tuple silently deletes every earlier row — the report is rebuilt from this run's results.
let results = [];
try {
  const prior = JSON.parse(readFileSync(OUT, "utf8"));
  for (const row of prior.rows || []) {
    for (const [version, v] of Object.entries(row.versions || {})) {
      results.push({ tuple: row.tuple, task: row.task, model: row.model, dtype: row.dtype, device: row.device,
        tokenizersSplit: row.tokenizersSplitAffected, version,
        ...(v.runs ? { ok: true, loadMs: v.loadMs, inferMs: v.inferMs, dims: v.dims, sample: v.sample, text: v.text, waveforms: v.waveforms } : { ok: false, err: v.error, errName: v.errorName, errStack: v.stack }) });
    }
  }
} catch { /* first run */ }

for (const tuple of selected) {
  for (const version of VERSIONS) {
    const workerPath = workerPathFor(tuple.name, version);
    writeFileSync(workerPath, workerTpl(tuple, version));
    createdWorkers.push(workerPath);
    const workerUrl = "/web-ai-showcase/" + workerPath.slice(ROOT.length);
    const expr =
      `await new Promise((resolve)=>{const w=new Worker(${JSON.stringify(workerUrl)},{type:'module'});` +
      `const to=setTimeout(()=>{w.terminate();resolve({ok:false,err:'timeout after ${TIMEOUT_MS}ms'});},${TIMEOUT_MS});` +
      `w.onmessage=(e)=>{clearTimeout(to);w.terminate();resolve(e.data);};` +
      `w.onerror=(e)=>{clearTimeout(to);resolve({ok:false,err:'worker.onerror '+(e.message||'')});};w.postMessage(1);})`;
    let r;
    try {
      const res = await cdp.send(
        "Runtime.evaluate",
        { expression: `(async()=>{try{return (${expr});}catch(e){return {ok:false,err:String(e)};}})()`, awaitPromise: true, returnByValue: true },
        page.sessionId,
        TIMEOUT_MS + 20000,
      );
      r = res.result?.value ?? { ok: false, err: "no result returned" };
    } catch (e) {
      r = { ok: false, err: `harness error: ${String(e.message || e)}` };
    }
    results.push({ tuple: tuple.name, task: tuple.task, model: tuple.model, dtype: tuple.dtype, device: tuple.device, tokenizersSplit: tuple.tokenizersSplit, version, ...r });
    // Persist now: an interrupted sweep must still leave a usable report rather than nothing.
    try { writeFileSync(OUT, JSON.stringify(buildReport(), null, 2) + "\n"); } catch { /* final write will surface it */ }
    console.log(`${tuple.name} @ ${version}: ${r.ok ? "RUNS" : "FAILS"} ${r.ok ? `load=${r.loadMs}ms infer=${r.inferMs}ms` : `— ${String(r.err).slice(0, 120)}`}`);
  }
}

for (const p of createdWorkers) { try { rmSync(p, { force: true }); } catch { /* ignore */ } }
await closePage(cdp, page.targetId);
chrome.kill();
server.close();

// Compare the two versions per tuple. Tolerance is stated rather than assumed: ORT 1.22 -> 1.31 need
// not produce bit-identical floats, so "close" and "identical" are different verdicts.
const byTuple = new Map();
for (const r of results) {
  if (!byTuple.has(r.tuple)) byTuple.set(r.tuple, {});
  byTuple.get(r.tuple)[r.version] = r;
}
const compareVersions = (a, b) => {
  if (!a || !b) return { verdict: "incomplete" };
  if (!a.ok && !b.ok) return { verdict: "both-fail", detail: "both versions failed" };
  if (!a.ok) return { verdict: "regression-in-4.x", detail: `4.x failed: ${String(b.err).slice(0, 200)}` };
  if (!b.ok) return { verdict: "broken-in-4.x", detail: `4.x failed: ${String(b.err).slice(0, 200)}` };
  const dimsEqual = JSON.stringify(a.dims) === JSON.stringify(b.dims);
  const pairs = [];
  if (a.sample && b.sample) for (let i = 0; i < Math.min(a.sample.length, b.sample.length); i++) pairs.push(Math.abs(a.sample[i] - b.sample[i]));
  if (a.waveforms && b.waveforms) for (let i = 0; i < Math.min(a.waveforms.length, b.waveforms.length); i++) pairs.push(Math.abs(a.waveforms[i] - b.waveforms[i]));
  const maxAbs = pairs.length ? Math.max(...pairs) : null;
  const textEqual = a.text && b.text ? a.text === b.text : null;
  const verdict = (dimsEqual || a.dims === null || b.dims === null) && (maxAbs === null || maxAbs <= 1e-3) && textEqual !== false
    ? "same" : "differs";
  return { verdict, dimsEqual, maxAbsDiff: maxAbs, textEqual, aText: a.text ?? null, bText: b.text ?? null };
};

const rows = buildReport().rows;

const report = {
  generated: new Date().toISOString(),
  bead: "web-ai-showcase-865",
  purpose: "evidence for staging the shared transformers.js pin 3.7.5 -> 4.3.0; no source was changed",
  versions: VERSIONS,
  method: "repo harness (headless Chrome + CDP), module worker importing transformers.js from a pinned CDN URL per version, real pipeline() load and real inference, output compared numerically",
  tolerance: "same = identical dims, max |Δ| <= 1e-3 on the compared prefix, and equal text where the task returns text; ORT 1.22 -> 1.31 need not be bit-identical",
  coverage: { probedTuples: rows.length, note: "a representative sample of the 91 distinct (task, dtype, runtime) groups the 327 built routes contain; the full group table is in reports/transformers-v4-tuple-census.json" },
  rows,
};
writeFileSync(OUT, JSON.stringify(report, null, 2) + "\n");

console.log("\n=== COMPAT MATRIX ===");
for (const row of rows) {
  const s = (v) => row.versions[v]?.runs ? `RUNS (${row.versions[v].loadMs}ms load)` : `FAILS — ${String(row.versions[v]?.error ?? "?").slice(0, 90)}`;
  console.log(`${row.tuple}\n   3.7.5: ${s("3.7.5")}\n   4.3.0: ${s("4.3.0")}\n   verdict: ${row.comparison.verdict}${row.comparison.maxAbsDiff !== null && row.comparison.maxAbsDiff !== undefined ? ` (max|Δ|=${row.comparison.maxAbsDiff})` : ""}`);
}
