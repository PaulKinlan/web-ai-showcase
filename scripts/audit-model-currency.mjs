#!/usr/bin/env node
// Model + runtime currency audit (bead web-ai-showcase-fb9).
//
// Three questions, answered with evidence:
//   1. Runtime currency — which transformers.js build does each route load, and is a newer one
//      published?
//   2. Checkpoint liveness — for the artifacts a route REALLY pulls weights from (not just the
//      catalogue `hfId`, which is often the upstream base model while the route loads an
//      `onnx-community/*-ONNX` or MLC build), does it still exist, is it gated, and can the
//      declared quant still be satisfied?
//   3. Catalogue accuracy — does `models.json` cite the repo the code requests, and does its
//      `dtype` match the quant the worker actually loads?
//
// Usage:
//   node scripts/audit-model-currency.mjs                    # sweep → reports/model-currency.{json,md}
//   node scripts/audit-model-currency.mjs --slug <slug>      # one route (extraction debug)
//   node scripts/audit-model-currency.mjs --limit 20         # smoke run
//   node scripts/audit-model-currency.mjs --check            # diff vs inventory/model-currency.json
//
// Scope: a refining lower bound over the catalogued built routes at this scan depth — never a
// completeness claim. Network-dependent; an offline run reports `error` rather than guessing.

import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { classifyTaskPair } from "./model-task-vocabulary.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const SNAPSHOT = ROOT + "inventory/model-currency.json";
const REPORT_JSON = ROOT + "reports/model-currency.json";
const REPORT_MD = ROOT + "reports/model-currency.md";
const ALLOWLIST_PATH = ROOT + "scripts/runtime-pin-allowlist.json";
export const PIN_SCAN_TARGETS = "models/ lib/ public/ scripts/ search/ models.json sw.js";

const args = process.argv.slice(2);
const CHECK_ONLY = args.includes("--check");
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const SLUG = flag("--slug");
const LIMIT = flag("--limit") ? Number(flag("--limit")) : Infinity;
const REFRESH = args.includes("--refresh"); // ignore the cache entirely
const MAX_LIVE = flag("--max-live") ? Number(flag("--max-live")) : 300; // polite request budget
const CACHE_TTL_MS = (flag("--cache-hours") ? Number(flag("--cache-hours")) : 24) * 3600 * 1000;
const CONCURRENCY = 3;
const REQUEST_SPACING_MS = 120;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- id extraction ---------------------------------------------------------------------
// Ids come only from model-loading call sites. A generic `org/name` scan over prose and HTML
// links yielded 87 ids from 12 routes, 70 of them bogus 401s — precision or nothing.
const ID_PATTERNS = [
  /from_pretrained\(\s*["'`]([^"'`]+)["'`]/g,
  /\bmodelId\s*:\s*["'`]([^"'`]+)["'`]/g,
  /\bmodel_id\s*[:=]\s*["'`]([^"'`]+)["'`]/g,
  /\bmodel\s*:\s*["'`]([^"'`]+)["'`]/g,
  /\bCreateMLCEngine\(\s*["'`]([^"'`]+)["'`]/g,
  /\bcreateEngine\(\s*["'`]([^"'`]+)["'`]/g,
  /["'`]([A-Za-z0-9_.\-]+\/[A-Za-z0-9_.\-]+)\/resolve\/main/g,
  /\bdata-model(?:-id)?\s*=\s*["']([^"']+)["']/g,
];
const ID_SHAPE = /^[A-Za-z0-9][\w.-]{0,79}\/[A-Za-z0-9][\w.-]{0,79}$/;
const NON_HF =
  /^(w3\.org|schema\.org|github\.com|unpkg\.com|cdn\.|fonts\.|www\.|developer\.|developers\.|docs?\.)/;
// A repo consulted for architecture/tokenizer only — it serves no weights, so its ONNX layout
// says nothing about whether the route works.
// (repo -> quant) from one call site. A short lookahead keeps the pair inside the same options
// object: crossing a route's whole dtype set against every repo it loads is what produced the
// fashion-clip false positive (fp32 for fashion-clip, q8 for Xenova/clip-vit-base-patch16).
const PAIR_PATTERNS = [
  /(?:from_pretrained\(|modelId\s*:|\bmodel\s*:)\s*["'`]([A-Za-z0-9][\w.-]{0,79}\/[A-Za-z0-9][\w.-]{0,79})["'`][^)]{0,400}?\bdtype:\s*["'`]([a-zA-Z0-9_]+)["'`]/gs,
  /\bdtype:\s*["'`]([a-zA-Z0-9_]+)["'`][^}]{0,400}?\b(?:model|modelId)\s*:\s*["'`]([A-Za-z0-9][\w.-]{0,79}\/[A-Za-z0-9][\w.-]{0,79})["'`]/gs,
];
const CONFIG_ONLY =
  /\b(?:AutoConfig|AutoTokenizer|AutoProcessor)\.from_pretrained\(\s*["'`]([^"'`]+)["'`]/g;

async function signalsFor(slug) {
  const dir = ROOT + `models/${slug}/`;
  const out = {
    runtime: "unknown",
    weightIds: new Set(),
    configIds: new Set(),
    mlcIds: new Set(),
    explicitOnnxFiles: new Set(),
    workerDtypes: [],
    declaredModelFile: null,
    callSitePairs: {},
  };
  if (!existsSync(dir)) return out;
  let files = [];
  try {
    files = (await readdir(dir)).filter((f) => /\.(js|mjs|html)$/.test(f));
  } catch {
    return out;
  }

  const workerPath = dir + "worker.js";
  for (const f of files) {
    const t = await readFile(dir + f, "utf8");
    const isWorker = dir + f === workerPath;

    if (/lib\/webllm\.js|@mlc-ai\/web-llm|CreateMLCEngine|MLCEngine|prebuiltAppConfig/.test(t)) {
      out.runtime = "webllm";
    } else if (/@mediapipe\/tasks-/.test(t)) {
      if (out.runtime !== "webllm") out.runtime = "mediapipe";
    } else if (/onnxruntime-web/.test(t) && !/@huggingface\/transformers/.test(t)) {
      if (out.runtime === "unknown") out.runtime = "raw-ort";
    } else if (/@huggingface\/transformers/.test(t) && out.runtime === "unknown") {
      out.runtime = "transformers.js";
    }

    const configOnly = new Set([...t.matchAll(CONFIG_ONLY)].map((m) => m[1]));
    for (const re of ID_PATTERNS) {
      for (const m of t.matchAll(re)) {
        const id = m[1];
        if (!ID_SHAPE.test(id) || NON_HF.test(id)) continue;
        if (configOnly.has(id)) out.configIds.add(id);
        else out.weightIds.add(id);
      }
    }
    for (const m of t.matchAll(/\b([A-Za-z0-9_.\-]+-q4f16_\d-MLC)\b/g)) out.mlcIds.add(m[1]);
    for (const m of t.matchAll(/["'`]([^"'`]*\.onnx)["'`]/g)) {
      out.explicitOnnxFiles.add(m[1].split("/").pop());
    }
    if (isWorker) {
      for (const m of t.matchAll(PAIR_PATTERNS[0])) out.callSitePairs[m[1]] = m[2];
      for (const m of t.matchAll(PAIR_PATTERNS[1])) out.callSitePairs[m[2]] = m[1];
      out.workerDtypes = [
        ...new Set([...t.matchAll(/\bdtype:\s*["'`]([a-zA-Z0-9_]+)["'`]/g)].map((m) => m[1])),
      ];
      out.declaredModelFile = t.match(/model_file_name:\s*["'`]([^"'`]+)["'`]/)?.[1] ?? null;
    }
  }
  return out;
}

// The vocabulary map lives in ./model-task-vocabulary.mjs — a committed list of (transformers.js
// task, Hub pipeline_tag) pairs that mean the same thing, each with its reason. Mismatches that are in
// the map are recorded on the route as vocabularyEquivalent; anything else is still reported here.

// --- HF API ----------------------------------------------------------------------------
let nextSlot = 0; // global pacing so a sweep never trips HuggingFace rate limiting
async function throttle() {
  const now = Date.now();
  const at = Math.max(now, nextSlot);
  nextSlot = at + REQUEST_SPACING_MS;
  if (at > now) await sleep(at - now);
}

async function hfModel(id, attempt = 0) {
  try {
    await throttle();
    const res = await fetch(`https://huggingface.co/api/models/${id}`, {
      headers: { "user-agent": "web-ai-showcase/currency-audit" },
      signal: AbortSignal.timeout(20000),
    });
    if (res.status === 404) return { id, missing: true };
    if (res.status === 401) return { id, unauthorized: true };
    if (res.status === 429) {
      const retryAfter = Number(res.headers.get("retry-after")) || 0;
      const wait = Math.min(retryAfter * 1000 || 1500 * 2 ** attempt, 30000);
      if (attempt < 6) {
        await sleep(wait + Math.random() * 400);
        return hfModel(id, attempt + 1);
      }
      return { id, rateLimited: true };
    }
    if (res.status >= 500) throw new Error(`HTTP ${res.status}`);
    if (!res.ok) return { id, error: `HTTP ${res.status}` };
    const j = await res.json();
    return {
      id,
      gated: j.gated ?? false,
      disabled: j.disabled ?? false,
      private: j.private ?? false,
      task: j.pipeline_tag ?? null,
      library: j.library_name ?? null,
      lastModified: j.lastModified ?? null,
      sha: j.sha ?? null,
      downloads: j.downloads ?? null,
      likes: j.likes ?? null,
      onnxFiles: (j.siblings || []).map((s) => s.rfilename).filter((f) => f.endsWith(".onnx")),
    };
  } catch (e) {
    if (attempt < 2) {
      await sleep(600 * (attempt + 1));
      return hfModel(id, attempt + 1);
    }
    return { id, error: e.message };
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0, done = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
      if (++done % 50 === 0) console.error(`  …${done}/${items.length}`);
    }
  }));
  return out;
}

// transformers.js quant token → the filename convention its ONNX repos use. Only meaningful for
// route+repo combinations that go through the pipeline's own file resolution.
const DTYPE_TOKENS = {
  fp32: [/^(onnx\/)?model\.onnx$/, /^onnx\/encoder_model\.onnx$/],
  float32: [/^(onnx\/)?model\.onnx$/, /^onnx\/encoder_model\.onnx$/],
  fp16: [/_fp16\.onnx$/],
  q8: [/_quantized\.onnx$/, /_int8\.onnx$/],
  int8: [/_quantized\.onnx$/, /_int8\.onnx$/],
  q4: [/_q4\.onnx$/],
  q4f16: [/_q4f16\.onnx$/],
};

// --- transformers.js pins ---------------------------------------------------------------
async function transformerPins() {
  const references = {};
  const raw = execSync(
    `grep -rhoE '@huggingface/transformers@[0-9]+\\.[0-9]+\\.[0-9]+' ${PIN_SCAN_TARGETS} 2>/dev/null || true`,
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  );
  for (const line of raw.split("\n")) {
    const v = line.split("@").pop();
    if (v) references[v] = (references[v] || 0) + 1;
  }
  const localOverrides = {};
  const files = execSync(
    `grep -rlE '@huggingface/transformers@[0-9]+\\.[0-9]+\\.[0-9]+' models/ 2>/dev/null || true`,
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  ).trim().split("\n").filter(Boolean);
  for (const rel of files) {
    const text = await readFile(ROOT + rel, "utf8");
    for (const m of text.matchAll(/@huggingface\/transformers@([0-9.]+)/g)) {
      localOverrides[m[1]] ||= new Set();
      localOverrides[m[1]].add(rel.split("/")[1]);
    }
  }
  let latest = null, recentStable = [];
  try {
    const j = await (await fetch("https://registry.npmjs.org/@huggingface/transformers", {
      signal: AbortSignal.timeout(20000),
    })).json();
    latest = j["dist-tags"]?.latest ?? null;
    recentStable = Object.keys(j.versions || {}).filter((v) => !v.includes("-")).slice(-6);
  } catch { /* offline: report null, never guess */ }
  const shared =
    (await readFile(ROOT + "lib/webai.js", "utf8")).match(/@huggingface\/transformers@([0-9.]+)/)
      ?.[1] ?? null;
  return {
    shared,
    latest,
    recentStable,
    references,
    localOverrides: Object.fromEntries(
      Object.entries(localOverrides).map(([v, s]) => [v, [...s].sort()]),
    ),
  };
}

// --- main ------------------------------------------------------------------------------
const cat = JSON.parse(await readFile(ROOT + "models.json", "utf8"));
const built = cat.models.filter((m) => m.status === "built");
const selected = SLUG ? built.filter((m) => m.slug === SLUG) : built;
if (SLUG && selected.length === 0) {
  console.error(`no built route with slug ${SLUG}`);
  process.exit(2);
}
const targets = selected.slice(0, LIMIT === Infinity ? undefined : LIMIT);
const routeRecords = {};

function checkRuntimePins() {
  if (!existsSync(ALLOWLIST_PATH)) {
    return ["missing scripts/runtime-pin-allowlist.json — runtime pin allowlist required"];
  }
  let allowlist;
  try {
    allowlist = JSON.parse(readFileSync(ALLOWLIST_PATH, "utf8"));
  } catch (e) {
    return [`failed to parse scripts/runtime-pin-allowlist.json: ${e.message}`];
  }

  const errors = [];

  // 1. Check onnxruntime-web versions
  const allowedOrt = new Set(
    (allowlist.onnxruntimeWeb?.allowedVersions || []).map((v) => v.version),
  );
  try {
    const raw = execSync(
      `grep -rhoE 'onnxruntime-web@[0-9]+\\.[0-9]+\\.[0-9]+' ${PIN_SCAN_TARGETS} 2>/dev/null || true`,
      { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
    );
    const foundOrt = new Set();
    for (const line of raw.split("\n")) {
      const v = line.split("@").pop()?.trim();
      if (v) foundOrt.add(v);
    }
    for (const v of foundOrt) {
      if (!allowedOrt.has(v)) {
        errors.push(
          `unauthorized onnxruntime-web version "${v}" — not in scripts/runtime-pin-allowlist.json`,
        );
      }
    }
  } catch (e) {
    errors.push(`failed to scan onnxruntime-web versions: ${e.message}`);
  }

  // 2. Check @huggingface/transformers versions
  const allowedTjsShared = allowlist.transformers?.shared;
  const allowedTjsOverrides = new Set(
    (allowlist.transformers?.allowedLocalOverrides || []).map((v) => v.version),
  );
  try {
    const raw = execSync(
      `grep -rhoE '@huggingface/transformers@[0-9]+\\.[0-9]+\\.[0-9]+' ${PIN_SCAN_TARGETS} 2>/dev/null || true`,
      { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
    );
    const foundTjs = new Set();
    for (const line of raw.split("\n")) {
      const v = line.split("@").pop()?.trim();
      if (v) foundTjs.add(v);
    }
    for (const v of foundTjs) {
      if (v !== allowedTjsShared && !allowedTjsOverrides.has(v)) {
        errors.push(
          `unauthorized @huggingface/transformers version "${v}" — not in scripts/runtime-pin-allowlist.json`,
        );
      }
    }
  } catch (e) {
    errors.push(`failed to scan @huggingface/transformers versions: ${e.message}`);
  }

  // 3. Check @mlc-ai/web-llm in lib/webllm.js
  if (existsSync(ROOT + "lib/webllm.js")) {
    const text = readFileSync(ROOT + "lib/webllm.js", "utf8");
    const m = text.match(/@mlc-ai\/web-llm@([0-9.]+)/);
    const v = m ? m[1] : null;
    if (v !== allowlist.webLlm?.shared) {
      errors.push(
        `lib/webllm.js pins web-llm version "${v}", expected "${allowlist.webLlm?.shared}" per scripts/runtime-pin-allowlist.json`,
      );
    }
  }

  // 4. Check @mediapipe/tasks-vision in lib/mediapipe.js
  if (existsSync(ROOT + "lib/mediapipe.js")) {
    const text = readFileSync(ROOT + "lib/mediapipe.js", "utf8");
    const m = text.match(/TASKS_VISION_VERSION\s*=\s*["']([0-9.]+)["']/) ||
      text.match(/tasks-vision@([0-9.]+)/);
    const v = m ? m[1] : null;
    if (v !== allowlist.mediapipe?.shared) {
      errors.push(
        `lib/mediapipe.js pins tasks-vision version "${v}", expected "${allowlist.mediapipe?.shared}" per scripts/runtime-pin-allowlist.json`,
      );
    }
  }

  return errors;
}

if (CHECK_ONLY) {
  // Offline gate: no network. Asserts only what committed evidence can support —
  //   exit 1: the catalogue changed since a route was verified (needs a --refresh re-audit),
  //           or an unauthorized runtime version was found.
  //   exit 2: a built route has never been verified (coverage gap)
  //   exit 0: every built route is covered, evidence matches catalogue, and runtime pins are authorized
  if (!existsSync(SNAPSHOT)) {
    console.error(
      "currency check FAILED: inventory/model-currency.json missing — run `node scripts/audit-model-currency.mjs` to generate",
    );
    process.exit(1);
  }
  let snap;
  try {
    snap = JSON.parse(readFileSync(SNAPSHOT, "utf8"));
  } catch (e) {
    console.error(
      `currency check FAILED: inventory/model-currency.json is corrupted: ${e.message}`,
    );
    process.exit(1);
  }
  const rec = snap.routes || {};
  const drifted = [], uncovered = [];
  for (const m of built) {
    const r = rec[m.slug];
    if (!r?.checkedAt) {
      uncovered.push(m.slug);
      continue;
    }
    const changes = [];
    if (r.hfId !== m.hfId) changes.push(`cited model ${r.hfId} → ${m.hfId}`);
    if (r.loadedId && m.loadedId && r.loadedId !== m.loadedId) {
      changes.push(`loaded model ${r.loadedId} → ${m.loadedId}`);
    }
    if (r.catalogueDtype !== (m.dtype ?? null)) {
      changes.push(
        `dtype ${JSON.stringify(r.catalogueDtype)} → ${JSON.stringify(m.dtype ?? null)}`,
      );
    }
    if (changes.length) drifted.push({ slug: m.slug, hfId: m.hfId, changes });
  }

  const pinErrors = checkRuntimePins();

  const report = {
    generated: new Date().toISOString(),
    mode: "check (offline)",
    builtRoutes: built.length,
    verifiedRoutes: built.length - uncovered.length,
    drifted,
    uncovered,
    pinErrors,
  };

  if (drifted.length || uncovered.length || pinErrors.length) {
    console.error("=== CURRENCY & RUNTIME PIN GATE ===");
    if (pinErrors.length) {
      console.error(`FAIL — ${pinErrors.length} unauthorized runtime pin(s):`);
      for (const e of pinErrors) console.error(`  ✗ ${e}`);
    }
    if (drifted.length) {
      console.error(
        `FAIL — ${drifted.length} route(s) changed since verification — re-run with --refresh:`,
      );
      for (const d of drifted) console.error(`  ✗ ${d.slug}: ${d.changes.join(", ")}`);
    }
    if (uncovered.length) {
      console.error(
        `FAIL — ${uncovered.length}/${built.length} built route(s) never verified in inventory: ${
          uncovered.slice(0, 5).join(", ")
        }`,
      );
    }
    process.exit(1);
  }

  console.log("=== CURRENCY & RUNTIME PIN GATE ===");
  console.log(
    `PASS — ${built.length} built routes covered; evidence matches catalogue; runtime pins authorized.`,
  );
  process.exit(0);
}

console.error(`Scanning ${targets.length} built route(s) for runtime + requested model ids…`);
const cited = new Set(),
  weightIds = new Set(),
  configIds = new Set(),
  perRoute = [],
  mlcRoutes = [];
for (const m of targets) {
  const sig = await signalsFor(m.slug);
  for (const id of sig.weightIds) weightIds.add(id);
  for (const id of sig.configIds) configIds.add(id);
  cited.add(m.hfId);
  if (sig.mlcIds.size) mlcRoutes.push({ slug: m.slug, mlcIds: [...sig.mlcIds] });
  const loadedId = m.loadedId || (sig.weightIds.size > 0 ? [...sig.weightIds][0] : m.hfId);
  const loadedVia = m.loadedVia ||
    (sig.runtime !== "unknown" ? sig.runtime : (m.runtime || "transformers.js"));
  if (loadedId && loadedVia !== "mediapipe" && loadedVia !== "mlc-cdn" && loadedVia !== "webllm") {
    weightIds.add(loadedId);
  }
  routeRecords[m.slug] = {
    hfId: m.hfId,
    loadedId,
    loadedVia,
    catalogueDtype: m.dtype ?? null,
    checkedAt: new Date().toISOString(),
  };
  perRoute.push({
    slug: m.slug,
    hfId: m.hfId,
    loadedId,
    loadedVia,
    task: m.task,
    catalogueDtype: m.dtype ?? null,
    runtime: sig.runtime,
    weightIds: [...sig.weightIds].sort(),
    configIds: [...sig.configIds].sort(),
    explicitOnnxFiles: [...sig.explicitOnnxFiles].sort(),
    declaredModelFile: sig.declaredModelFile,
    workerDtypes: sig.workerDtypes,
    callSitePairs: sig.callSitePairs,
  });
}

const looksLikeRepo = (id) => ID_SHAPE.test(id) && !NON_HF.test(id) && !id.startsWith("mediapipe/");
const sweep = [...new Set([...cited, ...weightIds, ...configIds])].filter(looksLikeRepo).sort();
const snapshot = existsSync(SNAPSHOT)
  ? JSON.parse(await readFile(SNAPSHOT, "utf8"))
  : { models: {} };
const prev = snapshot.models || {};
// Only a successful verification satisfies the cache; failures and throttles are retried next run.
const fresh = (id) =>
  !REFRESH && prev[id]?.outcome === "verified" && prev[id]?.verifiedAt &&
  Date.now() - Date.parse(prev[id].verifiedAt) < CACHE_TTL_MS;
const cachedCount = sweep.filter(fresh).length;

// Never-checked and stalest repos go first, so a budgeted run converges over successive runs
// instead of re-checking the head of an alphabetical list and starving the tail.
const queue = [...sweep].sort((a, b) => {
  const ca = prev[a]?.checkedAt ? Date.parse(prev[a].checkedAt) : 0;
  const cb = prev[b]?.checkedAt ? Date.parse(prev[b].checkedAt) : 0;
  return ca - cb || a.localeCompare(b);
}).filter((id) => !fresh(id));

console.error(
  `Health-checking ${sweep.length} unique HF repos (${cited.size} cited · ${weightIds.size} weights · ${configIds.size} config-only)…`,
);

let liveFetches = 0;
const deferred = new Set();
const fetched = await mapLimit(queue, CONCURRENCY, async (id) => {
  if (liveFetches >= MAX_LIVE) {
    deferred.add(id);
    return { id, deferred: true };
  }
  liveFetches++;
  return hfModel(id);
});
// Cached evidence is replayed into the same flag fields a live fetch would set, so a 401 or a
// 404 established on an earlier run keeps being reported instead of quietly disappearing.
const replay = (rec) =>
  !rec ? {} : {
    ...rec,
    ...(rec.outcome === "missing" ? { missing: true } : {}),
    ...(rec.outcome === "unauthorized" ? { unauthorized: true } : {}),
    ...(rec.outcome === "rate-limited" ? { rateLimited: true } : {}),
    ...(rec.outcome === "error" ? { error: rec.errorMessage || "recorded error" } : {}),
  };
const api = new Map([
  ...sweep.map((id) => [id, replay(prev[id])]),
  ...fetched.map((r) => [r.id, r]),
]);

const findings = [], records = {};
for (const id of sweep) {
  const r = api.get(id) ?? {};
  const asWeights = perRoute.filter((rt) => {
    if (
      rt.loadedVia === "mediapipe" ||
      rt.loadedVia === "mlc-cdn" ||
      rt.loadedVia === "webllm"
    ) {
      return false;
    }
    if (rt.loadedId && rt.loadedId !== rt.hfId) {
      return rt.loadedId === id;
    }
    return rt.loadedId === id || rt.weightIds.includes(id);
  });
  const asConfig = perRoute.filter((rt) => rt.configIds.includes(id));
  const f = {
    hfId: id,
    citedBy: perRoute.filter((rt) => rt.hfId === id).map((rt) => rt.slug),
    weightsFor: asWeights.map((rt) => rt.slug),
    configOnlyFor: asConfig.map((rt) => rt.slug),
  };
  // Gating only blocks routes that fetch weights from the Hub. WebLLM routes take quantised MLC
  // builds from the MLC CDN and merely cite the upstream card.
  const hubServed = asWeights.some((rt) =>
    rt.loadedVia !== "mlc-cdn" && rt.loadedVia !== "webllm" && rt.loadedVia !== "mediapipe"
  );

  if (r.missing) f.dead = "checkpoint 404";
  if (r.unauthorized) f.unauthorized = "HTTP 401 — private or non-HF identifier";
  if (r.error) f.error = r.error;
  if (r.disabled) f.disabled = "repo disabled upstream";
  if (r.private) f.private = "repo now private";
  if (r.gated && r.gated !== false) {
    f[hubServed ? "gated" : "gatedInformational"] = hubServed
      ? `gated=${r.gated} — weights are fetched from the Hub by ${
        asWeights.map((x) => x.slug).join(", ")
      }`
      : `gated=${r.gated} — cited only; weights arrive via a CDN build`;
  }

  if (hubServed && Array.isArray(r.onnxFiles)) {
    if (r.onnxFiles.length === 0) {
      f.noOnnx = "no .onnx files in the repo";
    } else {
      for (const rt of asWeights) {
        const paired = rt.callSitePairs?.[id];
        const dtypes = paired
          ? [paired]
          : (rt.workerDtypes.length ? rt.workerDtypes : [rt.catalogueDtype]);
        for (const d of dtypes) {
          const pats = DTYPE_TOKENS[d];
          if (!pats) continue;
          // If the route names a file (model_file_name or an explicit .onnx), its own resolution
          // wins — a filename-convention mismatch is not evidence of breakage.
          const named = rt.declaredModelFile || rt.explicitOnnxFiles.length > 0;
          if (named) continue;
          if (!r.onnxFiles.some((file) => pats.some((p) => p.test(file)))) {
            f.dtypeFileAbsent = {
              slug: rt.slug,
              dtype: d,
              onnxFiles: r.onnxFiles.slice(0, 14),
            };
          }
        }
      }
    }
  }

  for (const rt of perRoute.filter((x) => x.hfId === id)) {
    if (r.task && rt.task && r.task !== rt.task) {
      // `task` records the transformers.js pipeline the demo drives; the card records the author's
      // upstream tag. A recorded equivalence is knowledge, not a finding: it is kept on the route (so
      // reports/model-currency.json still shows the two vocabularies and why they agree) and omitted
      // from the findings list, which then contains only what a human still has to look at. An
      // unrecorded mismatch is reported exactly as before — the map is a whitelist, not a blanket
      // excuse (web-ai-showcase-4hv).
      const pair = classifyTaskPair(rt.task, r.task);
      if (pair.equivalent) {
        rt.vocabularyEquivalent = {
          recorded: rt.task,
          upstream: r.task,
          rationale: pair.rationale,
        };
      } else {
        f.taskDrift = {
          recorded: rt.task,
          upstream: r.task,
          reading: "review — confirm the page's claim matches how it is used",
        };
      }
    }
  }
  const drift = prev[id]?.sha && r.sha && prev[id].sha !== r.sha
    ? {
      from: String(prev[id].sha).slice(0, 10),
      to: String(r.sha).slice(0, 10),
      fromDate: prev[id].lastModified,
      toDate: r.lastModified,
    }
    : null;
  if (drift) f.upstreamMoved = drift;

  const attempted = !r.deferred;
  const ok = attempted && !r.missing && !r.unauthorized && !r.rateLimited && !r.error;
  const verdict = ok
    ? "verified"
    : r.deferred
    ? null
    : r.missing
    ? "missing"
    : r.unauthorized
    ? "unauthorized"
    : r.rateLimited
    ? "rate-limited"
    : "error";
  // Only an attempt is evidence. A deferred repo keeps whatever was known before.
  if (verdict && (ok || !prev[id]?.verifiedAt)) {
    records[id] = {
      outcome: verdict,
      errorMessage: ok ? null : (r.error ?? null),
      sha: r.sha ?? null,
      lastModified: r.lastModified ?? null,
      task: r.task ?? null,
      library: r.library ?? null,
      gated: r.gated ?? null,
      onnxCount: Array.isArray(r.onnxFiles) ? r.onnxFiles.length : null,
      onnxFiles: Array.isArray(r.onnxFiles) ? r.onnxFiles.slice(0, 30) : null,
      checked: new Date().toISOString().slice(0, 10),
      verifiedAt: ok ? new Date().toISOString() : (prev[id]?.verifiedAt ?? null),
      checkedAt: new Date().toISOString(),
    };
  }

  if (r.rateLimited) {
    f.rateLimited = "could not verify this run — HuggingFace rate limit (HTTP 429) after retries";
  }
  if (r.deferred) {
    f.deferred = `not re-checked this run (request budget ${MAX_LIVE}) — cached evidence ${
      prev[id]?.checkedAt ?? "absent"
    }`;
  }
  const kinds = [
    "dead",
    "unauthorized",
    "error",
    "disabled",
    "private",
    "gated",
    "noOnnx",
    "dtypeFileAbsent",
    "taskDrift",
    "rateLimited",
    "deferred",
  ]
    .filter((k) => f[k] !== undefined);
  if (drift) kinds.push("upstreamMoved");
  if (kinds.length === 0 && f.gatedInformational) kinds.push("gatedInformational");
  if (kinds.length) findings.push({ ...f, kinds });
}

// catalogue accuracy — cited id vs requested weights, and declared dtype vs loaded dtype
const accuracy = [], dtypeAudit = [];
const baseName = (s) =>
  s.split("/").pop().replace(/-ONNX$/i, "").replace(/-MLC$/, "").toLowerCase();
for (const rt of perRoute) {
  const targetId = rt.loadedId || rt.hfId;
  const isMatch = rt.weightIds.length === 0 ||
    rt.weightIds.includes(targetId) ||
    (rt.loadedId && rt.weightIds.includes(rt.hfId)) ||
    rt.loadedVia === "mediapipe" ||
    rt.loadedVia === "mlc-cdn" ||
    rt.loadedVia === "webllm";
  if (!isMatch) {
    accuracy.push({
      slug: rt.slug,
      runtime: rt.runtime,
      cited: rt.hfId,
      loadedId: rt.loadedId,
      requested: rt.weightIds,
      relation: rt.weightIds.some((id) =>
          baseName(id).startsWith(baseName(rt.hfId)) || baseName(rt.hfId).startsWith(baseName(id))
        )
        ? "same model — an ONNX/MLC build of the cited checkpoint"
        : "no name relation — verify the page's provenance claim",
    });
  }
  const catTokens = String(rt.catalogueDtype ?? "").split(/[^a-zA-Z0-9_]+/).filter(Boolean);
  const code = rt.workerDtypes;
  if (code.length && catTokens.length && !code.every((c) => catTokens.includes(c))) {
    dtypeAudit.push({
      slug: rt.slug,
      catalogue: rt.catalogueDtype,
      worker: code.join("+"),
      catalogueCoversWorker: false,
      kind: catTokens.length === 1 && code.length === 1
        ? "mismatch"
        : "catalogue-names-one-of-several",
    });
  }
}

const tjs = await transformerPins();
const rateLimited = findings.filter((f) => f.kinds.includes("rateLimited")).length;
const deferredCount = findings.filter((f) => f.kinds.includes("deferred")).length;
const tally = {};
for (const f of findings) for (const k of f.kinds) tally[k] = (tally[k] || 0) + 1;

// Recorded vocabulary equivalences: knowledge that the audit would otherwise have to re-report on
// every sweep. Grouped so the report shows the pairs and the reason, not just their absence.
const equivalenceGroups = new Map();
for (const rt of perRoute) {
  const eq = rt.vocabularyEquivalent;
  if (!eq) continue;
  const key = `${eq.recorded} -> ${eq.upstream}`;
  const entry = equivalenceGroups.get(key) || { count: 0, rationale: eq.rationale };
  entry.count += 1;
  equivalenceGroups.set(key, entry);
}
const equivalences = [...equivalenceGroups.entries()].sort((a, b) =>
  b[1].count - a[1].count || a[0].localeCompare(b[0])
);
const report = {
  generated: new Date().toISOString(),
  scope: {
    cataloguedNonPending: cat.models.length -
      cat.models.filter((m) => m.status === "pending").length,
    builtRoutes: built.length,
    scannedRoutes: targets.length,
    uniqueReposSwept: sweep.length,
    citedIds: cited.size,
    weightIds: weightIds.size,
    configOnlyIds: configIds.size,
    rateLimited,
    deferred: deferredCount,
    recheckedThisRun: liveFetches,
    reusedFromCache: cachedCount,
    verified: sweep.length - rateLimited - deferredCount,
    note:
      "Refining lower bound over the currently catalogued built routes — not a completeness claim.",
  },
  transformers: tjs,
  tally,
  findings,
  dtypeAudit,
  catalogueAccuracy: accuracy,
  routes: perRoute,
  mlcRoutes,
};
await mkdir(ROOT + "reports", { recursive: true });
await writeFile(REPORT_JSON, JSON.stringify(report, null, 2) + "\n");
await writeFile(
  SNAPSHOT,
  JSON.stringify(
    {
      generated: report.generated,
      routes: { ...(snapshot.routes || {}), ...routeRecords },
      models: { ...prev, ...records },
    },
    null,
    2,
  ) + "\n",
);

const md = [
  `# Model + runtime currency audit`,
  ``,
  `Generated ${report.generated} by \`scripts/audit-model-currency.mjs\`.`,
  ``,
  `- Built routes: **${built.length}** · scanned: **${targets.length}** · unique HF repos health-checked: **${sweep.length}**`,
  `  (${cited.size} cited · ${weightIds.size} weight-serving · ${configIds.size} config/tokenizer-only)`,
  `- transformers.js shared pin: **${tjs.shared}** · latest published: **${tjs.latest}**${
    tjs.recentStable.length ? ` (recent stable: ${tjs.recentStable.join(", ")})` : ""
  }`,
  `- Local version overrides: ${
    Object.entries(tjs.localOverrides).filter(([v]) => v !== tjs.shared).map(([v, s]) =>
      `\`${v}\` on ${s.length} route(s)`
    ).join(", ") || "none"
  }`,
  `- Checkpoint/pin findings: **${findings.length}**`,
  ...(equivalences.length
    ? [
      ``,
      `### Recorded vocabulary equivalences (${
        equivalences.reduce((n, [, e]) => n + e.count, 0)
      } routes)`,
      ``,
      "The transformers.js task a demo drives and the Hub `pipeline_tag` on the card are two" +
      " vocabularies for the same work. These pairs are recorded in `scripts/model-task-vocabulary.mjs`" +
      " as equivalent, with the reason, so they are reported as neither drift nor a defect — and an" +
      " unrecorded mismatch is still reported:",
      ``,
      ...equivalences.map(([pair, e]) => `- \`${pair}\` × ${e.count} — ${e.rationale}`),
    ]
    : []),
  ``,
  `## Findings by kind`,
  ``,
  ...(Object.keys(tally).length
    ? Object.entries(tally).sort((a, b) => b[1] - a[1]).map(([k, n]) => `- ${k}: ${n}`)
    : ["- none"]),
  ``,
  `## Findings`,
  ``,
  ...(findings.length
    ? findings.map((f) =>
      `- \`${f.hfId}\` — ${JSON.stringify(Object.fromEntries(f.kinds.map((k) => [k, f[k]])))}`
    )
    : ["- none"]),
  ``,
  `## Catalogue dtype vs worker dtype (${dtypeAudit.length})`,
  ``,
  ...(dtypeAudit.length
    ? dtypeAudit.map((d) =>
      `- \`${d.slug}\`: catalogue \`${d.catalogue}\`, worker \`${d.worker}\` — ${d.kind}`
    )
    : ["- none"]),
  ``,
  `## Catalogue accuracy — cited \`hfId\` vs requested weights (${accuracy.length})`,
  ``,
  ...(accuracy.length
    ? accuracy.map((a) =>
      `- \`${a.slug}\` (${a.runtime}): cites \`${a.cited}\`, requests ${
        a.requested.map((x) => `\`${x}\``).join(", ")
      } — ${a.relation}`
    )
    : ["- every scanned route requests the repo it cites"]),
  ``,
  `## Routes (runtime · requested weights)`,
  ``,
  ...perRoute.map((rt) =>
    `- \`${rt.slug}\` (${rt.runtime}, ${rt.catalogueDtype ?? "no dtype"}): ` +
    (rt.weightIds.length
      ? rt.weightIds.map((x) => `\`${x}\``).join(", ")
      : "no explicit model id in source") +
    (rt.configIds.length ? ` · config-only: ${rt.configIds.map((x) => `\`${x}\``).join(", ")}` : "")
  ),
  ``,
].join("\n");
await writeFile(REPORT_MD, md);
console.error(`Wrote ${REPORT_JSON}, ${REPORT_MD}, refreshed ${SNAPSHOT}`);
console.log(JSON.stringify(
  {
    scope: report.scope,
    tally,
    dtypeAudit: dtypeAudit.length,
    catalogueAccuracy: accuracy.length,
    transformers: {
      shared: tjs.shared,
      latest: tjs.latest,
      overrides: Object.keys(tjs.localOverrides),
    },
  },
  null,
  2,
));
