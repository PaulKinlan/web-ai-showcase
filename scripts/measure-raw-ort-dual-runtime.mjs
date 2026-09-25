#!/usr/bin/env node
// Raw-ORT dual-runtime measurement (bead web-ai-showcase-62m.3).
//
// Three routes load transformers.js AND a raw onnxruntime-web build in one page:
//   models/bert-base-turkish-cased-ner/multi-model/   transformers.js language detection + raw-ORT NER
//   models/model2vec-static-embeddings/multi-model/   raw-ORT potion + transformers.js reranker
//   models/yolo-world/multi-model/                    raw-ORT YOLO-World + transformers.js ViT
//
// Question: does the visitor actually pay for TWO ORT runtimes (both JS and both WASM), or does one
// path stay lazy? Method: open each page in a real headless Chrome at desktop and mobile, drive every
// model loader to ready (session creation is what instantiates a runtime), and attribute every network
// request to transformers.js, the raw CDN ORT build, or model weights. A single-runtime route is
// measured as the control.
//
// Usage:
//   node scripts/measure-raw-ort-dual-runtime.mjs                  # persistent profile (models cached)
//   node scripts/measure-raw-ort-dual-runtime.mjs --fresh          # wipe the profile first: cold bytes
//   node scripts/measure-raw-ort-dual-runtime.mjs --json           # machine-readable rows
//
// Read-only with respect to the repository; writes nothing except (with --fresh) the temp profile.

import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, setViewport, startServer } from "./browser.mjs";

const ARGS = process.argv.slice(2);
const FRESH = ARGS.includes("--fresh");
const JSON_OUT = ARGS.includes("--json");
const PROFILE = join(tmpdir(), "webai-dual-runtime-profile");

// route, label, expected engine families
const ROUTES = [
  {
    slug: "bert-base-turkish-cased-ner",
    route: "models/bert-base-turkish-cased-ner/multi-model/",
    control: "models/bert-base-turkish-cased-ner/",
    expect: "transformers.js language detection (q8) + raw ORT Turkish NER (fp32)",
  },
  {
    slug: "model2vec-static-embeddings",
    route: "models/model2vec-static-embeddings/multi-model/",
    control: "models/model2vec-static-embeddings/",
    expect: "raw ORT potion-base-8M + transformers.js jina reranker (q8)",
  },
  {
    slug: "yolo-world",
    route: "models/yolo-world/multi-model/",
    control: "models/yolo-world/",
    expect: "raw ORT YOLO-World + transformers.js ViT (q8)",
  },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function classify(url) {
  if (/@huggingface\/transformers@/.test(url)) {
    return /\.wasm(\?|$)/.test(url) ? "transformers-ort-wasm" : "transformers-lib-js";
  }
  if (/onnxruntime-web@/.test(url)) return /\.wasm(\?|$)/.test(url) ? "raw-ort-wasm" : "raw-ort-js";
  if (/huggingface\.co\/|hf\.co\/|xethub|cdn-lfs/.test(url)) return "model-weights";
  if (/web-ai-showcase|127\.0\.0\.1/.test(url)) return "app-shell";
  return "other";
}

if (FRESH) rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
const chrome = await launchChrome({
  userDataDir: PROFILE,
  resetProfile: false,
  removeProfileOnKill: false,
});
const cdp = new CDP(chrome.ws);

// Worker sessions (transformers.js and the raw-ORT worker both fetch their runtimes and weights from
// inside a Web Worker) need their own Network.enable, or their requests are invisible. Auto-attach and
// enable the network domain per worker session, exactly as the acceptance validators do.
const tracked = new Set();
async function enableNetwork(sessionId) {
  if (!sessionId || tracked.has(sessionId)) return;
  tracked.add(sessionId);
  await cdp.send("Network.enable", {}, sessionId).catch(() => tracked.delete(sessionId));
}

let events = [];
const requestUrl = new Map();
const requestStatus = new Map();
cdp.on((msg) => {
  if (msg.method === "Target.attachedToTarget") {
    const type = msg.params.targetInfo?.type;
    if (type === "worker" || type === "service_worker") void enableNetwork(msg.params.sessionId);
    return;
  }
  if (msg.method === "Network.requestWillBeSent") {
    requestUrl.set(msg.params.requestId, msg.params.request.url);
    events.push({ kind: "request", id: msg.params.requestId, url: msg.params.request.url });
  } else if (msg.method === "Network.loadingFinished") {
    events.push({
      kind: "finished",
      id: msg.params.requestId,
      encodedDataLength: msg.params.encodedDataLength,
    });
  } else if (msg.method === "Network.responseReceived") {
    requestStatus.set(msg.params.requestId, {
      fromDiskCache: Boolean(msg.params.response.fromDiskCache),
      fromServiceWorker: Boolean(msg.params.response.fromServiceWorker),
      status: msg.params.response.status,
    });
    events.push({ kind: "response", id: msg.params.requestId });
  } else if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
    events.push({
      kind: "console-error",
      text: String(msg.params.args?.map((a) => a.value ?? a.description ?? "").join(" ")).slice(0, 200),
    });
  }
});
await cdp.send("Target.setDiscoverTargets", { discover: true });
await cdp.send("Target.setAutoAttach", {
  autoAttach: true,
  flatten: true,
  waitForDebuggerOnStart: false,
});

const evaluate = (sid, expression, timeout = 120000) =>
  cdp.send(
    "Runtime.evaluate",
    {
      expression: `(async()=>{try{return (${expression})}catch(error){return {__error:String(error?.stack||error)}}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    timeout,
  ).then((r) => r.result?.value);

async function loaderStates(sid) {
  return evaluate(
    sid,
    `[...document.querySelectorAll('.model-loader')].map((n) => n.dataset.state)`,
  );
}

// Drive every loader to ready: click Download/Retry/Continue, and handle the honest check-timeout
// state (its local model check exceeded its deadline) instead of waiting forever.
async function loadAllEngines(sid, label, budgetMs = 30 * 60_000) {
  const started = Date.now();
  let lastLog = "";
  while (Date.now() - started < budgetMs) {
    const states = (await loaderStates(sid)) ?? [];
    const encoded = JSON.stringify(states);
    if (encoded !== lastLog) {
      console.log(`    [${label}] ${encoded} (${Math.round((Date.now() - started) / 1000)}s)`);
      lastLog = encoded;
    }
    if (states.length > 0 && states.every((s) => s === "ready")) return true;
    await evaluate(
      sid,
      `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Download|Retry|Re-download|Continue|Retry local check/i.test(b.textContent)&&!b.disabled){b.click();n++}}return n})()`,
    ).catch(() => null);
    await sleep(2000);
  }
  return false;
}

function summarize(routeEvents) {
  const byId = new Map();
  for (const e of routeEvents) {
    if (e.kind === "request") byId.set(e.id, { url: e.url, bytes: 0, cache: null, status: null });
  }
  for (const e of routeEvents) {
    const row = byId.get(e.id);
    if (!row) continue;
    if (e.kind === "finished") row.bytes = e.encodedDataLength;
    if (e.kind === "response") {
      const meta = requestStatus.get(e.id);
      if (meta) {
        row.cache = meta.fromDiskCache ? "disk" : meta.fromServiceWorker ? "sw" : "network";
        row.status = meta.status;
      }
    }
  }
  const groups = {};
  let total = 0;
  for (const row of byId.values()) {
    const cat = classify(row.url);
    groups[cat] ??= { files: 0, bytes: 0, urls: [], cached: 0 };
    groups[cat].files += 1;
    groups[cat].bytes += row.bytes || 0;
    groups[cat].cached += row.cache === "disk" ? 1 : 0;
    groups[cat].urls.push(row.url);
    total += row.bytes || 0;
  }
  const consoleErrors = routeEvents.filter((e) => e.kind === "console-error").map((e) => e.text);
  return { groups, total, consoleErrors };
}

async function measure(route, vpName) {
  events = [];
  requestUrl.clear();
  requestStatus.clear();
  const page = await openPage(cdp, base + route);
  try {
    await enableNetwork(cdp, page.sessionId);
    await cdp.send(
      "Target.setAutoAttach",
      { autoAttach: true, flatten: true, waitForDebuggerOnStart: false },
      page.sessionId,
    ).catch(() => {});
    await sleep(200);
    await setViewport(cdp, page.sessionId, vpName === "desktop" ? DESKTOP : MOBILE);
    await sleep(800);
    const ready = await loadAllEngines(page.sessionId, `${route.split("/").slice(-2, -1)[0]} ${vpName}`);
    const snap = summarize(events);
    return {
      route,
      viewport: vpName,
      ready,
      totalBytes: snap.total,
      groups: snap.groups,
      consoleErrors: snap.consoleErrors.slice(0, 3),
      pageErrors: page.errors.slice(0, 3),
    };
  } finally {
    await closePage(cdp, page.targetId);
  }
}

const rows = [];
for (const r of ROUTES) {
  for (const vpName of ["desktop", "mobile"]) {
    console.log(`\n== ${r.slug} [${vpName}] ${r.expect}`);
    const row = await measure(r.route, vpName);
    rows.push({ ...row, slug: r.slug, kind: "dual-runtime", expects: r.expect });
  }
}
// Control: the single-runtime overview of the same family (models already cached by the pass above).
for (const r of ROUTES) {
  const row = await measure(r.control, "desktop");
  rows.push({ ...row, slug: r.slug, kind: "control-single-runtime", expects: r.expect });
}

await chrome.kill();
server.close();

if (JSON_OUT) {
  console.log(JSON.stringify({ profile: PROFILE, fresh: FRESH, measuredAt: new Date().toISOString(), rows }, null, 2));
} else {
  const fmt = (b) => (b / 1024 / 1024).toFixed(2) + " MB";
  for (const row of rows) {
    console.log(`\n### ${row.slug} — ${row.kind} — ${row.route} [${row.viewport}] ready=${row.ready}`);
    console.log(`expected: ${row.expects}`);
    for (const [cat, g] of Object.entries(row.groups).sort((a, b) => b[1].bytes - a[1].bytes)) {
      console.log(`  ${cat.padEnd(22)} ${String(g.files).padStart(3)} file(s)  ${fmt(g.bytes).padStart(10)}  (cache: ${g.cached})`);
    }
    console.log(`  TOTAL ${fmt(row.totalBytes)}`);
    if (row.consoleErrors.length) console.log(`  console errors: ${JSON.stringify(row.consoleErrors)}`);
    if (row.pageErrors.length) console.log(`  page errors: ${JSON.stringify(row.pageErrors)}`);
  }
  console.log(`\nprofile: ${PROFILE} (fresh=${FRESH})`);
}
