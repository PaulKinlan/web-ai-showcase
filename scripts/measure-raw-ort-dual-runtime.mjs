#!/usr/bin/env node
// Raw-ORT dual-runtime measurement (bead web-ai-showcase-62m.3 / ij4 adapter).
//
// Routes that load transformers.js AND a raw onnxruntime-web build in one page:
//   models/bert-base-turkish-cased-ner/multi-model/   transformers.js language detection + raw-ORT NER
//   models/model2vec-static-embeddings/multi-model/   raw-ORT potion + transformers.js reranker
//   models/yolo-world/multi-model/                    raw-ORT YOLO-World + transformers.js ViT
//   models/embeddinggemma-2/basics/                   transformers.js embeddinggemma-2 (q4 webgpu)
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
// Scoping env vars (B4):
//   ROUTE_ONLY=embeddinggemma-2 node scripts/measure-raw-ort-dual-runtime.mjs
//   VIEWPORT_ONLY=desktop node scripts/measure-raw-ort-dual-runtime.mjs
//
// Read-only with respect to the repository; writes nothing except the temp profile.

import { createHash } from "node:crypto";
import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

// route, label, expected engine families
export const ROUTES = [
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
  {
    slug: "embeddinggemma-2",
    route: "models/embeddinggemma-2/basics/",
    control: "models/embeddinggemma-2/",
    expect: "transformers.js embeddinggemma-2 (q4 webgpu)",
    requiresWebGPU: true,
  },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// B2: Unique isolated profile per run.
// Rationale: A unique isolated profile per run ensures a cold cache visit every time.
// A warm profile can satisfy runtime requests from HTTP cache and make the captured set
// UNDERSTATE a cold visit. Removing it on both normal and timeout/failure paths (finally)
// ensures no disk leaks or cross-run cache pollution remain behind.
export function createUniqueProfilePath() {
  return join(
    tmpdir(),
    `webai-dual-runtime-profile-${process.pid}-${Date.now()}-${
      Math.random().toString(36).slice(2, 7)
    }`,
  );
}

// B4: Scoping env-var filter applied to BOTH loops.
// Authoritative route/viewport for runtime URL capture:
// For models/embeddinggemma-2/basics/, "models/embeddinggemma-2/basics/" at viewport "desktop"
// (cold profile) is authoritative for the captured URL set. The basics route initializes the full
// text encoder under WebGPU on the first visit, capturing all transitive jsDelivr runtime and
// WASM/WebGPU binary assets, while desktop provides the unthrottled canonical viewport.
export function getFilteredRoutes(routes = ROUTES, env = process.env) {
  const filter = (env.ROUTE_ONLY || env.ROUTE_FILTER || "").trim();
  if (!filter) return [...routes];
  return routes.filter((r) =>
    r.slug === filter ||
    r.slug.includes(filter) ||
    r.route.includes(filter) ||
    (r.control && r.control.includes(filter))
  );
}

export function getFilteredViewports(env = process.env) {
  const filter = (env.VIEWPORT_ONLY || env.VIEWPORT_FILTER || "").trim().toLowerCase();
  const allViewports = ["desktop", "mobile"];
  if (!filter) return allViewports;
  const filtered = allViewports.filter((vp) => vp === filter || vp.includes(filter));
  return filtered.length > 0 ? filtered : allViewports;
}

export function classify(url) {
  if (/@huggingface\/transformers@/.test(url)) {
    return /\.wasm(\?|$)/.test(url) ? "transformers-ort-wasm" : "transformers-lib-js";
  }
  if (/onnxruntime-web@/.test(url)) return /\.wasm(\?|$)/.test(url) ? "raw-ort-wasm" : "raw-ort-js";
  if (/huggingface\.co\/|hf\.co\/|xethub|cdn-lfs/.test(url)) return "model-weights";
  if (/web-ai-showcase|127\.0\.0\.1/.test(url)) return "app-shell";
  return "other";
}

// B5: Target auto-attach and worker network interception ordering.
// Worker sessions (transformers.js and the raw-ORT worker both fetch their runtimes and weights from
// inside a Web Worker) need their own Network.enable, or their requests are invisible.
// To guarantee no early requests are missed, auto-attach must be configured with
// waitForDebuggerOnStart: true, and Network.enable must be issued and resolved BEFORE
// resuming worker execution with Runtime.runIfWaitingForDebugger.
export async function setupTargetAutoAttach(cdp, sessionId = null) {
  return cdp.send(
    "Target.setAutoAttach",
    {
      autoAttach: true,
      flatten: true,
      waitForDebuggerOnStart: true,
    },
    sessionId,
  );
}

export async function handleTargetAttached(
  cdp,
  params,
  { tracked = new Set(), onNetworkEnabled = null } = {},
) {
  const { sessionId, targetInfo, waitingForDebugger } = params || {};
  if (!sessionId) return;
  const type = targetInfo?.type;
  const isWorker = type === "worker" || type === "service_worker";

  if (isWorker || waitingForDebugger) {
    if (!tracked.has(sessionId)) {
      tracked.add(sessionId);
      await cdp.send("Network.enable", {}, sessionId).catch((err) => {
        tracked.delete(sessionId);
        throw err;
      });
      if (onNetworkEnabled) onNetworkEnabled(sessionId, targetInfo);
    }
    if (waitingForDebugger) {
      await cdp.send("Runtime.runIfWaitingForDebugger", {}, sessionId);
    }
  }
}

// B0: WebGPU capability checking.
// Fail closed with a clear, non-zero-exit reason if WebGPU is unavailable so that
// a capability failure is never again reported as "this route fetched nothing".
export async function checkWebGPUAvailable(evaluateFn) {
  const result = await evaluateFn(`
    (async () => {
      if (typeof navigator === "undefined" || !("gpu" in navigator)) {
        return { ok: false, reason: "navigator.gpu missing" };
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { ok: false, reason: "requestAdapter returned null (no GPU adapter)" };
        return { ok: true };
      } catch (err) {
        return { ok: false, reason: String(err?.message || err) };
      }
    })()
  `);
  return result || { ok: false, reason: "evaluation returned empty" };
}

// B3: Decoded-byte SHA-256 calculation for exact observed successful cdn.jsdelivr.net URLs.
// The manifest policy hashes decoded bytes and the difference is material (e.g. 4.3.1 entrypoint
// is 167590 bytes wire vs 586230 decoded).
export function computeDecodedSha256(bufferOrUint8) {
  const buf = Buffer.isBuffer(bufferOrUint8) ? bufferOrUint8 : Buffer.from(bufferOrUint8);
  return {
    sha256: createHash("sha256").update(buf).digest("hex"),
    decodedBytes: buf.byteLength,
  };
}

export async function fetchDecodedSha256(url, fetchFn = globalThis.fetch) {
  const res = await fetchFn(url, { signal: AbortSignal.timeout(60_000) });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url} for decoded SHA-256 calculation: HTTP ${res.status}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  const buf = Buffer.from(arrayBuffer);
  return {
    url,
    ...computeDecodedSha256(buf),
    status: res.status,
  };
}

export async function resolveJsDelivrIntegrity(
  observedUrls,
  { fetchFn = globalThis.fetch, requestStatusMap = new Map() } = {},
) {
  const entries = {};
  const urls = [...new Set(observedUrls)].filter((u) => u && u.includes("cdn.jsdelivr.net"));
  for (const url of urls) {
    const statusMeta = requestStatusMap.get(url);
    if (statusMeta && statusMeta.status && (statusMeta.status < 200 || statusMeta.status >= 400)) {
      continue;
    }
    try {
      const result = await fetchDecodedSha256(url, fetchFn);
      entries[url] = {
        url,
        sha256: result.sha256,
        decodedBytes: result.decodedBytes,
        status: result.status,
      };
    } catch (err) {
      entries[url] = {
        url,
        error: err.message,
      };
    }
  }
  return entries;
}

// Critical behavioural rule: an unexercised route or zero-request result must be treated
// as FAILURE/INCONCLUSIVE, never as a pass.
export function verifyRunResults(rows) {
  const failures = [];
  if (!rows || rows.length === 0) {
    failures.push("No routes were executed");
  }
  for (const row of rows || []) {
    if (!row.ready) {
      failures.push(`${row.route} [${row.viewport}]: loader never reached 'ready' state`);
    }
    if (!row.totalBytes || row.totalBytes === 0) {
      failures.push(
        `${row.route} [${row.viewport}]: zero network requests recorded (unexercised route / inconclusive)`,
      );
    }
  }
  return {
    ok: failures.length === 0,
    failures,
  };
}

export const evaluate = (cdp, sid, expression, timeout = 120000) =>
  cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression})}catch(error){return {__error:String(error?.stack||error)}}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    timeout,
  ).then((r) => r.result?.value);

export async function loaderStates(cdp, sid) {
  return evaluate(
    cdp,
    sid,
    `[...document.querySelectorAll('.model-loader')].map((n) => n.dataset.state)`,
  );
}

// Drive every loader to ready: click Download/Retry/Continue, and handle the honest check-timeout
// state (its local model check exceeded its deadline) instead of waiting forever.
// B1 retracted: existing regex matches "Download model (~175 MB)".
// B0 fail-closed: abort immediately if loader enters 'unsupported' state.
export async function loadAllEngines(cdp, sid, label, budgetMs = 30 * 60_000) {
  const started = Date.now();
  let lastLog = "";
  while (Date.now() - started < budgetMs) {
    const states = (await loaderStates(cdp, sid)) ?? [];
    const encoded = JSON.stringify(states);
    if (encoded !== lastLog) {
      console.log(`    [${label}] ${encoded} (${Math.round((Date.now() - started) / 1000)}s)`);
      lastLog = encoded;
    }
    if (states.includes("unsupported")) {
      throw new Error(
        `FAIL CLOSED: Model loader reached 'unsupported' state on ${label}. ` +
          `WebGPU or hardware requirement is unmet in this browser environment.`,
      );
    }
    if (states.length > 0 && states.every((s) => s === "ready")) return true;
    await evaluate(
      cdp,
      sid,
      `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Download|Retry|Re-download|Continue|Retry local check/i.test(b.textContent)&&!b.disabled){b.click();n++}}return n})()`,
    ).catch(() => null);
    await sleep(2000);
  }
  return false;
}

export async function summarize(
  routeEvents,
  { fetchFn = globalThis.fetch, requestStatusMap = new Map() } = {},
) {
  const byId = new Map();
  for (const e of routeEvents) {
    if (e.kind === "request") byId.set(e.id, { url: e.url, bytes: 0, cache: null, status: null });
  }
  for (const e of routeEvents) {
    const row = byId.get(e.id);
    if (!row) continue;
    if (e.kind === "finished") row.bytes = e.encodedDataLength;
    if (e.kind === "response") {
      const meta = requestStatusMap.get(e.id);
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

  // B3: obtain and record DECODED-byte SHA-256 for exact observed successful cdn.jsdelivr.net URLs
  const successfulJsDelivrUrls = [];
  for (const row of byId.values()) {
    if (row.url && row.url.includes("cdn.jsdelivr.net")) {
      if (!row.status || (row.status >= 200 && row.status < 400)) {
        successfulJsDelivrUrls.push(row.url);
      }
    }
  }
  const jsDelivrEntries = await resolveJsDelivrIntegrity(successfulJsDelivrUrls, {
    fetchFn,
    requestStatusMap,
  });

  return { groups, total, consoleErrors, jsDelivrEntries };
}

export async function run(options = {}) {
  const args = options.args || process.argv.slice(2);
  const fresh = options.fresh ?? args.includes("--fresh");
  const jsonOut = options.jsonOut ?? args.includes("--json");
  const env = options.env || process.env;
  const fetchFn = options.fetchFn || globalThis.fetch;

  // B2: unique profile per run with finally removal
  const profile = env.CHROME_PROFILE_DIR || createUniqueProfilePath();
  if (fresh) {
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {}
  }
  mkdirSync(profile, { recursive: true });

  const filteredRoutes = getFilteredRoutes(ROUTES, env);
  const filteredViewports = getFilteredViewports(env);

  console.log(
    `Routes to measure (${filteredRoutes.length}): ${filteredRoutes.map((r) => r.slug).join(", ")}`,
  );
  console.log(`Viewports to test (${filteredViewports.length}): ${filteredViewports.join(", ")}`);

  let server = null;
  let chrome = null;

  try {
    const srv = await startServer();
    server = srv.server;
    const port = srv.port;
    const base = `http://127.0.0.1:${port}/web-ai-showcase/`;

    // B0: Launch Chrome with webgpu: true
    chrome = await launchChrome({
      userDataDir: profile,
      resetProfile: true,
      removeProfileOnKill: true,
      webgpu: true,
    });
    const cdp = new CDP(chrome.ws);

    const tracked = new Set();
    let events = [];
    const requestUrl = new Map();
    const requestStatus = new Map();

    cdp.on((msg) => {
      if (msg.method === "Target.attachedToTarget") {
        void handleTargetAttached(cdp, msg.params, { tracked }).catch((err) => {
          console.error(
            `[attach-error] Failed to handle attached target ${msg.params?.sessionId}:`,
            err,
          );
        });
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
          text: String(msg.params.args?.map((a) => a.value ?? a.description ?? "").join(" ")).slice(
            0,
            200,
          ),
        });
      }
    });

    await cdp.send("Target.setDiscoverTargets", { discover: true });
    // B5: Attach with waitForDebuggerOnStart: true
    await setupTargetAutoAttach(cdp);

    async function measure(route, vpName, routeConfig = null) {
      events = [];
      requestUrl.clear();
      requestStatus.clear();
      const page = await openPage(cdp, base + route);
      try {
        await cdp.send("Network.enable", {}, page.sessionId);
        await setupTargetAutoAttach(cdp, page.sessionId).catch(() => {});
        await sleep(200);
        await setViewport(cdp, page.sessionId, vpName === "desktop" ? DESKTOP : MOBILE);
        await sleep(800);

        // B0: Fail closed if route requires WebGPU and WebGPU is unavailable
        if (routeConfig?.requiresWebGPU) {
          const gpuCheck = await checkWebGPUAvailable((expr) =>
            evaluate(cdp, page.sessionId, expr)
          );
          if (!gpuCheck.ok) {
            throw new Error(
              `FAIL CLOSED: Route ${route} requires WebGPU, but WebGPU is unavailable in the browser: ${gpuCheck.reason}. ` +
                `A capability failure must never be reported as an empty network capture.`,
            );
          }
        }

        const ready = await loadAllEngines(
          cdp,
          page.sessionId,
          `${route.split("/").slice(-2, -1)[0]} ${vpName}`,
        );
        if (!ready) {
          const states = (await loaderStates(cdp, page.sessionId)) ?? [];
          if (states.includes("unsupported")) {
            throw new Error(
              `FAIL CLOSED: Model loader reached 'unsupported' state on ${route} [${vpName}]. ` +
                `WebGPU or hardware requirement is unmet in this browser environment.`,
            );
          }
        }

        const snap = await summarize(events, { fetchFn, requestStatusMap: requestStatus });
        return {
          route,
          viewport: vpName,
          ready,
          totalBytes: snap.total,
          groups: snap.groups,
          jsDelivrEntries: snap.jsDelivrEntries,
          consoleErrors: snap.consoleErrors.slice(0, 3),
          pageErrors: page.errors.slice(0, 3),
        };
      } finally {
        await closePage(cdp, page.targetId);
      }
    }

    const rows = [];
    // Loop 1: dual-runtime measurements
    for (const r of filteredRoutes) {
      for (const vpName of filteredViewports) {
        console.log(`\n== ${r.slug} [${vpName}] ${r.expect}`);
        const row = await measure(r.route, vpName, r);
        rows.push({ ...row, slug: r.slug, kind: "dual-runtime", expects: r.expect });
      }
    }

    // Loop 2: Control measurements (single-runtime overview of the same family)
    if (filteredViewports.includes("desktop")) {
      for (const r of filteredRoutes) {
        if (!r.control) continue;
        console.log(`\n== [control] ${r.slug} [desktop] ${r.control}`);
        const row = await measure(r.control, "desktop", r);
        rows.push({ ...row, slug: r.slug, kind: "control-single-runtime", expects: r.expect });
      }
    }

    // Verify run results: unexercised routes or zero-request results fail closed
    const verification = verifyRunResults(rows);
    if (!verification.ok) {
      console.error("\nRUN VERIFICATION FAILED:");
      for (const f of verification.failures) {
        console.error(`  - ${f}`);
      }
      process.exitCode = 1;
      throw new Error(`Measurement run failed verification:\n${verification.failures.join("\n")}`);
    }

    if (jsonOut) {
      console.log(
        JSON.stringify({ profile, fresh, measuredAt: new Date().toISOString(), rows }, null, 2),
      );
    } else {
      const fmt = (b) => (b / 1024 / 1024).toFixed(2) + " MB";
      for (const row of rows) {
        console.log(
          `\n### ${row.slug} — ${row.kind} — ${row.route} [${row.viewport}] ready=${row.ready}`,
        );
        console.log(`expected: ${row.expects}`);
        for (const [cat, g] of Object.entries(row.groups).sort((a, b) => b[1].bytes - a[1].bytes)) {
          console.log(
            `  ${cat.padEnd(22)} ${String(g.files).padStart(3)} file(s)  ${
              fmt(g.bytes).padStart(10)
            }  (cache: ${g.cached})`,
          );
        }
        console.log(`  TOTAL ${fmt(row.totalBytes)}`);
        if (row.jsDelivrEntries && Object.keys(row.jsDelivrEntries).length > 0) {
          console.log(`  cdn.jsdelivr.net integrity pins (decoded-byte sha256):`);
          for (const [url, entry] of Object.entries(row.jsDelivrEntries)) {
            if (entry.error) {
              console.log(`    ${url} — ERROR: ${entry.error}`);
            } else {
              console.log(`    ${url}`);
              console.log(`      decoded: ${entry.decodedBytes} bytes | sha256: ${entry.sha256}`);
            }
          }
        }
        if (row.consoleErrors.length) {
          console.log(`  console errors: ${JSON.stringify(row.consoleErrors)}`);
        }
        if (row.pageErrors.length) console.log(`  page errors: ${JSON.stringify(row.pageErrors)}`);
      }
      console.log(`\nprofile: ${profile} (fresh=${fresh})`);
    }

    return rows;
  } finally {
    if (chrome) {
      await chrome.kill({ removeProfile: true }).catch(() => {});
    }
    if (server) {
      try {
        server.close();
      } catch {}
    }
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {}
  }
}

export const isMain = Boolean(
  process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url),
);

if (isMain) {
  run().catch((err) => {
    console.error("\nFATAL:", err.message || err);
    process.exit(1);
  });
}
