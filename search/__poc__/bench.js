// Main-thread harness for the POC. Spawns the search worker via the repo's typed WorkerClient,
// drives build + query + probe methods, and MEASURES main-thread long tasks the whole time (the
// point of the off-main-thread architecture is that this number stays ~0). Publishes results to
// window.__RESULTS__ for the headless driver (drive-bench.mjs) to extract.
//
// modern-web-guidance retained: identify-inp-causes (Long Animation Frames / longtask observer used
// here as the lab proxy for INP main-thread cost), break-up-long-tasks (all heavy work is in worker).

import { WorkerClient } from "../../lib/worker-protocol.js";

const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const setStatus = (t, cls) => {
  statusEl.textContent = t;
  statusEl.className = cls || "";
};

// ── main-thread long-task observer (lab proxy for INP cost) ───────────────────────────────────────
const longTasks = [];
try {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) longTasks.push(+e.duration.toFixed(1));
  }).observe({ type: "longtask", buffered: true });
} catch { /* longtask unsupported → reported as null */ }

// Round-trip latency: main → worker → main, incl. postMessage + structured-clone of the result.
async function timeRoundTrip(fn) {
  const t = performance.now();
  const r = await fn();
  return { ms: +(performance.now() - t).toFixed(2), r };
}

async function run() {
  const results = {
    ua: navigator.userAgent,
    dpr: devicePixelRatio,
    vw: innerWidth,
    vh: innerHeight,
  };
  const client = new WorkerClient({
    url: new URL("./search-worker.js", import.meta.url),
    name: "search-poc",
    onState: (s) => setStatus(`worker: ${s}`),
  });
  await client.ready;

  setStatus("building index…");
  const build = await timeRoundTrip(() => client.request("build", {}, { onProgress: () => {} }));
  results.build = build.r.result;
  results.buildRoundTripMs = build.ms;

  setStatus("benching queries…");
  results.bench = (await client.request("bench", { iters: 60 })).result;

  // Individual round-trips (include the postMessage hop the bench numbers exclude).
  results.lexicalRT =
    (await timeRoundTrip(() =>
      client.request("lexical", { q: "speech to text webgpu", k: 20 }, { channel: "q" })
    )).ms;
  results.semanticF32RT =
    (await timeRoundTrip(() =>
      client.request("semantic", { seed: 7, k: 20, quant: "f32" }, { channel: "q" })
    )).ms;
  results.semanticI8RT =
    (await timeRoundTrip(() =>
      client.request("semantic", { seed: 7, k: 20, quant: "i8" }, { channel: "q" })
    )).ms;
  results.sampleLexical =
    (await client.request("lexical", { q: "background removal", k: 5 })).result;

  setStatus("probing OPFS…");
  results.opfs = (await client.request("probeOPFS", {})).result;

  setStatus("probing sqlite-wasm + FTS5 (network)…");
  try {
    results.sqlite = (await client.request("probeSqlite", {})).result;
  } catch (e) {
    results.sqlite = { ok: false, reason: String(e?.message || e) };
  }

  results.mainThreadLongTasks = { count: longTasks.length, durationsMs: longTasks };
  results.heapUsed = performance.memory?.usedJSHeapSize ?? null;

  window.__RESULTS__ = results;
  outEl.textContent = JSON.stringify(results, null, 2);
  setStatus(
    `done · ${results.build.docCount} docs · ${longTasks.length} main-thread long task(s)`,
    longTasks.length === 0 ? "ok" : "bad",
  );
  client.terminate();
}

run().catch((e) => {
  window.__RESULTS__ = { error: String(e?.stack || e) };
  outEl.textContent = String(e?.stack || e);
  setStatus("error: " + e.message, "bad");
});
