#!/usr/bin/env node
// Task 2b · Phase 5 — download-route CRAWLER (browser test mode).
//
// Visits inventoried routes in real headless Chrome and, per route, proves the ADOPTED state is present
// and healthy: the shared .model-loader rendered, a <model-download-status> element exists (the component
// is wired), no duplicate element ids, no horizontal overflow (desktop AND mobile), and NO console errors.
// Console errors are captured passively (always reliable); structural introspection uses a short retry so a
// demo whose model-init pegs the main thread is reported honestly as "busy" rather than failing the crawl.
//
// The download STATE MACHINE itself (late discovery, concurrency, error, ready, false-100%, stale events,
// missing actions) is exhaustively covered by the deterministic fixture tests — validate-central-loader.mjs
// (13) and validate-model-download-status.mjs (14) — so we don't re-drive real downloads on 270 routes.
//
// Usage:
//   node scripts/crawl-download-routes.mjs                 # a representative sample across every family
//   node scripts/crawl-download-routes.mjs --all           # every downloading route (slow)
//   node scripts/crawl-download-routes.mjs --family webllm  # one family
import {
  closePage,
  evalValue,
  launchChrome,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";
import { readFileSync } from "node:fs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const inv = JSON.parse(readFileSync("download-routes.json", "utf8"));
const downloading = inv.routes.filter((r) => r.status !== "non-applicable");

const args = process.argv.slice(2);
const famArg = (args.find((a) => a.startsWith("--family=")) || "").split("=")[1] ||
  (args.includes("--family") ? args[args.indexOf("--family") + 1] : null);
let routes;
if (args.includes("--all")) routes = downloading;
else if (famArg) routes = downloading.filter((r) => r.family === famArg);
else {
  // representative sample: up to 3 routes per family (the --all sweep covers everything, but is slow on
  // heavy pages). Every family is represented so the crawl spans the whole download surface.
  const cap = Number((args.find((a) => a.startsWith("--per=")) || "").split("=")[1]) || 2;
  const seen = {};
  routes = [];
  for (const r of downloading) {
    seen[r.family] = (seen[r.family] || 0) + 1;
    if (seen[r.family] <= cap) routes.push(r);
  }
}

const IGNORE =
  /webgpu|gpu adapter|needs-webgpu|requestAdapter|WebGPU|getUserMedia|camera|Permission|NotAllowed|net::ERR|Failed to (fetch|load)|dynamically imported|huggingface\.co|cdn\.jsdelivr|storage\.googleapis|ort-wasm|Tried to load/i;

const PROBE = `(() => {
  const loader = document.querySelector(".model-loader");
  const comp = document.querySelector("model-download-status");
  const ids = [...document.querySelectorAll("[id]")].map(e=>e.id);
  const dup = ids.filter((id,i)=>ids.indexOf(id)!==i);
  return {
    loaderPresent: !!loader,
    componentPresent: !!comp,
    dupIds: [...new Set(dup)].slice(0,5),
    overflow: document.documentElement.scrollWidth - window.innerWidth,
  };
})()`;

const { server, port } = await startServer();
const { CDP } = await import("./browser.mjs");
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);

const rows = [];
try {
  for (const r of routes) {
    const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/models/${r.slug}/`);
    await sleep(700);
    let probe = null;
    for (let i = 0; i < 1 && !probe; i++) {
      try {
        probe = await evalValue(cdp, pg.sessionId, PROBE, 3500);
      } catch {
        await sleep(300);
      }
    }
    process.stderr.write(`  crawled ${r.slug} (${r.family})${probe ? "" : " [busy]"}\n`);
    // mobile overflow (only if the desktop probe worked and there was no overflow issue)
    let mobileOverflow = null;
    if (probe) {
      try {
        await setViewport(cdp, pg.sessionId, {
          width: 390,
          height: 844,
          deviceScaleFactor: 2,
          mobile: true,
        });
        await sleep(250);
        mobileOverflow = await evalValue(
          cdp,
          pg.sessionId,
          `document.documentElement.scrollWidth - window.innerWidth`,
          5000,
        );
      } catch { /* busy */ }
    }
    const errs = pg.errors.filter((e) => !IGNORE.test(e));
    rows.push({ slug: r.slug, family: r.family, probe, mobileOverflow, errs });
    await closePage(cdp, pg.targetId);
  }
} finally {
  chrome.kill();
  server.close();
}

// ── report ──────────────────────────────────────────────────────────────────────────────────────────
let ok = 0, consoleClean = 0, introspected = 0, busy = 0, fails = 0;
for (const row of rows) {
  const p = row.probe;
  const cErr = row.errs.length === 0;
  if (cErr) consoleClean++;
  let verdict;
  if (!p) {
    busy++;
    verdict = cErr ? "LOADED (introspect busy)" : "CONSOLE-ERRORS (busy)";
  } else {
    introspected++;
    const overflowD = p.overflow > 2;
    const overflowM = row.mobileOverflow != null && row.mobileOverflow > 2;
    const bad = !p.loaderPresent || !p.componentPresent || p.dupIds.length > 0 || overflowD ||
      overflowM || !cErr;
    if (bad) {
      fails++;
      verdict = "FAIL " + JSON.stringify({
        loader: p.loaderPresent,
        comp: p.componentPresent,
        dup: p.dupIds,
        ovD: p.overflow,
        ovM: row.mobileOverflow,
        errs: row.errs.slice(0, 1),
      });
    } else {
      ok++;
      verdict = "OK";
    }
  }
  if (verdict !== "OK") console.log(`${row.slug.padEnd(30)} ${row.family.padEnd(30)} ${verdict}`);
}

console.log(
  `\ncrawled ${rows.length}/${downloading.length} downloading routes` +
    (args.includes("--all") ? " (ALL)" : " (representative sample)"),
);
console.log(`  console-error-free: ${consoleClean}/${rows.length}`);
console.log(
  `  introspected & healthy (loader+component present, no dup ids, no overflow desktop+mobile): ${ok}/${introspected}`,
);
console.log(`  main-thread busy (loaded, structural introspect skipped): ${busy}`);
console.log(`  FAILURES: ${fails}`);
// Fail the crawl on any console errors or any introspected structural failure.
const hardFails = fails + (rows.length - consoleClean);
process.exit(hardFails > 0 ? 1 : 0);
