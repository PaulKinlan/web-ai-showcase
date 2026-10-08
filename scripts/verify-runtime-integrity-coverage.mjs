#!/usr/bin/env node
// Completeness audit for the service-worker runtime integrity manifest (bead web-ai-showcase-mtu).
//
// QUESTION: does a REAL browser session fetch any cdn.jsdelivr.net URL that is NOT in
// runtime-integrity.json? A URL outside the manifest is served pass-through with NO integrity check at
// all, so each one is a coverage gap. This script does not try to prove pass-through is safe - it
// measures how much of the real traffic is actually covered, which is the number the manifest's
// limitation text claims.
//
// METHOD: drive built routes from each runtime family in a real headless Chrome and attribute every
// cdn.jsdelivr.net request. Worker-originated requests need their own Network.enable, or they are
// invisible - every one of these runtimes is loaded from inside a Web Worker, so without auto-attach
// this audit would report a clean zero and mean nothing.
//
// A route that never navigated, or that fetched no jsDelivr URL at all, is reported INCONCLUSIVE rather
// than covered: "we saw nothing" is not "there is nothing", and the whole point of this file is to avoid
// that confusion.
//
// Run: node scripts/verify-runtime-integrity-coverage.mjs
import { readFileSync } from "node:fs";
import { CDP, closePage, DESKTOP, launchChrome, openPage, setViewport, startServer } from "./browser.mjs";

const MANIFEST = JSON.parse(readFileSync(new URL("../runtime-integrity.json", import.meta.url), "utf8"));
// Same normalisation the worker uses: a query string cannot change immutable bytes, so the manifest key
// is origin + pathname.
const keyOf = (url) => {
  const parsed = new URL(url);
  return `${parsed.origin}${parsed.pathname}`;
};
const pinned = new Set(Object.keys(MANIFEST.urls).map(keyOf));

// One built route per runtime family the site actually loads from jsDelivr.
const ROUTES = [
  { family: "transformers.js", route: "models/embeddinggemma/basics/" },
  { family: "raw onnxruntime-web", route: "models/animegan-cartoonization/basics/" },
  { family: "mediapipe tasks-vision", route: "models/gesture-recognizer/basics/" },
  { family: "outetts", route: "models/outetts/basics/" },
];

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}/web-ai-showcase/`;

let chrome;
const report = [];
try {
  chrome = await launchChrome({});
  const cdp = new CDP(chrome.ws);

  const tracked = new Set();
  let requestUrls = new Map();
  let jsdelivr = new Map();
  cdp.on((msg) => {
    if (msg.method === "Target.attachedToTarget") {
      const type = msg.params.targetInfo?.type;
      if (type === "worker" || type === "service_worker") {
        const sid = msg.params.sessionId;
        if (!tracked.has(sid)) {
          tracked.add(sid);
          void cdp.send("Network.enable", {}, sid).catch(() => tracked.delete(sid));
        }
      }
      return;
    }
    if (msg.method === "Network.requestWillBeSent") {
      requestUrls.set(msg.params.requestId, msg.params.request.url);
      if (msg.params.request.url.startsWith("https://cdn.jsdelivr.net/")) {
        jsdelivr.set(msg.params.request.url, { status: null, fromCache: null });
      }
    } else if (msg.method === "Network.responseReceived") {
      const url = requestUrls.get(msg.params.requestId);
      if (url && jsdelivr.has(url)) {
        jsdelivr.set(url, {
          status: msg.params.response.status,
          fromCache: Boolean(msg.params.response.fromDiskCache || msg.params.response.fromServiceWorker),
        });
      }
    }
  });
  await cdp.send("Target.setDiscoverTargets", { discover: true });
  await cdp.send("Target.setAutoAttach", { autoAttach: true, flatten: true, waitForDebuggerOnStart: false });

  const evaluate = (sessionId, expression, timeout = 60000) =>
    cdp
      .send(
        "Runtime.evaluate",
        {
          expression: `(async()=>{try{return (${expression})}catch(error){return {__error:String(error?.stack||error)}}})()`,
          awaitPromise: true,
          returnByValue: true,
        },
        sessionId,
        timeout,
      )
      .then((r) => r.result?.value)
      .catch((error) => ({ __error: String(error?.message || error) }));

  for (const { family, route } of ROUTES) {
    requestUrls = new Map();
    jsdelivr = new Map();
    let page;
    const entry = { family, route, href: null, navigated: false, triggers: 0, urls: [], missing: [], errors: [] };
    try {
      page = await openPage(cdp, base + route);
      await setViewport(cdp, page.sessionId, DESKTOP);
      // Guard against the silent false negative: openPage does not inspect its own navigation result, so
      // confirm where we actually ended up before believing any measurement taken here.
      const state = await evaluate(page.sessionId, `({href: document.location.href, title: document.title, ready: document.readyState})`);
      entry.href = state?.href ?? null;
      entry.navigated = typeof entry.href === "string" && entry.href.includes(route.replace(/\/$/, ""));
      if (!entry.navigated) {
        entry.errors.push(`did not navigate to the route (landed on ${entry.href})`);
        continue;
      }

      // Trigger the runtime the page needs: the loaders are behind buttons and the runtime is fetched
      // from inside a worker, so a page that is merely open proves nothing.
      for (let attempt = 0; attempt < 3; attempt++) {
        const clicked = await evaluate(
          page.sessionId,
          `(() => {
            const wanted = /^(run|load|start|index|analy|detect|generate|speak|synthes|transcribe|cartoon|recogni|process)/i;
            const buttons = [...document.querySelectorAll("button")].filter(
              (b) => !b.disabled && wanted.test((b.textContent || "").trim()),
            );
            const target = buttons.find((b) => !b.dataset.eg2Triggered);
            if (!target) return null;
            target.dataset.eg2Triggered = "1";
            target.click();
            return (target.textContent || "").trim().slice(0, 40);
          })()`,
        );
        if (typeof clicked !== "string") break;
        entry.triggers++;
        // Bounded wait for the runtime to be requested and downloaded, then look for the next trigger.
        await new Promise((resolve) => setTimeout(resolve, 20000));
      }

      entry.urls = [...jsdelivr.entries()].map(([url, meta]) => ({
        url,
        pinned: pinned.has(keyOf(url)),
        status: meta.status,
        fromCache: meta.fromCache,
      }));
      entry.missing = entry.urls.filter((u) => !u.pinned).map((u) => u.url);
    } catch (error) {
      entry.errors.push(String(error?.message || error));
    } finally {
      // Always close the page: a route left open holds its workers and runtime alive, and this audit
      // would then be measuring its own leftovers.
      if (page) await closePage(cdp, page.targetId).catch(() => {});
      // Pushed from the finally, not after it: an earlier draft pushed after the try block, so a route
      // that bailed early with `continue` disappeared from the report entirely and the audit printed a
      // clean zero over routes it never measured. Every route must appear whatever happened to it.
      report.push(entry);
    }
  }
} finally {
  await chrome?.kill?.().catch(() => {});
  server?.close?.();
}

let gaps = 0;
let inconclusive = 0;
for (const entry of report) {
  console.log(`\n=== ${entry.family} · ${entry.route} ===`);
  if (entry.errors.length) for (const error of entry.errors) console.log(`  ERROR ${error}`);
  if (!entry.navigated) {
    inconclusive++;
    console.log("  INCONCLUSIVE: route did not navigate, so nothing was measured here");
    continue;
  }
  console.log(`  navigated to ${entry.href}`);
  console.log(`  triggered ${entry.triggers} control(s); ${entry.urls.length} cdn.jsdelivr.net request(s)`);
  for (const u of entry.urls) {
    console.log(`    ${u.pinned ? "COVERED   " : "UNCOVERED "} status=${u.status ?? "?"} ${keyOf(u.url).replace("https://cdn.jsdelivr.net/npm/", "")}`);
  }
  if (entry.urls.length === 0) {
    inconclusive++;
    console.log("  INCONCLUSIVE: no jsDelivr runtime was fetched, so this route proves nothing about coverage");
  } else if (entry.missing.length) {
    gaps += entry.missing.length;
    console.log(`  GAP: ${entry.missing.length} fetched URL(s) are not in the manifest and so are NOT integrity-checked`);
  } else {
    console.log("  COMPLETE: every jsDelivr URL this route fetched is pinned in the manifest");
  }
}
console.log(`\nRESULT routes=${report.length} uncoveredUrls=${gaps} inconclusive=${inconclusive}`);
process.exit(gaps > 0 || inconclusive > 0 ? 1 : 0);
