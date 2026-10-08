#!/usr/bin/env node
// Real-Chrome proof that the MediaPipe runtime path is actually verified by the service worker
// (bead web-ai-showcase-mtu, review finding F1).
//
// WHY THIS EXISTS: the service-worker unit tests run against a stubbed environment, so they establish
// the LOGIC and nothing about what Chrome does with these bytes. MediaPipe loads its runtime glue with
// importScripts(), which the Fetch spec defines as a no-cors request - the response is opaque (status 0,
// ok false) and cannot be read or hashed at all. A previous revision returned such a response early and
// handed the script over completely unchecked.
//
// HOW IT PROVES IT, rather than inferring it: sw.js stores a library response ONLY after its bytes have
// hashed to the pinned value. So finding the asset in SHELL_CACHE is direct evidence that the hash check
// ran and passed on bytes the browser actually served. This reads those cached bytes back IN THE PAGE
// and re-hashes them against runtime-integrity.json, which also proves the cached copy is the pinned
// artifact rather than something else.
//
// Run: node scripts/verify-mediapipe-integrity-runtime.mjs
import { readFileSync } from "node:fs";
import { CDP, DESKTOP, closePage, launchChrome, openPage, setViewport, startServer } from "./browser.mjs";

const MANIFEST = JSON.parse(readFileSync(new URL("../runtime-integrity.json", import.meta.url), "utf8"));
const keyOf = (url) => {
  const parsed = new URL(url);
  return `${parsed.origin}${parsed.pathname}`;
};
const expected = new Map(Object.entries(MANIFEST.urls).map(([url, meta]) => [keyOf(url), meta.sha256]));

// The three assets a MediaPipe vision route must fetch, one of which (.js) arrives via importScripts.
const WATCHED = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_internal.js",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_internal.wasm",
];

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
const route = "models/gesture-recognizer/basics/";

const checks = [];
const record = (ok, label, detail = "") => {
  checks.push(ok);
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
};

let chrome;
try {
  chrome = await launchChrome({});
  const cdp = new CDP(chrome.ws);

  // Track network + any 502 the worker returns, per page session.
  const requestUrls = new Map();
  const responses = new Map();
  const badResponses = [];
  const tracked = new Set();
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
    } else if (msg.method === "Network.responseReceived") {
      const url = requestUrls.get(msg.params.requestId);
      const status = msg.params.response.status;
      if (url) {
        responses.set(url, { status, fromServiceWorker: Boolean(msg.params.response.fromServiceWorker) });
        if (status === 502) badResponses.push(url);
      }
    }
  });
  await cdp.send("Target.setDiscoverTargets", { discover: true });
  await cdp.send("Target.setAutoAttach", { autoAttach: true, flatten: true, waitForDebuggerOnStart: false });

  const evaluate = (sessionId, expression, timeout = 90000) =>
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

  let page;
  try {
    page = await openPage(cdp, base + route);
    await setViewport(cdp, page.sessionId, DESKTOP);
    const landed = await evaluate(page.sessionId, "document.location.href");
    record(typeof landed === "string" && landed.includes(route), "page navigated to the MediaPipe route", String(landed));

    // Register and wait for the worker, then RELOAD so this page load is actually controlled by it.
    // Proving the runtime path on a page the worker does not control would prove nothing.
    const registration = await evaluate(
      page.sessionId,
      `(async () => {
        const reg = await navigator.serviceWorker.register("/web-ai-showcase/sw.js", { scope: "/web-ai-showcase/" });
        await navigator.serviceWorker.ready;
        return { scope: reg.scope, controller: Boolean(navigator.serviceWorker.controller) };
      })()`,
    );
    record(registration?.scope === "/web-ai-showcase/", "service worker registered for the site scope", registration?.scope ?? String(registration));
    await evaluate(page.sessionId, "location.reload()").catch(() => {});
    await new Promise((resolve) => setTimeout(resolve, 3000));
    const controlled = await evaluate(page.sessionId, "Boolean(navigator.serviceWorker.controller)");
    record(controlled === true, "page is controlled by the service worker (so its fetches are intercepted)", String(controlled));

    // Drive the two real controls: the model loader, then the run button.
    const clickControl = `(selector) => {
      const node = document.querySelector(selector);
      if (!node || node.disabled || node.dataset.eg2Triggered) return null;
      node.dataset.eg2Triggered = "1"; node.click();
      return (node.id ? "#" + node.id + " " : "") + (node.textContent || "").trim().slice(0, 32);
    }`;
    const clicked = [];
    const deadline = Date.now() + 210000;
    // The runtime is imported when the worker starts, so the glue fetch happens early; keep going until
    // all three watched assets have been seen or the bound expires.
    while (Date.now() < deadline) {
      for (const selector of [".model-loader button, .model-loader [role=button]", "button#run, button[id^=run-]"]) {
        const hit = await evaluate(page.sessionId, `(${clickControl})(${JSON.stringify(selector)})`);
        if (typeof hit === "string") clicked.push(hit);
      }
      const seen = WATCHED.filter((url) => responses.has(url)).length;
      if (seen === WATCHED.length) break;
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
    record(clicked.length > 0, "drove the real page controls", clicked.join(" -> ") || "none found");

    for (const url of WATCHED) {
      const res = responses.get(url);
      record(Boolean(res), `browser fetched ${keyOf(url).replace("https://cdn.jsdelivr.net/npm/", "")}`, res ? `status=${res.status} fromSW=${res.fromServiceWorker}` : "not seen");
    }
    record(badResponses.length === 0, "no 502 from the worker (nothing failed closed on this route)", badResponses.length ? badResponses.join(", ") : "none");

    // The decisive evidence: read the cached bytes back in the page and re-hash them.
    const cached = await evaluate(
      page.sessionId,
      `(async () => {
        const out = [];
        const names = await caches.keys();
        for (const url of ${JSON.stringify(WATCHED)}) {
          let found = null;
          for (const name of names) {
            const cache = await caches.open(name);
            const hit = await cache.match(url);
            if (hit) { found = { cache: name, type: hit.type, status: hit.status }; break; }
          }
          if (!found) { out.push({ url, present: false }); continue; }
          let hash = null;
          let bytes = null;
          try {
            const cache = await caches.open(found.cache);
            const hit = await cache.match(url);
            const buf = await hit.arrayBuffer();
            bytes = buf.byteLength;
            const digest = await crypto.subtle.digest("SHA-256", buf);
            hash = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
          } catch (error) { hash = "UNREADABLE: " + String(error && error.message); }
          out.push({ url, present: true, ...found, bytes, hash });
        }
        return out;
      })()`,
    );

    if (!Array.isArray(cached)) {
      record(false, "could read Cache Storage from the page", JSON.stringify(cached).slice(0, 200));
    } else {
      for (const entry of cached) {
        const short = keyOf(entry.url).replace("https://cdn.jsdelivr.net/npm/", "");
        if (!entry.present) {
          record(false, `cached (proves verification ran): ${short}`, "NOT in any cache, so the hash check did not complete for it");
          continue;
        }
        const want = expected.get(keyOf(entry.url));
        record(entry.hash === want, `cached bytes hash to the pinned value: ${short}`, entry.hash === want ? `${entry.bytes} bytes, hash matches` : `got ${entry.hash} want ${want}`);
      }
    }
  } finally {
    if (page) await closePage(cdp, page.targetId).catch(() => {});
  }
} finally {
  await chrome?.kill?.().catch(() => {});
  server?.close?.();
}

const passed = checks.filter(Boolean).length;
console.log(`\nRESULT ${passed}/${checks.length} checks passed`);
process.exit(passed === checks.length ? 0 : 1);
