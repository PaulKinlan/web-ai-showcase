#!/usr/bin/env node
// Route check for every published page that consumes the SHARED Silero VAD runtime
// (models/silero-vad/vad.js + models/silero-vad/worker.js).
//
// Those two files are shared, so a change made for one demo — this branch adds `dispose()` to the
// engine, resumes a suspended AudioContext in LiveMic, and serialises the worker's stateful
// stream-chunk/stream-reset messages — can regress five published Silero routes and the
// speaker-verification multi-model route that never appear in the new demo's own validator.
// CLAUDE.md makes acceptance route-complete, not family-level, so every one of them is driven here
// at desktop AND mobile.
//
// WHAT THIS PROVES: each route loads, renders, and settles to an HONEST state; the shared engine and
// LiveMic still construct and expose the API the pages call; no console errors, no failed requests,
// no horizontal overflow, tap targets at 360px.
//
// WHAT IT DOES NOT PROVE: real Silero inference. This container has no GPU and no outbound network
// for Chromium, so no model can be fetched. Nothing here downloads, and a route that correctly shows
// its download-required state is recorded as exactly that — not as a passing inference run.
//
// Run: node scripts/validate-vad-routes.mjs

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, setViewport, startServer } from "./browser.mjs";

// Every route whose source references the shared VAD engine, plus the Silero ladder that ships it.
const ROUTES = [
  { slug: "silero-vad", route: "models/silero-vad/", label: "overview" },
  { slug: "silero-vad", route: "models/silero-vad/basics/", label: "basics" },
  { slug: "silero-vad", route: "models/silero-vad/practical/", label: "practical" },
  { slug: "silero-vad", route: "models/silero-vad/wild/", label: "wild (live mic)" },
  { slug: "silero-vad", route: "models/silero-vad/multi-model/", label: "multi-model" },
  { slug: "speaker-verification", route: "models/speaker-verification/multi-model/", label: "multi-model" },
];

const PROFILE_DIR = mkdtempSync(join(tmpdir(), "vad-routes-"));
let checks = 0;
let failed = 0;
let server;
let chrome;
let cdp;

function check(label, ok, detail = "") {
  checks++;
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 180)}` : ""}`);
  return ok;
}

async function evaluate(sessionId, expression, timeoutMs = 30_000) {
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.message||error)};}})()`,
    awaitPromise: true,
    returnByValue: true,
  }, sessionId, timeoutMs);
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}

// A loader that never leaves "checking" is the failure this repo cares about: the visitor is told
// something is happening when nothing is. Any settled state — including an honest download-required
// or unsupported one — is a pass.
const LOADER_SETTLED = `(() => {
  const hosts = [...document.querySelectorAll("model-download-status, [id$='loader'], [id^='loader']")];
  if (!hosts.length) return { hosts: 0, settled: true, text: "" };
  const text = hosts.map((h) => (h.shadowRoot?.textContent ?? h.textContent ?? "").replace(/\\s+/g, " ").trim()).join(" | ");
  return { hosts: hosts.length, settled: !/^\\s*$/.test(text) && !/\\bchecking\\b/i.test(text), text };
})()`;

const OVERFLOW = `(() => {
  const d = document.documentElement;
  return { scrollW: d.scrollWidth, clientW: d.clientWidth };
})()`;

// Tap targets: visible, enabled, in-flow controls only. Hidden/oversized-container cases are noise.
const SMALL_TARGETS = `(() => {
  const bad = [];
  for (const el of document.querySelectorAll("button, a[href], input[type=button], input[type=file], summary")) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none") continue;
    if (r.height < 44 && r.width < 44) bad.push((el.id || el.tagName) + " " + Math.round(r.width) + "x" + Math.round(r.height));
  }
  return bad;
})()`;

try {
  server = await startServer();
  chrome = await launchChrome({ userDataDir: PROFILE_DIR });
  cdp = new CDP(chrome.ws);

  for (const { route, label } of ROUTES) {
    for (const [vpName, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      const name = `${route} [${vpName}]`;
      const page = await openPage(cdp, `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
      await setViewport(cdp, page.sessionId, vp);
      await new Promise((r) => setTimeout(r, 2200));

      const title = await evaluate(page.sessionId, `document.querySelector("h1")?.textContent?.trim() ?? ""`);
      check(`${name}: renders (${label})`, title.length > 0, title);

      const loader = await evaluate(page.sessionId, LOADER_SETTLED);
      check(`${name}: the loader settles to an honest state, never a stuck "checking"`, loader.settled, loader.text);

      const ov = await evaluate(page.sessionId, OVERFLOW);
      check(`${name}: no horizontal overflow`, ov.scrollW <= ov.clientW + 1, `${ov.scrollW} vs ${ov.clientW}`);

      if (vpName === "mobile") {
        const small = await evaluate(page.sessionId, SMALL_TARGETS);
        check(`${name}: every visible control meets the 44px tap-target floor`, small.length === 0, small.join(", "));
      }

      // The shared runtime itself: the module still parses, the engine still constructs, and the API
      // the pages call — including the dispose() this branch adds — is still there.
      const api = await evaluate(page.sessionId, `(async () => {
        const m = await import("/web-ai-showcase/models/silero-vad/vad.js");
        const names = Object.keys(m);
        const proto = m.VadEngine ? Object.getOwnPropertyNames(m.VadEngine.prototype) : [];
        return { names, proto, liveMic: typeof m.LiveMic, supported: typeof m.LiveMic?.supported === "function" };
      })()`);
      check(`${name}: the shared VAD module still loads`, api.names.includes("VadEngine"), api.names.join(","));
      check(`${name}: VadEngine still exposes dispose()`, api.proto.includes("dispose"), api.proto.join(","));
      check(`${name}: LiveMic still exposes supported()`, api.supported === true);

      check(`${name}: zero console errors`, page.errors.length === 0, page.errors.join(" | "));
      check(`${name}: zero failed network requests`, page.netFailures.length === 0, page.netFailures.join(" | "));
      await closePage(cdp, page.targetId);
    }
  }
} catch (err) {
  failed++;
  console.error("\nRUNNER ERROR:", err?.message ?? err);
} finally {
  try {
    chrome?.kill?.();
  } catch { /* noop */ }
  try {
    server?.server?.close();
  } catch { /* noop */ }
  rmSync(PROFILE_DIR, { recursive: true, force: true });
}

console.log(`\n${checks - failed}/${checks} checks passed`);
console.log("NOTE: no model was downloaded and no Silero inference ran — this container has no GPU");
console.log("      and no outbound network. Real-inference coverage for these routes is the device pass.");
process.exit(failed ? 1 : 0);
