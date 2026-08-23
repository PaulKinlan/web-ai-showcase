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
// WHAT THIS PROVES: each route loads, renders, and settles to an HONEST state; its visible controls
// are DRIVEN — every parameter slider moves and its readout follows, every disclosure opens, and the
// primary action is gated honestly (disabled with the reason on the page) rather than silently doing
// nothing; the shared engine and LiveMic still construct and expose the API the pages call; no
// console errors, no failed requests, no horizontal overflow, tap targets at 360px.
//
// WHAT IT DOES NOT PROVE: real Silero inference on any of these routes. This container has no GPU
// and no outbound network for Chromium, so no model can be fetched and no advertised model stage can
// run. Nothing here downloads. Every route therefore ends with its inference stages recorded
// EXPLICITLY UNVERIFIED (see the summary this prints) — never counted as a passing run. Those stages
// belong to the device pass, and the shared-runtime changes in this branch are accepted here only on
// the load/render/control/API axis they actually cover.
//
// Run: node scripts/validate-vad-routes.mjs

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, setViewport, startServer } from "./browser.mjs";

// Every route whose source references the shared VAD engine, plus the Silero ladder that ships it.
// `primary` is the control that would RUN the model; `stages` are the advertised model stages that
// cannot execute in this container and are reported as unverified.
const ROUTES = [
  { route: "models/silero-vad/", label: "overview", primary: "run", stages: ["silero VAD on an uploaded clip"] },
  { route: "models/silero-vad/basics/", label: "basics", primary: "run", stages: ["silero VAD speech/no-speech verdict"] },
  { route: "models/silero-vad/practical/", label: "practical", primary: "run", stages: ["silero VAD segmentation", "trimmed-WAV export"] },
  { route: "models/silero-vad/wild/", label: "wild (live mic)", primary: "micBtn", stages: ["live microphone VAD streaming"] },
  { route: "models/silero-vad/multi-model/", label: "multi-model", primary: "run", stages: ["silero VAD segmentation", "Whisper transcription"] },
  {
    route: "models/speaker-verification/multi-model/",
    label: "multi-model",
    primary: "run",
    stages: ["silero VAD trimming", "WavLM-SV embedding + cosine compare"],
  },
];
const unverified = [];

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

  for (const { route, label, primary, stages } of ROUTES) {
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

      // ---- drive the controls ----
      // Every range input paired with a readout: move it and confirm the page follows. These work
      // without the model, so "the control is wired" is genuinely provable here.
      const sliders = await evaluate(page.sessionId, `(() => {
        const out = [];
        for (const el of document.querySelectorAll('input[type=range]')) {
          const readout = document.getElementById(el.id + "Out") ||
            document.querySelector('[data-for="' + el.id + '"]');
          const before = readout?.textContent ?? null;
          const lo = Number(el.min || 0), hi = Number(el.max || 100), step = Number(el.step || 1);
          const target = Number(el.value) > (lo + hi) / 2 ? lo + step : hi - step;
          el.value = String(target);
          el.dispatchEvent(new Event("input", { bubbles: true }));
          out.push({ id: el.id, applied: Number(el.value) === target, readout: !!readout, changed: readout ? readout.textContent !== before : null });
        }
        return out;
      })()`);
      for (const sl of sliders) {
        check(`${name}: the ${sl.id} slider accepts a new value`, sl.applied, JSON.stringify(sl));
        if (sl.readout) check(`${name}: and its readout follows`, sl.changed === true, JSON.stringify(sl));
      }

      // Disclosures ("show file details", "how it works") must actually open.
      const disclosures = await evaluate(page.sessionId, `(() => {
        const els = [...document.querySelectorAll("details")].filter((d) => !d.open);
        const first = els[0];
        if (!first) return { total: 0 };
        first.querySelector("summary")?.click();
        return { total: els.length, opened: first.open };
      })()`);
      if (disclosures.total) {
        check(`${name}: a collapsed disclosure opens when its summary is activated`, disclosures.opened === true);
      }

      // The primary action must be gated HONESTLY while the model is absent: disabled, with the
      // reason visible on the page — never enabled-but-silent, and never quietly starting a
      // download this container could not complete anyway.
      const gate = await evaluate(page.sessionId, `(() => {
        const el = document.getElementById(${JSON.stringify(primary)});
        if (!el) return { missing: true };
        const before = document.body.textContent.length;
        el.click();
        return {
          disabled: !!el.disabled,
          label: (el.textContent || "").trim().slice(0, 40),
          reason: document.body.innerText.replace(/\\s+/g, " "),
          bodyGrew: document.body.textContent.length !== before,
        };
      })()`);
      check(`${name}: the primary control (#${primary}) exists`, !gate.missing);
      check(
        `${name}: it is disabled while the model is absent, not enabled-but-silent`,
        gate.disabled === true,
        gate.label,
      );
      check(
        `${name}: and the page says why the model is not ready`,
        /isn't on your device yet|download model|needs webgpu|not available/i.test(gate.reason || ""),
        (gate.reason || "").slice(0, 120),
      );
      check(`${name}: clicking the gated control changes nothing`, gate.bodyGrew === false);

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
      if (vpName === "desktop") unverified.push({ route, stages });
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

// A validator that lies is worse than no validator. Snippets sent to the page live in template
// literals, where an "invalid" escape silently COLLAPSES — `/\\s+/` arrives as `/s+/`, which strips
// the letter s instead of whitespace, and `/setTimeout\\(/` arrives as an unterminated group that
// throws at parse time and makes the whole evaluate return undefined. Both happened here. This scan
// fails the run rather than letting a weakened assertion pass quietly.
{
  const selfSrc = readFileSync(new URL(import.meta.url), "utf8");
  const offenders = [];
  for (const m of selfSrc.matchAll(/`([^`]*)`/gs)) {
    const line = selfSrc.slice(0, m.index).split("\n").length;
    for (const esc of new Set(m[1].match(/\\./g) ?? [])) {
      if (!"nrt`$\\".includes(esc[1])) offenders.push(`line ${line}: ${esc}`);
    }
  }
  check(
    "no template literal in this file carries a collapsing escape",
    offenders.length === 0,
    offenders.join(", "),
  );
}

console.log(`\n${checks - failed}/${checks} checks passed`);
console.log("\nEXPLICITLY UNVERIFIED — no model was downloaded and no inference ran here (no GPU, no");
console.log("outbound network for Chromium). These advertised stages are NOT covered by the run above");
console.log("and remain the device pass's job:");
for (const u of unverified) {
  for (const stage of u.stages) console.log(`  • ${u.route} — ${stage}`);
}
process.exit(failed ? 1 : 0);
