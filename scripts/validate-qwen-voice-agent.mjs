#!/usr/bin/env node
// Headless route check for models/qwen-tiny-llm/multi-model/ (the voice agent).
//
// What this DOES cover, for real, at desktop and mobile:
//   • the page loads with zero console errors and no failed requests
//   • all three loaders reach an HONEST state (download-required / unsupported-needs-WebGPU on a
//     GPU-less runner) and never auto-download
//   • no horizontal overflow, and the primary controls meet the 44px tap-target floor
//   • a COMPLETE turn: transcript → Qwen's <tool_call> → the real tool executing → page state
//     actually changing → the second-pass answer rendering, with the stage flow and readout filled in
//   • the honest no-tool-call path, and the tool-failure path
//
// The turn is driven with SUBSTITUTED engines (window.__voiceAgent) because this runner has no GPU
// and cannot fetch ~600 MB of weights. The transcript and the model's raw text are stubbed; the
// tool execution, the parsing, and every pixel of rendering are the page's own code. Real
// end-to-end inference still needs a WebGPU browser — see the run record.
//
// Usage: node scripts/validate-qwen-voice-agent.mjs

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

const ROUTE = "models/qwen-tiny-llm/multi-model/";
const REMOTE_HOST = "not-localhost.test"; // resolved to loopback by --host-resolver-rules
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "voice-agent-"));
let checks = 0;
let failed = 0;
let server;
let chrome;
let cdp;

function check(label, ok, detail = "") {
  checks++;
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 200)}` : ""}`);
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

async function waitFor(sessionId, expression, timeoutMs, label) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const v = await evaluate(sessionId, expression);
    if (v) return v;
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(`timed out after ${timeoutMs}ms waiting for ${label}`);
}

// A stub pair of engines. Whisper returns a fixed transcript; Qwen returns a fixed raw completion on
// the first pass and a fixed sentence on the second. Everything downstream is the page's real code.
const stubEngines = (transcript, firstText, secondText) => `
  (() => {
    const va = globalThis.__voiceAgent;
    va.engines.asr.transcribe = async () => ({ text: ${JSON.stringify(transcript)}, device: "wasm", ms: 42 });
    let pass = 0;
    va.engines.llm.chat = async (messages, opts = {}) => {
      pass++;
      const text = pass === 1 ? ${JSON.stringify(firstText)} : ${JSON.stringify(secondText)};
      opts.onPrompt?.("<|im_start|>system\\n(stubbed prompt)<|im_end|>");
      for (const tok of text.split(/(?<=\\s)/)) opts.onToken?.(tok, 1);
      return { ms: 30, tokens: text.split(" ").length, text };
    };
    va.markReady("vad"); va.markReady("asr"); va.markReady("llm");
    return true;
  })()`;

const turnSnapshot = `
  (() => {
    const card = document.querySelector("#turns .turn");
    if (!card) return null;
    const nodes = [...card.querySelectorAll(".flow .node")].map(n => n.dataset.key + ":" + n.dataset.state);
    return {
      heard: card.querySelector('[data-role="heard"]').textContent.trim(),
      answer: card.querySelector('[data-role="answer"]').textContent.trim(),
      readout: card.querySelector('[data-role="readout"]').textContent.trim(),
      body: card.querySelector('[data-role="body"]').textContent,
      nodes,
      notes: [...document.querySelectorAll("#notes li")].map(li => li.textContent),
      timers: [...document.querySelectorAll("#timers .timer")].map(t => t.textContent),
      lastCalc: document.getElementById("lastCalc").textContent.trim(),
    };
  })()`;

try {
  server = await startServer();
  // MAP a non-local hostname onto the loopback server so the published-origin behaviour of the
  // debug hook can be checked for real, rather than asserted from the source.
  chrome = await launchChrome({
    userDataDir: PROFILE_DIR,
    extraArgs: [`--host-resolver-rules=MAP ${REMOTE_HOST} 127.0.0.1`],
  });
  cdp = new CDP(chrome.ws);
  const base = `http://127.0.0.1:${server.port}/web-ai-showcase/${ROUTE}`;

  for (const [name, viewport] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
    console.log(`\n===== ${name} =====`);
    const page = await openPage(cdp, base);
    const sessionId = page.sessionId;
    await setViewport(cdp, sessionId, viewport);
    await new Promise((r) => setTimeout(r, 3000));

    check(`${name}: no console errors`, page.errors.length === 0, page.errors.join(" | "));
    check(`${name}: no failed requests`, page.netFailures.length === 0, page.netFailures.join(" | "));

    const layout = await evaluate(sessionId, `(() => {
      const d = document.documentElement;
      const primary = ["listen","typedGo","runClip","modelSize","typedCmd"]
        .map(id => document.getElementById(id))
        .filter(Boolean)
        .map(el => ({ id: el.id, h: Math.round(el.getBoundingClientRect().height) }));
      return { overflow: d.scrollWidth > d.clientWidth, scrollW: d.scrollWidth, clientW: d.clientWidth,
               small: primary.filter(p => p.h < 44).map(p => p.id + ":" + p.h + "px") };
    })()`);
    check(`${name}: no horizontal overflow`, !layout.overflow, `${layout.scrollW} vs ${layout.clientW}`);
    check(`${name}: primary controls are >= 44px tall`, layout.small.length === 0, layout.small.join(", "));

    const loaders = await evaluate(sessionId, `[...document.querySelectorAll(".model-loader")].map(l => l.dataset.state)`);
    check(`${name}: three loaders mounted`, loaders.length === 3, loaders.join(", "));
    const honest = loaders.every((st) => ["download-required", "unsupported", "checking", "ready", "partial", "update"].includes(st));
    check(`${name}: every loader is in an honest state`, honest, loaders.join(", "));
    check(
      `${name}: nothing auto-downloaded on a cache-absent profile`,
      !loaders.includes("downloading"),
      loaders.join(", "),
    );
    check(
      `${name}: the WebGPU-only model reports needs-WebGPU on this GPU-less runner`,
      loaders.includes("unsupported"),
      loaders.join(", "),
    );

    check(`${name}: the validator hook is present on localhost`, await evaluate(sessionId, `!!globalThis.__voiceAgent`));
    check(`${name}: six tools advertised`, (await evaluate(sessionId, `document.querySelectorAll("#toollist li").length`)) === 6);

    // ---- turn 1: a real tool call that changes page state ----
    await evaluate(sessionId, stubEngines(
      "Set a timer for 45 seconds for the pasta.",
      '<tool_call>\n{"name": "start_timer", "arguments": {"seconds": 45, "label": "pasta"}}\n</tool_call>',
      "Your pasta timer is running for 45 seconds.",
    ));
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "mic" })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 1");
    const t1 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: turn 1 shows the transcript`, t1.heard.includes("timer for 45 seconds"), t1.heard);
    check(`${name}: turn 1 parsed the tool call`, t1.body.includes('"name": "start_timer"'), t1.body.slice(0, 120));
    check(`${name}: turn 1 ran the tool for real`, t1.nodes.includes("tool:done"), t1.nodes.join(" "));
    check(`${name}: turn 1 really started a timer on the page`, t1.timers.length === 1, JSON.stringify(t1.timers));
    check(`${name}: turn 1 rendered the second-pass answer`, t1.answer.includes("45 seconds"), t1.answer);
    check(`${name}: turn 1 filled the timing readout`, /total/.test(t1.readout), t1.readout);
    check(`${name}: every stage node resolved`, t1.nodes.every((n) => !n.endsWith(":pending")), t1.nodes.join(" "));

    // ---- turn 2: the honest no-tool-call path ----
    await evaluate(sessionId, stubEngines(
      "Who wrote Middlemarch?",
      "Middlemarch was written by George Eliot.",
      "unused",
    ));
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ text: "Who wrote Middlemarch?" })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 2");
    const t2 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: no-tool turn is labelled, not hidden`, t2.body.includes("No tool call in that reply"), t2.body.slice(-160));
    check(`${name}: no-tool turn marks the tool stage failed`, t2.nodes.includes("tool:fail"), t2.nodes.join(" "));
    check(`${name}: no-tool turn still shows the direct answer`, t2.answer.includes("George Eliot"), t2.answer);

    // ---- turn 3: a tool that fails — the model is told, the page says so ----
    await evaluate(sessionId, stubEngines(
      "What is banana times fish?",
      '<tool_call>{"name":"calculate","arguments":{"expression":"banana * fish"}}</tool_call>',
      "I couldn't work that one out.",
    ));
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ text: "What is banana times fish?" })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 3");
    const t3 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a failing tool is reported, not swallowed`, t3.nodes.includes("tool:fail"), t3.nodes.join(" "));
    check(`${name}: the failure reason is on the page`, /unknown name|the tool failed/i.test(t3.body), t3.body.slice(-180));

    // ---- turn 4: a note, proving add_note mutates real page state ----
    await evaluate(sessionId, stubEngines(
      "Note that the wifi password is hunter2.",
      '<tool_call>{"name":"add_note","arguments":{"text":"wifi password is hunter2"}}</tool_call>',
      "Saved that note for you.",
    ));
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ text: "Note that the wifi password is hunter2." })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 4");
    const t4 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: add_note wrote to the on-page notepad`, t4.notes.some((n) => n.includes("hunter2")), JSON.stringify(t4.notes));

    check(`${name}: four turns are logged`, (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 4);
    check(`${name}: still no console errors after four turns`, page.errors.length === 0, page.errors.join(" | "));

    await closePage(cdp, page.targetId);
  }

  // ---- published-origin behaviour of the debug hook ----
  // Regression (PR #3 review): __voiceAgent.markReady() flips readiness flags without a loader
  // having finished — the exact "ready" lie the honest-state invariant exists to prevent. It must
  // not exist off localhost. Served from the SAME loopback server under a non-local hostname.
  console.log(`\n===== published origin (${REMOTE_HOST}) =====`);
  const remote = await openPage(cdp, `http://${REMOTE_HOST}:${server.port}/web-ai-showcase/${ROUTE}`);
  await setViewport(cdp, remote.sessionId, DESKTOP);
  await new Promise((r) => setTimeout(r, 3000));
  check(
    "the debug hook is ABSENT on a non-localhost origin",
    (await evaluate(remote.sessionId, `typeof globalThis.__voiceAgent`)) === "undefined",
  );
  check(
    "readiness cannot be spoofed from the page on a non-localhost origin",
    (await evaluate(remote.sessionId, `typeof globalThis.__voiceAgent?.markReady`)) === "undefined",
  );
  check(
    "the page still renders normally there",
    (await evaluate(remote.sessionId, `document.querySelectorAll("#toollist li").length`)) === 6,
  );
  check(
    "loaders still reach honest states there",
    (await evaluate(remote.sessionId, `[...document.querySelectorAll(".model-loader")].length`)) === 3,
  );
  check("no console errors on the published origin", remote.errors.length === 0, remote.errors.join(" | "));
  await closePage(cdp, remote.targetId);
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
process.exit(failed ? 1 : 0);
