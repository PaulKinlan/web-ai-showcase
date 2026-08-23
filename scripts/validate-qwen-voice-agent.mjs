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

    // ---- readiness gate, BEFORE anything is marked ready ----
    // Regression (PR #3 Codex review round 2): the text input stayed enabled while its button was
    // disabled, so Enter reached runTurn and the worker's ensureLoaded() started a 483 MB download
    // outside the loader's explicit Download action.
    check(
      `${name}: the command input is disabled while Qwen is absent`,
      await evaluate(sessionId, `document.getElementById("typedCmd").disabled`),
    );
    await evaluate(sessionId, `(() => {
      const el = document.getElementById("typedCmd");
      el.value = "what time is it in Tokyo?";
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      return true;
    })()`);
    await new Promise((r) => setTimeout(r, 800));
    check(
      `${name}: Enter cannot start a turn while Qwen is absent`,
      (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 0,
    );
    check(
      `${name}: refusing says the model must be downloaded first`,
      /not on this device yet/i.test(await evaluate(sessionId, `document.getElementById("status").textContent`)),
      await evaluate(sessionId, `document.getElementById("status").textContent`),
    );
    check(
      `${name}: the refusal did not start a download`,
      (await evaluate(sessionId, `[...document.querySelectorAll(".model-loader")].map(l=>l.dataset.state).join(",")`))
        .split(",").every((st) => st !== "downloading"),
    );
    check(
      `${name}: an answer live region exists for assistive tech`,
      await evaluate(sessionId, `document.getElementById("announcer")?.getAttribute("aria-live") === "polite"`),
    );
    check(`${name}: six tools advertised`, (await evaluate(sessionId, `document.querySelectorAll("#toollist li").length`)) === 6);
    // The repo rule is that a multi-model page must run every advertised stage or stay explicitly
    // unverified and unpublished. Nothing here has run end to end, so the page must say so and the
    // catalogue must not link it.
    check(
      `${name}: the page declares itself unverified`,
      /not yet been run end to end/i.test(await evaluate(sessionId, `document.getElementById("content").innerText`)),
    );

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

    // ---- turn 5: Whisper hears nothing — the turn must RESOLVE, not look stuck ----
    // Regression (PR #3 Codex review): the early return left llm1/tool/llm2 pinned at "pending", so
    // a finished turn read as a pipeline still working.
    await evaluate(sessionId, `(() => {
      const va = globalThis.__voiceAgent;
      va.engines.asr.transcribe = async () => ({ text: "   ", device: "wasm", ms: 12 });
      return true;
    })()`);
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "mic" })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 5");
    const t5 = await evaluate(sessionId, turnSnapshot);
    check(
      `${name}: a silent turn leaves NO stage pending`,
      t5.nodes.every((n) => !n.endsWith(":pending")),
      t5.nodes.join(" "),
    );
    check(
      `${name}: a silent turn marks the downstream stages skipped`,
      t5.nodes.filter((n) => n.endsWith(":skipped")).length === 3,
      t5.nodes.join(" "),
    );
    check(`${name}: a silent turn says what to do next`, /try again/i.test(t5.answer), t5.answer);

    // ---- switching the reasoning model must drop readiness, not inherit it ----
    // Regression (PR #3 Codex review): a superseded loader's onReady flipped ready.llm for a
    // checkpoint that was never fetched, which would then download outside any visible loader.
    await evaluate(sessionId, `globalThis.__voiceAgent.markReady("llm")`);
    check(`${name}: llm reads ready before the switch`, (await evaluate(sessionId, `globalThis.__voiceAgent.state().ready.llm`)) === true);
    await evaluate(sessionId, `(() => {
      const sel = document.getElementById("modelSize");
      sel.value = "onnx-community/Qwen2.5-1.5B-Instruct";
      sel.dispatchEvent(new Event("change"));
      return true;
    })()`);
    await new Promise((r) => setTimeout(r, 1500));
    const afterSwitch = await evaluate(sessionId, `globalThis.__voiceAgent.state()`);
    check(`${name}: switching selects the new checkpoint`, afterSwitch.currentModelId.includes("1.5B"), afterSwitch.currentModelId);
    check(`${name}: switching clears readiness rather than inheriting it`, afterSwitch.ready.llm === false);
    check(
      `${name}: the remounted loader advertises the new checkpoint`,
      await evaluate(sessionId, `document.getElementById("loader-llm").innerHTML.includes("Qwen2.5-1.5B-Instruct")`),
    );
    check(
      `${name}: nothing auto-downloaded after the switch`,
      (await evaluate(sessionId, `document.querySelector("#loader-llm .model-loader").dataset.state`)) !== "downloading",
    );

    // ---- a non-Latin transcript is speech ----
    // Regression (PR #3 Codex review round 2): the /[a-z0-9]/ test discarded Arabic, Chinese, Greek
    // and Cyrillic transcripts as "heard no words", though Whisper is multilingual and the page
    // claims no English-only restriction.
    await evaluate(sessionId, stubEngines(
      "東京は何時ですか",
      '<tool_call>{"name":"get_time","arguments":{"timezone":"Asia/Tokyo"}}</tool_call>',
      "東京の現在時刻です。",
    ));
    await evaluate(sessionId, `globalThis.__voiceAgent.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "mic" })`);
    await waitFor(sessionId, `!globalThis.__voiceAgent.state().busy`, 20_000, "turn 6");
    const t6 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a Japanese transcript is not discarded`, t6.heard.includes("東京"), t6.heard);
    check(`${name}: a non-Latin turn still reaches the tool`, t6.nodes.includes("tool:done"), t6.nodes.join(" "));

    check(`${name}: six turns are logged`, (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 6);
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
