#!/usr/bin/env node
// Headless route check for models/ultravox-audio-llm/ — the native audio-in demo.
//
// Covers, for real, at desktop and mobile:
//   • the page loads with zero console errors and no failed requests
//   • the WebGPU-only model reaches an HONEST unsupported state on a GPU-less runner, the status line
//     says so, and NOTHING is downloaded
//   • no horizontal overflow; primary controls meet the 44px tap-target floor
//   • a COMPLETE turn: audio in → the model's function-call JSON parsed → the real tool executing →
//     page state changing → the second-pass answer rendering, with the audio footprint reported
//   • the honest no-tool-call path
//   • the debug hook is absent on a non-localhost origin
//
// The turn is driven with a SUBSTITUTED engine (window.__ultravox), because this runner has no GPU
// and cannot fetch ~1.5 GB of weights. The model's output is stubbed; the parsing, tool execution and
// every pixel of rendering are the page's own code. Real audio-in inference still needs a WebGPU
// browser with a microphone — see the unverified banner on the page itself.

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
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

const ROUTE = "models/ultravox-audio-llm/";
const REMOTE_HOST = "not-localhost.test";
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "ultravox-"));
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
    if (await evaluate(sessionId, expression)) return true;
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(`timed out after ${timeoutMs}ms waiting for ${label}`);
}

// Stub the Ultravox engine. It returns the audio-footprint numbers a real run would report, so the
// rendering path that proves "audio entered the prompt as embeddings" is exercised.
const stubEngine = (firstText, secondText, audioFrames = 26) => `
  (() => {
    const uv = globalThis.__ultravox;
    let pass = 0;
    uv.engines.llm.generate = async (opts = {}) => {
      pass++;
      const text = pass === 1 ? ${JSON.stringify(firstText)} : ${JSON.stringify(secondText)};
      opts.onPrompt?.("<|start_header_id|>user<|end_header_id|>\\n\\n" + "<|audio|>".repeat(${audioFrames}));
      return { text, prepMs: 40, genMs: 120, promptTokens: 180, audioFrames: ${audioFrames}, newTokens: 24, device: "webgpu" };
    };
    uv.markReady("llm"); uv.markReady("vad");
    return true;
  })()`;

const turnSnapshot = `
  (() => {
    const card = document.querySelector("#turns .turn");
    if (!card) return null;
    return {
      nodes: [...card.querySelectorAll(".flow .node")].map(n => n.dataset.key + ":" + n.dataset.state),
      body: card.querySelector('[data-role="body"]').textContent,
      answer: card.querySelector('[data-role="answer"]').textContent.trim(),
      readout: card.querySelector('[data-role="readout"]').textContent,
      audioBars: card.querySelectorAll(".audioviz span").length,
      notes: [...document.querySelectorAll("#notes li")].map(li => li.textContent),
      timers: [...document.querySelectorAll("#timers .timer")].map(t => t.textContent),
    };
  })()`;

try {
  server = await startServer();
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
      const primary = ["listen","runClip","clipPrompt"].map(id => document.getElementById(id)).filter(Boolean)
        .map(el => ({ id: el.id, h: Math.round(el.getBoundingClientRect().height) }));
      return { overflow: d.scrollWidth > d.clientWidth, scrollW: d.scrollWidth, clientW: d.clientWidth,
               small: primary.filter(p => p.h < 44).map(p => p.id + ":" + p.h) };
    })()`);
    check(`${name}: no horizontal overflow`, !layout.overflow, `${layout.scrollW} vs ${layout.clientW}`);
    check(`${name}: primary controls are >= 44px tall`, layout.small.length === 0, layout.small.join(", "));

    const loaders = await evaluate(sessionId, `[...document.querySelectorAll(".model-loader")].map(l=>l.dataset.state)`);
    check(`${name}: two loaders mounted`, loaders.length === 2, loaders.join(", "));
    check(
      `${name}: the WebGPU-only model reports needs-WebGPU on this runner`,
      loaders.includes("unsupported"),
      loaders.join(", "),
    );
    check(`${name}: nothing auto-downloaded`, !loaders.includes("downloading"), loaders.join(", "));
    const status = await evaluate(sessionId, `document.getElementById("status").textContent`);
    check(`${name}: the status line explains why, not "preparing"`, /can't run Ultravox/i.test(status), status);
    check(`${name}: it promises no CPU fallback`, /no CPU fallback/i.test(status), status);

    check(
      `${name}: the page declares itself unverified`,
      /not yet been run end to end/i.test(await evaluate(sessionId, `document.getElementById("content").innerText`)),
    );
    check(`${name}: six tools advertised`, (await evaluate(sessionId, `document.querySelectorAll("#toollist li").length`)) === 6);
    check(
      `${name}: the page states there is no transcription`,
      /no transcript|not a transcript|no speech-to-text|never becomes text/i.test(
        await evaluate(sessionId, `document.getElementById("content").innerText`),
      ),
    );

    // ---- readiness gate ----
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "mic" })`);
    await new Promise((r) => setTimeout(r, 600));
    check(
      `${name}: a turn cannot start while the model is absent`,
      (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 0,
    );

    // ---- turn 1: a real tool call, driven by audio ----
    await evaluate(sessionId, stubEngine(
      '<|python_tag|>{"name": "start_timer", "parameters": {"seconds": 45, "label": "pasta"}}',
      "Your pasta timer is running for 45 seconds.",
    ));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(32000), seconds: 2, source: "mic" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 1");
    const t1 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: turn 1 parsed the llama-style call`, t1.body.includes('"name": "start_timer"'), t1.body.slice(0, 120));
    check(`${name}: turn 1 ran the tool for real`, t1.nodes.includes("tool:done"), t1.nodes.join(" "));
    check(`${name}: turn 1 really started a timer`, t1.timers.length === 1, JSON.stringify(t1.timers));
    check(`${name}: turn 1 rendered the answer`, t1.answer.includes("45 seconds"), t1.answer);
    check(`${name}: every stage resolved`, t1.nodes.every((n) => !n.endsWith(":pending")), t1.nodes.join(" "));
    // The distinguishing surface: the audio's footprint in the prompt, not a transcript.
    check(`${name}: the audio footprint is drawn`, t1.audioBars > 0, String(t1.audioBars));
    check(
      `${name}: the turn reports embedding positions, not words`,
      /embedding positions/.test(t1.body) && /No transcript was produced/.test(t1.body),
      t1.body.slice(-200),
    );
    check(`${name}: the readout counts audio positions`, /positions/.test(t1.readout), t1.readout);

    // ---- turn 2: honest no-tool-call path ----
    await evaluate(sessionId, stubEngine("He is asking people to serve their country.", "unused"));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 2");
    const t2 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a no-tool turn is labelled, not hidden`, /No tool call in that reply/.test(t2.body), t2.body.slice(-160));
    check(`${name}: a no-tool turn still answers`, t2.answer.includes("serve their country"), t2.answer);
    check(`${name}: a no-tool turn marks the tool stage skipped`, t2.nodes.includes("tool:skipped"), t2.nodes.join(" "));
    // Regression (PR #3 Codex round 5): the clip is handed straight to Ultravox, so crediting Silero
    // with a "done" endpoint stage would claim a model ran that never did.
    check(
      `${name}: a clip turn does NOT credit Silero with running`,
      t2.nodes.includes("vad:skipped"),
      t2.nodes.join(" "),
    );

    // ---- turn 3: the model returns nothing usable ----
    // Regression (PR #3 Codex round 4): an empty generation was rendered as a completed "direct
    // answer" — a blank turn dressed up as a success.
    await evaluate(sessionId, stubEngine("", "unused"));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 3");
    const t3 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: an empty generation is marked failed`, t3.nodes.includes("answer:fail"), t3.nodes.join(" "));
    check(`${name}: an empty generation says so`, /no usable output/i.test(t3.body), t3.body.slice(-160));
    check(`${name}: an empty generation renders no fake answer`, t3.answer === "", JSON.stringify(t3.answer));

    // ---- the clip is a separate mode from listening ----
    // Regression (PR #3 Codex round 4): sending the clip mid-utterance set busy while the partially
    // collected utterance stayed live, so the next command was spliced onto a stale fragment.
    const clipGating = await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      const before = document.getElementById("runClip").disabled;
      return { beforeDisabled: before };
    })()`);
    check(`${name}: the clip is enabled when not listening`, clipGating.beforeDisabled === false);

    // ---- a refused tool must not be announced as having run ----
    // Regression (PR #3 Codex round 5): the live region said "<tool> ran" regardless of outcome, so
    // assistive-technology users were told a side effect happened when the call had been rejected.
    await evaluate(sessionId, stubEngine(
      '{"name":"get_time","parameters":{"timezone":"Mars/Olympus"}}',
      "I could not look that up.",
    ));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 4");
    const t4 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a rejected tool is marked failed`, t4.nodes.includes("tool:fail"), t4.nodes.join(" "));
    const announced = await evaluate(sessionId, `document.getElementById("announcer").textContent`);
    check(
      `${name}: a rejected tool is NOT announced as having run`,
      /was not run/i.test(announced) && !/get_time ran/i.test(announced),
      announced,
    );

    // ---- a rejected tool plus an empty second reply must not render "undefined" ----
    // Regression (PR #3 Codex round 7): failed outcomes carry `error`, not `display`, so the
    // fallback produced the literal string "undefined" and announced it as the answer.
    await evaluate(sessionId, stubEngine('{"name":"get_time","parameters":{"timezone":"Mars/Olympus"}}', ""));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 5");
    const t5 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: never renders the string "undefined"`, !/undefined/.test(t5.answer), JSON.stringify(t5.answer));
    check(
      `${name}: an empty reply after a rejected tool falls back to the error`,
      t5.answer.length > 0 || t5.nodes.includes("answer:fail"),
      `${t5.answer} | ${t5.nodes.join(" ")}`,
    );
    const ann5 = await evaluate(sessionId, `document.getElementById("announcer").textContent`);
    check(`${name}: and never announces "undefined"`, !/undefined/.test(ann5), ann5);

    // ---- a multi-call reply must run NOTHING ----
    // Regression (PR #3 Codex round 9): the page took calls[0] and silently dropped the rest, so two
    // requested timers became one started timer and one vanished request, with neither the model nor
    // the visitor told. Every executor mutates page state, so a partial prefix is the worst outcome.
    const timersBefore = await evaluate(sessionId, `document.querySelectorAll("#timers li").length`);
    await evaluate(sessionId, stubEngine(
      '<tool_call>{"name":"start_timer","parameters":{"seconds":60}}</tool_call>' +
        '<tool_call>{"name":"start_timer","parameters":{"seconds":300}}</tool_call>',
      "I can only start one at a time — which first?",
    ));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "turn 6");
    const t6 = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a multi-call reply is marked failed`, t6.nodes.includes("tool:fail"), t6.nodes.join(" "));
    check(
      `${name}: NO timer was started from a multi-call reply`,
      (await evaluate(sessionId, `document.querySelectorAll("#timers li").length`)) === timersBefore,
    );
    check(
      `${name}: the page says both calls were discarded`,
      /none run/i.test(await evaluate(sessionId, `document.querySelector("#turns .turn").textContent`)),
      t6.nodes.join(" "),
    );
    check(`${name}: and still answers rather than dead-ending`, t6.answer.length > 0, t6.answer);

    // ---- releasing the LLM must cancel a microphone startup already in flight ----
    // Regression (PR #3 Codex round 9): cancellation keyed on vadGeneration alone, which releasing
    // Ultravox never advanced, so a startup awaiting the permission prompt went on to open the
    // microphone for a model that no longer existed. Driven here through the same counter the real
    // permission path checks.
    const cancelled = await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      const before = uv.captureGeneration();
      uv.releaseLLM();
      return { before, after: uv.captureGeneration(), listening: uv.state().listening, llmReady: uv.state().ready.llm };
    })()`);
    check(
      `${name}: releasing Ultravox advances the capture generation, cancelling any pending startup`,
      cancelled.after > cancelled.before,
      JSON.stringify(cancelled),
    );
    check(`${name}: and leaves the page not listening`, cancelled.listening === false && cancelled.llmReady === false);

    check(`${name}: six turns logged`, (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 6);
    check(`${name}: still no console errors`, page.errors.length === 0, page.errors.join(" | "));
    await closePage(cdp, page.targetId);
  }

  console.log(`\n===== published origin (${REMOTE_HOST}) =====`);
  const remote = await openPage(cdp, `http://${REMOTE_HOST}:${server.port}/web-ai-showcase/${ROUTE}`);
  await setViewport(cdp, remote.sessionId, DESKTOP);
  await new Promise((r) => setTimeout(r, 2500));
  check(
    "the debug hook is ABSENT on a non-localhost origin",
    (await evaluate(remote.sessionId, `typeof globalThis.__ultravox`)) === "undefined",
  );
  check(
    "the page still renders there",
    (await evaluate(remote.sessionId, `document.querySelectorAll("#toollist li").length`)) === 6,
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

// ---------------------------------------------------------------------------
// Source guards for the three paths this GPU-less, microphone-less runner cannot enter.
// These are NOT proof of behaviour — they are regression guards that the fixed code path is still
// present. The behavioural proof needs the device pass named in the unverified banner.
// ---------------------------------------------------------------------------
console.log("\n===== source guards (not behavioural proof) =====");
{
  const app = readFileSync(new URL("../models/ultravox-audio-llm/app.js", import.meta.url), "utf8");
  const vadWorker = readFileSync(new URL("../models/silero-vad/worker.js", import.meta.url), "utf8");

  // Codex round 8: resetting the endpointer left `pending` full, so the backlog could resume and
  // form a new utterance across the gap the UI had just called discarded.
  const overflow = app.slice(app.indexOf("This device can't keep up") - 1600, app.indexOf("This device can't keep up"));
  check(
    "a VAD overflow purges the queued audio, not just the endpointer",
    /staleVadReplies \+= pending\.length;[\s\S]{0,80}pending\.length = 0;/.test(overflow),
  );

  // Codex round 8: releasing the LLM left the microphone open, so every completed utterance fell
  // into the "not on this device" error.
  const llmDispose = app.slice(app.indexOf("onDispose: () => {\n    ready.llm = false;"));
  check(
    "releasing Ultravox also stops the microphone",
    /ready\.llm = false;[\s\S]{0,400}if \(listening\) stopListening\(\);/.test(llmDispose.slice(0, 600)),
  );

  // Codex round 8: a stream-reset off the serialisation queue could acknowledge a fresh session
  // while an older queued inference wrote the previous session's recurrent state back over it.
  const reset = vadWorker.slice(vadWorker.indexOf('d.type === "stream-reset"'), vadWorker.indexOf('d.type === "stream-chunk"'));
  check(
    "stream-reset joins the same serialisation queue as stream-chunk",
    /streamTail = streamTail[\s\S]{0,200}streamReset\(\)/.test(reset),
    reset.replace(/\s+/g, " ").slice(0, 120),
  );
  check("stream-reset acknowledges only after the queued reset runs", /\.then\(\(\) => post\(\{ type: "stream-ready"/.test(reset));
}

console.log(`\n${checks - failed}/${checks} checks passed`);
process.exit(failed ? 1 : 0);
