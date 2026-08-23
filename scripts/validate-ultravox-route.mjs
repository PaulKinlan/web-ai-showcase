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

    check(`${name}: two turns logged`, (await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`)) === 2);
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

console.log(`\n${checks - failed}/${checks} checks passed`);
process.exit(failed ? 1 : 0);
