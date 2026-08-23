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

// announce() clears the live region and re-fills it on the next frame, so reading it the instant a
// turn ends can catch the empty gap. Waiting for non-empty text also stops these assertions passing
// vacuously — a `!/undefined/` check is trivially true of "".
async function announcement(sessionId) {
  await waitFor(sessionId, `document.getElementById("announcer").textContent.trim().length > 0`, 5000, "an announcement");
  return await evaluate(sessionId, `document.getElementById("announcer").textContent`);
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
      globalThis.__lastMessages = opts.messages;
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
    extraArgs: [
      `--host-resolver-rules=MAP ${REMOTE_HOST} 127.0.0.1`,
      // A fake capture device, auto-granted. This is what lets the microphone path be driven for
      // real — LiveMic, its resample, and the endpointer's overflow handling all run as written,
      // with only the VAD transport stubbed (no model can be fetched here).
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required",
    ],
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
    const announced = await announcement(sessionId);
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
    const ann5 = await announcement(sessionId);
    check(`${name}: and never announces "undefined"`, !/undefined/.test(ann5), ann5);

    // ---- an empty reply after a SUCCESSFUL tool must not be dressed up as the model's answer ----
    // Regression (PR #3 Codex round 10): every successful tool supplies a non-empty display line, so
    // an empty second generation silently became "answer: done" showing text the model never
    // produced. The tool result is still worth showing — it just must not be presented as the reply.
    await evaluate(sessionId, stubEngine('{"name":"calculate","parameters":{"expression":"6 * 7"}}', ""));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "empty answer after a successful tool");
    const tEmpty = await evaluate(sessionId, turnSnapshot);
    const emptySource = await evaluate(
      sessionId,
      `document.querySelector("#turns .turn [data-role=answer]")?.dataset.source ?? ""`,
    );
    check(
      `${name}: an empty reply after a SUCCESSFUL tool is NOT marked done`,
      tEmpty.nodes.includes("answer:fail"),
      tEmpty.nodes.join(" "),
    );
    check(
      `${name}: the tool result is labelled as the tool's output, not the model's words`,
      emptySource === "tool" && /not the model's words/.test(tEmpty.body),
      `${emptySource} | ${tEmpty.body.slice(0, 140)}`,
    );
    check(`${name}: the answer itself is still shown`, /42/.test(tEmpty.answer), tEmpty.answer);
    const annEmpty = await announcement(sessionId);
    check(
      `${name}: and it is not announced as the model's reply`,
      /no reply/i.test(annEmpty),
      annEmpty,
    );

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

    // ---- capture must not resume mid-sentence after a busy turn ----
    // Regression (PR #3 Codex round 10): frames arriving while a turn generated were discarded with
    // the UI still saying "Mic is open". Speech that began during that gap resumed mid-sentence, so
    // its TAIL could be endpointed as a whole command — "…for five minutes" arriving as the request.
    // Driven through the real endpointer with real probabilities; only the model output is stubbed.
    await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      // The release test above tore the LLM down, so re-arm it before driving another turn.
      uv.markReady("llm");
      uv.markReady("vad");
      uv.engines.llm.generate = async () => {
        await new Promise((r) => setTimeout(r, 1500));
        return { text: "Done.", prepMs: 10, genMs: 20, promptTokens: 100, audioFrames: 8, newTokens: 3, device: "webgpu" };
      };
      uv.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" });
      return true;
    })()`);
    await waitFor(sessionId, `globalThis.__ultravox.state().busy`, 5000, "the slow turn to start");
    // Speech arriving mid-generation: recorded as a hole, never accumulated.
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(10).fill(0.95))`);
    const during = await evaluate(sessionId, `globalThis.__ultravox.endpointer()`);
    check(
      `${name}: speech during a busy turn is not collected`,
      during.awaitingResync === true && during.inSpeech === false && during.utterLen === 0,
      JSON.stringify(during),
    );
    check(
      `${name}: and the mic note says so instead of "Mic is open"`,
      /isn't collected/i.test(await evaluate(sessionId, `document.getElementById("micNote").textContent`)),
      await evaluate(sessionId, `document.getElementById("micNote").textContent`),
    );
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 15_000, "the slow turn to finish");
    // The visitor is still talking as the turn ends. The tail must NOT become a new utterance.
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(20).fill(0.95))`);
    const tail = await evaluate(sessionId, `globalThis.__ultravox.endpointer()`);
    check(
      `${name}: a sentence already in progress is NOT picked up mid-way`,
      tail.inSpeech === false && tail.utterLen === 0 && tail.awaitingResync === true,
      JSON.stringify(tail),
    );
    // A real pause re-syncs, and normal endpointing resumes from the next sentence.
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(30).fill(0.02))`);
    const synced = await evaluate(sessionId, `globalThis.__ultravox.endpointer()`);
    check(`${name}: a genuine silence gap re-syncs capture`, synced.awaitingResync === false, JSON.stringify(synced));
    check(
      `${name}: and the mic note goes back to inviting speech`,
      /Mic is open/i.test(await evaluate(sessionId, `document.getElementById("micNote").textContent`)),
    );
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(6).fill(0.95))`);
    const resumed = await evaluate(sessionId, `globalThis.__ultravox.endpointer()`);
    check(
      `${name}: the NEXT sentence is collected normally`,
      resumed.inSpeech === true && resumed.utterLen > 0,
      JSON.stringify(resumed),
    );

    // ---- a packed multi-call wrapper must also run NOTHING ----
    // Regression (PR #3 Codex round 11): two calls concatenated inside ONE <tool_call> wrapper
    // parsed as a single call, so the page executed it and never reached the round-9 refusal.
    const timersBeforePacked = await evaluate(sessionId, `document.querySelectorAll("#timers .timer").length`);
    await evaluate(sessionId, stubEngine(
      '<tool_call>{"name":"start_timer","parameters":{"seconds":60}}' +
        '{"name":"start_timer","parameters":{"seconds":300}}</tool_call>',
      "One at a time, please.",
    ));
    await evaluate(sessionId, `globalThis.__ultravox.runTurn({ audio: new Float32Array(16000), seconds: 1, source: "clip" })`);
    await waitFor(sessionId, `!globalThis.__ultravox.state().busy`, 20_000, "the packed multi-call turn");
    const tPacked = await evaluate(sessionId, turnSnapshot);
    check(`${name}: a PACKED multi-call reply is marked failed`, tPacked.nodes.includes("tool:fail"), tPacked.nodes.join(" "));
    check(
      `${name}: and starts no timer`,
      (await evaluate(sessionId, `document.querySelectorAll("#timers .timer").length`)) === timersBeforePacked,
    );

    // ---- a fatal worker error must leave the engine RECOVERABLE ----
    // Regression (PR #3 Codex round 11): the error handler rejected the waiters but left the dead
    // worker installed, so the loader's Retry posted into a corpse and its promise never settled.
    const recovery = await evaluate(sessionId, `(async () => {
      const { UltravoxEngine } = await import("/web-ai-showcase/models/ultravox-audio-llm/ultravox.js");
      const e = new UltravoxEngine();
      const before = !!e.worker;
      // Simulate the fatal case: a module worker whose graph never starts.
      e._fatal(new Error("Worker failed to start"));
      const afterFatal = { worker: !!e.worker, ready: e.ready };
      // The loader's Retry calls load() again — it must get a FRESH worker, not hang forever.
      const p = e.load();
      const respawned = !!e.worker;
      let settled = "pending";
      await Promise.race([
        p.then(() => (settled = "resolved"), () => (settled = "rejected")),
        new Promise((r) => setTimeout(r, 1200)),
      ]);
      e.dispose();
      return { before, afterFatal, respawned, settled, disposedWorker: !!e.worker };
    })()`, 40_000);
    check(`${name}: a fresh engine starts with a worker`, recovery.before === true);
    check(
      `${name}: a fatal error discards the dead worker and clears ready`,
      recovery.afterFatal.worker === false && recovery.afterFatal.ready === false,
      JSON.stringify(recovery.afterFatal),
    );
    check(`${name}: Retry builds a fresh worker instead of posting into the corpse`, recovery.respawned === true);
    check(`${name}: and that retry actually settles`, recovery.settled !== "pending", recovery.settled);
    check(`${name}: dispose leaves no worker behind`, recovery.disposedWorker === false);

    // ---- generate() on an unloaded engine must fail loudly, not hang ----
    const noModel = await evaluate(sessionId, `(async () => {
      const { UltravoxEngine } = await import("/web-ai-showcase/models/ultravox-audio-llm/ultravox.js");
      const e = new UltravoxEngine();
      try {
        await e.generate({ messages: [], tools: [], maxTokens: 4 });
        return "resolved";
      } catch (err) {
        return String(err.message);
      } finally {
        e.dispose();
      }
    })()`);
    check(`${name}: generate() without a loaded model rejects with a readable reason`, /not loaded/i.test(noModel), noModel);

    // ---- pagehide must cancel a microphone startup still in flight ----
    // Regression (PR #3 Codex round 11): pagehide called stopListening() or mic.stop(), neither of
    // which can cancel an unresolved permission request. With the back-forward cache the page comes
    // BACK, and beginListening() could then pass its cancellation check and mark the restored page
    // as listening over a microphone the visitor never re-authorised.
    const hidden = await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      const before = uv.captureGeneration();
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }));
      return { before, after: uv.captureGeneration(), listening: uv.state().listening };
    })()`);
    check(
      `${name}: pagehide advances the capture generation, invalidating a pending startup`,
      hidden.after > hidden.before,
      JSON.stringify(hidden),
    );
    check(`${name}: and the page is not left listening`, hidden.listening === false);

    // ---- audio captured DURING a turn stays quarantined even if its reply lands late ----
    // Regression (PR #3 Codex round 12): the busy decision was taken when the reply came back, not
    // when the PCM was captured, so on a backlogged device a chunk recorded during the advertised
    // "not collecting" gap could arrive after generation finished and be endpointed as a command.
    // Each queued chunk now carries the state that was true at capture time.
    const lateReply = await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      const before = uv.endpointer();
      // A chunk captured while busy, delivered now that the page is idle.
      uv.feedVad(Array(12).fill(0.95), true);
      return { before, after: uv.endpointer(), busyNow: uv.state().busy };
    })()`);
    check(
      `${name}: a chunk captured while busy is quarantined even when its reply lands late`,
      lateReply.busyNow === false && lateReply.after.inSpeech === false && lateReply.after.utterLen === 0,
      JSON.stringify(lateReply),
    );
    check(
      `${name}: and that late audio puts capture back into resync`,
      lateReply.after.awaitingResync === true,
      JSON.stringify(lateReply.after),
    );
    // Meanwhile a chunk genuinely captured while idle is still collected normally.
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(30).fill(0.02), false)`);
    await evaluate(sessionId, `globalThis.__ultravox.feedVad(Array(6).fill(0.95), false)`);
    const idleChunk = await evaluate(sessionId, `globalThis.__ultravox.endpointer()`);
    check(
      `${name}: audio captured while idle is still collected`,
      idleChunk.inSpeech === true && idleChunk.utterLen > 0,
      JSON.stringify(idleChunk),
    );

    // ---- a reply with no matching queued chunk is simply ignored ----
    // Regression (PR #3 Codex round 12): stale replies were counted, not correlated, so a chunk that
    // FAILED (an error, never a reply) made the counter over-count and every later reply was paired
    // with the previous chunk's PCM. Correlation by id makes an unmatched reply a no-op.
    const orphan = await evaluate(sessionId, `(() => {
      const uv = globalThis.__ultravox;
      const before = uv.endpointer();
      uv.engines.vad.onStream({ id: 999999, probs: Float32Array.from(Array(10).fill(0.95)) });
      return { before, after: uv.endpointer() };
    })()`);
    check(
      `${name}: a reply for a chunk that was never queued changes nothing`,
      JSON.stringify(orphan.before) === JSON.stringify(orphan.after),
      JSON.stringify(orphan),
    );

    // ---- the REAL bundled-clip control, not the debug hook ----
    // Regression (PR #3 Codex round 14): every clip turn above calls runTurn() with a synthetic
    // Float32Array, so #runClip's own path — the urlToMono16k() fetch and decode of the bundled WAV,
    // reading the prompt field, the clipPreparing state transitions — was never exercised, and a
    // missing or broken clip asset would still have produced a green run. Driven for real here; only
    // the model's output is stubbed, as everywhere else.
    await evaluate(sessionId, stubEngine(
      '{"name":"calculate","parameters":{"expression":"2 + 2"}}',
      "That comes to four.",
    ));
    const clipRun = await evaluate(sessionId, `(async () => {
      const before = document.querySelectorAll("#turns .turn").length;
      // A prompt containing the reserved placeholder the page documents on screen: the page must
      // strip it rather than hand the processor two placeholders for one recording.
      document.getElementById("clipPrompt").value = "What is 2 + 2? <|audio|>";
      const fetched = [];
      const realFetch = window.fetch;
      window.fetch = (...args) => { fetched.push(String(args[0])); return realFetch.apply(window, args); };
      document.getElementById("runClip").click();
      const deadline = Date.now() + 30000;
      while (Date.now() < deadline && document.querySelectorAll("#turns .turn").length === before) {
        await new Promise((r) => setTimeout(r, 150));
      }
      while (Date.now() < deadline && globalThis.__ultravox.state().busy) {
        await new Promise((r) => setTimeout(r, 150));
      }
      window.fetch = realFetch;
      const turn = document.querySelector("#turns .turn");
      return {
        added: document.querySelectorAll("#turns .turn").length - before,
        fetchedClip: fetched.some((u) => /jfk\\.wav$/.test(u)),
        userContent: (globalThis.__lastMessages ?? []).find((m) => m.role === "user")?.content ?? null,
        answer: turn?.querySelector('[data-role="answer"]')?.textContent.trim() ?? "",
        nodes: turn ? [...turn.querySelectorAll(".flow .node")].map((n) => n.dataset.key + ":" + n.dataset.state) : [],
        clipEnabledAgain: !document.getElementById("runClip").disabled,
      };
    })()`, 60_000);
    check(`${name}: clicking the real clip button runs a turn`, clipRun.added === 1, JSON.stringify(clipRun.nodes));
    check(
      `${name}: it actually fetched and decoded the bundled WAV`,
      clipRun.fetchedClip === true,
      clipRun.fetchedClip ? "" : "no request for jfk.wav was made",
    );
    check(`${name}: Silero is honestly marked unused for a pre-cut clip`, clipRun.nodes.includes("vad:skipped"), clipRun.nodes.join(" "));
    check(`${name}: the tool ran and the answer rendered`, /four/i.test(clipRun.answer), clipRun.answer);
    // Asserted on the prompt the MODEL was actually given, not on rendered text — a DOM assertion
    // here would pass vacuously if the page happened to render nothing.
    check(
      `${name}: the visitor's question reached the model`,
      /What is 2 \+ 2\?/.test(clipRun.userContent ?? ""),
      JSON.stringify(clipRun.userContent),
    );
    check(
      `${name}: with EXACTLY one audio placeholder — the page's, never the visitor's`,
      (clipRun.userContent ?? "").split("<|audio|>").length - 1 === 1,
      JSON.stringify(clipRun.userContent),
    );
    check(`${name}: the clip control is usable again afterwards`, clipRun.clipEnabledAgain === true);
    await evaluate(sessionId, `document.getElementById("clipPrompt").value = ""`);

    // ---- a suspended audio context is a FAILED start, for every caller ----
    // Regression (PR #3 Codex round 13): start() resolved and only recorded `suspended` as a
    // property, so each caller had to remember to check it — and the published silero-vad/wild
    // route did not: it showed "Live" over a context that will never deliver a frame. Driven here
    // against the shared class with a context that refuses to resume.
    const suspended = await evaluate(sessionId, `(async () => {
      const { LiveMic } = await import("/web-ai-showcase/models/silero-vad/vad.js");
      const Real = self.AudioContext;
      class NeverResumes extends Real {
        get state() { return "suspended"; }
        async resume() { /* the autoplay policy wins */ }
      }
      self.AudioContext = NeverResumes;
      try {
        await new LiveMic({ onFrames() {} }).start();
        return { outcome: "resolved" };
      } catch (err) {
        return { outcome: "threw", name: err.name, message: err.message };
      } finally {
        self.AudioContext = Real;
      }
    })()`, 40_000);
    check(
      `${name}: LiveMic.start() FAILS on a context that stays suspended`,
      suspended.outcome === "threw",
      JSON.stringify(suspended),
    );
    check(
      `${name}: with a name callers can branch on`,
      suspended.name === "AudioContextSuspendedError",
      suspended.name,
    );
    check(
      `${name}: and a message that tells the visitor what to do`,
      /tap the button again/i.test(suspended.message || ""),
      suspended.message,
    );

    // ---- VAD overflow must drain and reset the WORKER, not just forget the bookkeeping ----
    // Regression (PR #3 Codex round 13): pending.clear() cannot unsend the stream-chunk messages
    // already sitting on the worker's serialised tail — they keep mutating Silero's recurrent state
    // — so accepting new audio immediately could splice the discarded command's tail into a fresh
    // utterance. Driven with a REAL microphone (Chrome's fake capture device) through the real
    // LiveMic and the real overflow branch; only the VAD transport is stubbed, since no model can
    // be fetched in this container.
    const overflow = await evaluate(sessionId, `(async () => {
      const uv = globalThis.__ultravox;
      uv.stopListening();
      let sent = 0, resets = 0, releaseReset = null;
      uv.engines.vad.streamChunk = () => ++sent;          // ids, but never any reply — the backlog grows
      // beginListening() awaits a reset before opening the mic, so only the OVERFLOW-triggered reset
      // (the second call) is held open; holding the first would just stall startup.
      uv.engines.vad.streamReset = () => {
        resets++;
        if (resets === 1) return Promise.resolve();
        return new Promise((r) => (releaseReset = r));
      };
      uv.markReady("llm"); uv.markReady("vad");
      await uv.startListening();
      const deadline = Date.now() + 25000;
      while (Date.now() < deadline && !uv.endpointer().vadResetting) {
        await new Promise((r) => setTimeout(r, 100));
      }
      const atOverflow = { ...uv.endpointer(), sent, resets };
      const sentAtOverflow = sent;
      await new Promise((r) => setTimeout(r, 700));       // frames keep arriving from the fake mic
      const whileResetting = { ...uv.endpointer(), sentSince: sent - sentAtOverflow };
      releaseReset?.();
      await new Promise((r) => setTimeout(r, 700));
      const afterReset = { ...uv.endpointer(), sentSince: sent - sentAtOverflow };
      uv.stopListening();
      return { atOverflow, whileResetting, afterReset };
    })()`, 60_000);
    check(
      `${name}: the real microphone path reaches the overflow branch`,
      overflow.atOverflow?.vadResetting === true && overflow.atOverflow.sent >= 64,
      JSON.stringify(overflow.atOverflow),
    );
    check(
      `${name}: overflow requests a worker stream reset, not just a local clear`,
      overflow.atOverflow?.resets === 2,
      JSON.stringify(overflow.atOverflow),
    );
    check(
      `${name}: NO audio is sent while the reset is queued behind the backlog`,
      overflow.whileResetting?.sentSince === 0,
      JSON.stringify(overflow.whileResetting),
    );
    check(
      `${name}: capture resumes once the reset is acknowledged`,
      overflow.afterReset?.sentSince > 0,
      JSON.stringify(overflow.afterReset),
    );
    check(
      `${name}: and stays quarantined until a genuine silence gap`,
      overflow.afterReset?.awaitingResync === true,
      JSON.stringify(overflow.afterReset),
    );

    // ---- a stalled generation must fail, not hang the page forever ----
    // Regression (PR #3 Codex round 15): generate() had no deadline, so a WebGPU or worker hang left
    // the page permanently busy — controls disabled, microphone audio discarded, and no error
    // anywhere. The deadline is an INACTIVITY one, so a slow-but-alive device is never cut off.
    const stall = await evaluate(sessionId, `(async () => {
      const mod = await import("/web-ai-showcase/models/ultravox-audio-llm/ultravox.js");
      // A worker that accepts messages and never answers — no network needed, and deterministic.
      const RealWorker = self.Worker;
      self.Worker = class { addEventListener() {} postMessage() {} terminate() {} };
      const e = new mod.UltravoxEngine();
      self.Worker = RealWorker;
      e.ready = true;                       // pretend a model is loaded
      const started = Date.now();
      const outcome = await Promise.race([
        e.generate({ messages: [], tools: [], maxTokens: 8 }).then(() => "resolved", (err) => err.name),
        new Promise((r) => setTimeout(() => r("still-pending"), 4000)),
      ]);
      e.dispose();
      return { outcome, waited: Date.now() - started };
    })()`, 40_000);
    check(
      `${name}: a generation that gets no reply does not resolve early or silently`,
      stall.outcome === "still-pending",
      JSON.stringify(stall),
    );

    // ---- timers run on wall-clock time and announce together ----
    // Regression (PR #3 Codex round 14): countdowns measured on performance.now() could stall while
    // the machine slept, and two timers finishing in the same tick each cleared the shared live
    // region on the same frame — so only the last was ever spoken, and both were already flagged as
    // announced, so the lost one was never announced at all.
    const timers = await evaluate(sessionId, `(async () => {
      const uv = globalThis.__ultravox;
      // Two timers that are already due: the tool is driven for real, the clock is not faked.
      uv.toolCtx.startTimer(0.1, "pasta");
      uv.toolCtx.startTimer(0.1, "eggs");
      await new Promise((r) => setTimeout(r, 1400));
      return {
        announced: document.getElementById("announcer").textContent,
        rows: [...document.querySelectorAll("#timers .timer")].map((t) => t.textContent),
        done: [...document.querySelectorAll("#timers .timer")].filter((t) => t.dataset.done === "1").length,
      };
    })()`, 30_000);
    check(`${name}: both timers reach done`, timers.done === 2, JSON.stringify(timers.rows));
    check(
      `${name}: BOTH finished timers are named in one announcement`,
      /pasta/.test(timers.announced) && /eggs/.test(timers.announced),
      timers.announced,
    );
    check(
      `${name}: and it is a single combined message, not two that overwrite each other`,
      /2 timers finished/i.test(timers.announced),
      timers.announced,
    );

    // Counted LAST, so it covers every turn this pass drove — it sat mid-file before and silently
    // excluded anything added after it.
    const turnCount = await evaluate(sessionId, `document.querySelectorAll("#turns .turn").length`);
    check(`${name}: every driven turn is logged`, turnCount === 10, `got ${turnCount}, want 10`);

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
    /pending\.clear\(\);/.test(overflow),
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
  const engine = readFileSync(new URL("../models/ultravox-audio-llm/ultravox.js", import.meta.url), "utf8");
  // Codex round 15: a generation with no deadline left the page permanently busy on a WebGPU hang.
  check("a stall deadline is armed for every generation", /GENERATE_STALL_MS/.test(engine) && /rearm\(\);/.test(engine));
  check(
    "each streamed token re-arms it, so a slow-but-alive device is never cut off",
    /case "token": \{[\s\S]{0,160}rearm\?\.\(\)/.test(engine),
  );
  check(
    "a stalled generation tears the worker down so Retry can recover",
    /GenerationStalledError[\s\S]{0,240}this\._fatal\(err\)/.test(engine),
  );

  check("stream-reset acknowledges only after the queued reset runs", /\.then\(\(\) => post\(\{ type: "stream-ready"/.test(reset));
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
process.exit(failed ? 1 : 0);
