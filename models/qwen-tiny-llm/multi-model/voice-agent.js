// The voice-agent loop: mic → Silero VAD endpointing → Whisper → Qwen tool call → real tool → Qwen answer.
//
// Every model runs in its own worker (Silero on onnxruntime-web, Whisper and Qwen on Transformers.js),
// so the only work on this thread is the mic tap's linear resample, the meter draw, and DOM updates.
// Nothing here fabricates output: if a stage fails, or the model declines to call a tool, the turn
// records that plainly.

import { LiveMic, VadEngine } from "../../silero-vad/vad.js";
import { urlToMono16k, WhisperEngine } from "../../whisper-speech-to-text/whisper.js";
import { escapeHTML, QwenEngine } from "../qwen.js";
import { createModelLoader } from "/web-ai-showcase/lib/model-loader.js";
import {
  parseToolCalls,
  runTool,
  stripToolCalls,
  SYSTEM_PROMPT,
  TOOL_SCHEMAS,
  toolMessageContent,
} from "./tools.js";

const $ = (id) => document.getElementById(id);

// --- endpointing constants (frames are 512 samples at 16 kHz = 32 ms) ---
const FRAME_SEC = 512 / 16000;
const START_PROB = 0.6; // enter speech above this…
const STOP_PROB = 0.35; // …and only leave it below this (hysteresis, so a breath doesn't cut you off)
const START_FRAMES = 2; // ~64 ms of speech before we commit
const HANG_SEC = 0.8; // silence that ends the turn
const PREROLL_SEC = 0.4; // keep this much audio from before the trigger, so the first word survives
const MAX_UTTER_SEC = 15; // hard cap — a bounded buffer, never an unbounded one
const MIN_UTTER_SEC = 0.35; // shorter than this is a cough, not a command

const engines = {
  vad: new VadEngine(),
  asr: new WhisperEngine(),
  llm: new QwenEngine(),
};

const ready = { vad: false, asr: false, llm: false };
let llmDevice = null; // the real backend, learned from the loader — never assumed
let asrDevice = null;
let currentModelId = "onnx-community/Qwen2.5-0.5B-Instruct";
let busy = false; // a turn is in flight — we don't start another until it finishes
let listening = false; // declared up here: updateControls() runs before the mic section is reached

// ---------------------------------------------------------------------------
// Tool-side page state (what the executors actually mutate)
// ---------------------------------------------------------------------------
const notes = [];
const timers = [];
let timerSeq = 0;

function renderNotes() {
  const list = $("notes");
  list.replaceChildren();
  for (const n of notes) {
    const li = document.createElement("li");
    li.textContent = n;
    list.append(li);
  }
  $("notesEmpty").hidden = notes.length > 0;
}

function renderTimers() {
  const box = $("timers");
  box.replaceChildren();
  for (const t of timers) {
    const left = Math.max(0, Math.ceil((t.endsAt - performance.now()) / 1000));
    const row = document.createElement("div");
    row.className = "timer";
    row.dataset.done = left === 0 ? "1" : "0";
    const name = document.createElement("span");
    name.textContent = t.label;
    const clock = document.createElement("span");
    const m = Math.floor(left / 60);
    clock.textContent = left === 0 ? "done ✓" : `${m}:${String(left - m * 60).padStart(2, "0")}`;
    row.append(name, clock);
    box.append(row);
  }
  $("timersEmpty").hidden = timers.length > 0;
}

function startTimer(seconds, label) {
  const id = `t${++timerSeq}`;
  timers.push({ id, label, endsAt: performance.now() + seconds * 1000 });
  renderTimers();
  return id;
}

// One shared 1 Hz tick for every countdown — cheaper and steadier than a timer per timer.
setInterval(() => {
  if (timers.length) renderTimers();
}, 1000);

const toolCtx = {
  notes,
  startTimer,
  onNotesChanged: renderNotes,
};

for (const schema of TOOL_SCHEMAS) {
  const li = document.createElement("li");
  li.textContent = schema.function.name;
  $("toollist").append(li);
}

// ---------------------------------------------------------------------------
// Model loading — three independent auto-init loaders (shared loader UX)
// ---------------------------------------------------------------------------
function reportReadiness() {
  const count = Number(ready.vad) + Number(ready.asr) + Number(ready.llm);
  const el = $("status");
  el.classList.toggle("ok", count === 3);
  if (count === 3) {
    el.textContent = "All three models ready — press Start listening, or type a command.";
  } else {
    const missing = [!ready.vad && "Silero VAD", !ready.asr && "Whisper", !ready.llm && "Qwen"]
      .filter(Boolean).join(", ");
    el.textContent = `${count}/3 models ready · still preparing ${missing}…`;
  }
  updateControls();
}

function updateControls() {
  const canListen = ready.vad && ready.asr && ready.llm && LiveMic.supported() && !busy;
  $("listen").disabled = !canListen && !listening;
  $("typedGo").disabled = !ready.llm || busy;
  $("runClip").disabled = !(ready.vad && ready.asr && ready.llm) || busy;
  for (const b of $("examples").querySelectorAll("button")) b.disabled = !ready.llm || busy;
}

createModelLoader({
  mount: $("loader-vad"),
  model: {
    modelId: "onnx-community/silero-vad",
    runtime: "onnxruntime-web",
    dtype: "fp32",
    sizeMB: 2,
    requiresWebGPU: false,
  },
  load: (onProgress) => engines.vad.load(onProgress),
  onReady: () => {
    ready.vad = true;
    reportReadiness();
  },
});

createModelLoader({
  mount: $("loader-asr"),
  model: {
    modelId: "onnx-community/whisper-base_timestamped",
    runtime: "transformers.js",
    dtype: "q4 WebGPU / q8 WASM",
    sizeMB: 120,
    requiresWebGPU: false,
  },
  load: async (onProgress) => ({ device: await engines.asr.load(onProgress) }),
  onReady: (r) => {
    ready.asr = true;
    asrDevice = r?.device ?? "wasm";
    reportReadiness();
  },
});

function mountLlmLoader(modelId) {
  currentModelId = modelId;
  ready.llm = false;
  reportReadiness();
  createModelLoader({
    mount: $("loader-llm"),
    model: {
      modelId,
      runtime: "transformers.js",
      dtype: "q4f16",
      sizeMB: modelId.includes("1.5B") ? 1222 : 483,
      requiresWebGPU: true,
    },
    load: (onProgress) => engines.llm.load(onProgress, { modelId }),
    onReady: (device) => {
      ready.llm = true;
      llmDevice = device ? String(device) : null;
      reportReadiness();
    },
  });
}
mountLlmLoader(currentModelId);

$("modelSize").addEventListener("change", (e) => {
  if (busy) return;
  mountLlmLoader(e.target.value);
});

// ---------------------------------------------------------------------------
// The live meter — mic level bars with the VAD probability drawn over them
// ---------------------------------------------------------------------------
const HISTORY = 240; // ~7.7 s of 32 ms frames
const levels = new Float32Array(HISTORY);
const probs = new Float32Array(HISTORY);
let histLen = 0;
let meterQueued = false;

function pushFrame(level, prob) {
  if (histLen < HISTORY) {
    levels[histLen] = level;
    probs[histLen] = prob;
    histLen++;
  } else {
    levels.copyWithin(0, 1);
    probs.copyWithin(0, 1);
    levels[HISTORY - 1] = level;
    probs[HISTORY - 1] = prob;
  }
  // Coalesce to one draw per animation frame — the VAD emits ~31 frames/second per chunk and we
  // must not repaint per frame.
  if (!meterQueued) {
    meterQueued = true;
    requestAnimationFrame(() => {
      meterQueued = false;
      drawMeter();
    });
  }
}

function drawMeter() {
  const canvas = $("meter");
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
  const good = cs.getPropertyValue("--good").trim() || "#2a7";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 110;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // The threshold rule is drawn even when idle — an empty box reads as broken, an axis reads as ready.
  const drawThreshold = () => {
    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = accent;
    ctx.beginPath();
    const ty = h - START_PROB * (h - 6) - 3;
    ctx.moveTo(0, ty);
    ctx.lineTo(w, ty);
    ctx.stroke();
    ctx.restore();
  };
  if (!histLen) {
    drawThreshold();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = muted;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("waiting for audio", w / 2, h / 2);
    ctx.globalAlpha = 1;
    return;
  }

  const step = w / HISTORY;
  // level bars
  for (let i = 0; i < histLen; i++) {
    const x = i * step;
    const amp = Math.min(1, levels[i] * 4) * (h * 0.45);
    ctx.strokeStyle = probs[i] >= START_PROB ? good : muted;
    ctx.globalAlpha = probs[i] >= START_PROB ? 0.9 : 0.45;
    ctx.lineWidth = Math.max(1, step * 0.8);
    ctx.beginPath();
    ctx.moveTo(x + step / 2, h / 2 - amp);
    ctx.lineTo(x + step / 2, h / 2 + amp);
    ctx.stroke();
  }
  // probability line
  ctx.globalAlpha = 1;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < histLen; i++) {
    const x = i * step + step / 2;
    const y = h - probs[i] * (h - 6) - 3;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  drawThreshold();
}
addEventListener("resize", drawMeter);

/** Backend names are only ever printed once a model has actually told us which one it loaded on. */
function backendLabel(dev) {
  return dev ? String(dev).toUpperCase() : "–";
}

function setPhase(text, kind = "0") {
  const el = $("phase");
  el.textContent = text;
  el.dataset.on = kind;
}

// ---------------------------------------------------------------------------
// Mic + endpointing
// ---------------------------------------------------------------------------
let mic = null;
const pending = []; // chunks submitted to the VAD worker, awaiting their probabilities (FIFO)
const MAX_PENDING = 64; // if the worker stalls we drop audio rather than growing without bound

// Utterance accumulation
let inSpeech = false;
let startRun = 0;
let silentFrames = 0;
let utterance = [];
let utterLen = 0;
const preroll = []; // bounded ring of recent frames, so we keep the audio before the trigger
let prerollLen = 0;
const PREROLL_SAMPLES = Math.round(PREROLL_SEC * 16000);
const MAX_UTTER_SAMPLES = Math.round(MAX_UTTER_SEC * 16000);

function resetEndpointer() {
  inSpeech = false;
  startRun = 0;
  silentFrames = 0;
  utterance = [];
  utterLen = 0;
  preroll.length = 0;
  prerollLen = 0;
}

function pushPreroll(frame) {
  preroll.push(frame);
  prerollLen += frame.length;
  while (prerollLen - preroll[0].length >= PREROLL_SAMPLES) {
    prerollLen -= preroll.shift().length;
  }
}

function concat(chunks, total) {
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

engines.vad.onStream = (msg) => {
  const chunk = pending.shift();
  if (!chunk) return;
  const p = msg.probs;
  for (let f = 0; f < p.length; f++) {
    const frame = chunk.subarray(f * 512, f * 512 + 512);
    let sumSq = 0;
    for (let i = 0; i < frame.length; i++) sumSq += frame[i] * frame[i];
    pushFrame(Math.sqrt(sumSq / Math.max(1, frame.length)), p[f]);

    if (busy) continue; // meter keeps moving, but we don't collect a new turn mid-turn

    if (!inSpeech) {
      pushPreroll(frame.slice());
      startRun = p[f] >= START_PROB ? startRun + 1 : 0;
      if (startRun >= START_FRAMES) {
        inSpeech = true;
        silentFrames = 0;
        utterance = preroll.slice();
        utterLen = prerollLen;
        preroll.length = 0;
        prerollLen = 0;
        setPhase("hearing you", "speech");
      }
    } else {
      utterance.push(frame.slice());
      utterLen += frame.length;
      silentFrames = p[f] < STOP_PROB ? silentFrames + 1 : 0;
      const hangReached = silentFrames * FRAME_SEC >= HANG_SEC;
      const tooLong = utterLen >= MAX_UTTER_SAMPLES;
      if (hangReached || tooLong) {
        const pcm = concat(utterance, utterLen);
        resetEndpointer();
        const seconds = pcm.length / 16000;
        if (seconds < MIN_UTTER_SEC) {
          setPhase("listening", "1");
        } else {
          runTurn({ audio: pcm, seconds, truncated: tooLong });
        }
      }
    }
  }
};

async function startListening() {
  if (!LiveMic.supported()) {
    $("micFallback").hidden = false;
    return;
  }
  try {
    await engines.vad.streamReset();
  } catch (err) {
    $("status").textContent = `Couldn't reset the VAD: ${err.message}`;
    $("status").classList.add("err");
    return;
  }
  mic = new LiveMic({
    onFrames: (frames) => {
      // frames is always a whole number of 512-sample frames. Keep our own copy for the utterance
      // buffer; the engine transfers its own copy to the worker.
      if (pending.length >= MAX_PENDING) {
        pending.shift(); // the VAD is behind — drop the oldest rather than grow without bound
        $("micNote").textContent = "Dropping audio — the VAD worker is behind on this device.";
      }
      pending.push(frames);
      engines.vad.streamChunk(frames);
    },
  });
  try {
    await mic.start();
  } catch (err) {
    mic = null;
    $("micFallback").hidden = false;
    $("micNote").textContent = `Microphone unavailable (${err.name || "error"}).`;
    return;
  }
  listening = true;
  resetEndpointer();
  $("listen").innerHTML = '<span class="rec-dot"></span>Stop listening';
  $("listen").classList.remove("secondary");
  $("micNote").textContent = "Mic is open. Speak a command, then pause.";
  setPhase("listening", "1");
  updateControls();
}

function stopListening() {
  listening = false;
  try {
    mic?.stop();
  } catch { /* already gone */ }
  mic = null;
  pending.length = 0;
  resetEndpointer();
  $("listen").textContent = "🎙️ Start listening";
  $("micNote").textContent = "Mic closed.";
  setPhase("idle", "0");
  updateControls();
}

$("listen").addEventListener("click", () => (listening ? stopListening() : startListening()));
if (!LiveMic.supported()) {
  $("micFallback").hidden = false;
  $("micNote").textContent = "No microphone API in this browser.";
}
$("hangLabel").textContent = `${HANG_SEC.toFixed(1)} s`;
addEventListener("pagehide", () => {
  try {
    mic?.stop();
  } catch { /* noop */ }
});

// ---------------------------------------------------------------------------
// A turn: transcribe → choose a tool → run it → answer
// ---------------------------------------------------------------------------
const STAGES = [
  ["vad", "Silero VAD"],
  ["asr", "Whisper"],
  ["llm1", "Qwen · pick tool"],
  ["tool", "tool"],
  ["llm2", "Qwen · answer"],
];

let turnSeq = 0;

function newTurnCard() {
  $("turnsEmpty").hidden = true;
  const li = document.createElement("li");
  li.className = "turn";
  li.innerHTML = `
    <h3>Turn ${++turnSeq}</h3>
    <p class="heard" data-role="heard"><em class="muted">listening back…</em></p>
    <div class="flow" data-role="flow"></div>
    <div data-role="body"></div>
    <p class="answer" data-role="answer"></p>
    <p class="readout" data-role="readout"></p>`;
  const flow = li.querySelector('[data-role="flow"]');
  STAGES.forEach(([key, label], i) => {
    if (i) {
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "→";
      flow.append(arrow);
    }
    const node = document.createElement("span");
    node.className = "node";
    node.dataset.key = key;
    node.dataset.state = "pending";
    node.textContent = label;
    flow.append(node);
  });
  $("turns").prepend(li);
  return {
    el: li,
    heard: li.querySelector('[data-role="heard"]'),
    body: li.querySelector('[data-role="body"]'),
    answer: li.querySelector('[data-role="answer"]'),
    readout: li.querySelector('[data-role="readout"]'),
    stage(key, state, label) {
      const n = flow.querySelector(`[data-key="${key}"]`);
      if (!n) return;
      n.dataset.state = state;
      if (label) n.textContent = label;
    },
  };
}

function addBlock(card, title, text, mono = true) {
  const wrap = document.createElement("div");
  const h = document.createElement("p");
  h.className = "muted";
  h.style.cssText = "margin:.5rem 0 .15rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em";
  h.textContent = title;
  const body = document.createElement(mono ? "pre" : "p");
  if (mono) body.className = "call";
  body.textContent = text;
  wrap.append(h, body);
  card.body.append(wrap);
  return body;
}

function addInside(card, title, text) {
  const d = document.createElement("details");
  d.className = "inside";
  const s = document.createElement("summary");
  s.textContent = title;
  const pre = document.createElement("pre");
  pre.className = "tmpl";
  pre.textContent = text;
  d.append(s, pre);
  card.body.append(d);
}

function readout(card, parts) {
  card.readout.innerHTML = parts
    .filter(Boolean)
    .map(([k, v]) => `<span>${escapeHTML(k)} <b>${escapeHTML(String(v))}</b></span>`)
    .join("");
}

/** One end-to-end turn. `audio` may be null for a typed command (the ASR stage is then skipped). */
async function runTurn({ audio, seconds = 0, text = null, truncated = false, source = "mic" }) {
  if (busy) return;
  busy = true;
  updateControls();
  const card = newTurnCard();
  const timings = {};
  const t0 = performance.now();

  try {
    // --- stage 1/2: what did they say ---
    let heard = text;
    if (audio) {
      if (source === "clip") {
        // The clip didn't arrive through the live endpointer, so run Silero over it for real rather
        // than labelling a stage we skipped.
        card.stage("vad", "pending", "Silero…");
        const v0 = performance.now();
        const vad = await engines.vad.run(audio);
        timings.vad = Math.round(performance.now() - v0);
        card.stage("vad", "done", `VAD ${vad.segments.length} seg`);
        addBlock(
          card,
          "Silero VAD over the clip",
          `${vad.segments.length} speech segment(s), ${(vad.speechRatio * 100).toFixed(0)}% speech, ` +
            `${timings.vad} ms on ${vad.device}`,
        );
      } else {
        card.stage("vad", "done", `endpointed ${seconds.toFixed(1)}s`);
      }
      card.stage("asr", "pending", "Whisper…");
      setPhase("transcribing", "1");
      const a0 = performance.now();
      const asr = await engines.asr.transcribe(audio);
      timings.asr = Math.round(performance.now() - a0);
      heard = (asr.text || "").trim();
      asrDevice = asr.device || asrDevice;
      if (!heard || !/[a-z0-9]/i.test(heard)) {
        card.stage("asr", "fail", "Whisper — nothing");
        card.heard.innerHTML = '<em class="muted">Whisper heard no words in that clip.</em>';
        readout(card, [["asr", `${timings.asr} ms`], ["backend", backendLabel(asrDevice)]]);
        return;
      }
      card.stage("asr", "done", `Whisper ${timings.asr} ms`);
    } else {
      card.stage("vad", "done", "typed — no audio");
      card.stage("asr", "done", "typed — no audio");
    }
    card.heard.textContent = `“${heard}”`;
    if (truncated) {
      const warn = document.createElement("span");
      warn.className = "muted";
      warn.style.fontSize = ".8rem";
      warn.textContent = ` (cut off at the ${MAX_UTTER_SEC}s cap)`;
      card.heard.append(warn);
    }

    // --- stage 3: which tool ---
    setPhase("thinking", "1");
    card.stage("llm1", "pending", "Qwen…");
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: heard },
    ];
    let template1 = "";
    const l0 = performance.now();
    const first = await engines.llm.chat(messages, {
      tools: TOOL_SCHEMAS,
      modelId: currentModelId,
      doSample: false, // greedy — sampling makes small models mangle the JSON
      maxTokens: 160,
      onPrompt: (t) => (template1 = t),
    });
    timings.llm1 = Math.round(performance.now() - l0);
    const rawFirst = (first.text || "").trim();
    card.stage("llm1", "done", `Qwen ${timings.llm1} ms`);
    addBlock(card, "what the model produced", rawFirst || "(nothing)");
    addInside(card, "the exact prompt it saw (chat template + tools block)", template1);

    const calls = parseToolCalls(rawFirst);
    if (!calls.length) {
      // A real, honest outcome — not an error. The model answered from its own knowledge.
      card.stage("tool", "fail", "no tool call");
      card.stage("llm2", "done", "direct answer");
      const direct = stripToolCalls(rawFirst) || rawFirst;
      card.answer.textContent = direct;
      const note = document.createElement("p");
      note.className = "muted";
      note.style.fontSize = ".82rem";
      note.textContent =
        "No tool call in that reply — the model answered directly. That's a real outcome for a 0.5B model, " +
        "not a page error; the larger checkpoint above calls tools more often.";
      card.body.append(note);
      readout(card, [
        audio && ["asr", `${timings.asr} ms`],
        ["llm", `${timings.llm1} ms`],
        ["tokens", first.tokens],
        ["backend", backendLabel(llmDevice)],
        ["total", `${Math.round(performance.now() - t0)} ms`],
      ]);
      return;
    }

    // --- stage 4: run it for real ---
    const call = calls[0];
    card.stage("tool", "pending", call.name);
    addBlock(card, "parsed tool call", JSON.stringify(call, null, 2));
    const outcome = runTool(call, toolCtx);
    timings.tool = outcome.ms;
    card.stage("tool", outcome.ok ? "done" : "fail", `${call.name}${outcome.ok ? " ✓" : " ✗"}`);
    addBlock(
      card,
      outcome.ok ? "what the tool returned" : "the tool failed (the model is told, and explains)",
      outcome.ok ? `${outcome.display}\n${JSON.stringify(outcome.result)}` : outcome.error,
    );
    if (call.name === "calculate" && outcome.ok) $("lastCalc").textContent = outcome.display;

    // --- stage 5: answer with the result in hand ---
    card.stage("llm2", "pending", "Qwen…");
    const followUp = [
      ...messages,
      { role: "assistant", tool_calls: [{ type: "function", function: call }] },
      { role: "tool", content: toolMessageContent(outcome) },
    ];
    let streamed = "";
    const l1 = performance.now();
    const second = await engines.llm.chat(followUp, {
      tools: TOOL_SCHEMAS,
      modelId: currentModelId,
      doSample: false,
      maxTokens: 120,
      onToken: (tok) => {
        streamed += tok;
        card.answer.innerHTML = `${escapeHTML(streamed)}<span class="caret"></span>`;
      },
    });
    timings.llm2 = Math.round(performance.now() - l1);
    const answer = stripToolCalls(second.text || streamed).trim() ||
      outcome.display ||
      "(the model returned nothing on the second pass)";
    card.answer.textContent = answer;
    card.stage("llm2", "done", `Qwen ${timings.llm2} ms`);
    readout(card, [
      audio && ["heard", `${seconds.toFixed(1)} s`],
      timings.vad != null && ["silero", `${timings.vad} ms`],
      audio && ["whisper", `${timings.asr} ms · ${backendLabel(asrDevice)}`],
      ["qwen pick", `${timings.llm1} ms`],
      ["tool", `${timings.tool} ms`],
      ["qwen answer", `${timings.llm2} ms`],
      ["backend", backendLabel(llmDevice)],
      ["model", currentModelId.split("/").pop()],
      ["total", `${Math.round(performance.now() - t0)} ms`],
    ]);
  } catch (err) {
    // Errors go on the page, never only the console.
    card.answer.innerHTML = "";
    const p = document.createElement("p");
    p.className = "status err";
    p.textContent = `That turn failed: ${err?.message ?? err}`;
    card.body.append(p);
    for (const [key] of STAGES) {
      const n = card.el.querySelector(`[data-key="${key}"]`);
      if (n && n.dataset.state === "pending") n.dataset.state = "fail";
    }
  } finally {
    busy = false;
    setPhase(listening ? "listening" : "idle", listening ? "1" : "0");
    updateControls();
  }
}

// ---------------------------------------------------------------------------
// The two no-mic entry points
// ---------------------------------------------------------------------------
function sendTyped(value) {
  const cmd = String(value ?? "").trim();
  if (!cmd) return;
  runTurn({ text: cmd, source: "typed" });
}
$("typedGo").addEventListener("click", () => sendTyped($("typedCmd").value));
$("typedCmd").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendTyped($("typedCmd").value);
  }
});
for (const b of $("examples").querySelectorAll("button")) {
  b.addEventListener("click", () => {
    $("typedCmd").value = b.dataset.cmd;
    sendTyped(b.dataset.cmd);
  });
}

$("runClip").addEventListener("click", async () => {
  if (busy) return;
  $("runClip").disabled = true;
  try {
    const { pcm } = await urlToMono16k("../../whisper-speech-to-text/jfk.wav");
    await runTurn({ audio: pcm, seconds: pcm.length / 16000, source: "clip" });
  } catch (err) {
    $("status").textContent = `Couldn't decode the bundled clip: ${err.message}`;
    $("status").classList.add("err");
  } finally {
    updateControls();
  }
});

renderNotes();
renderTimers();
updateControls();
drawMeter();

// Lets scripts/validate-qwen-voice-agent.mjs drive a whole turn with substituted engines, so the
// parse → real-tool-execution → render path is covered on a machine that can't download 600 MB of
// weights.
//
// LOCAL ONLY. markReady() flips the readiness flags without a loader having actually finished, which
// is exactly the "ready" lie the honest-state invariant exists to prevent — so the hook is never
// attached on the published origin. The validator serves from 127.0.0.1, which is where it belongs.
function isLocalHost() {
  const h = location.hostname;
  return h === "127.0.0.1" || h === "localhost" || h === "[::1]" || h === "" || h === "::1";
}
if (isLocalHost()) {
  globalThis.__voiceAgent = {
    runTurn,
    engines,
    toolCtx,
    state: () => ({
      ready: { ...ready },
      busy,
      listening,
      currentModelId,
      notes: [...notes],
      timers: timers.length,
    }),
    markReady: (which) => {
      ready[which] = true;
      reportReadiness();
    },
  };
}
