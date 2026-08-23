// The Ultravox loop: mic → Silero VAD endpointing → the LLM HEARS the audio → tool call → answer.
//
// The audio never becomes text. Silero decides only when you stopped talking; the PCM itself goes to
// UltravoxProcessor, which turns it into embedding frames the language model attends to. If you are
// looking for the transcription step, there isn't one — that is the whole point of the page.

import { LiveMic, urlToMono16k, VadEngine } from "../silero-vad/vad.js";
import { escapeHTML, probeWebGPUMain, ULTRAVOX_CSS, UltravoxEngine } from "./ultravox.js";
import { createModelLoader } from "/web-ai-showcase/lib/model-loader.js";
import {
  AUDIO_PLACEHOLDER,
  parseToolCalls,
  runTool,
  stripToolCalls,
  SYSTEM_PROMPT,
  TOOL_SCHEMAS,
  toolMessageContent,
} from "./tools.js";

const style = document.createElement("style");
style.textContent = ULTRAVOX_CSS;
document.head.append(style);

const $ = (id) => document.getElementById(id);

const MODEL_ID = "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX";
// Silero frames are 512 samples at 16 kHz = 32 ms.
const FRAME_SEC = 512 / 16000;
const START_PROB = 0.6;
const STOP_PROB = 0.35;
const START_FRAMES = 2;
const HANG_SEC = 0.8;
const PREROLL_SEC = 0.4;
const MAX_UTTER_SEC = 20;
const MIN_VOICED_SEC = 0.35; // measured on VOICED frames, not on the padded utterance
const PREROLL_SAMPLES = Math.round(PREROLL_SEC * 16000);
const MAX_UTTER_SAMPLES = Math.round(MAX_UTTER_SEC * 16000);

const engines = { llm: new UltravoxEngine(), vad: new VadEngine() };
const ENGINE_FACTORIES = { llm: () => new UltravoxEngine(), vad: () => new VadEngine() };
const ready = { llm: false, vad: false };
// Bumped whenever the VAD engine is replaced. Microphone startup awaits a permission prompt that can
// outlast a Release/Clear, so anything begun before a reset must check it is still the current one.
let vadGeneration = 0;
let device = null;
let busy = false;
let listening = false;
let micStarting = false;
let clipPreparing = false;

function resetEngine(kind, reason = "Model released") {
  try {
    engines[kind]?.dispose?.(reason);
  } catch { /* already gone */ }
  engines[kind] = ENGINE_FACTORIES[kind]();
  ready[kind] = false;
  if (kind === "llm") device = null;
  if (kind === "vad") {
    vadGeneration++;
    staleVadReplies = 0;
    pending.length = 0;
    engines.vad.onStream = onVadStream;
    watchVadErrors();
  }
}

// ---------------------------------------------------------------------------
// Tool-side page state
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
    // One polite announcement on the transition to zero. Making the countdown itself live would read
    // every second aloud; saying nothing means a screen-reader user never learns the timer finished.
    if (left === 0 && !t.announced) {
      t.announced = true;
      announce(`Timer finished: ${t.label}.`);
    }
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

setInterval(() => {
  if (timers.length) renderTimers();
}, 1000);

const toolCtx = { notes, startTimer, onNotesChanged: renderNotes };

for (const schema of TOOL_SCHEMAS) {
  const li = document.createElement("li");
  li.textContent = schema.function.name;
  $("toollist").append(li);
}

// A little static illustration of what the prompt looks like: text tokens, then a run of audio
// positions, then text again.
{
  const viz = $("audiovizDemo");
  const pattern = [..."ttt", ..."a".repeat(26), ..."tt"];
  for (const c of pattern) {
    const el = document.createElement("span");
    if (c === "t") el.className = "txt";
    viz.append(el);
  }
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------
let gpuBlocked = null; // set once the main-thread probe answers

/** What the shared loaders are actually showing right now. */
function loaderStates() {
  return [...document.querySelectorAll(".model-loader")].map((l) => l.dataset.state);
}

function reportReadiness() {
  const count = Number(ready.llm) + Number(ready.vad);
  const el = $("status");
  el.classList.toggle("ok", count === 2);
  if (count === 2) {
    el.textContent = "Both models ready — press Start listening, or send the bundled clip.";
  } else if (
    !gpuBlocked && loaderStates().some((st) => st === "download-required" || st === "partial")
  ) {
    // "Preparing…" is only true while something is actually initialising. A loader sitting at
    // download-required is waiting for the visitor, and a screen-reader user hearing "preparing"
    // would wait forever for a transfer that is never going to start on its own.
    el.textContent =
      "Nothing is downloading. These models aren't on this device yet — use the Download buttons " +
      "above when you're ready; nothing is fetched without you asking.";
  } else if (gpuBlocked) {
    // A loader that lands on "unsupported" never calls onReady, so without this the page would sit
    // on "Preparing…" forever while the loader beside it plainly says the model cannot run.
    el.classList.add("err");
    el.textContent =
      `This device can't run Ultravox: ${gpuBlocked}. It needs WebGPU and about 1.5 GB of GPU ` +
      "memory. There is no CPU fallback for a model this size, so nothing will be downloaded.";
  } else {
    const missing = [!ready.llm && "Ultravox", !ready.vad && "Silero VAD"].filter(Boolean).join(", ");
    el.textContent = `${count}/2 models ready · still preparing ${missing}…`;
  }
  updateControls();
}

function updateControls() {
  const held = busy || clipPreparing;
  const canListen = ready.llm && ready.vad && LiveMic.supported() && !held && !micStarting;
  $("listen").disabled = (!canListen && !listening) || micStarting;
  // Sending the clip mid-utterance would set busy, and onVadStream skips frames while busy WITHOUT
  // clearing the partially collected utterance — so the next live command would be spliced onto a
  // stale fragment. Simplest correct answer: the clip is a separate mode from listening.
  $("runClip").disabled = !ready.llm || held || listening || micStarting;
  $("clipPrompt").disabled = !ready.llm || held;
}

createModelLoader({
  mount: $("loader-llm"),
  model: {
    modelId: MODEL_ID,
    runtime: "transformers.js",
    dtype: "q8/q4/q4",
    sizeMB: 1523,
    requiresWebGPU: true,
  },
  load: async (onProgress) => {
    const r = await engines.llm.load(onProgress);
    return r;
  },
  onReady: (r) => {
    ready.llm = true;
    device = r?.device ?? null;
    reportReadiness();
  },
  dispose: () => resetEngine("llm"),
  onDispose: () => {
    ready.llm = false;
    device = null;
    reportReadiness();
  },
});

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
  dispose: () => resetEngine("vad"),
  onDispose: () => {
    if (listening) stopListening();
    reportReadiness();
  },
});

// ---------------------------------------------------------------------------
// Live meter
// ---------------------------------------------------------------------------
const HISTORY = 240;
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

  const threshold = () => {
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
    threshold();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = muted;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("waiting for audio", w / 2, h / 2);
    ctx.globalAlpha = 1;
    return;
  }
  const step = w / HISTORY;
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
  threshold();
}
addEventListener("resize", drawMeter);

function setPhase(text, kind = "0") {
  const el = $("phase");
  el.textContent = text;
  el.dataset.on = kind;
}

function announce(text) {
  const region = $("announcer");
  region.textContent = "";
  requestAnimationFrame(() => {
    region.textContent = text;
  });
}

// ---------------------------------------------------------------------------
// Mic + endpointing (Silero decides WHEN you stopped — it never produces words)
// ---------------------------------------------------------------------------
let mic = null;
const pending = [];
const MAX_PENDING = 64;
let staleVadReplies = 0;

let inSpeech = false;
let startRun = 0;
let silentFrames = 0;
// Voiced frames only. The utterance's total length always includes 0.4 s of pre-roll and the 0.8 s
// hangover that ended it, so a duration test against it could never reject anything — the filter
// was dead code. Count the frames Silero actually called speech instead.
let voicedFrames = 0;
let utterance = [];
let utterLen = 0;
const preroll = [];
let prerollLen = 0;

function resetEndpointer() {
  inSpeech = false;
  startRun = 0;
  silentFrames = 0;
  voicedFrames = 0;
  utterance = [];
  utterLen = 0;
  preroll.length = 0;
  prerollLen = 0;
}

function pushPreroll(frame) {
  preroll.push(frame);
  prerollLen += frame.length;
  while (preroll.length > 1 && prerollLen - preroll[0].length >= PREROLL_SAMPLES) {
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

function onVadStream(msg) {
  if (staleVadReplies > 0) {
    staleVadReplies--;
    return;
  }
  const chunk = pending.shift();
  if (!chunk) return;
  const p = msg.probs;
  for (let f = 0; f < p.length; f++) {
    const frame = chunk.subarray(f * 512, f * 512 + 512);
    let sumSq = 0;
    for (let i = 0; i < frame.length; i++) sumSq += frame[i] * frame[i];
    pushFrame(Math.sqrt(sumSq / Math.max(1, frame.length)), p[f]);

    if (busy) continue;

    if (!inSpeech) {
      pushPreroll(frame.slice());
      startRun = p[f] >= START_PROB ? startRun + 1 : 0;
      if (startRun >= START_FRAMES) {
        inSpeech = true;
        silentFrames = 0;
        voicedFrames = startRun;
        utterance = preroll.slice();
        utterLen = prerollLen;
        preroll.length = 0;
        prerollLen = 0;
        setPhase("hearing you", "speech");
      }
    } else {
      utterance.push(frame.slice());
      utterLen += frame.length;
      if (p[f] >= STOP_PROB) voicedFrames++;
      silentFrames = p[f] < STOP_PROB ? silentFrames + 1 : 0;
      const hangReached = silentFrames * FRAME_SEC >= HANG_SEC;
      const tooLong = utterLen >= MAX_UTTER_SAMPLES;
      if (hangReached || tooLong) {
        const pcm = concat(utterance, utterLen);
        const voicedSec = voicedFrames * FRAME_SEC;
        resetEndpointer();
        const seconds = pcm.length / 16000;
        if (voicedSec < MIN_VOICED_SEC) {
          // A cough, a door, a single clipped syllable. Not worth ~1.5 GB of model's attention.
          setPhase("listening", "1");
          $("micNote").textContent =
            `Ignored ${voicedSec.toFixed(2)} s of sound — too short to be a command.`;
        } else {
          runTurn({ audio: pcm, seconds, voicedSec, truncated: tooLong, source: "mic" });
        }
      }
    }
  }
}
engines.vad.onStream = onVadStream;

function watchVadErrors() {
  engines.vad.worker.addEventListener("message", (e) => {
    if (e.data?.type !== "error" || !listening) return;
    stopListening();
    $("status").textContent = `Voice detection failed, so listening stopped: ${e.data.message}`;
    $("status").classList.add("err");
  });
}
watchVadErrors();

async function startListening() {
  if (micStarting || listening) return;
  if (!LiveMic.supported()) {
    $("micFallback").hidden = false;
    return;
  }
  micStarting = true;
  $("listen").disabled = true;
  $("micNote").textContent = "Asking for microphone access…";
  try {
    await beginListening();
  } finally {
    micStarting = false;
    updateControls();
  }
}

async function beginListening() {
  const generation = vadGeneration;
  const engine = engines.vad;
  try {
    await engine.streamReset();
  } catch (err) {
    $("status").textContent = `Couldn't reset the VAD: ${err.message}`;
    $("status").classList.add("err");
    return;
  }
  if (generation !== vadGeneration) return; // released while we were resetting
  const pendingMic = new LiveMic({
    onFrames: (frames) => {
      // Bound to the engine that was current when capture started. After a Release/Clear these
      // frames would otherwise reach a fresh, loader-unready engine and streamChunk() would fetch
      // Silero again behind the visitor's back, with the loader still saying download-required.
      if (generation !== vadGeneration) return;
      if (pending.length >= MAX_PENDING) {
        // Dropping samples mid-utterance would leave a hole: later frames get appended after a
        // missing span and Ultravox would hear a SPLICED command — "set a timer for five… minutes"
        // with the middle gone. On a page whose whole claim is that the model hears the real audio,
        // quietly stitching it is the worst option. Abandon this turn and say so.
        const wasCollecting = inSpeech;
        resetEndpointer();
        setPhase(listening ? "listening" : "idle", listening ? "1" : "0");
        $("micNote").textContent = wasCollecting
          ? "This device can't keep up — that turn was discarded rather than sent with a gap. Try again."
          : "Dropping audio — the voice detector is behind on this device.";
        return;
      }
      pending.push(frames);
      engine.streamChunk(frames);
    },
  });
  mic = pendingMic;
  try {
    await pendingMic.start();
  } catch (err) {
    try {
      pendingMic.stop();
    } catch { /* nothing to release */ }
    if (mic === pendingMic) mic = null;
    $("micFallback").hidden = false;
    $("micNote").textContent = `Microphone unavailable (${err.name || "error"}).`;
    return;
  }
  // The permission prompt can outlast a Release/Clear. If the engine was replaced while we waited,
  // close the microphone we just opened rather than streaming into a model that is no longer loaded.
  if (generation !== vadGeneration) {
    try {
      pendingMic.stop();
    } catch { /* already gone */ }
    if (mic === pendingMic) mic = null;
    $("micNote").textContent = "Listening cancelled — the voice detector was released.";
    return;
  }
  listening = true;
  staleVadReplies = 0;
  resetEndpointer();
  $("listen").innerHTML = '<span class="rec-dot"></span>Stop listening';
  $("micNote").textContent = "Mic is open. Speak, then pause.";
  setPhase("listening", "1");
}

function stopListening() {
  listening = false;
  try {
    mic?.stop();
  } catch { /* already gone */ }
  mic = null;
  staleVadReplies += pending.length;
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
// A turn
// ---------------------------------------------------------------------------
const STAGES = [
  ["vad", "Silero endpoint"],
  ["hear", "Ultravox hears"],
  ["tool", "tool"],
  ["answer", "Ultravox answers"],
];
let turnSeq = 0;

function newTurnCard() {
  $("turnsEmpty").hidden = true;
  const li = document.createElement("li");
  li.className = "turn";
  li.innerHTML = `
    <h3>Turn ${++turnSeq}</h3>
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

function addBlock(card, title, text) {
  const h = document.createElement("p");
  h.className = "muted";
  h.style.cssText = "margin:.5rem 0 .15rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em";
  h.textContent = title;
  const body = document.createElement("pre");
  body.className = "call";
  body.textContent = text;
  card.body.append(h, body);
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

/** Show the audio's footprint in the prompt: N embedding positions among the text tokens. */
function addAudioFootprint(card, audioFrames, promptTokens, seconds) {
  const p = document.createElement("p");
  p.className = "muted";
  p.style.cssText = "margin:.5rem 0 .15rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em";
  p.textContent = "your voice, as the model received it";
  const viz = document.createElement("div");
  viz.className = "audioviz";
  const shown = Math.min(audioFrames, 60);
  for (let i = 0; i < shown; i++) viz.append(document.createElement("span"));
  const note = document.createElement("p");
  note.className = "muted";
  note.style.fontSize = ".82rem";
  note.textContent =
    `${seconds.toFixed(1)} s of audio became ${audioFrames} embedding positions` +
    `${audioFrames > shown ? ` (${shown} shown)` : ""} in a ${promptTokens}-token prompt. ` +
    "No transcript was produced at any point.";
  card.body.append(p, viz, note);
}

function readout(card, parts) {
  card.readout.innerHTML = parts
    .filter(Boolean)
    .map(([k, v]) => `<span>${escapeHTML(k)} <b>${escapeHTML(String(v))}</b></span>`)
    .join("");
}

async function runTurn({ audio, seconds = 0, voicedSec = null, prompt = null, truncated = false, source = "mic" }) {
  if (busy) return;
  if (!ready.llm) {
    $("status").textContent =
      "Ultravox isn't on this device yet — use the Download button above first; nothing is fetched behind your back.";
    $("status").classList.add("err");
    return;
  }
  $("status").classList.remove("err");
  busy = true;
  updateControls();
  const card = newTurnCard();
  const t0 = performance.now();

  try {
    if (source === "clip") {
      // The clip is decoded and handed straight to Ultravox — endpointing is what Silero is FOR, and
      // a pre-cut file needs none. Showing that stage as "done" would credit a model that never ran.
      card.stage("vad", "skipped", "Silero not used");
    } else {
      card.stage("vad", "done", `${voicedSec != null ? voicedSec.toFixed(1) : seconds.toFixed(1)}s voiced`);
    }
    setPhase("listening back", "1");

    // The audio IS the user turn. Any text sits alongside the placeholder, never replacing it.
    const userContent = prompt ? `${prompt}\n${AUDIO_PLACEHOLDER}` : AUDIO_PLACEHOLDER;
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ];

    card.stage("hear", "pending", "Ultravox…");
    let template1 = "";
    const first = await engines.llm.generate({
      messages,
      tools: TOOL_SCHEMAS,
      audio: audio.slice(), // transferred to the worker
      maxTokens: 192,
      onPrompt: (t) => (template1 = t),
      onToken: (_tok, n) => card.stage("hear", "pending", `Ultravox… ${n} tok`),
    });
    card.stage("hear", "done", `heard ${first.genMs} ms`);
    addAudioFootprint(card, first.audioFrames, first.promptTokens, seconds);
    addBlock(card, "what the model produced", (first.text || "").trim() || "(nothing)");
    addInside(card, "the exact prompt it saw (audio placeholder expanded per frame)", template1);

    const calls = parseToolCalls(first.text);
    if (!calls.length) {
      const direct = stripToolCalls(first.text) || (first.text || "").trim();
      if (!direct) {
        // Nothing at all came back — no tool call, no prose, or only special tokens. Rendering that
        // as a completed "direct answer" would be a blank turn dressed up as a success.
        card.stage("tool", "skipped", "no tool call");
        card.stage("answer", "fail", "no output");
        const p = document.createElement("p");
        p.className = "status err";
        p.textContent =
          "The model produced no usable output for that turn — no tool call and no answer. " +
          "Try again, or say it a little differently.";
        card.body.append(p);
        announce("The model produced no output for that turn.");
        readout(card, [
          ["audio", `${seconds.toFixed(1)} s → ${first.audioFrames} positions`],
          ["generate", `${first.genMs} ms`],
          ["tokens", first.newTokens],
          ["backend", (device || "–").toUpperCase()],
          truncated && ["note", `cut at the ${MAX_UTTER_SEC}s cap`],
        ]);
        return;
      }
      card.stage("tool", "skipped", "no tool call");
      card.stage("answer", "done", "direct answer");
      card.answer.textContent = direct;
      announce(`Answered without a tool: ${direct}`);
      const note = document.createElement("p");
      note.className = "muted";
      note.style.fontSize = ".82rem";
      note.textContent =
        "No tool call in that reply — the model answered from what it heard. That is a real outcome " +
        "for a 1B model, not a page error.";
      card.body.append(note);
      readout(card, [
        ["audio", `${seconds.toFixed(1)} s → ${first.audioFrames} positions`],
        ["prep", `${first.prepMs} ms`],
        ["generate", `${first.genMs} ms`],
        ["tokens", first.newTokens],
        ["backend", (device || "–").toUpperCase()],
        ["total", `${Math.round(performance.now() - t0)} ms`],
        // The tool branch discloses this; the direct-answer branch must too, or the user is never
        // told the end of their command was cut off.
        truncated && ["note", `cut at the ${MAX_UTTER_SEC}s cap`],
      ]);
      return;
    }

    const call = calls[0];
    card.stage("tool", "pending", call.name);
    addBlock(card, "parsed tool call", JSON.stringify(call, null, 2));
    const outcome = runTool(call, toolCtx);
    card.stage("tool", outcome.ok ? "done" : "fail", `${call.name}${outcome.ok ? " ✓" : " ✗"}`);
    addBlock(
      card,
      outcome.ok ? "what the tool returned" : "the tool failed (the model is told, and explains)",
      outcome.ok ? `${outcome.display}\n${JSON.stringify(outcome.result)}` : outcome.error,
    );
    if (call.name === "calculate" && outcome.ok) $("lastCalc").textContent = outcome.display;

    // Second pass. The audio stays in the conversation — the model still has the original sound.
    card.stage("answer", "pending", "Ultravox…");
    const followUp = [
      ...messages,
      { role: "assistant", tool_calls: [{ type: "function", function: call }] },
      { role: "tool", content: toolMessageContent(outcome) },
    ];
    const second = await engines.llm.generate({
      messages: followUp,
      tools: TOOL_SCHEMAS,
      audio: audio.slice(),
      maxTokens: 128,
      onToken: (_tok, n) => card.stage("answer", "pending", `Ultravox… ${n} tok`),
    });
    const answer = stripToolCalls(second.text).trim() || outcome.display;
    card.answer.textContent = answer;
    announce(outcome.ok ? `${call.name} ran. ${answer}` : `${call.name} was not run. ${answer}`);
    card.stage("answer", "done", `${second.genMs} ms`);
    readout(card, [
      ["audio", `${seconds.toFixed(1)} s → ${first.audioFrames} positions`],
      ["hear", `${first.genMs} ms`],
      ["tool", `${outcome.ms} ms`],
      ["answer", `${second.genMs} ms`],
      ["backend", (device || "–").toUpperCase()],
      ["total", `${Math.round(performance.now() - t0)} ms`],
      truncated && ["note", `cut at the ${MAX_UTTER_SEC}s cap`],
    ]);
  } catch (err) {
    const p = document.createElement("p");
    p.className = "status err";
    p.textContent = `That turn failed: ${err?.message ?? err}`;
    card.body.append(p);
    for (const [key] of STAGES) {
      const n = card.el.querySelector(`[data-key="${key}"]`);
      if (n && n.dataset.state === "pending") n.dataset.state = "fail";
    }
    announce(`That turn failed: ${err?.message ?? err}`);
  } finally {
    busy = false;
    setPhase(listening ? "listening" : "idle", listening ? "1" : "0");
    updateControls();
  }
}

// ---------------------------------------------------------------------------
// The bundled clip
// ---------------------------------------------------------------------------
$("runClip").addEventListener("click", async () => {
  if (busy || clipPreparing) return;
  clipPreparing = true;
  updateControls();
  setPhase("decoding clip", "1");
  try {
    const { pcm } = await urlToMono16k("../whisper-speech-to-text/jfk.wav");
    clipPreparing = false;
    updateControls();
    await runTurn({
      audio: pcm,
      seconds: pcm.length / 16000,
      prompt: $("clipPrompt").value.trim() || null,
      source: "clip",
    });
  } catch (err) {
    $("status").textContent = `Couldn't decode the bundled clip: ${err.message}`;
    $("status").classList.add("err");
  } finally {
    clipPreparing = false;
    setPhase(listening ? "listening" : "idle", listening ? "1" : "0");
    updateControls();
  }
});

renderNotes();
renderTimers();
updateControls();
drawMeter();

// Gate honestly and EARLY: probe on the main thread so the page says why it can't run before the
// loaders finish their own checks, rather than showing a stale "preparing" line.
// The shared loader has no callback for "settled but not ready", so watch its state attribute.
for (const mount of ["loader-llm", "loader-vad"]) {
  const el = $(mount);
  new MutationObserver(() => reportReadiness())
    .observe(el, { attributes: true, attributeFilter: ["data-state"], subtree: true });
}

probeWebGPUMain().then((gpu) => {
  if (gpu.ok) return;
  const reasons = {
    "no-gpu": "this browser doesn't expose the WebGPU API",
    "no-adapter": "WebGPU is present but no GPU adapter is available (common in headless browsers and VMs)",
    "adapter-error": "requesting a WebGPU adapter threw an error",
  };
  gpuBlocked = reasons[gpu.reason] ?? "WebGPU isn't usable here";
  reportReadiness();
});

function isLocalHost() {
  const h = location.hostname;
  return h === "127.0.0.1" || h === "localhost" || h === "[::1]" || h === "" || h === "::1";
}
if (isLocalHost()) {
  globalThis.__ultravox = {
    runTurn,
    engines,
    toolCtx,
    state: () => ({ ready: { ...ready }, busy, listening, device, notes: [...notes] }),
    markReady: (which) => {
      ready[which] = true;
      reportReadiness();
    },
  };
}
