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
  stripReservedAudioTokens,
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
// Bumped by anything that should abandon a microphone startup already in flight. vadGeneration alone
// was not enough: releasing the LLM while the permission prompt was open left it unchanged, so
// beginListening() went on to open the microphone for a model that no longer existed.
let captureGeneration = 0;
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
  if (kind === "llm") {
    device = null;
    // Ultravox is the only thing on this page that can answer, so tearing it down must also abandon
    // a microphone startup already in flight. `listening` cannot cover that case — during the
    // permission prompt it is still false — and vadGeneration does not move when only the LLM goes.
    captureGeneration++;
  }
  if (kind === "vad") {
    vadGeneration++;
    captureGeneration++;
    vadResetting = false;
    pending.clear();
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
  const justFinished = [];
  box.replaceChildren();
  for (const t of timers) {
    // Wall-clock, not performance.now(): a countdown measured on a clock that can stall while the
    // machine sleeps would show five minutes left after a ten-minute nap. A timer must expire on
    // elapsed real time.
    const left = Math.max(0, Math.ceil((t.endsAt - Date.now()) / 1000));
    // One polite announcement on the transition to zero. Making the countdown itself live would read
    // every second aloud; saying nothing means a screen-reader user never learns the timer finished.
    // Collected, not announced here: two timers finishing in the same tick would each clear the live
    // region and schedule on the same frame, so only the last would be spoken — and both are already
    // flagged, so the lost one would never be announced at all.
    if (left === 0 && !t.announced) {
      t.announced = true;
      justFinished.push(t.label);
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
  if (justFinished.length === 1) announce(`Timer finished: ${justFinished[0]}.`);
  else if (justFinished.length > 1) {
    announce(`${justFinished.length} timers finished: ${justFinished.join(", ")}.`);
  }
}

function startTimer(seconds, label) {
  const id = `t${++timerSeq}`;
  timers.push({ id, label, endsAt: Date.now() + seconds * 1000 });
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
    // An already-open microphone has to be closed too — otherwise it stays live and drops every
    // completed utterance into the "not on this device" error. (A startup still in flight is
    // cancelled by the capture-generation bump in resetEngine.)
    if (listening) stopListening();
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

function micNote(text) {
  $("micNote").textContent = text;
}

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
// Queued audio, keyed by the VAD worker's request id. It used to be an array plus a counter of
// replies to discard, which was wrong twice over: a chunk that FAILED never produced a reply, so the
// counter over-counted and every subsequent reply was paired with the previous chunk's PCM; and the
// busy decision was taken when the reply came back rather than when the audio was captured, so on a
// backlogged device speech recorded during the "not collecting" gap could be endpointed anyway.
// Each entry now carries the state that was true AT CAPTURE TIME, and correlation is by id, so a
// missing or failed reply simply never matches.
const pending = new Map();
const MAX_PENDING = 64;
let feedSeq = 0; // only used by the localhost debug hook's feedVad
// True while a post-overflow stream reset is queued behind the chunks already sitting on the
// worker's serialised tail. Clearing `pending` forgets the bookkeeping but CANNOT unsend those
// messages: they keep running and keep mutating Silero's recurrent state, so accepting new audio
// immediately would splice the discarded command's tail into a fresh utterance. Frames are dropped
// until the reset is acknowledged, which — because the reset joins the same tail — means until the
// backlog has actually drained.
let vadResetting = false;

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

// Set while a turn is generating, because capture is NOT collecting then. Resuming the moment the
// turn ends would take whatever the visitor happens to be saying mid-sentence and endpoint the TAIL
// of it as a complete command — "…for five minutes" arriving as the whole request, which is exactly
// the sort of half-heard instruction that could start the wrong timer. Cleared only after a real
// silence gap proves we are at a boundary again.
let awaitingResync = false;
let resyncSilent = 0;
const RESYNC_FRAMES = Math.ceil(HANG_SEC / FRAME_SEC);

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
  // No entry means the chunk was discarded (stop, overflow, engine reset) — its reply is stale.
  const entry = pending.get(msg.id);
  if (!entry) return;
  pending.delete(msg.id);
  const chunk = entry.pcm;
  const capturedWhileBusy = entry.busy;
  const p = msg.probs;
  for (let f = 0; f < p.length; f++) {
    const frame = chunk.subarray(f * 512, f * 512 + 512);
    let sumSq = 0;
    for (let i = 0; i < frame.length; i++) sumSq += frame[i] * frame[i];
    pushFrame(Math.sqrt(sumSq / Math.max(1, frame.length)), p[f]);

    // The state that matters is the one at CAPTURE time, not now: on a slow device this reply can
    // arrive after generation finished, and treating that audio as freshly captured would let speech
    // recorded during the advertised gap be endpointed as a new command.
    if (capturedWhileBusy || busy) {
      // Not collecting: the answer is still generating. Remember that the stream has a hole in it.
      if (!awaitingResync) {
        awaitingResync = true;
        resyncSilent = 0;
        resetEndpointer();
        micNote("Still answering — speech during this gap isn't collected. Wait for the answer.");
      }
      continue;
    }
    if (awaitingResync) {
      // Wait for a genuine pause before trusting the stream again, so a sentence that began during
      // the gap is never mistaken for a complete command.
      resyncSilent = p[f] < STOP_PROB ? resyncSilent + 1 : 0;
      if (resyncSilent < RESYNC_FRAMES) continue;
      awaitingResync = false;
      resyncSilent = 0;
      resetEndpointer();
      micNote("Mic is open. Speak, then pause.");
      setPhase("listening", "1");
      continue;
    }

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
    if (e.data?.type !== "error") return;
    // A failed chunk produces an error carrying its id, never a `stream` reply. Drop its entry so
    // the invariant holds — every queued chunk is removed exactly once, on reply OR on failure —
    // regardless of what the policy below decides to do about listening.
    if (e.data.id != null) pending.delete(e.data.id);
    if (!listening) return;
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
  const capture = captureGeneration;
  // Either counter moving means this startup was overtaken — the VAD was replaced, or the model that
  // consumes the audio was released.
  const cancelled = () => generation !== vadGeneration || capture !== captureGeneration;
  const engine = engines.vad;
  try {
    await engine.streamReset();
  } catch (err) {
    $("status").textContent = `Couldn't reset the VAD: ${err.message}`;
    $("status").classList.add("err");
    return;
  }
  if (cancelled()) return; // released while we were resetting
  const pendingMic = new LiveMic({
    onFrames: (frames) => {
      // Bound to the engine that was current when capture started. After a Release/Clear these
      // frames would otherwise reach a fresh, loader-unready engine and streamChunk() would fetch
      // Silero again behind the visitor's back, with the loader still saying download-required.
      if (cancelled()) return;
      // Nothing is sent while the worker is draining and resetting — see `vadResetting`.
      if (vadResetting) return;
      if (pending.size >= MAX_PENDING) {
        // Dropping samples mid-utterance would leave a hole: later frames get appended after a
        // missing span and Ultravox would hear a SPLICED command — "set a timer for five… minutes"
        // with the middle gone. On a page whose whole claim is that the model hears the real audio,
        // quietly stitching it is the worst option. Abandon this turn and say so.
        const wasCollecting = inSpeech;
        resetEndpointer();
        // Clearing the endpointer alone was not enough: `pending` still held the queued audio, so as
        // soon as one reply freed a slot the backlog resumed and could form a new utterance spanning
        // the very gap the UI had just said was discarded. Discard the queued work too; their replies
        // no longer match anything and are ignored on arrival.
        pending.clear();
        // Quarantine the endpointer too: after the reset lands, capture waits for a genuine silence
        // gap before trusting the stream, so the tail of the discarded command cannot start a turn.
        awaitingResync = true;
        resyncSilent = 0;
        if (!vadResetting) {
          vadResetting = true;
          const gen = captureGeneration;
          engine.streamReset()
            .catch(() => { /* a failing VAD is reported by watchVadErrors */ })
            .finally(() => {
              // A reset that lands after a Release/Clear must not un-pause a stream nobody wants.
              if (gen === captureGeneration) vadResetting = false;
            });
        }
        setPhase(listening ? "listening" : "idle", listening ? "1" : "0");
        $("micNote").textContent = wasCollecting
          ? "This device can't keep up — that turn was discarded rather than sent with a gap. Try again."
          : "Dropping audio — the voice detector is behind on this device.";
        return;
      }
      // Tag the chunk with the state at capture time, then send. The id ties the two together.
      const id = engine.streamChunk(frames);
      pending.set(id, { pcm: frames, busy });
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
    if (err?.name === "AudioContextSuspendedError") {
      // start() now fails closed on a suspended context, so this is the honest report rather than a
      // "mic is open" line over silence.
      $("status").textContent = err.message;
      $("status").classList.add("err");
      $("micNote").textContent = "Audio engine suspended — capture did not start.";
      return;
    }
    $("micFallback").hidden = false;
    $("micNote").textContent = `Microphone unavailable (${err.name || "error"}).`;
    return;
  }
  // The permission prompt can outlast a Release/Clear. If the engine was replaced while we waited,
  // close the microphone we just opened rather than streaming into a model that is no longer loaded.
  if (cancelled()) {
    try {
      pendingMic.stop();
    } catch { /* already gone */ }
    if (mic === pendingMic) mic = null;
    $("micNote").textContent = "Listening cancelled — a model this needs was released.";
    return;
  }
  // A suspended audio context is now a THROWN start failure (handled in the catch above), not a
  // silently-resolved one, so there is nothing to re-check here.
  listening = true;
  pending.clear();
  resetEndpointer();
  // A previous failure may have shown the no-microphone panel; capture is plainly working now.
  $("micFallback").hidden = true;
  $("listen").innerHTML = '<span class="rec-dot"></span>Stop listening';
  $("micNote").textContent = "Mic is open. Speak, then pause.";
  $("status").classList.remove("err");
  setPhase("listening", "1");
}

function stopListening() {
  listening = false;
  awaitingResync = false;
  resyncSilent = 0;
  vadResetting = false;
  try {
    mic?.stop();
  } catch { /* already gone */ }
  mic = null;
  pending.clear();
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
  // Closing the track is not enough: with the back-forward cache the page comes BACK, and it would
  // return showing "Stop listening" over a microphone that no longer exists — the next press would
  // only clear that stale state instead of reopening capture. Tear the whole thing down.
  try {
    // A startup still awaiting the permission prompt is the case stopListening() cannot reach:
    // `listening` is false, and stopping a microphone that has not opened yet cancels nothing. Bump
    // the capture generation so that when the prompt finally resolves — possibly after the page has
    // been restored from the back-forward cache — beginListening() abandons it instead of marking
    // the restored page as listening.
    captureGeneration++;
    if (listening) stopListening();
    else mic?.stop();
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

    // The audio IS the user turn. Any text sits alongside the placeholder, never replacing it — and
    // the visitor's text cannot contribute a placeholder of its own (the page documents the token on
    // screen, so it does get typed; two placeholders for one recording breaks the turn).
    const askText = stripReservedAudioTokens(prompt);
    const userContent = askText ? `${askText}\n${AUDIO_PLACEHOLDER}` : AUDIO_PLACEHOLDER;
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
      // NOT `|| first.text`: a reply of nothing but special tokens (<|eot_id|>, <|python_tag|>)
      // strips to empty, and restoring the raw markup rendered it as a real answer. Now that
      // stripToolCalls preserves genuine content — including ordinary JSON — the stripped result IS
      // the model's answer, and empty means empty.
      const direct = stripToolCalls(first.text);
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
    // The prompt asks for exactly one call per reply, and every executor MUTATES page state. Running
    // an arbitrary prefix would leave one timer started and the second silently dropped, with neither
    // the model nor the visitor told. Nothing runs; the model is given the reason and answers from it.
    const multi = calls.length > 1;
    card.stage("tool", "pending", multi ? `${calls.length} calls` : call.name);
    addBlock(
      card,
      multi ? `parsed ${calls.length} tool calls — none run` : "parsed tool call",
      JSON.stringify(multi ? calls : call, null, 2),
    );
    const outcome = multi
      ? {
        ok: false,
        name: call.name,
        error: `That reply contained ${calls.length} tool calls (${
          calls.map((c) => c.name).join(", ")
        }). Only one tool may be called per reply, so none of them were run. Call exactly one tool.`,
      }
      : runTool(call, toolCtx);
    card.stage(
      "tool",
      outcome.ok ? "done" : "fail",
      multi ? `${calls.length} calls ✗ none run` : `${call.name}${outcome.ok ? " ✓" : " ✗"}`,
    );
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
    // A REFUSED tool carries `error`, not `display` — falling back to display rendered the literal
    // string "undefined" and announced it, while marking the stage done.
    const fallback = outcome.ok ? outcome.display : outcome.error;
    const generated = stripToolCalls(second.text).trim();
    const answer = generated || fallback || "";
    if (generated) {
      card.answer.textContent = generated;
      card.stage("answer", "done", `${second.genMs} ms`);
    } else if (fallback) {
      // The model said nothing on the second pass. The tool's own line still answers the question,
      // so it is worth showing — but marking the stage "done" and presenting it as the reply would
      // be a generation the model never produced. This page's whole claim is that what you read is
      // what the model said, so the fallback is shown and LABELLED as the tool's own output.
      card.answer.textContent = fallback;
      card.answer.dataset.source = "tool";
      card.stage("answer", "fail", "no reply — showing the tool result");
      const note = document.createElement("p");
      note.className = "status err";
      note.textContent = outcome.ok
        ? "Ultravox produced no reply after the tool ran, so the line above is the tool's own output, not the model's words."
        : "Ultravox produced no reply, so the line above is why the tool was refused, not the model's words.";
      card.body.append(note);
    } else {
      card.stage("answer", "fail", "no output");
      const p = document.createElement("p");
      p.className = "status err";
      p.textContent = "The model produced no answer after the tool step. Try again.";
      card.body.append(p);
    }
    announce(
      generated
        ? (outcome.ok ? `${call.name} ran. ${generated}` : `${call.name} was not run. ${generated}`)
        : answer
        ? `${call.name} ${outcome.ok ? "ran" : "was not run"}, but the model produced no reply. The tool reported: ${answer}`
        : "The model produced no answer after the tool step.",
    );
    readout(card, [
      ["audio", `${seconds.toFixed(1)} s → ${first.audioFrames} positions`],
      ["hear", `${first.genMs} ms`],
      ["tool", outcome.ms == null ? "not run" : `${outcome.ms} ms`],
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
    // Exposed so the headless runner can prove the cancellation counter moves on release without a
    // microphone or a permission prompt, which it has neither of.
    captureGeneration: () => captureGeneration,
    releaseLLM: () => resetEngine("llm", "Released by the validator"),
    // Feed the endpointer real VAD frames. This drives the SAME code path the worker drives — no
    // state is spoofed, only the speech probabilities the model would have returned — so the
    // endpointer and its post-busy resync gate can be tested without a microphone.
    feedVad: (probs, capturedWhileBusy = busy) => {
      const id = -(++feedSeq); // negative ids can never collide with the worker's
      pending.set(id, { pcm: new Float32Array(probs.length * 512), busy: capturedWhileBusy });
      onVadStream({ id, probs: Float32Array.from(probs) });
    },
    endpointer: () => ({
      inSpeech,
      awaitingResync,
      utterLen,
      voicedFrames,
      pending: pending.size,
      vadResetting,
    }),
    startListening,
    stopListening,
    markReady: (which) => {
      ready[which] = true;
      reportReadiness();
    },
  };
}
