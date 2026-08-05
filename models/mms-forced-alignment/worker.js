// MMS-300m forced-aligner worker — forced alignment off the main thread, WASM int8.
// Model: onnx-community/mms-300m-1130-forced-aligner-ONNX (task: automatic-speech-recognition, CTC).
//
// Forced alignment answers: "given this audio AND this transcript, WHEN does each word happen?"
// Unlike ASR (which guesses the words), alignment takes the words as ground truth and finds the
// time of each one. The MMS-300m aligner is a Wav2Vec2ForCTC model trained on a forced-alignment
// dataset: for every ~20ms audio frame it emits a distribution over a 31-symbol alphabet
// (<blank> + letters + apostrophe). We run the model directly (no ASR pipeline — the pipeline
// hides the frames) and:
//   1. tokenize the KNOWN transcript (normalized: lowercase, punctuation stripped, digits-only
//      words dropped) into letter ids with word boundaries (the model's emission contains only
//      letters + blank — verified: greedy argmax emits zero space labels — so words come from the
//      transcript, not the emission);
//   2. argmax the per-frame logits → emission strip, collapse CTC repeats + blanks;
//   3. monotonic-match the transcript letters against the emission (gaps allowed for insertions;
//      consecutive identical letters may reuse one emission — CTC doubling);
//   4. group matched letters into words by transcript boundaries → per-word start/end times.
// Every timestamp is derived from the model's real per-frame output — nothing canned.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/mms-300m-1130-forced-aligner-ONNX";

// id → character, from the model's vocab.json (deterministic for this pinned model).
// 0 = <blank> (the CTC blank), 3 = <unk> (the space label — the reference tokenizer maps ' ' to it),
// 1 = <pad>, 2 = </s>. 4..30 = lowercase letters + apostrophe.
const ID2CHAR = {
  0: "",
  1: "",
  2: "",
  3: " ",
  4: "a", 5: "i", 6: "e", 7: "n", 8: "o", 9: "u", 10: "t", 11: "s", 12: "r", 13: "m",
  14: "k", 15: "l", 16: "d", 17: "g", 18: "h", 19: "y", 20: "b", 21: "p", 22: "w", 23: "c",
  24: "v", 25: "j", 26: "z", 27: "f", 28: "'", 29: "q", 30: "x",
};
const CHAR2ID = Object.fromEntries(Object.entries(ID2CHAR).map(([id, ch]) => [ch, Number(id)]));
const BLANK = 0;
const SPACE = 3;

let model = null;
let processor = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoProcessor, AutoModelForCTC } = await import(TRANSFORMERS_URL);
  processor = await AutoProcessor.from_pretrained("onnx-community/mms-300m-1130-forced-aligner-ONNX", {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCTC.from_pretrained("onnx-community/mms-300m-1130-forced-aligner-ONNX", {
    device,
    dtype: "q4", // maps to onnx/model_q4.onnx
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

// Normalize a transcript for the aligner's 31-symbol alphabet: lowercase, strip punctuation
// (keep apostrophes inside words), drop digit-only words, collapse whitespace.
function normalizeTranscript(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z'\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w && !/^\d+$/.test(w.replace(/'/g, "")))
    .join(" ");
}

// Build the letter targets (no space labels): the model's emission contains only letters + blank
// (verified: greedy argmax over the JFK clip emits zero id-3 frames), so word boundaries come from
// the TRANSCRIPT, not the emission. Each target records its word index for grouping.
function buildTargets(normalized) {
  const words = normalized.split(" ");
  const targets = []; // {id, wordIdx}
  for (let w = 0; w < words.length; w++) {
    for (const ch of words[w]) {
      const id = CHAR2ID[ch];
      if (id !== undefined) targets.push({ id, wordIdx: w });
    }
  }
  return { targets, words };
}

// Collapse the per-frame argmax emission: drop blanks and id-3 (never emitted, but drop anyway),
// merge consecutive repeats (CTC standard).
function collapseEmission(ids) {
  const out = [];
  let prev = -1;
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    if (id === BLANK || id === SPACE) continue; // blank/space emit nothing
    if (id === prev) continue; // merge repeats
    out.push({ id, frame: t });
    prev = id;
  }
  return out;
}

// Monotonic forced alignment: match each target letter to its next occurrence in the emission,
// skipping over insertions (gaps). Consecutive identical targets may reuse one emission (CTC
// handles doubled letters). Unmatched letters stay null (honest — the model missed them).
function alignTargets(emission, targets) {
  const matches = new Array(targets.length).fill(null);
  let e = 0;
  for (let i = 0; i < targets.length; i++) {
    const tgt = targets[i].id;
    while (e < emission.length && emission[e].id !== tgt) e++;
    if (e >= emission.length) break;
    matches[i] = e;
    // If the NEXT target is identical, don't advance — it may reuse this emission.
    if (i + 1 >= targets.length || targets[i + 1].id !== tgt) e++;
  }
  return matches;
}

async function align(id, audio, transcript, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await processor(audio); // audio is already 16 kHz mono Float32Array
  const { logits } = await model(inputs); // Tensor [1, T, V]
  const ms = Math.round(performance.now() - t0);

  const [, T, V] = logits.dims;
  const data = logits.data;
  const emissionIds = new Array(T);
  for (let t = 0; t < T; t++) {
    let best = -Infinity;
    let bi = 0;
    const base = t * V;
    for (let v = 0; v < V; v++) {
      const val = data[base + v];
      if (val > best) {
        best = val;
        bi = v;
      }
    }
    emissionIds[t] = bi;
  }

  const audioSec = audioDur || audio.length / 16000;
  const frameSec = T ? audioSec / T : 0.02;

  // 1) The model's own greedy reading (honest "what it heard" — NOT the alignment result).
  const collapsed = collapseEmission(emissionIds);
  const heardText = collapsed
    .map((c) => ID2CHAR[c.id] ?? "")
    .join("")
    .replace(/\s+/g, " ")
    .trim();

  // 2) Forced alignment of the KNOWN transcript: letter targets matched monotonically to the
  // emission; words are grouped from the transcript, timed by their matched letters.
  const normalized = normalizeTranscript(transcript);
  const { targets, words: transcriptWords } = buildTargets(normalized);
  const emission = collapseEmission(emissionIds);
  const matches = alignTargets(emission, targets);

  // Group matched letters into words (transcript word boundaries). A word with zero matched
  // letters is skipped (honest gap — the model didn't emit it).
  const alignedWords = [];
  const wordStarts = new Map();
  const wordEnds = new Map();
  for (let i = 0; i < targets.length; i++) {
    if (matches[i] === null) continue;
    const w = targets[i].wordIdx;
    const f = emission[matches[i]].frame;
    if (!wordStarts.has(w) || f < wordStarts.get(w)) wordStarts.set(w, f);
    wordEnds.set(w, Math.max(wordEnds.get(w) ?? 0, f + 1));
  }
  for (let w = 0; w < transcriptWords.length; w++) {
    if (!wordStarts.has(w)) {
      alignedWords.push({ text: transcriptWords[w], start: null, end: null }); // honest gap
      continue;
    }
    alignedWords.push({
      text: transcriptWords[w],
      start: +(wordStarts.get(w) * frameSec).toFixed(2),
      end: +(wordEnds.get(w) * frameSec).toFixed(2),
    });
  }

  post({
    type: "result",
    id,
    words: alignedWords.map((w) => ({ text: w.text, start: w.start === null ? null : +(w.start.toFixed(2)), end: w.end === null ? null : +(w.end.toFixed(2)) })),
    transcript: normalized,
    heard: heardText,
    frames: T,
    frameMs: +(frameSec * 1000).toFixed(1),
    matched: matches.filter((m) => m !== null).length,
    total: targets.length,
    audioSec: +audioSec.toFixed(2),
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "align") await align(id, e.data.audio, e.data.transcript, e.data.audioDur);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
