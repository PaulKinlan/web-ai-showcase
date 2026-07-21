// Basic Pitch transcription engine + helpers. The engine talks to worker.js (raw ORT-web); the helpers
// decode/resample audio to the model's 22050 Hz mono input, render the note-activation matrix as a piano
// roll, and pick simple note events. No fake output — the piano roll IS the model's note head; the note
// list is a simple, clearly-labelled peak-pick over that same output.

export const SR = 22050;

export class TranscribeEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.onProgress = null;
    this.pending = new Map();
    this.seq = 0;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress") this.onProgress?.(m.p);
      else if (m.type === "ready") this._ready?.();
      else if (m.type === "roll") this.pending.get(m.id)?.resolve(m);
      else if (m.type === "error") {
        if (m.id != null && this.pending.has(m.id)) {
          this.pending.get(m.id).reject(new Error(m.message));
        } else this._readyReject?.(new Error(m.message));
      }
    });
  }
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this._ready = resolve;
      this._readyReject = reject;
      this.worker.postMessage({ type: "load" });
    });
  }
  /** Transcribe mono audio at 22050 Hz → { note, onset, frames, keys, midiLow, frameSec, ms }. */
  transcribe(audio) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const a = new Float32Array(audio); // copy so we can transfer
      this.worker.postMessage({ type: "transcribe", id, audio: a }, [a.buffer]);
    }).finally(() => this.pending.delete(id));
  }
  dispose() {
    this.worker.terminate();
  }
}

/** Decode an ArrayBuffer of encoded audio (wav/mp3/webm/…) to mono Float32 at 22050 Hz. */
export async function decodeToMono22k(arrayBuffer) {
  // decodeAudioData resamples to the context's sampleRate; length param doesn't bound the output.
  const ctx = new OfflineAudioContext(1, 1, SR);
  const buf = await ctx.decodeAudioData(arrayBuffer);
  if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < ch.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export function midiToName(m) {
  return NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
}

/**
 * Render the note-activation matrix as a piano roll onto a canvas. x = time, y = pitch (high at top).
 * Intensity = note activation; onsets are drawn a touch brighter so note starts read clearly.
 */
export function drawPianoRoll(canvas, roll, { threshold = 0.3 } = {}) {
  const { note, onset, frames, keys, midiLow, frameSec } = roll;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  // faint horizontal guide at each C
  ctx.fillStyle = "rgba(127,127,127,0.12)";
  for (let k = 0; k < keys; k++) {
    if ((midiLow + k) % 12 === 0) {
      const y = H - ((k + 0.5) / keys) * H;
      ctx.fillRect(0, y - 0.5, W, 1);
    }
  }
  const cellW = W / frames, cellH = H / keys;
  for (let f = 0; f < frames; f++) {
    for (let k = 0; k < keys; k++) {
      const v = note[f * keys + k];
      if (v < threshold) continue;
      const o = onset[f * keys + k];
      const x = f * cellW, y = H - (k + 1) * cellH;
      // teal→amber ramp with activation; brighten on an onset frame
      const a = Math.min(1, v);
      const r = Math.round(40 + 200 * a),
        g = Math.round(180 * a + (o > 0.5 ? 60 : 0)),
        b = Math.round(160 * (1 - a) + 60);
      ctx.fillStyle = `rgba(${r},${g},${b},${0.25 + 0.75 * a})`;
      ctx.fillRect(x, y, Math.max(1, cellW + 0.5), Math.max(1, cellH + 0.5));
    }
  }
  return { seconds: frames * frameSec };
}

/**
 * Simple, clearly-labelled note-event extraction over the model's note + onset heads (NOT Basic Pitch's
 * full note-creation algorithm): a note starts on an onset frame with sustained note activation and ends
 * when the note head falls away. Returns [{ midi, name, startSec, durSec, conf }].
 */
export function pickNotes(roll, { onsetThresh = 0.5, noteThresh = 0.3, minFrames = 3 } = {}) {
  const { note, onset, frames, keys, midiLow, frameSec } = roll;
  const notes = [];
  for (let k = 0; k < keys; k++) {
    let active = false, startF = 0, peak = 0, gap = 0;
    for (let f = 0; f < frames; f++) {
      const nv = note[f * keys + k], ov = onset[f * keys + k];
      if (!active) {
        if (ov > onsetThresh && nv > noteThresh) {
          active = true;
          startF = f;
          peak = nv;
          gap = 0;
        }
      } else {
        peak = Math.max(peak, nv);
        if (nv < noteThresh) {
          if (++gap >= 2) {
            if (f - gap - startF >= minFrames) {
              notes.push({
                midi: midiLow + k,
                name: midiToName(midiLow + k),
                startSec: startF * frameSec,
                durSec: (f - gap - startF) * frameSec,
                conf: +peak.toFixed(2),
              });
            }
            active = false;
          }
        } else gap = 0;
      }
    }
    if (active && frames - startF >= minFrames) {
      notes.push({
        midi: midiLow + k,
        name: midiToName(midiLow + k),
        startSec: startF * frameSec,
        durSec: (frames - startF) * frameSec,
        conf: +peak.toFixed(2),
      });
    }
  }
  return notes.sort((a, b) => a.startSec - b.startSec || a.midi - b.midi);
}

export const BP_CSS = `
  .bp-controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; margin: 0.6rem 0; }
  .bp-chip { display: inline-flex; align-items: center; gap: 0.4rem; }
  .bp-chip svg { width: 16px; height: 16px; flex: none; }
  .bp-roll-wrap { position: relative; margin: 0.6rem 0; overflow-x: auto; }
  #roll { width: 100%; max-width: 100%; height: 260px; background: #0b0f14; border-radius: 10px; display: block; }
  .bp-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
  .bp-notes { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0; max-height: 8.5rem; overflow-y: auto; }
  .bp-note { font-family: var(--font-mono, monospace); font-size: 0.78rem; padding: 0.12rem 0.45rem; border-radius: 999px; background: #2bb59a22; border: 1px solid #2bb59a55; }
  .bp-dropzone { border: 1.5px dashed #8884; border-radius: 10px; padding: 0.9rem; text-align: center; cursor: pointer; font-size: 0.9rem; }
  .bp-dropzone:focus-visible { outline: 2px solid #2bb59a; }
`;
