// media-pipeline.audio-feature-worker.js — the feature-extraction half of AudioCapturePipeline.
//
// Receives BOUNDED audio chunks (either transferred via postMessage — the default — or read from a
// SharedArrayBuffer ring buffer when cross-origin isolated) and computes lightweight per-chunk
// features (RMS energy in dBFS, peak, zero-crossing rate). It keeps only a BOUNDED ring of the most
// recent feature frames so memory never grows without limit, and posts each new frame back to the
// main thread. Swap the feature math for real DSP (mel filterbank, MFCC, etc.) in a real demo.

let transport = "postmessage";
let header = null; // Int32Array [writeIndex, readIndex] when transport === "sab"
let ring = null; // Float32Array samples
let capacity = 0;
let chunkSize = 2048;
let sampleRate = 48000;

// Bounded history of recent feature frames (never grows past MAX_FRAMES).
const MAX_FRAMES = 512;
const frames = [];

function featuresFor(samples) {
  let sumSq = 0;
  let peak = 0;
  let crossings = 0;
  let prev = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    sumSq += v * v;
    const a = Math.abs(v);
    if (a > peak) peak = a;
    if (i > 0 && ((v >= 0) !== (prev >= 0))) crossings++;
    prev = v;
  }
  const rms = Math.sqrt(sumSq / samples.length);
  const db = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
  return {
    t: Date.now(),
    rms,
    db: Number.isFinite(db) ? db : -120,
    peak,
    zcr: crossings / samples.length,
    zcrHz: (crossings / 2) * (sampleRate / samples.length),
  };
}

function emit(samples) {
  const f = featuresFor(samples);
  frames.push(f);
  if (frames.length > MAX_FRAMES) frames.shift(); // bounded
  self.postMessage({ type: "features", features: f });
}

function drainSab() {
  // Read everything between readIndex and writeIndex out of the ring, one chunk at a time.
  let r = Atomics.load(header, 1);
  const w = Atomics.load(header, 0);
  let available = (w - r + capacity) % capacity;
  while (available >= chunkSize) {
    const out = new Float32Array(chunkSize);
    for (let i = 0; i < chunkSize; i++) {
      out[i] = ring[r];
      r = (r + 1) % capacity;
    }
    Atomics.store(header, 1, r);
    emit(out);
    available -= chunkSize;
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    transport = msg.transport;
    sampleRate = msg.sampleRate || sampleRate;
    chunkSize = msg.chunkSize || chunkSize;
    if (transport === "sab") {
      header = new Int32Array(msg.sab, 0, 2);
      ring = new Float32Array(msg.sab, 8);
      capacity = msg.capacity;
    }
    self.postMessage({ type: "ready", transport });
  } else if (msg.type === "chunk") {
    // Default path: a bounded chunk transferred from the worklet via the main thread.
    emit(msg.samples);
  } else if (msg.type === "drain") {
    // SAB path: the worklet nudged us that new samples are in the ring.
    if (transport === "sab") drainSab();
  } else if (msg.type === "stop") {
    frames.length = 0;
    self.close();
  }
};
