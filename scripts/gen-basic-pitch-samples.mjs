// Generates the self-authored piano-like sample clips shipped with the basic-pitch-transcription demo.
// All audio is synthesized here (no third-party recordings) → the clips are our own, freely usable.
// Mono, 22050 Hz, 16-bit PCM WAV — Basic Pitch's native input rate. Run: node scripts/gen-basic-pitch-samples.mjs
import { writeFileSync } from "node:fs";
const SR = 22050, DIR = "models/basic-pitch-transcription/";
const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
const HARM = [1, 0.28, 0.1, 0.04]; // gentle piano-ish rolloff — in-distribution, minimal harmonic ghosting
// one piano-like note into buffer a at sample offset off, length dur, midi m, velocity v
function note(a, off, dur, m, v = 0.9) {
  const f = hz(m);
  for (let i = 0; i < dur; i++) {
    const t = off + i;
    if (t >= a.length) break;
    const s = i / SR;
    // pluck/piano envelope: fast attack, exponential decay, short release at tail
    const env = Math.min(1, s / 0.006) * Math.exp(-s / 0.9) * Math.min(1, (dur - i) / (0.02 * SR));
    let val = 0;
    for (let h = 0; h < HARM.length; h++) {
      val += HARM[h] * Math.sin(2 * Math.PI * f * (h + 1) * t / SR);
    }
    a[t] += v * env * val;
  }
}
function norm(a, peak = 0.9) {
  let mx = 0;
  for (const x of a) mx = Math.max(mx, Math.abs(x));
  const k = mx ? peak / mx : 1;
  for (let i = 0; i < a.length; i++) a[i] *= k;
  return a;
}
function wav(a) {
  const n = a.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const W = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  W(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  W(8, "WAVE");
  W(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, SR, true);
  dv.setUint32(28, SR * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  W(36, "data");
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, a[i]));
    dv.setInt16(44 + i * 2, s * 32767, true);
  }
  return Buffer.from(buf);
}
const DUR = Math.round(1.98 * SR); // one Basic Pitch window (~43844) → no windowing needed for samples
// 1) Chord progression (POLYPHONY showcase): C  F  G  C, three notes each
const chords = new Float32Array(DUR);
{
  const prog = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]];
  const seg = Math.floor(DUR / prog.length);
  prog.forEach((ch, i) => ch.forEach((m) => note(chords, i * seg, seg, m, 0.8)));
}
// 2) Arpeggio (melodic sequence): C E G B C
const arp = new Float32Array(DUR);
{
  const seq = [60, 64, 67, 71, 72], seg = Math.floor(DUR / seq.length);
  seq.forEach((m, i) => note(arp, i * seg, Math.floor(seg * 1.6), m, 0.95));
}
// 3) Two-hand (bass + melody, POLYPHONY): left-hand C2/G2 under a right-hand C-major melody
const duet = new Float32Array(DUR);
{
  const half = Math.floor(DUR / 2);
  note(duet, 0, half, 48, 0.75);
  note(duet, half, DUR - half, 43, 0.75); // bass C3 then G2 (reliably detected)
  const mel = [72, 74, 76, 77, 79], seg = Math.floor(DUR / mel.length);
  mel.forEach((m, i) => note(duet, i * seg, Math.floor(seg * 1.3), m, 0.8));
}
for (const [name, a] of [["chords", chords], ["arpeggio", arp], ["duet", duet]]) {
  writeFileSync(DIR + "sample-" + name + ".wav", wav(norm(a)));
  console.log("wrote", DIR + "sample-" + name + ".wav");
}
