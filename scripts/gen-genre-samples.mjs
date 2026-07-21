// Generate first-party (procedural, no third-party audio) sample clips for the music-genre demo.
// 16 kHz mono 16-bit PCM WAV. Two contrasting textures the model reads reliably to DISTINCT genres:
// a sustained chord progression (-> jazz) and a boom-bap drum beat (-> hip-hop). Real genre range comes
// from user upload. Verified in headless Chrome: sample-jazz -> jazz 0.85, sample-hiphop -> hiphop 0.76.
import fs from "node:fs";
const SR = 16000, DUR = 8, N = SR * DUR;
const DIR = "models/music-genre-classification/";
function writeWav(path, samples) {
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + samples.length * 2, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE((v * 32767) | 0, 44 + i * 2);
  }
  fs.writeFileSync(path, buf);
  return buf.length;
}
const noise = () => Math.random() * 2 - 1;
// 1) sustained major-triad chord progression -> jazz
{
  const x = new Float32Array(N);
  const prog = [[261.6, 329.6, 392], [293.7, 370, 440], [220, 277.2, 329.6], [261.6, 329.6, 392]];
  for (let i = 0; i < N; i++) {
    const t = i / SR;
    const ch = prog[Math.floor(t / 2) % prog.length];
    const nt = (t / 2) % 1;
    const e = Math.min(1, nt * 8) * Math.exp(-nt * 0.6);
    let s = 0;
    for (const f of ch) {
      s += Math.sin(2 * Math.PI * f * t) + 0.2 * Math.sin(2 * Math.PI * f * 2 * t);
    }
    x[i] = 0.2 * e * s;
  }
  console.log("jazz", writeWav(DIR + "sample-jazz.wav", x), "bytes");
}
// 2) boom-bap drum beat -> hip-hop
{
  const x = new Float32Array(N);
  const bpm = 90, spb = 60 / bpm;
  for (let i = 0; i < N; i++) {
    const t = i / SR;
    const beat = t / spb, ph = beat - Math.floor(beat), b = Math.floor(beat) % 4;
    const k = ((b === 0 || b === 2) && ph < 0.1)
      ? Math.sin(2 * Math.PI * (55 - 30 * ph) * t) * Math.exp(-ph * 10)
      : 0;
    const sn = ((b === 1 || b === 3) && ph < 0.1) ? noise() * Math.exp(-ph * 20) * 0.5 : 0;
    x[i] = 0.7 * k + sn + noise() * 0.02;
  }
  console.log("hiphop", writeWav(DIR + "sample-hiphop.wav", x), "bytes");
}
