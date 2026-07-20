// Front-end helpers for the MMS-TTS Vietnamese (VITS) pages. Thin: owns the worker handshake, WAV
// encoding and the waveform canvas. All inference runs in worker.js, off the main thread.
//
// Model: Xenova/mms-tts-vie — the VIETNAMESE VITS checkpoint from Meta's MMS family (ONNX q8, ~38 MB,
// 16 kHz mono output). This is a DISTINCT model from the built English/German/Spanish/French/Arabic
// MMS-TTS demos: Vietnamese character vocab + Vietnamese-trained VITS weights, so it speaks Vietnamese
// orthography — the full set of tone-marked vowels (à/á/ả/ã/ạ, ê/ô/ơ/ư, đ …) — natively.

const WORKER_URL = "/web-ai-showcase/models/mms-tts-vietnamese/worker.js";
export const MODEL_ID = "Xenova/mms-tts-vie";

export class MmsVieEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.loaded = new Set(); // modelIds that have been loaded at least once
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "audio") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress, modelId = MODEL_ID) {
    if (onProgress) this.onProgress = onProgress;
    if (this.loaded.has(modelId)) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({
        resolve: (d) => {
          this.loaded.add(modelId);
          resolve(d);
        },
        reject,
      });
      this.worker.postMessage({ type: "load", modelId });
    });
  }

  /** Synthesize Vietnamese text → { audio: Float32Array, rate, ms, samples, device } */
  speak(text, modelId = MODEL_ID) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "speak", id, modelId, text });
    });
  }
}

// A small set of Vietnamese example lines that exercise diacritics, tone marks and Vietnamese phrasing.
export const VIETNAMESE_SAMPLES = [
  {
    label: "Giới thiệu (Intro)",
    text: "Giọng nói này được tạo hoàn toàn bên trong trình duyệt của bạn, không cần máy chủ.",
  },
  {
    label: "Chào hỏi (Greeting)",
    text: "Xin chào. Rất vui được gặp bạn. Chúc bạn một ngày tốt lành.",
  },
  { label: "Câu líu lưỡi (Tongue twister)", text: "Nồi đồng nấu ốc, nồi đất nấu ếch; buổi trưa ăn bưởi, buổi tối ăn bòn bon." },
  {
    label: "Thông báo (Announcement)",
    text: "Lịch hẹn của bạn đã được xác nhận vào chín giờ ba mươi sáng thứ Ba. Vui lòng đến đúng giờ.",
  },
  {
    label: "Thời tiết (Weather)",
    text: "Cuối tuần trời phần lớn quang đãng, nhiệt độ cao nhất khoảng hai mươi hai độ.",
  },
  {
    label: "Dấu thanh (Tone marks)",
    text: "Mẹ mua mấy quả mận, mỗi quả một màu; má bảo mai mình mời mọi người.",
  },
];

/** Encode mono Float32 PCM → a 16-bit WAV Blob (playback needs no extra deps). */
export function wavBlob(samples, rate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const wr = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  wr(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  wr(36, "data");
  dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

export function wavUrl(samples, rate) {
  return URL.createObjectURL(wavBlob(samples, rate));
}

/** Draw a waveform into a canvas; `progress` 0..1 fills the played portion with the accent colour. */
export function drawWaveform(canvas, pcm, progress = 0, color) {
  const dpr = Math.min(2, self.devicePixelRatio || 1);
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 96;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const accent = color || cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--border").trim() || "#ccc";
  const mid = h / 2;
  const step = Math.max(1, Math.floor(pcm.length / w));
  const playX = progress * w;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const v = pcm[x * step + j] || 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.strokeStyle = x <= playX ? accent : muted;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export const MMS_VIE_CSS = `
.tts-grid { display: grid; grid-template-columns: 1fr auto; gap: .6rem; align-items: end; margin-top: .6rem; }
@media (max-width: 560px){ .tts-grid { grid-template-columns: 1fr; } }
textarea.tts { inline-size: 100%; min-block-size: 4.5rem; resize: vertical; font: inherit;
  padding: .6rem .7rem; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .35rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; min-block-size: 2.4rem; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.chip .chip-lang { font-family: var(--font-mono); color: var(--muted); font-size: .72rem; margin-inline-end: .3rem; }
.audio-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-top: .7rem; }
.audio-row audio { flex: 1; min-inline-size: 220px; }
.wave-wrap { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .4rem; margin-top: .6rem; }
canvas.wave { inline-size: 100%; block-size: 96px; display: block; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.run-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
.run-card { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .6rem; }
.run-head { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: .74rem; color: var(--muted); margin-bottom: .3rem; }
.run-card canvas { inline-size: 100%; block-size: 54px; display: block; }
.dur-bar-wrap { block-size: .7rem; border-radius: 999px; background: var(--bg-secondary); border: 1px solid var(--border); overflow: hidden; margin-top: .35rem; }
.dur-bar { display:block; block-size:100%; background: var(--accent); }
label.field { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; }
`;
