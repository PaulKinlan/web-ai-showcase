// Front-end helpers for the spoken-language-identification pages. Keeps each page thin: it owns the
// worker handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws the
// waveform, maps the 126 ISO-639-3 codes to readable language names, and renders the per-language score
// bars. All inference lives in worker.js (off the main thread). This is the AUDIO language detector —
// distinct from the text language-identification page: it listens, it doesn't read.

const WORKER_URL = "/web-ai-showcase/models/spoken-language-id/worker.js";
const TARGET_RATE = 16000;

export class LidEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
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
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Identify the language of a 16 kHz mono Float32Array. Returns { all, entropy, margin, ms, device, durationS }. */
  identify(audio) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: audio.buffer }, [audio.buffer]);
    });
  }
}

let _audioCtx = null;
function audioCtx() {
  if (!_audioCtx) {
    const AC = self.AudioContext || self.webkitAudioContext;
    _audioCtx = new AC();
  }
  return _audioCtx;
}

/** Decode any browser-supported audio ArrayBuffer to a 16 kHz mono Float32Array. */
export async function decodeToMono16k(arrayBuffer) {
  const decoded = await audioCtx().decodeAudioData(arrayBuffer.slice(0));
  const frames = Math.ceil(decoded.duration * TARGET_RATE);
  const off = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { pcm: rendered.getChannelData(0), duration: decoded.duration };
}

export async function urlToMono16k(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return decodeToMono16k(buf);
}

export async function blobToMono16k(blob) {
  return decodeToMono16k(await blob.arrayBuffer());
}

/** A tiny mic recorder: start() then stop() → { blob, url }. Honest about missing mic support. */
export class MicRecorder {
  constructor() {
    this.rec = null;
    this.chunks = [];
    this.stream = null;
  }
  static supported() {
    return !!(navigator.mediaDevices?.getUserMedia && self.MediaRecorder);
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.addEventListener("dataavailable", (e) => {
      if (e.data.size) this.chunks.push(e.data);
    });
    this.rec.start();
  }
  stop() {
    return new Promise((resolve) => {
      this.rec.addEventListener("stop", () => {
        const type = this.rec.mimeType || "audio/webm";
        const blob = new Blob(this.chunks, { type });
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
        resolve({ blob, url: URL.createObjectURL(blob) });
      }, { once: true });
      this.rec.stop();
    });
  }
}

// ISO-639-3 → readable language name for all 126 MMS-LID classes.
export const LANG_NAMES = {
  abk: "Abkhaz",
  afr: "Afrikaans",
  amh: "Amharic",
  ara: "Arabic",
  asm: "Assamese",
  ast: "Asturian",
  aze: "Azerbaijani",
  bak: "Bashkir",
  bel: "Belarusian",
  ben: "Bengali",
  bod: "Tibetan",
  bos: "Bosnian",
  bre: "Breton",
  bul: "Bulgarian",
  cat: "Catalan",
  ceb: "Cebuano",
  ces: "Czech",
  ckb: "Central Kurdish",
  cmn: "Mandarin Chinese",
  cym: "Welsh",
  dan: "Danish",
  deu: "German",
  ell: "Greek",
  eng: "English",
  epo: "Esperanto",
  est: "Estonian",
  eus: "Basque",
  fao: "Faroese",
  fas: "Persian",
  fin: "Finnish",
  fra: "French",
  ful: "Fula",
  gle: "Irish",
  glg: "Galician",
  glv: "Manx",
  grn: "Guarani",
  guj: "Gujarati",
  hat: "Haitian Creole",
  hau: "Hausa",
  haw: "Hawaiian",
  heb: "Hebrew",
  hin: "Hindi",
  hrv: "Croatian",
  hun: "Hungarian",
  hye: "Armenian",
  ibo: "Igbo",
  ina: "Interlingua",
  ind: "Indonesian",
  isl: "Icelandic",
  ita: "Italian",
  jav: "Javanese",
  jpn: "Japanese",
  kam: "Kamba",
  kan: "Kannada",
  kat: "Georgian",
  kaz: "Kazakh",
  kea: "Kabuverdianu",
  khm: "Khmer",
  kir: "Kyrgyz",
  kor: "Korean",
  lao: "Lao",
  lat: "Latin",
  lav: "Latvian",
  lin: "Lingala",
  lit: "Lithuanian",
  ltz: "Luxembourgish",
  lug: "Ganda",
  luo: "Luo",
  mal: "Malayalam",
  mar: "Marathi",
  mkd: "Macedonian",
  mlg: "Malagasy",
  mlt: "Maltese",
  mon: "Mongolian",
  mri: "Māori",
  mya: "Burmese",
  nld: "Dutch",
  nno: "Norwegian Nynorsk",
  nob: "Norwegian Bokmål",
  npi: "Nepali",
  nso: "Northern Sotho",
  nya: "Chichewa",
  oci: "Occitan",
  orm: "Oromo",
  ory: "Odia",
  pan: "Punjabi",
  pol: "Polish",
  por: "Portuguese",
  pus: "Pashto",
  ron: "Romanian",
  rus: "Russian",
  san: "Sanskrit",
  sco: "Scots",
  sin: "Sinhala",
  slk: "Slovak",
  slv: "Slovenian",
  sna: "Shona",
  snd: "Sindhi",
  som: "Somali",
  spa: "Spanish",
  sqi: "Albanian",
  srp: "Serbian",
  sun: "Sundanese",
  swe: "Swedish",
  swh: "Swahili",
  tam: "Tamil",
  tat: "Tatar",
  tel: "Telugu",
  tgk: "Tajik",
  tgl: "Tagalog",
  tha: "Thai",
  tuk: "Turkmen",
  tur: "Turkish",
  ukr: "Ukrainian",
  umb: "Umbundu",
  urd: "Urdu",
  uzb: "Uzbek",
  vie: "Vietnamese",
  war: "Waray",
  wol: "Wolof",
  xho: "Xhosa",
  yid: "Yiddish",
  yor: "Yoruba",
  yue: "Cantonese",
  zlm: "Malay",
  zul: "Zulu",
};

export function langName(code) {
  return LANG_NAMES[code] || code;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
export function drawWaveform(canvas, pcm) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 80;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pcm || !pcm.length) return;
  const mid = h / 2, step = Math.max(1, Math.floor(pcm.length / w));
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.9;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = pcm[x * step + i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** Render top language scores as accessible bars into `container`. `items` = [{code, score}]. */
export function renderLangBars(container, items, max = 6) {
  const top = items[0]?.code;
  container.replaceChildren(
    ...items.slice(0, max).map((it) => {
      const row = document.createElement("div");
      row.className = "lang-row" + (it.code === top ? " lang-top" : "");
      const pct = (it.score * 100).toFixed(1);
      const name = langName(it.code);
      row.innerHTML = `
        <div class="lang-head">
          <span class="lang-label">${escapeHTML(name)} <span class="lang-code">${
        escapeHTML(it.code)
      }</span></span>
          <span class="lang-val">${pct}%</span>
        </div>
        <div class="lang-track" role="meter" aria-valuemin="0" aria-valuemax="100"
             aria-valuenow="${pct}" aria-label="${escapeHTML(name)}: ${pct} percent">
          <div class="lang-fill" style="inline-size:${Math.max(1, it.score * 100)}%"></div>
        </div>`;
      return row;
    }),
  );
}

export const LID_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:80px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; max-inline-size:100%; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.chip { font:inherit; font-size:.82rem; padding:.35rem .7rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed="true"] { border-color:var(--accent); background:var(--bg-secondary); font-weight:600; }
.langs { display:flex; flex-direction:column; gap:.5rem; margin:.5rem 0; }
.lang-head { display:flex; justify-content:space-between; gap:.5rem; font-size:.85rem; }
.lang-label { font-family:var(--font-body); }
.lang-code { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.lang-val { font-family:var(--font-mono); color:var(--muted); white-space:nowrap; }
.lang-track { block-size:.7rem; background:var(--bg-raised); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; margin-top:.15rem; }
.lang-fill { block-size:100%; background:var(--muted); border-radius:999px; transition:inline-size .35s ease; }
.lang-top .lang-fill { background:var(--accent); }
.lang-top .lang-label { font-weight:600; }
.big-verdict { font-size:1.4rem; font-weight:600; display:flex; gap:.5rem; align-items:baseline; margin:.2rem 0; flex-wrap:wrap; }
.big-verdict .bv-code { font-family:var(--font-mono); font-size:.9rem; color:var(--muted); font-weight:400; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.conf-meter { block-size:.9rem; border-radius:999px; overflow:hidden; border:1px solid var(--border);
  background:linear-gradient(to right, var(--bad), var(--warn), var(--good)); position:relative; margin:.3rem 0; }
.conf-needle { position:absolute; top:-3px; bottom:-3px; inline-size:3px; background:var(--color); border-radius:2px; }
`;
