// Front-end helpers for the mBERT language-detection pages. Owns the worker handshake and the renderers
// (top result, per-language probability bars). All inference lives in worker.js.
//
// This model (onnx-community/language_detection-ONNX, a BERT fine-tune) emits 201 language labels using
// FLORES-200 codes of the form <iso639-3>_<Script> (e.g. fra_Latn, deu_Latn, zho_Hans, rus_Cyrl). We map
// the common ones to friendly names + a flag for display, and fall back to a parsed "ISO3 · Script" label
// for the long tail — the raw model code is always shown too, so nothing is hidden or invented.

const WORKER_URL = "/web-ai-showcase/models/mbert-language-detection/worker.js";

// Curated display names + flags for common FLORES-200 codes. The model knows 201; this covers the ones a
// visitor is most likely to try. Anything not here falls back to a parsed code (see langName/langFlag).
export const LANGS = {
  eng_Latn: { name: "English", flag: "🇬🇧" },
  fra_Latn: { name: "French", flag: "🇫🇷" },
  deu_Latn: { name: "German", flag: "🇩🇪" },
  spa_Latn: { name: "Spanish", flag: "🇪🇸" },
  ita_Latn: { name: "Italian", flag: "🇮🇹" },
  por_Latn: { name: "Portuguese", flag: "🇵🇹" },
  nld_Latn: { name: "Dutch", flag: "🇳🇱" },
  rus_Cyrl: { name: "Russian", flag: "🇷🇺" },
  ukr_Cyrl: { name: "Ukrainian", flag: "🇺🇦" },
  pol_Latn: { name: "Polish", flag: "🇵🇱" },
  ces_Latn: { name: "Czech", flag: "🇨🇿" },
  slk_Latn: { name: "Slovak", flag: "🇸🇰" },
  ron_Latn: { name: "Romanian", flag: "🇷🇴" },
  hun_Latn: { name: "Hungarian", flag: "🇭🇺" },
  ell_Grek: { name: "Greek", flag: "🇬🇷" },
  bul_Cyrl: { name: "Bulgarian", flag: "🇧🇬" },
  srp_Cyrl: { name: "Serbian", flag: "🇷🇸" },
  hrv_Latn: { name: "Croatian", flag: "🇭🇷" },
  bos_Latn: { name: "Bosnian", flag: "🇧🇦" },
  slv_Latn: { name: "Slovenian", flag: "🇸🇮" },
  swe_Latn: { name: "Swedish", flag: "🇸🇪" },
  dan_Latn: { name: "Danish", flag: "🇩🇰" },
  nob_Latn: { name: "Norwegian (Bokmål)", flag: "🇳🇴" },
  nno_Latn: { name: "Norwegian (Nynorsk)", flag: "🇳🇴" },
  fin_Latn: { name: "Finnish", flag: "🇫🇮" },
  est_Latn: { name: "Estonian", flag: "🇪🇪" },
  lvs_Latn: { name: "Latvian", flag: "🇱🇻" },
  isl_Latn: { name: "Icelandic", flag: "🇮🇸" },
  gle_Latn: { name: "Irish", flag: "🇮🇪" },
  cym_Latn: { name: "Welsh", flag: "🏴󠁧󠁢󠁷󠁬󠁳󠁿" },
  cat_Latn: { name: "Catalan", flag: "🇪🇸" },
  eus_Latn: { name: "Basque", flag: "🇪🇸" },
  glg_Latn: { name: "Galician", flag: "🇪🇸" },
  tur_Latn: { name: "Turkish", flag: "🇹🇷" },
  azj_Latn: { name: "Azerbaijani", flag: "🇦🇿" },
  kaz_Cyrl: { name: "Kazakh", flag: "🇰🇿" },
  uzn_Latn: { name: "Uzbek", flag: "🇺🇿" },
  arb_Arab: { name: "Arabic (MSA)", flag: "🇸🇦" },
  ary_Arab: { name: "Arabic (Moroccan)", flag: "🇲🇦" },
  arz_Arab: { name: "Arabic (Egyptian)", flag: "🇪🇬" },
  heb_Hebr: { name: "Hebrew", flag: "🇮🇱" },
  pes_Arab: { name: "Persian", flag: "🇮🇷" },
  urd_Arab: { name: "Urdu", flag: "🇵🇰" },
  pbt_Arab: { name: "Pashto", flag: "🇦🇫" },
  ckb_Arab: { name: "Central Kurdish", flag: "🏳️" },
  hin_Deva: { name: "Hindi", flag: "🇮🇳" },
  mar_Deva: { name: "Marathi", flag: "🇮🇳" },
  npi_Deva: { name: "Nepali", flag: "🇳🇵" },
  san_Deva: { name: "Sanskrit", flag: "🇮🇳" },
  ben_Beng: { name: "Bengali", flag: "🇧🇩" },
  asm_Beng: { name: "Assamese", flag: "🇮🇳" },
  guj_Gujr: { name: "Gujarati", flag: "🇮🇳" },
  pan_Guru: { name: "Punjabi", flag: "🇮🇳" },
  ory_Orya: { name: "Odia", flag: "🇮🇳" },
  tam_Taml: { name: "Tamil", flag: "🇮🇳" },
  tel_Telu: { name: "Telugu", flag: "🇮🇳" },
  kan_Knda: { name: "Kannada", flag: "🇮🇳" },
  mal_Mlym: { name: "Malayalam", flag: "🇮🇳" },
  sin_Sinh: { name: "Sinhala", flag: "🇱🇰" },
  tha_Thai: { name: "Thai", flag: "🇹🇭" },
  lao_Laoo: { name: "Lao", flag: "🇱🇦" },
  mya_Mymr: { name: "Burmese", flag: "🇲🇲" },
  khm_Khmr: { name: "Khmer", flag: "🇰🇭" },
  vie_Latn: { name: "Vietnamese", flag: "🇻🇳" },
  ind_Latn: { name: "Indonesian", flag: "🇮🇩" },
  zsm_Latn: { name: "Malay", flag: "🇲🇾" },
  tgl_Latn: { name: "Tagalog", flag: "🇵🇭" },
  ceb_Latn: { name: "Cebuano", flag: "🇵🇭" },
  jav_Latn: { name: "Javanese", flag: "🇮🇩" },
  sun_Latn: { name: "Sundanese", flag: "🇮🇩" },
  zho_Hans: { name: "Chinese (Simplified)", flag: "🇨🇳" },
  zho_Hant: { name: "Chinese (Traditional)", flag: "🇹🇼" },
  yue_Hant: { name: "Cantonese", flag: "🇭🇰" },
  jpn_Jpan: { name: "Japanese", flag: "🇯🇵" },
  kor_Hang: { name: "Korean", flag: "🇰🇷" },
  swh_Latn: { name: "Swahili", flag: "🇰🇪" },
  hau_Latn: { name: "Hausa", flag: "🇳🇬" },
  yor_Latn: { name: "Yoruba", flag: "🇳🇬" },
  ibo_Latn: { name: "Igbo", flag: "🇳🇬" },
  zul_Latn: { name: "Zulu", flag: "🇿🇦" },
  xho_Latn: { name: "Xhosa", flag: "🇿🇦" },
  afr_Latn: { name: "Afrikaans", flag: "🇿🇦" },
  amh_Ethi: { name: "Amharic", flag: "🇪🇹" },
  som_Latn: { name: "Somali", flag: "🇸🇴" },
  kat_Geor: { name: "Georgian", flag: "🇬🇪" },
  hye_Armn: { name: "Armenian", flag: "🇦🇲" },
  epo_Latn: { name: "Esperanto", flag: "🏳️" },
  mlt_Latn: { name: "Maltese", flag: "🇲🇹" },
  bel_Cyrl: { name: "Belarusian", flag: "🇧🇾" },
  mkd_Cyrl: { name: "Macedonian", flag: "🇲🇰" },
  lit_Latn: { name: "Lithuanian", flag: "🇱🇹" },
  tat_Cyrl: { name: "Tatar", flag: "🏳️" },
  kir_Cyrl: { name: "Kyrgyz", flag: "🇰🇬" },
  tgk_Cyrl: { name: "Tajik", flag: "🇹🇯" },
  mri_Latn: { name: "Māori", flag: "🇳🇿" },
  smo_Latn: { name: "Samoan", flag: "🇼🇸" },
  fij_Latn: { name: "Fijian", flag: "🇫🇯" },
};

// Script code → readable name, for the long-tail fallback label.
const SCRIPTS = {
  Latn: "Latin",
  Cyrl: "Cyrillic",
  Arab: "Arabic",
  Deva: "Devanagari",
  Beng: "Bengali",
  Hang: "Hangul",
  Jpan: "Japanese",
  Hans: "Simplified Han",
  Hant: "Traditional Han",
  Grek: "Greek",
  Hebr: "Hebrew",
  Thai: "Thai",
  Taml: "Tamil",
  Telu: "Telugu",
  Knda: "Kannada",
  Mlym: "Malayalam",
  Gujr: "Gujarati",
  Guru: "Gurmukhi",
  Orya: "Odia",
  Sinh: "Sinhala",
  Mymr: "Myanmar",
  Khmr: "Khmer",
  Laoo: "Lao",
  Geor: "Georgian",
  Armn: "Armenian",
  Ethi: "Ethiopic",
  Tibt: "Tibetan",
  Tfng: "Tifinagh",
  Olck: "Ol Chiki",
};

/** Friendly name for a FLORES-200 code; falls back to "iso3 · ScriptName" for the long tail. */
export function langName(code) {
  if (LANGS[code]) return LANGS[code].name;
  const [iso, script] = String(code).split("_");
  const scriptName = SCRIPTS[script] || script || "";
  return scriptName ? `${iso} · ${scriptName}` : iso || code;
}
export function langFlag(code) {
  return LANGS[code]?.flag || "🏳️";
}

export class LangIdEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
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

  /** Detect → { text, scores:[{label,score} …sorted], ms, device }. */
  detect(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Big headline result: flag + language name + confidence, with an honest "uncertain" hint. */
export function renderTop(container, scores) {
  const top = scores[0];
  const conf = (top.score * 100).toFixed(1);
  const uncertain = top.score < 0.6;
  container.innerHTML = `<div class="li-top ${uncertain ? "unsure" : ""}">` +
    `<span class="li-flag">${langFlag(top.label)}</span>` +
    `<span class="li-name">${langName(top.label)}</span>` +
    `<span class="li-code">${top.label}</span>` +
    `<span class="li-conf">${conf}%</span></div>` +
    (uncertain
      ? `<p class="li-hint muted">Low confidence — the text is short, mixed, or between closely related languages.</p>`
      : "");
}

/** Per-language probability bars for the top-N languages (the softmax distribution). */
export function renderBars(container, scores, topN = 6) {
  const top = scores.slice(0, topN);
  container.replaceChildren(...top.map((s, i) => {
    const row = document.createElement("div");
    row.className = "li-bar-row" + (i === 0 ? " top" : "");
    const pct = (s.score * 100).toFixed(1);
    row.innerHTML = `<span class="li-bar-label">${langFlag(s.label)} ${langName(s.label)} <span class="li-bar-code">${s.label}</span></span>` +
      `<span class="li-bar-track"><span class="li-bar-fill" style="inline-size:${pct}%"></span></span>` +
      `<span class="li-bar-pct">${pct}%</span>`;
    return row;
  }));
}

export const LANGID_CSS = `
.li-top { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; margin-top: .5rem;
  padding: .7rem .9rem; border: 1px solid var(--accent); border-inline-start: 4px solid var(--accent);
  border-radius: 10px; background: var(--bg-raised); }
.li-top.unsure { border-color: var(--border-strong); border-inline-start-color: var(--warn); }
.li-flag { font-size: 1.8rem; line-height: 1; }
.li-name { font-family: var(--font-display); font-size: 1.4rem; }
.li-code { font-family: var(--font-mono); font-size: .8rem; color: var(--muted); }
.li-conf { margin-inline-start: auto; font-family: var(--font-mono); font-size: 1.1rem; font-weight: 600; }
.li-hint { font-size: .82rem; margin: .4rem 0 0; }
.li-bars { display: flex; flex-direction: column; gap: .4rem; margin-top: .5rem; }
.li-bar-row { display: grid; grid-template-columns: minmax(8rem, 12rem) 1fr auto; gap: .6rem; align-items: center; }
.li-bar-label { font-size: .82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.li-bar-code { font-family: var(--font-mono); font-size: .7rem; color: var(--muted); }
.li-bar-track { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.li-bar-fill { display: block; block-size: 100%; background: var(--accent); }
.li-bar-row.top .li-bar-fill { background: var(--accent); }
.li-bar-pct { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); min-inline-size: 3.2rem; text-align: end; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
`;
