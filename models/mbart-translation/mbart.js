// Front-end helpers for the mBART-50 many-to-many translation pages. Owns the worker handshake,
// streaming, the searchable 50-language pickers, and render helpers. All inference runs in worker.js
// (off the main thread).
//
// mBART-50 uses language codes of the form `xx_XX` (e.g. `en_XX`, `fr_XX`, `zh_CN`) — 52 codes for its
// 50 languages. Like NLLB/M2M100 it is many-to-many (no English pivot), and the output language is
// chosen by FORCING the decoder to begin with the target language's special token
// (`forced_bos_token_id`). What makes mBART distinct: it was pretrained by DENOISING — corrupting and
// reconstructing monolingual text in all 50 languages — before being fine-tuned to translate. That
// denoising lineage (BART, scaled multilingual) is different from M2M100/NLLB's translation-centric
// pretraining.

const WORKER_URL = "/web-ai-showcase/models/mbart-translation/worker.js";

// The 52 mBART-50 language codes (code → English name), sorted by name. Taken from the model's own
// tokenizer_config.json `language_codes`, so every entry is a code the model actually accepts.
export const LANGS = [
  ["af_ZA", "Afrikaans"],
  ["ar_AR", "Arabic"],
  ["az_AZ", "Azerbaijani"],
  ["bn_IN", "Bengali"],
  ["my_MM", "Burmese"],
  ["zh_CN", "Chinese"],
  ["hr_HR", "Croatian"],
  ["cs_CZ", "Czech"],
  ["nl_XX", "Dutch"],
  ["en_XX", "English"],
  ["et_EE", "Estonian"],
  ["fi_FI", "Finnish"],
  ["fr_XX", "French"],
  ["gl_ES", "Galician"],
  ["ka_GE", "Georgian"],
  ["de_DE", "German"],
  ["gu_IN", "Gujarati"],
  ["he_IL", "Hebrew"],
  ["hi_IN", "Hindi"],
  ["id_ID", "Indonesian"],
  ["it_IT", "Italian"],
  ["ja_XX", "Japanese"],
  ["kk_KZ", "Kazakh"],
  ["km_KH", "Khmer"],
  ["ko_KR", "Korean"],
  ["lv_LV", "Latvian"],
  ["lt_LT", "Lithuanian"],
  ["mk_MK", "Macedonian"],
  ["ml_IN", "Malayalam"],
  ["mr_IN", "Marathi"],
  ["mn_MN", "Mongolian"],
  ["ne_NP", "Nepali"],
  ["ps_AF", "Pashto"],
  ["fa_IR", "Persian"],
  ["pl_PL", "Polish"],
  ["pt_XX", "Portuguese"],
  ["ro_RO", "Romanian"],
  ["ru_RU", "Russian"],
  ["si_LK", "Sinhala"],
  ["sl_SI", "Slovenian"],
  ["es_XX", "Spanish"],
  ["sw_KE", "Swahili"],
  ["sv_SE", "Swedish"],
  ["tl_XX", "Tagalog"],
  ["ta_IN", "Tamil"],
  ["te_IN", "Telugu"],
  ["th_TH", "Thai"],
  ["tr_TR", "Turkish"],
  ["uk_UA", "Ukrainian"],
  ["ur_PK", "Urdu"],
  ["vi_VN", "Vietnamese"],
  ["xh_ZA", "Xhosa"],
];

export const CODE_TO_NAME = Object.fromEntries(LANGS);
export const NAME_TO_CODE = Object.fromEntries(LANGS.map(([c, n]) => [n.toLowerCase(), c]));
export const LANG_COUNT = LANGS.length;

/** Populate a <datalist> with every language name (value = name) for a native searchable picker. */
export function fillLangDatalist(datalist) {
  datalist.replaceChildren(...LANGS.map(([code, name]) => {
    const o = document.createElement("option");
    o.value = name;
    o.label = code;
    return o;
  }));
}

/** Resolve a picker's text value to an mBART code. Accepts an exact name, a raw code, or null. */
export function resolveCode(value) {
  if (!value) return null;
  const v = value.trim();
  if (CODE_TO_NAME[v]) return v; // already a code
  const byName = NAME_TO_CODE[v.toLowerCase()];
  if (byName) return byName;
  const hit = LANGS.find(([, n]) => n.toLowerCase() === v.toLowerCase());
  return hit ? hit[0] : null;
}

export function nameFor(code) {
  return CODE_TO_NAME[code] || code;
}

/** Wire an <input list> picker so it always reflects a valid language; shows the resolved code. */
export function bindPicker(input, codeOut, initialCode) {
  if (initialCode) input.value = nameFor(initialCode);
  const sync = () => {
    const code = resolveCode(input.value);
    if (codeOut) codeOut.textContent = code || "—";
    input.setAttribute("aria-invalid", code ? "false" : "true");
    return code;
  };
  input.addEventListener("input", sync);
  input.addEventListener("change", sync);
  sync();
  return sync;
}

export class MBartEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._streams = new Map();
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
    } else if (msg.type === "stream") {
      this._streams.get(msg.id)?.(msg.text);
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
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

  /** Translate text between two mBART codes. onStream(partial) receives streamed greedy output. */
  translate(text, srcCode, tgtCode, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, srcCode, tgtCode, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the language/token/timing readout for a completed translation. */
export function renderStats(els, r) {
  if (els.codes) els.codes.textContent = `${r.srcCode} → ${r.tgtCode}`;
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
  if (els.bos) {
    els.bos.textContent = r.forcedBos != null ? `#${r.forcedBos} (${r.tgtCode})` : "–";
  }
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
}

export const MBART_CSS = `
.tr-io { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.tr-io textarea { inline-size:100%; min-block-size:130px; resize:vertical; font-family:var(--font-body); }
.tr-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:130px; white-space:pre-wrap; font-size:1.05rem; overflow-wrap:anywhere; }
.tr-out:empty::before { content:"The translation will stream in here."; color:var(--muted); font-size:.95rem; }
.lang-row { display:grid; gap:.9rem 1.2rem; grid-template-columns:1fr auto 1fr; align-items:end; margin:.6rem 0; }
.lang-row label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; min-inline-size:0; }
.lang-row input { inline-size:100%; }
.lang-row .code { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.lang-row input[aria-invalid="true"] { border-color:var(--bad); }
.swap-btn { align-self:end; min-block-size:2.4rem; min-inline-size:2.4rem; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; overflow-wrap:anywhere; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.25rem .65rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2rem; }
.chip:hover, .chip:focus-visible { border-color:var(--accent); }
.relay-step { border:1px solid var(--border); border-inline-start:4px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); padding:.5rem .7rem; margin-top:.5rem; overflow-wrap:anywhere; }
.relay-step .meta { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.diff-same { color:var(--good); } .diff-drift { color:var(--warn); }
`;
