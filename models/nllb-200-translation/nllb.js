// Front-end helpers for the NLLB-200 translation pages. Owns the worker handshake, streaming, the
// searchable 200-language pickers (FLORES-200 codes), and render helpers. All inference runs in
// worker.js (off the main thread). NLLB uses FLORES-200 language codes like `eng_Latn`, `zul_Latn`,
// `yor_Latn` — three-letter ISO 639-3 language + four-letter ISO 15924 script — and a forced target
// BOS token selects the output language.

const WORKER_URL = "/web-ai-showcase/models/nllb-200-translation/worker.js";

// All 202 FLORES-200 languages NLLB-200 supports (code → English name), sorted by name. Extracted from
// the model's own special_tokens_map.json, so every entry is a code the model actually accepts.
export const LANGS = [
  ["ace_Arab", "Acehnese (Arabic)"],
  ["ace_Latn", "Acehnese (Latin)"],
  ["afr_Latn", "Afrikaans"],
  ["aka_Latn", "Akan"],
  ["amh_Ethi", "Amharic"],
  ["hye_Armn", "Armenian"],
  ["asm_Beng", "Assamese"],
  ["ast_Latn", "Asturian"],
  ["awa_Deva", "Awadhi"],
  ["quy_Latn", "Ayacucho Quechua"],
  ["ban_Latn", "Balinese"],
  ["bam_Latn", "Bambara"],
  ["bjn_Arab", "Banjar (Arabic)"],
  ["bjn_Latn", "Banjar (Latin)"],
  ["bak_Cyrl", "Bashkir"],
  ["eus_Latn", "Basque"],
  ["bel_Cyrl", "Belarusian"],
  ["bem_Latn", "Bemba"],
  ["ben_Beng", "Bengali"],
  ["bho_Deva", "Bhojpuri"],
  ["bos_Latn", "Bosnian"],
  ["bug_Latn", "Buginese"],
  ["bul_Cyrl", "Bulgarian"],
  ["mya_Mymr", "Burmese"],
  ["yue_Hant", "Cantonese"],
  ["cat_Latn", "Catalan"],
  ["ceb_Latn", "Cebuano"],
  ["tzm_Tfng", "Central Atlas Tamazight"],
  ["ayr_Latn", "Central Aymara"],
  ["knc_Arab", "Central Kanuri (Arabic)"],
  ["knc_Latn", "Central Kanuri (Latin)"],
  ["ckb_Arab", "Central Kurdish"],
  ["hne_Deva", "Chhattisgarhi"],
  ["zho_Hans", "Chinese (Simplified)"],
  ["zho_Hant", "Chinese (Traditional)"],
  ["cjk_Latn", "Chokwe"],
  ["crh_Latn", "Crimean Tatar"],
  ["hrv_Latn", "Croatian"],
  ["ces_Latn", "Czech"],
  ["dan_Latn", "Danish"],
  ["prs_Arab", "Dari"],
  ["nld_Latn", "Dutch"],
  ["dyu_Latn", "Dyula"],
  ["dzo_Tibt", "Dzongkha"],
  ["pan_Guru", "Eastern Panjabi"],
  ["ydd_Hebr", "Eastern Yiddish"],
  ["arz_Arab", "Egyptian Arabic"],
  ["eng_Latn", "English"],
  ["epo_Latn", "Esperanto"],
  ["est_Latn", "Estonian"],
  ["ewe_Latn", "Ewe"],
  ["fao_Latn", "Faroese"],
  ["fij_Latn", "Fijian"],
  ["fin_Latn", "Finnish"],
  ["fon_Latn", "Fon"],
  ["fra_Latn", "French"],
  ["fur_Latn", "Friulian"],
  ["glg_Latn", "Galician"],
  ["lug_Latn", "Ganda"],
  ["kat_Geor", "Georgian"],
  ["deu_Latn", "German"],
  ["ell_Grek", "Greek"],
  ["grn_Latn", "Guarani"],
  ["guj_Gujr", "Gujarati"],
  ["hat_Latn", "Haitian Creole"],
  ["khk_Cyrl", "Halh Mongolian"],
  ["hau_Latn", "Hausa"],
  ["heb_Hebr", "Hebrew"],
  ["hin_Deva", "Hindi"],
  ["hun_Latn", "Hungarian"],
  ["isl_Latn", "Icelandic"],
  ["ibo_Latn", "Igbo"],
  ["ilo_Latn", "Ilocano"],
  ["ind_Latn", "Indonesian"],
  ["gle_Latn", "Irish"],
  ["ita_Latn", "Italian"],
  ["jpn_Jpan", "Japanese"],
  ["jav_Latn", "Javanese"],
  ["kac_Latn", "Jingpho"],
  ["kbp_Latn", "Kabiye"],
  ["kea_Latn", "Kabuverdianu"],
  ["kab_Latn", "Kabyle"],
  ["kam_Latn", "Kamba"],
  ["kan_Knda", "Kannada"],
  ["kas_Arab", "Kashmiri (Arabic)"],
  ["kas_Deva", "Kashmiri (Devanagari)"],
  ["kaz_Cyrl", "Kazakh"],
  ["khm_Khmr", "Khmer"],
  ["kon_Latn", "Kikongo"],
  ["kik_Latn", "Kikuyu"],
  ["kmb_Latn", "Kimbundu"],
  ["kin_Latn", "Kinyarwanda"],
  ["kor_Hang", "Korean"],
  ["kir_Cyrl", "Kyrgyz"],
  ["lao_Laoo", "Lao"],
  ["ltg_Latn", "Latgalian"],
  ["lij_Latn", "Ligurian"],
  ["lim_Latn", "Limburgish"],
  ["lin_Latn", "Lingala"],
  ["lit_Latn", "Lithuanian"],
  ["lmo_Latn", "Lombard"],
  ["lua_Latn", "Luba-Kasai"],
  ["luo_Latn", "Luo"],
  ["ltz_Latn", "Luxembourgish"],
  ["mkd_Cyrl", "Macedonian"],
  ["mag_Deva", "Magahi"],
  ["mai_Deva", "Maithili"],
  ["mal_Mlym", "Malayalam"],
  ["mlt_Latn", "Maltese"],
  ["mri_Latn", "Maori"],
  ["mar_Deva", "Marathi"],
  ["mni_Beng", "Meitei (Bengali)"],
  ["acm_Arab", "Mesopotamian Arabic"],
  ["min_Latn", "Minangkabau"],
  ["lus_Latn", "Mizo"],
  ["arb_Arab", "Modern Standard Arabic"],
  ["ary_Arab", "Moroccan Arabic"],
  ["mos_Latn", "Mossi"],
  ["ars_Arab", "Najdi Arabic"],
  ["npi_Deva", "Nepali"],
  ["fuv_Latn", "Nigerian Fulfulde"],
  ["azj_Latn", "North Azerbaijani"],
  ["apc_Arab", "North Levantine Arabic"],
  ["kmr_Latn", "Northern Kurdish"],
  ["nso_Latn", "Northern Sotho"],
  ["uzn_Latn", "Northern Uzbek"],
  ["nob_Latn", "Norwegian Bokmal"],
  ["nno_Latn", "Norwegian Nynorsk"],
  ["nus_Latn", "Nuer"],
  ["nya_Latn", "Nyanja (Chichewa)"],
  ["oci_Latn", "Occitan"],
  ["ory_Orya", "Odia"],
  ["pag_Latn", "Pangasinan"],
  ["pap_Latn", "Papiamento"],
  ["plt_Latn", "Plateau Malagasy"],
  ["pol_Latn", "Polish"],
  ["por_Latn", "Portuguese"],
  ["ron_Latn", "Romanian"],
  ["run_Latn", "Rundi"],
  ["rus_Cyrl", "Russian"],
  ["smo_Latn", "Samoan"],
  ["sag_Latn", "Sango"],
  ["san_Deva", "Sanskrit"],
  ["sat_Beng", "Santali"],
  ["srd_Latn", "Sardinian"],
  ["gla_Latn", "Scottish Gaelic"],
  ["srp_Cyrl", "Serbian"],
  ["shn_Mymr", "Shan"],
  ["sna_Latn", "Shona"],
  ["scn_Latn", "Sicilian"],
  ["szl_Latn", "Silesian"],
  ["snd_Arab", "Sindhi"],
  ["sin_Sinh", "Sinhala"],
  ["slk_Latn", "Slovak"],
  ["slv_Latn", "Slovenian"],
  ["som_Latn", "Somali"],
  ["azb_Arab", "South Azerbaijani"],
  ["ajp_Arab", "South Levantine Arabic"],
  ["pbt_Arab", "Southern Pashto"],
  ["sot_Latn", "Southern Sotho"],
  ["dik_Latn", "Southwestern Dinka"],
  ["spa_Latn", "Spanish"],
  ["lvs_Latn", "Standard Latvian"],
  ["zsm_Latn", "Standard Malay"],
  ["bod_Tibt", "Standard Tibetan"],
  ["sun_Latn", "Sundanese"],
  ["swh_Latn", "Swahili"],
  ["ssw_Latn", "Swati"],
  ["swe_Latn", "Swedish"],
  ["acq_Arab", "Ta'izzi-Adeni Arabic"],
  ["tgl_Latn", "Tagalog"],
  ["tgk_Cyrl", "Tajik"],
  ["taq_Latn", "Tamasheq (Latin)"],
  ["taq_Tfng", "Tamasheq (Tifinagh)"],
  ["tam_Taml", "Tamil"],
  ["tat_Cyrl", "Tatar"],
  ["tel_Telu", "Telugu"],
  ["tha_Thai", "Thai"],
  ["tir_Ethi", "Tigrinya"],
  ["tpi_Latn", "Tok Pisin"],
  ["als_Latn", "Tosk Albanian"],
  ["tso_Latn", "Tsonga"],
  ["tsn_Latn", "Tswana"],
  ["tum_Latn", "Tumbuka"],
  ["aeb_Arab", "Tunisian Arabic"],
  ["tur_Latn", "Turkish"],
  ["tuk_Latn", "Turkmen"],
  ["twi_Latn", "Twi"],
  ["ukr_Cyrl", "Ukrainian"],
  ["umb_Latn", "Umbundu"],
  ["urd_Arab", "Urdu"],
  ["uig_Arab", "Uyghur"],
  ["vec_Latn", "Venetian"],
  ["vie_Latn", "Vietnamese"],
  ["war_Latn", "Waray"],
  ["cym_Latn", "Welsh"],
  ["gaz_Latn", "West Central Oromo"],
  ["pes_Arab", "Western Persian"],
  ["wol_Latn", "Wolof"],
  ["xho_Latn", "Xhosa"],
  ["yor_Latn", "Yoruba"],
  ["zul_Latn", "Zulu"],
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

/** Resolve a picker's text value to a FLORES code. Accepts an exact name, a raw code, or null. */
export function resolveCode(value) {
  if (!value) return null;
  const v = value.trim();
  if (CODE_TO_NAME[v]) return v; // already a code
  const byName = NAME_TO_CODE[v.toLowerCase()];
  if (byName) return byName;
  // tolerate a trailing "(code)" or partial: match the first name that starts with the text
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

export class NllbEngine {
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

  /** Translate text between two FLORES codes. onStream(partial) receives streamed greedy output. */
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
  if (els.codes) els.codes.textContent = `${r.srcCode} \u2192 ${r.tgtCode}`;
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
  if (els.bos) {
    els.bos.textContent = r.forcedBos != null ? `#${r.forcedBos} (${r.tgtCode})` : "\u2013";
  }
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "\u2013";
  }
}

export const NLLB_CSS = `
.tr-io { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.tr-io textarea { inline-size:100%; min-block-size:130px; resize:vertical; font-family:var(--font-body); }
.tr-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:130px; white-space:pre-wrap; font-size:1.05rem; }
.tr-out:empty::before { content:"The translation will stream in here."; color:var(--muted); font-size:.95rem; }
.lang-row { display:grid; gap:.9rem 1.2rem; grid-template-columns:1fr auto 1fr; align-items:end; margin:.6rem 0; }
.lang-row label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.lang-row input { inline-size:100%; }
.lang-row .code { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.lang-row input[aria-invalid="true"] { border-color:var(--bad); }
.swap-btn { align-self:end; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.relay-step, .telephone-step { border:1px solid var(--border); border-inline-start:4px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); padding:.5rem .7rem; margin-top:.5rem; }
.relay-step .meta { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.diff-same { color:var(--good); } .diff-drift { color:var(--warn); }
`;
