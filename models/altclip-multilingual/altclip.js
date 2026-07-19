// Front-end helpers for the multilingual zero-shot image pages. Keeps each page thin: it owns the
// worker handshake, turns files/samples into data URLs, and renders the similarity bars. ALL inference
// runs off the main thread in worker.js.
//
// Model: jinaai/jina-clip-v2 (JinaCLIPModel), WASM backend, q8. AltCLIP itself has no Transformers.js
// model class (see the page's honesty note), so this demo uses jina-clip-v2 — a genuinely multilingual
// CLIP (89 languages) — for the same capability: match an image to text labels written in ANY language.
//
// jina-clip-v2 returns L2-normalised image + text embeddings (NOT logits_per_image like OpenAI CLIP),
// so the worker computes the cosine similarity itself. Same concept in different languages → similar
// score against the same image: that's the whole point.

const WORKER_URL = "/web-ai-showcase/models/altclip-multilingual/worker.js";

export class MultilingualClipEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
  /** Score an image against text labels (any languages). Returns { labels, cosines, probs, dim, ms, device }. */
  classify(imageURL, labels) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, labels });
    });
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Parse a comma/newline label field into a clean, de-duped list. */
export function parseLabels(text) {
  return [...new Set(text.split(/[,\n]/).map((s) => s.trim()).filter(Boolean))];
}

/** Softmax over the cosine similarities → relative percentages for the bars. */
export function softmax(arr, temp = 20) {
  const scaled = arr.map((v) => v * temp);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

/**
 * Render similarity bars. `items` = [{label, cosine, prob, lang?}], sorted by cosine. The bar length is
 * the cosine (mapped 0..1); the % is the softmax share.
 */
export function renderBars(container, items) {
  const sorted = [...items].sort((a, b) => b.cosine - a.cosine);
  const top = sorted[0]?.label;
  container.replaceChildren(...sorted.map((it) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (it.label === top ? " bar-top" : "");
    const pct = (it.prob * 100).toFixed(1);
    const cos = it.cosine.toFixed(3);
    const width = Math.max(2, Math.min(100, it.cosine * 100));
    const lang = it.lang ? `<span class="bar-lang">${escapeHTML(it.lang)}</span>` : "";
    row.innerHTML = `
      <div class="bar-head">
        <span class="bar-label" dir="auto">${lang}${escapeHTML(it.label)}</span>
        <span class="bar-val">cos ${cos} · ${pct}%</span>
      </div>
      <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100"
           aria-valuenow="${width.toFixed(0)}" aria-label="${escapeHTML(it.label)}: cosine ${cos}">
        <div class="bar-fill" style="inline-size:${width.toFixed(1)}%"></div>
      </div>`;
    return row;
  }));
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// A small phrasebook: one concept written in several languages. Used by the "5 languages" demo and the
// presets so the cross-lingual behaviour is easy to feel. (Translations are common dictionary forms.)
export const PHRASEBOOK = {
  "a cat": {
    en: "a cat",
    es: "un gato",
    fr: "un chat",
    de: "eine Katze",
    zh: "一只猫",
    ja: "猫",
    ar: "قطة",
    ru: "кошка",
    hi: "एक बिल्ली",
  },
  "a dog": {
    en: "a dog",
    es: "un perro",
    fr: "un chien",
    de: "ein Hund",
    zh: "一只狗",
    ja: "犬",
    ar: "كلب",
    ru: "собака",
    hi: "एक कुत्ता",
  },
  "a city street": {
    en: "a city street",
    es: "una calle de la ciudad",
    fr: "une rue de la ville",
    de: "eine Stadtstraße",
    zh: "城市街道",
    ja: "街の通り",
    ar: "شارع المدينة",
    ru: "городская улица",
    hi: "एक शहर की सड़क",
  },
  "a beach at sunset": {
    en: "a beach at sunset",
    es: "una playa al atardecer",
    fr: "une plage au coucher du soleil",
    de: "ein Strand bei Sonnenuntergang",
    zh: "日落时的海滩",
    ja: "夕暮れのビーチ",
    ar: "شاطئ عند غروب الشمس",
    ru: "пляж на закате",
    hi: "सूर्यास्त के समय समुद्र तट",
  },
  "a bowl of food": {
    en: "a bowl of food",
    es: "un plato de comida",
    fr: "un bol de nourriture",
    de: "eine Schüssel Essen",
    zh: "一碗食物",
    ja: "食べ物のボウル",
    ar: "وعاء من الطعام",
    ru: "миска еды",
    hi: "भोजन का एक कटोरा",
  },
  "a cozy room": {
    en: "a cozy room",
    es: "una habitación acogedora",
    fr: "une pièce confortable",
    de: "ein gemütliches Zimmer",
    zh: "一个舒适的房间",
    ja: "居心地の良い部屋",
    ar: "غرفة مريحة",
    ru: "уютная комната",
    hi: "एक आरामदायक कमरा",
  },
};

export const LANG_NAMES = {
  en: "EN",
  es: "ES",
  fr: "FR",
  de: "DE",
  zh: "ZH",
  ja: "JA",
  ar: "AR",
  ru: "RU",
  hi: "HI",
};

// Shared inline styles for the multilingual widgets (injected once per page).
export const ALTCLIP_CSS = `
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.preview-wrap { position:relative; display:inline-block; max-inline-size:100%; }
.preview-img { max-inline-size:100%; max-block-size:340px; border-radius:8px; display:block; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px; border:2px solid transparent;
  cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.bars { display:flex; flex-direction:column; gap:.55rem; margin-top:.5rem; }
.bar-head { display:flex; justify-content:space-between; gap:.5rem; font-size:.85rem; }
.bar-label { font-family:var(--font-body); min-inline-size:0; }
.bar-lang { display:inline-block; font-family:var(--font-mono); font-size:.68rem; color:var(--accent-ink,var(--accent));
  border:1px solid var(--border); border-radius:4px; padding:0 .3rem; margin-inline-end:.4rem; }
.bar-val { font-family:var(--font-mono); color:var(--muted); white-space:nowrap; }
.bar-track { block-size:.7rem; background:var(--bg-raised); border:1px solid var(--border); border-radius:999px;
  overflow:hidden; margin-top:.15rem; }
.bar-fill { block-size:100%; background:var(--muted); border-radius:999px; transition:inline-size .35s ease; }
.bar-top .bar-fill { background:var(--accent); }
.bar-top .bar-label { font-weight:600; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:end; margin:.6rem 0; }
textarea { inline-size:100%; font:inherit; padding:.4rem .5rem; border-radius:8px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.chip { font:inherit; font-size:.8rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.warn-box { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:.7rem .9rem;
  margin:.6rem 0; font-size:.85rem; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.6rem 0; }
.metric { border:1px solid var(--border); border-radius:8px; padding:.6rem .7rem; background:var(--bg-raised); }
.metric .k { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.metric .v { font-family:var(--font-mono); font-size:1.15rem; font-weight:600; }
`;
