// Grapheme-to-phoneme (G2P) helpers: the ByT5 byte tokenizer (shared by worker.js), the language list,
// the worker handshake, and render/CSS helpers. All inference lives in worker.js (off the main thread).
//
// The model is CharsiuG2P (multilingual byT5) — it maps SPELLING to SOUND (IPA phonemes) per word, in a
// chosen language. Like ByT5 it is token-free: text is UTF-8 bytes, each byte -> id = byte + 3, with
// pad=0, eos=1, unk=2. So the "tokenizer" is a fixed byte map (no vocab file), which is why ANY script —
// Latin, Cyrillic, kana, Greek — encodes losslessly and can be phonemized.

const EOS = 1;
const BYTE_OFFSET = 3; // id = byte + 3; ids 3..258 are the 256 byte values

/** A CharsiuG2P query is "<lang-code>: word". Encode it to ByT5 byte ids + EOS. */
export function textToIds(text) {
  const bytes = new TextEncoder().encode(text);
  const ids = new Array(bytes.length + 1);
  for (let i = 0; i < bytes.length; i++) ids[i] = bytes[i] + BYTE_OFFSET;
  ids[bytes.length] = EOS;
  return ids;
}

/** Decode output ids back to text: ids 3..258 are bytes (id - 3); specials (<3, >258) are dropped. */
export function idsToText(ids) {
  const bytes = [];
  for (const id of ids) {
    const n = Number(id);
    if (n >= BYTE_OFFSET && n <= BYTE_OFFSET + 255) bytes.push(n - BYTE_OFFSET);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

// A curated subset of CharsiuG2P's 100 language codes — a spread of scripts and families. The code is the
// literal prefix the model was trained on ("<eng-us>: word").
export const LANGUAGES = [
  { code: "eng-us", name: "English (US)" },
  { code: "eng-uk", name: "English (UK)" },
  { code: "fra", name: "French" },
  { code: "ger", name: "German" },
  { code: "spa", name: "Spanish" },
  { code: "ita", name: "Italian" },
  { code: "por", name: "Portuguese" },
  { code: "dut", name: "Dutch" },
  { code: "swe", name: "Swedish" },
  { code: "pol", name: "Polish" },
  { code: "tur", name: "Turkish" },
  { code: "gre", name: "Greek" },
  { code: "rus", name: "Russian" },
  { code: "ara", name: "Arabic" },
  { code: "hin", name: "Hindi" },
  { code: "jpn", name: "Japanese" },
  { code: "kor", name: "Korean" },
  { code: "vie-n", name: "Vietnamese (N)" },
];

export class G2pEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
  /** Phonemize a whitespace-separated phrase in `code` → { words:[{word,ipa}], ipa, ms }. */
  phonemize(code, text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, code, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const G2P_CSS = `
  .g2p-row { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; margin: 0.6rem 0; }
  .g2p-row input[type=text] { flex: 1; min-width: 12rem; padding: 0.5rem 0.6rem; border-radius: 8px; border: 1px solid #8886; font-size: 1rem; }
  .g2p-row select { padding: 0.5rem; border-radius: 8px; }
  .g2p-out { font-size: 1.6rem; margin: 0.6rem 0; min-height: 2rem; word-break: break-word; }
  .g2p-ipa { font-family: "Doulos SCS", "Charis SIL", "Gentium Plus", serif; }
  .g2p-words { display: flex; flex-wrap: wrap; gap: 0.5rem 0.9rem; margin: 0.4rem 0; }
  .g2p-word { font-size: 0.95rem; }
  .g2p-word b { font-family: "Doulos SCS", "Charis SIL", "Gentium Plus", serif; }
  .g2p-word span { color: var(--muted, #888); }
  .g2p-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .g2p-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
