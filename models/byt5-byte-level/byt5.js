// Front-end helpers for the ByT5 (byte-level seq2seq) pages. Keeps each page thin: the worker
// handshake + the ByT5 byte tokenizer (the star of the demo) + render helpers + CSS. All inference
// lives in worker.js (off the main thread). The byte-tokenizer functions are the SINGLE source of
// truth and are imported by BOTH the main thread (live "see inside" as you type) and worker.js (the
// exact ids fed to the model) — so what you see is exactly what the model reads.
//
// ByT5's tokenizer is TOKENIZER-FREE in the usual sense: there is no learned vocabulary. Text is UTF-8
// encoded to raw bytes, and each byte maps to a token id by a fixed offset. The full "vocab" is 384:
//   id 0 = <pad>, 1 = </s> (eos), 2 = <unk>
//   id 3..258   = the 256 possible byte values (id = byte + 3)
//   id 258+     = span sentinels <extra_id_0>.. — this ONNX model empirically emits/reads id 258 for
//                 <extra_id_0>, and byte 0xFF (which would also be id 258) is never a valid UTF-8 byte,
//                 so ids 3..257 are unambiguously real text bytes and ids >= 258 are span markers.
// Because ANY text is just bytes, nothing is ever out-of-vocabulary — every script, emoji, math
// symbol, or corrupted byte encodes losslessly. That is the property this demo makes legible.

const WORKER_URL = "/web-ai-showcase/models/byt5-byte-level/worker.js";

export const PAD = 0, EOS = 1, UNK = 2, BYTE_OFFSET = 3;
// The 256 byte values map to ids 3..258 (byte + 3). Byte 0xFF (id 258) is never a valid UTF-8 byte,
// and this ONNX model emits id 258 as its <extra_id_0> span sentinel, so we treat ids 3..257 as real
// text bytes and ids >= 258 as span sentinels — which cleanly separates content from fill markers.
export const LAST_BYTE_ID = 257; // 0xFE — highest id we decode back to a text byte
export const SENT0 = 258; // <extra_id_0> — the first span-fill sentinel (as this model emits/reads it)
export const BLANK_MARK = "[BLANK]"; // authoring marker in the input; mapped to <extra_id_0>

/** Text → UTF-8 bytes (Uint8Array). The whole tokenizer, step one. */
export function textToBytes(text) {
  return new TextEncoder().encode(text);
}

/** Text → ByT5 token ids (byte + 3). No vocabulary lookup, no unknown tokens — ever. */
export function textToByteIds(text) {
  return [...textToBytes(text)].map((b) => b + BYTE_OFFSET);
}

/**
 * Split an authored string on BLANK_MARK into ByT5 "parts": literal text runs interleaved with the
 * <extra_id_0> sentinel. Both threads use this so the model reads exactly what the page shows.
 */
export function textToParts(text) {
  const parts = [];
  const chunks = text.split(BLANK_MARK);
  chunks.forEach((c, i) => {
    if (c) parts.push({ t: c });
    if (i < chunks.length - 1) parts.push({ s: SENT0 });
  });
  return parts;
}

/** Parts → the flat id sequence fed to the encoder (bytes for text, sentinel ids for blanks, + eos). */
export function partsToIds(parts) {
  const ids = [];
  for (const p of parts) {
    if (p.s != null) ids.push(p.s);
    else for (const b of textToBytes(p.t)) ids.push(b + BYTE_OFFSET);
  }
  ids.push(EOS);
  return ids;
}

/** Decode a byte-id sequence back to text. Stops at eos; skips pad/unk; renders sentinels as markers. */
export function idsToText(ids, { markSentinels = false } = {}) {
  const bytes = [];
  const out = [];
  const flush = () => {
    if (bytes.length) {
      try {
        out.push(new TextDecoder().decode(new Uint8Array(bytes)));
      } catch {
        out.push("�");
      }
      bytes.length = 0;
    }
  };
  for (const id of ids) {
    if (id === EOS) break;
    if (id === PAD || id === UNK) continue;
    if (id >= BYTE_OFFSET && id <= LAST_BYTE_ID) bytes.push(id - BYTE_OFFSET);
    else if (id >= SENT0) {
      flush();
      if (markSentinels) out.push(`⟦${id - SENT0}⟧`);
    }
  }
  flush();
  return out.join("");
}

/** From a raw denoising output [pad, <extra_id_0>, …fill…, <extra_id_1>, …], take just the first fill. */
export function firstFill(ids) {
  const bytes = [];
  let started = false;
  for (const id of ids) {
    if (id === EOS) break;
    if (id >= SENT0) {
      if (started) break; // reached the next sentinel — stop after the first fill
      started = true;
      continue;
    }
    if (started && id >= BYTE_OFFSET && id <= LAST_BYTE_ID) bytes.push(id - BYTE_OFFSET);
  }
  try {
    return new TextDecoder().decode(new Uint8Array(bytes)).trim();
  } catch {
    return "�";
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** The worker-backed engine: load once (auto-init via createModelLoader), then run(text). */
export class ByT5Engine {
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

  /** Run ByT5 on an authored string (may contain BLANK_MARK). opts: { maxTokens, noRepeat, beams }. */
  run(text, opts = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, opts });
    });
  }
}

/** Render a text as a live byte breakdown: one chip per character showing its UTF-8 bytes + ids. */
export function renderByteChips(container, text, { limit = 120 } = {}) {
  const chars = [...text]; // iterate by code point so emoji/surrogates stay whole
  const frag = document.createDocumentFragment();
  let shown = 0;
  for (const ch of chars) {
    if (shown >= limit) {
      const more = document.createElement("span");
      more.className = "bl-more";
      more.textContent = `+${chars.length - shown} more…`;
      frag.append(more);
      break;
    }
    const bytes = [...new TextEncoder().encode(ch)];
    const chip = document.createElement("span");
    chip.className = "bl-chip";
    const g = document.createElement("span");
    g.className = "bl-glyph";
    g.textContent = ch === " " ? "␠" : ch === "\n" ? "␤" : ch;
    const ids = document.createElement("span");
    ids.className = "bl-ids";
    ids.textContent = bytes.map((b) => b + BYTE_OFFSET).join(" ");
    chip.append(g, ids);
    chip.title = `“${ch}” → ${bytes.length} byte(s) ${
      bytes.map((b) => "0x" + b.toString(16).padStart(2, "0")).join(" ")
    } → id ${bytes.map((b) => b + BYTE_OFFSET).join(", ")}`;
    frag.append(chip);
    shown++;
  }
  container.replaceChildren(frag);
}

export const BYT5_CSS = `
.bl-io { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.bl-col { flex:1 1 300px; min-inline-size:0; }
.bl-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:56px; white-space:pre-wrap; word-break:break-word; line-height:1.6; }
.bl-out:empty::before { content:"The model's output appears here."; color:var(--muted); }
.bl-fill { color:var(--accent); font-weight:600; background:color-mix(in srgb, var(--accent) 12%, transparent);
  border-radius:4px; padding:0 .15em; }
.bl-chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:.4rem; }
.bl-chip { display:inline-flex; flex-direction:column; align-items:center; gap:1px; min-inline-size:0;
  border:1px solid var(--border); border-radius:6px; background:var(--bg-secondary); padding:.15rem .3rem; }
.bl-glyph { font-family:var(--font-mono); font-size:.9rem; line-height:1.1; max-inline-size:2.4rem;
  overflow:hidden; text-overflow:ellipsis; }
.bl-ids { font-family:var(--font-mono); font-size:.6rem; color:var(--muted); white-space:nowrap; }
.bl-more { align-self:center; font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.bl-stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.bl-stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.bl-stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.bl-stat .v { font-family:var(--font-display); font-size:1.4rem; }
.bl-controls { display:flex; flex-wrap:wrap; gap:.5rem 1rem; align-items:end; margin:.6rem 0; }
.bl-controls label { display:flex; flex-direction:column; gap:.3rem; font-size:.8rem; }
.bl-controls input[type=range] { inline-size:min(100%, 12rem); }
.bl-controls .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.preset-grid { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); margin:.6rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised);
  padding:.55rem .7rem; cursor:pointer; font:inherit; min-inline-size:0; }
.preset:hover { border-color:var(--accent); }
.preset:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.preset .t { font-weight:600; font-size:.9rem; display:block; }
.preset .d { color:var(--muted); font-size:.76rem; font-family:var(--font-mono); }
.chip-btn { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip-btn:hover { border-color:var(--accent); }
.chip-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.bl-note { font-size:.82rem; color:var(--muted); }
.bl-textarea { inline-size:100%; font-family:var(--font-body); box-sizing:border-box; }
`;
