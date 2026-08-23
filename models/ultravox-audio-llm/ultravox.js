// Front-end helpers for the Ultravox page: the worker handshake, a real WebGPU probe for the honest
// unsupported state, and the widget CSS. All inference lives in worker.js.

const WORKER_URL = "/web-ai-showcase/models/ultravox-audio-llm/worker.js";

export class UltravoxEngine {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.device = null;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this._disposed = false;
    this._spawn();
  }

  /**
   * Build the worker. Called again after a FATAL worker error — a module worker whose graph 404s or
   * fails to parse never becomes usable, and the old code left that dead worker installed: the
   * loader offered Retry, load() posted into it, and the promise simply never settled. A fatal error
   * now tears the worker down, so the next load() gets a fresh one and Retry can actually recover.
   */
  _spawn() {
    if (this._disposed) return;
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      this._fatal(new Error(e.message || "Worker failed to start"));
    });
  }

  /** A failure the worker cannot continue past: reject everything and discard the worker. */
  _fatal(err) {
    this.ready = false;
    this.device = null;
    const dead = this.worker;
    this.worker = null;
    try {
      dead?.terminate();
    } catch { /* already gone */ }
    this._rejectAll(err);
    // A probe that can never answer must not hang the honest-capability gate either.
    for (const w of this._probeWaiters) w.resolve(false);
    this._probeWaiters = [];
  }

  _rejectAll(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    for (const [, p] of this._pending) p.reject(err);
    this._pending.clear();
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "probe-result":
        for (const w of this._probeWaiters) w.resolve(msg.gpu);
        this._probeWaiters = [];
        break;
      case "ready":
        this.ready = true;
        this.device = msg.device;
        this.dtype = msg.dtype;
        for (const w of this._loadWaiters) w.resolve(msg);
        this._loadWaiters = [];
        break;
      case "prompt":
        this._pending.get(msg.id)?.onPrompt?.(msg.template);
        break;
      case "token":
        this._pending.get(msg.id)?.onToken?.(msg.token, msg.n);
        break;
      case "result": {
        const p = this._pending.get(msg.id);
        if (p) {
          this._pending.delete(msg.id);
          p.resolve(msg);
        }
        break;
      }
      case "error":
        if (msg.id != null && this._pending.has(msg.id)) {
          this._pending.get(msg.id).reject(new Error(msg.message));
          this._pending.delete(msg.id);
        } else {
          this._rejectAll(new Error(msg.message));
        }
        break;
    }
  }

  probeGPU() {
    if (this._disposed) return Promise.resolve(false);
    if (!this.worker) this._spawn();
    return new Promise((resolve) => {
      this._probeWaiters.push({ resolve });
      this.worker.postMessage({ type: "probe" });
    });
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve({ device: this.device, dtype: this.dtype });
    if (this._disposed) return Promise.reject(new Error("Engine disposed"));
    // Retry after a fatal error lands here with no worker; build a fresh one rather than posting
    // into the corpse and waiting forever.
    if (!this.worker) this._spawn();
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /**
   * Generate from a message list plus (optionally) a 16 kHz mono Float32Array of audio.
   * The audio is TRANSFERRED, so the caller must pass a copy it no longer needs.
   */
  generate({ messages, tools, audio, maxTokens, onPrompt, onToken }) {
    // Never silently start a fresh worker here: a generate without a loaded model would sit waiting
    // while the page believed a turn was running. Fail loudly and let the loader's Retry reload.
    if (!this.worker || !this.ready) {
      return Promise.reject(new Error("The model is not loaded — reload it and try again."));
    }
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onPrompt, onToken });
      const payload = { type: "generate", id, messages, tools, audio, maxTokens };
      this.worker.postMessage(payload, audio ? [audio.buffer] : []);
    });
  }

  /** Reject anything in flight, then terminate. Not reusable afterwards — construct a new one. */
  dispose(reason = "Engine disposed") {
    this._disposed = true;
    this.ready = false;
    this._rejectAll(new Error(reason));
    for (const w of this._probeWaiters) w.resolve(false);
    this._probeWaiters = [];
    try {
      this.worker?.terminate();
    } catch { /* already gone */ }
    this.worker = null;
  }
}

/** Probe WebGPU on the main thread too, so the page can gate before it ever loads a worker. */
export async function probeWebGPUMain() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter" };
    return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export const ULTRAVOX_CSS = `
.mic-bar { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.mic-bar button { min-block-size:44px; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad);
  display:inline-block; margin-inline-end:.45rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.meter { inline-size:100%; block-size:110px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
@media (max-width:420px) { .meter { block-size:88px; } }
.phase { display:inline-flex; align-items:center; gap:.4rem; font-family:var(--font-mono);
  font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; padding:.25rem .6rem;
  border-radius:999px; border:1px solid var(--border); background:var(--bg-secondary); color:var(--muted); }
.phase[data-on="1"] { border-color:var(--accent); color:var(--accent); }
.phase[data-on="speech"] { border-color:var(--good); color:var(--good); }
.turns { list-style:none; margin:.6rem 0 0; padding:0; display:flex; flex-direction:column; gap:.9rem; }
.turn { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised); padding:.85rem; }
.turn h3 { margin:0 0 .5rem; font-size:1.05rem; }
.flow { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.4rem 0; font-size:.8rem; }
.flow .node { border:1px solid var(--border); border-radius:6px; padding:.15rem .45rem;
  font-family:var(--font-mono); font-size:.72rem; background:var(--bg-secondary); }
.flow .node[data-state="done"] { border-color:var(--good); }
.flow .node[data-state="fail"] { border-color:var(--bad); color:var(--bad); }
.flow .node[data-state="skipped"] { border-style:dashed; color:var(--muted); opacity:.75; }
.flow .arrow { color:var(--muted); }
.call { font-family:var(--font-mono); font-size:.8rem; background:var(--bg-secondary);
  border:1px solid var(--border); border-radius:var(--radius); padding:.5rem .6rem; margin:.4rem 0;
  overflow-x:auto; white-space:pre; }
.answer { font-size:1.05rem; line-height:1.6; margin:.5rem 0 0; }
.readout { display:flex; flex-wrap:wrap; gap:.9rem; font-family:var(--font-mono); font-size:.75rem;
  color:var(--muted); margin-top:.55rem; }
.readout b { color:var(--color); font-weight:600; }
.tool-state { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); }
.tool-state > div { border:1px solid var(--border); border-radius:var(--radius);
  background:var(--bg-raised); padding:.8rem; }
.tool-state h3 { margin:0 0 .4rem; font-size:.95rem; }
.tool-state ul { margin:0; padding-inline-start:1.1rem; }
.timer { display:flex; justify-content:space-between; gap:.6rem; font-family:var(--font-mono); font-size:.85rem; }
.timer[data-done="1"] { color:var(--good); font-weight:600; }
.toollist { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; padding:0; list-style:none; }
.toollist li { font-family:var(--font-mono); font-size:.75rem; border:1px solid var(--border);
  border-radius:999px; padding:.2rem .6rem; background:var(--bg-secondary); }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; margin-block-start:.6rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
details.inside { margin-top:.7rem; }
details.inside > summary { cursor:pointer; font-family:var(--font-mono); font-size:.8rem; color:var(--muted); }
pre.tmpl { font-family:var(--font-mono); font-size:.72rem; white-space:pre-wrap; word-break:break-word;
  background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius);
  padding:.6rem; max-block-size:22rem; overflow:auto; }
.audioviz { display:flex; flex-wrap:wrap; gap:2px; margin:.4rem 0; }
.audioviz span { inline-size:.55rem; block-size:1.1rem; border-radius:2px; background:var(--accent); opacity:.75; }
.audioviz span.txt { background:var(--border); }
.visually-hidden { position:absolute; inline-size:1px; block-size:1px; overflow:hidden;
  clip-path:inset(50%); white-space:nowrap; }
`;
