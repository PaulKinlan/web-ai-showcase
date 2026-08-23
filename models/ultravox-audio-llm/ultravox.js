// Front-end helpers for the Ultravox page: the worker handshake, a real WebGPU probe for the honest
// unsupported state, and the widget CSS. All inference lives in worker.js.

import { WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

const WORKER_URL = "/web-ai-showcase/models/ultravox-audio-llm/worker.js";

// How long a generation may go with NO word from the worker before it is declared stuck. This is an
// INACTIVITY deadline, not a total budget: every streamed token re-arms it, so a slow-but-alive
// device is never cut off, while a WebGPU hang — which produces silence, not slowness — is caught.
// Without it a stalled worker left the page permanently `busy`: controls disabled, microphone audio
// discarded, and no error anywhere, which is the worst failure mode this page has.
const GENERATE_STALL_MS = 45_000;
// Loading has the same failure mode and needed the same guard: a cold ~1.5 GB fetch or a WebGPU
// session creation that stalls without throwing left the shared loader stuck in "downloading" with
// no Retry and nothing on the page. Progress re-arms this too, so a slow link is never cut off.
const LOAD_STALL_MS = 120_000;

export class UltravoxEngine {
  constructor() {
    this.client = null;
    this.ready = false;
    this.device = null;
    this.dtype = null;
    this.onProgress = null;
    this._disposed = false;
    this._spawn();
  }

  /**
   * Build the worker client. Called again after a FATAL worker error — a module worker whose graph
   * 404s or fails to parse never becomes usable, and leaving that dead client installed meant the
   * loader offered Retry, load() posted into it, and the promise simply never settled.
   */
  _spawn() {
    if (this._disposed) return;
    this.client = new WorkerClient({
      url: WORKER_URL,
      name: "ultravox",
      maxInFlight: 1,
      maxQueue: 4,
      onState: (state) => {
        // "error" is terminal for a WorkerClient: drop it so the next load() builds a fresh one.
        if (state === "error" || state === "terminated") {
          this.ready = false;
          this.device = null;
          if (!this._disposed) this.client = null;
        }
      },
    });
  }

  /** A deadline that RE-ARMS on progress: slow-but-alive is fine, silence is not. */
  _deadline(ms) {
    const ctrl = new AbortController();
    let timer = null;
    const stop = () => {
      if (timer !== null) clearTimeout(timer);
      timer = null;
    };
    const bump = () => {
      stop();
      timer = setTimeout(() => {
        const err = new Error(
          `The model stopped responding (no output for ${Math.round(ms / 1000)}s). ` +
            "The worker was reset — load it again and retry.",
        );
        err.name = "StalledError";
        ctrl.abort(err);
      }, ms);
    };
    bump();
    return { signal: ctrl.signal, bump, stop, reason: () => ctrl.signal.reason };
  }

  async probeGPU() {
    if (this._disposed) return { ok: false, reason: "disposed" };
    if (!this.client) this._spawn();
    try {
      return await this.client.request("probe", {});
    } catch {
      // A probe that cannot answer must not hang the honest-capability gate.
      return { ok: false, reason: "worker-unavailable" };
    }
  }

  async load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return { device: this.device, dtype: this.dtype };
    if (this._disposed) throw new Error("Engine disposed");
    // Retry after a fatal error lands here with no client; build a fresh one rather than posting
    // into the corpse and waiting forever.
    if (!this.client) this._spawn();
    const dl = this._deadline(LOAD_STALL_MS);
    try {
      const res = await this.client.request("load", {}, {
        signal: dl.signal,
        onProgress: (p) => {
          // Every byte of progress re-arms the clock — a cold 1.5 GB download on a slow link is
          // slow, not stuck, and only genuine silence should fail it.
          dl.bump();
          if (p?.kind === "download") this.onProgress?.(p.p);
        },
      });
      this.ready = true;
      this.device = res?.device ?? null;
      this.dtype = res?.dtype ?? null;
      return res;
    } catch (err) {
      if (dl.signal.aborted) {
        this._fatal(dl.reason() ?? err);
        throw dl.reason() ?? err;
      }
      throw err;
    } finally {
      dl.stop();
    }
  }

  /**
   * Generate from a message list plus (optionally) a 16 kHz mono Float32Array of audio.
   * The audio is TRANSFERRED, so the caller must pass a copy it no longer needs.
   */
  async generate({ messages, tools, audio, maxTokens, onPrompt, onToken }) {
    // Never silently start a fresh worker here: a generate without a loaded model would sit waiting
    // while the page believed a turn was running. Fail loudly and let the loader's Retry reload.
    if (!this.client || !this.ready) {
      throw new Error("The model is not loaded — reload it and try again.");
    }
    const dl = this._deadline(GENERATE_STALL_MS);
    try {
      return await this.client.request(
        "generate",
        { messages, tools, audio, maxTokens },
        {
          transfer: audio ? [audio.buffer] : undefined,
          signal: dl.signal,
          onProgress: (p) => {
            dl.bump();
            if (p?.kind === "prompt") onPrompt?.(p.template);
            else if (p?.kind === "token") onToken?.(p.token, p.n);
          },
        },
      );
    } catch (err) {
      if (dl.signal.aborted) {
        // A worker that has gone quiet mid-generation cannot be trusted to finish anything else.
        this._fatal(dl.reason() ?? err);
        throw dl.reason() ?? err;
      }
      throw err;
    } finally {
      dl.stop();
    }
  }

  /** A failure the worker cannot continue past: discard the client so the next load() respawns. */
  _fatal(err) {
    this.ready = false;
    this.device = null;
    const dead = this.client;
    this.client = null;
    try {
      dead?.terminate?.(err);
    } catch { /* already gone */ }
  }

  /** Reject anything in flight, then terminate. Not reusable afterwards — construct a new one. */
  dispose(reason = "Engine disposed") {
    this._disposed = true;
    this.ready = false;
    this.device = null;
    const dead = this.client;
    this.client = null;
    try {
      dead?.terminate?.(new Error(reason));
    } catch { /* already gone */ }
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
