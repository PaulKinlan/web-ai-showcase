// Typed, versioned main-thread ↔ worker message protocol for web-ai-showcase.
//
// A NEW capability (additive) — it does NOT replace lib/model-loader.js / lib/model-cache.js /
// lib/webai.js and changes no existing demo. It gives model pages a disciplined way to run inference
// off the main thread (AGENTS.md: "Inference off the main thread, stream progress/tokens, keep INP
// low"), with real Transferables, streamed progress, cooperative cancellation, stale-response
// suppression, bounded backpressure, and deterministic teardown.
//
// modern-web-guidance retained + applied:
//   • performance — INP & Main-Thread Unblocking: "> 250ms: Offload to a Web Worker" and "separate UI
//     updates from heavy computations" — this protocol is the offload channel; progress is streamed so
//     the main thread only paints. The 50ms-rule / scheduler.yield() slicing is applied INSIDE worker
//     handlers via the `yieldToMain` helper below (cooperative + cancellation checkpoints).
//   • deprioritize-background-fetches — worker fetches for non-critical assets can pass
//     `priority:'low'`; the protocol is transport-agnostic so handlers own that.
// Primary specs consulted (MDN): Worker (`{type:'module'}`), Worker.postMessage transfer list,
// Transferable objects (ArrayBuffer, ImageBitmap), MessagePort, AbortController/AbortSignal,
// DOMException("AbortError"). No dedicated worker/transferable guide exists in the guidance catalogue
// today; these MDN primitives are the source of truth for the semantics below.
//
// ── Message envelope (every message) ─────────────────────────────────────────────────────────────
//   { p: PROTOCOL_VERSION, kind, ...fields }
// Kinds:
//   main → worker : "request" {id, method, payload} · "abort" {id} · "dispose" {}
//   worker → main : "ready" {} · "response" {id, result} · "progress" {id, progress} · "error" {id?, error}
// Transfer: request.payload and response.result may carry ArrayBuffers/ImageBitmaps that are
// *transferred* (ownership moved, buffers detached) rather than structured-cloned, in BOTH directions.
//
// ── Lifecycle states (WorkerClient.state, emitted via onState) ───────────────────────────────────
//   "starting" → "ready" → "busy" ⇄ "ready" → "terminated"      (and "error" on fatal worker failure)
//
// ── Usage (copyable) ─────────────────────────────────────────────────────────────────────────────
//   // main thread
//   import { WorkerClient } from "/lib/worker-protocol.js";
//   const client = new WorkerClient({
//     url: new URL("./worker.js", import.meta.url), // dedicated MODULE worker
//     name: "asr",
//     maxInFlight: 1,   // concurrent requests dispatched to the worker
//     maxQueue: 8,      // pending requests waiting for a slot; overflow rejects deterministically
//     onState: (s) => console.log("worker", s),
//   });
//   await client.ready;                              // resolves when the worker posts "ready"
//   const controller = new AbortController();
//   const { result } = await client.request("transcribe", { audio: floatBuf.buffer }, {
//     transfer: [floatBuf.buffer],                   // detaches floatBuf.buffer on the main thread
//     signal: controller.signal,                     // controller.abort() → cooperative worker cancel
//     channel: "live",                               // latest-wins: a newer "live" supersedes older ones
//     onProgress: (p) => updateBar(p),
//   });
//   client.terminate();                              // dispose → free model + revoke object URLs → terminate
//
//   // worker.js (module worker)
//   import { serveWorker, yieldToMain } from "/lib/worker-protocol.js";
//   serveWorker({
//     async init() { self.pipe = await loadModel(); },          // runs before "ready" is posted
//     methods: {
//       async transcribe(payload, { signal, onProgress }) {
//         const audio = new Float32Array(payload.audio);        // took ownership of the transferred buffer
//         for (let i = 0; i < steps; i++) {
//           if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
//           onProgress({ status: "progress", progress: (i / steps) * 100 });
//           await yieldToMain();                                // INP: slice work, stay cancellable
//         }
//         const out = new Float32Array(n);
//         return { result: { text, logits: out.buffer }, transfer: [out.buffer] }; // transfers back
//       },
//     },
//     onDispose() { self.pipe?.dispose?.(); },                  // free the model on terminate()
//   });

/** Protocol version — bump on any breaking envelope change; both sides assert compatibility. */
export const PROTOCOL_VERSION = 1;

/** @typedef {"request"|"abort"|"dispose"|"ready"|"response"|"progress"|"error"} MessageKind */
/** @typedef {"starting"|"ready"|"busy"|"error"|"terminated"} WorkerState */
/**
 * @typedef {Object} RequestOptions
 * @property {Transferable[]} [transfer] Objects to transfer (ownership moved) with the payload.
 * @property {AbortSignal}    [signal]   Aborting it cancels the request (worker gets an abort message).
 * @property {(p:any)=>void}  [onProgress] Called for every worker progress event for this request.
 * @property {string}         [channel]  Latest-wins key: a newer request on the same channel supersedes
 *                                        (aborts) older in-flight/queued ones; their late responses drop.
 */

/** Thrown when the bounded queue is full — deterministic backpressure signal (not a network/worker error). */
export class QueueOverflowError extends Error {
  constructor(message) {
    super(message);
    this.name = "QueueOverflowError";
  }
}

/** Thrown to a request that a newer same-channel request superseded (latest-wins). */
export class SupersededError extends Error {
  constructor(message) {
    super(message);
    this.name = "SupersededError";
  }
}

function abortError(reason) {
  if (reason instanceof Error) return reason;
  return new DOMException(typeof reason === "string" ? reason : "Aborted", "AbortError");
}

/**
 * Main-thread client for a dedicated module worker speaking this protocol.
 *
 * Guarantees:
 *  - per-request `id`; responses/progress/errors are routed by id;
 *  - **stale-response suppression**: once a request is settled (resolved, aborted, superseded, or
 *    overflowed) its id is retired and any later message for that id is dropped;
 *  - **backpressure**: at most `maxInFlight` requests are dispatched to the worker at once; up to
 *    `maxQueue` wait; beyond that `request()` rejects synchronously-ish with {@link QueueOverflowError};
 *  - **cancellation**: an aborted `signal` posts an `abort` message and rejects locally with the signal
 *    reason (AbortError by default);
 *  - **latest-wins channels**: a new request on an existing `channel` supersedes older ones on it;
 *  - **deterministic teardown**: `terminate()` posts `dispose` (letting the worker free its model),
 *    revokes any registered object URLs, then calls `Worker.terminate()`.
 */
export class WorkerClient {
  /**
   * @param {Object} opts
   * @param {string|URL} opts.url            Worker script URL (served same-origin).
   * @param {string}     [opts.name]         Worker name (devtools + diagnostics).
   * @param {number}     [opts.maxInFlight]  Max concurrent dispatched requests (default 1).
   * @param {number}     [opts.maxQueue]     Max pending (queued) requests (default 16).
   * @param {(s:WorkerState)=>void} [opts.onState] Lifecycle state change callback.
   * @param {boolean}    [opts.module]       Use a module worker (default true — preferred).
   * @param {number}     [opts.disposeGraceMs] Ms to let the worker free resources before terminate (300).
   */
  constructor(
    { url, name, maxInFlight = 1, maxQueue = 16, onState, module = true, disposeGraceMs = 300 },
  ) {
    this.name = name || "worker";
    this.maxInFlight = Math.max(1, maxInFlight | 0);
    this.maxQueue = Math.max(0, maxQueue | 0);
    this.disposeGraceMs = disposeGraceMs;
    this._onState = onState;
    /** @type {WorkerState} */
    this.state = "starting";
    this._seq = 0;
    /** in-flight + queued requests, keyed by id. */
    this._reqs = new Map();
    /** ordered queue of ids waiting for an in-flight slot. */
    this._queue = [];
    this._inflight = 0;
    /** channel → set of live request ids (for latest-wins supersession). */
    this._channels = new Map();
    /** ids that were settled locally but may still get a late worker message → drop it. */
    this._retired = new Set();
    this._objectUrls = new Set();

    // Prefer a dedicated MODULE worker (import maps / ESM). Fall back to classic only if asked.
    this.worker = new Worker(url, { type: module ? "module" : "classic", name: this.name });
    this._ownsUrl = false;

    this.worker.addEventListener("message", (ev) => this._onMessage(ev.data));
    this.worker.addEventListener("error", (ev) => this._onFatal(ev.message || "worker error"));
    this.worker.addEventListener(
      "messageerror",
      () => this._onFatal("worker message deserialisation error"),
    );

    this.ready = new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });
    // Avoid an unhandled rejection if nobody awaits .ready before a fatal error.
    this.ready.catch(() => {});
  }

  /** Register an object URL (e.g. a Blob worker URL) to be revoked on terminate(). */
  registerObjectURL(u) {
    this._objectUrls.add(u);
    this._ownsUrl = true;
    return u;
  }

  _setState(s) {
    if (this.state === s || this.state === "terminated") return;
    this.state = s;
    try {
      this._onState?.(s);
    } catch { /* listener errors never break the protocol */ }
  }

  _nextId() {
    return `${this.name}-${++this._seq}`;
  }

  /**
   * Dispatch a request to the worker.
   * @param {string} method
   * @param {any} payload
   * @param {RequestOptions} [options]
   * @returns {Promise<{result:any}>}
   */
  request(method, payload, options = {}) {
    if (this.state === "terminated") {
      return Promise.reject(new Error(`WorkerClient ${this.name} is terminated`));
    }
    const { transfer, signal, onProgress, channel } = options;
    if (signal?.aborted) return Promise.reject(abortError(signal.reason));

    // Backpressure: reject deterministically when the bounded queue is already full.
    if (this._inflight >= this.maxInFlight && this._queue.length >= this.maxQueue) {
      return Promise.reject(
        new QueueOverflowError(
          `${this.name}: queue full (maxInFlight=${this.maxInFlight}, maxQueue=${this.maxQueue})`,
        ),
      );
    }

    const id = this._nextId();
    // Latest-wins: supersede older live requests on the same channel BEFORE enqueuing this one.
    if (channel) {
      const live = this._channels.get(channel);
      if (live) {
        for (const olderId of [...live]) this._supersede(olderId);
      }
      this._channels.set(channel, new Set([id]));
    }

    return new Promise((resolve, reject) => {
      const rec = {
        id,
        method,
        payload,
        transfer,
        onProgress,
        channel,
        signal,
        resolve,
        reject,
        dispatched: false,
        onAbort: null,
      };
      this._reqs.set(id, rec);

      if (signal) {
        rec.onAbort = () => this._abort(id, signal.reason);
        signal.addEventListener("abort", rec.onAbort, { once: true });
      }

      // Unified path: enqueue, then pump. _pumpQueue dispatches while an in-flight slot is free — this
      // avoids a stale would-queue decision deadlocking a request whose channel-supersede just freed a slot.
      this._queue.push(id);
      this._pumpQueue();
    });
  }

  _dispatch(id) {
    const rec = this._reqs.get(id);
    if (!rec || rec.dispatched) return;
    rec.dispatched = true;
    this._inflight++;
    this._setState("busy");
    this._post(
      { p: PROTOCOL_VERSION, kind: "request", id, method: rec.method, payload: rec.payload },
      rec.transfer,
    );
  }

  _post(msg, transfer) {
    try {
      if (transfer && transfer.length) this.worker.postMessage(msg, transfer);
      else this.worker.postMessage(msg);
    } catch (err) {
      // e.g. an already-detached/non-transferable in the list — surface to the request, don't wedge.
      const rec = this._reqs.get(msg.id);
      if (rec) this._settle(rec, () => rec.reject(err));
    }
  }

  _pumpQueue() {
    while (this._inflight < this.maxInFlight && this._queue.length) {
      const id = this._queue.shift();
      if (this._reqs.has(id)) this._dispatch(id);
    }
    if (this._inflight === 0 && this.state === "busy") this._setState("ready");
  }

  /** Remove a request from all bookkeeping and retire its id so late worker messages are dropped. */
  _settle(rec, finalize) {
    if (!this._reqs.has(rec.id)) return;
    this._reqs.delete(rec.id);
    const qi = this._queue.indexOf(rec.id);
    if (qi >= 0) this._queue.splice(qi, 1);
    if (rec.dispatched) this._inflight = Math.max(0, this._inflight - 1);
    if (rec.channel) {
      const live = this._channels.get(rec.channel);
      live?.delete(rec.id);
      if (live && live.size === 0) this._channels.delete(rec.channel);
    }
    if (rec.signal && rec.onAbort) rec.signal.removeEventListener("abort", rec.onAbort);
    this._retired.add(rec.id);
    // Bound the retired set so it can't grow forever in a long-lived page.
    if (this._retired.size > 512) {
      const it = this._retired.values();
      for (let i = 0; i < 256; i++) this._retired.delete(it.next().value);
    }
    try {
      finalize();
    } finally {
      this._pumpQueue();
    }
  }

  _abort(id, reason) {
    const rec = this._reqs.get(id);
    if (!rec) return;
    // Tell the worker to stop cooperatively (only meaningful once dispatched, but harmless otherwise).
    if (rec.dispatched) this._post({ p: PROTOCOL_VERSION, kind: "abort", id });
    this._settle(rec, () => rec.reject(abortError(reason)));
  }

  _supersede(id) {
    const rec = this._reqs.get(id);
    if (!rec) return;
    if (rec.dispatched) this._post({ p: PROTOCOL_VERSION, kind: "abort", id });
    this._settle(
      rec,
      () => rec.reject(new SupersededError(`${this.name}: superseded on channel "${rec.channel}"`)),
    );
  }

  _onMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.p !== PROTOCOL_VERSION) {
      this._onFatal(`protocol mismatch: worker v${msg.p} vs client v${PROTOCOL_VERSION}`);
      return;
    }
    switch (msg.kind) {
      case "ready":
        this._setState("ready");
        this._readyResolve?.(this);
        this._readyResolve = null;
        return;
      case "progress": {
        const rec = this._reqs.get(msg.id);
        if (!rec) return; // stale/superseded/aborted — drop.
        try {
          rec.onProgress?.(msg.progress);
        } catch { /* progress listener errors never break the protocol */ }
        return;
      }
      case "response": {
        const rec = this._reqs.get(msg.id);
        if (!rec) return; // STALE-RESPONSE SUPPRESSION: id already retired → drop silently.
        this._settle(rec, () => rec.resolve({ result: msg.result }));
        return;
      }
      case "error": {
        if (msg.id == null) {
          // Request-less worker error (e.g. init failure) → fatal.
          this._onFatal(msg.error?.message || "worker error");
          return;
        }
        const rec = this._reqs.get(msg.id);
        if (!rec) return; // stale — drop.
        const err = new Error(msg.error?.message || "worker error");
        if (msg.error?.name) err.name = msg.error.name;
        if (msg.error?.stack) err.stack = msg.error.stack;
        this._settle(rec, () => rec.reject(err));
        return;
      }
      default:
        return; // unknown kind — ignore forward-compatibly.
    }
  }

  _onFatal(message) {
    if (this.state === "terminated") return;
    this._setState("error");
    const err = new Error(`${this.name}: ${message}`);
    this._readyReject?.(err);
    this._readyReject = null;
    // Reject everything still outstanding; the worker is no longer trustworthy.
    for (const rec of [...this._reqs.values()]) this._settle(rec, () => rec.reject(err));
  }

  /** True while the worker is ready and not currently processing a request. */
  get idle() {
    return this.state === "ready" && this._inflight === 0;
  }

  /** Number of requests dispatched-but-unsettled plus queued. */
  get pending() {
    return this._reqs.size;
  }

  /**
   * Deterministic teardown: reject outstanding requests, ask the worker to free its model, revoke any
   * registered object URLs, then terminate the underlying Worker. Idempotent.
   * @param {any} [reason] Rejection reason for outstanding requests.
   */
  async terminate(reason) {
    if (this.state === "terminated") return;
    const err = reason instanceof Error ? reason : new Error(reason || `${this.name}: terminated`);
    for (const rec of [...this._reqs.values()]) this._settle(rec, () => rec.reject(err));
    this._queue.length = 0;
    // Give the worker a chance to free the model / caches before we hard-terminate.
    try {
      this.worker.postMessage({ p: PROTOCOL_VERSION, kind: "dispose" });
      await new Promise((r) => setTimeout(r, this.disposeGraceMs));
    } catch { /* ignore */ }
    try {
      this.worker.terminate();
    } catch { /* ignore */ }
    for (const u of this._objectUrls) {
      try {
        URL.revokeObjectURL(u);
      } catch { /* ignore */ }
    }
    this._objectUrls.clear();
    this._setState("terminated");
  }
}

/** INP helper: yield to the main thread mid-task. Front-of-queue via scheduler.yield when available.
 *  Applies the `performance` guidance's 50ms-rule slicing inside worker handlers. */
export async function yieldToMain() {
  // In a worker there is no user input to unblock, but yielding still lets abort messages + progress
  // posts interleave and keeps long compute cooperative. scheduler.yield exists in workers on Chrome.
  const s = typeof self !== "undefined" ? self.scheduler : undefined;
  if (s && typeof s.yield === "function") return s.yield();
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Worker-side counterpart. Wires `self` message handling to your method table, posting the correct
 * protocol messages and honouring cancellation.
 *
 * @param {Object} opts
 * @param {() => (void|Promise<void>)} [opts.init] Runs once before "ready" is posted (load the model).
 * @param {Record<string, (payload:any, ctx:{signal:AbortSignal, onProgress:(p:any)=>void, id:string}) =>
 *   Promise<any|{result:any, transfer?:Transferable[]}>>} opts.methods Request handlers by method name.
 * @param {() => (void|Promise<void>)} [opts.onDispose] Free the model / revoke URLs on dispose.
 */
export function serveWorker({ init, methods, onDispose }) {
  if (typeof self === "undefined" || typeof self.postMessage !== "function") {
    throw new Error("serveWorker() must run inside a Worker");
  }
  /** id → AbortController for in-flight handlers (cooperative cancellation). */
  const active = new Map();
  let disposed = false;

  const post = (msg, transfer) => {
    if (disposed) return;
    if (transfer && transfer.length) self.postMessage(msg, transfer);
    else self.postMessage(msg);
  };

  async function handleRequest(id, method, payload) {
    const handler = methods?.[method];
    if (typeof handler !== "function") {
      post({
        p: PROTOCOL_VERSION,
        kind: "error",
        id,
        error: { name: "MethodNotFound", message: `Unknown method: ${method}` },
      });
      return;
    }
    const controller = new AbortController();
    active.set(id, controller);
    const ctx = {
      id,
      signal: controller.signal,
      onProgress: (progress) => {
        if (!controller.signal.aborted) {
          post({ p: PROTOCOL_VERSION, kind: "progress", id, progress });
        }
      },
    };
    try {
      const out = await handler(payload, ctx);
      if (controller.signal.aborted) return; // cancelled mid-flight — main side already settled; drop.
      const result = out && typeof out === "object" && "result" in out ? out.result : out;
      const transfer = out && typeof out === "object" ? out.transfer : undefined;
      post({ p: PROTOCOL_VERSION, kind: "response", id, result }, transfer);
    } catch (err) {
      if (controller.signal.aborted) return; // aborted → main side already rejected; stay silent.
      post({
        p: PROTOCOL_VERSION,
        kind: "error",
        id,
        error: {
          name: err?.name || "Error",
          message: String(err?.message ?? err),
          stack: err?.stack,
        },
      });
    } finally {
      active.delete(id);
    }
  }

  async function ready() {
    try {
      await init?.();
      post({ p: PROTOCOL_VERSION, kind: "ready" });
    } catch (err) {
      post({
        p: PROTOCOL_VERSION,
        kind: "error",
        error: {
          name: err?.name || "InitError",
          message: String(err?.message ?? err),
          stack: err?.stack,
        },
      });
    }
  }

  self.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (!msg || msg.p !== PROTOCOL_VERSION) return;
    switch (msg.kind) {
      case "request":
        handleRequest(msg.id, msg.method, msg.payload);
        return;
      case "abort": {
        const c = active.get(msg.id);
        if (c) c.abort(new DOMException("Aborted by main thread", "AbortError"));
        return;
      }
      case "dispose":
        disposed = true;
        for (const c of active.values()) {
          try {
            c.abort(new DOMException("Worker disposing", "AbortError"));
          } catch { /* ignore */ }
        }
        active.clear();
        Promise.resolve()
          .then(() => onDispose?.())
          .catch(() => {})
          .finally(() => self.close());
        return;
      default:
        return;
    }
  });

  ready();
}
