// media-pipeline.js — off-main-thread media preprocessing for the web-ai-showcase demos.
//
// Two independent pipelines, each with an honest fallback so they run on both the cross-origin-
// isolated Deno deployment and GitHub Pages (no COOP/COEP), plus browsers without worker canvas:
//
//   A. IMAGE / VIDEO-FRAME preprocessing (ImagePreprocessor)
//      Decode + resize with createImageBitmap on the main thread, then TRANSFER the ImageBitmap to a
//      worker that owns an OffscreenCanvas, draws it, getImageData()s it THERE, and normalizes into a
//      Float32 tensor. The main thread never calls getImageData (keeps INP low — performance /
//      efficient-background-processing guidance). When OffscreenCanvas-in-workers is unavailable we
//      fall back to a main-thread canvas and MEASURE the cost so the demo can report which path ran.
//
//   B. REAL-TIME AUDIO capture (AudioCapturePipeline)
//      An AudioWorklet taps the mic (or any AudioNode) on the audio render thread, batches samples into
//      BOUNDED, fixed-size chunks, and hands them to a feature-extraction worker.
//      • SharedArrayBuffer ring buffer (zero-copy) needs cross-origin isolation — COOP:
//        same-origin + COEP: require-corp, i.e. `self.crossOriginIsolated === true`. The Deno
//        deployment sets both headers and uses SAB; GitHub Pages cannot, so SAB is unavailable there.
//      • The portable fallback is postMessage: the worklet posts each bounded chunk to the main
//        thread, which forwards (transfers) it to the feature worker. This works everywhere.
//      `AudioCapturePipeline.mode()` reports which path will run.
//
// All capture is USER-INITIATED upstream (see lib/capture-ux.js) — this module only processes a
// MediaStream / AudioNode you already obtained; it never calls getUserMedia.
//
// ── Copyable usage — image ──────────────────────────────────────────────────────────────────────────
//   import { ImagePreprocessor } from "/web-ai-showcase/lib/media-pipeline.js";
//   const pre = new ImagePreprocessor();                 // forceMainThread:true to test the fallback
//   const { tensor, width, height, path, ms } =
//     await pre.preprocess(imageBitmapOrBlobOrElement, { width: 224, height: 224,
//       mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], layout: "CHW" });
//   inferenceWorker.postMessage({ tensor }, [tensor.buffer]); // hand the normalized tensor onward
//   pre.destroy();
//
// ── Copyable usage — audio ──────────────────────────────────────────────────────────────────────────
//   import { AudioCapturePipeline } from "/web-ai-showcase/lib/media-pipeline.js";
//   console.log(AudioCapturePipeline.mode());            // "postmessage" on GitHub Pages
//   const pipe = new AudioCapturePipeline({ chunkSize: 2048, onFeatures: (f) => draw(f) });
//   await pipe.start(micStream);                          // micStream from capture-ux (user-initiated)
//   // …later…
//   pipe.stop();
// ─────────────────────────────────────────────────────────────────────────────────────────────────

const IMG_WORKER_URL = new URL("./media-pipeline.image-worker.js", import.meta.url).href;
const AUDIO_WORKLET_URL = new URL("./media-pipeline.audio-worklet.js", import.meta.url).href;
const AUDIO_FEATURE_WORKER_URL =
  new URL("./media-pipeline.audio-feature-worker.js", import.meta.url).href;

/** OffscreenCanvas exists in this (main-thread) realm. Workers are probed separately at init. */
export function supportsOffscreenCanvas() {
  return typeof OffscreenCanvas !== "undefined";
}

/** Cross-origin isolation is required for SharedArrayBuffer (COOP + COEP). */
export function isCrossOriginIsolated() {
  return typeof self !== "undefined" && self.crossOriginIsolated === true &&
    typeof SharedArrayBuffer !== "undefined";
}

/**
 * Decode + resize any drawable source to an ImageBitmap sized for the model, WITHOUT touching pixels
 * on the main thread. createImageBitmap does the decode + high-quality downscale for us, and the
 * returned bitmap is transferable to a worker.
 * `source` may be a Blob/File, ImageBitmap, HTMLImageElement, HTMLVideoElement, HTMLCanvasElement,
 * VideoFrame, or ImageData.
 */
export async function prepareImageBitmap(source, { width, height, resizeQuality = "high" } = {}) {
  const opts = (width && height)
    ? { resizeWidth: width, resizeHeight: height, resizeQuality }
    : undefined;
  return await createImageBitmap(source, opts);
}

/** Normalize RGBA bytes into a Float32 tensor. Shared by the worker and the main-thread fallback. */
export function rgbaToTensor(rgba, width, height, {
  mean = null,
  std = null,
  layout = "CHW",
  normalize = true,
} = {}) {
  const n = width * height;
  const out = new Float32Array(n * 3);
  const m = mean || [0, 0, 0];
  const s = std || [1, 1, 1];
  const scale = normalize ? 1 / 255 : 1;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] * scale;
    const g = rgba[i * 4 + 1] * scale;
    const b = rgba[i * 4 + 2] * scale;
    const rv = (r - m[0]) / s[0];
    const gv = (g - m[1]) / s[1];
    const bv = (b - m[2]) / s[2];
    if (layout === "CHW") {
      out[i] = rv; // R plane
      out[n + i] = gv; // G plane
      out[2 * n + i] = bv; // B plane
    } else { // HWC
      out[i * 3] = rv;
      out[i * 3 + 1] = gv;
      out[i * 3 + 2] = bv;
    }
  }
  return out;
}

/**
 * Off-main-thread image/video-frame preprocessor. Lazily spins up a worker with an OffscreenCanvas;
 * transparently falls back to (and measures) a main-thread canvas when that's unavailable or forced.
 */
export class ImagePreprocessor {
  constructor({ forceMainThread = false } = {}) {
    this.forceMainThread = forceMainThread;
    this.worker = null;
    this._ready = null; // Promise<boolean> — resolves true if the worker path is usable
    this._id = 0;
    this._pending = new Map();
    this._mainCanvas = null;
  }

  /** Resolve whether the worker+OffscreenCanvas path is available (probed once). */
  _ensureWorker() {
    if (this.forceMainThread || typeof Worker === "undefined" || !supportsOffscreenCanvas()) {
      return Promise.resolve(false);
    }
    if (this._ready) return this._ready;
    this._ready = new Promise((resolve) => {
      let settled = false;
      try {
        this.worker = new Worker(IMG_WORKER_URL, { type: "module" });
      } catch {
        resolve(false);
        return;
      }
      const onInit = (e) => {
        if (e.data?.type === "ready") {
          settled = true;
          this.worker.removeEventListener("message", onInit);
          this.worker.addEventListener("message", (ev) => this._onMessage(ev.data));
          // The worker reports whether ITS realm has OffscreenCanvas (some UAs lack it in workers).
          resolve(e.data.offscreen === true);
        }
      };
      this.worker.addEventListener("message", onInit);
      this.worker.addEventListener("error", () => {
        if (!settled) resolve(false);
      });
      this.worker.postMessage({ type: "init" });
      // If the worker never answers, fall back after a short probe window.
      setTimeout(() => {
        if (!settled) resolve(false);
      }, 1500);
    });
    return this._ready;
  }

  _onMessage(msg) {
    if (msg.type === "result" && this._pending.has(msg.id)) {
      const { resolve } = this._pending.get(msg.id);
      this._pending.delete(msg.id);
      resolve(msg);
    } else if (msg.type === "error" && this._pending.has(msg.id)) {
      const { reject } = this._pending.get(msg.id);
      this._pending.delete(msg.id);
      reject(new Error(msg.message));
    }
  }

  /**
   * Preprocess `source` to a normalized Float32 tensor.
   * Returns { tensor, width, height, channels, layout, path, ms } where
   * path is "worker-offscreen" or "main-thread".
   */
  async preprocess(source, {
    width = 224,
    height = 224,
    mean = null,
    std = null,
    layout = "CHW",
    normalize = true,
    resizeQuality = "high",
  } = {}) {
    const useWorker = await this._ensureWorker();
    // Decode + resize off-pixel on the main thread (createImageBitmap), yielding a transferable bitmap.
    const bitmap = await prepareImageBitmap(source, { width, height, resizeQuality });

    if (useWorker) {
      const id = ++this._id;
      const result = await new Promise((resolve, reject) => {
        this._pending.set(id, { resolve, reject });
        // Transfer the ImageBitmap — getImageData happens in the worker, never here.
        this.worker.postMessage(
          { type: "preprocess", id, bitmap, width, height, mean, std, layout, normalize },
          [bitmap],
        );
      });
      return {
        tensor: result.tensor,
        width: result.width,
        height: result.height,
        channels: 3,
        layout,
        path: "worker-offscreen",
        ms: result.ms,
      };
    }

    // ── Measured main-thread fallback (OffscreenCanvas unavailable, or forced) ──
    const t0 = performance.now();
    if (!this._mainCanvas) this._mainCanvas = document.createElement("canvas");
    const canvas = this._mainCanvas;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, width, height);
    try {
      bitmap.close();
    } catch { /* noop */ }
    const { data } = ctx.getImageData(0, 0, width, height); // on the main thread — the measured cost
    const tensor = rgbaToTensor(data, width, height, { mean, std, layout, normalize });
    const ms = performance.now() - t0;
    return { tensor, width, height, channels: 3, layout, path: "main-thread", ms };
  }

  destroy() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this._pending.clear();
    this._mainCanvas = null;
    this._ready = null;
  }
}

/**
 * Real-time audio capture via AudioWorklet → bounded chunks → feature-extraction worker.
 *
 * The AudioWorklet runs on the audio render thread and batches samples into fixed-size chunks
 * (bounded — never an unbounded buffer). Those chunks reach the feature worker one of two ways:
 *   • "sab-isolated": a SharedArrayBuffer ring buffer shared worklet↔worker (zero-copy). Requires
 *      cross-origin isolation (COOP + COEP). NOT available on GitHub Pages.
 *   • "postmessage" (fallback): the worklet posts each chunk to the main thread, which transfers it
 *      to the feature worker. Works with no special headers.
 */
export class AudioCapturePipeline {
  /** Which transport will be used given the current isolation state. */
  static mode() {
    return isCrossOriginIsolated() ? "sab-isolated" : "postmessage";
  }

  constructor(
    { chunkSize = 2048, ringSeconds = 2, onFeatures = () => {}, onError = () => {} } = {},
  ) {
    this.chunkSize = chunkSize;
    this.ringSeconds = ringSeconds;
    this.onFeatures = onFeatures;
    this.onError = onError;
    this.mode = AudioCapturePipeline.mode();
    this.ctx = null;
    this.node = null;
    this.source = null;
    this.worker = null;
    this.sab = null;
    this._ownsCtx = false;
    this._oscillator = null;
  }

  /** Start from a mic MediaStream (obtained user-initiated elsewhere). */
  async start(stream) {
    const AC = self.AudioContext || self.webkitAudioContext;
    const ctx = new AC();
    this._ownsCtx = true;
    const source = ctx.createMediaStreamSource(stream);
    await this._wire(ctx, source);
  }

  /** Start from any AudioNode + context you already have (e.g. an OscillatorNode for testing). */
  async startFromNode(sourceNode, audioContext) {
    await this._wire(audioContext, sourceNode);
  }

  /** Device-free synthetic source — an oscillator — handy for tests and demos without a mic. */
  async startFromOscillator({ frequency = 220, type = "sine" } = {}) {
    const AC = self.AudioContext || self.webkitAudioContext;
    const ctx = new AC();
    this._ownsCtx = true;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = frequency;
    this._oscillator = osc;
    await this._wire(ctx, osc);
    osc.start();
    return ctx.state;
  }

  async _wire(ctx, source) {
    this.ctx = ctx;
    this.source = source;
    if (!ctx.audioWorklet) throw new Error("AudioWorklet is not supported in this browser.");

    // Feature-extraction worker (bounded ring of recent frames → features back to the main thread).
    this.worker = new Worker(AUDIO_FEATURE_WORKER_URL, { type: "module" });
    this.worker.addEventListener("message", (e) => {
      if (e.data?.type === "features") this.onFeatures(e.data.features);
    });
    this.worker.addEventListener("error", (e) => this.onError(e));

    let sab = null;
    if (this.mode === "sab-isolated") {
      // Ring buffer: [writeIndex, readIndex, ...float32 samples]. Both threads map the same memory.
      const capacity = Math.max(this.chunkSize * 8, ctx.sampleRate * this.ringSeconds);
      sab = new SharedArrayBuffer(8 + capacity * 4);
      this.sab = sab;
      this.worker.postMessage({
        type: "init",
        transport: "sab",
        sab,
        capacity,
        sampleRate: ctx.sampleRate,
        chunkSize: this.chunkSize,
      });
    } else {
      this.worker.postMessage({
        type: "init",
        transport: "postmessage",
        sampleRate: ctx.sampleRate,
        chunkSize: this.chunkSize,
      });
    }

    await ctx.audioWorklet.addModule(AUDIO_WORKLET_URL);
    const node = new AudioWorkletNode(ctx, "bounded-capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      processorOptions: {
        chunkSize: this.chunkSize,
        transport: this.mode === "sab-isolated" ? "sab" : "postmessage",
        sab,
      },
    });
    this.node = node;

    if (this.mode === "sab-isolated") {
      // Worklet writes into the SAB ring; it only needs to nudge the worker to drain. We route the
      // nudge through the main thread port (the worklet and worker can't hold each other's ports).
      node.port.onmessage = (e) => {
        if (e.data?.type === "wrote") this.worker.postMessage({ type: "drain" });
      };
    } else {
      // DEFAULT: forward each bounded chunk from the worklet to the feature worker (transfer buffer).
      node.port.onmessage = (e) => {
        const d = e.data;
        if (d?.type === "chunk") {
          this.worker.postMessage({ type: "chunk", samples: d.samples }, [d.samples.buffer]);
        }
      };
    }

    source.connect(node);
    // Keep the graph pulling without making noise: a zero-gain sink to destination.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    node.connect(sink).connect(ctx.destination);
    this._sink = sink;

    if (ctx.state === "suspended") {
      // Time-box resume(): under an autoplay policy without a user gesture it can stay pending forever
      // (e.g. headless). We don't block the pipeline on it — the graph is wired either way.
      try {
        await Promise.race([
          ctx.resume(),
          new Promise((r) => setTimeout(r, 800)),
        ]);
      } catch { /* autoplay policy may keep it suspended until a gesture; caller can retry */ }
    }
  }

  stop() {
    try {
      this._oscillator?.stop();
    } catch { /* noop */ }
    this._oscillator = null;
    try {
      this.source?.disconnect();
    } catch { /* noop */ }
    try {
      this.node?.disconnect();
    } catch { /* noop */ }
    try {
      this._sink?.disconnect();
    } catch { /* noop */ }
    if (this.worker) {
      this.worker.postMessage({ type: "stop" });
      this.worker.terminate();
      this.worker = null;
    }
    if (this._ownsCtx && this.ctx) {
      try {
        this.ctx.close();
      } catch { /* noop */ }
    }
    this.ctx = null;
    this.node = null;
    this.source = null;
    this.sab = null;
  }
}
