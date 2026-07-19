// media-pipeline.audio-worklet.js — the audio-render-thread half of AudioCapturePipeline.
//
// Registered as "bounded-capture-processor". process() runs in 128-sample render quanta on the audio
// thread. We accumulate mono samples into a FIXED-SIZE chunk (bounded — never an unbounded buffer)
// and hand each full chunk onward two ways:
//
//   • transport "postmessage" (DEFAULT, works on GitHub Pages): post a COPY of the chunk to the main
//     thread via this.port. The main thread transfers it to the feature worker.
//   • transport "sab" (only when cross-origin isolated): write the chunk into a SharedArrayBuffer ring
//     buffer that the feature worker also maps, then post a tiny "wrote" nudge so the worker drains.
//     Uses Atomics for the ring indices. GitHub Pages can't enable this (no COOP/COEP) — it's here to
//     show the isolated-context path; the default above is what actually runs there.

class BoundedCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.chunkSize = o.chunkSize || 2048;
    this.transport = o.transport === "sab" && o.sab ? "sab" : "postmessage";
    this.buf = new Float32Array(this.chunkSize);
    this.fill = 0;

    if (this.transport === "sab") {
      // Ring layout: Int32 header [writeIndex, readIndex] over the first 8 bytes, then Float32 samples.
      this.header = new Int32Array(o.sab, 0, 2);
      this.ring = new Float32Array(o.sab, 8);
      this.capacity = this.ring.length;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0]; // mono (channelCount:1 on the node)
    if (!ch) return true;

    for (let i = 0; i < ch.length; i++) {
      this.buf[this.fill++] = ch[i];
      if (this.fill === this.chunkSize) {
        this._flush();
        this.fill = 0;
      }
    }
    return true; // keep the processor alive
  }

  _flush() {
    if (this.transport === "sab") {
      // Write the chunk into the ring at the current write index (wrapping), then advance it.
      let w = Atomics.load(this.header, 0);
      for (let i = 0; i < this.chunkSize; i++) {
        this.ring[w] = this.buf[i];
        w = (w + 1) % this.capacity;
      }
      Atomics.store(this.header, 0, w);
      this.port.postMessage({ type: "wrote", count: this.chunkSize });
    } else {
      // Copy out (the internal buffer is reused) and transfer the copy to the main thread.
      const out = new Float32Array(this.chunkSize);
      out.set(this.buf);
      this.port.postMessage({ type: "chunk", samples: out }, [out.buffer]);
    }
  }
}

registerProcessor("bounded-capture-processor", BoundedCaptureProcessor);
