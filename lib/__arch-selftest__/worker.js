// Tiny compute MODULE worker for the reference-architecture self-test. Exercises the worker protocol:
// transferables in/out, streamed progress, cooperative cancellation, and a per-request abort signal.
import { serveWorker, yieldToMain } from "../worker-protocol.js";

serveWorker({
  async init() {
    // No model here — just prove init runs before "ready" is posted.
    self.__ready = true;
  },
  methods: {
    // Sum a transferred ArrayBuffer of Float64 values and transfer a derived buffer back.
    async sumBuffer(payload, { onProgress, signal }) {
      const view = new Float64Array(payload.buffer); // took ownership of the transferred buffer
      let sum = 0;
      const doubled = new Float64Array(view.length);
      for (let i = 0; i < view.length; i++) {
        if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
        sum += view[i];
        doubled[i] = view[i] * 2;
        if (i % 1000 === 0) {
          onProgress({ status: "progress", progress: (i / view.length) * 100 });
          await yieldToMain();
        }
      }
      onProgress({ status: "progress", progress: 100 });
      return {
        result: { sum, length: view.length, doubled: doubled.buffer },
        transfer: [doubled.buffer], // transfer the result buffer back to the main thread
      };
    },

    // A long, cooperative task that streams progress and honours cancellation at each step.
    async slow(payload, { onProgress, signal }) {
      const steps = payload?.steps ?? 40;
      for (let i = 0; i < steps; i++) {
        if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
        onProgress({ status: "progress", progress: (i / steps) * 100, step: i });
        // Busy-wait a little so a mid-flight abort is realistic, yielding to stay cancellable.
        const until = performance.now() + 15;
        while (performance.now() < until) { /* spin briefly */ }
        await yieldToMain();
      }
      return { done: true, steps, tag: payload?.tag };
    },
  },
  onDispose() {
    self.__ready = false;
  },
});
