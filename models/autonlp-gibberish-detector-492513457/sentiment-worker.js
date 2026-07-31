// Pinned SST-2 stage used only by the multi-model route. It never receives text until the detector's
// user-set acceptance policy allows it through.
import { serveWorker } from "/web-ai-showcase/lib/worker-protocol.js";
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
const REVISION = "0b6928efcb76139cae2c6881d49cda67fe119f42";
let pipe = null;

serveWorker({
  async init() {},
  methods: {
    async load(_payload, { onProgress }) {
      const loaded = await loadPipeline({
        task: "text-classification",
        model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        revision: REVISION,
        dtype: "q8",
        backend: "wasm",
        onProgress: (event) => onProgress?.(event),
      });
      pipe = loaded.pipe;
      return {
        result: { backend: loaded.device, sourceModel: MODEL_ID, revision: REVISION, dtype: "q8" },
      };
    },
    async classify({ text }, { signal }) {
      if (!pipe) throw new Error("Sentiment model not loaded");
      if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const started = performance.now();
      const scores = await pipe(String(text), { top_k: null });
      return {
        result: { scores, ms: Math.round(performance.now() - started), sourceModel: MODEL_ID },
      };
    },
  },
  onDispose() {
    pipe = null;
  },
});
