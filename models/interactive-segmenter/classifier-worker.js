// Second genuine model for the multi-model route: MobileViT classifies the MagicTouch cut-out.
import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";
import { serveWorker } from "/web-ai-showcase/lib/worker-protocol.js";

const REVISION = "a0e5dde3e1b3d9e49894dcaeeb1e046ed01edfc8";
let classifier;

serveWorker({
  methods: {
    async configure(_payload, { onProgress }) {
      const { pipeline, env } = await import(TRANSFORMERS_URL);
      env.allowLocalModels = false;
      classifier = await pipeline("image-classification", "Xenova/mobilevit-small", {
        device: "wasm",
        dtype: "q8",
        revision: REVISION,
        progress_callback: onProgress,
      });
      return {
        result: {
          device: "wasm",
          modelId: "Xenova/mobilevit-small",
          revision: REVISION,
        },
      };
    },
    async classify(payload) {
      if (!classifier) throw new Error("MobileViT not configured");
      const started = performance.now();
      const labels = await classifier(payload.image, { top_k: payload.topK ?? 5 });
      return {
        result: {
          labels,
          latencyMs: Math.round((performance.now() - started) * 10) / 10,
        },
      };
    },
  },
  onDispose() {
    classifier?.dispose?.();
    classifier = null;
  },
});
