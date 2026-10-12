// Regression fixture for web-ai-showcase-x4ba. This comment NAMES another runtime's API and used to
// flip the whole route to raw-ort (sitely-controlled bytes, restart-only resume).
// ort.env.wasm.numThreads = 1 and InferenceSession are MENTIONED HERE ONLY — never called.
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

export const load = () => loadPipeline({ task: "image-classification", model: "fixture/model" });
