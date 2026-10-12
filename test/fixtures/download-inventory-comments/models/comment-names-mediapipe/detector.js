// model-prefetch.mjs / prefetchModel( are the FIRST family in FAMILY_ORDER, and FilesetResolver
// (mediapipe) is named here too. Both are MENTIONED IN THIS COMMENT ONLY; if comments were matched this
// route would be misclassified as the earliest family. The executable import below is WebLLM.
import { CreateMLCEngine } from "/web-ai-showcase/lib/webllm.js";

export const boot = () => CreateMLCEngine("fixture-model");
