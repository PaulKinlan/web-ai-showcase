// The Hugging Face Hub `pipeline_tag` on a model card and the transformers.js pipeline task a demo
// drives are two vocabularies for the same work. The currency audit compares them per route, so every
// honest difference showed up as `taskDrift` — 34 of 59 findings in the fb9 sweep, none of them a
// defect (web-ai-showcase-4hv).
//
// This map records the pairs that mean the same thing, with the reason, so the audit can stop
// reporting them while still reporting anything it has not been told about. It is a whitelist, not a
// blanket excuse: an unmapped mismatch stays a finding, which is the property the audit is for.
//
// Keys are `recorded -> upstream`, i.e. `<transformers.js task the demo drives> -> <Hub pipeline_tag>`.
// The route records the transformers.js task, the card records the author's own tag.
export const KNOWN_EQUIVALENT_TASK_PAIRS = {
  "feature-extraction -> sentence-similarity":
    "same behaviour — demo drives the TJS feature-extraction pipeline; the card advertises retrieval",
  "sentence-similarity -> feature-extraction":
    "same behaviour — TJS pipeline name vs card tag; the embedding is the retrieval model",
  "text-classification -> text-ranking":
    "reranker driven as a classification pipeline (TJS has no text-ranking task)",
  "text-classification -> zero-shot-classification": "NLI checkpoint driven as classification",
  "text-to-speech -> text-to-audio": "same behaviour — TJS task name vs card tag",
  "audio-feature-extraction -> feature-extraction":
    "same behaviour — TJS audio task name vs card tag",
  "zero-shot-audio-classification -> feature-extraction":
    "demo runs the zero-shot audio pipeline over the encoder",
  "zero-shot-image-classification -> feature-extraction":
    "demo runs the zero-shot image pipeline over the encoder",
  "zero-shot-object-detection -> object-detection":
    "demo runs the zero-shot detection pipeline (more specific than the card)",
  "text2text-generation -> text-generation": "TJS seq2seq task vs card tag",
  "image-to-image -> text-to-image":
    "card tag is a poor fit for this restoration model; demo uses image-to-image",
  "image-to-image -> image-to-text":
    "card tag is a poor fit for this unwarping model; demo uses image-to-image",
  "image-text-to-text -> text-generation":
    "card tag is a poor fit for this VLM; demo uses image-text-to-text",
  "fill-mask -> text-generation":
    "spell-correction checkpoint driven as fill-mask over masked spans",
};

/**
 * Classify a route's task against the upstream card's tag.
 *
 * Returns { equivalent: true, key, rationale } for a recorded vocabulary pair, and
 * { equivalent: false, key } for anything unknown — the caller decides what an unknown pair means, but
 * it must remain reportable.
 */
export function classifyTaskPair(recorded, upstream) {
  const key = `${recorded} -> ${upstream}`;
  const rationale = KNOWN_EQUIVALENT_TASK_PAIRS[key];
  return rationale ? { equivalent: true, key, rationale } : { equivalent: false, key };
}
