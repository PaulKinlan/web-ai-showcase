// Shared PaliGemma asset manifest — the SINGLE source of the big weight files, imported by both the
// worker (which prefetches them resumably) and the pages (whose "Discard partial downloads" clears the
// same URLs). Keeping one list means the prefetch and the discard can never target different keys.
//
// These names are the dtype-suffixed component files for the worker's dtype map
// {embed_tokens:q8→_quantized, vision_encoder:fp16→_fp16, decoder_model_merged:q4f16→_q4f16}.
// (No .onnx_data external-data files exist for these variants.)
export const PALIGEMMA_MODEL_ID = "onnx-community/paligemma2-3b-pt-224";
export const PALIGEMMA_BIG_FILES = [
  "onnx/embed_tokens_quantized.onnx",
  "onnx/vision_encoder_fp16.onnx",
  "onnx/decoder_model_merged_q4f16.onnx",
];
