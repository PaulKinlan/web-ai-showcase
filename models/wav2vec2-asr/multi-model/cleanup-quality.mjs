// A cleanup must add sentence casing and terminal punctuation WITHOUT changing
// the CTC transcript's words. Shared by the browser worker and acceptance gate.
export function evaluateCleanup(raw, cleaned) {
  const source = String(raw ?? "").trim();
  const output = String(cleaned ?? "").trim();
  const words = (text) => text.toLocaleLowerCase("en-US")
    .match(/[a-z0-9]+(?:'[a-z0-9]+)*/g) ?? [];
  const original = words(source);
  const proposed = words(output);
  const sameWords = original.length > 0 && original.length === proposed.length &&
    original.every((word, index) => word === proposed[index]);
  const sentenceCase = /^[A-Z]/.test(output);
  const terminalPunctuation = /[.!?]$/.test(output);
  return { valid: sameWords && sentenceCase && terminalPunctuation,
    sameWords, sentenceCase, terminalPunctuation };
}

// Honest minimum, not a Qwen result: restore ONLY the first capital and a
// terminal period. The UI explicitly labels it as a basic rule-based fallback.
export function basicFallback(raw) {
  const source = String(raw ?? "").trim();
  if (!source) return "";
  const cased = source.toLocaleLowerCase("en-US").replace(/^[a-z]/, c => c.toUpperCase());
  return /[.!?]$/.test(cased) ? cased : `${cased}.`;
}
