// home-core.mjs — pure, DOM-free helpers shared by the homepage controller (home.js, browser) and the
// deterministic node tests (test/home-core.test.mjs). No imports; no side effects. Everything here is
// data-in/data-out so it can be unit-tested without a browser and reused verbatim by the page.
//
// Design notes (modern-web-guidance ids retained: accessibility, search-hidden-content, size-aware-styling):
// - Section ids derive from the DURABLE modality key, never array position or display text (which drift).
// - Homepage search is built-only + lexical (BM25 in the worker) so it works with no model download.
// - Query state round-trips through `?q=` for bookmarkable/history-safe search.

// ── Section taxonomy — the REAL generated categories (models.json `modality`), not an invented one ────
// Order is deterministic; only sections with ≥1 built demo render (computed in orderedSections()).
export const MODALITY_ORDER = [
  "vision",
  "vision-language",
  "audio",
  "audio-language",
  "text",
  "code",
];

export const MODALITY_LABEL = {
  "vision": "Vision & sight",
  "vision-language": "Vision + language (VLMs)",
  "audio": "Audio & speech",
  "audio-language": "Audio + language",
  "text": "Text & language",
  "code": "Code",
};

export const MODALITY_BLURB = {
  "vision": "See, segment, restore and understand images.",
  "vision-language": "Models that look at an image and talk about it.",
  "audio": "Transcribe, classify, and synthesize sound and speech.",
  "audio-language": "Cross-modal audio ↔ text.",
  "text": "Generate, translate, classify, and embed language.",
  "code": "Code generation and understanding.",
};

/**
 * Deterministic, stable section id from a durable modality key.
 * `cat-<modality>` — the modality strings are already safe kebab-case tokens
 * (vision, vision-language, audio, audio-language, text, code). We still sanitize
 * defensively so any future modality value yields a valid, unique HTML id.
 */
export function sectionId(modality) {
  const key = String(modality || "other")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "other";
  return `cat-${key}`;
}

/**
 * Given the built models, return the ordered list of sections actually present.
 * Known modalities come first in MODALITY_ORDER; any unknown modality is appended
 * (sorted) so a new category is NEVER silently dropped. Each entry has a unique id.
 */
export function orderedSections(builtModels) {
  const counts = new Map();
  for (const m of builtModels) {
    const key = m.modality || "other";
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const known = MODALITY_ORDER.filter((k) => counts.has(k));
  const unknown = [...counts.keys()].filter((k) => !MODALITY_ORDER.includes(k)).sort();
  const out = [...known, ...unknown].map((key) => ({
    key,
    id: sectionId(key),
    label: MODALITY_LABEL[key] || labelize(key),
    blurb: MODALITY_BLURB[key] || "",
    count: counts.get(key),
  }));
  // Guard: ids must be unique (they are, since one section per distinct modality).
  return out;
}

function labelize(key) {
  return String(key).replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── HTML escaping for any untrusted catalogue text (name/blurb/task/unlocks/hfId) ─────────────────────
export function esc(s) {
  return String(s == null ? "" : s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

// ── Query URL state (bookmarkable, history-safe) ──────────────────────────────────────────────────────
/** Parse the homepage search query from a location.search string (`?q=...`). */
export function readQuery(searchString) {
  const p = new URLSearchParams(searchString || "");
  return { q: (p.get("q") || "").trim() };
}

/** Serialize `{q}` back to a canonical `?q=...` (empty when no query) for pushState/replaceState. */
export function writeQuery({ q } = {}) {
  const p = new URLSearchParams();
  const query = (q || "").trim();
  if (query) p.set("q", query);
  const s = p.toString();
  return s ? `?${s}` : "";
}

/**
 * The homepage search payload for the shared search worker. ALWAYS built-only and lexical:
 * lexical BM25 needs no model download (the 25 MB intent model stays on /explore/ only), and
 * status:["built"] keeps homepage results to the interactive demos.
 */
// k defaults to 300 — comfortably above the built-demo count, so a built-only homepage search returns
// EVERY match in one page (hits.length === total ⇒ the shown count is exact, no silent truncation).
export function homepageSearchPayload(q, { k = 300, cursor } = {}) {
  const payload = {
    q: (q || "").trim(),
    filters: { status: ["built"] },
    mode: "lexical",
    expandAliases: true,
    k,
  };
  if (cursor != null) payload.cursor = cursor;
  return payload;
}

// ── Discovery: "Try a task" intent chips (set the query; NOT a separate engine) ───────────────────────
// Each chip is a plain query string fed to the same lexical search. Labels are human intents; the
// query carries the terms/aliases that surface the relevant built demos.
export const INTENT_CHIPS = [
  { label: "Transcribe speech", q: "transcribe speech audio to text" },
  { label: "Describe an image", q: "describe image caption visual question answering" },
  { label: "Chat with an LLM", q: "chat assistant text generation language model" },
  { label: "Remove a background", q: "remove background image cut out subject" },
  { label: "Depth from a photo", q: "depth estimation distance 3d from single image" },
  { label: "Translate text", q: "translate translation between languages" },
  { label: "Text to speech", q: "text to speech synthesize voice" },
  { label: "Detect objects", q: "object detection bounding boxes in image" },
  { label: "Embed / semantic search", q: "sentence embeddings semantic similarity search" },
  { label: "Upscale an image", q: "super resolution upscale enhance image" },
];

// ── Curated "good places to start" — an explicit tiny config with capability rationale ────────────────
// Chosen for REPRESENTATIVE CAPABILITY breadth, not popularity, and never a fabricated quality score.
// curatedFor() filters to entries that are actually present + built, so a retired slug just drops out.
export const CURATED = [
  { slug: "whisper-speech-to-text", why: "Speech → text, fully in the browser" },
  { slug: "smolvlm-vision-language", why: "Ask questions about any image (VLM)" },
  { slug: "qwen-tiny-llm", why: "A chat LLM running on-device" },
  { slug: "rmbg-background-removal", why: "One-click background removal" },
  { slug: "depth-anything", why: "Depth from a single photo" },
  { slug: "kokoro-text-to-speech", why: "Natural text-to-speech voices" },
  { slug: "clip-zero-shot-image", why: "Classify an image by your own labels" },
  { slug: "detr-object-detection", why: "Find objects in a photo" },
  { slug: "swin2sr-super-resolution", why: "Upscale a low-res image 2×" },
  { slug: "minilm-embeddings", why: "Semantic embeddings for search" },
];

/** Keep only curated entries whose slug is present and status==="built"; preserve config order. */
export function curatedFor(builtModels) {
  const bySlug = new Map(builtModels.map((m) => [m.slug, m]));
  const out = [];
  for (const c of CURATED) {
    const m = bySlug.get(c.slug);
    if (m) out.push({ ...c, model: m });
  }
  return out;
}
