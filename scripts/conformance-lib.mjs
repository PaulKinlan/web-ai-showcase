#!/usr/bin/env node
// Shared library for the web-ai-showcase critique + immutable-conformance + goal lifecycle.
//
// This is a WEB-AI-domain analog of chrome-platform-showcase's conformance/critique system — NOT a
// copy of its Chrome-platform assertions. Assertions here describe what a *browser-runnable model
// demo* must hold: real local inference, expected I/O shape + semantic sanity, runtime/backend/
// model-id/quantisation honesty, auto-init-on-cached behaviour, honest download/error/offline states,
// every visible control, no fake output, accessibility, a mobile+desktop matrix, and that the
// frontend consulted modern-web-guidance.
//
// Suites are DERIVED from real models.json metadata + per-task templates (genuine, not fake), and are
// IMMUTABLE once committed: you fix the demo, never weaken the assertion. scripts/check-conformance.mjs
// recomputes each suiteHash and diffs against origin/main to detect weakening/removal without a
// migration record.

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const SCHEMA_VERSION = 1;
export const CRITIQUE_SCHEMA_VERSION = 1;

const repoRoot = fileURLToPath(new URL("..", import.meta.url));

// ── Catalogue helpers ───────────────────────────────────────────────────────

export function loadCatalogue() {
  const raw = JSON.parse(readFileSync(`${repoRoot}models.json`, "utf8"));
  return Array.isArray(raw) ? raw : (raw?.models ?? []);
}

export function builtModels(catalogue = loadCatalogue()) {
  return catalogue.filter((m) => m.status === "built");
}

export function modelDir(slug) {
  return `${repoRoot}models/${slug}/`;
}

// Concatenate the model page HTML + its colocated JS (the demo's real source) for static assertions.
export function modelSource(slug) {
  const dir = modelDir(slug);
  const html = existsSync(`${dir}index.html`) ? readFileSync(`${dir}index.html`, "utf8") : "";
  let js = "";
  if (existsSync(dir)) {
    for (const f of readdirSync(dir)) {
      if (f.endsWith(".js")) js += "\n" + readFileSync(`${dir}${f}`, "utf8");
    }
  }
  return { html, js, all: html + "\n" + js };
}

const isMediaPipe = (m) => typeof m.hfId === "string" && m.hfId.startsWith("mediapipe/");

// ── Hashing / normalization (immutability) ───────────────────────────────────

// Canonical, order-independent representation of a single assertion. Only the fields that define the
// CONTRACT are hashed — a cosmetic re-order never changes the hash, but weakening `describe`/`test`/
// `kind`/`deviceClass`/`category` does.
export function normalizeAssertion(a) {
  return {
    id: a.id,
    category: a.category,
    describe: a.describe,
    kind: a.kind,
    deviceClass: a.deviceClass,
    test: a.test ?? null,
    expect: a.expect ?? null,
  };
}

export function computeSuiteHash(assertions) {
  const norm = assertions
    .map(normalizeAssertion)
    .sort((x, y) => (x.id < y.id ? -1 : x.id > y.id ? 1 : 0));
  return "sha256:" + createHash("sha256").update(JSON.stringify(norm)).digest("hex");
}

// ── Per-task I/O templates (expected output shape + semantic sanity) ─────────
// Each entry describes what a correct run of that task must produce, in reviewer-checkable terms.
// These become the `io-shape` (manual-evidenced) assertion's `describe`, so semantic sanity is judged
// against the REAL model output, never faked.
const IO_SHAPE = {
  "image-text-to-text":
    "produces coherent generated text that actually references the given image + prompt (VLM), streamed token-by-token",
  "automatic-speech-recognition": "produces a transcript string whose words match the spoken audio",
  "text-to-speech": "produces playable audio whose speech matches the input text",
  "text-to-audio": "produces playable audio that corresponds to the text/description prompt",
  "text-generation": "streams coherent continuation/answer tokens conditioned on the prompt",
  "text2text-generation": "produces a transformed text output appropriate to the instruction",
  "summarization": "produces a shorter text that faithfully summarises the input",
  "translation": "produces text in the target language that preserves the source meaning",
  "fill-mask": "produces ranked token candidates for the masked position with probabilities",
  "feature-extraction": "produces a fixed-length embedding vector (numeric array) for the input",
  "image-feature-extraction": "produces a fixed-length image embedding vector (numeric array)",
  "sentence-similarity": "produces similarity scores between the query and candidate sentences",
  "text-classification":
    "produces labelled class scores that sum sensibly and match the input tone",
  "token-classification":
    "produces per-token labels/spans (e.g. entities) aligned to the input text",
  "question-answering": "produces an answer span drawn from the supplied context",
  "zero-shot-classification": "produces scores across the supplied candidate labels",
  "zero-shot-image-classification":
    "produces scores across the supplied candidate labels for the image",
  "zero-shot-audio-classification":
    "produces scores across the supplied candidate labels for the audio",
  "zero-shot-object-detection": "produces boxes+labels+scores for the supplied candidate labels",
  "object-detection":
    "produces bounding boxes with class labels and confidence scores drawn on the image",
  "image-segmentation": "produces per-pixel masks/regions overlaid on the image",
  "mask-generation": "produces segmentation masks for the prompted point/box on the image",
  "depth-estimation": "produces a per-pixel depth map visualised over the image",
  "image-classification": "produces ranked class labels with probabilities for the image",
  "image-to-text": "produces a caption/description that matches the image content",
  "image-to-image": "produces a transformed output image (e.g. upscaled/cleaned) from the input",
  "document-question-answering": "produces an answer grounded in the document image + question",
  "audio-classification": "produces ranked audio class labels with probabilities",
  "face-landmark-detection": "produces face landmark points overlaid live on the video/image",
  "hand-landmark-detection": "produces hand landmark points overlaid live on the video/image",
  "pose-landmark-detection": "produces body pose landmark points overlaid live on the video/image",
  "gesture-recognition": "produces recognised gesture labels with scores from the live video",
  "face-detection": "produces face bounding boxes overlaid on the video/image",
  "any-to-any":
    "produces the modality-appropriate output for the selected sub-task, grounded in the input",
};

function ioShapeText(model) {
  return IO_SHAPE[model.task] ||
    `produces the expected ${model.task} output, semantically consistent with the input`;
}

// ── Assertion builders ───────────────────────────────────────────────────────
// Every assertion carries a stable id, a domain `category`, a human `describe` (the contract), a
// `kind` the runner understands, and a `deviceClass`. `test` semantics depend on kind:
//   console-clean  — page loads with no uncaught error / console.error         (auto, in-page)
//   dom            — { selector, min } querySelectorAll(selector).length >= min (auto, in-page)
//   page-text      — substring must appear in the rendered document text/HTML   (auto, in-page)
//   script         — JS expression evaluated in the page, coerced to boolean    (auto, in-page)
//   source         — { file?, pattern, mode } static check over the demo source (auto, on-disk)
//   capability     — { probe } real capability probe; genuinely-unavailable ⇒ blocked, never a pass
//   responsive     — { assert } viewport-scoped layout check (mobile/desktop)   (auto, in-page)
//   manual-evidenced — screenshot captured; an agent Reads it and records a verdict + evidence
const A = (id, category, describe, kind, deviceClass, test = null, expect = null) => ({
  id,
  category,
  describe,
  kind,
  deviceClass,
  ...(test !== null ? { test } : {}),
  ...(expect !== null ? { expect } : {}),
});

// Derive the immutable assertion array for one built model from its real metadata + task template.
export function deriveAssertions(model) {
  const mp = isMediaPipe(model);
  const out = [];

  // ── real-local-inference correctness ──
  out.push(A(
    "loads-clean",
    "real-inference",
    "The model page loads and initialises its control UI with no uncaught errors on the page or console.",
    "console-clean",
    "both",
  ));
  out.push(A(
    "calls-real-inference",
    "real-inference",
    "The demo source calls a REAL in-browser inference path (transformers.js pipeline / WebLLM engine / MediaPipe Tasks) — never a canned result.",
    "source",
    "both",
    {
      pattern:
        "pipeline\\(|createModelLoader|CreateMLCEngine|FilesetResolver|\\.generate\\(|\\.transcribe|MediaPipe|reranker|extractor|classifier",
      mode: "contains",
    },
  ));
  out.push(A(
    "io-shape-semantic",
    "io-shape",
    `Expected output shape + semantic sanity: a real run ${ioShapeText(model)}.`,
    "manual-evidenced",
    "both",
  ));

  // ── runtime / backend / model-id / quantisation honesty ──
  if (mp) {
    out.push(A(
      "declares-mediapipe-runtime",
      "runtime-config",
      "At-a-glance declares the MediaPipe Tasks runtime and the exact task the demo runs.",
      "page-text",
      "both",
      "MediaPipe",
    ));
  } else {
    out.push(A(
      "declares-model-id",
      "runtime-config",
      `At-a-glance shows the exact source model id (${model.hfId}) linked to its card — no invented model.`,
      "page-text",
      "both",
      model.hfId,
    ));
  }
  out.push(A(
    "declares-task",
    "runtime-config",
    `At-a-glance names the real pipeline task (${model.task}).`,
    "page-text",
    "both",
    model.task,
  ));
  out.push(A(
    "declares-backend",
    "runtime-config",
    `The page declares the execution backend (${
      model.backend === "webgpu" ? "WebGPU" : "WASM/WebGPU"
    }).`,
    "script",
    "both",
    "/webgpu|wasm/i.test(document.documentElement.innerText)",
  ));
  if (model.dtype) {
    // dtype can be compound (e.g. "q4f16 (WebGPU) / q8 (WASM)"); assert on the primary dtype token.
    const dtypeToken = String(model.dtype).match(/[a-z0-9]+/i)?.[0] || model.dtype;
    out.push(A(
      "declares-quantisation",
      "runtime-config",
      `At-a-glance states the quantisation/dtype (${model.dtype}) the demo actually loads.`,
      "page-text",
      "both",
      dtypeToken,
    ));
  }
  out.push(A(
    mp ? "mediapipe-gpu-delegate" : "inference-off-main-thread",
    "runtime-config",
    mp
      ? "Inference runs via the MediaPipe GPU/WASM delegate (FilesetResolver), keeping the control UI responsive."
      : "Inference runs off the main thread (a Web Worker) so the control UI stays responsive and INP stays low.",
    "source",
    "both",
    mp
      ? { pattern: "FilesetResolver|createFromOptions|delegate", mode: "contains" }
      : { pattern: "new Worker|worker\\.js", mode: "contains" },
  ));

  // ── download / cache / current-version auto-init behaviour ──
  out.push(A(
    "shared-auto-init-loader",
    "cache-init",
    "The page loads its model through the shared createModelLoader auto-init architecture — not a hand-rolled Load button.",
    "source",
    "both",
    { pattern: "createModelLoader", mode: "contains" },
  ));
  out.push(A(
    "auto-init-on-cached",
    "cache-init",
    "A valid current on-device model auto-initialises (checking → initialising → ready) with no Load click for a returning user; only Download/Re-download/Update surface otherwise.",
    "manual-evidenced",
    "both",
  ));
  out.push(A(
    "live-status-region",
    "states",
    "Loading/progress/ready/error state is announced in an accessible live region (role=status, aria-live).",
    "dom",
    "both",
    { selector: '[role="status"][aria-live], [role="status"]', min: 1 },
  ));

  // ── progress / error / retry / offline states ──
  out.push(A(
    "errors-on-the-page",
    "states",
    "Load/inference/WebGPU/memory failures surface on the page (an .err/error status the visitor can read), never only in the console.",
    "source",
    "both",
    { pattern: "err|error|Retry|retry", mode: "contains" },
  ));
  out.push(A(
    "honest-progress-retry-offline",
    "states",
    "Download progress, retry/recovery on failure, and cached offline use are handled honestly (no fake success, no silent large re-download).",
    "manual-evidenced",
    "both",
  ));

  // ── capability / device honesty ──
  if (model.backend === "webgpu") {
    out.push(A(
      "webgpu-capability-honest",
      "states",
      "This model prefers WebGPU; on a device with no GPU adapter the page shows a labelled HONEST state — either the WASM-fallback/download path or a needs-WebGPU/unsupported explanation with enable steps — never a blank panel or faked output.",
      "capability",
      "both",
      { probe: "webgpu" },
    ));
  } else {
    out.push(A(
      "runs-without-webgpu-fallback",
      "states",
      "This model runs on the WASM path (no GPU required); on a device without WebGPU it still works or degrades honestly with a labelled reason.",
      "manual-evidenced",
      "both",
    ));
  }

  // ── every visible control ──
  out.push(A(
    "controls-operable",
    "controls",
    "Every visible control (inputs, buttons, samples, parameter widgets) is real, labelled, and keyboard-operable — no dead or decorative controls.",
    "manual-evidenced",
    "both",
  ));

  // ── no fake output ──
  out.push(A(
    "no-fake-output",
    "no-fake-output",
    "No canned/hardcoded result is presented as live model output; the visible output is produced by the on-device run (or an honest unsupported state).",
    "manual-evidenced",
    "both",
  ));

  // ── accessibility ──
  out.push(A(
    "images-have-alt",
    "accessibility",
    "All content images expose alternative text (or are marked decorative).",
    "script",
    "both",
    "Array.from(document.querySelectorAll('img')).every(i=>i.hasAttribute('alt'))",
  ));
  out.push(A(
    "buttons-have-names",
    "accessibility",
    "Every button exposes an accessible name (text, aria-label, or aria-labelledby).",
    "script",
    "both",
    "Array.from(document.querySelectorAll('button')).every(b=>((b.textContent||'').trim()||b.getAttribute('aria-label')||b.getAttribute('aria-labelledby')))",
  ));

  // ── mobile + desktop matrix ──
  out.push(A(
    "responsive-desktop",
    "responsive",
    "At desktop (≈1280×800) the layout has no unintended horizontal overflow and no clipped controls/text.",
    "responsive",
    "desktop",
    { assert: "no-horizontal-overflow" },
  ));
  out.push(A(
    "responsive-mobile",
    "responsive",
    "At narrow mobile (≈360×740, DPR3, touch) the layout has no horizontal overflow, legible text, and ≈44px tap targets.",
    "responsive",
    "mobile",
    { assert: "no-horizontal-overflow" },
  ));
  out.push(A(
    "parity-both-classes",
    "responsive",
    "The demo is a usable, polished experience on BOTH mobile and desktop, or is honestly recorded unsupported on a class with evidence (never left unfinished).",
    "manual-evidenced",
    "both",
  ));

  // ── performance / INP ──
  out.push(A(
    "inp-stays-low",
    "performance",
    "Interacting with the controls keeps the main thread responsive (inference off-thread, long tasks broken up) — no INP regression from a blocking loop.",
    "manual-evidenced",
    "both",
  ));

  // ── modern-web-guidance build process ──
  out.push(A(
    "guidance-consulted",
    "build-process",
    "The frontend implementation consulted modern-web-guidance for its UI/API topics and applied or justified the recommendations (recorded in the critique's guidanceConsulted).",
    "manual-evidenced",
    "both",
  ));

  return out;
}

export function deriveSuite(model, { generatedAt, author } = {}) {
  const assertions = deriveAssertions(model);
  return {
    schemaVersion: SCHEMA_VERSION,
    id: model.slug,
    immutable: true,
    route: `models/${model.slug}/`,
    identity: {
      hfId: model.hfId ?? null,
      task: model.task ?? null,
      backend: model.backend ?? null,
    },
    generatedAt: generatedAt || "1970-01-01T00:00:00Z",
    author: author || "derive-conformance",
    suiteHash: computeSuiteHash(assertions),
    assertions,
  };
}

// ── Lightweight structural schema validation (no external deps) ──────────────
// Enough to catch malformed artifacts in the gate; the JSON Schema files under schemas/ are the
// human-facing contract.
const CATEGORIES = new Set([
  "real-inference",
  "io-shape",
  "runtime-config",
  "cache-init",
  "states",
  "controls",
  "no-fake-output",
  "accessibility",
  "responsive",
  "performance",
  "build-process",
]);
const KINDS = new Set([
  "console-clean",
  "dom",
  "page-text",
  "script",
  "source",
  "capability",
  "responsive",
  "manual-evidenced",
]);
const DEVICE_CLASSES = new Set(["both", "desktop", "mobile", "n/a"]);

export function validateSuite(suite) {
  const errs = [];
  if (suite.schemaVersion !== SCHEMA_VERSION) errs.push(`schemaVersion must be ${SCHEMA_VERSION}`);
  if (!suite.id) errs.push("missing id");
  if (suite.immutable !== true) errs.push("immutable must be true");
  if (!Array.isArray(suite.assertions) || suite.assertions.length === 0) {
    errs.push("assertions must be a non-empty array");
    return errs;
  }
  const ids = new Set();
  for (const a of suite.assertions) {
    if (!a.id) errs.push("assertion missing id");
    if (ids.has(a.id)) errs.push(`duplicate assertion id: ${a.id}`);
    ids.add(a.id);
    if (!CATEGORIES.has(a.category)) errs.push(`${a.id}: bad category ${a.category}`);
    if (!KINDS.has(a.kind)) errs.push(`${a.id}: bad kind ${a.kind}`);
    if (!DEVICE_CLASSES.has(a.deviceClass)) errs.push(`${a.id}: bad deviceClass ${a.deviceClass}`);
  }
  const wantHash = computeSuiteHash(suite.assertions);
  if (suite.suiteHash !== wantHash) {
    errs.push(`suiteHash mismatch: stored ${suite.suiteHash} != computed ${wantHash}`);
  }
  return errs;
}

const CRITIQUE_DIMENSIONS = new Set([
  "real-inference",
  "io-shape",
  "runtime-config",
  "cache-init",
  "states",
  "controls",
  "no-fake-output",
  "accessibility",
  "responsive",
  "performance",
]);

export function validateCritique(c) {
  const errs = [];
  if (c.schemaVersion !== CRITIQUE_SCHEMA_VERSION) {
    errs.push(`schemaVersion must be ${CRITIQUE_SCHEMA_VERSION}`);
  }
  if (!c.id) errs.push("missing id");
  if (typeof c.revision !== "number") errs.push("revision must be a number");
  if (!Array.isArray(c.rubric) || c.rubric.length === 0) {
    errs.push("rubric must be a non-empty array");
  } else {
    for (const r of c.rubric) {
      if (!CRITIQUE_DIMENSIONS.has(r.dimension)) errs.push(`bad rubric dimension: ${r.dimension}`);
      if (typeof r.score !== "number" || r.score < 0 || r.score > 5) {
        errs.push(`${r.dimension}: score must be 0-5`);
      }
      if (!["info", "minor", "major", "critical"].includes(r.severity)) {
        errs.push(`${r.dimension}: bad severity ${r.severity}`);
      }
      if (!r.evidence) errs.push(`${r.dimension}: evidence is required (retain REAL evidence)`);
    }
  }
  if (!Array.isArray(c.guidanceConsulted)) errs.push("guidanceConsulted must be an array");
  // A frontend critique with empty guidanceConsulted is INCOMPLETE (the mandate).
  if (
    Array.isArray(c.guidanceConsulted) && c.guidanceConsulted.length === 0 && c.frontend !== false
  ) {
    errs.push(
      "INCOMPLETE: guidanceConsulted is empty for a frontend critique (modern-web-guidance mandate)",
    );
  }
  if (!Array.isArray(c.followUpGoals)) errs.push("followUpGoals must be an array");
  return errs;
}

export { isMediaPipe, repoRoot };
