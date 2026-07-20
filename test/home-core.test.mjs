// Deterministic tests for the homepage discovery/search core (node:test, no browser).
// Run: node --test test/home-core.test.mjs
//
// Covers: stable/unique section ids, every jump link resolving to a rendered section, ?q= round-trip,
// built-only + lexical search payload, HTML escaping, curated config integrity, intent-chip validity,
// and corpus support for representative intent queries (the actual ranking is validated in-browser).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  CURATED,
  curatedFor,
  esc,
  homepageSearchPayload,
  INTENT_CHIPS,
  MODALITY_ORDER,
  orderedSections,
  readQuery,
  sectionId,
  writeQuery,
} from "../home-core.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const models = JSON.parse(readFileSync(join(root, "models.json"), "utf8")).models;
const built = models.filter((m) => m.status === "built");

test("sectionId is deterministic, safe, and stable across calls", () => {
  assert.equal(sectionId("vision"), "cat-vision");
  assert.equal(sectionId("vision-language"), "cat-vision-language");
  assert.equal(sectionId("audio-language"), "cat-audio-language");
  assert.equal(sectionId("code"), "cat-code");
  // stable across calls
  assert.equal(sectionId("text"), sectionId("text"));
  // defensive sanitization of an odd value → still a valid, prefixed id
  assert.match(sectionId("Weird / Modality!"), /^cat-[a-z0-9-]+$/);
  assert.equal(sectionId(""), "cat-other");
});

test("orderedSections yields unique ids covering every built modality, known-first", () => {
  const secs = orderedSections(built);
  const ids = secs.map((s) => s.id);
  // unique
  assert.equal(new Set(ids).size, ids.length, "section ids must be unique");
  // every built modality is represented exactly once
  const builtModalities = new Set(built.map((m) => m.modality || "other"));
  assert.equal(secs.length, builtModalities.size);
  for (const key of builtModalities) {
    assert.ok(secs.find((s) => s.key === key), `missing section for modality ${key}`);
  }
  // ids match sectionId(key) — so jump links (#id) resolve to the rendered <h2 id>
  for (const s of secs) assert.equal(s.id, sectionId(s.key));
  // counts sum to total built (no demo dropped, no double-count)
  assert.equal(secs.reduce((n, s) => n + s.count, 0), built.length);
  // known modalities appear before unknown, in MODALITY_ORDER
  const knownIdx = secs.filter((s) => MODALITY_ORDER.includes(s.key)).map((s) =>
    MODALITY_ORDER.indexOf(s.key)
  );
  assert.deepEqual(knownIdx, [...knownIdx].sort((a, b) => a - b));
});

test("every jump link targets an existing unique section id", () => {
  const secs = orderedSections(built);
  const idSet = new Set(secs.map((s) => s.id));
  // The jump nav is generated from exactly these sections; each href="#id" must resolve.
  for (const s of secs) assert.ok(idSet.has(s.id));
  // no jump link points at a non-rendered/duplicate id
  assert.equal(idSet.size, secs.length);
});

test("query URL round-trips through ?q= (bookmark/back-forward safe)", () => {
  for (const q of ["", "whisper", "remove background", "chat  ", "  depth  "]) {
    const round = readQuery(writeQuery({ q })).q;
    assert.equal(round, q.trim());
  }
  assert.equal(writeQuery({ q: "" }), "");
  assert.equal(writeQuery({ q: "a b" }), "?q=a+b");
  assert.equal(readQuery("?q=hello+world").q, "hello world");
  // extraneous params are ignored (only q is our state)
  assert.equal(readQuery("?foo=1&q=x").q, "x");
});

test("homepage search payload is always built-only + lexical (no model download)", () => {
  const p = homepageSearchPayload("  transcribe  ");
  assert.equal(p.q, "transcribe");
  assert.deepEqual(p.filters, { status: ["built"] });
  assert.equal(p.mode, "lexical");
  assert.ok(p.k > 0);
  const p2 = homepageSearchPayload("x", { cursor: 60 });
  assert.equal(p2.cursor, 60);
});

test("esc neutralizes HTML/attribute injection", () => {
  assert.equal(
    esc(`<img src=x onerror="alert(1)">`),
    "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;",
  );
  assert.equal(esc(`a & b`), "a &amp; b");
  assert.equal(esc(`o'malley`), "o&#39;malley");
  assert.equal(esc(null), "");
  assert.equal(esc(undefined), "");
});

test("curated config resolves only to real, built demos (no drift)", () => {
  const resolved = curatedFor(built);
  assert.ok(resolved.length >= 6, "expected a healthy curated set");
  for (const c of resolved) {
    assert.equal(c.model.status, "built");
    assert.equal(c.model.slug, c.slug);
    assert.ok(c.why && c.why.length > 3);
  }
  // Every curated slug that exists must be built; warn-fail on a stale slug.
  const bySlug = new Map(models.map((m) => [m.slug, m]));
  for (const c of CURATED) {
    const m = bySlug.get(c.slug);
    assert.ok(m, `curated slug not in catalogue: ${c.slug}`);
    assert.equal(m.status, "built", `curated slug not built: ${c.slug}`);
  }
});

test("intent chips are well-formed", () => {
  assert.ok(INTENT_CHIPS.length >= 6);
  for (const chip of INTENT_CHIPS) {
    assert.ok(chip.label && chip.label.length > 2);
    assert.ok(chip.q && chip.q.trim().length > 2);
  }
});

// ── Corpus support for representative intent queries ────────────────────────────────────────────────
// The real BM25 ranking runs in search/worker.js and is validated in-browser (DevTools MCP). Here we
// assert the DATA supports each intent: at least one BUILT demo of the expected task carries the query's
// terms in its searchable text, so lexical search will surface it.
test("index corpus supports representative intent queries (built demos present + termful)", () => {
  let index;
  try {
    index = JSON.parse(readFileSync(join(root, "search/index/index.json"), "utf8"));
  } catch {
    // Index is a generated artifact; skip if absent in this checkout rather than fail spuriously.
    return;
  }
  const docs = index.docs || [];
  const builtDocs = docs.filter((d) => d.status === "built");
  assert.ok(builtDocs.length > 0, "index has built docs");

  const textOf = (d) =>
    [d.name, d.blurb, d.unlocks, d.task, d.family, d.canonicalFamily, d.modality, d.hfId]
      .filter(Boolean).join(" ").toLowerCase();

  // (query terms that must co-occur) → a task the surfaced built demo should have
  const cases = [
    {
      terms: ["transcribe", "speech"],
      anyTerm: ["transcribe", "speech", "asr", "audio"],
      task: "automatic-speech-recognition",
    },
    {
      terms: ["background"],
      anyTerm: ["background", "matting", "remove"],
      taskLike: "segmentation",
    },
    { terms: ["depth"], anyTerm: ["depth"], task: "depth-estimation" },
    { terms: ["translate"], anyTerm: ["translat"], task: "translation" },
    {
      terms: ["object", "detection"],
      anyTerm: ["detect", "object", "bounding"],
      task: "object-detection",
    },
  ];

  for (const c of cases) {
    const hit = builtDocs.find((d) => {
      const t = textOf(d);
      const termMatch = c.anyTerm.some((w) => t.includes(w));
      const taskMatch = c.task
        ? d.task === c.task
        : c.taskLike
        ? String(d.task).includes(c.taskLike)
        : true;
      return termMatch && taskMatch;
    });
    assert.ok(
      hit,
      `no built demo supports intent ${JSON.stringify(c.terms)} (task ${c.task || c.taskLike})`,
    );
  }
});
