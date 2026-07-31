// Focused regressions for canonical Turkish NER BIO repair, exact text transforms, integrity, and
// request lifecycle. Run: node --test test/turkish-ner.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  alignPieces,
  assertArtifactIntegrity,
  entitySpans,
  mergeWords,
  replaceEntitySpans,
  turkishEntityKey,
} from "../models/bert-base-turkish-cased-ner/ner-core.js";

const ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

test("WordPiece continuation disagreement keeps the first-piece BIO label and one exact span", () => {
  const text = "Ahmet Hamdi Tanpınar konuştu.";
  const pieces = ["[CLS]", "Ahmet", "Hamdi", "Tan", "##pınar", "konuştu", ".", "[SEP]"];
  const aligned = alignPieces(pieces, text);
  const labels = ["O", "B-PER", "I-PER", "I-PER", "O", "O", "O", "O"];
  const tokens = pieces.map((word, index) => ({
    word,
    entity: labels[index],
    score: index === 4 ? 0.2 : 0.9,
    ...aligned[index],
  }));
  const words = mergeWords(tokens, text);
  assert.equal(words[2].surface, "Tanpınar");
  assert.equal(words[2].entity, "I-PER");
  assert.deepEqual(entitySpans(words, text).map(({ type, text, start, end }) => ({
    type,
    text,
    start,
    end,
  })), [{ type: "PER", text: "Ahmet Hamdi Tanpınar", start: 0, end: 20 }]);
});

test("orphan I-* and type transitions are repaired into separate BIO spans", () => {
  const text = "Ankara Baykar";
  const words = [
    { surface: "Ankara", entity: "I-LOC", type: "LOC", score: 0.8, start: 0, end: 6 },
    { surface: "Baykar", entity: "I-ORG", type: "ORG", score: 0.9, start: 7, end: 13 },
  ];
  assert.deepEqual(entitySpans(words, text).map((span) => [span.type, span.text]), [
    ["LOC", "Ankara"],
    ["ORG", "Baykar"],
  ]);
});

test("all Wild replacements use original offsets end-to-start", () => {
  const text = "Fatih Sultan Mehmet İstanbul'u fethetti ve Topkapı Sarayı'nı yaptırdı.";
  const names = ["Fatih Sultan Mehmet", "İstanbul", "Topkapı Sarayı"];
  const types = ["PER", "LOC", "LOC"];
  const entities = names.map((name, index) => {
    const start = text.indexOf(name);
    return { type: types[index], text: name, start, end: start + name.length };
  });
  const replacements = new Map([[0, "Zeynep"], [1, "Paris"], [2, "Buckingham"]]);
  assert.equal(
    replaceEntitySpans(text, entities, replacements),
    "Zeynep Paris'u fethetti ve Buckingham'nı yaptırdı.",
  );
});

test("Turkish locale casing de-duplicates dotted and dotless I correctly", () => {
  assert.equal(turkishEntityKey("LOC", "IĞDIR"), turkishEntityKey("LOC", "Iğdır"));
  assert.equal(turkishEntityKey("LOC", "İZMİR"), turkishEntityKey("LOC", "İzmir"));
  assert.notEqual(turkishEntityKey("LOC", "IĞDIR"), turkishEntityKey("LOC", "İZMİR"));
});

test("artifact integrity requires both exact length and SHA-256", async () => {
  const bytes = new TextEncoder().encode("abc").buffer;
  assert.deepEqual(await assertArtifactIntegrity(bytes, {
    expectedBytes: 3,
    expectedSha256: ABC_SHA256,
    label: "fixture",
  }), { byteLength: 3, sha256: ABC_SHA256 });
  await assert.rejects(
    assertArtifactIntegrity(bytes, { expectedBytes: 4, expectedSha256: ABC_SHA256, label: "fixture" }),
    /expected 4 bytes, received 3/,
  );
  await assert.rejects(
    assertArtifactIntegrity(bytes, { expectedBytes: 3, expectedSha256: "0".repeat(64), label: "fixture" }),
    /expected SHA-256/,
  );
});

test("NerEngine is latest-wins, bounded, rejects disposal, and suppresses stale replies", async () => {
  class FakeWorker {
    static instances = [];
    constructor() { this.listeners = new Map(); this.messages = []; this.terminated = false; FakeWorker.instances.push(this); }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    emit(data) { this.listeners.get("message")?.({ data }); }
  }
  globalThis.Worker = FakeWorker;
  const { NerEngine } = await import(`../models/bert-base-turkish-cased-ner/ner.js?test=${Date.now()}`);
  const engine = new NerEngine();
  const worker = FakeWorker.instances[0];
  const load = engine.load();
  worker.emit({ type: "ready", device: "wasm" });
  await load;

  const first = engine.tag("ilk");
  const firstId = worker.messages.at(-1).id;
  const second = engine.tag("ikinci");
  const secondId = worker.messages.at(-1).id;
  await assert.rejects(first, (error) => error.name === "AbortError");
  assert.equal(engine._pending.size, 1);
  assert.ok(worker.messages.some((message) => message.type === "cancel" && message.id === firstId));
  worker.emit({ type: "tag", id: firstId, text: "stale" });
  assert.equal(engine._pending.size, 1);
  worker.emit({ type: "tag", id: secondId, text: "ikinci" });
  assert.equal((await second).text, "ikinci");

  const pending = engine.tag("dispose race");
  const pendingId = worker.messages.at(-1).id;
  engine.dispose("test release");
  await assert.rejects(pending, (error) => error.name === "AbortError" && /test release/.test(error.message));
  assert.equal(engine._pending.size, 0);
  assert.equal(worker.terminated, true);
  worker.emit({ type: "tag", id: pendingId, text: "late" });
  assert.equal(engine._pending.size, 0);
  delete globalThis.Worker;
});
