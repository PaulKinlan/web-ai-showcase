import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { inScope } from "../scripts/audio-provenance-lib.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const TED_SHA = "0013db1382f6fa95c4f3482594ba4cae0fd7da2104e6e36b79f4163540c676bf";
const LEDGER = JSON.parse(readFileSync(join(ROOT, "audio-provenance/ledger.json"), "utf8"));
const TRACKED = execFileSync("git", ["ls-files", "-z"], { cwd: ROOT })
  .toString("utf8").split("\0").filter(Boolean);
const sha256 = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");
const SOURCE = "https://www.openslr.org/12";
const LICENSE = "https://creativecommons.org/licenses/by/4.0/";
const CREDITS = "audio-provenance/librispeech-replacements.md";

// The immutable baseline is allowed to name the retired incident hash; the current tree is not.
test("no tracked media contains the retired TED bytes, nor any current ledger entry", () => {
  assert.ok(LEDGER.legacyBaseline.hashes.includes(TED_SHA), "historical baseline stays pinned");
  assert.ok(!LEDGER.entries.some((entry) => entry.hash === TED_SHA), "retired TED entry must be absent");
  for (const relative of TRACKED.filter(inScope)) {
    assert.notEqual(sha256(join(ROOT, relative)), TED_SHA, relative);
  }
});

test("eleven replacement paths have ten declared, licensed and correctly hashed recordings", () => {
  const paths = TRACKED.filter((path) => /^models\/[^/]+\/(?:speech-)?librispeech\.wav$/.test(path));
  assert.equal(paths.length, 11);
  const entries = LEDGER.entries.filter((entry) => entry.paths.some((path) => paths.includes(path)));
  assert.equal(entries.length, 10);
  assert.equal(entries.reduce((n, entry) => n + entry.paths.filter((p) => paths.includes(p)).length, 0), 11);
  for (const relative of paths) {
    const hash = sha256(join(ROOT, relative));
    const entry = entries.find((candidate) => candidate.hash === hash);
    assert.ok(entry?.paths.includes(relative), `missing licensed hash/path ${relative}`);
    assert.equal(entry.rightsCleared, true);
    assert.equal(entry.archetype, "licensed");
    assert.equal(entry.provenance.license, "CC BY 4.0");
    assert.equal(entry.provenance.sourceUrl, SOURCE);
    assert.match(entry.provenance.attribution, /Vassil Panayotov/);
    assert.match(entry.provenance.attribution, /utterance[s]? \d+-\d+-\d+/);
    assert.match(entry.provenance.sourceAsset, /LibriSpeech\/dev-clean\/\d+\/\d+\/\d+-\d+-\d+\.flac/);
  }
  const composite = entries.find((entry) => entry.paths.includes("models/speaker-diarization/librispeech.wav"));
  assert.match(composite.provenance.attribution, /0\.300 s digital silence/);
  assert.match(composite.provenance.creator, /iamartin.*Winston Tharp/);
  assert.ok(existsSync(join(ROOT, CREDITS)));
});

test("replacement WAV files are non-silent 16 kHz mono PCM with bounded durations", () => {
  for (const path of TRACKED.filter((p) => /\/(?:speech-)?librispeech\.wav$/.test(p))) {
    const data = readFileSync(join(ROOT, path));
    assert.equal(data.toString("ascii", 0, 4), "RIFF", path);
    assert.equal(data.toString("ascii", 8, 12), "WAVE", path);
    assert.equal(data.toString("ascii", 12, 16), "fmt ", path);
    assert.equal(data.readUInt16LE(20), 1, `PCM ${path}`);
    assert.equal(data.readUInt16LE(22), 1, `mono ${path}`);
    assert.equal(data.readUInt32LE(24), 16000, `16 kHz ${path}`);
    assert.equal(data.readUInt16LE(34), 16, `16-bit ${path}`);
    assert.equal(data.toString("ascii", 36, 40), "data", path);
    const frames = data.readUInt32LE(40) / 2;
    assert.ok(frames / 16000 >= 6 && frames / 16000 <= 16, `duration ${path}`);
    assert.ok(data.subarray(44).some((byte) => byte !== 0), `not silent ${path}`);
  }
});

test("every migrated demo route resolves its sample and displays source, license and reader credits", () => {
  const pages = TRACKED.filter((p) => p.startsWith("models/") && p.endsWith("/index.html"));
  const migrated = [];
  for (const relative of pages) {
    const html = readFileSync(join(ROOT, relative), "utf8");
    // Exclude unrelated `translated.wav` and keep any historic test fixture prose out of route checks.
    assert.doesNotMatch(html, /(?<![a-z])(?:speech-)?ted\.wav|\bTED\b/, relative);
    if (!html.includes("librispeech.wav")) continue;
    migrated.push(relative);
    assert.equal((html.match(/data-audio-credit="librispeech"/g) ?? []).length, 1, relative);
    assert.ok(html.includes(`href="${SOURCE}"`), `source link: ${relative}`);
    assert.ok(html.includes(`href="${LICENSE}"`), `license link: ${relative}`);
    assert.ok(html.includes(`blob/main/${CREDITS}`), `reader/changes record: ${relative}`);
    for (const [, url] of html.matchAll(/["']([^"']*librispeech\.wav)["']/g)) {
      const path = url.startsWith("/web-ai-showcase/")
        ? join(ROOT, url.slice("/web-ai-showcase/".length))
        : resolve(ROOT, dirname(relative), url);
      assert.ok(path.startsWith(`${ROOT}/`), `outside repo: ${url} on ${relative}`);
      assert.ok(existsSync(path), `missing sample ${url} on ${relative}`);
    }
  }
  assert.equal(migrated.length, 46, "all previously shipped TED consumer pages remain covered");
  for (const family of ["longt5-summarization", "distilbart-summarization"]) {
    const page = join(ROOT, `models/${family}/multi-model/index.html`);
    assert.match(readFileSync(page, "utf8"), /\/models\/whisper-speech-to-text\/librispeech\.wav/);
  }
});
