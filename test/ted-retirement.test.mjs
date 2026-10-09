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
const ORIGINAL_FLAC_SHA = new Map(Object.entries({
  "5895-34615-0010": "dc6759cf467a74ce415d70279b343ce24ef8bbb95f3cde44ac89ca20dfb1930b",
  "5694-64025-0020": "a29e98c98718e1c0ef1fc9ebb58658175d3666efab32be8b968ad337fddad84c",
  "251-118436-0000": "8a8a2f34930ea37432f1cc1082061cd889eceed949b3c13a79cbdc10a96f99e0",
  "1272-135031-0010": "cec411cc16334eae88f793cf4cc0c2c6f83087c06b4e11a451c329a33b26b884",
  "8842-302203-0005": "f3e0a032bc3e29600efe37c26edbaa5f134d55aafb8bbf3850ac893a938fdb72",
  "6295-244435-0009": "fd89d4d906579b2135ddcf6bdc53ec03923f7dffcc27c3795fad17a2dc06f264",
  "1919-142785-0022": "e8f1c17ac9dd8957e0181cd2fbc2d44442655587f2ff95d8839124f9992826e8",
  "6313-66129-0016": "0640055049684429839aa7109e12f660b6a0680b9d3283a69a6cc5317c06ee0c",
  "2412-153954-0006": "3feac5890500534685cde5eca5835e47d06bb2b7d0923b2ae3a544f2d68e30d1",
  "7976-110523-0006": "61d7839fa23dbe0febabe4d067ee25bbdadaeb8583de6ffdd8edbd9a5942ea25",
}));

// The immutable baseline is allowed to name the retired incident hash; the current tree is not.
test("no tracked media contains the retired TED bytes, nor any current ledger entry", () => {
  assert.ok(LEDGER.legacyBaseline.hashes.includes(TED_SHA), "historical baseline stays pinned");
  assert.ok(!LEDGER.entries.some((entry) => entry.hash === TED_SHA), "retired TED entry must be absent");
  for (const relative of TRACKED.filter(inScope)) {
    assert.notEqual(sha256(join(ROOT, relative)), TED_SHA, relative);
  }
  for (const relative of TRACKED.filter((p) => /^models\/.*\.(?:html|js|mjs|css)$/.test(p))) {
    assert.doesNotMatch(readFileSync(join(ROOT, relative), "utf8"), /(?<![a-z])(?:speech-)?ted\.wav/,
      `retired sample path remains active: ${relative}`);
  }
});

test("eleven replacement paths have ten declared, licensed and correctly hashed recordings", () => {
  const paths = TRACKED.filter((path) => /^models\/[^/]+\/(?:speech-)?librispeech\.wav$/.test(path));
  assert.equal(paths.length, 11);
  const entries = LEDGER.entries.filter((entry) => entry.paths.some((path) => paths.includes(path)));
  assert.equal(entries.length, 10);
  assert.equal(entries.reduce((n, entry) => n + entry.paths.filter((p) => paths.includes(p)).length, 0), 11);
  assert.equal(LEDGER.sourceCorpora.librispeechDevClean.archiveMd5, "42e2234ba48799c1f50f24a7926300a1");
  assert.equal(LEDGER.sourceCorpora.librispeechDevClean.archiveBytes, 337926286);
  const selection = LEDGER.sourceCorpora.librispeechDevClean.selection;
  assert.equal(selection.seed, "web-ai-showcase-z79-2026-10-09-openslr12-dev-clean-v1");
  assert.match(selection.algorithm, /archive's member order.*random\.Random\(seed\).*distinct speaker ID/i);
  assert.deepEqual(selection.selectedSpeakerDeduplicatedRanks, [1, 2, 5, 7, 10, 15, 19, 21, 24, 28]);
  assert.deepEqual(selection.selectedRawShuffledPositions, [1, 2, 5, 7, 10, 15, 20, 24, 28, 36]);
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
    assert.equal(entry.provenance.sourceFileHashes.length, relative.includes("speaker-diarization") ? 2 : 1);
    for (const { asset, sha256: originalSha } of entry.provenance.sourceFileHashes) {
      const id = asset.match(/\/(\d+-\d+-\d+)\.flac$/)?.[1];
      assert.equal(originalSha, ORIGINAL_FLAC_SHA.get(id), `original FLAC hash for ${asset}`);
      assert.ok(entry.provenance.sourceAsset.includes(asset), `declared source path ${asset}`);
      assert.ok(readFileSync(join(ROOT, CREDITS), "utf8").includes(originalSha), `credits contain ${originalSha}`);
    }
  }
  const shared = entries.find((entry) => entry.paths.includes("models/whisper-speech-to-text/librispeech.wav"));
  assert.deepEqual(shared.paths, ["models/wav2vec2-asr/librispeech.wav", "models/whisper-speech-to-text/librispeech.wav"]);
  const composite = entries.find((entry) => entry.paths.includes("models/speaker-diarization/librispeech.wav"));
  assert.match(composite.provenance.attribution, /0\.300 s digital silence/);
  assert.match(composite.provenance.creator, /iamartin.*Winston Tharp/);
  const segments = composite.provenance.sourceAlignedSegments;
  assert.deepEqual(segments.map(({ startSeconds, endSeconds }) => [startSeconds, endSeconds]), [[0, 7.77], [7.77, 8.07], [8.07, 15.42]]);
  assert.deepEqual(segments.map(({ speaker, sourceId }) => [speaker, sourceId]),
    [["iamartin", "5895-34615-0010"], [null, null], ["Winston Tharp", "5694-64025-0020"]]);
  assert.match(segments[0].referenceTranscript, /^ALL HIS EMOTIONS/);
  assert.equal(segments[1].referenceTranscript, "");
  assert.match(segments[2].referenceTranscript, /^MULE DID NOT DESIRE/);
  const pcm = readFileSync(join(ROOT, "models/speaker-diarization/librispeech.wav")).subarray(44);
  assert.equal(pcm.length, 15.42 * 16000 * 2);
  assert.ok(pcm.subarray(7.77 * 16000 * 2, 8.07 * 16000 * 2).every((byte) => byte === 0),
    "the disclosed 0.300s insert is exact digital silence");
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
