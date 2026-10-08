import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { matchingSound, parseSoundTargets } from "../models/ast-audio-classification/target-match.js";

const html = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)),
  "../models/ast-audio-classification/practical/index.html"), "utf8");
const defaultChip = html.match(/<button\b[^>]*data-match="([^"]+)"[^>]*aria-pressed="true"/);
const chip = (name) => {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const value = html.match(new RegExp(`data-match="(${escaped})"`));
  assert.ok(value, `${name} chip still exists`);
  return parseSoundTargets(value[1]);
};
const speech = [{ label:"Speech", score:0.903 }, { label:"Male speech, man speaking", score:0.009 }];

test("initially pressed AST Speech target passes real-shaped classifier labels", () => {
  assert.ok(defaultChip, "one native button exposes the default pressed target");
  assert.match(html, /let targets = parseSoundTargets\(\$\("targets"\)\.querySelector\('\[aria-pressed="true"\]'\)\.dataset\.match\)/,
    "initial selection uses the normalized path");
  assert.match(html, /const hit = matchingSound\(r\.labels, \+\$\("thresh"\)\.value, targets\)/,
    "live classification uses the same tested matcher");
  const initial = parseSoundTargets(defaultChip[1]);
  assert.deepEqual(initial, ["speech"]);
  assert.equal(matchingSound(speech, 0.12, initial)?.label, "Speech");
  assert.equal(matchingSound(speech, 0.95, initial), undefined,
    "sensitivity threshold still guards alerts");
});

test("chip changes and reselecting Speech use the same normalized target path", () => {
  assert.match(html, /targets = parseSoundTargets\(b\.dataset\.match\)/,
    "click handler must update targets through the same parser");
  const music = chip("Music");
  assert.deepEqual(music, ["music"]);
  assert.equal(matchingSound(speech, 0.12, music), undefined,
    "switching to Music must not alert on Speech");
  assert.equal(matchingSound([{label:"Music", score:0.84}], 0.12, music)?.label, "Music");
  const whistle = chip("Whistling,Whistle");
  assert.deepEqual(whistle, ["whistling", "whistle"]);
  assert.equal(matchingSound([{label:"Whistle", score:0.69}], 0.12, whistle)?.label, "Whistle");
  assert.equal(matchingSound(speech, 0.12, parseSoundTargets(defaultChip[1]))?.label, "Speech",
    "reselecting the default returns to a working Speech target");
});
