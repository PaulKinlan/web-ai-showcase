#!/usr/bin/env node
import { strict as assert } from "node:assert";
import { basicFallback, evaluateCleanup } from "../models/wav2vec2-asr/multi-model/cleanup-quality.mjs";

const raw="AND SO MY FELLOW AMERICA AND NOT WHAT YOUR COUNTRY CAN DO FOR YOU AND WHAT YOU CAN DO FOR YOUR COUNTRY";
const observed="and so my fellow america and not what your country can do for you and what you can do for your country";
const red=evaluateCleanup(raw,observed);
assert.equal(red.valid,false,"actual W7 browser output must be RED");
assert.equal(red.sameWords,true);
assert.equal(red.sentenceCase,false);
assert.equal(red.terminalPunctuation,false);

const fixed="And so, my fellow America, and not what your country can do for you. And what you can do for your country.";
assert.deepEqual(evaluateCleanup(raw,fixed),{
  valid:true,sameWords:true,sentenceCase:true,terminalPunctuation:true,
},"genuinely punctuated/cased unchanged words must be GREEN");
for(const candidate of [
  "And so, my fellow Americans, ask not what your country can do for you.",
  "And so my fellow america and not what your country can do for you",
  "and so, my fellow america and not what your country can do for you.",
]) assert.equal(evaluateCleanup(raw,candidate).valid,false,`must reject ${candidate}`);
assert.equal(basicFallback(raw),
  "And so my fellow america and not what your country can do for you and what you can do for your country.");
assert.equal(evaluateCleanup(raw,basicFallback(raw)).valid,true,
  "a fallback can be legible; acceptance independently rejects it via its visible fallback flag");
assert.equal(basicFallback(""),"");
console.log("PASS wav2vec2 cleanup: actual red, true punctuation green, changed words red, fallback visibly separate");
