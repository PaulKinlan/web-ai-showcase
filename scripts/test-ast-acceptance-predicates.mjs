#!/usr/bin/env node
// Focused regression checks for the EXACT browser predicate functions: no Chrome is launched.
import assert from "node:assert/strict";
import { alertWaitTerminal, hasSpeechAlert, isFreshMultiTerminal } from "./ast-acceptance-predicates.mjs";

const tone = {
  runDisabled:false, status:'AST tagged this as Sine wave — not speech, so Whisper is skipped.',
  readoutVisible:true, scores:5, route:'tagged, no ASR', whisperVisible:false,
};
const jfkStarted = { ...tone, runDisabled:true, status:'AST is listening…', readoutVisible:false, scores:0 };
assert.equal(tone.route === 'tagged, no ASR', true, 'reproduces the old stale-route check');
assert.equal(isFreshMultiTerminal(jfkStarted), false, 'old tone route cannot pass during JFK inference');
assert.equal(isFreshMultiTerminal({ ...jfkStarted, runDisabled:false }), false,
  'old tone route cannot pass without new JFK scores/readout even if button re-enables');
assert.equal(isFreshMultiTerminal({ ...jfkStarted, scores:5, readoutVisible:true }), false,
  'stale state cannot pass while inference is still in progress');
assert.equal(isFreshMultiTerminal({ ...tone, status:'AST tagged this as Sine wave — not speech' }), true,
  'completed tone pass remains a terminal observable state');
assert.equal(isFreshMultiTerminal({ ...tone, route:'→ Whisper', whisperVisible:true,
  status:'AST heard speech — routing to Whisper to transcribe.' }), true,
  'completed JFK/Whisper pass is terminal');
assert.equal(isFreshMultiTerminal({ ...jfkStarted, runDisabled:false, status:'Pipeline failed: model error' }), true,
  'terminal pipeline errors stop the wait but never meet the downstream success assertion');

const initial = { hidden:true, text:'Heard it!', count:1, status:'Model ready.' };
assert.equal(alertWaitTerminal(initial), false, 'first non-matching mic window is not success');
assert.equal(alertWaitTerminal({ ...initial, count:3 }), false, 'waits for later real audio windows');
assert.equal(alertWaitTerminal({ ...initial, count:4 }), true, 'bounded after four completed windows');
assert.equal(hasSpeechAlert({ ...initial, count:4 }), false, 'four windows with no alert fail closed');
assert.equal(alertWaitTerminal({ ...initial, status:'Classify error: decoding failed' }), true,
  'a real page error stops waiting for immediate failure');
assert.equal(hasSpeechAlert({ ...initial, status:'Classify error: decoding failed' }), false);
assert.equal(alertWaitTerminal({ hidden:false, text:'Heard it — Speech (87%)', count:2, status:'Model ready.' }), true);
assert.equal(hasSpeechAlert({ hidden:false, text:'Heard it — Speech (87%)', count:2 }), true,
  'only a visible actual speech alert passes');
assert.equal(hasSpeechAlert({ hidden:false, text:'Heard it — Sine wave (87%)', count:2 }), false,
  'a non-speech alert does not satisfy the speech claim');
console.log('PASS AST acceptance predicates: stale multi route rejected; no Speech alert fails closed');
