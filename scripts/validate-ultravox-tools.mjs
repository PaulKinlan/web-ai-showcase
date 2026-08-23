#!/usr/bin/env node
// Unit tests for the Ultravox tool layer (models/ultravox-audio-llm/tools.js).
//
// These are the parts that must be right no matter what the model says: the arithmetic evaluator
// (which parses untrusted model output and must NEVER reach eval), the unit table, and the
// function-call parser, which must accept Llama-3.2's canonical shapes and reject JSON the model is
// merely talking about.
//
// Pure Node, no browser, no network. Run: node scripts/validate-ultravox-tools.mjs

import {
  AUDIO_PLACEHOLDER,
  convert,
  evaluateExpression,
  MAX_NOTE_CHARS,
  parseToolCalls,
  runTool,
  stripToolCalls,
  SYSTEM_PROMPT,
  TOOL_NAMES,
  TOOL_SCHEMAS,
  toolMessageContent,
} from "../models/ultravox-audio-llm/tools.js";

let checks = 0;
let failed = 0;

function check(label, ok, detail = "") {
  checks++;
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
}

function near(label, actual, expected, tol = 1e-6) {
  check(label, Math.abs(actual - expected) <= tol, `got ${actual}, want ${expected}`);
}

function throws(label, fn) {
  try {
    const v = fn();
    check(label, false, `expected a throw, got ${JSON.stringify(v)}`);
  } catch {
    check(label, true);
  }
}

console.log("— arithmetic evaluator —");
near("18 * 7", evaluateExpression("18 * 7"), 126);
near("(120 + 45) / 3", evaluateExpression("(120 + 45) / 3"), 55);
near("precedence 2 + 3 * 4", evaluateExpression("2 + 3 * 4"), 14);
near("right-assoc 2^3^2", evaluateExpression("2^3^2"), 512);
near("unary minus", evaluateExpression("-5 + 2"), -3);
near("sqrt(144)", evaluateExpression("sqrt(144)"), 12);
near("max(3, 9, 4)", evaluateExpression("max(3, 9, 4)"), 9);
near("15% of 80 as 80 * 15 / 100", evaluateExpression("80 * 15 / 100"), 12);
near("modulo", evaluateExpression("17 % 5"), 2);
near("unicode ×", evaluateExpression("6 × 7"), 42);
near("pi constant", evaluateExpression("pi"), Math.PI);
throws("rejects eval-style code", () => evaluateExpression("globalThis.alert(1)"));
throws("rejects property access", () => evaluateExpression("(1).constructor"));
throws("rejects a bare identifier", () => evaluateExpression("process"));
throws("rejects unbalanced parens", () => evaluateExpression("(1 + 2"));
throws("rejects trailing junk", () => evaluateExpression("1 + 2 foo"));
throws("rejects division by zero", () => evaluateExpression("1/0"));
throws("rejects an empty expression", () => evaluateExpression("   "));
throws("rejects an over-long expression", () => evaluateExpression("1+".repeat(200) + "1"));

// Regression (PR #3 Codex round 3): a leading sign must NOT bind tighter than exponentiation.
// -2^2 is -(2^2) = -4 everywhere except a broken parser, and the agent would have read out "4".
near("-2^2 is -(2^2)", evaluateExpression("-2^2"), -4);
near("-3^2+1", evaluateExpression("-3^2+1"), -8);
near("a negative exponent still works", evaluateExpression("2^-2"), 0.25);
near("parentheses still force a negative base", evaluateExpression("(-2)^2"), 4);
near("exponentiation stays right-associative", evaluateExpression("2^3^2"), 512);
near("unary minus still binds to multiplication", evaluateExpression("-2*3"), -6);

console.log("— unit conversion —");
near("100 km → miles", convert(100, "km", "miles").value, 62.137119, 1e-5);
near("5 kg → lb", convert(5, "kg", "lb").value, 11.023113, 1e-5);
near("20 celsius → fahrenheit", convert(20, "celsius", "fahrenheit").value, 68);
near("degrees prefix tolerated", convert(20, "degrees celsius", "fahrenheit").value, 68);
near("0 c → kelvin", convert(0, "c", "kelvin").value, 273.15);
near("60 mph → kph", convert(60, "mph", "kph").value, 96.56064, 1e-4);
near("2 litres → pints", convert(2, "litres", "pints").value, 4.226753, 1e-5);
check("dimension is reported", convert(1, "m", "ft").dimension === "length");
throws("rejects a cross-dimension convert", () => convert(1, "km", "kg"));
throws("rejects an unknown unit", () => convert(1, "furlong", "m"));
throws("rejects a non-number value", () => convert("banana", "m", "ft"));

// Regression (PR #3 Codex review round 2): "pint" and "gallon" mean different volumes in the US and
// the UK. Bare names resolve to US, but the answer must NAME the system so it can never quietly
// answer in the wrong one, and explicit imperial names must be honoured.
near("2 litres → pints defaults to US", convert(2, "litres", "pints").value, 4.226753, 1e-5);
check("a bare pint names its system", convert(2, "litres", "pints").system === "US");
check("a bare pint is labelled in the result", convert(2, "litres", "pints").to === "US pints", convert(2, "litres", "pints").to);
near("imperial pints are a different number", convert(2, "litres", "imperial pints").value, 3.519508, 1e-5);
near('"uk pints" is the same as imperial', convert(2, "litres", "uk pints").value, 3.519508, 1e-5);
near("hyphenated us-gallon still parses", convert(1, "us-gallon", "litres").value, 3.785412, 1e-5);
near("imperial gallons differ from US", convert(1, "imperial gallons", "litres").value, 4.54609, 1e-5);
check(
  "an imperial answer is labelled imperial",
  convert(2, "litres", "imperial pints").system === "imperial",
);
check("non-volume conversions carry no system label", convert(1, "km", "miles").system === undefined);
const pintOut = runTool({ name: "convert_units", arguments: { value: 2, from: "litres", to: "pints" } }, {});
check("the displayed conversion names the system", /US pints/.test(pintOut.display), pintOut.display);

console.log("— tool-call parsing —");
const canonical = '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}\n</tool_call>';
check("canonical <tool_call>", JSON.stringify(parseToolCalls(canonical)) ===
  JSON.stringify([{ name: "get_time", arguments: { timezone: "Asia/Tokyo" } }]));
check(
  "unterminated <tool_call> (truncated generation)",
  parseToolCalls('<tool_call>\n{"name": "list_notes", "arguments": {}}').length === 1,
);
// Superseded by the canonical-channel rule below: a fence introduced by prose is the model TALKING
// about a call, not making one, so it must no longer execute. A fence that is the whole message
// still does — see "a fence containing only the call still counts".
check(
  "a fence introduced by prose does NOT execute",
  parseToolCalls('Sure!\n```json\n{"name":"calculate","arguments":{"expression":"2+2"}}\n```').length === 0,
);
check(
  "bare object",
  parseToolCalls('{"name": "add_note", "arguments": {"text": "milk"}}')[0].name === "add_note",
);
check(
  "stringified arguments",
  parseToolCalls('<tool_call>{"name":"calculate","arguments":"{\\"expression\\":\\"3*3\\"}"}</tool_call>')[0]
    .arguments.expression === "3*3",
);
check(
  "nested braces in arguments survive",
  parseToolCalls('<tool_call>{"name":"add_note","arguments":{"text":"use {braces} here"}}</tool_call>')[0]
    .arguments.text === "use {braces} here",
);
check("unknown tool names are ignored", parseToolCalls('{"name":"rm_rf","arguments":{}}').length === 0);
check("plain prose yields no calls", parseToolCalls("The capital of France is Paris.").length === 0);
check("duplicate identical calls collapse", parseToolCalls(canonical + canonical).length === 1);
check(
  "two distinct calls both parse",
  parseToolCalls(
    '<tool_call>{"name":"calculate","arguments":{"expression":"1+1"}}</tool_call>' +
      '<tool_call>{"name":"list_notes","arguments":{}}</tool_call>',
  ).length === 2,
);
check(
  "stripToolCalls leaves the prose",
  stripToolCalls("Let me check.\n" + canonical) === "Let me check.",
);

// Llama-3.2 (Ultravox's backbone) does not use Qwen's <tool_call> wrapper — it emits a bare object,
// often behind a <|python_tag|> marker, and names the arguments "parameters".
check(
  "llama bare {name, parameters}",
  parseToolCalls('{"name": "get_time", "parameters": {"timezone": "Asia/Tokyo"}}')[0]?.arguments.timezone === "Asia/Tokyo",
);
check(
  "llama <|python_tag|> prefix",
  parseToolCalls('<|python_tag|>{"name": "calculate", "parameters": {"expression": "2+2"}}')[0]?.name === "calculate",
);
check(
  "python_tag with trailing eom",
  parseToolCalls('<|python_tag|>{"name":"list_notes","parameters":{}}<|eom_id|>')[0]?.name === "list_notes",
);
// Regression (PR #3 Codex round 3): a call must BE the message, not something the message quotes.
// Asked to "show the JSON but don't run it", a model produces exactly that JSON inside a sentence;
// executing it would perform an action the user explicitly declined.
check(
  "JSON quoted inside prose does NOT execute",
  parseToolCalls('The JSON would be {"name": "start_timer", "parameters": {"seconds": 300}} but do not run it.').length === 0,
);
check(
  "a trailing explanation after a call does NOT execute",
  parseToolCalls('{"name": "list_notes", "parameters": {}} — that is what I would send.').length === 0,
);
check(
  "an explicit <tool_call> wrapper still counts anywhere",
  parseToolCalls('Here you go: <tool_call>{"name":"list_notes","arguments":{}}</tool_call>')[0]?.name === "list_notes",
);
check(
  "a bare object alone still counts",
  parseToolCalls('  {"name": "list_notes", "parameters": {}}  ')[0]?.name === "list_notes",
);
check(
  "a fence containing only the call still counts",
  parseToolCalls('```json\n{"name":"list_notes","parameters":{}}\n```')[0]?.name === "list_notes",
);
check(
  "stripToolCalls removes a bare llama call",
  stripToolCalls('{"name": "list_notes", "parameters": {}}') === "",
  JSON.stringify(stripToolCalls('{"name": "list_notes", "parameters": {}}')),
);
check(
  "stripToolCalls removes the python_tag marker",
  !/python_tag/.test(stripToolCalls('<|python_tag|>{"name":"list_notes","parameters":{}}')),
);
check("the audio placeholder is the one the processor expects", AUDIO_PLACEHOLDER === "<|audio|>", AUDIO_PLACEHOLDER);
check(
  "the system prompt does not claim a transcript",
  !/transcript/i.test(SYSTEM_PROMPT),
  SYSTEM_PROMPT,
);

// Regression (PR #3 Codex round 4): unparseable arguments were silently replaced with {}, which
// CHANGES the request — get_time with a mangled timezone became a successful LOCAL time lookup and
// the model would then state the wrong city's time with full confidence.
const malformed = parseToolCalls('{"name":"get_time","arguments":"{not json"}')[0];
check("a malformed arguments string is flagged, not emptied", !!malformed?.argsError, JSON.stringify(malformed));
const malformedRun = runTool(malformed, {});
check("a call with unreadable arguments is refused", malformedRun.ok === false, malformedRun.display ?? malformedRun.error);
check(
  "the refusal tells the model not to guess",
  /do not guess/i.test(malformedRun.error),
  malformedRun.error,
);
check(
  "a well-formed stringified arguments object still runs",
  runTool(parseToolCalls('<tool_call>{"name":"calculate","arguments":"{\\"expression\\":\\"3*3\\"}"}</tool_call>')[0], {}).result?.value === 9,
);

// Regression (PR #3 Codex round 4): the schema invited percentage questions the evaluator cannot
// parse. Neither natural form works, so the description must tell the model to use division.
throws("a percent sign in a percentage does not parse", () => evaluateExpression("15% * 80"));
throws('"15% of 80" does not parse', () => evaluateExpression("15% of 80"));
near("the documented form does parse", evaluateExpression("80 * 15 / 100"), 12);
near("a bare % is still remainder", evaluateExpression("17 % 5"), 2);
const calcDesc = TOOL_SCHEMAS.find((t) => t.function.name === "calculate").function.description;
check("the calculate schema documents the division form", /80 \* 15 \/ 100/.test(calcDesc), calcDesc);
check("the calculate schema warns off the percent sign", /no percent sign/i.test(calcDesc), calcDesc);

console.log("— executors —");
const notes = [];
const ctx = { notes, now: "2026-08-23T12:00:00Z", startTimer: () => "t1" };
const timeOut = runTool({ name: "get_time", arguments: { timezone: "Asia/Tokyo" } }, ctx);
check("get_time succeeds", timeOut.ok, timeOut.display);
check("get_time is really Tokyo (21:00 on 23 Aug 2026)", /21:00/.test(timeOut.display), timeOut.display);
check("get_time rejects a bogus zone", !runTool({ name: "get_time", arguments: { timezone: "Mars/Olympus" } }, ctx).ok);
const calcOut = runTool({ name: "calculate", arguments: { expression: "12 * 12" } }, ctx);
check("calculate result", calcOut.ok && calcOut.result.value === 144, calcOut.display);
check("calculate failure is captured, not thrown", runTool({ name: "calculate", arguments: { expression: "oops" } }, ctx).ok === false);
const noteOut = runTool({ name: "add_note", arguments: { text: "buy milk" } }, ctx);
check("add_note stores the note", noteOut.ok && notes.length === 1 && notes[0] === "buy milk");
check("list_notes reads it back", runTool({ name: "list_notes", arguments: {} }, ctx).result.count === 1);
const timerOut = runTool({ name: "start_timer", arguments: { seconds: 90, label: "pasta" } }, ctx);
check("start_timer succeeds", timerOut.ok && timerOut.result.seconds === 90, timerOut.display);
check("start_timer rejects zero", !runTool({ name: "start_timer", arguments: { seconds: 0 } }, ctx).ok);
check("start_timer caps at an hour", !runTool({ name: "start_timer", arguments: { seconds: 99999 } }, ctx).ok);
check("unknown tool is reported, not thrown", runTool({ name: "nope", arguments: {} }, ctx).ok === false);
check("tool message content is JSON", JSON.parse(toolMessageContent(calcOut)).value === 144);
check("error outcomes serialise an error field", "error" in JSON.parse(toolMessageContent(runTool({ name: "nope" }, ctx))));
check("six tools published", TOOL_NAMES.length === 6, TOOL_NAMES.join(","));

// Regression (PR #3 review): a note longer than the cap was stored truncated but reported back to
// the model in full, so a follow-up list_notes contradicted the confirmation the model had just
// given. What is stored and what is reported must be the same string.
const longNotes = [];
const longCtx = { notes: longNotes };
const long = "x".repeat(MAX_NOTE_CHARS + 50);
const longOut = runTool({ name: "add_note", arguments: { text: long } }, longCtx);
check("an over-long note is stored truncated", longNotes[0].length === MAX_NOTE_CHARS, String(longNotes[0].length));
check(
  "the truncated note is what the model is told",
  longOut.result.note === longNotes[0],
  `reported ${longOut.result.note.length} chars, stored ${longNotes[0].length}`,
);
check("truncation is flagged to the model", longOut.result.truncated === true);
check(
  "list_notes agrees with what add_note reported",
  runTool({ name: "list_notes", arguments: {} }, longCtx).result.notes[0] === longOut.result.note,
);
check("a short note is not marked truncated", runTool({ name: "add_note", arguments: { text: "milk" } }, longCtx).result.truncated === false);

// Regression (PR #3 review): the schema described a ring the demo never plays. Tool descriptions are
// repeated to the user by the model, so a false one becomes a false promise.
const timerDesc = TOOL_SCHEMAS.find((t) => t.function.name === "start_timer").function.description;
check("start_timer does not promise a sound it never makes", !/\bring/i.test(timerDesc), timerDesc);
check("start_timer says what it actually does", /counts down|makes no sound/i.test(timerDesc), timerDesc);

console.log(`\n${checks - failed}/${checks} checks passed`);
process.exit(failed ? 1 : 0);
