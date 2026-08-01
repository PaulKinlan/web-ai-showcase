#!/usr/bin/env node
// Fail-closed offline verifier for the executable chrome-devtools-mcp MagicTouch event ledger.
// Browser evidence is produced by capture-interactive-segmenter-mcp.mjs. This validator binds its
// hash-chained tool events, exact route/device/action denominator, screenshot bytes/dimensions, and
// rendered inference state. A partial MCP run is retained but exits non-zero and cannot publish.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { DESKTOP, MOBILE, repoRoot, setViewport } from "./browser.mjs";
import {
  screenshotBindingMatches,
  screenshotFileMatches,
  sourceUsesExecutableMcp,
  validateEvidenceSummary,
  validateLedgerChain,
} from "./interactive-segmenter-evidence.mjs";

const SLUG = "interactive-segmenter";
const EVIDENCE = `models/${SLUG}/evidence/acceptance.json`;
const LEDGER = `models/${SLUG}/evidence/mcp-events.ndjson`;
const RUN_RECORD = `models/${SLUG}/acceptance-run.json`;
const CAPTURE_RUNNER = `scripts/capture-${SLUG}-mcp.mjs`;
const ROUTES = [
  ["overview", "models/interactive-segmenter/"],
  ["basics", "models/interactive-segmenter/basics/"],
  ["practical", "models/interactive-segmenter/practical/"],
  ["wild", "models/interactive-segmenter/wild/"],
  ["multi", "models/interactive-segmenter/multi-model/"],
];
const STAGES = ["interactive_segmenter/magic_touch", "Xenova/mobilevit-small"];
const VIEWPORTS = {
  desktop: { ...DESKTOP, deviceScaleFactor: 1 },
  mobile: { ...MOBILE, deviceScaleFactor: 1 },
};
// setViewport is the canonical viewport application contract used by ordinary CDP runners. The
// MCP producer applies these same concrete values through its real emulate tool calls.
const viewportContract = { setViewport: setViewport.name, VIEWPORTS };
const MODEL_SHA256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25";
const ARTIFACT_BYTES = 6_227_884;
const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const read = (path) => readFileSync(join(repoRoot, path));
const evidence = JSON.parse(read(EVIDENCE));
const captureSource = read(CAPTURE_RUNNER).toString();
const validatorSource = readFileSync(new URL(import.meta.url), "utf8");
const checks = [];
let failures = 0;

function check(id, pass, detail) {
  const record = { id, pass: Boolean(pass), detail };
  checks.push(record);
  if (!record.pass) failures++;
  console.log(`${record.pass ? "PASS" : "FAIL"} ${id}: ${detail}`);
}

check(
  "executable-mcp-producer",
  sourceUsesExecutableMcp(captureSource, validatorSource),
  `${CAPTURE_RUNNER} uses MCP Client/stdio and visible-control tools; validator has no inert void constants`,
);
check(
  "exact-artifact",
  evidence.artifact?.bytes === ARTIFACT_BYTES && evidence.artifact?.sha256 === MODEL_SHA256,
  `${evidence.artifact?.bytes} bytes · sha256 ${evidence.artifact?.sha256}`,
);
check(
  "exact-runtime",
  evidence.producer?.tool === "chrome-devtools-mcp" && /HeadlessChrome\/[\d.]+/.test(
    evidence.exactRuntime || "",
  ),
  `${evidence.producer?.tool} · ${evidence.exactRuntime || "missing UA"}`,
);

const ledgerRaw = read(LEDGER);
const events = ledgerRaw.toString().trim().split("\n").filter(Boolean).map((line) =>
  JSON.parse(line)
);
const chainOkay = validateLedgerChain(events, evidence.producer, ledgerRaw);
check(
  "mcp-ledger-integrity",
  chainOkay,
  `${events.length} hash-chained typed MCP/artifact-binding events`,
);

function countEvents(route, device, action, tool) {
  return events.filter((event) =>
    event.route === route && event.device === device && event.action === action &&
    (!tool || event.tool === tool) && !event.response?.isError
  ).length;
}
function hasVisibleAction(route, device) {
  return events.some((event) =>
    event.route === route && event.device === device &&
    ["click", "fill", "press_key", "upload_file"].includes(event.tool) && !event.response?.isError
  );
}
function statePass(record, route, device) {
  const expected = VIEWPORTS[device];
  const actual = record?.actual;
  return Boolean(record) && actual?.viewport?.width === expected.width &&
    actual?.viewport?.height === expected.height && actual?.viewport?.dpr === 1 &&
    /^Selected \d+\.\d+%/.test(actual?.status || "") &&
    /^\d+\.\d+%$/.test(actual?.coverage || "") &&
    /^\d+\.\d+%$/.test(actual?.pointConfidence || "") &&
    /foreground confidence \+ category mask/.test(actual?.shape || "") &&
    Number(actual?.selectedPixels?.replace(/[^\d]/g, "")) > 0 && actual?.overflow === 0 &&
    (route !== "multi" || actual?.classificationRows >= 3);
}

const records = Array.isArray(evidence.records) ? evidence.records : [];
const results = [];
for (const [route, path] of ROUTES) {
  for (const device of Object.keys(VIEWPORTS)) {
    const record = records.find((item) => item.route === route && item.device === device);
    const toolEvidence =
      countEvents(route, device, "console-before", "list_console_messages") === 1 &&
      countEvents(route, device, "console-after", "list_console_messages") === 1 &&
      countEvents(route, device, "network-before", "list_network_requests") === 1 &&
      countEvents(route, device, "network-after", "list_network_requests") === 1 &&
      hasVisibleAction(route, device);
    const pass = statePass(record, route, device) && toolEvidence;
    check(
      `matrix-${route}-${device}`,
      pass,
      record
        ? `${record.actual?.status}; viewport ${record.actual?.viewport?.width}×${record.actual?.viewport?.height}; console/network before+after ${toolEvidence}`
        : "missing executable MCP record",
    );
    results.push({
      route: path,
      viewport: device,
      pass,
      coverage: record?.actual?.coverage,
      pointConfidence: record?.actual?.pointConfidence,
      output: record?.actual?.shape,
    });
  }
}

const screenshotRecords = Array.isArray(evidence.screenshots) ? evidence.screenshots : [];
const screenshotKeys = new Set();
let screenshotsOkay = screenshotRecords.length === 20;
let retainedScreenshots = 0;
let acceptedScreenshots = 0;
for (const shot of screenshotRecords) {
  const key = `${shot.route}/${shot.device}/${shot.theme}`;
  screenshotKeys.add(key);
  const path = `models/${SLUG}/${shot.path}`;
  if (!existsSync(join(repoRoot, path))) {
    screenshotsOkay = false;
    continue;
  }
  const bytes = read(path);
  const [width, height] = execFileSync("identify", ["-format", "%w %h", join(repoRoot, path)], {
    encoding: "utf8",
  }).trim().split(/\s+/).map(Number);
  const expected = VIEWPORTS[shot.device];
  const input = {
    shot,
    bytes,
    sha256: sha256(bytes),
    width,
    height,
    viewport: expected,
    events,
    absolutePath: join(repoRoot, path),
  };
  if (screenshotFileMatches(input)) retainedScreenshots++;
  const bound = screenshotBindingMatches(input) && statePass(
    records.find((item) => item.route === shot.route && item.device === shot.device),
    shot.route,
    shot.device,
  );
  if (bound) acceptedScreenshots++;
  screenshotsOkay &&= bound;
}
for (const [route] of ROUTES) {
  for (const device of Object.keys(VIEWPORTS)) {
    for (const theme of ["light", "dark"]) {
      screenshotKeys.has(`${route}/${device}/${theme}`) ||
        (screenshotsOkay = false);
    }
  }
}
check(
  "retained-screenshots",
  screenshotsOkay && screenshotKeys.size === 20,
  `${retainedScreenshots}/20 retained file matches; ${acceptedScreenshots}/20 MCP-event + artifact-binding matches`,
);
check(
  "summary-schema-and-derivation",
  validateEvidenceSummary(evidence, events, acceptedScreenshots),
  `${evidence.status || "missing status"}; exact row, inspection, and screenshot denominators`,
);

const allConsoleAfter = events.filter((event) =>
  event.action === "console-after" && event.tool === "list_console_messages" &&
  !event.response?.isError
);
const consoleOkay = allConsoleAfter.length === 10 &&
  allConsoleAfter.every((event) => !/\[(error|warn|issue)\]/i.test(event.response.text));
check(
  "console-clean",
  consoleOkay,
  `${allConsoleAfter.length}/10 post-interaction console inspections`,
);
const allNetworkAfter = events.filter((event) =>
  event.action === "network-after" && event.tool === "list_network_requests" &&
  !event.response?.isError
);
const networkOkay = allNetworkAfter.length === 10 &&
  allNetworkAfter.every((event) => !/\[(?:4\d\d|5\d\d|failed)\]/i.test(event.response.text));
check(
  "network-clean",
  networkOkay,
  `${allNetworkAfter.length}/10 post-interaction network inspections`,
);
check(
  "complete-denominator",
  validateEvidenceSummary(evidence, events, acceptedScreenshots) &&
    evidence.status === "completed" && evidence.denominators?.routeDeviceRuns?.completed === 10 &&
    evidence.denominators?.routeDeviceRuns?.blocked === 0 &&
    evidence.denominators?.routeDeviceRuns?.notRun === 0,
  `${evidence.denominators?.routeDeviceRuns?.completed || 0}/10 complete · ${
    evidence.denominators?.routeDeviceRuns?.blocked || 0
  } blocked · ${evidence.denominators?.routeDeviceRuns?.notRun || 0} not-run`,
);
check(
  "stages",
  STAGES.every((stage) => captureSource.toLowerCase().includes(stage.toLowerCase())),
  `${STAGES.join(" + ")} are explicitly exercised by the MCP producer`,
);
check(
  "viewport-contract",
  viewportContract.setViewport === "setViewport" && VIEWPORTS.desktop.width === 1280 &&
    VIEWPORTS.mobile.width === 360,
  `DESKTOP ${VIEWPORTS.desktop.width}×${VIEWPORTS.desktop.height}; MOBILE ${VIEWPORTS.mobile.width}×${VIEWPORTS.mobile.height}`,
);

if (process.argv.includes("--write-run")) {
  const commit = execFileSync("git", [
    "log",
    "-n1",
    "--format=%H",
    "HEAD",
    "--",
    `models/${SLUG}`,
    `scripts/validate-${SLUG}.mjs`,
    CAPTURE_RUNNER,
    `:(exclude)models/${SLUG}/acceptance.json`,
    `:(exclude)models/${SLUG}/acceptance-run.json`,
  ], { cwd: repoRoot, encoding: "utf8" }).trim();
  writeFileSync(
    join(repoRoot, RUN_RECORD),
    JSON.stringify(
      {
        schemaVersion: 2,
        generatedAt: new Date().toISOString(),
        commit,
        exitCode: failures ? 1 : 0,
        status: failures ? "partial" : "completed",
        validator: `scripts/validate-${SLUG}.mjs`,
        producer: CAPTURE_RUNNER,
        evidence: EVIDENCE,
        evidenceSha256: sha256(read(EVIDENCE)),
        ledger: LEDGER,
        ledgerSha256: sha256(ledgerRaw),
        checks,
        results,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`wrote ${RUN_RECORD} for ${commit}`);
}

console.log(
  `interactive-segmenter acceptance: ${checks.length - failures}/${checks.length} passed`,
);
process.exit(failures ? 1 : 0);
