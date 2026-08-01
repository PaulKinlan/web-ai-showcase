#!/usr/bin/env node
// Fail-closed offline verifier for the executable chrome-devtools-mcp MagicTouch event ledger.
// Browser evidence is produced by capture-interactive-segmenter-mcp.mjs. This validator binds its
// fully typed hash chain, exact route/device/action denominator, screenshot artifacts, and rendered
// inference state. The exported pure function is the same acceptance path used by this CLI.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
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
const viewportContract = { setViewport: setViewport.name, VIEWPORTS };
const MODEL_SHA256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25";
const ARTIFACT_BYTES = 6_227_884;
const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const read = (path) => readFileSync(join(repoRoot, path));

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

export function validateInteractiveSegmenterEvidence({
  evidence,
  events,
  ledgerRaw,
  captureSource,
  validatorSource,
  loadScreenshot,
  logger = null,
}) {
  const checks = [];
  let failures = 0;
  const check = (id, pass, detail) => {
    const record = { id, pass: Boolean(pass), detail };
    checks.push(record);
    if (!record.pass) failures++;
    logger?.(`${record.pass ? "PASS" : "FAIL"} ${id}: ${detail}`);
  };
  const countEvents = (route, device, action, tool) =>
    events.filter((event) =>
      event.route === route && event.device === device && event.action === action &&
      (!tool || event.tool === tool) && !event.response?.isError
    ).length;
  const hasVisibleAction = (route, device) =>
    events.some((event) =>
      event.route === route && event.device === device &&
      ["click", "fill", "press_key", "upload_file"].includes(event.tool) &&
      !event.response?.isError
    );

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
    evidence.producer?.tool === "chrome-devtools-mcp" &&
      /HeadlessChrome\/[\d.]+/.test(evidence.exactRuntime || ""),
    `${evidence.producer?.tool} · ${evidence.exactRuntime || "missing UA"}`,
  );
  check(
    "mcp-ledger-integrity",
    validateLedgerChain(events, evidence.producer, ledgerRaw),
    `${events.length} hash-chained fully typed MCP/artifact-binding events`,
  );

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
    let file;
    try {
      file = loadScreenshot(shot);
    } catch {
      file = null;
    }
    if (!file) {
      screenshotsOkay = false;
      continue;
    }
    const expected = VIEWPORTS[shot.device];
    const input = {
      shot,
      bytes: file.bytes,
      sha256: sha256(file.bytes),
      width: file.width,
      height: file.height,
      viewport: expected,
      events,
      absolutePath: file.absolutePath,
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
        if (!screenshotKeys.has(`${route}/${device}/${theme}`)) screenshotsOkay = false;
      }
    }
  }
  check(
    "retained-screenshots",
    screenshotsOkay && screenshotKeys.size === 20,
    `${retainedScreenshots}/20 retained file matches; ${acceptedScreenshots}/20 MCP-event + artifact-binding matches`,
  );
  const summaryOkay = validateEvidenceSummary(evidence, events, acceptedScreenshots);
  check(
    "summary-schema-and-derivation",
    summaryOkay,
    `${
      evidence.status || "missing status"
    }; exact typed summary, row, inspection, and screenshot denominators`,
  );

  const allConsoleAfter = events.filter((event) =>
    event.action === "console-after" && event.tool === "list_console_messages" &&
    !event.response?.isError
  );
  check(
    "console-clean",
    allConsoleAfter.length === 10 &&
      allConsoleAfter.every((event) => !/\[(error|warn|issue)\]/i.test(event.response.text)),
    `${allConsoleAfter.length}/10 post-interaction console inspections`,
  );
  const allNetworkAfter = events.filter((event) =>
    event.action === "network-after" && event.tool === "list_network_requests" &&
    !event.response?.isError
  );
  check(
    "network-clean",
    allNetworkAfter.length === 10 &&
      allNetworkAfter.every((event) => !/\[(?:4\d\d|5\d\d|failed)\]/i.test(event.response.text)),
    `${allNetworkAfter.length}/10 post-interaction network inspections`,
  );
  check(
    "complete-denominator",
    summaryOkay && evidence.status === "completed" &&
      evidence.denominators?.routeDeviceRuns?.completed === 10 &&
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

  return { checks, failures, results, acceptedScreenshots, retainedScreenshots };
}

function loadRepositoryScreenshot(shot) {
  const path = join(repoRoot, `models/${SLUG}/${shot.path}`);
  if (!existsSync(path)) return null;
  const bytes = readFileSync(path);
  const [width, height] = execFileSync("identify", ["-format", "%w %h", path], {
    encoding: "utf8",
  }).trim().split(/\s+/).map(Number);
  return { bytes, width, height, absolutePath: path };
}

function runCli() {
  const evidence = JSON.parse(read(EVIDENCE));
  const ledgerRaw = read(LEDGER);
  const events = ledgerRaw.toString().trim().split("\n").filter(Boolean).map((line) =>
    JSON.parse(line)
  );
  const captureSource = read(CAPTURE_RUNNER).toString();
  const validatorSource = readFileSync(new URL(import.meta.url), "utf8");
  const outcome = validateInteractiveSegmenterEvidence({
    evidence,
    events,
    ledgerRaw,
    captureSource,
    validatorSource,
    loadScreenshot: loadRepositoryScreenshot,
    logger: console.log,
  });

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
          exitCode: outcome.failures ? 1 : 0,
          status: outcome.failures ? "partial" : "completed",
          validator: `scripts/validate-${SLUG}.mjs`,
          producer: CAPTURE_RUNNER,
          evidence: EVIDENCE,
          evidenceSha256: sha256(read(EVIDENCE)),
          ledger: LEDGER,
          ledgerSha256: sha256(ledgerRaw),
          checks: outcome.checks,
          results: outcome.results,
        },
        null,
        2,
      ) + "\n",
    );
    console.log(`wrote ${RUN_RECORD} for ${commit}`);
  }
  console.log(
    `interactive-segmenter acceptance: ${
      outcome.checks.length - outcome.failures
    }/${outcome.checks.length} passed`,
  );
  process.exitCode = outcome.failures ? 1 : 0;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) runCli();
