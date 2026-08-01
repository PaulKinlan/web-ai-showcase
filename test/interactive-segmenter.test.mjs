import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { composeMask, summarizeConfidence } from "../models/interactive-segmenter/mask-math.js";
import { createHash } from "node:crypto";
import { writeAtomicCaptureOutputs } from "../scripts/interactive-segmenter-capture-summary.mjs";
import {
  completeEvidencePasses,
  eventHash,
  expectedRouteDeviceRows,
  screenshotBindingMatches,
  sourceUsesExecutableMcp,
  validateEvidenceSummary,
  validateLedgerChain,
} from "../scripts/interactive-segmenter-evidence.mjs";
import { validateInteractiveSegmenterEvidence } from "../scripts/validate-interactive-segmenter.mjs";
import { perfectInteractiveSegmenterEvidence } from "./fixtures/interactive-segmenter-evidence.mjs";

const root = new URL("../", import.meta.url);
const read = (path) => readFileSync(new URL(path, root), "utf8");

function catalogue() {
  return JSON.parse(read("models.json")).models;
}

test("catalogue integrates exactly one verified pending family without changing denominator", () => {
  const models = catalogue();
  assert.equal(models.length, 2679);
  assert.deepEqual(
    Object.fromEntries(
      ["built", "pending", "blocked"].map((status) => [
        status,
        models.filter((model) => model.status === status).length,
      ]),
    ),
    { built: 314, pending: 2302, blocked: 63 },
  );
  const model = models.find((entry) => entry.slug === "interactive-segmenter");
  assert.equal(model.status, "built");
  assert.equal(model.hfId, "mediapipe/interactive-segmenter");
  assert.equal(model.task, "mask-generation");
  assert.equal(model.license, "apache-2.0");
});

test("five stable portfolio routes call the genuine shared demo", () => {
  for (
    const route of [
      "index.html",
      "basics/index.html",
      "practical/index.html",
      "wild/index.html",
      "multi-model/index.html",
    ]
  ) {
    const html = read(`models/interactive-segmenter/${route}`);
    assert.match(html, /mountInteractiveDemo/);
    assert.match(html, /public\/styles\.css/);
    assert.match(html, /<main>/);
  }
  assert.match(
    read("models/interactive-segmenter/multi-model/index.html"),
    /MagicTouch[\s\S]*MobileViT/,
  );
});

test("artifact pin and worker integrity constant match", () => {
  const artifact = JSON.parse(read("models/interactive-segmenter/artifact.json"));
  const worker = read("models/interactive-segmenter/worker.js");
  assert.equal(artifact.artifact.bytes, 6_227_884);
  assert.equal(artifact.artifact.sha256.length, 64);
  assert.match(worker, new RegExp(artifact.artifact.sha256));
  assert.match(worker, /crypto\.subtle\.digest\("SHA-256"/);
  assert.match(worker, /modelAssetBuffer/);
  assert.match(worker, /InteractiveSegmenter\.createFromOptions/);
});

test("published API example uses the task API foreground mask at index zero", () => {
  const html = read("models/interactive-segmenter/index.html");
  assert.match(html, /result\.confidenceMasks\[0\]\.getAsFloat32Array\(\)/);
  assert.doesNotMatch(html, /result\.confidenceMasks\[1\]/);
  assert.match(html, /raw MagicTouch model tensor has background and foreground channels/i);
  assert.match(html, /task API[^<]*exposes one normalized foreground confidence mask/i);
});

test("MagicTouch download is honestly site-controlled and restart-only", () => {
  const route = JSON.parse(read("download-routes.json")).routes.find((entry) =>
    entry.slug === "interactive-segmenter"
  );
  assert.equal(route.family, "mediapipe");
  assert.equal(route.byteControl, "site-controlled");
  assert.equal(route.resume, "restart-only");
  assert.match(route.note, /interruption restarts from byte zero/);
  assert.doesNotMatch(route.note, /runtime owns|runtime-owned/i);
  const generator = read("scripts/route-download-inventory.mjs");
  assert.match(generator, /family === "mediapipe"[\s\S]*modelAssetBuffer/);
});

test("MCP evidence helpers reject inert validators and wide mobile screenshots", () => {
  const capture = read("scripts/capture-interactive-segmenter-mcp.mjs");
  const validator = read("scripts/validate-interactive-segmenter.mjs");
  assert.equal(sourceUsesExecutableMcp(capture, validator), true);
  assert.equal(sourceUsesExecutableMcp(capture, `${validator}\nvoid [DESKTOP, MOBILE];`), false);
  assert.match(capture, /execFileSync\("identify"/);
  assert.match(capture, /writeAtomicCaptureOutputs/);
  const bytes = new Uint8Array([1, 2, 3]);
  const shot = {
    bytes: 3,
    sha256: "abc",
    image: { width: 1905, height: 740 },
    expectedViewport: { width: 360, height: 740, dpr: 1 },
  };
  assert.equal(
    screenshotBindingMatches({
      shot,
      bytes,
      sha256: "abc",
      width: 1905,
      height: 740,
      viewport: { width: 360, height: 740 },
    }),
    false,
  );
});

test("perfect 10/10 and 20/20 fixture requires typed chained screenshot provenance", () => {
  const fixture = perfectInteractiveSegmenterEvidence();
  assert.equal(
    validateLedgerChain(fixture.events, fixture.evidence.producer, fixture.ledgerRaw),
    true,
  );
  assert.equal(validateEvidenceSummary(fixture.evidence, fixture.events, 20), true);
  assert.equal(completeEvidencePasses(fixture.evidence, fixture.events, 20), true);
  for (const input of fixture.screenshotInputs) {
    assert.equal(screenshotBindingMatches(input), true);
  }

  const unknownEventField = structuredClone(fixture.events);
  unknownEventField[0].unexpected = true;
  assert.equal(validateLedgerChain(unknownEventField, fixture.evidence.producer), false);
  const unknownResponseField = structuredClone(fixture.events);
  unknownResponseField[0].response.unexpected = true;
  assert.equal(validateLedgerChain(unknownResponseField, fixture.evidence.producer), false);
  const unknownSummaryField = structuredClone(fixture.evidence);
  unknownSummaryField.unexpected = true;
  assert.equal(validateEvidenceSummary(unknownSummaryField, fixture.events, 20), false);
  const unknownProducerField = structuredClone(fixture.evidence);
  unknownProducerField.producer.unexpected = true;
  assert.equal(validateEvidenceSummary(unknownProducerField, fixture.events, 20), false);

  const digestTamper = structuredClone(fixture.screenshotInputs[0]);
  digestTamper.shot.sha256 = "0".repeat(64);
  assert.equal(screenshotBindingMatches(digestTamper), false);

  const sameSizeReplacement = structuredClone(fixture.screenshotInputs[0]);
  sameSizeReplacement.bytes = Buffer.alloc(sameSizeReplacement.bytes.length, 120);
  sameSizeReplacement.sha256 = "1".repeat(64);
  assert.equal(screenshotBindingMatches(sameSizeReplacement), false);

  const dimensionTamper = structuredClone(fixture.screenshotInputs[0]);
  dimensionTamper.shot.image.height++;
  assert.equal(screenshotBindingMatches(dimensionTamper), false);

  const missingBinding = {
    ...fixture.screenshotInputs[0],
    events: fixture.events.filter((event) =>
      event.request?.screenshotEventHash !== fixture.events[5].hash
    ),
  };
  assert.equal(screenshotBindingMatches(missingBinding), false);

  const partial = structuredClone(fixture.evidence);
  partial.status = "partial";
  partial.blocker = {
    route: "multi",
    device: "mobile",
    action: "fixture-failure",
    tool: "click",
    code: "fixture-blocker",
    detail: "synthetic partial fixture",
    recoverable: true,
    retryDisposition: "capture again",
  };
  partial.denominators.routeDeviceRuns.rows[9].status = "blocked";
  partial.denominators.routeDeviceRuns.completed = 9;
  partial.denominators.routeDeviceRuns.blocked = 1;
  // A status-only demotion with all ten completed records is internally inconsistent.
  assert.equal(validateEvidenceSummary(partial, fixture.events, 20), false);
  assert.equal(completeEvidencePasses(partial, fixture.events, 20), false);
});

function fixtureValidatorInput(fixture, loadScreenshot) {
  return {
    evidence: fixture.evidence,
    events: fixture.events,
    ledgerRaw: fixture.ledgerRaw,
    captureSource: read("scripts/capture-interactive-segmenter-mcp.mjs"),
    validatorSource: read("scripts/validate-interactive-segmenter.mjs"),
    loadScreenshot: loadScreenshot || ((shot) => {
      const input = fixture.screenshotInputs.find((item) => item.shot.path === shot.path);
      return input && {
        bytes: input.bytes,
        width: input.width,
        height: input.height,
        absolutePath: input.absolutePath,
      };
    }),
  };
}

test("importable portfolio validator accepts synthetic executable 10/10 and 20/20 evidence", () => {
  const fixture = perfectInteractiveSegmenterEvidence();
  const outcome = validateInteractiveSegmenterEvidence(fixtureValidatorInput(fixture));
  assert.equal(outcome.checks.length, 21);
  assert.equal(outcome.failures, 0);
  assert.equal(outcome.acceptedScreenshots, 20);
});

test("portfolio validator rejects an eleventh duplicate route/device record", () => {
  const fixture = perfectInteractiveSegmenterEvidence();
  fixture.evidence.records.push(structuredClone(fixture.evidence.records[0]));
  assert.equal(validateEvidenceSummary(fixture.evidence, fixture.events, 20), false);
  const outcome = validateInteractiveSegmenterEvidence(fixtureValidatorInput(fixture));
  assert.ok(outcome.failures > 0);
});

test("portfolio validator rejects tampered, missing, and partial perfect fixtures", () => {
  const tampered = perfectInteractiveSegmenterEvidence();
  tampered.evidence.screenshots[0].sha256 = "0".repeat(64);
  assert.ok(validateInteractiveSegmenterEvidence(fixtureValidatorInput(tampered)).failures > 0);

  const missing = perfectInteractiveSegmenterEvidence();
  const missingPath = missing.evidence.screenshots[0].path;
  const missingInput = fixtureValidatorInput(missing, (shot) => {
    if (shot.path === missingPath) return null;
    const input = missing.screenshotInputs.find((item) => item.shot.path === shot.path);
    return {
      bytes: input.bytes,
      width: input.width,
      height: input.height,
      absolutePath: input.absolutePath,
    };
  });
  assert.ok(validateInteractiveSegmenterEvidence(missingInput).failures > 0);

  const partial = perfectInteractiveSegmenterEvidence();
  partial.evidence.status = "partial";
  partial.evidence.blocker = {
    route: "multi",
    device: "mobile",
    action: "fixture-failure",
    tool: "click",
    code: "fixture-partial",
    detail: "synthetic partial fixture",
    recoverable: true,
    retryDisposition: "capture again",
  };
  partial.evidence.denominators.routeDeviceRuns.rows[9].status = "blocked";
  partial.evidence.denominators.routeDeviceRuns.completed = 9;
  partial.evidence.denominators.routeDeviceRuns.blocked = 1;
  assert.ok(validateInteractiveSegmenterEvidence(fixtureValidatorInput(partial)).failures > 0);
});

test("re-chained malformed typed MCP event is rejected after refreshed producer digests", () => {
  const fixture = perfectInteractiveSegmenterEvidence();
  const events = structuredClone(fixture.events);
  delete events[0].startedAt;
  delete events[0].endedAt;
  events[0].route = 42;
  events[0].device = { kind: "desktop" };
  delete events[0].response.text;
  let previousHash = "0".repeat(64);
  for (const [index, event] of events.entries()) {
    event.sequence = index + 1;
    event.previousHash = previousHash;
    event.hash = eventHash(event);
    previousHash = event.hash;
  }
  const ledgerRaw = events.map((event) => JSON.stringify(event)).join("\n") + "\n";
  const evidence = structuredClone(fixture.evidence);
  evidence.producer.eventCount = events.length;
  evidence.producer.finalEventHash = previousHash;
  evidence.producer.ledgerSha256 = createHash("sha256").update(ledgerRaw).digest("hex");
  assert.equal(validateLedgerChain(events, evidence.producer, ledgerRaw), false);
  assert.equal(validateEvidenceSummary(evidence, events, 20), false);
  const outcome = validateInteractiveSegmenterEvidence({
    ...fixtureValidatorInput(fixture),
    evidence,
    events,
    ledgerRaw,
  });
  assert.equal(outcome.checks.find((check) => check.id === "mcp-ledger-integrity").pass, false);
  assert.ok(outcome.failures > 0);
});

function atomicFailureCase(
  kind,
  completed,
  blocked,
  ledgerLength,
  screenshotCount = completed * 2,
) {
  const directory = mkdtempSync(join(tmpdir(), `magic-touch-${kind}-`));
  try {
    const fixture = perfectInteractiveSegmenterEvidence();
    const ledger = fixture.events.slice(0, ledgerLength);
    const rows = expectedRouteDeviceRows();
    for (let index = 0; index < completed; index++) rows[index].status = "completed";
    for (let index = completed; index < completed + blocked; index++) {
      rows[index].status = "blocked";
    }
    const ledgerPath = join(directory, "mcp-events.ndjson");
    const summaryPath = join(directory, "acceptance.json");
    const summary = writeAtomicCaptureOutputs({
      ledgerPath,
      summaryPath,
      ledger,
      routeDeviceRows: rows,
      records: fixture.evidence.records.slice(0, completed),
      screenshots: fixture.evidence.screenshots.slice(0, screenshotCount),
      blocker: {
        route: completed + blocked ? rows[Math.max(0, completed + blocked - 1)].route : null,
        device: completed + blocked ? rows[Math.max(0, completed + blocked - 1)].device : null,
        action: `${kind}-failure`,
        tool: kind === "close" ? "producer" : "fixture-tool",
        code: `fixture-${kind}-exception`,
        detail: `${kind} failed`,
        recoverable: true,
        retryDisposition: "requires a fresh capture",
      },
      generatedAt: "2026-08-02T00:00:02.000Z",
    });
    assert.notEqual(summary.status, "completed");
    assert.equal(summary.denominators.routeDeviceRuns.completed, completed);
    assert.equal(summary.denominators.routeDeviceRuns.blocked, blocked);
    assert.ok(summary.blocker);
    assert.equal(
      readFileSync(ledgerPath, "utf8"),
      ledger.map((event) => JSON.stringify(event)).join("\n") + (ledger.length ? "\n" : ""),
    );
    assert.deepEqual(JSON.parse(readFileSync(summaryPath, "utf8")), summary);
    assert.equal(readdirSync(directory).some((name) => name.includes(".tmp-")), false);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test("setup failure atomically writes blocked summary and retains its ledger", () => {
  atomicFailureCase("setup", 0, 0, 0);
});

test("tool failure atomically writes blocked summary and retains its ledger", () => {
  atomicFailureCase("tool", 0, 1, 1);
});

test("screenshot failure atomically writes partial summary and retains its ledger", () => {
  atomicFailureCase("screenshot", 1, 1, 9);
});

test("close failure atomically demotes completion and retains its ledger", () => {
  atomicFailureCase("close", 9, 1, 90, 20);
});

test("inference and dense compositing remain worker-first with deterministic disposal", () => {
  const worker = read("models/interactive-segmenter/worker.js");
  const client = read("models/interactive-segmenter/demo.js");
  assert.match(worker, /segmenter\.segment\(bitmap/);
  assert.match(worker, /composeMask/);
  assert.match(worker, /transfer: \[rgba\.buffer\]/);
  assert.match(worker, /segmenter\?\.close/);
  assert.match(client, /WorkerClient/);
  assert.match(client, /pagehide/);
  assert.match(client, /URL\.revokeObjectURL/);
});

test("confidence summaries discriminate target, area and bounds", () => {
  const confidence = new Float32Array([
    0.05,
    0.1,
    0.2,
    0.1,
    0.9,
    0.8,
    0.05,
    0.7,
    0.65,
  ]);
  const summary = summarizeConfidence(confidence, 0.5, { x: 0.5, y: 0.5 }, 3, 3);
  assert.equal(summary.selectedPixels, 4);
  assert.equal(summary.coverage, 4 / 9);
  assert.ok(summary.targetConfidence > 0.89);
  assert.deepEqual(summary.bbox, { minX: 1, minY: 1, maxX: 2, maxY: 2 });
});

test("compositor makes distinct overlay, mask, cutout, and spotlight outputs", () => {
  const source = new Uint8ClampedArray([
    200,
    100,
    50,
    255,
    20,
    40,
    80,
    255,
  ]);
  const confidence = new Float32Array([0.9, 0.1]);
  const outputs = ["overlay", "mask", "cutout", "spotlight"].map((mode) =>
    Array.from(composeMask(source, confidence, { mode, threshold: 0.5, opacity: 0.6 }))
  );
  assert.equal(new Set(outputs.map(JSON.stringify)).size, 4);
  assert.ok(outputs[2][3] > 0);
  assert.equal(outputs[2][7], 0);
  assert.equal(outputs[3][4], outputs[3][5]);
  assert.equal(outputs[3][5], outputs[3][6]);
});
