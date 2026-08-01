import { createHash } from "node:crypto";
import {
  eventHash,
  expectedRouteDeviceRows,
} from "../../scripts/interactive-segmenter-evidence.mjs";

const digest = (value) => createHash("sha256").update(value).digest("hex");

export function perfectInteractiveSegmenterEvidence() {
  const events = [];
  const records = [];
  const screenshots = [];
  const screenshotInputs = [];
  let previousHash = "0".repeat(64);
  const append = (fields) => {
    const base = {
      sequence: events.length + 1,
      startedAt: "2026-08-02T00:00:00.000Z",
      endedAt: "2026-08-02T00:00:00.001Z",
      route: fields.route,
      device: fields.device,
      eventType: fields.eventType || "mcp-tool",
      action: fields.action,
      tool: fields.tool,
      request: fields.request || {},
      response: fields.response || { isError: false, text: "fixture success" },
      previousHash,
    };
    const event = { ...base, hash: eventHash(base) };
    events.push(event);
    previousHash = event.hash;
    return event;
  };

  for (const row of expectedRouteDeviceRows()) {
    const viewport = row.device === "desktop"
      ? { width: 1280, height: 800, dpr: 1 }
      : { width: 360, height: 740, dpr: 1 };
    append({ ...row, action: "visible-control", tool: "click", request: { uid: "fixture" } });
    for (
      const [action, tool] of [
        ["console-before", "list_console_messages"],
        ["network-before", "list_network_requests"],
        ["console-after", "list_console_messages"],
        ["network-after", "list_network_requests"],
      ]
    ) append({ ...row, action, tool });

    records.push({
      route: row.route,
      device: row.device,
      expectedViewport: viewport,
      actual: {
        url: `https://fixture.invalid/models/interactive-segmenter/${row.route}/`,
        ua: "Mozilla/5.0 HeadlessChrome/150.0.0.0",
        viewport,
        mobileLayout: row.device === "mobile",
        status: "Selected 12.3% of the image at 98.7% point confidence.",
        coverage: "12.3%",
        pointConfidence: "98.7%",
        shape: "32×32 foreground confidence + category mask",
        selectedPixels: "126 (12.3%)",
        loaders: [],
        classificationRows: row.route === "multi" ? 3 : 0,
        overflow: 0,
        controls: { threshold: "50", opacity: "60", view: "overlay" },
      },
    });

    for (const theme of ["light", "dark"]) {
      const path = `evidence/screenshots/${row.route}-${row.device}-${theme}.webp`;
      const absolutePath = `/repo/models/interactive-segmenter/${path}`;
      const bytes = Buffer.from(`${row.route}/${row.device}/${theme}`);
      const shot = {
        route: row.route,
        device: row.device,
        theme,
        path,
        expectedViewport: viewport,
        image: { width: viewport.width, height: viewport.height },
        bytes: bytes.length,
        sha256: digest(bytes),
      };
      const screenshotEvent = append({
        ...row,
        action: `screenshot-${theme}`,
        tool: "take_screenshot",
        request: { filePath: absolutePath, fullPage: true },
      });
      append({
        ...row,
        eventType: "artifact-binding",
        action: "bind-screenshot-artifact",
        tool: "artifact-binding",
        request: {
          screenshotEventHash: screenshotEvent.hash,
          artifact: {
            route: row.route,
            device: row.device,
            theme,
            path,
            outputPath: absolutePath,
            image: shot.image,
            bytes: shot.bytes,
            sha256: shot.sha256,
            viewport,
          },
        },
        response: { isError: false, text: "" },
      });
      screenshots.push(shot);
      screenshotInputs.push({
        shot,
        bytes,
        sha256: digest(bytes),
        width: viewport.width,
        height: viewport.height,
        viewport,
        events,
        absolutePath,
      });
    }
  }

  const ledgerRaw = events.map((event) => JSON.stringify(event)).join("\n") + "\n";
  const evidence = {
    schemaVersion: 3,
    generatedAt: "2026-08-02T00:00:01.000Z",
    status: "completed",
    producer: {
      name: "scripts/capture-interactive-segmenter-mcp.mjs",
      tool: "chrome-devtools-mcp",
      packageVersion: "1.6.0",
      transport: "stdio",
      eventCount: events.length,
      ledger: "evidence/mcp-events.ndjson",
      ledgerSha256: digest(ledgerRaw),
      finalEventHash: previousHash,
    },
    exactRuntime: "Mozilla/5.0 HeadlessChrome/150.0.0.0",
    stages: ["interactive_segmenter/magic_touch", "Xenova/mobilevit-small"],
    artifact: {
      url:
        "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite",
      bytes: 6_227_884,
      sha256: "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25",
    },
    records,
    screenshots,
    denominators: {
      routeDeviceRuns: {
        completed: 10,
        blocked: 0,
        notRun: 0,
        expected: 10,
        rows: expectedRouteDeviceRows().map((row) => ({ ...row, status: "completed" })),
      },
      consoleBefore: { completed: 10, expected: 10 },
      consoleAfter: { completed: 10, expected: 10 },
      networkBefore: { completed: 10, expected: 10 },
      networkAfter: { completed: 10, expected: 10 },
      screenshots: { accepted: 20, blocked: 0, expected: 20 },
    },
    blocker: null,
  };
  return { evidence, events, ledgerRaw, screenshotInputs };
}
