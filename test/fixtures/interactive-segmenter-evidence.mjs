import { createHash } from "node:crypto";
import {
  eventHash,
  expectedRouteDeviceRows,
} from "../../scripts/interactive-segmenter-evidence.mjs";

const digest = (value) => createHash("sha256").update(value).digest("hex");

export function perfectInteractiveSegmenterEvidence() {
  const events = [];
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
      response: { isError: false, text: "fixture success" },
      previousHash,
    };
    const event = { ...base, hash: eventHash(base) };
    events.push(event);
    previousHash = event.hash;
    return event;
  };

  for (const row of expectedRouteDeviceRows()) {
    for (
      const [action, tool] of [
        ["console-before", "list_console_messages"],
        ["network-before", "list_network_requests"],
        ["console-after", "list_console_messages"],
        ["network-after", "list_network_requests"],
      ]
    ) append({ ...row, action, tool });

    for (const theme of ["light", "dark"]) {
      const viewport = row.device === "desktop"
        ? { width: 1280, height: 800, dpr: 1 }
        : { width: 360, height: 740, dpr: 1 };
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
            ...shot,
            outputPath: absolutePath,
            viewport,
          },
        },
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
    status: "completed",
    producer: {
      eventCount: events.length,
      finalEventHash: previousHash,
      ledgerSha256: digest(ledgerRaw),
    },
    records: [],
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
