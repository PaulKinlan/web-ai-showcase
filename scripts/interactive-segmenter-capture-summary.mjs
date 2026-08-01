import { createHash } from "node:crypto";
import { renameSync, writeFileSync } from "node:fs";
import { deriveCaptureStatus } from "./interactive-segmenter-evidence.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

export function createCaptureSummary({
  ledger,
  routeDeviceRows,
  records,
  screenshots,
  blocker,
  generatedAt = new Date().toISOString(),
}) {
  const ledgerText = ledger.length
    ? ledger.map((entry) => JSON.stringify(entry)).join("\n") + "\n"
    : "";
  const completed = routeDeviceRows.filter((row) => row.status === "completed").length;
  const blocked = routeDeviceRows.filter((row) => row.status === "blocked").length;
  const successful = (action, tool) =>
    ledger.filter((event) =>
      event.eventType === "mcp-tool" && event.action === action && event.tool === tool &&
      !event.response.isError
    ).length;
  const denominator = (action, tool) => ({ completed: successful(action, tool), expected: 10 });
  return {
    ledgerText,
    summary: {
      schemaVersion: 3,
      generatedAt,
      status: deriveCaptureStatus(routeDeviceRows),
      producer: {
        name: "scripts/capture-interactive-segmenter-mcp.mjs",
        tool: "chrome-devtools-mcp",
        packageVersion: "1.6.0",
        transport: "stdio",
        eventCount: ledger.length,
        ledger: "evidence/mcp-events.ndjson",
        ledgerSha256: sha256(ledgerText),
        finalEventHash: ledger.at(-1)?.hash || "0".repeat(64),
      },
      exactRuntime: records[0]?.actual?.ua || null,
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
          completed,
          blocked,
          notRun: 10 - completed - blocked,
          expected: 10,
          rows: routeDeviceRows,
        },
        consoleBefore: denominator("console-before", "list_console_messages"),
        consoleAfter: denominator("console-after", "list_console_messages"),
        networkBefore: denominator("network-before", "list_network_requests"),
        networkAfter: denominator("network-after", "list_network_requests"),
        screenshots: {
          accepted: screenshots.length,
          blocked: 20 - screenshots.length,
          expected: 20,
        },
      },
      blocker,
    },
  };
}

export function writeAtomicCaptureOutputs(input) {
  const { ledgerPath, summaryPath } = input;
  const { ledgerText, summary } = createCaptureSummary(input);
  writeFileSync(ledgerPath, ledgerText);
  const temporarySummaryPath = `${summaryPath}.tmp-${process.pid}`;
  writeFileSync(temporarySummaryPath, JSON.stringify(summary, null, 2) + "\n");
  renameSync(temporarySummaryPath, summaryPath);
  return summary;
}
