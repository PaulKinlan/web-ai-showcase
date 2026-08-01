#!/usr/bin/env node
// Executable chrome-devtools-mcp producer for the MagicTouch portfolio acceptance ledger.
// This script uses MCP tools for navigation and every user interaction. evaluate_script only reads
// rendered state or waits for an already-triggered event; it never injects inference results.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeAtomicCaptureOutputs } from "./interactive-segmenter-capture-summary.mjs";
import { expectedRouteDeviceRows } from "./interactive-segmenter-evidence.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..");
const slug = "interactive-segmenter";
const evidenceDir = join(repoRoot, "models", slug, "evidence");
const screenshotDir = join(evidenceDir, "screenshots");
const ledgerPath = join(evidenceDir, "mcp-events.ndjson");
const summaryPath = join(evidenceDir, "acceptance.json");
const baseArg = process.argv.find((arg) => arg.startsWith("--base-url="));
const baseUrl = (baseArg?.slice("--base-url=".length) || "http://127.0.0.1:8919/web-ai-showcase/")
  .replace(/\/?$/, "/");
const mcpBundle = process.env.CHROME_DEVTOOLS_MCP_BUNDLE ||
  "/home/paulkinlan/.npm/_npx/15c61037b1978c83/node_modules/chrome-devtools-mcp/build/src/third_party/index.js";
const mcpServer = process.env.CHROME_DEVTOOLS_MCP_SERVER ||
  "/home/paulkinlan/.npm/_npx/15c61037b1978c83/node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const textOf = (result) =>
  (result?.content || []).filter((item) => item.type === "text")
    .map((item) => item.text).join("\n");
const stages = ["interactive_segmenter/magic_touch", "Xenova/mobilevit-small"];
const routes = [
  { id: "overview", path: "models/interactive-segmenter/" },
  { id: "basics", path: "models/interactive-segmenter/basics/" },
  { id: "practical", path: "models/interactive-segmenter/practical/" },
  { id: "wild", path: "models/interactive-segmenter/wild/" },
  { id: "multi", path: "models/interactive-segmenter/multi-model/" },
];
const devices = [
  { id: "desktop", viewport: "1280x800x1", width: 1280, height: 800, mobile: false },
  { id: "mobile", viewport: "360x740x1,mobile,touch", width: 360, height: 740, mobile: true },
];

mkdirSync(screenshotDir, { recursive: true });
let client;
let sequence = 0;
let previousHash = "0".repeat(64);
const ledger = [];
let context = { route: null, device: null, action: "start" };
function appendEvent(fields) {
  const base = {
    sequence: ++sequence,
    startedAt: fields.startedAt || new Date().toISOString(),
    endedAt: fields.endedAt || new Date().toISOString(),
    route: fields.route ?? context.route,
    device: fields.device ?? context.device,
    eventType: fields.eventType,
    action: fields.action,
    tool: fields.tool,
    request: fields.request || {},
    response: fields.response || { isError: false, text: "" },
    previousHash,
  };
  const event = { ...base, hash: sha256(JSON.stringify(base)) };
  ledger.push(event);
  previousHash = event.hash;
  writeFileSync(ledgerPath, ledger.map((entry) => JSON.stringify(entry)).join("\n") + "\n");
  return event;
}
async function call(tool, request = {}, next = {}) {
  context = { ...context, ...next };
  const startedAt = new Date().toISOString();
  let response;
  let responseText = "";
  let thrown;
  try {
    response = await client.callTool(
      { name: tool, arguments: request },
      undefined,
      { timeout: 240_000, maxTotalTimeout: 240_000 },
    );
    responseText = textOf(response);
  } catch (error) {
    thrown = error;
    responseText = String(error?.stack || error);
  }
  const event = appendEvent({
    startedAt,
    endedAt: new Date().toISOString(),
    eventType: "mcp-tool",
    action: context.action,
    tool,
    request,
    response: { isError: Boolean(thrown || response?.isError), text: responseText },
  });
  if (thrown) throw thrown;
  if (response?.isError) throw new Error(`${tool} failed: ${responseText}`);
  return { response, text: responseText, event };
}

function bindScreenshot(shot, screenshotEvent, absolutePath) {
  const bytes = readFileSync(absolutePath);
  const [width, height] = execFileSync("identify", ["-format", "%w %h", absolutePath], {
    encoding: "utf8",
  }).trim().split(/\s+/).map(Number);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`identify returned invalid dimensions for ${absolutePath}`);
  }
  const artifact = {
    route: shot.route,
    device: shot.device,
    theme: shot.theme,
    path: shot.path,
    outputPath: absolutePath,
    image: { width, height },
    bytes: bytes.length,
    sha256: sha256(bytes),
    viewport: shot.expectedViewport,
  };
  appendEvent({
    eventType: "artifact-binding",
    action: "bind-screenshot-artifact",
    tool: "artifact-binding",
    request: { screenshotEventHash: screenshotEvent.hash, artifact },
  });
  return {
    route: shot.route,
    device: shot.device,
    theme: shot.theme,
    path: shot.path,
    expectedViewport: shot.expectedViewport,
    image: artifact.image,
    bytes: artifact.bytes,
    sha256: artifact.sha256,
  };
}

function snapshotUid(snapshot, role, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = snapshot.match(new RegExp(`uid=([^\\s]+) ${role} "${escaped}"(?:\\s|$)`));
  if (!match) throw new Error(`No ${role} named ${name} in MCP snapshot`);
  return match[1];
}

async function snapshot(action) {
  return (await call("take_snapshot", {}, { action })).text;
}
async function clickNamed(role, name, action) {
  const tree = await snapshot(`${action}:snapshot`);
  const uid = snapshotUid(tree, role, name);
  await call("click", { uid }, { action });
  return uid;
}
async function waitFor(text, action, timeout = 120_000) {
  return await call("wait_for", { text: [text], timeout }, { action });
}
async function waitForSnapshot(predicate, action, timeout = 120_000) {
  const deadline = Date.now() + timeout;
  let attempt = 0;
  while (Date.now() < deadline) {
    const tree = await snapshot(`${action}:poll-${++attempt}`);
    if (predicate(tree)) return tree;
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
  throw new Error(`${action} did not reach its required rendered state in ${timeout} ms`);
}
const inferenceComplete = (tree) => /StaticText "Selected \d+(?:\.\d+)?% of the image/.test(tree);
async function settle(action, milliseconds = 700) {
  await call("evaluate_script", {
    function:
      `async () => { await new Promise(resolve => setTimeout(resolve, ${milliseconds})); return "settled"; }`,
  }, { action });
}

const records = [];
const screenshots = [];
const routeDeviceRows = expectedRouteDeviceRows();
let activeRow = null;
let blocker = null;

try {
  const { Client, StdioClientTransport } = await import(mcpBundle);
  const transport = new StdioClientTransport({
    command: "node",
    args: [
      mcpServer,
      "--channel=stable",
      "--headless",
      "--isolated",
      "--viewport=1280x800",
      "--no-usage-statistics",
      "--no-performance-crux",
      "--redactNetworkHeaders",
      "--allowUnrestrictedPaths",
      "--screenshotFormat=webp",
      "--screenshotQuality=82",
    ],
  });
  client = new Client({ name: "webai-magic-touch-acceptance", version: "1.0.0" }, {
    capabilities: {},
  });
  await client.connect(transport);
  await call("new_page", { url: "about:blank", timeout: 30_000 }, {
    route: null,
    device: null,
    action: "new-page",
  });

  for (const device of devices) {
    for (const route of routes) {
      activeRow = routeDeviceRows.find((row) => row.route === route.id && row.device === device.id);
      context = { route: route.id, device: device.id, action: "emulate-light" };
      await call("emulate", { viewport: device.viewport, colorScheme: "light" });
      await call("navigate_page", {
        type: "url",
        url: new URL(route.path, baseUrl).href,
        timeout: 30_000,
      }, { action: "navigate" });
      await call("list_console_messages", {}, { action: "console-before" });
      await call("list_network_requests", {}, { action: "network-before" });

      let tree = await snapshot("initial-snapshot");
      if (/button "Download model \(~5\.94 MB\)"/.test(tree)) {
        await call("click", { uid: snapshotUid(tree, "button", "Download model (~5.94 MB)") }, {
          action: "download-model",
        });
      } else if (/button "Load model into memory"/.test(tree)) {
        await call("click", { uid: snapshotUid(tree, "button", "Load model into memory") }, {
          action: "load-cached-model",
        });
      }
      await waitFor("MagicTouch ready", "wait-model-ready");
      tree = await snapshot("ready-snapshot");

      if (route.id === "overview") {
        await call("click", {
          uid: snapshotUid(tree, "button", "Image prompt: choose an object point"),
        }, { action: "point-click" });
        await waitForSnapshot(inferenceComplete, "wait-point-inference");
        for (const mode of ["Confidence", "Cut-out", "Spotlight", "Overlay"]) {
          await clickNamed("button", mode, `mask-mode-${mode.toLowerCase()}`);
          await settle(`settle-mask-mode-${mode.toLowerCase()}`);
        }
        tree = await snapshot("slider-snapshot");
        await call("fill", {
          uid: snapshotUid(tree, "slider", "Foreground threshold"),
          value: "65",
        }, {
          action: "threshold-65",
        });
        tree = await snapshot("opacity-snapshot");
        await call("fill", { uid: snapshotUid(tree, "slider", "Overlay strength"), value: "75" }, {
          action: "opacity-75",
        });
        await settle("settle-sliders");
        tree = await snapshot("upload-snapshot");
        await call("upload_file", {
          uid: snapshotUid(tree, "button", "Use your own image"),
          filePath: join(repoRoot, "media", "assets", "dog-outdoor.jpg"),
        }, { action: "upload-rights-safe-fixture" });
        await clickNamed("button", "Segment at target", "segment-upload");
        await waitForSnapshot(inferenceComplete, "wait-upload-inference");
      } else if (route.id === "basics") {
        await clickNamed("button", "top centre", "preset-point");
        await waitForSnapshot(inferenceComplete, "wait-preset-inference");
      } else if (route.id === "practical") {
        await clickNamed("button", "Segment at target", "segment-practical");
        await waitForSnapshot(inferenceComplete, "wait-practical-inference");
        await clickNamed("button", "Download transparent PNG", "export-cutout");
      } else if (route.id === "wild") {
        await call("click", {
          uid: snapshotUid(tree, "button", "Image prompt: choose an object point"),
        }, { action: "focus-point-canvas" });
        await waitForSnapshot(inferenceComplete, "wait-initial-wild-inference");
        await call("press_key", { key: "ArrowRight" }, { action: "keyboard-move-point" });
        await call("press_key", { key: "Enter" }, { action: "keyboard-run" });
        await settle("wait-keyboard-inference", 2_000);
      } else if (route.id === "multi") {
        tree = await snapshot("classifier-loader-snapshot");
        if (/button "Download model \(~6\.1 MB\)"/.test(tree)) {
          await call("click", {
            uid: snapshotUid(tree, "button", "Download model (~6.1 MB)"),
          }, { action: "download-mobilevit" });
        } else if ((tree.match(/button "Load model into memory"/g) || []).length > 0) {
          const matches = [...tree.matchAll(/uid=([^\s]+) button "Load model into memory"/g)];
          await call("click", { uid: matches.at(-1)[1] }, { action: "load-cached-mobilevit" });
        }
        await waitForSnapshot(
          (snapshotText) => (snapshotText.match(/button "Release from memory"/g) || []).length >= 2,
          "wait-mobilevit-ready",
          180_000,
        );
        await clickNamed("button", "Segment at target", "segment-multi-model");
        await waitForSnapshot(
          (snapshotText) => /StaticText "MobileViT latency \d/.test(snapshotText),
          "wait-mobilevit-classification",
          180_000,
        );
      }

      const state = await call("evaluate_script", {
        function: `() => ({
          url: location.href,
          ua: navigator.userAgent,
          viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
          mobileLayout: getComputedStyle(document.querySelector('.interactive-toolbar')).display === 'flex' &&
            [...document.querySelector('.interactive-toolbar').children].every(el => el.getBoundingClientRect().width >= innerWidth * 0.75),
          status: document.querySelector('#status')?.textContent?.trim(),
          coverage: document.querySelector('#coverage')?.textContent?.trim(),
          pointConfidence: document.querySelector('#point-confidence')?.textContent?.trim(),
          shape: document.querySelector('#inside-shape')?.textContent?.trim(),
          selectedPixels: document.querySelector('#inside-pixels')?.textContent?.trim(),
          loaders: [...document.querySelectorAll('[data-model-loader-state]')].map(el => el.dataset.modelLoaderState),
          classificationRows: document.querySelectorAll('.classification-row').length,
          overflow: Math.max(0, document.documentElement.scrollWidth - innerWidth),
          controls: {
            threshold: document.querySelector('#threshold')?.value,
            opacity: document.querySelector('#opacity')?.value,
            view: document.querySelector('[data-view][aria-pressed="true"]')?.dataset.view
          }
        })`,
      }, { action: "read-completed-state" });
      const parsed = JSON.parse(state.text.match(/```json\n([\s\S]*?)\n```/)?.[1] || "null");
      records.push({
        route: route.id,
        device: device.id,
        expectedViewport: { width: device.width, height: device.height, dpr: 1 },
        actual: parsed,
      });

      await call("list_console_messages", {}, { action: "console-after" });
      await call("list_network_requests", {}, { action: "network-after" });
      for (const theme of ["light", "dark"]) {
        // The MCP emulate tool replaces the entire emulation state; omitting viewport here
        // resets Puppeteer's capture viewport to the 1280px launch default.
        await call("emulate", { viewport: device.viewport, colorScheme: theme }, {
          action: `theme-${theme}`,
        });
        await settle(`settle-theme-${theme}`, 250);
        const relative = `evidence/screenshots/${route.id}-${device.id}-${theme}.webp`;
        const absolute = join(repoRoot, "models", slug, relative);
        const capture = await call("take_screenshot", {
          format: "webp",
          quality: 82,
          // chrome-devtools-mcp full-page capture uses the launch viewport width even after
          // mobile emulation. Viewport capture preserves the exact 360px mobile evidence width.
          fullPage: !device.mobile,
          filePath: absolute,
        }, { action: `screenshot-${theme}` });
        screenshots.push(bindScreenshot(
          {
            route: route.id,
            device: device.id,
            theme,
            path: relative,
            expectedViewport: { width: device.width, height: device.height, dpr: 1 },
          },
          capture.event,
          absolute,
        ));
      }
      activeRow.status = "completed";
      activeRow = null;
    }
  }
} catch (error) {
  if (activeRow) activeRow.status = "blocked";
  blocker = {
    route: context.route,
    device: context.device,
    action: context.action,
    tool: ledger.at(-1)?.tool || "producer",
    code: "capture-exception",
    detail: String(error?.stack || error),
    recoverable: true,
    retryDisposition: "requires one fresh-process capture; this producer does not retry",
  };
  process.exitCode = 1;
} finally {
  if (client) {
    try {
      await client.close();
    } catch (error) {
      if (!blocker) {
        const lastCompleted = [...routeDeviceRows].reverse().find((row) =>
          row.status === "completed"
        );
        if (lastCompleted) lastCompleted.status = "blocked";
        blocker = {
          route: context.route,
          device: context.device,
          action: "close-mcp-client",
          tool: "producer",
          code: "capture-close-exception",
          detail: String(error?.stack || error),
          recoverable: true,
          retryDisposition: "requires one fresh-process capture; this producer does not retry",
        };
        process.exitCode = 1;
      }
    }
  }

  const summary = writeAtomicCaptureOutputs({
    ledgerPath,
    summaryPath,
    ledger,
    routeDeviceRows,
    records,
    screenshots,
    blocker,
  });
  console.log(`wrote ${summaryPath} (${summary.status})`);
  console.log(`wrote ${ledgerPath} (${ledger.length} hash-chained evidence events)`);
}
