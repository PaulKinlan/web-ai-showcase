import { createHash } from "node:crypto";

export const EVIDENCE_ROUTES = ["overview", "basics", "practical", "wild", "multi"];
export const EVIDENCE_DEVICES = ["desktop", "mobile"];
export const EVIDENCE_THEMES = ["light", "dark"];
export const EXPECTED_ROUTE_DEVICE_RUNS = 10;
export const EXPECTED_SCREENSHOTS = 20;

const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const expectedRowKeys = new Set(
  EVIDENCE_DEVICES.flatMap((device) => EVIDENCE_ROUTES.map((route) => `${route}/${device}`)),
);

export function expectedRouteDeviceRows() {
  return EVIDENCE_DEVICES.flatMap((device) =>
    EVIDENCE_ROUTES.map((route) => ({ route, device, status: "notRun" }))
  );
}

export function deriveCaptureStatus(rows) {
  const completed = rows.filter((row) => row.status === "completed").length;
  if (completed === EXPECTED_ROUTE_DEVICE_RUNS) return "completed";
  return completed === 0 ? "blocked" : "partial";
}

export function eventHash(event) {
  const { hash: _hash, ...base } = event;
  return sha256(JSON.stringify(base));
}

export function validateLedgerChain(events, producer = {}, ledgerRaw) {
  let previousHash = "0".repeat(64);
  let okay = Array.isArray(events) && events.length > 0;
  for (let index = 0; index < (events?.length || 0); index++) {
    const event = events[index];
    const eventShape = event?.eventType === "mcp-tool"
      ? typeof event.tool === "string" && event.tool.length > 0 &&
        typeof event.action === "string" && event.action.length > 0 &&
        event.request && typeof event.request === "object" &&
        event.response && typeof event.response.isError === "boolean"
      : event?.eventType === "artifact-binding" &&
        event.action === "bind-screenshot-artifact" && event.tool === "artifact-binding" &&
        event.request?.screenshotEventHash && event.request?.artifact;
    okay &&= Boolean(eventShape) && event.sequence === index + 1 &&
      event.previousHash === previousHash && event.hash === eventHash(event);
    previousHash = event.hash;
  }
  okay &&= producer.eventCount === events.length && producer.finalEventHash === previousHash;
  if (ledgerRaw !== undefined) okay &&= producer.ledgerSha256 === sha256(ledgerRaw);
  return Boolean(okay);
}

export function screenshotFileMatches({
  shot,
  bytes,
  sha256: actualSha256,
  width,
  height,
  viewport,
}) {
  return Boolean(
    shot && bytes && viewport && shot.sha256 === actualSha256 && shot.bytes === bytes.length &&
      shot.image?.width === width && shot.image?.height === height && width === viewport.width &&
      height >= viewport.height && shot.expectedViewport?.width === viewport.width &&
      shot.expectedViewport?.height === viewport.height && shot.expectedViewport?.dpr === 1 &&
      EVIDENCE_ROUTES.includes(shot.route) && EVIDENCE_DEVICES.includes(shot.device) &&
      EVIDENCE_THEMES.includes(shot.theme),
  );
}

export function screenshotBindingMatches(input) {
  const { shot, events, absolutePath } = input;
  if (!screenshotFileMatches(input) || !Array.isArray(events) || !absolutePath) return false;

  return events.some((binding, index) => {
    if (binding.eventType !== "artifact-binding" || binding.action !== "bind-screenshot-artifact") {
      return false;
    }
    const screenshotEvent = events[index - 1];
    const artifact = binding.request?.artifact;
    return screenshotEvent?.eventType === "mcp-tool" &&
      screenshotEvent.tool === "take_screenshot" && !screenshotEvent.response?.isError &&
      screenshotEvent.route === shot.route && screenshotEvent.device === shot.device &&
      screenshotEvent.action === `screenshot-${shot.theme}` &&
      screenshotEvent.request?.filePath === absolutePath &&
      binding.route === shot.route && binding.device === shot.device &&
      binding.request?.screenshotEventHash === screenshotEvent.hash &&
      artifact?.path === shot.path && artifact?.outputPath === absolutePath &&
      artifact?.route === shot.route && artifact?.device === shot.device &&
      artifact?.theme === shot.theme && artifact?.bytes === shot.bytes &&
      artifact?.sha256 === shot.sha256 && artifact?.image?.width === shot.image.width &&
      artifact?.image?.height === shot.image.height &&
      artifact?.viewport?.width === shot.expectedViewport.width &&
      artifact?.viewport?.height === shot.expectedViewport.height &&
      artifact?.viewport?.dpr === shot.expectedViewport.dpr;
  });
}

function successfulCount(events, action, tool) {
  return events.filter((event) =>
    event.eventType === "mcp-tool" && event.action === action && event.tool === tool &&
    !event.response?.isError
  ).length;
}

export function validateEvidenceSummary(evidence, events, acceptedScreenshots) {
  if (
    evidence?.schemaVersion !== 3 ||
    !["completed", "partial", "blocked"].includes(evidence?.status) ||
    !Array.isArray(evidence?.records) || !Array.isArray(evidence?.screenshots) ||
    !Array.isArray(events)
  ) return false;
  const rows = evidence?.denominators?.routeDeviceRuns?.rows;
  if (!Array.isArray(rows) || rows.length !== EXPECTED_ROUTE_DEVICE_RUNS) return false;
  const keys = new Set(rows.map((row) => `${row.route}/${row.device}`));
  if (
    keys.size !== expectedRowKeys.size ||
    [...expectedRowKeys].some((key) => !keys.has(key)) ||
    rows.some((row) => !["completed", "blocked", "notRun"].includes(row.status))
  ) return false;

  const routeRuns = evidence.denominators.routeDeviceRuns;
  const counts = Object.fromEntries(
    ["completed", "blocked", "notRun"].map((status) => [
      status,
      rows.filter((row) => row.status === status).length,
    ]),
  );
  if (
    routeRuns.expected !== EXPECTED_ROUTE_DEVICE_RUNS ||
    routeRuns.completed !== counts.completed || routeRuns.blocked !== counts.blocked ||
    routeRuns.notRun !== counts.notRun
  ) return false;

  for (
    const [name, action, tool] of [
      ["consoleBefore", "console-before", "list_console_messages"],
      ["consoleAfter", "console-after", "list_console_messages"],
      ["networkBefore", "network-before", "list_network_requests"],
      ["networkAfter", "network-after", "list_network_requests"],
    ]
  ) {
    const denominator = evidence.denominators[name];
    if (
      denominator?.expected !== EXPECTED_ROUTE_DEVICE_RUNS ||
      denominator?.completed !== successfulCount(events, action, tool)
    ) return false;
  }

  const screenshots = evidence.denominators.screenshots;
  if (
    screenshots?.expected !== EXPECTED_SCREENSHOTS ||
    screenshots?.accepted !== acceptedScreenshots ||
    screenshots?.blocked !== EXPECTED_SCREENSHOTS - acceptedScreenshots ||
    evidence.screenshots?.length !== acceptedScreenshots
  ) return false;

  const derivedStatus = deriveCaptureStatus(rows);
  const fullSuccess = counts.completed === EXPECTED_ROUTE_DEVICE_RUNS &&
    ["consoleBefore", "consoleAfter", "networkBefore", "networkAfter"].every((name) =>
      evidence.denominators[name].completed === EXPECTED_ROUTE_DEVICE_RUNS
    ) && acceptedScreenshots === EXPECTED_SCREENSHOTS;
  if (
    evidence.status !== derivedStatus || (evidence.status === "completed") !== fullSuccess ||
    (evidence.status === "completed" ? evidence.blocker !== null : !evidence.blocker)
  ) return false;
  return true;
}

export function completeEvidencePasses(evidence, events, acceptedScreenshots) {
  return evidence?.status === "completed" &&
    validateEvidenceSummary(evidence, events, acceptedScreenshots);
}

export function sourceUsesExecutableMcp(captureSource, validatorSource) {
  return /new Client\(/.test(captureSource) && /StdioClientTransport/.test(captureSource) &&
    /client\.callTool/.test(captureSource) && /"click"/.test(captureSource) &&
    /"press_key"/.test(captureSource) && /"upload_file"/.test(captureSource) &&
    /"list_console_messages"/.test(captureSource) &&
    /"list_network_requests"/.test(captureSource) &&
    /bind-screenshot-artifact/.test(captureSource) && !/void\s*\[/.test(validatorSource);
}
