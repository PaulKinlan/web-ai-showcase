import { createHash } from "node:crypto";

export const EVIDENCE_ROUTES = ["overview", "basics", "practical", "wild", "multi"];
export const EVIDENCE_DEVICES = ["desktop", "mobile"];
export const EVIDENCE_THEMES = ["light", "dark"];
export const EXPECTED_ROUTE_DEVICE_RUNS = 10;
export const EXPECTED_SCREENSHOTS = 20;

const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const digestPattern = /^[a-f0-9]{64}$/;
const expectedRowKeys = new Set(
  EVIDENCE_DEVICES.flatMap((device) => EVIDENCE_ROUTES.map((route) => `${route}/${device}`)),
);

const isObject = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value) &&
  Object.getPrototypeOf(value) === Object.prototype;
const exactKeys = (value, keys) =>
  isObject(value) && Object.keys(value).length === keys.length &&
  keys.every((key) => Object.hasOwn(value, key));
const isDigest = (value) => typeof value === "string" && digestPattern.test(value);
const isPositiveInteger = (value) => Number.isInteger(value) && value > 0;
const isNonNegativeInteger = (value) => Number.isInteger(value) && value >= 0;
const isNonEmptyString = (value) => typeof value === "string" && value.length > 0;
const isIsoTimestamp = (value) => {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
};
const isDimensions = (value, withDpr = false) => {
  const keys = withDpr ? ["width", "height", "dpr"] : ["width", "height"];
  return exactKeys(value, keys) && isPositiveInteger(value.width) &&
    isPositiveInteger(value.height) && (!withDpr || value.dpr === 1);
};

function isSafeJsonValue(value, seen = new Set()) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value !== "object" || seen.has(value)) return false;
  seen.add(value);
  if (Array.isArray(value)) return value.every((item) => isSafeJsonValue(item, seen));
  if (!isObject(value)) return false;
  return Object.entries(value).every(([key, item]) =>
    !["__proto__", "prototype", "constructor"].includes(key) &&
    isSafeJsonValue(item, seen)
  );
}

function validResponse(response) {
  return exactKeys(response, ["isError", "text"]) &&
    typeof response.isError === "boolean" && typeof response.text === "string";
}

function validArtifact(artifact) {
  return exactKeys(artifact, [
    "route",
    "device",
    "theme",
    "path",
    "outputPath",
    "image",
    "bytes",
    "sha256",
    "viewport",
  ]) && EVIDENCE_ROUTES.includes(artifact.route) && EVIDENCE_DEVICES.includes(artifact.device) &&
    EVIDENCE_THEMES.includes(artifact.theme) &&
    /^evidence\/screenshots\/[a-z0-9-]+\.webp$/.test(artifact.path) &&
    isNonEmptyString(artifact.outputPath) && isDimensions(artifact.image) &&
    isPositiveInteger(artifact.bytes) && isDigest(artifact.sha256) &&
    isDimensions(artifact.viewport, true);
}

export function validateTypedEvent(event, previousEvent = null) {
  if (
    !exactKeys(event, [
      "sequence",
      "startedAt",
      "endedAt",
      "route",
      "device",
      "eventType",
      "action",
      "tool",
      "request",
      "response",
      "previousHash",
      "hash",
    ])
  ) return false;
  if (
    !isPositiveInteger(event.sequence) || !isIsoTimestamp(event.startedAt) ||
    !isIsoTimestamp(event.endedAt) || Date.parse(event.endedAt) < Date.parse(event.startedAt) ||
    !(event.route === null || EVIDENCE_ROUTES.includes(event.route)) ||
    !(event.device === null || EVIDENCE_DEVICES.includes(event.device)) ||
    !isNonEmptyString(event.action) || !isNonEmptyString(event.tool) ||
    !isObject(event.request) || !isSafeJsonValue(event.request) || !validResponse(event.response) ||
    !isDigest(event.previousHash) || !isDigest(event.hash)
  ) return false;

  if (event.eventType === "mcp-tool") return event.tool !== "artifact-binding";
  if (event.eventType !== "artifact-binding") return false;
  if (
    event.route === null || event.device === null || event.action !== "bind-screenshot-artifact" ||
    event.tool !== "artifact-binding" || event.response.isError || event.response.text !== "" ||
    !exactKeys(event.request, ["screenshotEventHash", "artifact"]) ||
    !isDigest(event.request.screenshotEventHash) || !validArtifact(event.request.artifact)
  ) return false;
  const artifact = event.request.artifact;
  return previousEvent?.eventType === "mcp-tool" && previousEvent.tool === "take_screenshot" &&
    !previousEvent.response.isError && previousEvent.hash === event.request.screenshotEventHash &&
    previousEvent.route === event.route && previousEvent.device === event.device &&
    previousEvent.action === `screenshot-${artifact.theme}` &&
    previousEvent.request.filePath === artifact.outputPath && artifact.route === event.route &&
    artifact.device === event.device;
}

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
    okay &&= validateTypedEvent(event, events[index - 1]) && event.sequence === index + 1 &&
      event.previousHash === previousHash && event.hash === eventHash(event);
    previousHash = event?.hash;
  }
  okay &&= producer.eventCount === events.length && producer.finalEventHash === previousHash;
  if (ledgerRaw !== undefined) okay &&= producer.ledgerSha256 === sha256(ledgerRaw);
  return Boolean(okay);
}

function validScreenshot(shot) {
  return exactKeys(shot, [
    "route",
    "device",
    "theme",
    "path",
    "expectedViewport",
    "image",
    "bytes",
    "sha256",
  ]) && EVIDENCE_ROUTES.includes(shot.route) && EVIDENCE_DEVICES.includes(shot.device) &&
    EVIDENCE_THEMES.includes(shot.theme) &&
    /^evidence\/screenshots\/[a-z0-9-]+\.webp$/.test(shot.path) &&
    isDimensions(shot.expectedViewport, true) && isDimensions(shot.image) &&
    isPositiveInteger(shot.bytes) && isDigest(shot.sha256);
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
    validScreenshot(shot) && bytes?.length > 0 && isObject(viewport) &&
      isPositiveInteger(viewport.width) && isPositiveInteger(viewport.height) &&
      isDigest(actualSha256) && shot.sha256 === actualSha256 && shot.bytes === bytes.length &&
      shot.image.width === width && shot.image.height === height && width === viewport.width &&
      height >= viewport.height && shot.expectedViewport.width === viewport.width &&
      shot.expectedViewport.height === viewport.height && shot.expectedViewport.dpr === 1,
  );
}

export function screenshotBindingMatches(input) {
  const { shot, events, absolutePath } = input;
  if (!screenshotFileMatches(input) || !Array.isArray(events) || !isNonEmptyString(absolutePath)) {
    return false;
  }
  return events.some((binding, index) => {
    const screenshotEvent = events[index - 1];
    const artifact = binding?.request?.artifact;
    return validateTypedEvent(binding, screenshotEvent) &&
      binding.eventType === "artifact-binding" && screenshotEvent.route === shot.route &&
      screenshotEvent.device === shot.device &&
      screenshotEvent.action === `screenshot-${shot.theme}` &&
      screenshotEvent.request.filePath === absolutePath && artifact.path === shot.path &&
      artifact.outputPath === absolutePath && artifact.route === shot.route &&
      artifact.device === shot.device && artifact.theme === shot.theme &&
      artifact.bytes === shot.bytes &&
      artifact.sha256 === shot.sha256 && artifact.image.width === shot.image.width &&
      artifact.image.height === shot.image.height &&
      artifact.viewport.width === shot.expectedViewport.width &&
      artifact.viewport.height === shot.expectedViewport.height &&
      artifact.viewport.dpr === shot.expectedViewport.dpr;
  });
}

function successfulCount(events, action, tool) {
  return events.filter((event) =>
    event.eventType === "mcp-tool" && event.action === action && event.tool === tool &&
    !event.response.isError
  ).length;
}

function validRow(row) {
  return exactKeys(row, ["route", "device", "status"]) && EVIDENCE_ROUTES.includes(row.route) &&
    EVIDENCE_DEVICES.includes(row.device) &&
    ["completed", "blocked", "notRun"].includes(row.status);
}

function validRecord(record) {
  if (
    !exactKeys(record, ["route", "device", "expectedViewport", "actual"]) ||
    !EVIDENCE_ROUTES.includes(record.route) || !EVIDENCE_DEVICES.includes(record.device) ||
    !isDimensions(record.expectedViewport, true)
  ) return false;
  const actual = record.actual;
  if (
    !exactKeys(actual, [
      "url",
      "ua",
      "viewport",
      "mobileLayout",
      "status",
      "coverage",
      "pointConfidence",
      "shape",
      "selectedPixels",
      "loaders",
      "classificationRows",
      "overflow",
      "controls",
    ])
  ) return false;
  return isNonEmptyString(actual.url) && URL.canParse(actual.url) && isNonEmptyString(actual.ua) &&
    isDimensions(actual.viewport, true) && typeof actual.mobileLayout === "boolean" &&
    [actual.status, actual.coverage, actual.pointConfidence, actual.shape, actual.selectedPixels]
      .every((value) => typeof value === "string") &&
    Array.isArray(actual.loaders) &&
    actual.loaders.every((value) => typeof value === "string") &&
    isNonNegativeInteger(actual.classificationRows) && isNonNegativeInteger(actual.overflow) &&
    exactKeys(actual.controls, ["threshold", "opacity", "view"]) &&
    [actual.controls.threshold, actual.controls.opacity, actual.controls.view].every((value) =>
      value === null || typeof value === "string"
    );
}

function validBlocker(blocker) {
  return exactKeys(blocker, [
    "route",
    "device",
    "action",
    "tool",
    "code",
    "detail",
    "recoverable",
    "retryDisposition",
  ]) && (blocker.route === null || EVIDENCE_ROUTES.includes(blocker.route)) &&
    (blocker.device === null || EVIDENCE_DEVICES.includes(blocker.device)) &&
    [blocker.action, blocker.tool, blocker.code, blocker.detail, blocker.retryDisposition].every(
      isNonEmptyString,
    ) && typeof blocker.recoverable === "boolean";
}

function validProducer(producer) {
  return exactKeys(producer, [
    "name",
    "tool",
    "packageVersion",
    "transport",
    "eventCount",
    "ledger",
    "ledgerSha256",
    "finalEventHash",
  ]) && producer.name === "scripts/capture-interactive-segmenter-mcp.mjs" &&
    producer.tool === "chrome-devtools-mcp" && producer.packageVersion === "1.6.0" &&
    producer.transport === "stdio" && isNonNegativeInteger(producer.eventCount) &&
    producer.ledger === "evidence/mcp-events.ndjson" && isDigest(producer.ledgerSha256) &&
    isDigest(producer.finalEventHash);
}

const validTenDenominator = (value) =>
  exactKeys(value, ["completed", "expected"]) && isNonNegativeInteger(value.completed) &&
  value.completed <= 10 && value.expected === 10;

export function validateEvidenceSummary(evidence, events, acceptedScreenshots) {
  if (
    !exactKeys(evidence, [
      "schemaVersion",
      "generatedAt",
      "status",
      "producer",
      "exactRuntime",
      "stages",
      "artifact",
      "records",
      "screenshots",
      "denominators",
      "blocker",
    ]) || evidence.schemaVersion !== 3 || !isIsoTimestamp(evidence.generatedAt) ||
    !["completed", "partial", "blocked"].includes(evidence.status) ||
    !validProducer(evidence.producer) ||
    !(evidence.exactRuntime === null || isNonEmptyString(evidence.exactRuntime)) ||
    !Array.isArray(evidence.stages) || evidence.stages.length !== 2 ||
    evidence.stages[0] !== "interactive_segmenter/magic_touch" ||
    evidence.stages[1] !== "Xenova/mobilevit-small" ||
    !exactKeys(evidence.artifact, ["url", "bytes", "sha256"]) ||
    !isNonEmptyString(evidence.artifact.url) || !URL.canParse(evidence.artifact.url) ||
    !isPositiveInteger(evidence.artifact.bytes) || !isDigest(evidence.artifact.sha256) ||
    !Array.isArray(evidence.records) || !evidence.records.every(validRecord) ||
    !Array.isArray(evidence.screenshots) || evidence.screenshots.length > 20 ||
    !evidence.screenshots.every(validScreenshot) || !Array.isArray(events) ||
    !events.every((event, index) => validateTypedEvent(event, events[index - 1])) ||
    !isNonNegativeInteger(acceptedScreenshots) || acceptedScreenshots > 20 ||
    !exactKeys(evidence.denominators, [
      "routeDeviceRuns",
      "consoleBefore",
      "consoleAfter",
      "networkBefore",
      "networkAfter",
      "screenshots",
    ])
  ) return false;

  const routeRuns = evidence.denominators.routeDeviceRuns;
  if (
    !exactKeys(routeRuns, ["completed", "blocked", "notRun", "expected", "rows"]) ||
    ![routeRuns.completed, routeRuns.blocked, routeRuns.notRun].every(isNonNegativeInteger) ||
    routeRuns.expected !== 10 || !Array.isArray(routeRuns.rows) || routeRuns.rows.length !== 10 ||
    !routeRuns.rows.every(validRow)
  ) return false;
  const keys = new Set(routeRuns.rows.map((row) => `${row.route}/${row.device}`));
  if (keys.size !== expectedRowKeys.size || [...expectedRowKeys].some((key) => !keys.has(key))) {
    return false;
  }
  const counts = Object.fromEntries(
    ["completed", "blocked", "notRun"].map((status) => [
      status,
      routeRuns.rows.filter((row) => row.status === status).length,
    ]),
  );
  if (
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
      !validTenDenominator(denominator) ||
      denominator.completed !== successfulCount(events, action, tool)
    ) return false;
  }

  const screenshots = evidence.denominators.screenshots;
  if (
    !exactKeys(screenshots, ["accepted", "blocked", "expected"]) ||
    !isNonNegativeInteger(screenshots.accepted) || !isNonNegativeInteger(screenshots.blocked) ||
    screenshots.expected !== 20 || screenshots.accepted !== acceptedScreenshots ||
    screenshots.blocked !== 20 - acceptedScreenshots ||
    evidence.screenshots.length !== acceptedScreenshots
  ) return false;

  const derivedStatus = deriveCaptureStatus(routeRuns.rows);
  const fullSuccess = counts.completed === 10 &&
    ["consoleBefore", "consoleAfter", "networkBefore", "networkAfter"].every((name) =>
      evidence.denominators[name].completed === 10
    ) && acceptedScreenshots === 20;
  if (
    evidence.status !== derivedStatus || (evidence.status === "completed") !== fullSuccess ||
    (evidence.status === "completed" ? evidence.blocker !== null : !validBlocker(evidence.blocker))
  ) {
    return false;
  }
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
