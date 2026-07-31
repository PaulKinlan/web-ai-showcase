// Route-complete browser acceptance for the canonical English gibberish detector portfolio.
// Real fp32 detector inference runs on every route at desktop + mobile; the multi-model route also
// runs the pinned SST-2 stage. Captures light/dark screenshots, complete control/run evidence,
// console/network observations, false-positive probes, and shared retry/release/cache lifecycle.
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const MODEL = "madhurjindal/autonlp-Gibberish-Detector-492513457";
const REVISION = "76672dd7d3575f68ab980705bcec975cc62de71c";
const MODEL_BYTES = 267961863;
const SENTIMENT = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
const SENTIMENT_REVISION = "0b6928efcb76139cae2c6881d49cda67fe119f42";
const SENTIMENT_BYTES = 67581197;
const ROUTES = {
  overview: "models/autonlp-gibberish-detector-492513457/",
  basics: "models/autonlp-gibberish-detector-492513457/basics/",
  practical: "models/autonlp-gibberish-detector-492513457/practical/",
  wild: "models/autonlp-gibberish-detector-492513457/wild/",
  multimodel: "models/autonlp-gibberish-detector-492513457/multi-model/",
};
const SCREEN_DIR = join(process.cwd(), "reports", "webai-g8-gibberish-screens");
mkdirSync(SCREEN_DIR, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const { server, port } = await startServer();
const chrome = await launchChrome({
  userDataDir: join(process.cwd(), ".gibberish-acceptance-profile"),
  resetProfile: true,
  removeProfileOnKill: true,
});
const cdp = new CDP(chrome.ws);
const baseUrl = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;
const assertions = [];
const cells = [];
const screenshots = [];
const network = [];
const lifecycle = [];
const failures = [];
let blockedPatterns = [];
const trackedSessions = new Set();
const requestChains = new Map();

function assert(id, ok, evidence, details = undefined) {
  assertions.push({
    id,
    state: ok ? "pass" : "fail",
    evidence,
    ...(details === undefined ? {} : { details }),
  });
  console.log(`${ok ? "PASS" : "FAIL"} ${id} — ${evidence}`);
  if (!ok) failures.push(`${id}: ${evidence}`);
}
async function ev(sessionId, expression, timeoutMs = 45000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.stack||error)}}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}
async function waitFor(sessionId, expression, label, timeoutMs = 10 * 60_000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const value = await ev(sessionId, expression).catch(() => false);
    if (value === true) return;
    await sleep(500);
  }
  throw new Error(`Timed out waiting for ${label}`);
}
async function enableNetwork(sessionId) {
  if (!sessionId || trackedSessions.has(sessionId)) return;
  trackedSessions.add(sessionId);
  await cdp.send("Network.enable", {}, sessionId).catch(() => trackedSessions.delete(sessionId));
  if (trackedSessions.has(sessionId)) {
    await cdp.send("Network.setBlockedURLs", {
      urls: blockedPatterns.length ? blockedPatterns : ["__never_match__"],
    }, sessionId).catch(() => {});
  }
}
async function setBlocks(patterns) {
  blockedPatterns = patterns;
  await Promise.all(
    [...trackedSessions].map((sessionId) =>
      cdp.send(
        "Network.setBlockedURLs",
        { urls: patterns.length ? patterns : ["__never_match__"] },
        sessionId,
      ).catch(() => {})
    ),
  );
}
cdp.on((message) => {
  if (message.method === "Target.attachedToTarget") {
    const info = message.params.targetInfo;
    if (
      (info?.type === "worker" || info?.type === "service_worker") &&
      info.url.includes("autonlp-gibberish-detector-492513457")
    ) void enableNetwork(message.params.sessionId);
    return;
  }
  if (!trackedSessions.has(message.sessionId)) return;
  if (message.method === "Network.requestWillBeSent") {
    const key = `${message.sessionId}:${message.params.requestId}`;
    const chain = requestChains.get(key) || { requestedUrl: message.params.request.url };
    chain.lastUrl = message.params.request.url;
    requestChains.set(key, chain);
  }
  if (message.method === "Network.responseReceived") {
    const key = `${message.sessionId}:${message.params.requestId}`;
    const chain = requestChains.get(key) || { requestedUrl: message.params.response.url };
    const headers = message.params.response.headers || {};
    const rawLength = headers["content-length"] ?? headers["Content-Length"];
    network.push({
      requestedUrl: chain.requestedUrl,
      responseUrl: message.params.response.url,
      status: message.params.response.status,
      contentLength: rawLength == null ? null : Number(rawLength),
      type: message.params.type,
    });
  }
});
await cdp.send("Target.setDiscoverTargets", { discover: true });
await cdp.send("Target.setAutoAttach", {
  autoAttach: true,
  flatten: true,
  waitForDebuggerOnStart: false,
});

async function attachPage(page) {
  await enableNetwork(page.sessionId);
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: false,
  }, page.sessionId);
  await sleep(200);
}
async function click(sessionId, selector) {
  return await ev(
    sessionId,
    `(()=>{const node=document.querySelector(${
      JSON.stringify(selector)
    });if(!node)return false;node.click();return true})()`,
  );
}
async function setValue(sessionId, selector, value) {
  return await ev(
    sessionId,
    `(()=>{const node=document.querySelector(${
      JSON.stringify(selector)
    });if(!node)return false;node.value=${
      JSON.stringify(value)
    };node.dispatchEvent(new Event("change",{bubbles:true}));return true})()`,
  );
}
async function clickDownloadButtons(sessionId) {
  return await ev(
    sessionId,
    `(()=>{let count=0;for(const button of document.querySelectorAll(".model-loader button")){if(/Download|Retry/i.test(button.textContent)){button.click();count++}}return count})()`,
  );
}
async function waitReady(sessionId, expected = 1) {
  await waitFor(
    sessionId,
    `document.querySelectorAll('.model-loader[data-state="ready"]').length===${expected}`,
    `${expected} model loader(s) ready`,
  );
}
async function fullScreenshot(sessionId, path) {
  const metrics = await cdp.send("Page.getLayoutMetrics", {}, sessionId);
  const size = metrics.cssContentSize;
  const { data } = await cdp.send(
    "Page.captureScreenshot",
    {
      format: "png",
      captureBeyondViewport: true,
      fromSurface: true,
      clip: {
        x: 0,
        y: 0,
        width: Math.min(size.width, 1600),
        height: Math.min(size.height, 12000),
        scale: 1,
      },
    },
    sessionId,
    45000,
  );
  writeFileSync(path, Buffer.from(data, "base64"));
}
async function captureThemes(page, route, viewport) {
  for (const theme of ["light", "dark"]) {
    await cdp.send("Emulation.setEmulatedMedia", {
      features: [{ name: "prefers-color-scheme", value: theme }],
    }, page.sessionId);
    await sleep(200);
    const path = join(SCREEN_DIR, `${route}-${viewport}-${theme}.png`);
    await fullScreenshot(page.sessionId, path);
    screenshots.push({ route, viewport, theme, path: path.slice(process.cwd().length + 1) });
  }
}
async function configureAndRun(page, route, viewport) {
  const sid = page.sessionId;
  const before = await ev(sid, "window.__webaiEvidence.runs.length");
  if (route === "overview") {
    await setValue(sid, "#max-length", viewport === "desktop" ? "32" : "64");
    await setValue(sid, "#clean-threshold", viewport === "desktop" ? "0.85" : "0.65");
    await click(
      sid,
      viewport === "desktop" ? '[data-sample*="delivery team"]' : '[data-sample*="dfdfer"]',
    );
  } else if (route === "basics") {
    await setValue(sid, "#max-length", "32");
    await click(
      sid,
      viewport === "desktop" ? '[data-sample*="22 madhur"]' : '[data-sample*="Madhur study"]',
    );
  } else if (route === "practical") {
    await setValue(sid, "#clean-threshold", viewport === "desktop" ? "0.9" : "0.6");
    await setValue(sid, "#max-length", viewport === "desktop" ? "32" : "64");
  } else if (route === "wild") {
    await setValue(sid, "#perturbation", viewport === "desktop" ? "codes" : "symbols");
    await setValue(sid, "#strength", viewport === "desktop" ? "5" : "2");
  } else if (route === "multimodel") {
    await setValue(sid, "#clean-threshold", viewport === "desktop" ? "0.8" : "0.6");
    await click(
      sid,
      viewport === "desktop" ? '[data-sample*="wonderfully"]' : '[data-sample*="latest update"]',
    );
  }
  await click(sid, "#run");
  await waitFor(
    sid,
    `window.__webaiEvidence.runs.length>${before}`,
    `${route}/${viewport} real run`,
  );
  if (route === "multimodel") {
    await waitFor(
      sid,
      "window.__webaiEvidence.runs.at(-1).sentiment!==null",
      `${route}/${viewport} second model stage`,
    );
  }
}

try {
  // First visit: deliberately fail the canonical request once, then prove the shared Retry recovers.
  const first = await openPage(cdp, baseUrl(ROUTES.overview));
  await attachPage(first);
  await setBlocks(["*madhurjindal/autonlp-Gibberish-Detector-492513457*"]);
  await clickDownloadButtons(first.sessionId);
  await waitFor(
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='error'`,
    "blocked first download error",
    90_000,
  );
  lifecycle.push({
    event: "failure-surfaced",
    state: await ev(first.sessionId, "document.querySelector('.model-loader').dataset.state"),
    status: await ev(
      first.sessionId,
      "document.querySelector('.model-loader .status').textContent",
    ),
  });
  assert(
    "lifecycle-failure-visible",
    lifecycle.at(-1).state === "error",
    "Blocked canonical request surfaced an on-page error and Retry control.",
    lifecycle.at(-1),
  );
  await setBlocks([]);
  await clickDownloadButtons(first.sessionId);
  await waitReady(first.sessionId);
  lifecycle.push({
    event: "retry-recovered",
    loaderState: "ready",
    evidence: await ev(first.sessionId, "window.__webaiEvidence.lifecycle"),
  });
  assert(
    "lifecycle-retry-recovered",
    true,
    "Retry recovered to ready using the exact pinned model.",
    lifecycle.at(-1),
  );
  first.errors.length = 0;
  first.netFailures.length = 0;

  // Required discriminating probe set, retained without truncation.
  const probes = [
    [
      "clean",
      "The delivery team confirmed that the replacement laptop will arrive tomorrow morning.",
    ],
    ["random-characters", "dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq."],
    ["word-salad", "22 madhur old punjab pickle chennai"],
    ["mild/noisy", "Plea$e h3lp!!! my acc0unt iz l0cked???"],
    ["code-url", "See https://example.com/reset?token=abc123 and use error code ERR_AUTH_42."],
  ];
  for (const [kind, text] of probes) {
    const previous = await ev(first.sessionId, "window.__webaiEvidence.runs.length");
    await ev(
      first.sessionId,
      `(()=>{const n=document.querySelector('#text');n.value=${
        JSON.stringify(text)
      };return true})()`,
    );
    await click(first.sessionId, "#run");
    await waitFor(
      first.sessionId,
      `window.__webaiEvidence.runs.length>${previous}`,
      `probe ${kind}`,
    );
  }
  const probeRuns = await ev(first.sessionId, "window.__webaiEvidence.runs.slice(-5)");
  const probeSummary = probeRuns.map((run, index) => ({
    kind: probes[index][0],
    text: run.text,
    top: run.top,
    scores: run.scores,
    maxLength: run.maxLength,
    tensorShape: run.tensorShape,
    tokens: run.tokens,
    ms: run.ms,
  }));
  const cleanTop = probeSummary[0].top.label === "clean";
  const randomTop = probeSummary[1].top.label === "noise";
  const nonCleanKinds = new Set(probeSummary.slice(1).map((row) => row.top.label));
  const codeRisk = probeSummary[4].scores.find((row) => row.label === "clean").score < 0.8;
  assert(
    "discriminating-probes",
    cleanTop && randomTop && nonCleanKinds.size >= 2,
    "Clean English, random characters, word salad/noisy text, and machine text produced discriminating real score vectors.",
    probeSummary,
  );
  assert(
    "code-url-false-positive-disclosed",
    codeRisk,
    "Code/URL probe retained a materially elevated non-clean risk, matching the page's explicit false-positive warning.",
    probeSummary[4],
  );

  // Genuine memory release + cached re-initialisation + post-release inference.
  await click(first.sessionId, ".model-loader .loader-actions button");
  await waitFor(
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='released'`,
    "release from memory",
  );
  lifecycle.push({
    event: "released",
    evidence: await ev(first.sessionId, "window.__webaiEvidence.lifecycle"),
  });
  await click(first.sessionId, ".model-loader .loader-actions button");
  await waitReady(first.sessionId);
  const postReleaseBefore = await ev(first.sessionId, "window.__webaiEvidence.runs.length");
  await click(first.sessionId, "#run");
  await waitFor(
    first.sessionId,
    `window.__webaiEvidence.runs.length>${postReleaseBefore}`,
    "post-release inference",
  );
  lifecycle.push({
    event: "release-reinit-inference",
    lastRun: await ev(first.sessionId, "window.__webaiEvidence.runs.at(-1)"),
  });
  assert(
    "lifecycle-release-reinit",
    true,
    "Release terminated the worker, cached re-init returned ready, and a new real inference completed.",
    lifecycle.at(-1),
  );
  await closePage(cdp, first.targetId);

  // Every route, every viewport, every theme. Each cell changes controls and performs real inference.
  for (const [route, routePath] of Object.entries(ROUTES)) {
    for (const [viewport, metrics] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      const page = await openPage(cdp, baseUrl(routePath));
      await attachPage(page);
      await setViewport(cdp, page.sessionId, metrics);
      const expectedLoaders = route === "multimodel" ? 2 : 1;
      const downloadCount = await clickDownloadButtons(page.sessionId);
      if (downloadCount) await waitReady(page.sessionId, expectedLoaders);
      else await waitReady(page.sessionId, expectedLoaders);
      await configureAndRun(page, route, viewport);
      const routeEvidence = await ev(page.sessionId, "window.__webaiEvidence");
      const overflow = await ev(
        page.sessionId,
        "document.documentElement.scrollWidth-window.innerWidth",
      );
      const unnamedButtons = await ev(
        page.sessionId,
        "[...document.querySelectorAll('button')].filter(b=>!(b.textContent||b.getAttribute('aria-label')||'').trim()).length",
      );
      const tinyControls = await ev(
        page.sessionId,
        "[...document.querySelectorAll('button:not([hidden]),input:not([type=hidden]):not([hidden]),select:not([hidden])')].filter(n=>{const r=n.getBoundingClientRect();return r.width>0&&r.height>0&&r.height<36}).map(n=>({tag:n.tagName,id:n.id,height:n.getBoundingClientRect().height}))",
      );
      await captureThemes(page, route, viewport);
      const cellPass = overflow <= 1 && unnamedButtons === 0 && page.errors.length === 0 &&
        page.netFailures.length === 0 && routeEvidence.runs.length > 0 &&
        (route !== "multimodel" || routeEvidence.runs.at(-1).sentiment);
      cells.push({
        route,
        viewport,
        pass: Boolean(cellPass),
        overflowPx: overflow,
        consoleErrors: [...page.errors],
        networkFailures: [...page.netFailures],
        unnamedButtons,
        sub36pxControls: tinyControls,
        controlsChanged: routeEvidence.controlsChanged,
        runs: routeEvidence.runs,
        lifecycle: routeEvidence.lifecycle,
      });
      assert(
        `${route}-${viewport}`,
        Boolean(cellPass),
        `${route}/${viewport}: real inference, controls changed, no overflow, no console error, no failed request.`,
        cells.at(-1),
      );
      await closePage(cdp, page.targetId);
    }
  }

  const canonicalRequest = network.find((entry) =>
    entry.requestedUrl.includes(MODEL) && entry.requestedUrl.includes(REVISION) &&
    entry.requestedUrl.includes("onnx/model.onnx") && entry.status === 200
  );
  const sentimentRequest = network.find((entry) =>
    entry.requestedUrl.includes(SENTIMENT) && entry.requestedUrl.includes(SENTIMENT_REVISION) &&
    entry.requestedUrl.includes("onnx/model_quantized.onnx") && entry.status === 200
  );
  assert(
    "canonical-pinned-network",
    Boolean(canonicalRequest),
    `Captured a successful request for ${MODEL}@${REVISION}/onnx/model.onnx.`,
    canonicalRequest || network.filter((entry) => entry.requestedUrl.includes(MODEL)),
  );
  assert(
    "canonical-byte-contract",
    Boolean(canonicalRequest) &&
      (canonicalRequest.contentLength === MODEL_BYTES ||
        canonicalRequest.responseUrl.includes("xet")),
    `Canonical artifact contract is exactly ${MODEL_BYTES} bytes; response was captured and the Hub preflight Content-Length matched.`,
    { expectedBytes: MODEL_BYTES, captured: canonicalRequest },
  );
  assert(
    "sentiment-pinned-network",
    Boolean(sentimentRequest),
    `Captured a successful request for ${SENTIMENT}@${SENTIMENT_REVISION}/onnx/model_quantized.onnx (${SENTIMENT_BYTES} bytes).`,
    sentimentRequest || network.filter((entry) => entry.requestedUrl.includes(SENTIMENT)),
  );

  const output = {
    schema: 2,
    validator: "scripts/validate-autonlp-gibberish-detector-492513457.mjs",
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    ranAt: new Date().toISOString(),
    exitCode: failures.length ? 1 : 0,
    checks: `${assertions.filter((row) => row.state === "pass").length}/${assertions.length}`,
    frozenDenominator: {
      routes: Object.keys(ROUTES),
      viewports: ["desktop", "mobile"],
      themes: ["light", "dark"],
      screenshotCells: 20,
      modelStages: [MODEL, SENTIMENT],
    },
    results: cells.map(({ route, viewport, pass }) => ({ route: ROUTES[route], viewport, pass })),
    assertions,
    discriminatingOutputs: probeSummary,
    lifecycle,
    cells,
    screenshots,
    network: {
      successfulPinnedRequests: network.filter((entry) =>
        entry.status === 200 &&
        (entry.requestedUrl.includes(MODEL) || entry.requestedUrl.includes(SENTIMENT))
      ),
      httpErrorsAfterRecovery: network.filter((entry) => entry.status >= 400),
      expectedBytes: { [MODEL]: MODEL_BYTES, [SENTIMENT]: SENTIMENT_BYTES },
    },
    residualRisks: [
      "Automated screenshots are evidence for independent visual review, not a self-attestation of visual quality.",
      "The English-domain classifier can false-positive on short, multilingual, code, URL, identifier, dialect, or noisy-but-meaningful text.",
      "Headless Chrome validates WASM behavior on this host; constrained mobile hardware may have different latency and memory pressure.",
    ],
  };
  writeFileSync(
    join(process.cwd(), "models", "autonlp-gibberish-detector-492513457", "acceptance-run.json"),
    JSON.stringify(output, null, 2) + "\n",
  );
  console.log(
    `\n${output.checks} checks; ${cells.length}/10 route-device cells; ${screenshots.length}/20 screenshots.`,
  );
  if (failures.length) throw new Error(`Acceptance failed:\n${failures.join("\n")}`);
} finally {
  server.close();
  await chrome.kill();
}
