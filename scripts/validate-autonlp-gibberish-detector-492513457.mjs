// Fail-closed, route-complete browser acceptance for the canonical English gibberish portfolio.
// Every route control is changed from its rendered value at desktop + mobile, every sample control
// is activated, the primary action is keyboard-operated, and Multi-model proves both pass and stop.
// The shared loader lifecycle is exercised where the immutable WASM route supports a state; states
// that cannot honestly exist (mutable Update / feature-gated unsupported) are retained as not-applicable.
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
const MOBILE_TARGET_PX = 44;
const ROUTES = {
  overview: "models/autonlp-gibberish-detector-492513457/",
  basics: "models/autonlp-gibberish-detector-492513457/basics/",
  practical: "models/autonlp-gibberish-detector-492513457/practical/",
  wild: "models/autonlp-gibberish-detector-492513457/wild/",
  multimodel: "models/autonlp-gibberish-detector-492513457/multi-model/",
};
const CONTROL_PLAN = {
  overview: { text: "#text", values: { "#max-length": "32", "#clean-threshold": "0.85", "#allow-mild": true }, samples: 5 },
  basics: { text: "#text", values: { "#max-length": "32", "#clean-threshold": "0.85", "#allow-mild": true }, samples: 5 },
  practical: { text: "#text-lines", values: { "#max-length": "32", "#clean-threshold": "0.85", "#allow-mild": true }, samples: 0 },
  wild: { text: "#text", values: { "#perturbation": "codes", "#strength": "5", "#max-length": "32" }, samples: 0 },
  multimodel: { text: "#text", values: { "#max-length": "32", "#clean-threshold": "0.85", "#allow-mild": true }, samples: 4 },
};
const RESULT_SELECTORS = {
  overview: "#scores .score-row",
  basics: "#scores .score-row",
  practical: "#queue-body tr",
  wild: "#wild-results:not([hidden])",
  multimodel: "#gate-output:not(:empty)",
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
  assertions.push({ id, state: ok ? "pass" : "fail", evidence, ...(details === undefined ? {} : { details }) });
  console.log(`${ok ? "PASS" : "FAIL"} ${id} — ${evidence}`);
  if (!ok) failures.push(`${id}: ${evidence}`);
}
async function ev(sessionId, expression, timeoutMs = 45000) {
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.stack||error)}}})()`,
    awaitPromise: true,
    returnByValue: true,
  }, sessionId, timeoutMs);
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
  await Promise.all([...trackedSessions].map((sessionId) => cdp.send("Network.setBlockedURLs", {
    urls: patterns.length ? patterns : ["__never_match__"],
  }, sessionId).catch(() => {})));
}
cdp.on((message) => {
  if (message.method === "Target.attachedToTarget") {
    const info = message.params.targetInfo;
    if ((info?.type === "worker" || info?.type === "service_worker") &&
      info.url.includes("autonlp-gibberish-detector-492513457")) void enableNetwork(message.params.sessionId);
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
await cdp.send("Target.setAutoAttach", { autoAttach: true, flatten: true, waitForDebuggerOnStart: false });

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
  return await ev(sessionId, `(()=>{const node=document.querySelector(${JSON.stringify(selector)});if(!node)return false;node.click();return true})()`);
}
async function setValue(sessionId, selector, value) {
  return await ev(sessionId, `(()=>{const node=document.querySelector(${JSON.stringify(selector)});if(!node)return false;if(node.type==='checkbox')node.checked=${Boolean(value)};else node.value=${JSON.stringify(String(value))};node.dispatchEvent(new Event('input',{bubbles:true}));node.dispatchEvent(new Event('change',{bubbles:true}));return true})()`);
}
async function pressKey(sessionId, key, code, keyCode) {
  // CDP only performs the browser's native default action when keyDown carries the key's text.
  // This is a trusted input event (unlike dispatchEvent(new KeyboardEvent(...))), so Enter/Space
  // genuinely activate the focused native button through its keyboard behavior.
  const text = key === "Enter" ? "\r" : key;
  await cdp.send("Input.dispatchKeyEvent", { type: "keyDown", key, code, text, unmodifiedText: text, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode }, sessionId);
  await cdp.send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode }, sessionId);
}
async function keyboardActivate(sessionId, selector, key = "Enter") {
  const focused = await ev(sessionId, `(()=>{const node=document.querySelector(${JSON.stringify(selector)});if(!node||node.disabled)return false;node.focus();return document.activeElement===node})()`);
  if (!focused) return false;
  if (key === "Space") await pressKey(sessionId, " ", "Space", 32);
  else await pressKey(sessionId, "Enter", "Enter", 13);
  return true;
}
async function clickDownloadButtons(sessionId) {
  return await ev(sessionId, `(()=>{let count=0;for(const button of document.querySelectorAll('.model-loader button')){if(/Download|Retry/i.test(button.textContent)){button.click();count++}}return count})()`);
}
async function waitReady(sessionId, expected = 1) {
  await waitFor(sessionId, `document.querySelectorAll('.model-loader[data-state="ready"]').length===${expected}`, `${expected} model loader(s) ready`);
}
async function fullScreenshot(sessionId, path) {
  const metrics = await cdp.send("Page.getLayoutMetrics", {}, sessionId);
  const size = metrics.cssContentSize;
  const { data } = await cdp.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    fromSurface: true,
    clip: { x: 0, y: 0, width: Math.min(size.width, 1600), height: Math.min(size.height, 12000), scale: 1 },
  }, sessionId, 45000);
  writeFileSync(path, Buffer.from(data, "base64"));
}
async function materializeContent(sessionId, route) {
  return await ev(sessionId, `(async()=>{
    let style=document.querySelector('#acceptance-content-visibility');
    if(!style){style=document.createElement('style');style.id='acceptance-content-visibility';style.textContent='.model-card{content-visibility:visible!important;contain-intrinsic-size:auto!important}';document.head.append(style)}
    const regions=[...document.querySelectorAll('header,aside,main>section,main .model-card,footer')];
    for(const region of regions){region.scrollIntoView({block:'center'});await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))}
    scrollTo(0,0);await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
    const cards=[...document.querySelectorAll('.model-card')].map(node=>({text:node.innerText.trim(),width:node.getBoundingClientRect().width,height:node.getBoundingClientRect().height}));
    const sections=[...document.querySelectorAll('main>section')].map(node=>({id:node.getAttribute('aria-labelledby'),textLength:node.innerText.trim().length,height:node.getBoundingClientRect().height}));
    const resultCount=document.querySelectorAll(${JSON.stringify(RESULT_SELECTORS[route])}).length;
    return {route,resultCount,cards,sections,allCardsPainted:cards.every(x=>x.text&&x.width>0&&x.height>0),allSectionsPainted:sections.every(x=>x.textLength>0&&x.height>0)};
  })()`);
}
async function captureThemes(page, route, viewport) {
  const captures = [];
  for (const theme of ["light", "dark"]) {
    await cdp.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: theme }] }, page.sessionId);
    await sleep(200);
    const painted = await materializeContent(page.sessionId, route);
    const path = join(SCREEN_DIR, `${route}-${viewport}-${theme}.png`);
    await fullScreenshot(page.sessionId, path);
    const record = { route, viewport, theme, path: path.slice(process.cwd().length + 1), painted };
    screenshots.push(record);
    captures.push(record);
  }
  return captures;
}
async function waitForNewRun(sid, before, label) {
  await waitFor(sid, `window.__webaiEvidence.runs.length>${before}`, label);
  return await ev(sid, "window.__webaiEvidence.runs.at(-1)");
}
async function runByKeyboard(sid, label) {
  const before = await ev(sid, "window.__webaiEvidence.runs.length");
  const operated = await keyboardActivate(sid, "#run", "Enter");
  const run = operated ? await waitForNewRun(sid, before, label) : null;
  return { operated, run };
}
async function configureAndRun(page, route, viewport) {
  const sid = page.sessionId;
  const plan = CONTROL_PLAN[route];
  const initial = await ev(sid, `(()=>Object.fromEntries([...document.querySelectorAll('.controls textarea,.controls input:not([type=hidden]),.controls select')].map(node=>['#'+node.id,node.type==='checkbox'?node.checked:node.value])))()`);
  const sampleAudit = [];
  const samples = await ev(sid, "document.querySelectorAll('.controls [data-sample]').length");
  for (let index = 0; index < samples; index++) {
    const seed = `acceptance seed ${route} ${viewport} ${index}`;
    await setValue(sid, plan.text, seed);
    const selector = `.controls [data-sample]:nth-of-type(${index + 1})`;
    let activated;
    if (index === 0) activated = await keyboardActivate(sid, selector, "Space");
    else activated = await click(sid, selector);
    const observed = await ev(sid, `document.querySelector(${JSON.stringify(plan.text)}).value`);
    const expected = await ev(sid, `document.querySelector(${JSON.stringify(selector)})?.dataset.sample`);
    sampleAudit.push({ index, activated, inputChanged: observed !== seed && observed === expected, expected, observed });
  }
  const textValue = route === "practical"
    ? `A carefully edited ${viewport} queue item.\ndfdfer qsqskdsd acceptance noise.\nPlease confirm tomorrow's delivery appointment.`
    : route === "wild"
    ? `Please review the deliberately edited ${viewport} account recovery message.`
    : `The deliberately edited ${viewport} message is clear, complete, and useful.`;
  await setValue(sid, plan.text, textValue);
  for (const [selector, value] of Object.entries(plan.values)) await setValue(sid, selector, value);

  const changed = await ev(sid, `(()=>Object.fromEntries([...document.querySelectorAll('.controls textarea,.controls input:not([type=hidden]),.controls select')].map(node=>['#'+node.id,node.type==='checkbox'?node.checked:node.value])))()`);
  const expectedSelectors = [plan.text, ...Object.keys(plan.values)];
  const changedValues = expectedSelectors.map((selector) => ({ selector, before: initial[selector], after: changed[selector], changed: initial[selector] !== changed[selector] }));
  const focusAudit = await ev(sid, `(()=>{const nodes=[...document.querySelectorAll('.controls button:not([disabled]),.controls textarea,.controls input:not([type=hidden]):not([disabled]),.controls select:not([disabled])')].filter(n=>n.getClientRects().length);const rows=[];for(const node of nodes){node.focus();rows.push({tag:node.tagName,id:node.id||null,label:node.textContent?.trim()||null,focused:document.activeElement===node})}return rows})()`);

  let keyboard;
  let gateOutcomes = null;
  if (route === "multimodel") {
    await setValue(sid, "#text", `The ${viewport} replacement arrived early and support was wonderfully helpful.`);
    keyboard = await runByKeyboard(sid, `${route}/${viewport} keyboard gate pass`);
    const pass = keyboard.run;
    await setValue(sid, "#text", "dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq.");
    const stopBefore = await ev(sid, "window.__webaiEvidence.runs.length");
    await click(sid, "#run");
    const stop = await waitForNewRun(sid, stopBefore, `${route}/${viewport} gate stop`);
    await setValue(sid, "#text", `The final ${viewport} result is clear and wonderfully helpful.`);
    const finalBefore = await ev(sid, "window.__webaiEvidence.runs.length");
    await click(sid, "#run");
    const final = await waitForNewRun(sid, finalBefore, `${route}/${viewport} final populated result`);
    gateOutcomes = {
      pass: Boolean(pass?.decision?.accepted && pass.sentiment),
      stop: Boolean(stop?.decision?.accepted === false && stop.sentiment === null),
      finalPopulated: Boolean(final?.decision?.accepted && final.sentiment),
      passRun: pass,
      stopRun: stop,
    };
  } else {
    keyboard = await runByKeyboard(sid, `${route}/${viewport} keyboard run`);
  }
  const routeEvidence = await ev(sid, "window.__webaiEvidence");
  const evidenceControls = new Set(routeEvidence.controlsChanged.map((row) => row.control));
  const expectedEvidence = expectedSelectors.map((selector) => selector.slice(1));
  return {
    initial,
    final: changed,
    expectedSelectors,
    changedValues,
    sampleAudit,
    sampleDenominator: plan.samples,
    samplesFound: samples,
    focusAudit,
    keyboardRun: Boolean(keyboard?.operated && keyboard?.run),
    gateOutcomes,
    evidenceHasEveryValueControl: expectedEvidence.every((id) => evidenceControls.has(id)),
    routeEvidence,
  };
}
async function targetAudit(sid) {
  return await ev(sid, `(()=>[...document.querySelectorAll('.controls button,.controls textarea,.controls input:not([type=hidden]),.controls select,.model-loader button')].filter(n=>n.getClientRects().length).map(node=>{const target=node.type==='checkbox'?(node.closest('label')?.querySelector('span:last-child')||node):node;const r=target.getBoundingClientRect();return {tag:node.tagName,id:node.id||null,label:node.textContent?.trim()||null,effectiveTarget:target===node?'control':'associated label',width:Number(r.width.toFixed(2)),height:Number(r.height.toFixed(2)),pass:r.width>=${MOBILE_TARGET_PX}&&r.height>=${MOBILE_TARGET_PX}}}))()`);
}

try {
  // First visit: absent → deliberate fetch failure → visible Retry → exact pinned recovery.
  const first = await openPage(cdp, baseUrl(ROUTES.overview));
  await attachPage(first);
  await waitFor(first.sessionId, `document.querySelector('.model-loader')?.dataset.state==='download-required'`, "first visit absent");
  lifecycle.push({ event: "first-visit", state: "download-required", control: "Download model" });
  await setBlocks(["*madhurjindal/autonlp-Gibberish-Detector-492513457*"]);
  await clickDownloadButtons(first.sessionId);
  await waitFor(first.sessionId, `document.querySelector('.model-loader')?.dataset.state==='error'`, "blocked first download error", 90_000);
  lifecycle.push({ event: "failure-surfaced", state: "error", status: await ev(first.sessionId, "document.querySelector('.model-loader .status').textContent"), retry: await ev(first.sessionId, "/Retry/.test(document.querySelector('.model-loader button')?.textContent||'')") });
  assert("lifecycle-failure-visible", lifecycle.at(-1).retry === true, "Absent first download failed visibly and exposed Retry.", lifecycle.at(-1));
  await setBlocks([]);
  await clickDownloadButtons(first.sessionId);
  await waitReady(first.sessionId);
  lifecycle.push({ event: "retry-recovered", state: "ready", evidence: await ev(first.sessionId, "window.__webaiEvidence.lifecycle") });
  first.errors.length = 0;
  first.netFailures.length = 0;

  const probes = [
    ["clean", "The delivery team confirmed that the replacement laptop will arrive tomorrow morning."],
    ["random-characters", "dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq."],
    ["word-salad", "22 madhur old punjab pickle chennai"],
    ["mild/noisy", "Plea$e h3lp!!! my acc0unt iz l0cked???"],
    ["code-url", "See https://example.com/reset?token=abc123 and use error code ERR_AUTH_42."],
  ];
  for (const [kind, text] of probes) {
    const previous = await ev(first.sessionId, "window.__webaiEvidence.runs.length");
    await setValue(first.sessionId, "#text", text);
    await click(first.sessionId, "#run");
    await waitForNewRun(first.sessionId, previous, `probe ${kind}`);
  }
  const probeRuns = await ev(first.sessionId, "window.__webaiEvidence.runs.slice(-5)");
  const probeSummary = probeRuns.map((run, index) => ({ kind: probes[index][0], text: run.text, top: run.top, scores: run.scores, maxLength: run.maxLength, tensorShape: run.tensorShape, tokens: run.tokens, ms: run.ms }));
  const cleanTop = probeSummary[0].top.label === "clean";
  const randomTop = probeSummary[1].top.label === "noise";
  const nonCleanKinds = new Set(probeSummary.slice(1).map((row) => row.top.label));
  const codeRisk = probeSummary[4].scores.find((row) => row.label === "clean").score < 0.8;
  assert("discriminating-probes", cleanTop && randomTop && nonCleanKinds.size >= 2, "Five retained real score vectors discriminate clean, random, word-salad/noisy, and machine text.", probeSummary);
  assert("code-url-false-positive-disclosed", codeRisk, "Code/URL retained materially elevated non-clean risk matching the warning.", probeSummary[4]);

  // Release, then block every remote Hub/CDN URL: cached re-init and inference must still work.
  await click(first.sessionId, ".model-loader .loader-actions button");
  await waitFor(first.sessionId, `document.querySelector('.model-loader')?.dataset.state==='released'`, "release before offline cached use");
  await setBlocks(["https://huggingface.co/*", "https://*.hf.co/*", "https://*.xethub.hf.co/*"]);
  await click(first.sessionId, ".model-loader .loader-actions button");
  await waitReady(first.sessionId);
  const offlineBefore = await ev(first.sessionId, "window.__webaiEvidence.runs.length");
  await click(first.sessionId, "#run");
  const offlineRun = await waitForNewRun(first.sessionId, offlineBefore, "offline cached inference");
  lifecycle.push({ event: "offline-cached-use", state: "ready", remoteUrlsBlocked: true, realInference: Boolean(offlineRun?.tensorShape?.join() === "1,4"), networkFailures: [...first.netFailures] });
  await setBlocks([]);

  // Cache Storage entries are atomic, so partial/corrupt is honestly represented by deleting one
  // recorded response (browser eviction), then requiring the shared Re-download state and recovery.
  await waitFor(first.sessionId, `(async()=>{const {listValidationRecords}=await import('/web-ai-showcase/lib/model-cache.js');return (await listValidationRecords()).some(r=>r.modelId===${JSON.stringify(MODEL)}&&r.files?.length)})()`, "validation record with cached files", 120_000);
  // recordValidated() is intentionally background work; let the offline re-init record settle before
  // deleting a response so a late metadata write cannot race the eviction simulation.
  await sleep(6000);
  const eviction = await ev(first.sessionId, `(async()=>{const cacheApi=await import('/web-ai-showcase/lib/model-cache.js');const record=(await cacheApi.listValidationRecords()).find(r=>r.modelId===${JSON.stringify(MODEL)});const url=record?.files?.find(u=>!u.includes('/onnx/model.onnx'))||record?.files?.[0];let removed=false,cacheName=null;for(const name of await caches.keys()){const cache=await caches.open(name);if(url&&await cache.delete(url)){removed=true;cacheName=name;break}}const inspected=record?await cacheApi.inspectModel({key:record.key,timeoutMs:2000}):null;return {removed,url,cacheName,recordedFiles:record?.files?.length||0,inspectedState:inspected?.state||null,missing:inspected?.missing?.length||0}})()`);
  lifecycle.push({ event: "eviction", state: "recorded-file-deleted", ...eviction });
  await closePage(cdp, first.targetId);

  const partial = await openPage(cdp, baseUrl(ROUTES.overview));
  await attachPage(partial);
  await waitFor(partial.sessionId, `document.querySelector('.model-loader')?.dataset.state==='partial'`, "eviction produces partial state");
  const partialState = { event: "partial-or-corrupt", state: "partial", action: await ev(partial.sessionId, "document.querySelector('.model-loader button')?.textContent"), corruptRepresentation: "not separately representable: Cache Storage response writes are atomic; a corrupt/incomplete local set is fail-closed as missing/evicted (partial)" };
  lifecycle.push(partialState);
  await clickDownloadButtons(partial.sessionId);
  await waitReady(partial.sessionId);
  lifecycle.push({ event: "redownload-recovered", state: "ready", realMissingAssetFetch: true });
  await closePage(cdp, partial.targetId);

  const current = await openPage(cdp, baseUrl(ROUTES.overview));
  await attachPage(current);
  await waitReady(current.sessionId);
  const noDownloadNeeded = await ev(current.sessionId, "![...document.querySelectorAll('.model-loader button')].some(b=>/^Download|Re-download/.test(b.textContent))");
  lifecycle.push({ event: "current-cached-auto-init", state: "ready", automatic: noDownloadNeeded === true });
  await closePage(cdp, current.targetId);
  lifecycle.push({ event: "stale-update", state: "not-applicable", reason: `This family deliberately pins immutable revision ${REVISION}; mutable latest/update discovery is disabled, so claiming an Update state would be false.` });
  lifecycle.push({ event: "unsupported-device", state: "not-applicable", reason: "The fp32 WASM route has no WebGPU or optional-device-feature requirement in supported Chrome; desktop and mobile are exercised instead of fabricating an unsupported state." });

  // Every route × viewport changes every value control, activates every sample, runs via keyboard,
  // and captures both themes only after all content-visibility regions have been painted.
  for (const [route, routePath] of Object.entries(ROUTES)) {
    for (const [viewport, metrics] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      const page = await openPage(cdp, baseUrl(routePath));
      await attachPage(page);
      await setViewport(cdp, page.sessionId, metrics);
      const expectedLoaders = route === "multimodel" ? 2 : 1;
      await clickDownloadButtons(page.sessionId);
      await waitReady(page.sessionId, expectedLoaders);
      const controls = await configureAndRun(page, route, viewport);
      const overflow = await ev(page.sessionId, "document.documentElement.scrollWidth-window.innerWidth");
      const unnamedButtons = await ev(page.sessionId, "[...document.querySelectorAll('button')].filter(b=>!(b.textContent||b.getAttribute('aria-label')||'').trim()).length");
      const targets = await targetAudit(page.sessionId);
      const captures = await captureThemes(page, route, viewport);
      const allValuesChanged = controls.changedValues.length === controls.expectedSelectors.length && controls.changedValues.every((item) => item.changed);
      const allSamplesDriven = controls.samplesFound === controls.sampleDenominator && controls.sampleAudit.length === controls.sampleDenominator && controls.sampleAudit.every((item) => item.activated && item.inputChanged);
      const keyboardOperable = controls.keyboardRun && controls.focusAudit.length > 0 && controls.focusAudit.every((item) => item.focused);
      const gateComplete = route !== "multimodel" || (controls.gateOutcomes?.pass && controls.gateOutcomes?.stop && controls.gateOutcomes?.finalPopulated);
      const touchTargetsPass = viewport !== "mobile" || targets.every((target) => target.pass);
      const painted = captures.length === 2 && captures.every((capture) => capture.painted.resultCount > 0 && capture.painted.allCardsPainted && capture.painted.allSectionsPainted);
      const routeEvidence = controls.routeEvidence;
      const cellPass = overflow <= 1 && unnamedButtons === 0 && page.errors.length === 0 && page.netFailures.length === 0 && routeEvidence.runs.length > 0 && allValuesChanged && allSamplesDriven && controls.evidenceHasEveryValueControl && keyboardOperable && gateComplete && touchTargetsPass && painted;
      cells.push({
        route,
        viewport,
        pass: Boolean(cellPass),
        overflowPx: overflow,
        consoleErrors: [...page.errors],
        networkFailures: [...page.netFailures],
        unnamedButtons,
        touchTargetThresholdPx: MOBILE_TARGET_PX,
        touchTargets: targets,
        controlAudit: controls,
        screenshotPaintAudit: captures.map((capture) => ({ theme: capture.theme, painted: capture.painted })),
        runs: routeEvidence.runs,
        lifecycle: routeEvidence.lifecycle,
      });
      assert(`${route}-${viewport}`, Boolean(cellPass), `${route}/${viewport}: every visible route control changed/driven, keyboard run, real inference, ${route === "multimodel" ? "pass+stop gate, " : ""}${viewport === "mobile" ? `all effective targets ≥${MOBILE_TARGET_PX}px, ` : ""}all sections painted in both themes, no overflow/error/failed request.`, cells.at(-1));
      await closePage(cdp, page.targetId);
    }
  }

  // Finish in a genuine cleared state and verify both validation metadata and model cache entries are gone.
  const finalPage = await openPage(cdp, baseUrl(ROUTES.overview));
  await attachPage(finalPage);
  await waitReady(finalPage.sessionId);
  await click(finalPage.sessionId, ".model-loader button:nth-child(2)");
  await waitFor(finalPage.sessionId, `document.querySelector('.model-loader')?.dataset.state==='download-required'`, "final cache clear");
  const finalCache = await ev(finalPage.sessionId, `(async()=>{const {listValidationRecords,scanCachedFiles}=await import('/web-ai-showcase/lib/model-cache.js');const records=(await listValidationRecords()).filter(r=>r.modelId===${JSON.stringify(MODEL)}).length;const files=(await scanCachedFiles(${JSON.stringify(MODEL)})).length;return {records,files,state:document.querySelector('.model-loader').dataset.state,downloadShown:/Download/.test(document.querySelector('.model-loader button')?.textContent||'')}})()`);
  lifecycle.push({ event: "final-cache-clear", ...finalCache });
  await closePage(cdp, finalPage.targetId);

  const lifecycleOk = lifecycle.some((row) => row.event === "first-visit" && row.state === "download-required") &&
    lifecycle.some((row) => row.event === "failure-surfaced" && row.retry) &&
    lifecycle.some((row) => row.event === "retry-recovered" && row.state === "ready") &&
    lifecycle.some((row) => row.event === "offline-cached-use" && row.realInference) &&
    lifecycle.some((row) => row.event === "eviction" && row.removed) &&
    lifecycle.some((row) => row.event === "partial-or-corrupt" && row.state === "partial" && /Re-download/.test(row.action || "")) &&
    lifecycle.some((row) => row.event === "redownload-recovered" && row.state === "ready") &&
    lifecycle.some((row) => row.event === "current-cached-auto-init" && row.automatic) &&
    lifecycle.some((row) => row.event === "stale-update" && row.state === "not-applicable") &&
    lifecycle.some((row) => row.event === "unsupported-device" && row.state === "not-applicable") &&
    lifecycle.some((row) => row.event === "final-cache-clear" && row.records === 0 && row.files === 0 && row.downloadShown);
  assert("loader-cache-lifecycle-matrix", lifecycleOk, "First/failure/Retry/current/offline/eviction-partial/Re-download/final-clear were exercised; immutable-update and unsupported-device states are precisely retained as not-applicable.", lifecycle);

  const canonicalRequest = network.find((entry) => entry.requestedUrl.includes(MODEL) && entry.requestedUrl.includes(REVISION) && entry.requestedUrl.includes("onnx/model.onnx") && entry.status === 200 && entry.contentLength === MODEL_BYTES);
  const sentimentRequest = network.find((entry) => entry.requestedUrl.includes(SENTIMENT) && entry.requestedUrl.includes(SENTIMENT_REVISION) && entry.requestedUrl.includes("onnx/model_quantized.onnx") && entry.status === 200 && entry.contentLength === SENTIMENT_BYTES);
  assert("canonical-pinned-network-and-bytes", Boolean(canonicalRequest), `Fail-closed capture requires exact ${MODEL}@${REVISION}/onnx/model.onnx and Content-Length ${MODEL_BYTES}; redirects/Xet do not bypass the byte equality.`, canonicalRequest || network.filter((entry) => entry.requestedUrl.includes(MODEL)));
  assert("sentiment-pinned-network-and-bytes", Boolean(sentimentRequest), `Fail-closed capture requires exact ${SENTIMENT}@${SENTIMENT_REVISION}/onnx/model_quantized.onnx and Content-Length ${SENTIMENT_BYTES}.`, sentimentRequest || network.filter((entry) => entry.requestedUrl.includes(SENTIMENT)));

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
      routeDeviceCells: 10,
      mobileEffectiveTargetPx: MOBILE_TARGET_PX,
      modelStages: [MODEL, SENTIMENT],
      lifecycleStates: ["first-visit", "failure", "Retry", "current cached", "offline cached", "eviction→partial", "Re-download", "final clear", "immutable Update: not-applicable", "feature unsupported: not-applicable"],
    },
    results: cells.map(({ route, viewport, pass }) => ({ route: ROUTES[route], viewport, pass })),
    assertions,
    discriminatingOutputs: probeSummary,
    lifecycle,
    cells,
    screenshots,
    network: {
      successfulPinnedRequests: network.filter((entry) => entry.status === 200 && (entry.requestedUrl.includes(MODEL) || entry.requestedUrl.includes(SENTIMENT))),
      httpErrorsAfterRecovery: network.filter((entry) => entry.status >= 400),
      expectedBytes: { [MODEL]: MODEL_BYTES, [SENTIMENT]: SENTIMENT_BYTES },
      byteChecksFailClosed: true,
    },
    residualRisks: [
      "Automated populated screenshots are evidence for independent visual review, not a self-attestation of visual quality.",
      "The English-domain classifier can false-positive on short, multilingual, code, URL, identifier, dialect, or noisy-but-meaningful text.",
      "Headless Chrome validates WASM behavior on this host; constrained physical mobile hardware may have different latency and memory pressure.",
      "A mutable stale-Update state does not exist for this intentionally immutable revision; unsupported-device UI does not exist because this WASM path has no optional feature gate.",
    ],
  };
  writeFileSync(join(process.cwd(), "models", "autonlp-gibberish-detector-492513457", "acceptance-run.json"), JSON.stringify(output, null, 2) + "\n");
  console.log(`\n${output.checks} checks; ${cells.filter((cell) => cell.pass).length}/10 route-device cells; ${screenshots.length}/20 populated screenshots.`);
  if (failures.length) throw new Error(`Acceptance failed:\n${failures.join("\n")}`);
} finally {
  server.close();
  await chrome.kill();
}
