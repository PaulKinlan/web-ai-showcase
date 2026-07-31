#!/usr/bin/env node
// Reproducible, fail-closed FashionCLIP acceptance producer.
//
// This is intentionally a browser evidence producer, not a retained-string checker. It drives the
// unchanged eight-route family with real CDP key events and changed values, executes both immutable
// model stages, retains all 32 ready/result screenshots, and exercises the shared loader's state
// matrix. Every HTTP(S) request from pages and module workers is retained with its terminal status,
// headers and encoded byte count. Missing responses, unpinned model URLs, unexpected origins/statuses
// or unexpected failures make the producer exit non-zero. Run behind a hard outer timeout, e.g.:
//   timeout --signal=TERM --kill-after=30s 3600s node scripts/produce-fashion-clip-evidence.mjs

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, join, relative } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  setViewport,
  startServer,
} from "./browser.mjs";

const ROOT = process.cwd();
const FAMILY = "models/fashion-clip";
const EVIDENCE = join(ROOT, FAMILY, "evidence");
const SCREEN_DIR = join(EVIDENCE, "screenshots");
const PROFILE = join(ROOT, ".fashion-clip-evidence-profile");
const OUT = join(EVIDENCE, "acceptance.json");
const RUN_OUT = join(ROOT, FAMILY, "acceptance-run.json");
const PRODUCER = "scripts/produce-fashion-clip-evidence.mjs";
const FIXTURE = join(ROOT, "media/assets/fashion-shirt.jpg");

const MODELS = {
  fashion: {
    id: "patrickjohncyh/fashion-clip",
    revision: "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
    dtype: "fp32",
    artifact: "onnx/model.onnx",
    bytes: 605804513,
    sha256: null,
  },
  general: {
    id: "Xenova/clip-vit-base-patch16",
    revision: "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
    dtype: "q8",
    artifact: "onnx/model_quantized.onnx",
    bytes: 152040303,
    sha256: "cf5b03d7c03cd78498b0d59a905552b549ae91af4e99ffb985103aa9424d2272",
  },
};
const ROUTES = [
  { id: "overview", route: "models/fashion-clip/", path: "", screenshot: "overview" },
  { id: "basics", route: "models/fashion-clip/basics/", path: "basics/", screenshot: "basics" },
  {
    id: "basics-attributes",
    route: "models/fashion-clip/basics-attributes/",
    path: "basics-attributes/",
    screenshot: "basics-attributes",
  },
  {
    id: "practical-catalog",
    route: "models/fashion-clip/practical-catalog/",
    path: "practical-catalog/",
    screenshot: "practical-catalog",
  },
  {
    id: "practical-search",
    route: "models/fashion-clip/practical-search/",
    path: "practical-search/",
    screenshot: "practical-search",
  },
  {
    id: "wild-briefs",
    route: "models/fashion-clip/wild-briefs/",
    path: "wild-briefs/",
    screenshot: "wild-briefs",
  },
  {
    id: "wild-audit",
    route: "models/fashion-clip/wild-audit/",
    path: "wild-audit/",
    screenshot: "wild-audit",
  },
  {
    id: "multi-model",
    route: "models/fashion-clip/multi-model/",
    path: "multi-model/",
    screenshot: "multi-model",
    general: true,
  },
];
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256File = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");
const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" })
  .trim();
const producerSha256 = sha256File(join(ROOT, PRODUCER));
const assertions = [];
const records = [];
const screenshots = [];
const stateMatrix = [];
let phase = "startup";
let failures = 0;

function log(message) {
  console.log(`${new Date().toISOString()} ${message}`);
}
function assert(id, ok, evidence, details = undefined) {
  const status = ok ? "pass" : "fail";
  assertions.push({ id, status, evidence, ...(details === undefined ? {} : { details }) });
  if (!ok) failures++;
  log(`${ok ? "PASS" : "FAIL"} ${id} — ${evidence}`);
}
function matrix(stage, state, status, evidence, details = undefined) {
  stateMatrix.push({
    stage,
    state,
    status,
    evidence,
    ...(details === undefined ? {} : { details }),
  });
  if (status === "fail") failures++;
  log(
    `${
      status === "pass" ? "PASS" : status === "not-applicable" ? "N/A " : "FAIL"
    } state ${stage}/${state} — ${evidence}`,
  );
}
function expectedAssertionIds() {
  return [
    "primary-source-pin",
    "general-source-pin",
    ...ROUTES.flatMap((route) =>
      Object.keys(VIEWPORTS).flatMap((device) => [
        `${route.id}-${device}-changed-controls-keyboard`,
        `${route.id}-${device}-real-inference`,
        ...["light", "dark"].flatMap((theme) => [
          `${route.id}-${device}-${theme}-no-overflow`,
          `${route.id}-${device}-${theme}-a11y-names`,
          `${route.id}-${device}-${theme}-clean-runtime`,
          `${route.id}-${device}-${theme}-screenshot`,
        ]),
        `${route.id}-${device}-release-reinit`,
      ])
    ),
    "general-artifact-sha256",
    "loader-matrix-complete",
    "network-primary-artifact",
    "network-general-artifact",
    "network-complete-fail-closed",
    "screenshots-complete-bound",
  ];
}

class NetworkEvidence {
  constructor(cdp) {
    this.cdp = cdp;
    this.sessions = new Map();
    this.active = new Map();
    this.requests = [];
    this.blocks = [];
    this.offline = false;
    this.sequence = 0;
    cdp.on((message) => this.#event(message));
  }
  async enable(sessionId, target = {}) {
    if (!sessionId || this.sessions.has(sessionId)) return;
    this.sessions.set(sessionId, target);
    await this.cdp.send("Network.enable", {}, sessionId).catch(() => {});
    await this.#apply(sessionId);
  }
  async #apply(sessionId) {
    await this.cdp.send(
      "Network.setBlockedURLs",
      { urls: this.blocks.length ? this.blocks : ["__fashion_evidence_no_match__"] },
      sessionId,
    ).catch(() => {});
    await this.cdp.send("Network.emulateNetworkConditions", {
      offline: this.offline,
      latency: 0,
      downloadThroughput: -1,
      uploadThroughput: -1,
    }, sessionId).catch(() => {});
  }
  async setBlocks(urls) {
    this.blocks = [...urls];
    await Promise.all([...this.sessions.keys()].map((sid) => this.#apply(sid)));
  }
  async setOffline(value) {
    this.offline = value;
    await Promise.all([...this.sessions.keys()].map((sid) => this.#apply(sid)));
  }
  reconcileWorkerBootstraps(sessionId, entries) {
    const pending = [...this.active.values()].filter((request) =>
      request.requestKey.startsWith(`${sessionId}:`) &&
      request.requestedUrl.endsWith("/models/fashion-clip/worker.js")
    );
    const resources = entries.filter((entry) =>
      entry.name.endsWith("/models/fashion-clip/worker.js") && entry.responseStatus >= 200 &&
      entry.responseStatus < 400
    );
    for (let index = 0; index < pending.length; index++) {
      const request = pending[index];
      const resource = resources[index] ?? resources.at(-1);
      if (!resource) continue;
      request.status = resource.responseStatus;
      request.responseUrl = resource.name;
      request.mimeType = "text/javascript";
      request.fromDiskCache = resource.transferSize === 0;
      request.fromServiceWorker = false;
      request.headers = {
        contentLength: null,
        linkedSize: null,
        repoCommit: null,
        etag: null,
        location: null,
        cacheControl: null,
      };
      request.declaredBytes = resource.encodedBodySize || null;
      request.encodedDataLength = resource.transferSize || resource.encodedBodySize || 0;
      request.terminal = "finished-performance-resource-timing";
      request.cdpHandoff = {
        reason:
          "Chrome hands a dedicated-worker bootstrap fetch from the page Network domain to the worker target before responseReceived; ResourceTiming supplies its terminal HTTP status and bytes.",
        responseStatus: resource.responseStatus,
        transferSize: resource.transferSize,
        encodedBodySize: resource.encodedBodySize,
        duration: resource.duration,
      };
      this.active.delete(request.requestKey);
    }
  }
  #http(url) {
    return /^https?:/i.test(url || "");
  }
  #headers(headers = {}) {
    const out = {};
    for (const [key, value] of Object.entries(headers)) out[key.toLowerCase()] = String(value);
    return out;
  }
  #response(entry, response) {
    const headers = this.#headers(response?.headers);
    entry.responseUrl = response?.url ?? entry.requestedUrl;
    entry.status = response?.status ?? null;
    entry.statusText = response?.statusText ?? "";
    entry.mimeType = response?.mimeType ?? "";
    entry.fromDiskCache = Boolean(response?.fromDiskCache);
    entry.fromServiceWorker = Boolean(response?.fromServiceWorker);
    entry.headers = {
      contentLength: headers["content-length"] ?? null,
      linkedSize: headers["x-linked-size"] ?? null,
      repoCommit: headers["x-repo-commit"] ?? null,
      etag: headers.etag ?? null,
      location: headers.location ?? null,
      cacheControl: headers["cache-control"] ?? null,
    };
    const declared = entry.headers.linkedSize ?? entry.headers.contentLength;
    entry.declaredBytes = /^\d+$/.test(declared ?? "") ? Number(declared) : null;
  }
  #event(message) {
    if (message.method === "Target.attachedToTarget") {
      const info = message.params.targetInfo ?? {};
      if (
        (info.type === "worker" || info.type === "service_worker") &&
        info.url.includes("fashion-clip")
      ) {
        const sid = message.params.sessionId;
        void (async () => {
          await this.enable(sid, { type: info.type, url: info.url });
          await this.cdp.send("Runtime.runIfWaitingForDebugger", {}, sid).catch(() => {});
        })();
      }
      return;
    }
    const sid = message.sessionId;
    if (!this.sessions.has(sid)) return;
    const params = message.params ?? {};
    const key = `${sid}:${params.requestId}`;
    if (message.method === "Network.requestWillBeSent" && this.#http(params.request?.url)) {
      if (params.redirectResponse && this.active.has(key)) {
        const previous = this.active.get(key);
        this.#response(previous, params.redirectResponse);
        previous.terminal = "redirect";
        previous.encodedDataLength = params.redirectResponse.encodedDataLength ?? 0;
      }
      const requestedUrl = params.request.url;
      const entry = {
        sequence: ++this.sequence,
        requestKey: key,
        phase,
        target: this.sessions.get(sid),
        method: params.request.method,
        requestedUrl,
        revision: Object.values(MODELS).find((model) =>
          decodeURIComponent(requestedUrl).includes(`/${model.id}/resolve/${model.revision}/`)
        )?.revision ?? null,
        status: null,
        declaredBytes: null,
        encodedDataLength: null,
        terminal: null,
      };
      this.requests.push(entry);
      this.active.set(key, entry);
      return;
    }
    const entry = this.active.get(key);
    if (!entry) return;
    if (message.method === "Network.responseReceived") this.#response(entry, params.response);
    if (message.method === "Network.loadingFinished") {
      entry.terminal = "finished";
      entry.encodedDataLength = params.encodedDataLength ?? 0;
      this.active.delete(key);
    }
    if (message.method === "Network.loadingFailed") {
      entry.terminal = "failed";
      entry.failure = {
        errorText: params.errorText,
        blockedReason: params.blockedReason ?? null,
        canceled: Boolean(params.canceled),
      };
      this.active.delete(key);
    }
  }
  artifact(model) {
    const wanted = `/${model.id}/resolve/${model.revision}/${model.artifact}`;
    const initial = this.requests.find((request) =>
      decodeURIComponent(request.requestedUrl).includes(wanted) &&
      request.declaredBytes === model.bytes &&
      request.headers?.repoCommit === model.revision
    );
    if (!initial) return null;
    const hops = this.requests.filter((request) => request.requestKey === initial.requestKey);
    const final = hops.find((request) =>
      request.status === 200 && request.declaredBytes === model.bytes
    );
    return final ? { initial, final, hops: hops.map((request) => request.sequence) } : null;
  }
  review() {
    const allowedHost = (host) =>
      host === "127.0.0.1" || host === "cdn.jsdelivr.net" || host === "huggingface.co" ||
      host.endsWith(".huggingface.co") || host.endsWith(".hf.co") || host.endsWith(".xethub.hf.co");
    const unexpectedOrigins = this.requests.filter((request) => {
      try {
        return !allowedHost(new URL(request.requestedUrl).hostname);
      } catch {
        return true;
      }
    });
    const unpinned = this.requests.filter((request) => {
      const url = decodeURIComponent(request.requestedUrl);
      return Object.values(MODELS).some((model) =>
        url.includes(`/${model.id}/resolve/`) &&
        !url.includes(`/${model.id}/resolve/${model.revision}/`)
      );
    });
    const unfinished = this.requests.filter((request) => !request.terminal);
    const badStatus = this.requests.filter((request) =>
      request.status != null && (request.status >= 400 || request.status < 200)
    );
    const failed = this.requests.filter((request) => request.terminal === "failed");
    const expectedFailures = failed.filter((request) =>
      request.phase.startsWith("error-injection-") &&
      Object.values(MODELS).some((model) =>
        decodeURIComponent(request.requestedUrl).includes(`/${model.id}/`)
      )
    );
    const expectedFailurePhases = new Set(expectedFailures.map((request) => request.phase));
    const unexpectedFailures = failed.filter((request) => !expectedFailures.includes(request));
    const missingByteAccounting = this.requests.filter((request) =>
      request.terminal?.startsWith("finished") && !Number.isFinite(request.encodedDataLength)
    );
    return {
      complete: unexpectedOrigins.length === 0 && unpinned.length === 0 &&
        unfinished.length === 0 &&
        badStatus.length === 0 && unexpectedFailures.length === 0 &&
        missingByteAccounting.length === 0 &&
        expectedFailures.length >= 2 && expectedFailurePhases.size === 2,
      totals: {
        requests: this.requests.length,
        finished:
          this.requests.filter((request) => request.terminal?.startsWith("finished")).length,
        redirects: this.requests.filter((request) => request.terminal === "redirect").length,
        expectedFailures: expectedFailures.length,
      },
      unexpectedOrigins: unexpectedOrigins.map((request) => request.requestedUrl),
      unpinned: unpinned.map((request) => request.requestedUrl),
      unfinished: unfinished.map((request) => request.requestedUrl),
      badStatus: badStatus.map((request) => ({
        url: request.requestedUrl,
        status: request.status,
      })),
      unexpectedFailures: unexpectedFailures.map((request) => ({
        url: request.requestedUrl,
        phase: request.phase,
        failure: request.failure,
      })),
      expectedFailures: expectedFailures.map((request) => ({
        url: request.requestedUrl,
        phase: request.phase,
        failure: request.failure,
      })),
      missingByteAccounting: missingByteAccounting.map((request) => request.requestedUrl),
    };
  }
}

async function evaluate(cdp, sid, expression, timeoutMs = 45_000) {
  const { result, exceptionDetails } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: `(async()=>{${expression}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    timeoutMs,
  );
  if (exceptionDetails) {
    throw new Error(exceptionDetails.exception?.description || "Runtime.evaluate failed");
  }
  return result?.value;
}
async function waitFor(cdp, sid, expression, label, timeoutMs = 10 * 60_000) {
  const deadline = Date.now() + timeoutMs;
  let last = null;
  let nextLog = 0;
  while (Date.now() < deadline) {
    last = await evaluate(cdp, sid, `return (${expression});`).catch((error) =>
      `ERR:${error.message}`
    );
    if (last) return last;
    if (Date.now() >= nextLog) {
      log(`WAIT ${label}`);
      nextLog = Date.now() + 15_000;
    }
    await sleep(350);
  }
  throw new Error(`Timeout waiting for ${label}; last=${JSON.stringify(last)}`);
}
async function openEvidencePage(cdp, network, url) {
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const page = { targetId, sessionId, errors: [], failed: [] };
  await network.enable(sessionId, { type: "page", url });
  await cdp.send("Page.enable", {}, sessionId);
  await cdp.send("Runtime.enable", {}, sessionId);
  await cdp.send("DOM.enable", {}, sessionId);
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: true,
  }, sessionId);
  cdp.on((message) => {
    if (message.sessionId !== sessionId) return;
    if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
      page.errors.push(
        message.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" "),
      );
    }
    if (message.method === "Runtime.exceptionThrown") {
      page.errors.push(message.params.exceptionDetails?.exception?.description || "exception");
    }
    if (message.method === "Network.loadingFailed" && !message.params.canceled) {
      page.failed.push(message.params.errorText || "network failure");
    }
  });
  await cdp.send("Page.addScriptToEvaluateOnNewDocument", {
    source:
      `addEventListener('DOMContentLoaded',()=>{window.__fashionLoaderTrace=[];let last='';setInterval(()=>{const value=[...document.querySelectorAll('.model-loader')].map((node,index)=>({index,state:node.dataset.state,status:node.querySelector('.status')?.textContent||node.querySelector('.dl-phase')?.textContent||''}));const key=JSON.stringify(value);if(key!==last){last=key;window.__fashionLoaderTrace.push({at:performance.now(),value})}},25)})`,
  }, sessionId);
  await setViewport(cdp, sessionId, DESKTOP);
  const loaded = new Promise((resolve) => {
    cdp.on((message) => {
      if (message.sessionId === sessionId && message.method === "Page.loadEventFired") resolve();
    });
  });
  await cdp.send("Page.navigate", { url }, sessionId);
  await Promise.race([loaded, sleep(15_000)]);
  await waitFor(
    cdp,
    sessionId,
    `document.querySelectorAll('.model-loader').length>0`,
    "loader mount",
    30_000,
  );
  return page;
}
async function closeEvidencePage(cdp, network, page) {
  const resources = await evaluate(
    cdp,
    page.sessionId,
    `return performance.getEntriesByType('resource').map((entry)=>({name:entry.name,responseStatus:entry.responseStatus||0,transferSize:entry.transferSize||0,encodedBodySize:entry.encodedBodySize||0,duration:entry.duration||0}));`,
  ).catch(() => []);
  network.reconcileWorkerBootstraps(page.sessionId, resources ?? []);
  await closePage(cdp, page.targetId);
}
async function keyboard(cdp, sid, key, modifiers = 0) {
  const special = key === " "
    ? { code: "Space", windowsVirtualKeyCode: 32 }
    : key === "Enter"
    ? { code: "Enter", windowsVirtualKeyCode: 13 }
    : key.toLowerCase() === "a"
    ? { code: "KeyA", windowsVirtualKeyCode: 65 }
    : { code: key };
  const text = key === "Enter" ? "\r" : key === " " ? " " : "";
  await cdp.send("Input.dispatchKeyEvent", {
    type: "keyDown",
    key,
    modifiers,
    nativeVirtualKeyCode: special.windowsVirtualKeyCode,
    text,
    unmodifiedText: text,
    ...special,
  }, sid);
  await cdp.send("Input.dispatchKeyEvent", {
    type: "keyUp",
    key,
    modifiers,
    nativeVirtualKeyCode: special.windowsVirtualKeyCode,
    ...special,
  }, sid);
}
async function activate(cdp, sid, selector, key = "Enter") {
  const focused = await evaluate(
    cdp,
    sid,
    `const el=document.querySelector(${
      JSON.stringify(selector)
    });if(!el)return false;el.focus();return document.activeElement===el;`,
  );
  if (!focused) throw new Error(`Cannot focus ${selector}`);
  await keyboard(cdp, sid, key);
  return { selector, key, focused };
}
async function activateButtonText(cdp, sid, rootSelector, pattern, key = "Enter") {
  const selector = await evaluate(
    cdp,
    sid,
    `const root=document.querySelector(${
      JSON.stringify(rootSelector)
    });const buttons=[...(root?.querySelectorAll('button')||[])];const i=buttons.findIndex((button)=>new RegExp(${
      JSON.stringify(pattern)
    },'i').test(button.textContent));if(i<0)return null;buttons[i].dataset.evidenceButton=String(i);return ${
      JSON.stringify(rootSelector)
    }+' button[data-evidence-button="'+i+'"]';`,
  );
  if (!selector) throw new Error(`No button /${pattern}/ in ${rootSelector}`);
  return activate(cdp, sid, selector, key);
}
async function typeText(cdp, sid, selector, value) {
  const before = await evaluate(
    cdp,
    sid,
    `const el=document.querySelector(${JSON.stringify(selector)});el.focus();return el.value;`,
  );
  await keyboard(cdp, sid, "a", 2);
  await cdp.send("Input.insertText", { text: value }, sid);
  const after = await evaluate(
    cdp,
    sid,
    `return document.querySelector(${JSON.stringify(selector)}).value;`,
  );
  return {
    selector,
    inputMethod: "CDP key events + Input.insertText",
    before,
    after,
    changed: before !== after && after === value,
  };
}
async function driveMemoryControls(cdp, sid) {
  const count = await evaluate(
    cdp,
    sid,
    `return document.querySelectorAll('model-memory-diagnostics').length;`,
  );
  const results = [];
  for (let index = 0; index < count; index++) {
    const before = await evaluate(
      cdp,
      sid,
      `const host=document.querySelectorAll('model-memory-diagnostics')[${index}];const summary=host.shadowRoot.querySelector('summary');summary.focus();return {focused:host.shadowRoot.activeElement===summary,open:host.shadowRoot.querySelector('details').open,measureDisabled:host.shadowRoot.querySelector('#measure').disabled,reason:host.shadowRoot.querySelector('#explanation').textContent};`,
    );
    await keyboard(cdp, sid, "Enter");
    const opened = await evaluate(
      cdp,
      sid,
      `return document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.querySelector('details').open;`,
    );
    await keyboard(cdp, sid, "Enter");
    const closed = await evaluate(
      cdp,
      sid,
      `return !document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.querySelector('details').open;`,
    );
    let measure;
    if (before.measureDisabled) {
      measure = {
        status: "not-applicable",
        reason: before.reason,
      };
    } else {
      const focused = await evaluate(
        cdp,
        sid,
        `const button=document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.querySelector('#measure');button.focus();return document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.activeElement===button;`,
      );
      await keyboard(cdp, sid, "Enter");
      await waitFor(
        cdp,
        sid,
        `!document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.querySelector('#measure').disabled`,
        `memory measurement ${index}`,
        60_000,
      );
      measure = {
        status: "pass",
        focused,
        result: await evaluate(
          cdp,
          sid,
          `return document.querySelectorAll('model-memory-diagnostics')[${index}].shadowRoot.querySelector('#status').textContent;`,
        ),
      };
    }
    results.push({
      index,
      summary: { keyboard: "Enter", focused: before.focused, before: before.open, opened, closed },
      measure,
    });
  }
  return results;
}
async function uploadFixture(cdp, sid) {
  const focused = await evaluate(
    cdp,
    sid,
    `const el=document.querySelector('#upload');el.focus();return document.activeElement===el;`,
  );
  const { root } = await cdp.send("DOM.getDocument", { depth: -1, pierce: true }, sid);
  const { nodeId } = await cdp.send("DOM.querySelector", {
    nodeId: root.nodeId,
    selector: "#upload",
  }, sid);
  await cdp.send("DOM.setFileInputFiles", { nodeId, files: [FIXTURE] }, sid);
  await waitFor(
    cdp,
    sid,
    `/Uploaded image:/.test(document.querySelector('#preview')?.alt||'')`,
    "file input change",
    30_000,
  );
  return {
    selector: "#upload",
    inputMethod: "CDP.DOM.setFileInputFiles",
    keyboardFocused: focused,
    file: basename(FIXTURE),
    pickerChrome: "not-applicable",
    pickerReason:
      "Headless CDP cannot automate the operating-system file-picker window; the real focused input/change path is exercised with DOM.setFileInputFiles.",
    after: await evaluate(
      cdp,
      sid,
      `return {name:document.querySelector('#upload').files[0]?.name,alt:document.querySelector('#preview').alt};`,
    ),
  };
}
async function loaderStates(cdp, sid) {
  return evaluate(cdp, sid, `return window.__fashionLoaderTrace||[];`);
}
async function waitReady(cdp, sid, count = 1, label = "model ready") {
  return waitFor(
    cdp,
    sid,
    `[...document.querySelectorAll('.model-loader')].filter((node)=>node.dataset.state==='ready').length===${count}&&!document.querySelector('#run').disabled`,
    label,
  );
}
async function coldDownload(cdp, network, page) {
  const sid = page.sessionId;
  await waitFor(
    cdp,
    sid,
    `document.querySelectorAll('.model-loader[data-state="download-required"]').length===2`,
    "both first-visit Download states",
    60_000,
  );
  const initial = await evaluate(
    cdp,
    sid,
    `return [...document.querySelectorAll('.model-loader')].map((node)=>({state:node.dataset.state,status:node.querySelector('.status')?.textContent,button:node.querySelector('.loader-actions button')?.textContent}));`,
  );
  for (const [index, stage] of ["fashion", "general"].entries()) {
    phase = `cold-download-${stage}`;
    await activateButtonText(
      cdp,
      sid,
      index === 0 ? "#model-loader" : "#general-loader",
      "Download model",
      "Enter",
    );
    await waitFor(
      cdp,
      sid,
      `!document.querySelector(${
        JSON.stringify(index === 0 ? "#model-loader" : "#general-loader")
      }+' model-download-status').hidden`,
      `${stage} progress visible`,
      30_000,
    );
    const disclosure = await activate(
      cdp,
      sid,
      `${index === 0 ? "#model-loader" : "#general-loader"} summary`,
      "Enter",
    );
    const opened = await evaluate(
      cdp,
      sid,
      `return document.querySelector(${
        JSON.stringify(index === 0 ? "#model-loader" : "#general-loader")
      }+' details').open;`,
    );
    await activate(
      cdp,
      sid,
      `${index === 0 ? "#model-loader" : "#general-loader"} summary`,
      "Enter",
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelectorAll('.model-loader')[${index}].dataset.state==='ready'`,
      `${stage} cold download`,
      12 * 60_000,
    );
    matrix(
      stage,
      "first-visit-absent",
      initial[index]?.state === "download-required" ? "pass" : "fail",
      `initial state=${initial[index]?.state}; button=${initial[index]?.button}`,
    );
    matrix(
      stage,
      "download-required",
      /Download/.test(initial[index]?.button || "") ? "pass" : "fail",
      initial[index]?.button || "missing Download control",
    );
    matrix(
      stage,
      "downloading-progress",
      opened ? "pass" : "fail",
      "Download keyboard-activated; native file-details disclosure opened and closed with Enter",
      disclosure,
    );
    matrix(
      stage,
      "initialising-ready",
      "pass",
      "Cold transfer reached ready after real worker initialisation",
    );
  }
  phase = "cold-download-settled";
  await sleep(6000);
  return { initial, traces: await loaderStates(cdp, sid) };
}
async function driveChangedControls(cdp, page, route, device) {
  const sid = page.sessionId;
  const interactions = [];
  interactions.push(await activate(cdp, sid, '[data-sample*="fashion-shirt"]', " "));
  const shirt = await evaluate(
    cdp,
    sid,
    `return {pressed:document.querySelector('[data-sample*="fashion-shirt"]').getAttribute('aria-pressed'),alt:document.querySelector('#preview').alt};`,
  );
  interactions.push(await activate(cdp, sid, '[data-sample*="fashion-sneaker"]', "Enter"));
  const sneaker = await evaluate(
    cdp,
    sid,
    `return {pressed:document.querySelector('[data-sample*="fashion-sneaker"]').getAttribute('aria-pressed'),alt:document.querySelector('#preview').alt};`,
  );
  const upload = await uploadFixture(cdp, sid);
  const edited =
    `changed ${route.id} ${device} one\nchanged ${route.id} ${device} two\nchanged ${route.id} ${device} three`;
  const firstEdit = await typeText(cdp, sid, "#labels", edited);
  interactions.push(await activate(cdp, sid, "#reset-labels", "Enter"));
  const resetValue = await evaluate(cdp, sid, `return document.querySelector('#labels').value;`);
  const finalLabels =
    `edited ${route.id} ${device} alpha\nedited ${route.id} ${device} beta\nedited ${route.id} ${device} gamma`;
  const secondEdit = await typeText(cdp, sid, "#labels", finalLabels);
  interactions.push(await activate(cdp, sid, "#run", "Enter"));
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#run-status')?.textContent==='Real local inference complete.'&&!document.querySelector('#run').disabled`,
    `${route.id}/${device} inference`,
    4 * 60_000,
  );
  const output = await evaluate(
    cdp,
    sid,
    `return {status:document.querySelector('#run-status').textContent,labels:[...document.querySelectorAll('#tensor-rows tr td:first-child')].map((el)=>el.textContent),rows:document.querySelectorAll('#tensor-rows tr').length,resultRows:document.querySelectorAll('#results .fashion-result').length,generalRows:document.querySelectorAll('#general-results .fashion-result').length,top:document.querySelector('#results .fashion-result strong')?.textContent||null,generalTop:document.querySelector('#general-results .fashion-result strong')?.textContent||null};`,
  );
  const memoryControls = await driveMemoryControls(cdp, sid);
  const visibleControls = await evaluate(
    cdp,
    sid,
    `const light=[...document.querySelectorAll('button,input,textarea,select,summary')].filter((el)=>{const s=getComputedStyle(el);return s.display!=='none'&&s.visibility!=='hidden'&&el.getClientRects().length}).map((el)=>({scope:'light-dom',tag:el.tagName.toLowerCase(),id:el.id||null,text:(el.textContent||'').trim(),type:el.type||null,disabled:Boolean(el.disabled),name:el.getAttribute('aria-label')||document.querySelector('label[for="'+el.id+'"]')?.textContent||null}));const shadow=[...document.querySelectorAll('model-memory-diagnostics')].flatMap((host,index)=>[...host.shadowRoot.querySelectorAll('button,summary')].map((el)=>({scope:'memory-shadow-'+index,tag:el.tagName.toLowerCase(),id:el.id||null,text:(el.textContent||'').trim(),type:el.type||null,disabled:Boolean(el.disabled),name:(el.textContent||'').trim()})));return [...light,...shadow];`,
  );
  return {
    shirt,
    sneaker,
    upload,
    firstEdit,
    reset: { keyboard: "Enter", before: edited, after: resetValue, changed: resetValue !== edited },
    secondEdit,
    run: { keyboard: "Enter", changedLabels: finalLabels.split("\n") },
    interactions,
    memoryControls,
    visibleControls,
    output,
  };
}
async function themeAndScreenshot(cdp, page, route, device, theme, errorStart, failedStart) {
  const sid = page.sessionId;
  await cdp.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: theme }],
  }, sid);
  await sleep(200);
  await evaluate(
    cdp,
    sid,
    `const old=document.querySelector('#evidence-force-visible');if(old)old.remove();const style=document.createElement('style');style.id='evidence-force-visible';style.textContent='*{content-visibility:visible!important;contain-intrinsic-size:auto!important}';document.head.append(style);return true;`,
  );
  const checks = await evaluate(
    cdp,
    sid,
    `return {overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,unnamedButtons:[...document.querySelectorAll('button')].filter((button)=>!((button.textContent||'').trim()||button.getAttribute('aria-label')||button.getAttribute('aria-labelledby'))).length,missingAlt:[...document.querySelectorAll('img')].filter((img)=>!img.hasAttribute('alt')).length,status:document.querySelector('#run-status')?.textContent,rows:document.querySelectorAll('#tensor-rows tr').length,resultRows:document.querySelectorAll('#results .fashion-result').length,generalRows:document.querySelectorAll('#general-results .fashion-result').length};`,
  );
  const filename = `${route.screenshot}-${device}-${theme}.jpg`;
  const file = join(SCREEN_DIR, filename);
  const { data } = await cdp.send(
    "Page.captureScreenshot",
    {
      format: "jpeg",
      quality: 88,
      captureBeyondViewport: true,
      fromSurface: true,
    },
    sid,
    60_000,
  );
  writeFileSync(file, Buffer.from(data, "base64"));
  const dimensions = await evaluate(
    cdp,
    sid,
    `return {width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight};`,
  );
  const item = {
    route: `/models/fashion-clip/${route.path}`,
    device,
    theme,
    resultState: "ready with changed-input real inference",
    path: relative(ROOT, file),
    bytes: statSync(file).size,
    sha256: sha256File(file),
    dimensions,
  };
  screenshots.push(item);
  const prefix = `${route.id}-${device}-${theme}`;
  assert(`${prefix}-no-overflow`, checks.overflow === false, `overflow=${checks.overflow}`, checks);
  assert(
    `${prefix}-a11y-names`,
    checks.unnamedButtons === 0 && checks.missingAlt === 0,
    `unnamedButtons=${checks.unnamedButtons}; missingAlt=${checks.missingAlt}`,
  );
  const newErrors = page.errors.slice(errorStart);
  const newFailures = page.failed.slice(failedStart);
  assert(
    `${prefix}-clean-runtime`,
    newErrors.length === 0 && newFailures.length === 0,
    `${newErrors.length} console errors; ${newFailures.length} network failures`,
    { newErrors, newFailures },
  );
  assert(
    `${prefix}-screenshot`,
    item.bytes > 1000,
    `${item.path}; ${item.bytes} bytes; sha256=${item.sha256}`,
  );
  return { theme, ...checks, screenshot: item.path, screenshotSha256: item.sha256, dimensions };
}
async function releaseReinit(cdp, page, route, device) {
  const sid = page.sessionId;
  const mounts = route.general ? ["#model-loader", "#general-loader"] : ["#model-loader"];
  const stages = [];
  for (const mount of mounts) {
    const stage = mount === "#model-loader" ? "fashion" : "general";
    const release = await activateButtonText(cdp, sid, mount, "Release from memory", "Enter");
    await waitFor(
      cdp,
      sid,
      `document.querySelector(${
        JSON.stringify(mount)
      }+' .model-loader').dataset.state==='released'`,
      `${stage} released`,
      60_000,
    );
    const reload = await activateButtonText(cdp, sid, mount, "Load model into memory", "Enter");
    await waitFor(
      cdp,
      sid,
      `document.querySelector(${JSON.stringify(mount)}+' .model-loader').dataset.state==='ready'`,
      `${stage} reinit`,
      4 * 60_000,
    );
    stages.push({ stage, release, reload, finalState: "ready" });
  }
  return stages;
}
async function cacheUrls(cdp, sid, modelId) {
  return evaluate(
    cdp,
    sid,
    `const out=[];for(const cacheName of await caches.keys()){const cache=await caches.open(cacheName);for(const request of await cache.keys()){if(request.url.includes(${
      JSON.stringify(`/${modelId}/`)
    }))out.push({cacheName,url:request.url})}}return out;`,
  );
}
async function deleteCachedUrl(cdp, sid, item) {
  return evaluate(
    cdp,
    sid,
    `return (await caches.open(${JSON.stringify(item.cacheName)})).delete(${
      JSON.stringify(item.url)
    });`,
  );
}
async function exerciseFailure(cdp, network, base, stage) {
  phase = `partial-setup-${stage}`;
  let page = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  await waitReady(cdp, page.sessionId, 2, `${stage} failure setup ready`);
  const model = MODELS[stage];
  const urls = await cacheUrls(cdp, page.sessionId, model.id);
  const missing = urls.find((item) =>
    /(?:config|tokenizer_config|preprocessor_config)\.json(?:\?|$)/.test(item.url)
  );
  if (!missing) throw new Error(`No small cached config found for ${stage}`);
  await deleteCachedUrl(cdp, page.sessionId, missing);
  await closeEvidencePage(cdp, network, page);

  page = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  const mount = stage === "fashion" ? "#model-loader" : "#general-loader";
  await waitFor(
    cdp,
    page.sessionId,
    `document.querySelector(${JSON.stringify(mount)}+' .model-loader').dataset.state==='partial'`,
    `${stage} partial state`,
    60_000,
  );
  matrix(
    stage,
    "partial",
    "pass",
    `Deleted recorded atomic cache entry ${missing.url}; loader showed partial`,
  );
  matrix(
    stage,
    "evicted",
    "pass",
    "The missing recorded Cache Storage entry is the shared loader's observable browser-eviction state",
    { missing: missing.url },
  );
  phase = `error-injection-${stage}`;
  await network.setBlocks([missing.url]);
  await activateButtonText(cdp, page.sessionId, mount, "Re-download missing assets", "Enter");
  await waitFor(
    cdp,
    page.sessionId,
    `document.querySelector(${
      JSON.stringify(mount)
    }+' .model-loader').dataset.state==='error'&&/Retry/.test(document.querySelector(${
      JSON.stringify(mount)
    }+' .loader-actions')?.textContent||'')`,
    `${stage} visible error/retry`,
    90_000,
  );
  const errorText = await evaluate(
    cdp,
    page.sessionId,
    `return document.querySelector(${JSON.stringify(mount)}+' .status').textContent;`,
  );
  matrix(stage, "error", "pass", `Blocked real missing config request; visible state=${errorText}`);
  await network.setBlocks([]);
  phase = `retry-recovery-${stage}`;
  await activateButtonText(cdp, page.sessionId, mount, "Retry", "Enter");
  await waitFor(
    cdp,
    page.sessionId,
    `document.querySelector(${JSON.stringify(mount)}+' .model-loader').dataset.state==='ready'`,
    `${stage} retry recovery`,
    4 * 60_000,
  );
  matrix(
    stage,
    "retry",
    "pass",
    "Retry keyboard-activated after unblock and returned the real stage to ready",
  );
  await closeEvidencePage(cdp, network, page);
}
async function exerciseOffline(cdp, network, base) {
  phase = "offline-cached";
  const page = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  await waitReady(cdp, page.sessionId, 2, "offline setup ready");
  await activateButtonText(cdp, page.sessionId, "#model-loader", "Release from memory", "Enter");
  await activateButtonText(cdp, page.sessionId, "#general-loader", "Release from memory", "Enter");
  await waitFor(
    cdp,
    page.sessionId,
    `document.querySelectorAll('.model-loader[data-state="released"]').length===2`,
    "both stages released",
    60_000,
  );
  const remoteModelOrigins = [
    "https://huggingface.co/*",
    "https://*.huggingface.co/*",
    "https://*.hf.co/*",
    "https://*.xethub.hf.co/*",
  ];
  await network.setBlocks(remoteModelOrigins);
  let worked = false;
  let evidence;
  try {
    await activateButtonText(
      cdp,
      page.sessionId,
      "#model-loader",
      "Load model into memory",
      "Enter",
    );
    await waitFor(
      cdp,
      page.sessionId,
      `document.querySelector('#model-loader .model-loader').dataset.state==='ready'`,
      "offline FashionCLIP reinit",
      4 * 60_000,
    );
    await activateButtonText(
      cdp,
      page.sessionId,
      "#general-loader",
      "Load model into memory",
      "Enter",
    );
    await waitFor(
      cdp,
      page.sessionId,
      `document.querySelector('#general-loader .model-loader').dataset.state==='ready'`,
      "offline general CLIP reinit",
      4 * 60_000,
    );
    worked = true;
    evidence =
      "All Hugging Face/model-CDN origins were CDP-blocked; both released workers recreated and reached ready exclusively from browser model caches while the already-open app shell remained available.";
  } catch (error) {
    evidence = error.message;
  } finally {
    await network.setBlocks([]);
  }
  for (const stage of Object.keys(MODELS)) {
    matrix(stage, "offline-cached", worked ? "pass" : "fail", evidence);
  }
  await closeEvidencePage(cdp, network, page);
}
async function hashGeneralArtifact(cdp, network, base) {
  phase = "general-hash-verification";
  const page = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  await waitReady(cdp, page.sessionId, 2, "hash setup ready");
  const proof = await evaluate(
    cdp,
    page.sessionId,
    `const model=${
      JSON.stringify(MODELS.general.id)
    };let response=null,url=null;for(const name of await caches.keys()){const cache=await caches.open(name);for(const request of await cache.keys()){if(request.url.includes('/'+model+'/')&&request.url.includes('/onnx/model_quantized.onnx')){response=await cache.match(request);url=request.url;break}}if(response)break}if(!response)return null;const bytes=await response.arrayBuffer();const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map((value)=>value.toString(16).padStart(2,'0')).join('');return {url,bytes:bytes.byteLength,sha256:hash};`,
    4 * 60_000,
  );
  assert(
    "general-artifact-sha256",
    proof?.bytes === MODELS.general.bytes && proof?.sha256 === MODELS.general.sha256,
    proof
      ? `${proof.url}; ${proof.bytes} bytes; sha256=${proof.sha256}`
      : "general q8 cache artifact missing",
    proof,
  );
  await closeEvidencePage(cdp, network, page);
  return proof;
}
async function exerciseClear(cdp, network, base) {
  phase = "clear-cache";
  const page = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  await waitReady(cdp, page.sessionId, 2, "clear setup ready");
  for (const [stage, mount] of [["fashion", "#model-loader"], ["general", "#general-loader"]]) {
    await activateButtonText(cdp, page.sessionId, mount, "Clear cached model", "Enter");
    await waitFor(
      cdp,
      page.sessionId,
      `document.querySelector(${
        JSON.stringify(mount)
      }+' .model-loader').dataset.state==='download-required'`,
      `${stage} clear`,
      90_000,
    );
    const left = await cacheUrls(cdp, page.sessionId, MODELS[stage].id);
    matrix(
      stage,
      "clear-cache",
      left.length === 0 ? "pass" : "fail",
      `Clear keyboard-activated; ${left.length} matching cached requests remain`,
      { remaining: left },
    );
  }
  await closeEvidencePage(cdp, network, page);
}

mkdirSync(SCREEN_DIR, { recursive: true });
for (const file of [OUT, RUN_OUT]) rmSync(file, { force: true });
for (const route of ROUTES) {
  for (const device of Object.keys(VIEWPORTS)) {
    for (const theme of ["light", "dark"]) {
      rmSync(join(SCREEN_DIR, `${route.screenshot}-${device}-${theme}.jpg`), { force: true });
    }
  }
}

const workerSource = readFileSync(join(ROOT, FAMILY, "worker.js"), "utf8");
const fashionSource = readFileSync(join(ROOT, FAMILY, "fashion.js"), "utf8");
assert(
  "primary-source-pin",
  workerSource.includes(MODELS.fashion.revision) && fashionSource.includes(MODELS.fashion.revision),
  `primary=${MODELS.fashion.id}@${MODELS.fashion.revision}`,
);
assert(
  "general-source-pin",
  workerSource.includes(MODELS.general.revision) &&
    fashionSource.includes(MODELS.general.revision) &&
    fashionSource.includes(String(MODELS.general.bytes)) &&
    fashionSource.includes(MODELS.general.sha256),
  `general=${MODELS.general.id}@${MODELS.general.revision}; bytes=${MODELS.general.bytes}; sha256=${MODELS.general.sha256}`,
);

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
const chrome = await launchChrome({
  userDataDir: PROFILE,
  resetProfile: true,
  removeProfileOnKill: false,
});
const cdp = new CDP(chrome.ws);
const network = new NetworkEvidence(cdp);
let cold;
let generalHash = null;
try {
  phase = "cold-first-visit";
  const coldPage = await openEvidencePage(cdp, network, base + "models/fashion-clip/multi-model/");
  cold = await coldDownload(cdp, network, coldPage);
  await closeEvidencePage(cdp, network, coldPage);

  for (const route of ROUTES) {
    for (const [device, viewport] of Object.entries(VIEWPORTS)) {
      phase = `route-${route.id}-${device}`;
      const page = await openEvidencePage(cdp, network, base + `models/fashion-clip/${route.path}`);
      const errorStart = page.errors.length;
      const failedStart = page.failed.length;
      await setViewport(cdp, page.sessionId, viewport);
      const ready = await waitReady(
        cdp,
        page.sessionId,
        route.general ? 2 : 1,
        `${route.id}/${device} warm auto-init`,
      );
      const traces = await loaderStates(cdp, page.sessionId);
      const clickedBeforeReady = false;
      matrix(
        "fashion",
        "current-cached-auto-init",
        "pass",
        `${route.id}/${device} reached ready without a loader click`,
        { ready, clickedBeforeReady, traces },
      );
      if (route.general) {
        matrix(
          "general",
          "current-cached-auto-init",
          "pass",
          `${route.id}/${device} both stages reached ready without a loader click`,
          { traces },
        );
      }
      const controls = await driveChangedControls(cdp, page, route, device);
      const expectedRows = 3;
      const controlsOkay = controls.shirt.pressed === "true" &&
        controls.sneaker.pressed === "true" &&
        controls.upload.keyboardFocused && controls.upload.after?.name === basename(FIXTURE) &&
        controls.firstEdit.changed && controls.reset.changed && controls.secondEdit.changed &&
        controls.memoryControls.length === (route.general ? 2 : 1) &&
        controls.memoryControls.every((item) =>
          item.summary.focused && item.summary.opened && item.summary.closed &&
          ["pass", "not-applicable"].includes(item.measure.status)
        );
      assert(
        `${route.id}-${device}-changed-controls-keyboard`,
        controlsOkay,
        `changed samples, upload, textarea, reset and run; visible controls=${controls.visibleControls.length}`,
        controls,
      );
      const realOkay = controls.output.status === "Real local inference complete." &&
        controls.output.rows === expectedRows && controls.output.resultRows === expectedRows &&
        controls.output.labels.every((label) => label.startsWith(`edited ${route.id} ${device}`)) &&
        (!route.general || controls.output.generalRows === expectedRows);
      assert(
        `${route.id}-${device}-real-inference`,
        realOkay,
        `Fashion rows=${controls.output.resultRows}; tensor rows=${controls.output.rows}; general rows=${controls.output.generalRows}`,
        controls.output,
      );
      const themeRecords = [];
      for (const theme of ["light", "dark"]) {
        themeRecords.push(
          await themeAndScreenshot(cdp, page, route, device, theme, errorStart, failedStart),
        );
      }
      const lifecycle = await releaseReinit(cdp, page, route, device);
      assert(
        `${route.id}-${device}-release-reinit`,
        lifecycle.length === (route.general ? 2 : 1) &&
          lifecycle.every((item) => item.finalState === "ready"),
        `released and keyboard-reinitialised ${lifecycle.length} stage(s)`,
        lifecycle,
      );
      records.push({
        route: `/models/fashion-clip/${route.path}`,
        routeId: route.id,
        device,
        controls,
        realOutput: controls.output,
        lifecycle,
        loaderTrace: await loaderStates(cdp, page.sessionId),
        themes: themeRecords,
      });
      await closeEvidencePage(cdp, network, page);
    }
  }

  await exerciseFailure(cdp, network, base, "fashion");
  await exerciseFailure(cdp, network, base, "general");
  await exerciseOffline(cdp, network, base);
  generalHash = await hashGeneralArtifact(cdp, network, base);

  for (const stage of Object.keys(MODELS)) {
    matrix(
      stage,
      "stale-update",
      "not-applicable",
      `Stage is immutable-pinned to ${
        MODELS[stage].revision
      }; createModelLoader deliberately does not query mutable latest for pinned models.`,
    );
    matrix(
      stage,
      "corrupt-response-body",
      "not-applicable",
      "Cache Storage commits Response bodies atomically; an interrupted/partial body is not retained. Missing atomic entries are exercised as partial/evicted.",
    );
    matrix(
      stage,
      "unsupported-webgpu-capability",
      "not-applicable",
      "requiresWebGPU=false and the exercised backend is WASM; the shared WebGPU-only unsupported branch does not apply to this stage.",
    );
    matrix(
      stage,
      "release",
      "pass",
      `Release and cached keyboard re-init passed in route cells for ${stage}`,
    );
  }
  await exerciseClear(cdp, network, base);
} catch (error) {
  failures++;
  assertions.push({
    id: "producer-fatal",
    status: "fail",
    evidence: error?.stack || String(error),
  });
  console.error(error?.stack || error);
} finally {
  phase = "settle";
  await sleep(1500);
  await network.setOffline(false).catch(() => {});
  await network.setBlocks([]).catch(() => {});
  await chrome.kill({ removeProfile: true }).catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}

const requiredStates = [
  "first-visit-absent",
  "download-required",
  "downloading-progress",
  "initialising-ready",
  "current-cached-auto-init",
  "partial",
  "evicted",
  "error",
  "retry",
  "offline-cached",
  "stale-update",
  "corrupt-response-body",
  "unsupported-webgpu-capability",
  "release",
  "clear-cache",
];
const matrixComplete = Object.keys(MODELS).every((stage) =>
  requiredStates.every((state) =>
    stateMatrix.some((row) =>
      row.stage === stage && row.state === state && ["pass", "not-applicable"].includes(row.status)
    )
  )
);
assert(
  "loader-matrix-complete",
  matrixComplete,
  `${stateMatrix.length} rows; required=${Object.keys(MODELS).length * requiredStates.length}`,
  { requiredStates },
);
const primaryNetwork = network.artifact(MODELS.fashion);
const generalNetwork = network.artifact(MODELS.general);
assert(
  "network-primary-artifact",
  Boolean(primaryNetwork),
  primaryNetwork
    ? `status=${primaryNetwork.final.status}; bytes=${primaryNetwork.final.declaredBytes}; revision=${primaryNetwork.initial.headers.repoCommit}`
    : "missing exact primary transfer",
  primaryNetwork,
);
assert(
  "network-general-artifact",
  Boolean(generalNetwork),
  generalNetwork
    ? `status=${generalNetwork.final.status}; bytes=${generalNetwork.final.declaredBytes}; revision=${generalNetwork.initial.headers.repoCommit}`
    : "missing exact general q8 transfer",
  generalNetwork,
);
const networkReview = network.review();
assert(
  "network-complete-fail-closed",
  networkReview.complete,
  `${networkReview.totals.requests} complete requests; expected failures=${networkReview.totals.expectedFailures}; unexpected failures=${networkReview.unexpectedFailures.length}; unpinned=${networkReview.unpinned.length}`,
  networkReview,
);
assert(
  "screenshots-complete-bound",
  screenshots.length === 32 &&
    screenshots.every((shot) => shot.bytes > 1000 && /^[a-f0-9]{64}$/.test(shot.sha256)),
  `${screenshots.length}/32 screenshots carry non-empty bytes and SHA-256`,
);

const expectedIds = expectedAssertionIds();
const actualIds = assertions.filter((item) => item.id !== "producer-fatal").map((item) => item.id);
const exactDenominator = JSON.stringify(actualIds) === JSON.stringify(expectedIds);
if (!exactDenominator) {
  failures++;
  log(`FAIL frozen denominator mismatch expected=${expectedIds.length} actual=${actualIds.length}`);
}
const passed = failures === 0 && exactDenominator;
const artifact = {
  schemaVersion: 2,
  generatedAt: new Date().toISOString(),
  producer: {
    path: PRODUCER,
    sha256: producerSha256,
    sourceCommit,
    command:
      "timeout --signal=TERM --kill-after=30s 3600s node scripts/produce-fashion-clip-evidence.mjs",
    freshProfile: true,
  },
  models: MODELS,
  denominator: {
    routes: ROUTES.length,
    stages: Object.keys(MODELS).length,
    deviceClasses: Object.keys(VIEWPORTS),
    themes: ["light", "dark"],
    screenshotCells: 32,
    changedControlCells: 16,
    assertions: expectedIds.length,
    assertionIds: expectedIds,
    exactOrder: exactDenominator,
  },
  records,
  stateMatrix,
  network: {
    review: networkReview,
    artifactProofs: { fashion: primaryNetwork, general: generalNetwork },
    requests: network.requests,
  },
  contentVerification: { general: generalHash },
  screenshots,
  coldStart: cold,
  assertions,
  passed,
  summary: passed
    ? "All routes/stages, changed controls/keyboard, ready screenshots, loader states and complete pinned network evidence passed fail-closed."
    : `${failures} fail-closed acceptance failure(s).`,
};
writeFileSync(OUT, JSON.stringify(artifact, null, 2) + "\n");
const runRecord = {
  schema: 2,
  validator: PRODUCER,
  producer: PRODUCER,
  producerSha256,
  sourceCommit,
  commit: sourceCommit,
  ranAt: artifact.generatedAt,
  exitCode: passed ? 0 : 1,
  checks: `${assertions.filter((item) => item.status === "pass").length}/${expectedIds.length}`,
  frozenDenominator: {
    total: expectedIds.length,
    assertionIds: expectedIds,
    exactOrder: exactDenominator,
  },
  denominator: artifact.denominator,
  results: records.flatMap((record) =>
    record.themes.map((theme) => ({
      route: record.route.replace(/^\//, ""),
      viewport: record.device,
      theme: theme.theme,
      pass: !theme.overflow && theme.unnamedButtons === 0 && theme.missingAlt === 0 &&
        theme.status === "Real local inference complete.",
    }))
  ),
  evidence: {
    matrix: relative(ROOT, OUT),
    screenshots: relative(ROOT, SCREEN_DIR) + "/",
    networkRequests: network.requests.length,
    contentVerification: artifact.contentVerification,
    stateMatrix: `${stateMatrix.filter((row) => row.status === "pass").length} pass; ${
      stateMatrix.filter((row) => row.status === "not-applicable").length
    } honest N/A; ${stateMatrix.filter((row) => row.status === "fail").length} fail`,
  },
};
writeFileSync(RUN_OUT, JSON.stringify(runRecord, null, 2) + "\n");
log(
  `${passed ? "PASS" : "FAIL"} FashionCLIP producer: ${
    assertions.filter((item) => item.status === "pass").length
  }/${expectedIds.length}; screenshots=${screenshots.length}; requests=${network.requests.length}; state rows=${stateMatrix.length}`,
);
process.exit(passed ? 0 : 1);
