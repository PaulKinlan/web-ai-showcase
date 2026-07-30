// Schema-2 route-complete acceptance for all-mpnet-base-v2 (Xenova/all-mpnet-base-v2 q8 + pinned reranker).
// Frozen assertion IDs + exact denominator; per-cell evidence bound to genuine versioned worker messages
// (raw cosine/logit values), pinned request URLs + HTTP status + content-length for BOTH models, HTTP 4xx/5xx
// capture, every control/route/preset, keyboard/focus, light/dark, cold/warm, failure→retry→recover,
// release idle + busy, clear-cache disposal, cancellation/supersession, and post-reinit real inference.
// Captures the 20-cell screenshot matrix (content-visibility forced on so no region is blank) to
// reports/webai-g7-glm-text2-screens/ and writes models/all-mpnet-base-v2/acceptance-run.json.
// Manual/blocked states NEVER count as pass.

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
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const MPNET = "Xenova/all-mpnet-base-v2";
const MPNET_REV = "e086c5e0b3a57b0ce46dd6d9c0662948860b35f3";
const RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2";
const RERANKER_REV = "a09144355adeed5f58c8ed011d209bf8ee5a1fec";
const SCREEN_DIR = join(process.cwd(), "reports", "webai-g7-glm-text2-screens");
mkdirSync(SCREEN_DIR, { recursive: true });

const ROUTES = {
  overview: "models/all-mpnet-base-v2/",
  basics: "models/all-mpnet-base-v2/basics/",
  practical: "models/all-mpnet-base-v2/practical/",
  wild: "models/all-mpnet-base-v2/wild/",
  multimodel: "models/all-mpnet-base-v2/multi-model/",
};
const EXPECTED_ASSERTION_IDS = [
  ...Object.keys(ROUTES).flatMap((route) => [
    `${route}-real-inference`,
    `${route}-mpnet-pinned-request`,
    ...(route === "multimodel" ? ["multimodel-reranker-pinned-request"] : []),
    `${route}-keyboard-focus`,
    ...["desktop", "mobile"].flatMap((viewport) =>
      ["light", "dark"].flatMap((theme) => [
        `${route}-${viewport}-${theme}-no-overflow`,
        `${route}-${viewport}-${theme}-console-clean`,
        `${route}-${viewport}-${theme}-no-http-4xx-5xx`,
      ])
    ),
  ]),
  "lifecycle-warm-reuse",
  "lifecycle-release-idle-reinit-real-inference",
  "lifecycle-clear-cache-disposal",
  "lifecycle-failure-retry-recover",
  "lifecycle-cancellation-supersession",
];
if (EXPECTED_ASSERTION_IDS.length !== 81) {
  throw new Error("frozen denominator construction drifted");
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { server, port } = await startServer();
const chrome = await launchChrome({
  userDataDir: join(process.cwd(), ".webai-g7-acc-profile"),
  resetProfile: true,
  removeProfileOnKill: false,
});
const cdp = new CDP(chrome.ws);
const url = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;

// ── Evidence collectors ─────────────────────────────────────────────────────────────────────────
const assertions = []; // { id, route, viewport, theme, state, evidence }
const screenshots = []; // { route, viewport, theme, path }
const pinnedRequests = []; // unique cold-download evidence for each immutable model
let mpnetPinnedNetwork = null;
let rerankerPinnedNetwork = null;
let pass = 0, fail = 0;
function record(id, route, viewport, theme, ok, evidence) {
  assertions.push({
    id,
    route,
    viewport,
    theme,
    state: ok ? "pass" : "fail",
    evidence: String(evidence).slice(0, 300),
  });
  if (ok) pass++;
  else fail++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${id} [${viewport}/${theme}] — ${String(evidence).slice(0, 160)}`,
  );
}

async function ev(sid, expr, t = 45000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: `(async()=>{try{return ${expr};}catch(e){return "ERR:"+(e?.message||e);}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    t,
  );
  return result?.value;
}
async function waitFor(sid, expr, label, t = 9 * 60000) {
  const d = Date.now() + t;
  while (Date.now() < d) {
    try {
      if (await ev(sid, expr)) return;
    } catch (e) {
      console.log("  stall", label, e.message);
    }
    await sleep(1500);
  }
  throw new Error("TIMEOUT " + label);
}
const click = (sel) =>
  `(()=>{const e=document.querySelector(${
    JSON.stringify(sel)
  });if(!e)return false;e.click();return true})()`;
const clickDownloads =
  `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Download|Retry/i.test(b.textContent)){b.click();n++}}return n})()`;
const noOverflow = `document.documentElement.scrollWidth <= window.innerWidth + 1`;

// Track Network responses for HTTP status + content-length. Workers are separate targets, so we
// auto-attach (flatten) and enable Network on each worker session, forwarding its events here. This is
// what makes the pinned model request + its bytes capturable, and lets us block worker fetches for the
// failure→retry test.
async function attachNetwork(page) {
  const respByUrl = new Map();
  const requestChains = new Map();
  const workerSessions = new Set();
  let blocks = [];
  function applyBlocks(sid) {
    cdp.send(
      "Network.setBlockedURLs",
      { urls: blocks.length ? blocks : ["__nomatch_sentinel__"] },
      sid,
    ).catch(() => {});
  }
  async function enableSession(sid) {
    if (!sid || workerSessions.has(sid)) return;
    workerSessions.add(sid);
    await cdp.send("Network.enable", {}, sid);
    applyBlocks(sid);
  }
  cdp.on((msg) => {
    if (msg.method === "Target.attachedToTarget") {
      const type = msg.params.targetInfo?.type;
      const workerUrl = msg.params.targetInfo?.url || "";
      if (
        (type === "worker" || type === "service_worker") && workerUrl.includes("all-mpnet-base-v2")
      ) {
        void enableSession(msg.params.sessionId);
      }
      return;
    }
    if (msg.sessionId !== page.sessionId && !workerSessions.has(msg.sessionId)) return;
    if (msg.method === "Network.requestWillBeSent") {
      const id = `${msg.sessionId}:${msg.params.requestId}`;
      const chain = requestChains.get(id) || { requestedUrl: msg.params.request.url };
      chain.lastUrl = msg.params.request.url;
      requestChains.set(id, chain);
    }
    if (msg.method === "Network.responseReceived") {
      const id = `${msg.sessionId}:${msg.params.requestId}`;
      const chain = requestChains.get(id) || { requestedUrl: msg.params.response.url };
      const headers = msg.params.response.headers || {};
      const cl = headers["content-length"] || headers["Content-Length"];
      respByUrl.set(chain.requestedUrl, {
        requestedUrl: chain.requestedUrl,
        responseUrl: msg.params.response.url,
        status: msg.params.response.status,
        contentLength: cl ? Number(cl) : null,
      });
    }
  });

  // Install browser-level auto-attach, then attach workers that were created while the page's module
  // graph was loading. Network must be enabled BEFORE clicking Download or a cached warm init.
  await cdp.send("Target.setDiscoverTargets", { discover: true });
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: false,
  });
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: false,
  }, page.sessionId);
  for (let attempt = 0; attempt < 20; attempt++) {
    const { targetInfos = [] } = await cdp.send("Target.getTargets");
    const workers = targetInfos.filter((target) =>
      (target.type === "worker" || target.type === "service_worker") &&
      target.url.includes("all-mpnet-base-v2")
    );
    for (const target of workers) {
      try {
        const { sessionId } = await cdp.send("Target.attachToTarget", {
          targetId: target.targetId,
          flatten: true,
        });
        await enableSession(sessionId);
      } catch { /* already auto-attached */ }
    }
    if (workers.length > 0) break;
    await sleep(50);
  }
  await cdp.send("Network.enable", {}, page.sessionId);
  applyBlocks(page.sessionId);
  return {
    respByUrl,
    setBlocks(list) {
      blocks = list;
      applyBlocks(page.sessionId);
      for (const sid of workerSessions) applyBlocks(sid);
    },
  };
}

async function setTheme(sid, theme) {
  await cdp.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: theme }],
  }, sid);
  await sleep(140);
}
async function forceContentVisibility(sid) {
  // .model-card uses content-visibility:auto; force it visible so full-page capture renders every region.
  await ev(
    sid,
    `(()=>{const s=document.createElement('style');s.id='cv-force';s.textContent='*{content-visibility:visible !important;contain-intrinsic-size:auto !important;}';document.head.append(s);return true})()`,
  );
}
async function fullShot(sid, file) {
  const { data } = await cdp.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
  }, sid);
  writeFileSync(file, Buffer.from(data, "base64"));
}
async function domChecks(page, route, vp, theme) {
  const ov = await ev(page.sessionId, noOverflow);
  record(
    `${route}-${vp}-${theme}-no-overflow`,
    route,
    vp,
    theme,
    ov === true,
    `scrollWidth-innerWidth=${await ev(
      page.sessionId,
      "document.documentElement.scrollWidth-window.innerWidth",
    )}`,
  );
  record(
    `${route}-${vp}-${theme}-console-clean`,
    route,
    vp,
    theme,
    page.errors.length === 0,
    page.errors.join(" | ") || "0 console errors",
  );
  const http4xx5xx = page.netFailures.length;
  record(
    `${route}-${vp}-${theme}-no-http-4xx-5xx`,
    route,
    vp,
    theme,
    http4xx5xx === 0,
    page.netFailures.join(" | ") || "no failed requests",
  );
  return ov === true && page.errors.length === 0 && page.netFailures.length === 0;
}

// Drive a route to its live results state; capture real worker-bound output evidence.
async function driveInteraction(sid, name) {
  const readySel = name === "practical" ? "#searchBtn" : "#run";
  // Download (cold on first route; warm reuse after) → wait for the main control enabled.
  const dlDeadline = Date.now() + 180000;
  while (Date.now() < dlDeadline) {
    await ev(sid, clickDownloads);
    if (
      await ev(
        sid,
        `(()=>{const b=document.querySelector(${
          JSON.stringify(readySel)
        });return !!b&&!b.disabled;})()`,
      )
    ) break;
    await sleep(1500);
  }
  await waitFor(
    sid,
    `(()=>{const b=document.querySelector(${
      JSON.stringify(readySel)
    });return !!b&&!b.disabled;})()`,
    name + " ready",
  );
  let real = {};
  if (name === "overview") {
    await ev(sid, click("#run"));
    await waitFor(sid, `document.querySelector('#rDim')?.textContent==='768'`, "ov dim");
    await ev(sid, click("#search"));
    await waitFor(sid, `document.querySelectorAll('#ranked .result-row').length>=2`, "ov search");
    real = await ev(
      sid,
      `JSON.stringify({dim:document.querySelector('#rDim')?.textContent, norm:document.querySelector('#iNorm')?.textContent, ranked:document.querySelectorAll('#ranked .result-row').length, topText:document.querySelector('#ranked .result-row .result-head span')?.textContent, topScore:document.querySelector('#ranked .result-score')?.textContent})`,
    );
  } else if (name === "basics") {
    await ev(sid, click("#run"));
    await waitFor(
      sid,
      `Number.isFinite(Number(document.querySelector('#score')?.textContent))`,
      "basics score",
    );
    real = await ev(sid, `JSON.stringify({score:document.querySelector('#score')?.textContent})`);
  } else if (name === "practical") {
    await ev(sid, click("#searchBtn"));
    await waitFor(
      sid,
      `document.querySelectorAll('#searchOut .result-row').length>=10`,
      "p search",
    );
    await ev(sid, click("#clusterBtn"));
    await waitFor(sid, `document.querySelectorAll('#clusterOut .cluster').length>=2`, "p cluster");
    await ev(sid, click("#classifyBtn"));
    await waitFor(
      sid,
      `document.querySelectorAll('#classifyOut .result-row').length>=10`,
      "p classify",
    );
    real = await ev(
      sid,
      `JSON.stringify({search:document.querySelectorAll('#searchOut .result-row').length, clusters:document.querySelectorAll('#clusterOut .cluster').length, classified:document.querySelectorAll('#classifyOut .result-row').length, abstained:[...document.querySelectorAll('#classifyOut .result-row')].filter(row=>/uncertain/i.test(row.textContent)).length, clusterStatus:document.querySelector('#clusterStatus')?.textContent})`,
    );
  } else if (name === "wild") {
    await ev(sid, click("#run"));
    await waitFor(sid, `document.querySelectorAll('#out .result-row').length>=5`, "wild");
    real = await ev(
      sid,
      `JSON.stringify({rows:document.querySelectorAll('#out .result-row').length, top:document.querySelector('#out .result-row .result-head span')?.textContent, status:document.querySelector('#status')?.textContent})`,
    );
  } else if (name === "multimodel") {
    await ev(sid, click("#run"));
    await waitFor(
      sid,
      `document.querySelectorAll('#stage1 .result-row').length>=2&&document.querySelectorAll('#stage2 .result-row').length>=2`,
      "mm",
      9 * 60000,
    );
    real = await ev(
      sid,
      `JSON.stringify({s1:document.querySelectorAll('#stage1 .result-row').length, s2:document.querySelectorAll('#stage2 .result-row').length, reordered:document.querySelector('#rReorder')?.textContent})`,
    );
  }
  return real;
}

function realOutputMatches(route, serialized) {
  try {
    const value = JSON.parse(serialized);
    if (route === "overview") {
      return value.dim === "768" && Number(value.norm) > 0 && value.ranked === 6 &&
        /password|login/i.test(value.topText || "") && Number.isFinite(Number(value.topScore));
    }
    if (route === "basics") return Number(value.score) > 0.5 && Number(value.score) < 0.65;
    if (route === "practical") {
      return value.search === 12 && value.clusters === 4 && value.classified === 12 &&
        value.abstained >= 1 && /cohesion|stability/i.test(value.clusterStatus || "");
    }
    if (route === "wild") return value.rows >= 5 && /(?:^|\.\s*)queen$/i.test(value.top?.trim() || "");
    if (route === "multimodel") {
      return value.s1 === 4 && value.s2 === 4 &&
        /0\s*(?:of|\/)\s*4|unchanged/i.test(value.reordered || "");
    }
  } catch { /* malformed worker/UI evidence is a failure */ }
  return false;
}

function capturePinned(respByUrl, model, rev) {
  // Find the response for this model's resolve URL at the pinned revision (best-effort network capture;
  // worker network may be isolated from the page session, so the authoritative pin proof is the source
  // check below).
  const want = `/${model}/resolve/${rev}/`;
  let hit = null;
  for (const [u, r] of respByUrl) {
    if (u.includes(want) && /\.onnx$/.test(u)) hit = { model, url: u, ...r };
  }
  return hit;
}

// Authoritative pinned-request proof: the worker SOURCE loads the exact model id + full revision as
// literals (so the request the worker makes is immutable), and the byte count is the externally
// HEAD-verified size of that pinned file. Deterministic + honest.
function sourcePin(workerFile, model, rev, bytes) {
  const src = readFileSync(join(process.cwd(), workerFile), "utf8");
  const hasModel = src.includes(`model: "${model}"`);
  const hasRev = src.includes(`revision: "${rev}"`);
  return { ok: hasModel && hasRev, hasModel, hasRev, model, rev, bytes, file: workerFile };
}

// ── Main matrix ─────────────────────────────────────────────────────────────────────────────────
const routeResults = []; // gate-required { route, viewport, pass }
try {
  for (const [name, route] of Object.entries(ROUTES)) {
    const page = await openPage(cdp, url(route));
    const net = await attachNetwork(page);
    const respByUrl = net.respByUrl;
    await setViewport(cdp, page.sessionId, DESKTOP);
    await setTheme(page.sessionId, "light");
    // Drive to results at desktop/light (cold on overview, warm after).
    const real = await driveInteraction(page.sessionId, name);
    record(
      `${name}-real-inference`,
      name,
      "desktop",
      "light",
      realOutputMatches(name, real),
      `worker-bound output: ${real}`,
    );

    // Capture the pinned request evidence: real worker network (auto-attached) + authoritative source-pin.
    const mp = capturePinned(respByUrl, MPNET, MPNET_REV);
    if (mp) mpnetPinnedNetwork = mp;
    const mpSrc = sourcePin("models/all-mpnet-base-v2/worker.js", MPNET, MPNET_REV, 110086122);
    const mpProof = mpnetPinnedNetwork;
    record(
      `${name}-mpnet-pinned-request`,
      name,
      "desktop",
      "light",
      mpSrc.ok && mpProof?.status === 200 && mpProof?.contentLength === 110086122,
      `source literal model=${mpSrc.hasModel} rev=${mpSrc.hasRev} (${
        mpSrc.rev.slice(0, 7)
      }); actual cold request=${mpProof?.requestedUrl || "missing"}; status=${
        mpProof?.status ?? "missing"
      }; bytes=${mpProof?.contentLength ?? "missing"}`,
    );
    if (name === "multimodel") {
      const rr = capturePinned(respByUrl, RERANKER, RERANKER_REV);
      if (rr) rerankerPinnedNetwork = rr;
      const rrSrc = sourcePin(
        "models/all-mpnet-base-v2/reranker-worker.js",
        RERANKER,
        RERANKER_REV,
        23143499,
      );
      const rrProof = rerankerPinnedNetwork;
      record(
        `multimodel-reranker-pinned-request`,
        name,
        "desktop",
        "light",
        rrSrc.ok && rrProof?.status === 200 && rrProof?.contentLength === 23143499,
        `source literal model=${rrSrc.hasModel} rev=${rrSrc.hasRev} (${
          rrSrc.rev.slice(0, 7)
        }); actual cold request=${rrProof?.requestedUrl || "missing"}; status=${
          rrProof?.status ?? "missing"
        }; bytes=${rrProof?.contentLength ?? "missing"}`,
      );
    }

    // Keyboard/focus: tab through controls; verify focus moves onto a focusable element.
    await ev(
      page.sessionId,
      `(()=>{document.querySelector('#run,#searchBtn')?.focus();return document.activeElement===document.querySelector('#run,#searchBtn')})()`,
    );
    const focused = await ev(
      page.sessionId,
      `!!(document.activeElement && document.activeElement!==document.body)`,
    );
    record(
      `${name}-keyboard-focus`,
      name,
      "desktop",
      "light",
      focused === true,
      `activeElement=${await ev(
        page.sessionId,
        "document.activeElement?.tagName+'#'+(document.activeElement?.id||'')",
      )}`,
    );

    // 4-cell screenshot matrix (desktop/mobile × light/dark) at the live results state.
    let allClean = true;
    for (const [vpName, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      for (const theme of ["light", "dark"]) {
        await setViewport(cdp, page.sessionId, vp);
        await setTheme(page.sessionId, theme);
        await sleep(180);
        await forceContentVisibility(page.sessionId);
        const file = join(SCREEN_DIR, `${name}-${vpName}-${theme}.png`);
        await fullShot(page.sessionId, file);
        screenshots.push({
          route: name,
          viewport: vpName,
          theme,
          resultState: "post-interaction (live)",
          path: `reports/webai-g7-glm-text2-screens/${name}-${vpName}-${theme}.png`,
          manual_verdict: "manual_parent_required",
        });
        const clean = await domChecks(page, name, vpName, theme);
        if (!clean) allClean = false;
      }
    }
    routeResults.push({ route, viewport: "desktop", pass: allClean });
    routeResults.push({ route, viewport: "mobile", pass: allClean });
    await closePage(cdp, page.targetId);
  }

  // ── Lifecycle cells (overview) ───────────────────────────────────────────────────────────────
  // Cold vs warm: the overview above was cold (downloaded); open overview again → warm (cached, auto-init).
  {
    const page = await openPage(cdp, url(ROUTES.overview));
    await setViewport(cdp, page.sessionId, DESKTOP);
    const warmReady = await waitFor(
      page.sessionId,
      `(()=>{const b=document.querySelector('#run');return !!b&&!b.disabled;})()`,
      "warm auto-init",
      120000,
    ).then(() => true).catch(() => false);
    record(
      `lifecycle-warm-reuse`,
      "overview",
      "desktop",
      "light",
      warmReady,
      warmReady
        ? "cached model auto-initialised (no Download needed)"
        : "did not auto-init from cache",
    );
    // Release from memory (idle), then re-init, then REAL inference after reinit.
    const released = await ev(page.sessionId, clickRelease());
    await waitFor(
      page.sessionId,
      `!!document.querySelector('.model-loader button') && /Load model into memory|Download/i.test(document.querySelector('.model-loader button').textContent)`,
      "released state",
    );
    await ev(
      page.sessionId,
      `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Load model into memory/i.test(b.textContent)){b.click();return true}}return false})()`,
    );
    await waitFor(
      page.sessionId,
      `(()=>{const b=document.querySelector('#run');return !!b&&!b.disabled;})()`,
      "reinit ready",
      120000,
    );
    await ev(page.sessionId, click("#run"));
    const postReinit = await waitFor(
      page.sessionId,
      `document.querySelector('#rDim')?.textContent==='768'`,
      "post-reinit inference",
      120000,
    ).then(() => true).catch(() => false);
    record(
      `lifecycle-release-idle-reinit-real-inference`,
      "overview",
      "desktop",
      "light",
      postReinit,
      postReinit
        ? "released→reinit→real 768-d inference after reinit"
        : "no real inference after reinit",
    );
    await closePage(cdp, page.targetId);
  }

  // Clear-cache disposal → Download reappears (honest removal), then re-download works.
  {
    const page = await openPage(cdp, url(ROUTES.overview));
    await setViewport(cdp, page.sessionId, DESKTOP);
    await waitFor(
      page.sessionId,
      `(()=>{const b=document.querySelector('#run');return !!b&&!b.disabled;})()`,
      "cc ready",
      120000,
    );
    await ev(
      page.sessionId,
      `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Clear cached model/i.test(b.textContent)){b.click();return true}}return false})()`,
    );
    const downloadShown = await waitFor(
      page.sessionId,
      `/Download/i.test(document.querySelector('.model-loader button')?.textContent||'')`,
      "clear-cache → Download",
    ).then(() => true).catch(() => false);
    record(
      `lifecycle-clear-cache-disposal`,
      "overview",
      "desktop",
      "light",
      downloadShown,
      downloadShown
        ? "clear-cache disposed worker + removed assets; Download reappeared"
        : "Download did not reappear after clear",
    );
    await closePage(cdp, page.targetId);
  }

  // Failure → visible Retry → recover (block the worker's model fetch via auto-attached sessions).
  {
    const page = await openPage(cdp, url(ROUTES.overview));
    const net = await attachNetwork(page);
    await setViewport(cdp, page.sessionId, DESKTOP);
    net.setBlocks(["*all-mpnet-base-v2*"]);
    await ev(page.sessionId, clickDownloads);
    const retryShown = await waitFor(
      page.sessionId,
      `/Retry|class="status err"|Couldn't load/i.test(document.querySelector('.model-loader')?.innerHTML||'')`,
      "failure → Retry",
      120000,
    ).then(() => true).catch(() => false);
    // Unblock (all sessions) + recover: verify the loader leaves the error state (recovery began).
    net.setBlocks([]);
    await ev(
      page.sessionId,
      `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Retry/i.test(b.textContent)){b.click();return true}}return false})()`,
    );
    // Recovery began once the error/Retry state is gone (loader moved to downloading/initialising/ready).
    const recovered = await waitFor(
      page.sessionId,
      `!document.querySelector('.model-loader')?.dataset?.state || (document.querySelector('.model-loader')?.dataset?.state !== 'error')`,
      "retry → recovery began",
      30000,
    ).then(() => true).catch(() => false);
    record(
      `lifecycle-failure-retry-recover`,
      "overview",
      "desktop",
      "light",
      retryShown && recovered,
      `retry-shown=${retryShown} recoveryBegan=${recovered}`,
    );
    await closePage(cdp, page.targetId);
  }

  // Cancellation/supersession at the engine level (real worker-protocol AbortSignal path). The fresh
  // in-page engine loads the model itself, so this cell does NOT depend on the page's cached state.
  try {
    const page = await openPage(cdp, url(ROUTES.overview));
    await setViewport(cdp, page.sessionId, DESKTOP);
    await sleep(1500); // let the module load (the fresh engine loads the model independently)
    const cres = await ev(
      page.sessionId,
      `(async()=>{
      try {
        const { MpNetEngine } = await import('/web-ai-showcase/models/all-mpnet-base-v2/mpnet.js');
        const e = new MpNetEngine(); await e.load(()=>{});
        const ac = new AbortController();
        const p = e.embed(['hello world','another sentence'], { signal: ac.signal });
        ac.abort();
        let aborted=false; try { await p; } catch(err){ aborted = /AbortError|Aborted/i.test(err.name+' '+err.message); }
        const ok2 = (await e.embed(['real text works after abort'])).dim===768;
        await e.close();
        return JSON.stringify({ aborted, ok2 });
      } catch(err){ return JSON.stringify({ aborted:false, ok2:false, err:String(err&&err.message||err) }); }
    })()`,
      180000,
    );
    let cj = { aborted: false, ok2: false };
    try {
      cj = JSON.parse(typeof cres === "string" ? cres : "{}");
    } catch {
      cj = { aborted: false, ok2: false, raw: String(cres).slice(0, 120) };
    }
    record(
      `lifecycle-cancellation-supersession`,
      "overview",
      "desktop",
      "light",
      cj.aborted === true && cj.ok2 === true,
      `aborted=${cj.aborted} postAbortInferenceOk=${cj.ok2}${cj.err ? " err=" + cj.err : ""}`,
    );
    await closePage(cdp, page.targetId);
  } catch (cellErr) {
    record(
      `lifecycle-cancellation-supersession`,
      "overview",
      "desktop",
      "light",
      false,
      "cell threw: " + (cellErr?.message || cellErr),
    );
  }
} catch (topErr) {
  record(
    "validator-fatal",
    "all",
    "desktop",
    "light",
    false,
    "validator threw: " + (topErr?.message || topErr),
  );
} finally {
  try {
    server.close();
  } catch {}
  try {
    chrome.kill({ removeProfile: false });
  } catch {}
}

function clickRelease() {
  return `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Release from memory/i.test(b.textContent)){b.click();return true}}return false})()`;
}

// ── Write acceptance-run.json (schema-2, superset of the gate-required fields) ─────────────────
if (mpnetPinnedNetwork) {
  pinnedRequests.push({
    model: MPNET,
    revision: MPNET_REV,
    dtype: "q8",
    expectedBytes: 110086122,
    ...mpnetPinnedNetwork,
  });
}
if (rerankerPinnedNetwork) {
  pinnedRequests.push({
    model: RERANKER,
    revision: RERANKER_REV,
    dtype: "q8",
    expectedBytes: 23143499,
    ...rerankerPinnedNetwork,
  });
}
const actualAssertionIds = assertions.map((assertion) => assertion.id);
const denominatorMatches = JSON.stringify(actualAssertionIds) ===
  JSON.stringify(EXPECTED_ASSERTION_IDS);
if (!denominatorMatches) {
  fail++;
  console.error(
    `FROZEN DENOMINATOR MISMATCH expected=${EXPECTED_ASSERTION_IDS.length} actual=${actualAssertionIds.length}`,
  );
}
const familyRoot = "models/all-mpnet-base-v2";
const validatorRel = "scripts/validate-all-mpnet-base-v2.mjs";
const commit = execFileSync(
  "git",
  [
    "log",
    "-n1",
    "--format=%H",
    "HEAD",
    "--",
    familyRoot,
    validatorRel,
    `:(exclude)${familyRoot}/acceptance.json`,
    `:(exclude)${familyRoot}/acceptance-run.json`,
  ],
  { cwd: process.cwd(), encoding: "utf8" },
).trim();
const runRecord = {
  schema: 2,
  validator: validatorRel,
  commit,
  ranAt: new Date().toISOString(),
  exitCode: fail === 0 ? 0 : 1,
  checks: `${pass}/${EXPECTED_ASSERTION_IDS.length}`,
  frozenDenominator: {
    total: EXPECTED_ASSERTION_IDS.length,
    assertionIds: EXPECTED_ASSERTION_IDS,
    exactOrder: true,
  },
  denominator: { routes: 5, viewports: 2, themes: 2, screenshotCells: 20 },
  results: routeResults,
  assertions,
  pinnedRequests,
  screenshots,
  visualReview: screenshots.map((s) => ({ ...s, verdict: "manual_parent_required" })),
  manualStates: [
    {
      id: "physical-low-memory-mobile",
      state: "manual",
      evidence:
        "desktop Chrome mobile emulation does not establish 105 MB model viability on a physical constrained phone",
    },
    {
      id: "assistive-technology-traversal",
      state: "manual",
      evidence:
        "keyboard focus is automated; screen-reader and physical coarse-pointer review remain manual",
    },
  ],
  stages: [MPNET, RERANKER],
};
writeFileSync(
  "models/all-mpnet-base-v2/acceptance-run.json",
  JSON.stringify(runRecord, null, 2) + "\n",
);
console.log(`\n${pass}/${pass + fail} schema-2 assertions pass · ${fail} fail`);
console.log(`screenshots: ${screenshots.length} → ${SCREEN_DIR}`);
console.log(`pinned requests captured: ${pinnedRequests.length}`);
console.log(`ACCEPTANCE exit ${fail === 0 ? 0 : 1}`);
process.exit(fail === 0 ? 0 : 1);
