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
    try {
      await cdp.send("Network.enable", {}, sid);
      applyBlocks(sid);
    } catch {
      // A short-lived worker may disappear between Target attachment and Network.enable. It cannot
      // contribute request evidence after detaching, so remove it without creating an unhandled rejection.
      workerSessions.delete(sid);
    }
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
    await waitFor(
      sid,
      `document.querySelector('#rDim')?.textContent==='768'&&!document.querySelector('#run')?.disabled`,
      "ov dim",
    );
    await ev(
      sid,
      `(()=>{document.querySelector('#ranked').replaceChildren();document.querySelector('#search').click();return true})()`,
    );
    await waitFor(
      sid,
      `document.querySelectorAll('#ranked .result-row').length>=2&&!document.querySelector('#search')?.disabled`,
      "ov search",
    );
    const resetWorked = await ev(
      sid,
      `(()=>{document.querySelector('#sentences').value='changed';document.querySelector('#reset').click();return /password/i.test(document.querySelector('#sentences').value)})()`,
    );
    await ev(
      sid,
      `(()=>{const q=document.querySelector('#query');q.value='refund policy';q.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#search').click();return true})()`,
    );
    await waitFor(
      sid,
      `/refund/i.test(document.querySelector('#ranked')?.textContent||'')&&!document.querySelector('#search')?.disabled`,
      "ov edited search",
    );
    await ev(
      sid,
      `(()=>{const q=document.querySelector('#query');q.value="I can't sign in";q.dispatchEvent(new Event('input',{bubbles:true}));const p=document.querySelector('#pick');if(p.options.length>1){p.selectedIndex=1;p.dispatchEvent(new Event('change',{bubbles:true}))}document.querySelector('#search').click();return true})()`,
    );
    await waitFor(
      sid,
      `document.querySelectorAll('#ranked .result-row').length===6&&!document.querySelector('#search')?.disabled`,
      "ov restored search",
    );
    real = await ev(
      sid,
      `JSON.stringify({dim:document.querySelector('#rDim')?.textContent, norm:document.querySelector('#iNorm')?.textContent, ranked:document.querySelectorAll('#ranked .result-row').length, topText:document.querySelector('#ranked .result-row .result-head span')?.textContent, topScore:document.querySelector('#ranked .result-score')?.textContent, resetWorked:${resetWorked}, controls:['sentences','run','reset','query','search','pick']})`,
    );
  } else if (name === "basics") {
    await ev(sid, click("#run"));
    await waitFor(
      sid,
      `Number.isFinite(Number(document.querySelector('#score')?.textContent))`,
      "basics score",
    );
    const defaultScore = await ev(sid, `document.querySelector('#score')?.textContent`);
    const presets = await ev(
      sid,
      `(async()=>{const out=[];for(const chip of document.querySelectorAll('.chip')){chip.click();document.querySelector('#run').click();while(document.querySelector('#run').disabled)await new Promise(r=>setTimeout(r,25));out.push({label:chip.textContent.trim(),score:Number(document.querySelector('#score').textContent)})}const a=document.querySelector('#a'),b=document.querySelector('#b');a.value='A browser stores data locally.';b.value='Local storage keeps browser data.';a.dispatchEvent(new Event('input',{bubbles:true}));b.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#run').click();while(document.querySelector('#run').disabled)await new Promise(r=>setTimeout(r,25));const edited=Number(document.querySelector('#score').textContent);document.querySelector('.chip').click();document.querySelector('#run').click();while(document.querySelector('#run').disabled)await new Promise(r=>setTimeout(r,25));return JSON.stringify({presets:out,edited})})()`,
      180000,
    );
    real = JSON.stringify({ defaultScore, ...JSON.parse(presets) });
  } else if (name === "practical") {
    await ev(
      sid,
      `(()=>{const q=document.querySelector('#q');q.value="I'm locked out and can't sign in";q.dispatchEvent(new Event('input',{bubbles:true}));const k=document.querySelector('#k');k.value='3';k.dispatchEvent(new Event('input',{bubbles:true}));k.value='4';k.dispatchEvent(new Event('input',{bubbles:true}));const l=document.querySelector('#labels');l.dispatchEvent(new Event('input',{bubbles:true}));return true})()`,
    );
    await ev(sid, click("#searchBtn"));
    await waitFor(
      sid,
      `document.querySelectorAll('#searchOut .result-row').length>=10&&!document.querySelector('#searchBtn')?.disabled`,
      "p search",
    );
    await ev(sid, click("#clusterBtn"));
    await waitFor(
      sid,
      `document.querySelectorAll('#clusterOut .cluster').length>=2&&!document.querySelector('#clusterBtn')?.disabled`,
      "p cluster",
    );
    await ev(sid, click("#classifyBtn"));
    await waitFor(
      sid,
      `document.querySelectorAll('#classifyOut .result-row').length>=10&&!document.querySelector('#classifyBtn')?.disabled`,
      "p classify",
    );
    real = await ev(
      sid,
      `JSON.stringify({search:document.querySelectorAll('#searchOut .result-row').length, clusters:document.querySelectorAll('#clusterOut .cluster').length, classified:document.querySelectorAll('#classifyOut .result-row').length, abstained:[...document.querySelectorAll('#classifyOut .result-row')].filter(row=>/uncertain/i.test(row.textContent)).length, clusterStatus:document.querySelector('#clusterStatus')?.textContent, controls:['corpus','q','searchBtn','k','clusterBtn','labels','classifyBtn']})`,
    );
  } else if (name === "wild") {
    const presetJson = await ev(
      sid,
      `(async()=>{const out=[];for(const chip of document.querySelectorAll('.chip')){chip.click();document.querySelector('#run').click();while(document.querySelector('#run').disabled)await new Promise(r=>setTimeout(r,25));out.push({label:chip.textContent.trim(),top:document.querySelector('#out .result-row .result-head span')?.textContent})}const x=document.querySelector('#excl');x.click();x.click();document.querySelector('.chip').click();document.querySelector('#run').click();while(document.querySelector('#run').disabled)await new Promise(r=>setTimeout(r,25));return JSON.stringify(out)})()`,
      180000,
    );
    await waitFor(sid, `document.querySelectorAll('#out .result-row').length>=5`, "wild");
    real = await ev(
      sid,
      `JSON.stringify({rows:document.querySelectorAll('#out .result-row').length, top:document.querySelector('#out .result-row .result-head span')?.textContent, status:document.querySelector('#status')?.textContent, presets:${presetJson}, controls:['preset chips','a','b','c','run','vocab','excl']})`,
    );
  } else if (name === "multimodel") {
    await ev(
      sid,
      `(()=>{const q=document.querySelector('#q');q.dispatchEvent(new Event('input',{bubbles:true}));const k=document.querySelector('#k');k.value='3';k.dispatchEvent(new Event('input',{bubbles:true}));k.value='4';k.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#corpus').dispatchEvent(new Event('input',{bubbles:true}));return true})()`,
    );
    await ev(sid, click("#run"));
    await waitFor(
      sid,
      `document.querySelectorAll('#stage1 .result-row').length>=2&&document.querySelectorAll('#stage2 .result-row').length>=2&&!document.querySelector('#run')?.disabled`,
      "mm",
      9 * 60000,
    );
    real = await ev(
      sid,
      `JSON.stringify({s1:document.querySelectorAll('#stage1 .result-row').length, s2:document.querySelectorAll('#stage2 .result-row').length, reordered:document.querySelector('#rReorder')?.textContent, controls:['corpus','q','k','run'], stages:['mpnet','cross-encoder']})`,
    );
  }
  return real;
}

function realOutputMatches(route, serialized) {
  try {
    const value = JSON.parse(serialized);
    if (route === "overview") {
      return value.dim === "768" && Number(value.norm) > 0 && value.ranked === 6 &&
        /password|login/i.test(value.topText || "") && Number.isFinite(Number(value.topScore)) &&
        value.resetWorked === true && value.controls?.length === 6;
    }
    if (route === "basics") {
      return Number(value.defaultScore) > 0.5 && Number(value.defaultScore) < 0.65 &&
        value.presets?.length === 4 && value.presets.every((preset) =>
          Number.isFinite(preset.score)
        ) &&
        Number.isFinite(value.edited);
    }
    if (route === "practical") {
      return value.search === 12 && value.clusters === 4 && value.classified === 12 &&
        value.abstained >= 1 && /cohesion|stability/i.test(value.clusterStatus || "") &&
        value.controls?.length === 7;
    }
    if (route === "wild") {
      return value.rows >= 5 && /(?:^|\.\s*)queen$/i.test(value.top?.trim() || "") &&
        value.presets?.length === 4 && value.presets.every((preset) => preset.top) &&
        value.controls?.length === 7;
    }
    if (route === "multimodel") {
      return value.s1 === 4 && value.s2 === 4 &&
        /0\s*(?:of|\/)\s*4|unchanged/i.test(value.reordered || "") &&
        value.controls?.length === 4 && value.stages?.length === 2;
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

  // ── Lifecycle cells ─────────────────────────────────────────────────────────────────────────
  // Warm reuse is real cache reuse, not a source-only claim.
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
    await closePage(cdp, page.targetId);
  }

  // Exercise busy release and reinitialisation with real post-reinit inference on every route.
  const releaseEvidence = [];
  for (const [name, route] of Object.entries(ROUTES)) {
    const page = await openPage(cdp, url(route));
    await setViewport(cdp, page.sessionId, DESKTOP);
    await driveInteraction(page.sessionId, name);
    const main = name === "practical" ? "#clusterBtn" : "#run";
    const busy = await ev(
      page.sessionId,
      `(()=>{const run=document.querySelector(${
        JSON.stringify(main)
      });run.click();const busy=run.disabled;let released=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Release from memory/i.test(b.textContent)){b.click();released++}}return {busy,released,loaders:document.querySelectorAll('.model-loader').length}})()`,
    );
    const released = await waitFor(
      page.sessionId,
      `[...document.querySelectorAll('.model-loader button')].filter(b=>/Load model into memory|Download/i.test(b.textContent)).length===document.querySelectorAll('.model-loader').length`,
      `${name} busy release`,
      120000,
    ).then(() => true).catch(() => false);
    await ev(
      page.sessionId,
      `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Load model into memory/i.test(b.textContent)){b.click();n++}}return n})()`,
    );
    const post = await driveInteraction(page.sessionId, name);
    const postOk = realOutputMatches(name, post);
    releaseEvidence.push({
      name,
      busy: busy?.busy === true,
      released,
      loaders: busy?.loaders,
      releasedCount: busy?.released,
      postOk,
    });
    await closePage(cdp, page.targetId);
  }
  record(
    `lifecycle-release-idle-reinit-real-inference`,
    "all routes",
    "desktop",
    "light",
    releaseEvidence.length === 5 &&
      releaseEvidence.every((item) =>
        item.busy && item.released && item.releasedCount === item.loaders && item.postOk
      ),
    JSON.stringify(releaseEvidence),
  );

  // Exercise busy clear-cache, redownload/reinit, and real inference on every route and both Multi-model stages.
  const clearEvidence = [];
  for (const [name, route] of Object.entries(ROUTES)) {
    const page = await openPage(cdp, url(route));
    await setViewport(cdp, page.sessionId, DESKTOP);
    await driveInteraction(page.sessionId, name);
    const main = name === "practical" ? "#clusterBtn" : "#run";
    const busy = await ev(
      page.sessionId,
      `(()=>{const run=document.querySelector(${
        JSON.stringify(main)
      });run.click();const busy=run.disabled;let cleared=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Clear cached model/i.test(b.textContent)){b.click();cleared++}}return {busy,cleared,loaders:document.querySelectorAll('.model-loader').length}})()`,
    );
    const downloadShown = await waitFor(
      page.sessionId,
      `[...document.querySelectorAll('.model-loader button')].filter(b=>/Download/i.test(b.textContent)).length===document.querySelectorAll('.model-loader').length`,
      `${name} busy clear → Download`,
      120000,
    ).then(() => true).catch(() => false);
    await ev(page.sessionId, clickDownloads);
    const post = await driveInteraction(page.sessionId, name);
    const postOk = realOutputMatches(name, post);
    clearEvidence.push({
      name,
      busy: busy?.busy === true,
      downloadShown,
      loaders: busy?.loaders,
      clearedCount: busy?.cleared,
      postOk,
    });
    await closePage(cdp, page.targetId);
  }
  record(
    `lifecycle-clear-cache-disposal`,
    "all routes",
    "desktop",
    "light",
    clearEvidence.length === 5 &&
      clearEvidence.every((item) =>
        item.busy && item.downloadShown && item.clearedCount === item.loaders && item.postOk
      ),
    JSON.stringify(clearEvidence),
  );

  // Failure → visible Retry → recover (block the worker's model fetch via auto-attached sessions).
  {
    const page = await openPage(cdp, url(ROUTES.overview));
    const net = await attachNetwork(page);
    await setViewport(cdp, page.sessionId, DESKTOP);
    await waitFor(
      page.sessionId,
      `!document.querySelector('#run')?.disabled`,
      "failure setup ready",
      120000,
    );
    await ev(
      page.sessionId,
      `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Clear cached model/i.test(b.textContent)){b.click();return true}}return false})()`,
    );
    await waitFor(
      page.sessionId,
      `/Download/i.test(document.querySelector('.model-loader button')?.textContent||'')`,
      "failure setup cleared",
      120000,
    );
    net.setBlocks(["*all-mpnet-base-v2*"]);
    await ev(page.sessionId, clickDownloads);
    const retryShown = await waitFor(
      page.sessionId,
      `/Retry|class="status err"|Couldn't load/i.test(document.querySelector('.model-loader')?.innerHTML||'')`,
      "failure → Retry",
      120000,
    ).then(() => true).catch(() => false);
    // Unblock every attached worker target, then require ready plus real post-retry inference.
    net.setBlocks([]);
    await ev(
      page.sessionId,
      `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Retry/i.test(b.textContent)){b.click();return true}}return false})()`,
    );
    const recovered = await waitFor(
      page.sessionId,
      `!document.querySelector('#run')?.disabled`,
      "retry → ready",
      180000,
    ).then(() => true).catch(() => false);
    if (recovered) await ev(page.sessionId, click("#run"));
    const postRetryInference = recovered && await waitFor(
      page.sessionId,
      `document.querySelector('#rDim')?.textContent==='768'`,
      "retry → real inference",
      120000,
    ).then(() => true).catch(() => false);
    record(
      `lifecycle-failure-retry-recover`,
      "overview",
      "desktop",
      "light",
      retryShown && recovered && postRetryInference,
      `retry-shown=${retryShown} ready=${recovered} postRetry768=${postRetryInference}`,
    );
    await closePage(cdp, page.targetId);
  }

  // Exercise actual model cancellation plus protocol latest-wins, stale suppression, bounded queue,
  // fatal rejection, and recovery through a genuinely fresh worker instance.
  try {
    const page = await openPage(cdp, url(ROUTES.overview));
    await setViewport(cdp, page.sessionId, DESKTOP);
    await sleep(1500);
    const cres = await ev(
      page.sessionId,
      `(async()=>{
      try {
        const { MpNetEngine } = await import('/web-ai-showcase/models/all-mpnet-base-v2/mpnet.js');
        const { WorkerClient } = await import('/web-ai-showcase/lib/worker-protocol.js');
        const e = new MpNetEngine(); await e.load(()=>{});
        const ac = new AbortController();
        const p = e.embed(['hello world','another sentence'], { signal: ac.signal });
        ac.abort();
        let aborted=false; try { await p; } catch(err){ aborted = /AbortError|Aborted/i.test(err.name+' '+err.message); }
        const modelSuperseded = e.embed(['obsolete request'], { channel:'live' }).then(()=>false,err=>/Superseded/i.test(err.name+' '+err.message));
        const modelLatest = e.embed(['real text works after supersession'], { channel:'live' });
        const superseded = await modelSuperseded;
        const latestDim = (await modelLatest).dim;
        await e.close();

        const workerCode = "self.postMessage({p:1,kind:'ready'});self.onmessage=e=>{const m=e.data;if(m.kind==='dispose')return close();if(m.kind!=='request')return;if(m.method==='fatal'){setTimeout(()=>{throw new Error('deliberate fatal')},0);return}const delay=m.method==='slow'?80:10;setTimeout(()=>self.postMessage({p:1,kind:'response',id:m.id,result:{method:m.method}}),delay)}";
        const makeClient = async (name,maxQueue) => { const u=URL.createObjectURL(new Blob([workerCode],{type:'text/javascript'}));const c=new WorkerClient({url:u,name,maxInFlight:1,maxQueue,module:false,disposeGraceMs:0});c.registerObjectURL(u);await c.ready;return c };
        const s = await makeClient('supersession',1);
        const old = s.request('slow',{}, {channel:'live'}).then(()=>false,err=>/Superseded/i.test(err.name+' '+err.message));
        const newest = s.request('fast',{}, {channel:'live'});
        const oldRejected = await old;
        const newestResult = (await newest).result?.method;
        await new Promise(r=>setTimeout(r,110));
        const staleSuppressed = s.state==='ready' && s._reqs.size===0 && s._inflight===0;
        s.terminate();

        const q = await makeClient('queue',0);
        const q1=q.request('slow',{});
        const overflow=await q.request('fast',{}).then(()=>false,err=>err.name==='QueueOverflowError');
        await q1; q.terminate();

        const f = await makeClient('fatal',1);
        const currentRejected=await f.request('fatal',{}).then(()=>false,()=>true);
        const futureRejected=await f.request('fast',{}).then(()=>false,()=>true);
        const fatalState=f.state==='error' && f._reqs.size===0;
        f.terminate();
        const fresh=await makeClient('fresh',1);
        const freshWorked=(await fresh.request('fast',{})).result?.method==='fast';
        fresh.terminate();
        return JSON.stringify({aborted,superseded,latestDim,oldRejected,newestResult,staleSuppressed,overflow,currentRejected,futureRejected,fatalState,freshWorked});
      } catch(err){ return JSON.stringify({err:String(err&&err.stack||err)}); }
    })()`,
      240000,
    );
    let cj = {};
    try {
      cj = JSON.parse(typeof cres === "string" ? cres : "{}");
    } catch {
      cj = { raw: String(cres).slice(0, 200) };
    }
    const complete = cj.aborted === true && cj.superseded === true && cj.latestDim === 768 &&
      cj.oldRejected === true && cj.newestResult === "fast" && cj.staleSuppressed === true &&
      cj.overflow === true && cj.currentRejected === true && cj.futureRejected === true &&
      cj.fatalState === true && cj.freshWorked === true;
    record(
      `lifecycle-cancellation-supersession`,
      "shared protocol + real MPNet worker",
      "desktop",
      "light",
      complete,
      JSON.stringify(cj),
    );
    await closePage(cdp, page.targetId);
  } catch (cellErr) {
    record(
      `lifecycle-cancellation-supersession`,
      "shared protocol + real MPNet worker",
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
