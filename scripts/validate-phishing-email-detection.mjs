// Route-complete real-browser acceptance for cybersectony/phishing-email-detection-distilbert_v2.4.1.
// Drives every overview/Basics/Practical/Wild/Multi-model route at desktop + mobile.
// Every advertised stage runs for real: onnx-community/phishing-email-detection-distilbert_v2.4.1-ONNX
// (all routes) and Xenova/distilbart-cnn-6-6 (multi-model summarizer).
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

const MODEL_ID = "onnx-community/phishing-email-detection-distilbert_v2.4.1-ONNX";
const ROUTES = {
  overview: "models/phishing-email-detection/",
  basics: "models/phishing-email-detection/basics/",
  practical: "models/phishing-email-detection/practical/",
  wild: "models/phishing-email-detection/wild/",
  multimodel: "models/phishing-email-detection/multi-model/",
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const url = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;
let passed = 0;
let total = 0;
const results = [];

function check(name, condition, detail = "") {
  total++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${String(detail).slice(0, 240)}` : ""}`,
  );
  return Boolean(condition);
}
async function evaluate(sessionId, expression, timeoutMs = 45_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(e){return "ERR:"+(e?.message||e)}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}
async function waitFor(sessionId, expression, label, timeoutMs = 10 * 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await evaluate(sessionId, expression)) return;
    } catch (error) {
      console.log(`  [${label}] transient CDP stall: ${error.message}`);
    }
    await sleep(2000);
  }
  throw new Error(`TIMEOUT: ${label}`);
}
const click = (selector) =>
  `(()=>{const e=document.querySelector(${
    JSON.stringify(selector)
  });if(!e)return false;e.click();return true})()`;
const clickDownloads =
  `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Download|Retry/i.test(b.textContent)){b.click();n++}}return n})()`;
const noOverflow = `document.documentElement.scrollWidth <= window.innerWidth + 1`;

async function readyTarget(page, firstVisit) {
  await sleep(1500);
  if (firstVisit) {
    const state = await evaluate(
      page.sessionId,
      `JSON.stringify([...document.querySelectorAll('.model-loader button')].map(b=>b.textContent.trim()))`,
    );
    check(`fresh profile exposes Download for ${MODEL_ID}`, /Download/.test(state), state);
  }
  await evaluate(page.sessionId, clickDownloads);
  await waitFor(page.sessionId, `!document.querySelector('#text')?.disabled`, "target model ready");
}
async function hygiene(page, route, viewport) {
  const overflow = await evaluate(page.sessionId, noOverflow);
  return [
    check(`${viewport} ${route}: no horizontal overflow`, overflow),
    check(
      `${viewport} ${route}: zero console errors`,
      page.errors.length === 0,
      page.errors.join(" | "),
    ),
    check(
      `${viewport} ${route}: zero failed network requests`,
      page.netFailures.length === 0,
      page.netFailures.join(" | "),
    ),
  ].every(Boolean);
}
async function exercise(routeName, viewportName, viewport, firstVisit = false) {
  const route = ROUTES[routeName];
  const page = await openPage(cdp, url(route));
  await setViewport(cdp, page.sessionId, viewport);
  let ok = true;
  try {
    if (routeName === "overview") {
      await readyTarget(page, firstVisit);
      // Default phishing sample auto-scores on ready.
      await waitFor(
        page.sessionId,
        `document.querySelector('#vLabel')?.textContent==='PHISHING'`,
        "overview auto-verdict",
      );
      const snap = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({label:document.querySelector('#vLabel')?.textContent,conf:document.querySelector('#vConf')?.textContent,bars:document.querySelectorAll('#bars .class-row').length,backend:document.querySelector('#rBackend')?.textContent})`,
        ),
      );
      ok = check(
        `${viewportName} overview: real PHISHING verdict + 4 class bars`,
        snap.label === "PHISHING" && parseFloat(snap.conf) > 50 && snap.bars === 4 &&
          snap.backend === "WASM",
        JSON.stringify(snap),
      ) && ok;
      // See-inside occlusion attribution over the same message.
      await evaluate(page.sessionId, click("#explain"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#attr .attr-word').length>=5`,
        "overview occlusion attribution",
      );
      ok = check(
        `${viewportName} overview: real occlusion attribution words`,
        await evaluate(
          page.sessionId,
          `document.querySelectorAll('#attr .attr-word').length>=5`,
        ),
      ) && ok;
    } else if (routeName === "basics") {
      await readyTarget(page, false);
      await waitFor(
        page.sessionId,
        `document.querySelector('#vLabel')?.textContent==='PHISHING'`,
        "Basics auto-verdict",
      );
      const snap = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({label:document.querySelector('#vLabel')?.textContent,bars:document.querySelectorAll('#bars .class-row').length,ms:document.querySelector('#rMs')?.textContent})`,
        ),
      );
      ok = check(
        `${viewportName} Basics: real verdict + 4-class split + latency`,
        snap.label === "PHISHING" && snap.bars === 4 && /ms/.test(snap.ms),
        JSON.stringify(snap),
      ) && ok;
      // Flip to a legitimate chip and watch the verdict swing.
      await evaluate(
        page.sessionId,
        `(()=>{for(const b of document.querySelectorAll('.chip')){if(/colleague/i.test(b.textContent)){b.click();return true}}return false})()`,
      );
      await waitFor(
        page.sessionId,
        `document.querySelector('#vLabel')?.textContent==='LEGITIMATE'`,
        "Basics verdict swing",
      );
      ok = check(
        `${viewportName} Basics: verdict swings to LEGITIMATE on benign text`,
        await evaluate(
          page.sessionId,
          `document.querySelector('#vLabel')?.textContent==='LEGITIMATE'`,
        ),
      ) && ok;
    } else if (routeName === "practical") {
      await sleep(1200);
      await evaluate(page.sessionId, clickDownloads);
      await waitFor(page.sessionId, `!document.querySelector('#run')?.disabled`, "Practical ready");
      await evaluate(page.sessionId, click("#run"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#inboxList .triage-row').length + document.querySelectorAll('#junkList .triage-row').length >= 10`,
        "Practical triage",
      );
      const counts = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({clear:document.querySelectorAll('#inboxList .triage-row').length,threat:document.querySelectorAll('#junkList .triage-row').length,status:document.querySelector('#runStatus')?.textContent})`,
        ),
      );
      ok = check(
        `${viewportName} Practical: real queue triage (both columns populated)`,
        counts.clear >= 1 && counts.threat >= 1 && counts.clear + counts.threat >= 10,
        JSON.stringify(counts),
      ) && ok;
    } else if (routeName === "wild") {
      await readyTarget(page, false);
      await waitFor(
        page.sessionId,
        `!document.querySelector('#cmp')?.hidden && document.querySelector('#oLabel')?.textContent.length>1`,
        "Wild comparison",
      );
      const snap = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({o:document.querySelector('#oLabel')?.textContent,x:document.querySelector('#xLabel')?.textContent,drop:document.querySelector('#drop')?.textContent,obf:document.querySelector('#obfOut')?.textContent?.length})`,
        ),
      );
      ok = check(
        `${viewportName} Wild: original vs obfuscated both really scored`,
        snap.o === "PHISHING" && snap.x.length > 1 && /pts/.test(snap.drop) && snap.obf > 20,
        JSON.stringify(snap),
      ) && ok;
    } else {
      await sleep(1200);
      await evaluate(page.sessionId, clickDownloads);
      await waitFor(page.sessionId, `!document.querySelector('#gate')?.disabled`, "gate ready");
      await evaluate(page.sessionId, click("#gate"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#keptList .triage-row').length + document.querySelectorAll('#junkList .triage-row').length >= 8`,
        "phish gate",
      );
      const gate = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({kept:document.querySelectorAll('#keptList .triage-row').length,junk:document.querySelectorAll('#junkList .triage-row').length})`,
        ),
      );
      ok = check(
        `${viewportName} Multi-model: phishing gate really classified`,
        gate.kept >= 1 && gate.junk >= 1,
        JSON.stringify(gate),
      ) && ok;
      // Stage 2: DistilBART digest of the kept (legitimate) messages only.
      await evaluate(page.sessionId, clickDownloads);
      await waitFor(
        page.sessionId,
        `!document.querySelector('#summarize')?.disabled`,
        "summarizer ready",
        10 * 60_000,
      );
      await evaluate(page.sessionId, click("#summarize"));
      await waitFor(
        page.sessionId,
        `(document.querySelector('#out')?.textContent||'').length>30`,
        "digest generated",
        10 * 60_000,
      );
      const digest = await evaluate(
        page.sessionId,
        `(document.querySelector('#out')?.textContent||'').slice(0,120)`,
      );
      ok = check(
        `${viewportName} Multi-model: DistilBART digest of kept mail`,
        typeof digest === "string" && digest.length > 30,
        digest,
      ) && ok;
    }
    ok = await hygiene(page, route, viewportName) && ok;
  } catch (error) {
    ok = false;
    check(`${viewportName} ${route}: route completed`, false, error.stack || error.message);
  } finally {
    results.push({ route, viewport: viewportName, pass: ok });
    await closePage(cdp, page.targetId);
  }
}

try {
  for (const [name] of Object.entries(ROUTES)) {
    await exercise(name, "desktop", DESKTOP, name === "overview");
  }
  for (const [name] of Object.entries(ROUTES)) await exercise(name, "mobile", MOBILE, false);
} finally {
  server.close();
  chrome.kill();
}
console.log(`\n${passed}/${total} checks passed`);
console.log(`ACCEPTANCE_RESULTS=${JSON.stringify(results)}`);
const allRoutesPassed = results.length === 10 && results.every((result) => result.pass);
process.exit(passed === total && allRoutesPassed ? 0 : 1);
