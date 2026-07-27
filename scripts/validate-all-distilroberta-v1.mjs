// Route-complete real-browser acceptance for sentence-transformers/all-distilroberta-v1.
// Drives every overview/Basics/Practical/Wild/Multi-model route at desktop + mobile.
// Every advertised stage runs for real: sentence-transformers/all-distilroberta-v1 and
// Xenova/ms-marco-MiniLM-L-6-v2.
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

const MODEL_ID = "sentence-transformers/all-distilroberta-v1";
const ROUTES = {
  overview: "models/all-distilroberta-v1/",
  basics: "models/all-distilroberta-v1/basics/",
  practical: "models/all-distilroberta-v1/practical/",
  wild: "models/all-distilroberta-v1/wild/",
  multimodel: "models/all-distilroberta-v1/multi-model/",
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
  await waitFor(page.sessionId, `!document.querySelector('#run')?.disabled`, "target model ready");
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
      ok = check(
        `${viewportName} overview: Embed & compare click`,
        await evaluate(page.sessionId, click("#run")),
      ) && ok;
      await waitFor(
        page.sessionId,
        `document.querySelector('#rDim')?.textContent==='768'`,
        "overview 768-d inference",
      );
      const inside = JSON.parse(
        await evaluate(
          page.sessionId,
          `JSON.stringify({dim:document.querySelector('#rDim')?.textContent,cells:document.querySelectorAll('#vecStrip .vec-cell').length,norm:document.querySelector('#iNorm')?.textContent})`,
        ),
      );
      ok = check(
        `${viewportName} overview: real 768-d output + inside vector`,
        inside.dim === "768" && inside.cells === 96 && Number(inside.norm) > 0,
        JSON.stringify(inside),
      ) && ok;
      await evaluate(page.sessionId, click("#search"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#ranked .result-row').length>=2`,
        "overview semantic search",
      );
      ok = check(
        `${viewportName} overview: real semantic-search ranking`,
        await evaluate(
          page.sessionId,
          `document.querySelectorAll('#ranked .result-row').length>=2`,
        ),
      ) && ok;
    } else if (routeName === "basics") {
      await readyTarget(page, false);
      await evaluate(page.sessionId, click("#run"));
      await waitFor(
        page.sessionId,
        `Number.isFinite(Number(document.querySelector('#score')?.textContent))`,
        "Basics similarity",
      );
      ok = check(
        `${viewportName} Basics: real cosine`,
        await evaluate(
          page.sessionId,
          `Number(document.querySelector('#score')?.textContent)>0.4`,
        ),
        await evaluate(page.sessionId, `document.querySelector('#score')?.textContent`),
      ) && ok;
    } else if (routeName === "practical") {
      await sleep(1200);
      await evaluate(page.sessionId, clickDownloads);
      await waitFor(
        page.sessionId,
        `!document.querySelector('#searchBtn')?.disabled`,
        "Practical ready",
      );
      await evaluate(page.sessionId, click("#searchBtn"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#searchOut .result-row').length>=10`,
        "Practical search",
      );
      await evaluate(page.sessionId, click("#clusterBtn"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#clusterOut .cluster').length>=2`,
        "Practical cluster",
      );
      await evaluate(page.sessionId, click("#classifyBtn"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#classifyOut .result-row').length>=10`,
        "Practical classify",
      );
      ok = check(
        `${viewportName} Practical: search + clustering + nearest-label classification inferred`,
        await evaluate(
          page.sessionId,
          `document.querySelectorAll('#searchOut .result-row').length>=10&&document.querySelectorAll('#clusterOut .cluster').length>=2&&document.querySelectorAll('#classifyOut .result-row').length>=10`,
        ),
      ) && ok;
    } else if (routeName === "wild") {
      await readyTarget(page, false);
      await evaluate(page.sessionId, click("#run"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#out .result-row').length>=5`,
        "Wild analogy inference",
      );
      ok = check(
        `${viewportName} Wild: real vector arithmetic ranking`,
        await evaluate(page.sessionId, `document.querySelectorAll('#out .result-row').length>=5`),
        await evaluate(page.sessionId, `document.querySelector('#status')?.textContent`),
      ) && ok;
    } else {
      await sleep(1200);
      await evaluate(page.sessionId, clickDownloads);
      await waitFor(
        page.sessionId,
        `!document.querySelector('#run')?.disabled`,
        "Multi-model stages ready",
      );
      await evaluate(page.sessionId, click("#run"));
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#stage1 .result-row').length>=2&&document.querySelectorAll('#stage2 .result-row').length>=2`,
        "retrieve then rerank",
        10 * 60_000,
      );
      ok = check(
        `${viewportName} Multi-model: embedding retrieval + cross-encoder rerank both inferred`,
        await evaluate(
          page.sessionId,
          `document.querySelectorAll('#stage1 .result-row').length>=2&&document.querySelectorAll('#stage2 .result-row').length>=2`,
        ),
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
