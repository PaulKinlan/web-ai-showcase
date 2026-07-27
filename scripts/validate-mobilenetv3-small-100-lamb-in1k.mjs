// Route-complete acceptance validator for the distinct MobileNetV3-Small LAMB ImageNet family.
// Drives overview + Basics + Practical + Wild + Multi-model at desktop and mobile. The first
// overview visit performs the explicit real download; every later route proves cached auto-init.
// Multi-model is forced down its uncertain path so BOTH MobileNetV3 and ViT execute for real.
import { writeFileSync } from "node:fs";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const SLUG = "mobilenetv3-small-100-lamb-in1k";
const MODEL_ID = "onnx-community/mobilenetv3_small_100.lamb_in1k";
const ROUTES = {
  overview: "models/mobilenetv3-small-100-lamb-in1k/",
  basics: "models/mobilenetv3-small-100-lamb-in1k/basics/",
  practical: "models/mobilenetv3-small-100-lamb-in1k/practical/",
  wild: "models/mobilenetv3-small-100-lamb-in1k/wild/",
  multimodel: "models/mobilenetv3-small-100-lamb-in1k/multimodel/",
};
const CONFIRM_MODEL_ID = "Xenova/vit-base-patch16-224"; // actual model loaded by the delegated ViT worker
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const url = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;
const results = [];
let checks = 0;
let passes = 0;

async function value(sessionId, expression, timeoutMs = 45000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(e){return "ERR:"+(e?.message||e);}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}

async function waitFor(sessionId, expression, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await value(sessionId, expression, 45000)) return;
    } catch (error) {
      console.log(`  ${label}: renderer busy, continuing (${String(error.message).slice(0, 90)})`);
    }
    await sleep(1200);
  }
  throw new Error(`Timeout waiting for ${label}`);
}

function check(name, condition, detail = "") {
  checks++;
  if (condition) passes++;
  console.log(`${condition ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!condition) throw new Error(`${name}: ${detail}`);
}

const loader = `(() => { const x=document.querySelector('.model-loader'); return x && ({
  state:x.dataset.state||'', buttons:[...x.querySelectorAll('button')].map(b=>b.textContent.trim()),
  text:x.textContent.replace(/\\s+/g,' ').trim().slice(0,240) }); })()`;
const clickDownload = `(() => { const b=[...document.querySelectorAll('.model-loader button')]
  .find(x=>/Download/i.test(x.textContent)); if(!b)return false; b.click(); return true; })()`;
const noOverflow = `document.documentElement.scrollWidth <= window.innerWidth + 1`;
const namedControls =
  `[...document.querySelectorAll('button,input')].every(el => el.type === 'hidden' ||
  el.getAttribute('aria-label') || el.labels?.length || (el.textContent||'').trim() || el.id === 'file')`;

async function firstDownload(page) {
  await waitFor(
    page.sessionId,
    `(${loader})?.buttons?.some(x=>/Download/i.test(x))`,
    30000,
    "Download gate",
  );
  const before = await value(page.sessionId, loader);
  check(
    `first visit exposes explicit Download for ${MODEL_ID}`,
    before.buttons.some((x) => /Download/i.test(x)),
    JSON.stringify(before),
  );
  check("Download action accepted", await value(page.sessionId, clickDownload) === true);
  let sawProgress = false;
  const deadline = Date.now() + 6 * 60 * 1000;
  while (Date.now() < deadline) {
    const state = await value(page.sessionId, loader).catch(() => null);
    if (state) {
      if (/MB|%|files/i.test(state.text)) sawProgress = true;
      console.log(`  loader ${state.state}: ${state.text.slice(0, 130)}`);
    }
    if (
      await value(page.sessionId, `!document.querySelector('#run')?.disabled`).catch(() => false)
    ) break;
    await sleep(2000);
  }
  check("real shared-loader progress surfaced", sawProgress);
  check(
    "download completed and overview control enabled",
    await value(page.sessionId, `!document.querySelector('#run').disabled`) === true,
  );
}

async function assertPage(page, route, viewport) {
  check(
    `${viewport} ${route} no horizontal overflow`,
    await value(page.sessionId, noOverflow) === true,
  );
  check(
    `${viewport} ${route} controls have accessible names`,
    await value(page.sessionId, namedControls) === true,
  );
  check(
    `${viewport} ${route} zero console errors`,
    page.errors.length === 0,
    page.errors.join(" | "),
  );
  check(
    `${viewport} ${route} zero failed network`,
    page.netFailures.length === 0,
    page.netFailures.join(" | "),
  );
}

async function runRoute(routeName, viewportName, viewport, first = false) {
  const route = ROUTES[routeName];
  const page = await openPage(cdp, url(route));
  await setViewport(cdp, page.sessionId, viewport);
  let pass = false;
  try {
    if (first) await firstDownload(page);
    else {
      await waitFor(
        page.sessionId,
        `!document.querySelector('#run, #start')?.disabled`,
        4 * 60 * 1000,
        `${route} cached auto-init`,
      );
      const state = await value(page.sessionId, loader);
      check(
        `${viewportName} ${routeName} cached auto-init has no Download gate`,
        state && !state.buttons.some((x) => /Download/i.test(x)),
        JSON.stringify(state),
      );
    }

    if (routeName === "overview") {
      await value(page.sessionId, `document.querySelector('#run').click()`);
      await waitFor(
        page.sessionId,
        `document.querySelector('#insideWrap')?.hidden === false`,
        120000,
        "overview inference",
      );
      const out = await value(
        page.sessionId,
        `({n:document.querySelector('#rN').textContent,
        labels:[...document.querySelectorAll('#bars .bar-label')].map(x=>x.textContent),
        latency:document.querySelector('#rMs').textContent, rows:document.querySelectorAll('#insideRows tr').length})`,
      );
      check(
        `${viewportName} overview real [1000] inference + inside surface`,
        out.n === "1000" && out.rows >= 3 && /ms/.test(out.latency),
        JSON.stringify(out),
      );
      check(
        `${viewportName} overview semantic cats result`,
        out.labels.some((x) => /cat|tabby/i.test(x)),
        out.labels.join(", "),
      );
    } else if (routeName === "basics") {
      await value(page.sessionId, `document.querySelector('#run').click()`);
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#bars .bar-row').length >= 3`,
        120000,
        "basics inference",
      );
      check(
        `${viewportName} basics real top-k`,
        await value(page.sessionId, `document.querySelectorAll('#bars .bar-row').length >= 3`) ===
          true,
      );
    } else if (routeName === "practical") {
      await value(page.sessionId, `document.querySelector('#start').click()`);
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#bars .bar-row').length >= 3`,
        150000,
        "camera/fallback frame inference",
      );
      const live = await value(
        page.sessionId,
        `({bars:document.querySelectorAll('#bars .bar-row').length,
        ms:document.querySelector('#rMs').textContent,status:document.querySelector('#camStatus').textContent})`,
      );
      check(
        `${viewportName} practical user-initiated camera/fallback runs real frame`,
        live.bars >= 3 && /ms/.test(live.ms),
        JSON.stringify(live),
      );
      await value(page.sessionId, `document.querySelector('#stop').click()`);
    } else if (routeName === "wild") {
      await value(
        page.sessionId,
        `(() => { const r=document.querySelector('#res'), q=document.querySelector('#q'); r.value=32; r.dispatchEvent(new Event('input',{bubbles:true})); q.value=25; q.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('#run').click(); return true; })()`,
      );
      await waitFor(
        page.sessionId,
        `/Still|Broken/.test(document.querySelector('#deltaNote')?.textContent||'')`,
        180000,
        "wild degradation inference",
      );
      const result = await value(
        page.sessionId,
        `({bars:document.querySelectorAll('#bars .bar-row').length,note:document.querySelector('#deltaNote').textContent})`,
      );
      check(
        `${viewportName} wild clean+degraded real inference`,
        result.bars >= 3 && /Still|Broken/.test(result.note),
        JSON.stringify(result),
      );
    } else {
      await value(
        page.sessionId,
        `(() => { const t=document.querySelector('#th'); t.value=95; t.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('#run').click(); return true; })()`,
      );
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#mnBars .bar-row').length >= 3 && document.querySelectorAll('#vitBars .bar-row').length >= 3 && /Done/.test(document.querySelector('#runStatus')?.textContent||'')`,
        10 * 60 * 1000,
        "both cascade stages",
      );
      const chain = await value(
        page.sessionId,
        `({mn:document.querySelectorAll('#mnBars .bar-row').length,
        vit:document.querySelectorAll('#vitBars .bar-row').length, decision:document.querySelector('#decision').textContent,
        mnMs:document.querySelector('#mnMs').textContent, vitSub:document.querySelector('#vitSub').textContent})`,
      );
      check(
        `${viewportName} multimodel runs MobileNetV3 AND ViT stages`,
        chain.mn >= 3 && chain.vit >= 3 && /ms/.test(chain.mnMs) && /ms/.test(chain.vitSub),
        JSON.stringify(chain),
      );
    }

    await assertPage(page, route, viewportName);
    await screenshot(cdp, page.sessionId, `/tmp/mobilenetv3-${routeName}-${viewportName}.png`);
    pass = true;
  } finally {
    results.push({ route, viewport: viewportName, pass });
    await closePage(cdp, page.targetId);
  }
}

try {
  for (const name of Object.keys(ROUTES)) {
    await runRoute(name, "desktop", DESKTOP, name === "overview");
  }
  for (const name of Object.keys(ROUTES)) await runRoute(name, "mobile", MOBILE, false);
  check(
    "route matrix complete",
    results.length === 10 && results.every((r) => r.pass),
    JSON.stringify(results),
  );
} finally {
  const record = {
    commit: (await import("node:child_process")).execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: new URL("..", import.meta.url),
    }).toString().trim(),
    ranAt: new Date().toISOString(),
    exitCode: passes === checks && results.length === 10 ? 0 : 1,
    checks: { passed: passes, total: checks },
    results,
  };
  writeFileSync(
    new URL(`../models/${SLUG}/acceptance-run.json`, import.meta.url),
    JSON.stringify(record, null, 2) + "\n",
  );
  chrome.kill();
  server.close();
}
console.log(
  `SUMMARY ${passes}/${checks} checks; ${
    results.filter((r) => r.pass).length
  }/10 route-viewports passed`,
);
if (passes !== checks || results.length !== 10 || results.some((r) => !r.pass)) {
  process.exitCode = 1;
}
