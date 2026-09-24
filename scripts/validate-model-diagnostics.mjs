#!/usr/bin/env node
// Bounded diagnostics checks: fault-injected component/loader tests, then REAL llama2.c inference
// on all four routes at desktop/mobile. --fixtures-only is download-free. No fabricated model output.
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { tokenMetrics } from "../lib/model-run-status.mjs";
import {
  BASE,
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

const fixturesOnly = process.argv.includes("--fixtures-only");
const out = process.env.DIAGNOSTICS_OUTPUT || mkdtempSync(join(tmpdir(), "webai-diagnostics-"));
mkdirSync(out, { recursive: true });
const report = {
  startedAt: new Date().toISOString(),
  fixturesOnly,
  checks: [],
  routes: [],
  screenshots: [],
  blocked: [],
};
const check = (name, ok, detail = null) => {
  report.checks.push({ name, pass: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? " " + JSON.stringify(detail) : ""}`);
  assert.ok(ok, name);
};
async function evaluate(cdp, sid, expression) {
  const r = await cdp.send(
    "Runtime.evaluate",
    { expression, awaitPromise: true, returnByValue: true },
    sid,
    20000,
  );
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  }
  return r.result.value;
}
async function wait(cdp, sid, expression, ms = 10000) {
  const deadline = Date.now() + ms;
  let nextLog = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (await evaluate(cdp, sid, expression)) return;
    if (Date.now() >= nextLog) {
      console.log("  waiting", expression.slice(0, 90));
      nextLog += 10000;
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Timed out after ${ms}ms: ${expression}`);
}
async function click(cdp, sid, selector) {
  const p = await evaluate(
    cdp,
    sid,
    `(()=>{const el=document.querySelector(${
      JSON.stringify(selector)
    });el.scrollIntoView({block:'center'});const r=el.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`,
  );
  for (const type of ["mousePressed", "mouseReleased"]) {
    await cdp.send("Input.dispatchMouseEvent", { type, ...p, button: "left", clickCount: 1 }, sid);
  }
}

let chrome;
const { server, port } = await startServer();
// Local test server only; production header policy is not changed.
server.prependListener("request", (req, res) => {
  if (!req.url.includes("nonisolated")) {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  }
});
try {
  check(
    "finite exact-token average",
    tokenMetrics(32, 2000) === "32 tokens · 16.0 tok/s average (including prefill)",
  );
  check(
    "invalid metrics never claim a rate",
    [[NaN, 1], [-1, 1], [1.5, 1], [1, 0], [1, Infinity]].every(([n, ms]) =>
      tokenMetrics(n, ms) === "Token rate unavailable"
    ),
  );
  check("zero generated tokens is valid", tokenMetrics(0, 1000).startsWith("0 tokens · 0.0"));
  chrome = await launchChrome({
    userDataDir: join(out, "profile"),
    extraArgs: ["--enable-blink-features=ForceEagerMeasureMemory"],
  });
  const cdp = new CDP(chrome.ws);
  report.browser = await cdp.send("Browser.getVersion");
  report.browserFlags = [
    "headless",
    "disable-gpu",
    "ForceEagerMeasureMemory (test-only; shortens the native GC wait)",
  ];
  const fixture = await openPage(cdp, `http://127.0.0.1:${port}${BASE}lib/__dispose-selftest__/`);
  const sid = fixture.sessionId;
  const fixtureResult = await evaluate(
    cdp,
    sid,
    `(async()=>{
    const m=document.querySelector('model-memory-diagnostics');
    const results={isolated:crossOriginIsolated};
    performance.measureUserAgentSpecificMemory=async()=>({bytes:10485760,breakdown:[{types:['JavaScript']} ]});
    await m.capture('Baseline');
    performance.measureUserAgentSpecificMemory=async()=>({bytes:12582912,breakdown:[]});
    await m.capture('After inference');
    results.delta=m.snapshots.at(-1).delta===2097152;
    for(let i=0;i<9;i++) await m.capture('Sample '+i);
    results.boundedHistory=m.snapshots.length===8;
    performance.measureUserAgentSpecificMemory=async()=>{throw new DOMException('refused','SecurityError')};
    results.failure=await m.capture('Failure')===null && m.shadowRoot.querySelector('#status').textContent.includes('refused');
    let finish,calls=0;
    performance.measureUserAgentSpecificMemory=()=>{calls++;return new Promise(r=>finish=r)};
    const count=m.snapshots.length;
    const start=performance.now();
    results.timeout=await m.capture('Before load',{timeoutMs:30})===null && performance.now()-start<1000;
    results.busy=await m.capture('Do not queue')===null && calls===1;
    finish({bytes:999999,breakdown:[]});await new Promise(r=>setTimeout(r,0));
    results.lateDiscarded=m.snapshots.length===count && !m.shadowRoot.querySelector('#measure').disabled;
    performance.measureUserAgentSpecificMemory=async()=>({bytes:NaN,breakdown:[]});
    results.invalid=await m.capture('Invalid')===null && m.snapshots.length===count;
    performance.measureUserAgentSpecificMemory=async()=>({bytes:10485760,breakdown:[]});
    results.retry=!!await m.capture('Retry');
    await import('${BASE}lib/model-run-status.mjs');
    const r=document.createElement('model-run-status');document.body.append(r);
    await r.start();r.token(32,2000);r.tick();
    results.liveRate=r.shadowRoot.querySelector('#metrics').textContent.includes('16.0 tok/s');
    r.complete({tokens:32,ms:2000});r.token(99,1);
    results.final=r.shadowRoot.querySelector('#live').textContent.includes('32 tokens') && !r.shadowRoot.querySelector('#metrics').hasAttribute('aria-hidden');
    await r.start();results.reset=r.shadowRoot.querySelector('#metrics').textContent==='';
    r.fail('Fixture failure');results.error=r.dataset.state==='error' && r._timer===null;
    r.remove();
    // A non-settling native request cannot stop either loading or disposal. Synthetic model only.
    window.nativeCalls=0;performance.measureUserAgentSpecificMemory=()=>{nativeCalls++;return new Promise(()=>{})};
    window.__dispose.click('download');
    return results;
  })()`,
  );
  for (const [name, pass] of Object.entries(fixtureResult)) check(`fixture: ${name}`, pass);
  await wait(cdp, sid, "window.__dispose.state()==='ready'", 5000);
  check(
    "fixture: model ready despite stalled memory API",
    await evaluate(cdp, sid, "window.__dispose.runEnabled()"),
  );
  await evaluate(cdp, sid, "window.__dispose.click('release from memory')");
  await wait(cdp, sid, "window.__dispose.state()==='released'", 5000);
  // Measurement is manual-only (web-ai-showcase-cgr): the loader no longer samples on load or release,
  // because the native API answers only after a GC pass (~19s idle, ~60s with a model resident, measured
  // in Chrome 150 and Chromium 152), which no non-blocking phase budget can cover. The requirement this
  // check exists for — a stalled memory request must never queue behind, or block, disposal — now holds
  // by construction: the release path makes ZERO native calls and still completes. Asserting 0 rather
  // than 1 keeps the check meaningful instead of vacuous; if sampling ever returns to this path, the
  // count moves and this fails again.
  check(
    "fixture: disposal not queued behind stalled memory API",
    await evaluate(cdp, sid, "window.__dispose.counters().disposeCalls===1 && nativeCalls===0"),
  );
  await closePage(cdp, fixture.targetId);

  const plain = await openPage(
    cdp,
    `http://127.0.0.1:${port}${BASE}lib/__dispose-selftest__/?nonisolated`,
  );
  check(
    "non-isolated host explains unavailable memory",
    await evaluate(
      cdp,
      plain.sessionId,
      "!crossOriginIsolated && document.querySelector('model-memory-diagnostics').shadowRoot.querySelector('#measure').disabled && document.querySelector('model-memory-diagnostics').shadowRoot.querySelector('#explanation').textContent.includes('not cross-origin isolated')",
    ),
  );
  await closePage(cdp, plain.targetId);

  if (!fixturesOnly) {
    for (const [viewport, vp] of Object.entries({ desktop: DESKTOP, mobile: MOBILE })) {
      for (const rung of ["overview", "basics", "practical", "wild"]) {
        console.log(`REAL ${viewport} ${rung}`);
        const route = `models/llama2-c-stories/${rung === "overview" ? "" : rung + "/"}`;
        const page = await openPage(cdp, `http://127.0.0.1:${port}${BASE}${route}`);
        try {
          await setViewport(cdp, page.sessionId, vp);
          await cdp.send("Emulation.setEmulatedMedia", {
            features: [{
              name: "prefers-color-scheme",
              value: viewport === "mobile" ? "dark" : "light",
            }],
          }, page.sessionId);
          await cdp.send("Performance.enable", {}, page.sessionId);
          await evaluate(
            cdp,
            page.sessionId,
            `window.longTasks=[];new PerformanceObserver(list=>longTasks.push(...list.getEntries().map(e=>e.duration))).observe({type:'longtask'});window.metricsBefore=performance.now()`,
          );
          const before = await cdp.send("Performance.getMetrics", {}, page.sessionId);
          const initial = await evaluate(
            cdp,
            page.sessionId,
            "document.querySelector('.model-loader').dataset.state",
          );
          if (initial === "download-required") {
            await click(cdp, page.sessionId, ".loader-actions button");
          }
          await wait(
            cdp,
            page.sessionId,
            "document.querySelector('.model-loader').dataset.state==='ready'",
            150000,
          );
          await evaluate(
            cdp,
            page.sessionId,
            `(()=>{
            const n=document.querySelector('#maxNew');n.value='40';n.dispatchEvent(new Event('input',{bubbles:true}));
            document.querySelector('#mGreedy')?.click();
            if(document.querySelector('#character')){document.querySelector('#character').selectedIndex=1;document.querySelector('#problem').selectedIndex=1;}
            if(document.querySelector('#temp')){document.querySelector('#temp').value='0.3';document.querySelector('#temp').dispatchEvent(new Event('input',{bubbles:true}));}
            document.querySelector('model-memory-diagnostics').shadowRoot.querySelector('details').open=true;
          })()`,
          );
          await click(cdp, page.sessionId, "#run");
          await wait(
            cdp,
            page.sessionId,
            "document.querySelector('model-run-status').dataset.state==='complete'",
            90000,
          );
          const result = await evaluate(
            cdp,
            page.sessionId,
            `(()=>{
            const r=document.querySelector('model-run-status');const m=document.querySelector('model-memory-diagnostics');
            return {initial:${
              JSON.stringify(initial)
            },output:document.querySelector('#out').textContent,tokens:Number(document.querySelector('#rTok').textContent),chunks:document.querySelectorAll('#out .new-span').length,rate:document.querySelector('#rTps').textContent,metrics:r.shadowRoot.querySelector('#metrics').textContent,announcement:r.shadowRoot.querySelector('#live').textContent,backend:document.querySelector('#rBackend').textContent,overflow:document.documentElement.scrollWidth-innerWidth,viewportWidth:innerWidth,clipped:[...document.querySelectorAll('button,input,select,textarea')].filter(el=>{const b=el.getBoundingClientRect();return b.width>0&&(b.right>innerWidth+1||b.left< -1)}).length,longTasks,snapshots:m.snapshots};
          })()`,
          );
          check(
            `${viewport}/${rung}: real exact-token generation`,
            result.output.length > 20 && result.tokens > 0 && result.tokens <= 40 &&
              result.backend === "WASM",
            { tokens: result.tokens, chunks: result.chunks, rate: result.rate },
          );
          if (rung === "overview") {
            check(
              `${viewport}/${rung}: counts token IDs, not word chunks`,
              result.tokens === 40 && result.tokens > result.chunks,
            );
          }
          check(
            `${viewport}/${rung}: accessible rate agrees`,
            result.metrics.startsWith(`${result.tokens} tokens`) &&
              result.metrics.includes(`${result.rate} tok/s`) &&
              result.announcement.includes(result.metrics),
          );
          check(
            `${viewport}/${rung}: no clipped controls or expanded mobile viewport`,
            result.overflow <= 1 && result.viewportWidth === vp.width && result.clipped === 0,
            {
              overflow: result.overflow,
              viewportWidth: result.viewportWidth,
              clipped: result.clipped,
            },
          );
          const nativeMemory = await evaluate(
            cdp,
            page.sessionId,
            "document.querySelector('model-memory-diagnostics').capture('Manual snapshot',{timeoutMs:1000})",
          );
          if (nativeMemory) {
            check(
              `${viewport}/${rung}: real native memory estimate`,
              nativeMemory.bytes > 0 && nativeMemory.breakdownEntries > 0,
              nativeMemory,
            );
          } else {
            const reason = await evaluate(
              cdp,
              page.sessionId,
              "document.querySelector('model-memory-diagnostics').shadowRoot.querySelector('#status').textContent",
            );
            report.blocked.push({ route, viewport, check: "native model-memory estimate", reason });
            console.log(`BLOCKED native memory: ${reason}`);
            check(
              `${viewport}/${rung}: missing measurement is visible, not fabricated`,
              /took too long|Measuring|unavailable|late memory/.test(reason),
              reason,
            );
          }
          await evaluate(
            cdp,
            page.sessionId,
            "document.querySelector('model-memory-diagnostics').scrollIntoView({block:'start'})",
          );
          const shot = join(out, `${viewport}-${rung}.png`);
          await screenshot(cdp, page.sessionId, shot);
          report.screenshots.push(shot);
          // Exercise rerun/sample controls, not just a first rendering.
          await evaluate(
            cdp,
            page.sessionId,
            "document.querySelectorAll('#samples button')[1]?.click()",
          );
          await click(cdp, page.sessionId, rung === "wild" ? "#again" : "#run");
          await wait(
            cdp,
            page.sessionId,
            "document.querySelector('model-run-status').dataset.state==='complete'",
            90000,
          );
          check(
            `${viewport}/${rung}: rerun enabled`,
            await evaluate(
              cdp,
              page.sessionId,
              "!document.querySelector('#run').disabled && Number(document.querySelector('#rTok').textContent)>0",
            ),
          );
          check(
            `${viewport}/${rung}: console/network clean`,
            !page.errors.length && !page.netFailures.length,
            { errors: page.errors, network: page.netFailures },
          );
          report.routes.push({
            route,
            viewport,
            ...result,
            nativeMemory,
            devToolsBefore: before.metrics,
            devToolsAfter: (await cdp.send("Performance.getMetrics", {}, page.sessionId)).metrics,
          });
        } finally {
          await closePage(cdp, page.targetId);
        }
      }
    }
  }
} catch (error) {
  report.error = String(error.stack || error);
  console.error(report.error);
  process.exitCode = 1;
} finally {
  if (chrome) await chrome.kill();
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
  report.finishedAt = new Date().toISOString();
  writeFileSync(join(out, "results.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(`Evidence: ${out}/results.json`);
}
