#!/usr/bin/env node
// Route-complete grammar-correction acceptance: real browser inference on every published route
// at desktop and mobile. Advertised stages driven for real:
//   Xenova/grammar-synthesis-small (all routes)
//   onnx-community/moonshine-base-ONNX (multi-model, public-domain JFK sample)
// The harness owns one Chrome child and only terminates that child; every wait has a hard deadline
// and long downloads emit incremental state logs.
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

const GEC_ID = "Xenova/grammar-synthesis-small";
const ASR_ID = "onnx-community/moonshine-base-ONNX";
const ROUTES = {
  overview: "models/grammar-correction/",
  basics: "models/grammar-correction/basics/",
  practical: "models/grammar-correction/practical/",
  wild: "models/grammar-correction/wild/",
  multimodel: "models/grammar-correction/multi-model/",
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
let checks = 0;
let passed = 0;
let server;
let chrome;

function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 240)}` : ""}`,
  );
  return condition;
}

async function evaluate(cdp, sessionId, expression, timeoutMs = 45_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.message || error)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}

async function waitFor(cdp, sessionId, expression, deadlineMs, label, intervalMs = 2_000) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < deadlineMs) {
    try {
      if (await evaluate(cdp, sessionId, expression)) return;
    } catch (error) {
      if (Date.now() >= nextLog) {
        console.log(`  [${label}] poll stalled: ${String(error.message).slice(0, 120)}`);
      }
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

const loaderSnapshot =
  `JSON.stringify([...document.querySelectorAll('.model-loader')].map((loader) => ({
  state: loader.dataset.state || '',
  status: (loader.querySelector('.status')?.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 180),
  buttons: [...loader.querySelectorAll('button')].map((button) => button.textContent.trim())
})))`;

async function ensureReady(cdp, sessionId, expectedCount, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 12 * 60_000) {
    const snapshot = JSON.parse(await evaluate(cdp, sessionId, loaderSnapshot));
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(snapshot)}`,
      );
      nextLog = Date.now() + 8_000;
    }
    if (snapshot.length === expectedCount && snapshot.every((item) => item.state === "ready")) {
      return snapshot;
    }
    await evaluate(
      cdp,
      sessionId,
      `(() => {
      for (const loader of document.querySelectorAll('.model-loader')) {
        const button = [...loader.querySelectorAll('button')].find((item) => /Download|Retry|Re-download/i.test(item.textContent));
        if (button && !button.disabled) button.click();
      }
      return true;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, rung === "multimodel" ? 2 : 1, `${viewport} ${rung}`);

  if (rung === "overview") {
    await evaluate(
      cdp,
      sid,
      `(() => { const el=document.querySelector('#prompt'); el.value='she dont has no time for to finish the report'; document.querySelector('#run').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      120_000,
      `${viewport} overview inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      input: document.querySelector('#fedInput')?.textContent || '',
      output: document.querySelector('#out')?.textContent || '',
      tokens: document.querySelectorAll('#tokens .tok').length,
      timeline: document.querySelector('#timeline')?.width || 0
    })`,
    );
    check(
      `${viewport} overview: real ${GEC_ID} output + see-inside`,
      evidence.output.length > 8 && evidence.output !== evidence.input && evidence.tokens > 0 &&
        evidence.timeline > 0,
      JSON.stringify(evidence),
    );
  } else if (rung === "basics") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      120_000,
      `${viewport} basics inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ output:document.querySelector('#out')?.textContent || '', edits:document.querySelector('#editCount')?.textContent || '' })`,
    );
    check(
      `${viewport} basics: real correction and diff`,
      evidence.output.length > 8 && /edit/.test(evidence.edits),
      JSON.stringify(evidence),
    );
  } else if (rung === "practical") {
    await evaluate(
      cdp,
      sid,
      `(() => { const el=document.querySelector('#draft'); el.value='she dont has time to finish\\nwe has sent the reports yesterday'; document.querySelector('#run').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelectorAll('#lines .line-card').length === 2 && !document.querySelector('#copy').disabled`,
      180_000,
      `${viewport} practical inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ lines:document.querySelectorAll('#lines .line-card').length, edits:document.querySelector('#rEdits')?.textContent || '', backend:document.querySelector('#rBackend')?.textContent || '' })`,
    );
    check(
      `${viewport} practical: both lines corrected by ${GEC_ID}`,
      evidence.lines === 2 && Number(evidence.edits) > 0 && /WASM/.test(evidence.backend),
      JSON.stringify(evidence),
    );
  } else if (rung === "wild") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      150_000,
      `${viewport} wild inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ output:document.querySelector('#out')?.textContent || '', tokens:document.querySelectorAll('#tokens .tok').length, ms:document.querySelector('#rMs')?.textContent || '' })`,
    );
    check(
      `${viewport} wild: mangled text really reconstructed`,
      evidence.output.length > 12 && evidence.tokens > 0 && /s$/.test(evidence.ms),
      JSON.stringify(evidence),
    );
  } else {
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#sampleJfk').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `/ready/.test(document.querySelector('#clipState')?.textContent || '')`,
      45_000,
      `${viewport} public-domain JFK decode`,
    );
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      240_000,
      `${viewport} ${ASR_ID} → ${GEC_ID}`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ raw:document.querySelector('#raw')?.textContent || '', clean:document.querySelector('#clean')?.textContent || '', asr:document.querySelector('#rAsrMs')?.textContent || '', gec:document.querySelector('#rGecMs')?.textContent || '' })`,
    );
    check(
      `${viewport} multi-model: ${ASR_ID} then ${GEC_ID} both ran`,
      evidence.raw.length > 8 && evidence.clean.length > 8 && /s$/.test(evidence.asr) &&
        /s$/.test(evidence.gec),
      JSON.stringify(evidence),
    );
  }

  const hygiene = await evaluate(
    cdp,
    sid,
    `({ overflow:document.documentElement.scrollWidth-window.innerWidth, named:[...document.querySelectorAll('button')].every((b)=>(b.textContent||b.getAttribute('aria-label')||'').trim()) })`,
  );
  check(
    `${viewport} ${rung}: responsive controls`,
    hygiene.overflow <= 1 && hygiene.named,
    JSON.stringify(hygiene),
  );
  check(
    `${viewport} ${rung}: console/network clean`,
    page.errors.length === 0 && page.netFailures.length === 0,
    JSON.stringify({ errors: page.errors, network: page.netFailures }),
  );
}

try {
  const started = await startServer();
  server = started.server;
  chrome = await launchChrome();
  const cdp = new CDP(chrome.ws);
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}`;

  for (const viewport of ["desktop", "mobile"]) {
    for (const [rung, route] of Object.entries(ROUTES)) {
      const cell = { route, viewport, pass: false };
      results.push(cell);
      let page;
      try {
        console.log(`\n=== ${viewport} × ${rung}: ${route} ===`);
        page = await openPage(cdp, url(route));
        const before = passed;
        await exercise(cdp, page, rung, viewport);
        cell.pass = passed - before === 3;
      } catch (error) {
        console.log(`FAIL  ${viewport} ${rung}: ${String(error.stack || error).slice(0, 500)}`);
      } finally {
        if (page) await closePage(cdp, page.targetId);
      }
    }
  }
} finally {
  console.log(`\n${passed}/${checks} checks passed`);
  console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
  if (chrome) chrome.kill();
  if (server) await new Promise((resolve) => server.close(resolve));
}

process.exit(
  checks === 30 && passed === checks && results.length === 10 && results.every((item) => item.pass)
    ? 0
    : 1,
);
