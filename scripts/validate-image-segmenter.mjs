#!/usr/bin/env node
// Route-complete image-segmenter acceptance: real MediaPipe ImageSegmenter (deeplab_v3.tflite,
// 21 VOC classes) inference on every published route at desktop and mobile. The harness owns one
// fresh Chrome process tree per route cell while reusing its own cache profile (the 2.8MB model is
// downloaded once per run, then auto-init from cache is exercised on later cells); every wait has
// a hard deadline.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const RUN_RECORD = join(repoRoot, "models/image-segmenter/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "image-segmenter-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

// Advertised stage (acceptance.json) — the catalogue id for Google's MediaPipe image segmenter.
const MODEL_ID = "mediapipe/image-segmenter";

const ROUTES = {
  overview: "models/image-segmenter/",
  basics: "models/image-segmenter/basics/",
  practical: "models/image-segmenter/practical/",
  wild: "models/image-segmenter/wild/",
};
const EXPECTED = { overview: 3, basics: 2, practical: 2, wild: 2 };
const TOTAL = Object.values(EXPECTED).reduce((a, b) => a + b, 0) * 2;
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
        nextLog = Date.now() + 10_000;
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

async function ensureReady(cdp, sessionId, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 6 * 60_000) {
    const snapshot = JSON.parse(await evaluate(cdp, sessionId, loaderSnapshot));
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(snapshot)}`,
      );
      nextLog = Date.now() + 8_000;
    }
    if (snapshot.length === 1 && snapshot.every((item) => item.state === "ready")) return snapshot;
    await evaluate(
      cdp,
      sessionId,
      `(() => {
      const buttons = [...document.querySelectorAll('.model-loader')].flatMap((loader) =>
        [...loader.querySelectorAll('button')].filter((item) =>
          /Download|Retry|Re-download/i.test(item.textContent) && !item.disabled
        )
      );
      setTimeout(() => buttons.forEach((button) => button.click()), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 360000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false`,
    180_000,
    `${viewport} ${rung} inference`,
  );

  if (rung === "overview") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      chips: document.querySelectorAll('#legend .count-chip').length,
      classes: document.querySelector('#rClasses')?.textContent || '',
      delegate: document.querySelector('#rDelegate')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || '',
      rows: document.querySelectorAll('#insideRows tr').length,
      px: document.querySelector('#iPx')?.textContent || '',
      status: document.querySelector('#status')?.textContent || ''
    })`,
    );
    check(
      `${viewport} overview: real ${MODEL_ID} segmentation + legend + readout`,
      evidence.chips >= 2 && Number(evidence.classes) >= 2 && /ms/.test(evidence.ms) &&
        /CPU|GPU/.test(evidence.delegate),
      JSON.stringify(evidence),
    );
    check(
      `${viewport} overview: see-inside per-class table`,
      evidence.rows >= 2 && /[0-9]/.test(evidence.px),
      JSON.stringify(evidence),
    );
    // View-mode segmented control + opacity slider are real controls.
    const controls = await evaluate(
      cdp,
      sid,
      `(() => {
      document.querySelector('#modeMask').click();
      const maskPressed = document.querySelector('#modeMask').getAttribute('aria-pressed');
      document.querySelector('#modeOverlay').click();
      const slider = document.querySelector('#opacity');
      slider.value = 80;
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      return { maskPressed, overlayBack: document.querySelector('#modeOverlay').getAttribute('aria-pressed'), val: document.querySelector('#opacityVal')?.textContent || '' };
    })()`,
    );
    check(
      `${viewport} overview: view-mode toggle + opacity slider work`,
      controls.maskPressed === "true" && controls.overlayBack === "true" &&
        controls.val === "80%",
      JSON.stringify(controls),
    );
  } else if (rung === "basics") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      chips: document.querySelectorAll('#legend .count-chip').length,
      top: document.querySelector('#rTop')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || ''
    })`,
    );
    check(
      `${viewport} basics: real segmentation + largest-class readout`,
      evidence.chips >= 2 && /person|dog|bicycle|background/.test(evidence.top) &&
        /ms/.test(evidence.ms),
      JSON.stringify(evidence),
    );
    const legend = await evaluate(
      cdp,
      sid,
      `[...document.querySelectorAll('#legend .count-chip')].map((c) => c.textContent).join(' | ')`,
    );
    check(
      `${viewport} basics: legend lists coverage + confidence`,
      /%/.test(legend) && /conf [0-9]/.test(legend),
      legend.slice(0, 200),
    );
  } else if (rung === "practical") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      cov: document.querySelector('#rCov')?.textContent || '',
      conf: document.querySelector('#rConf')?.textContent || '',
      dl: document.querySelector('#download')?.disabled,
      noPerson: document.querySelector('#noPerson')?.hidden
    })`,
    );
    check(
      `${viewport} practical: person cut out with coverage + confidence`,
      /%/.test(evidence.cov) && /[0-9]/.test(evidence.conf) && evidence.noPerson === true,
      JSON.stringify(evidence),
    );
    check(
      `${viewport} practical: PNG download enabled after cutout`,
      evidence.dl === false,
      JSON.stringify(evidence),
    );
  } else if (rung === "wild") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      chips: document.querySelectorAll('#legend .count-chip').length,
      ms: document.querySelector('#rMs')?.textContent || ''
    })`,
    );
    check(
      `${viewport} wild: scene segmented into clickable class chips`,
      evidence.chips >= 2 && /ms/.test(evidence.ms),
      JSON.stringify(evidence),
    );
    const spot = await evaluate(
      cdp,
      sid,
      `(() => {
      const chips = [...document.querySelectorAll('#legend .count-chip')];
      const person = chips.find((c) => /person/.test(c.textContent)) || chips[1] || chips[0];
      person.click();
      // The legend re-renders on selection, so re-query for the pressed chip.
      const pressed = document.querySelector('#legend .count-chip[aria-pressed="true"]');
      return {
        pressed: pressed ? /person/.test(pressed.textContent) || pressed.textContent : false,
        spot: document.querySelector('#rSpot')?.textContent || '',
        state: document.querySelector('#spotState')?.textContent || ''
      };
    })()`,
    );
    check(
      `${viewport} wild: class spotlight toggles`,
      !!spot.pressed && spot.spot !== "–" && /Spotlighting/.test(spot.state),
      JSON.stringify(spot),
    );
  }
}

try {
  const started = await startServer();
  server = started.server;
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}`;

  for (const viewport of ["desktop", "mobile"]) {
    for (const [rung, route] of Object.entries(ROUTES)) {
      const cell = { route, viewport, pass: false };
      results.push(cell);
      let cdp;
      let page;
      try {
        console.log(`\n=== ${viewport} × ${rung}: ${route} ===`);
        chrome = await launchChrome({
          userDataDir: PROFILE_DIR,
          resetProfile: false,
          removeProfileOnKill: false,
        });
        cdp = new CDP(chrome.ws);
        page = await openPage(cdp, url(route));
        const before = passed;
        await exercise(cdp, page, rung, viewport);
        cell.pass = passed - before === EXPECTED[rung];
      } catch (error) {
        console.log(`FAIL  ${viewport} ${rung}: ${String(error.stack || error).slice(0, 500)}`);
      } finally {
        if (page) await closePage(cdp, page.targetId);
        if (chrome) await chrome.kill({ removeProfile: false });
        chrome = null;
      }
    }
  }
} finally {
  console.log(`\n${passed}/${checks} checks passed`);
  console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
  if (chrome) await chrome.kill({ removeProfile: false });
  if (server) await new Promise((resolve) => server.close(resolve));
  rmSync(PROFILE_DIR, { recursive: true, force: true });
}

const succeeded = checks === TOTAL && passed === checks && results.length === 8 &&
  results.every((item) => item.pass);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify({ commit, ranAt: new Date().toISOString(), exitCode: 0, results }, null, 2) +
      "\n",
  );
  console.log(`WROTE ${RUN_RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
