#!/usr/bin/env node
// Route-complete stanford-deidentifier-base acceptance: real browser inference on every published
// route at desktop and mobile. Advertised stage driven for real:
//   onnx-community/stanford-deidentifier-base-ONNX  (all routes — PHI token-classification, q8 WASM)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline and the ?auto hook downloads + runs the
// model on ready. Runs the demo's real controls where a route exposes them (copy, mode toggle).
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
const RUN_RECORD = join(repoRoot, "models/stanford-deidentifier-base/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "stanford-deidentifier-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "onnx-community/stanford-deidentifier-base-ONNX"; // the one advertised model stage
const ROUTES = {
  overview: "models/stanford-deidentifier-base/",
  basics: "models/stanford-deidentifier-base/basics/",
  practical: "models/stanford-deidentifier-base/practical/",
  wild: "models/stanford-deidentifier-base/wild/",
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

async function ensureReady(cdp, sessionId, label) {
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
    if (snapshot.length === 1 && snapshot.every((item) => item.state === "ready")) {
      return snapshot;
    }
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
  throw new Error(`hard timeout after 720000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  // Every route auto-runs the demo on ready (?auto hook) — wait for the readout + output.
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false`,
    240_000,
    `${viewport} ${rung} inference`,
  );

  if (rung === "overview") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#out')?.textContent || '',
      spans: document.querySelector('#rSpans')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      chips: document.querySelectorAll('#chain .t').length,
      counts: document.querySelectorAll('#counts .c').length,
      catCounts: [...document.querySelectorAll('#counts .c')].map((c) => c.textContent.trim())
    })`,
    );
    check(
      `${viewport} overview: real ${STAGE} PHI spans + see-inside`,
      evidence.out.includes("Maria Gonzalez") && Number(evidence.spans) >= 5 &&
        /WASM/.test(evidence.backend) && evidence.chips >= 4 && evidence.counts >= 4,
      JSON.stringify(evidence).slice(0, 240),
    );
  } else if (rung === "basics") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#out')?.textContent || '',
      spans: document.querySelector('#rSpans')?.textContent || '',
      counts: document.querySelectorAll('#counts .c').length
    })`,
    );
    check(
      `${viewport} basics: real PATIENT span highlighted`,
      evidence.out.includes("Maria Gonzalez") && Number(evidence.spans) >= 1 &&
        evidence.counts >= 1,
      JSON.stringify(evidence).slice(0, 200),
    );
  } else if (rung === "practical") {
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#out')?.textContent || '',
      copyDisabled: document.querySelector('#copy')?.disabled
    })`,
    );
    check(
      `${viewport} practical: redacted copy ready (PHI gone)`,
      /\[REDACTED\]/.test(evidence.out) && !evidence.out.includes("Maria Gonzalez") &&
        !evidence.out.includes("567-493-1234") && evidence.copyDisabled === false,
      JSON.stringify(evidence).slice(0, 200),
    );
    // Drive the [CATEGORY] mode toggle (a real control).
    await evaluate(cdp, sid, `(() => { document.querySelector('#mCategory').click(); return true; })()`);
    const catEvidence = await evaluate(
      cdp,
      sid,
      `({ out: document.querySelector('#out')?.textContent || '' })`,
    );
    check(
      `${viewport} practical: category-tagged redaction`,
      /\[DATE\]/.test(catEvidence.out) && /\[PATIENT\]/.test(catEvidence.out) &&
        !catEvidence.out.includes("Maria Gonzalez"),
      catEvidence.out.slice(0, 200),
    );
  } else {
    // wild — surrogate replacement
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#out')?.textContent || '',
      copyDisabled: document.querySelector('#copy')?.disabled
    })`,
    );
    check(
      `${viewport} wild: real PHI replaced by synthetic surrogates`,
      !evidence.out.includes("Maria Gonzalez") && !evidence.out.includes("567-493-1234") &&
        /(Jamie Rivera|Alex Chen|Morgan Lee|Sam Patel|Casey Nguyen|Jordan Kim|Taylor Brooks|Riley Morgan|St\. Mary|Riverside|Northgate|Mercy Regional|Lakeview|Cedar Hill|02\/14\/2023|June 9, 2023|11\/03\/2022|August 22, 2023|05\/30\/2022|December 4, 2023|MRN 441089|Acme Diagnostics|MedSystems|Aurora Health|Vertex Medical|Nimbus Records|\(555\))/.test(evidence.out) &&
        evidence.copyDisabled === false,
      JSON.stringify(evidence).slice(0, 200),
    );
    // Drive "New surrogates" (a real control) — output must change.
    await evaluate(cdp, sid, `(() => { document.querySelector('#reshuffle').click(); return true; })()`);
    const after = await evaluate(
      cdp,
      sid,
      `document.querySelector('#out')?.textContent || ''`,
    );
    check(
      `${viewport} wild: reshuffle produces a different surrogate set`,
      after.length > 0 && (after !== evidence.out.slice(0, after.length)),
      `changed=${after !== evidence.out.slice(0, after.length)}`,
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
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}?auto`;

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
        cell.pass = passed - before === (rung === "practical" || rung === "wild" ? 4 : 3);
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

const succeeded = checks === 28 && passed === checks && results.length === 8 &&
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
