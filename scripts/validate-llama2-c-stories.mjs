#!/usr/bin/env node
// Route-complete llama2-c-stories acceptance: real browser inference on every published route at desktop
// and mobile. Advertised stage driven for real:
//   Xenova/llama2.c-stories15M  (all routes — text-generation instruction following, WASM int8, 140MB)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline and the ?auto hook downloads + runs the
// model on ready. Every route is driven through real controls: a sample chip + the Answer button.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
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
const RUN_RECORD = join(repoRoot, "models/llama2-c-stories/acceptance-run.json");
mkdirSync(join(homedir(), ".cache", "webai-validator-profiles"), { recursive: true });
const PROFILE_DIR = mkdtempSync(join(homedir(), ".cache", "webai-validator-profiles", "llama2-c-stories-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "Xenova/llama2.c-stories15M"; // the one advertised model stage
const ROUTES = {
  overview: "models/llama2-c-stories/",
  basics: "models/llama2-c-stories/basics/",
  practical: "models/llama2-c-stories/practical/",
  wild: "models/llama2-c-stories/wild/",
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
  while (Date.now() - started < 14 * 60_000) {
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
  throw new Error(`hard timeout after 840000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  // Overview auto-runs on ?auto; ladder pages need the Answer button driven.
  const alreadyRan = await evaluate(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false`,
  );
  if (!alreadyRan) {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
  }
  // Streaming generation: wait for the readout (set when the stream completes). Completion
  // status text varies by route ("Done." / "Rewritten.") so the readout is the reliable signal.
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false`,
    300_000,
    `${viewport} ${rung} default generation`,
  );
  const first = await evaluate(
    cdp,
    sid,
    `({
      out: document.querySelector('#out')?.textContent || '',
      tokens: document.querySelector('#rTok')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      chips: document.querySelectorAll('#chain .t').length
    })`,
  );
  check(
    `${viewport} ${rung}: real ${STAGE} streamed generation`,
    first.out.trim().length >= 20 && Number(first.tokens) >= 5 && /WASM/.test(first.backend),
    JSON.stringify(first).slice(0, 200),
  );

  // Drive real controls: a sample chip (or, on the spec-driven practical rung, the character box) +
  // Answer → a new real stream. Completion status text varies by route ("The end." / "Story draft
  // ready." / "Another version.") so the out-change + status both gate the wait.
  const hasChips = await evaluate(
    cdp,
    sid,
    `document.querySelectorAll('#samples .chip').length > 0`,
  );
  const prompt0 = await evaluate(cdp, sid, `document.querySelector('#prompt')?.value || ''`);
  if (hasChips) {
    await evaluate(
      cdp,
      sid,
      `(() => { const chip = document.querySelectorAll('#samples .chip')[1]; if (chip) chip.click(); return !!chip; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#prompt')?.value !== ${JSON.stringify(prompt0)}`,
      15_000,
      `${viewport} ${rung} chip applied`,
      500,
    );
  } else {
    // practical rung: character + problem <select> boxes drive the draft
    await evaluate(
      cdp,
      sid,
      `(() => { const c = document.querySelector('#character'); if (c && c.options.length > 1) { c.selectedIndex = 1; c.dispatchEvent(new Event('change', { bubbles: true })); } return !!c; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#character')?.selectedIndex === 1`,
      15_000,
      `${viewport} ${rung} character applied`,
      500,
    );
  }
  await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
  await waitFor(
    cdp,
    sid,
    `(document.querySelector('#out')?.textContent || '') !== ${JSON.stringify(first.out)} &&
     /(The end\.|Story draft ready\.|Another version\.)/.test(document.querySelector('#status')?.textContent || '')`,
    300_000,
    `${viewport} ${rung} second generation`,
  );
  const second = await evaluate(
    cdp,
    sid,
    `({ out: document.querySelector('#out')?.textContent || '', tokens: document.querySelector('#rTok')?.textContent || '' })`,
  );
  check(
    `${viewport} ${rung}: sample chip + Answer drive a real second stream`,
    second.out.trim().length >= 20 && second.out !== first.out && Number(second.tokens) >= 5,
    JSON.stringify(second).slice(0, 200),
  );

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
        cell.pass = passed - before === 4;
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

}

const succeeded = checks === 32 && passed === checks && results.length === 8 &&
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
