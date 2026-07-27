#!/usr/bin/env node
// Route-complete camembert-ner acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   Xenova/camembert-ner  (all routes — French WikiNER token-classification, q8 WASM)
//   Xenova/camembert-base (multi-model — fill-mask recasting of detected entities)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile;
// every wait has a hard deadline and long downloads emit incremental state logs.
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
const RUN_RECORD = join(repoRoot, "models/camembert-ner/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "camembert-ner-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const NER_ID = "Xenova/camembert-ner";
const FM_ID = "Xenova/camembert-base";
const ROUTES = {
  overview: "models/camembert-ner/",
  basics: "models/camembert-ner/basics/",
  practical: "models/camembert-ner/practical/",
  wild: "models/camembert-ner/wild/",
  multimodel: "models/camembert-ner/multi-model/",
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
      const buttons = [...document.querySelectorAll('.model-loader')].flatMap((loader) =>
        [...loader.querySelectorAll('button')].filter((item) =>
          /Download|Retry|Re-download/i.test(item.textContent) && !item.disabled
        )
      );
      // Let CDP receive this evaluation result before model download/initialisation can saturate
      // the renderer. The next ensureReady poll observes the real state transition.
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
  await ensureReady(cdp, sid, rung === "multimodel" ? 2 : 1, `${viewport} ${rung}`);

  if (rung === "overview") {
    // Default sentence auto-runs on ready; wait for the readout then verify real entities + inside.
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
      out: document.querySelector('#out')?.textContent || '',
      entities: document.querySelector('#entities')?.textContent || '',
      entCards: document.querySelectorAll('#entities .ent-card').length,
      tokens: document.querySelectorAll('#tokens .tok-chip').length,
      entCount: document.querySelector('#rEnts')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || ''
    })`,
    );
    check(
      `${viewport} overview: real ${NER_ID} entities + see-inside`,
      evidence.out.includes("Marie Curie") && evidence.entCards >= 2 &&
        /Person|Organisation|Location/.test(evidence.entities) && evidence.tokens > 0 &&
        Number(evidence.entCount) >= 2 && /WASM/.test(evidence.backend),
      JSON.stringify(evidence),
    );
  } else if (rung === "basics") {
    // Drive a chip (a real control) — "two Washingtons" exercises PER vs LOC disambiguation.
    await evaluate(
      cdp,
      sid,
      `(() => { [...document.querySelectorAll('.ner-chip')][0].click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false && document.querySelectorAll('#entities .ent-card').length >= 1`,
      120_000,
      `${viewport} basics inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#out')?.textContent || '',
      entities: document.querySelector('#entities')?.textContent || '',
      chips: document.querySelectorAll('#out .ner-word').length
    })`,
    );
    check(
      `${viewport} basics: real entity highlight from ${NER_ID}`,
      evidence.out.includes("Washington") && /Person|Location/.test(evidence.entities) &&
        evidence.chips >= 1,
      JSON.stringify(evidence),
    );
  } else if (rung === "practical") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#results')?.hidden === false && document.querySelectorAll('#index .ent-card').length >= 3`,
      120_000,
      `${viewport} practical inference`,
    );
    const indexEvidence = await evaluate(
      cdp,
      sid,
      `({
      index: document.querySelector('#index')?.textContent || '',
      json: document.querySelector('#json')?.textContent || '',
      unique: document.querySelector('#rUnique')?.textContent || ''
    })`,
    );
    check(
      `${viewport} practical: entity index + JSON export`,
      /Renault/.test(indexEvidence.index) && /"type": "ORG"/.test(indexEvidence.json) &&
        Number(indexEvidence.unique) >= 3,
      JSON.stringify(indexEvidence).slice(0, 220),
    );
    // Second workflow on the same route: privacy redaction (PER+ORG checked by default).
    await evaluate(cdp, sid, `(() => { document.querySelector('#redact').click(); return true; })()`);
    const redactEvidence = await evaluate(
      cdp,
      sid,
      `({ text: document.querySelector('#redactOut')?.textContent || '', hidden: document.querySelector('#redactOut')?.hidden })`,
    );
    check(
      `${viewport} practical: privacy redaction masks names`,
      redactEvidence.hidden === false && /\[PERSONNE\]/.test(redactEvidence.text) &&
        /\[ORGANISATION\]/.test(redactEvidence.text) && !/Luca de Meo/.test(redactEvidence.text),
      JSON.stringify(redactEvidence).slice(0, 220),
    );
  } else if (rung === "wild") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#fillArea')?.hidden === false && document.querySelectorAll('#cast .remix-input').length >= 1`,
      120_000,
      `${viewport} wild inference`,
    );
    const castEvidence = await evaluate(
      cdp,
      sid,
      `({ cast: document.querySelector('#cast')?.textContent || '', inputs: document.querySelectorAll('#cast .remix-input').length, ents: document.querySelector('#rEnts')?.textContent || '' })`,
    );
    check(
      `${viewport} wild: real cast found by ${NER_ID}`,
      /Louis XIV/.test(castEvidence.cast) && Number(castEvidence.ents) >= 1,
      JSON.stringify(castEvidence),
    );
    // Recast with our own name and rebuild — the rebuild is driven by the real spans.
    await evaluate(
      cdp,
      sid,
      `(() => { const inp = document.querySelector('#cast .remix-input'); inp.value = 'Minuit le chat'; document.querySelector('#rebuild').click(); return true; })()`,
    );
    const rebuilt = await evaluate(
      cdp,
      sid,
      `document.querySelector('#result')?.textContent || ''`,
    );
    check(
      `${viewport} wild: story rebuilt with the recast name`,
      rebuilt.includes("Minuit le chat") && !rebuilt.includes("Louis XIV"),
      rebuilt.slice(0, 160),
    );
  } else {
    // Multi-model: NER (PER) then CamemBERT fill-mask recasts Marie Curie in context.
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      240_000,
      `${viewport} ${NER_ID} → ${FM_ID}`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      tagged: document.querySelector('#tagged')?.textContent || '',
      swaps: document.querySelector('#swaps')?.textContent || '',
      after: document.querySelector('#after')?.textContent || '',
      n: document.querySelector('#rSwaps')?.textContent || ''
    })`,
    );
    check(
      `${viewport} multi-model: ${NER_ID} then ${FM_ID} both ran`,
      evidence.tagged.includes("Marie Curie") && evidence.swaps.includes("Marie Curie") &&
        /After:/.test(evidence.after) && evidence.after.length > 20 && Number(evidence.n) >= 1,
      JSON.stringify(evidence).slice(0, 220),
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
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}`;

  // Each cell gets a fresh process tree so released route workers cannot starve later CDP or WASM
  // work. The validator-owned profile persists only between cells, proving cached auto-init without
  // retaining renderer memory.
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
        cell.pass = passed - before === (rung === "wild" || rung === "practical" ? 4 : 3);
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

const succeeded = checks === 34 && passed === checks && results.length === 10 &&
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
