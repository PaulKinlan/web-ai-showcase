#!/usr/bin/env node
// Route-complete mms-forced-alignment acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   onnx-community/mms-300m-1130-forced-aligner-ONNX  (all routes — MMS 300M CTC word alignment, WASM q4, 241 MB)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline. Every route is driven through real
// controls: the bundled JFK sample auto-loads, then the Align button runs real CTC alignment. The
// practical rung renders an SRT export instead of word chips, so the checks branch on the element
// actually present.
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
const RUN_RECORD = join(repoRoot, "models/mms-forced-alignment/acceptance-run.json");
const PROFILE_DIR = join(homedir(), ".cache", "webai-validator-profiles", "mms-forced-alignment");
mkdirSync(PROFILE_DIR, { recursive: true });
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "onnx-community/mms-300m-1130-forced-aligner-ONNX"; // the one advertised model stage
// The wild rung has no bundled audio — it drives a real mic recording. In the harness we point
// Chrome's fake audio device at the bundled JFK clip and align it against its known transcript, so
// the record → transcript → align flow runs end to end for real.
const JFK_WAV = join(repoRoot, "models/mms-forced-alignment/jfk.wav");
const JFK_TRANSCRIPT =
  "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.";
const ROUTES = {
  overview: "models/mms-forced-alignment/",
  basics: "models/mms-forced-alignment/basics/",
  practical: "models/mms-forced-alignment/practical/",
  wild: "models/mms-forced-alignment/wild/",
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
          /Download|Retry|Re-download|Continue/i.test(item.textContent) && !item.disabled
        )
      );
      // In check-timeout the Retry button re-enters the same stalled check while Continue is the
      // honest forward path (it verifies the cache / downloads); prefer Continue when present.
      const cont = buttons.find((b) => /^Continue/i.test(b.textContent.trim()));
      setTimeout(() => (cont || buttons[0])?.click(), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 840000ms: ${label} model download/init`);
}

// Wait for the alignment completion status AND a populated result surface, then snapshot it.
async function runAlign(cdp, sessionId, label) {
  await evaluate(cdp, sessionId, `(() => { document.querySelector('#run').click(); return true; })()`);
  // Reach a TERMINAL state: success with rendered output, or the page's explicit failure message
  // (fail fast with the real error instead of burning the whole timeout).
  await waitFor(
    cdp,
    sessionId,
    `(() => {
      const s = document.querySelector('#status')?.textContent || '';
      if (/Alignment failed|Record a sentence first/.test(s)) return true;
      return /Aligned\.|SRT built\./.test(s) &&
        (document.querySelectorAll('#words > *').length > 0 ||
         (document.querySelector('#srt')?.textContent || '').includes('-->'));
    })()`,
    300_000,
    `${label} alignment done`,
  );
  const snap = await evaluate(
    cdp,
    sessionId,
    `({
      status: document.querySelector('#status')?.textContent || '',
      matched: document.querySelector('#rMatched')?.textContent || document.querySelector('#rWords')?.textContent || '',
      frames: document.querySelector('#rFrames')?.textContent || '',
      frameMs: document.querySelector('#rFrameMs')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || '',
      words: document.querySelectorAll('#words > *').length,
      srt: (document.querySelector('#srt')?.textContent || '').slice(0, 120),
      transcript: (document.querySelector('#transcript')?.textContent || '').trim().slice(0, 160)
    })`,
  );
  if (/Alignment failed|Record a sentence first/.test(snap.status)) {
    throw new Error(`page alignment stopped: ${snap.status}`);
  }
  return snap;
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  if (rung === "wild") {
    // Real mic flow against Chrome's fake audio device (the bundled JFK clip).
    await evaluate(cdp, sid, `(() => { document.querySelector('#rec').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `(document.querySelector('#recState')?.textContent || '').includes('recording')`,
      15_000,
      `${viewport} ${rung} recording started`,
      500,
    );
    await sleep(13_500); // let the 11 s fake clip fully record
    await evaluate(cdp, sid, `(() => { document.querySelector('#rec').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `(document.querySelector('#recState')?.textContent || '').includes('recorded')`,
      30_000,
      `${viewport} ${rung} recording stopped`,
      1_000,
    );
    // recState flips BEFORE blobToMono16k finishes; the page assigns pcm then reveals the player,
    // so the player reveal is the decode-complete signal — clicking run before it means pcm is
    // still null and the page answers "Record a sentence first."
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#player')?.hidden === false && (document.querySelector('#player')?.src || '').length > 0`,
      30_000,
      `${viewport} ${rung} recording decoded`,
      1_000,
    );
    await evaluate(
      cdp,
      sid,
      `(() => { const t = document.querySelector('#transcript'); t.value = ${JSON.stringify(JFK_TRANSCRIPT)}; return t.value.length; })()`,
    );
  }

  // Real run #1: the bundled JFK sample (11 s) is the default audio; the Align button drives CTC.
  const first = await runAlign(cdp, sid, `${viewport} ${rung} run 1`);
  const m1 = (first.matched || "").match(/^(\d+)(\/|$)/); // "54/83 chars" (overview) or "22" (practical)
  check(
    `${viewport} ${rung}: real ${STAGE} forced alignment`,
    first.backend === "WASM" && !!m1 && Number(m1[1]) > 0 &&
      /^\d+ ms$/.test(first.ms) && first.transcript.length > 0,
    JSON.stringify(first).slice(0, 240),
  );
  check(
    `${viewport} ${rung}: word-level output rendered`,
    (rung === "practical" ? first.srt.includes("-->") : first.words > 0),
    JSON.stringify({ words: first.words, srt: first.srt }),
  );

  // Drive real controls: Align again → a second real alignment pass, output persists.
  const second = await runAlign(cdp, sid, `${viewport} ${rung} run 2`);
  const m2 = (second.matched || "").match(/^(\d+)(\/|$)/);
  check(
    `${viewport} ${rung}: second Align re-runs real alignment`,
    !!m2 && Number(m2[1]) > 0 && /^\d+ ms$/.test(second.ms) &&
      (rung === "practical" ? second.srt.includes("-->") : second.words > 0),
    JSON.stringify(second).slice(0, 240),
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
          // The wild rung records from the mic; drive Chrome's fake device with the bundled clip.
          extraArgs: rung === "wild"
            ? [
              "--use-fake-ui-for-media-stream",
              "--use-fake-device-for-media-stream",
              `--use-file-for-fake-audio-capture=${JFK_WAV}`,
            ]
            : [],
        });
        cdp = new CDP(chrome.ws);
        page = await openPage(cdp, url(route));
        const before = passed;
        await exercise(cdp, page, rung, viewport);
        cell.pass = passed - before === 5;
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

const succeeded = checks === 40 && passed === checks && results.length === 8 &&
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
