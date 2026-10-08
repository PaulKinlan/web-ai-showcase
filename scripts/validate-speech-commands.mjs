#!/usr/bin/env node
// Route-complete Speech Commands acceptance: all five published routes on desktop and mobile.
// Real stages: Xenova/ast-finetuned-speech-commands-v2 on every route, and
// onnx-community/whisper-base_timestamped only after the real wake-word gate on multi-model.
// The JFK sample-sentence.wav is used ONLY by multi-model after wake; the other routes' trigger
// clips are separate media and must not inherit the JFK attribution.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const RECORD = join(repoRoot, "models/speech-commands/acceptance-run.json");
const PROFILE = mkdtempSync(join(tmpdir(), "speech-commands-acceptance-"));
const EVIDENCE = process.env.SPEECH_CREDIT_EVIDENCE_DIR || join(tmpdir(), "7nr-w2-speech-credit");
mkdirSync(EVIDENCE, { recursive: true });
const CREDIT_URL = "https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES = {
  overview: "models/speech-commands/",
  basics: "models/speech-commands/basics/",
  practical: "models/speech-commands/practical/",
  wild: "models/speech-commands/wild/",
  multi: "models/speech-commands/multi-model/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
let checks = 0;
let passed = 0;
let server;
let chrome;
let cdp;

function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 260)}` : ""}`);
  return condition;
}

async function evaluate(sid, expression, timeoutMs = 45_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: `(async()=>{try{return (${expression});}catch(e){return {__error:String(e?.message||e)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    timeoutMs,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}

async function waitFor(sid, expression, deadlineMs, label, intervalMs = 1_000) {
  const start = Date.now();
  let nextLog = 0;
  while (Date.now() - start < deadlineMs) {
    try {
      if (await evaluate(sid, expression)) return;
    } catch (e) {
      if (Date.now() >= nextLog) console.log(`  [${label}] evaluation: ${String(e.message).slice(0, 120)}`);
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - start) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

async function ensureReady(sid, label) {
  const start = Date.now();
  let nextLog = 0;
  while (Date.now() - start < 12 * 60_000) {
    const snapshot = await evaluate(sid, `(() => {
      const loaders=[...document.querySelectorAll('.model-loader')];
      return {states:loaders.map(x=>x.dataset.state),
        checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim()),
        buttons:loaders.flatMap(x=>[...x.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim()))};
    })()`);
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] ${Math.round((Date.now() - start) / 1000)}s ${JSON.stringify(snapshot)}`);
      nextLog = Date.now() + 10_000;
    }
    if (snapshot.states.length === (label.startsWith("multi") ? 2 : 1) &&
        snapshot.states.every((state) => state === "ready")) return snapshot;
    await evaluate(sid, `(() => {
      const buttons=[...document.querySelectorAll('.model-loader button')].filter(b=>
        /Download|Retry|Re-download|Continue/i.test(b.textContent) && !b.disabled);
      buttons.forEach(b=>b.click()); return buttons.length;
    })()`);
    await sleep(1_500);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model preparation`);
}

async function verifyCredit(sid, label, rung, viewport, mark) {
  const credit = await evaluate(sid, `(() => {
    const a=[...document.querySelectorAll('a')].find(x=>x.textContent.trim()===
      'JFK audio source and attribution record');
    if (!a) return null;
    a.scrollIntoView({block:'center'});
    return {href:a.href, visible:!!a.getClientRects().length,
      context:a.parentElement.textContent.trim()};
  })()`);
  await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-${rung}.png`));
  if (rung !== "multi") {
    mark("no false JFK attribution on non-JFK trigger routes", !credit);
    return;
  }
  const visible = credit?.visible && credit.href.startsWith(CREDIT_URL) &&
    /John F\. Kennedy/.test(credit.context) && /public domain/.test(credit.context) &&
    /follow-up sentence \(not the wake-word clips\)/.test(credit.context);
  let responseStatus = 0;
  cdp.on((msg) => {
    if (msg.sessionId === sid && msg.method === "Network.responseReceived" &&
        msg.params.type === "Document" && msg.params.response.url.startsWith(CREDIT_URL)) {
      responseStatus = msg.params.response.status;
    }
  });
  if (visible) {
    await evaluate(sid, `(() => {
      const a=[...document.querySelectorAll('a')].find(x=>x.textContent.trim()===
        'JFK audio source and attribution record');
      setTimeout(()=>a.click(),0); return true;
    })()`);
    await waitFor(sid, `location.href.startsWith(${JSON.stringify(CREDIT_URL)})`,
      30_000, `${label} credit navigation`, 500);
    await waitFor(sid,
      `document.body?.innerText.includes('627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b')`,
      30_000, `${label} credit content`, 500);
  }
  const content = visible && await evaluate(sid, `(() => {
    const body=document.body?.innerText||'';
    return ['627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b',
      'John F. Kennedy','Public domain','models/speech-commands/sample-sentence.wav']
      .map(text=>body.includes(text));
  })()`);
  mark("visible JFK credit click reaches content-verified ledger",
    visible && responseStatus >= 200 && responseStatus < 400 && content?.every(Boolean),
    JSON.stringify({credit, responseStatus, content}));
}

async function drive(rung, viewport) {
  const route = ROUTES[rung];
  const label = `${rung}@${viewport}`;
  let page;
  let ok = true;
  const mark = (name, condition, detail = "") => {
    ok = check(`${label} ${name}`, condition, detail) && ok;
  };
  try {
    console.log(`\n=== ${label}: ${route} ===`);
    page = await openPage(cdp, `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
    const sid = page.sessionId;
    await setViewport(cdp, sid, VIEWPORTS[viewport]);
    const loader = await ensureReady(sid, label);
    mark("bounded local cache check", loader.checks.length > 0 &&
      loader.checks.every((n) => Number.isFinite(n) && n <= 450), JSON.stringify(loader.checks));
    if (rung === "overview" || rung === "basics") {
      await waitFor(sid, `!document.querySelector('#run')?.disabled`, 30_000, `${label} command audio ready`);
      await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#readout')?.hidden &&
        document.querySelector('#scores')?.children.length > 0`, 120_000, `${label} real keyword inference`);
      const result = await evaluate(sid, `({backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,top:document.querySelector('#rTop')?.textContent,
        scores:document.querySelector('#scores')?.children.length,
        inside:document.querySelector('#insideRows')?.children.length||0,
        status:document.querySelector('#status')?.textContent})`);
      mark("real keyword output", result.backend === "WASM" && /^\d+ ms$/.test(result.ms) &&
        result.scores > 0 && /yes/i.test(result.top) && !/failed/i.test(result.status), JSON.stringify(result));
      if (rung === "overview") mark("see-inside score table", result.inside > 0);
    } else if (rung === "practical") {
      await evaluate(sid, `(() => { document.querySelector('#samples button').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#readout')?.hidden`, 120_000, `${label} real command action`);
      const result = await evaluate(sid, `({heard:document.querySelector('#heard')?.textContent,
        dotTop:document.querySelector('#dot')?.style.top,
        backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real command moved the pad", /up/i.test(result.heard) &&
        result.dotTop !== "43%" && result.backend === "WASM" && /^\d+ ms$/.test(result.ms) &&
        !/failed/i.test(result.status), JSON.stringify(result));
    } else if (rung === "wild") {
      await evaluate(sid, `(() => { document.querySelector('#samples button').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#readout')?.hidden`, 120_000, `${label} real wake-meter inference`);
      const result = await evaluate(sid, `({word:document.querySelector('#word')?.value,
        meter:document.querySelector('#meter')?.getAttribute('aria-valuenow'),
        backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real keyword score updates wake meter", result.word === "yes" &&
        Number(result.meter) > 0 && result.backend === "WASM" && /^\d+ ms$/.test(result.ms) &&
        !/failed/i.test(result.status), JSON.stringify(result));
    } else {
      await waitFor(sid, `!document.querySelector('#run')?.disabled`, 30_000, `${label} both stages ready`);
      await evaluate(sid, `(() => { document.querySelector('#word').value='go';
        document.querySelector('#samples button[data-src="../sample-go.wav"]').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#run')?.disabled`, 30_000, `${label} wake sample decoded`);
      await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#whisperOut')?.hidden ||
        document.querySelector('#rRoute')?.textContent === 'asleep' ||
        /Pipeline failed/.test(document.querySelector('#route')?.textContent||'')`,
        180_000, `${label} real wake → Whisper`);
      const result = await evaluate(sid, `({route:document.querySelector('#rRoute')?.textContent,
        kws:document.querySelector('#rKws')?.textContent,
        whisper:document.querySelector('#rWhisper')?.textContent,
        text:document.querySelector('#transcript')?.textContent,
        visible:!document.querySelector('#whisperOut')?.hidden,
        status:document.querySelector('#route')?.textContent})`);
      mark("real top-1 wake gate triggers Whisper on bundled JFK sentence",
        result.route === "→ wake" && /go/i.test(result.kws) &&
        result.visible && /country/i.test(result.text) && /^\d+ ms$/.test(result.whisper),
        JSON.stringify(result));
    }
    const hygiene = await evaluate(sid, `({overflow:document.documentElement.scrollWidth-innerWidth,
      named:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim())})`);
    mark("responsive controls", hygiene.overflow <= 1 && hygiene.named, JSON.stringify(hygiene));
    mark("console/network clean", page.errors.length === 0 && page.netFailures.length === 0,
      JSON.stringify({errors:page.errors,network:page.netFailures}));
    await verifyCredit(sid, label, rung, viewport, mark);
  } catch (e) {
    ok = false;
    check(`${label} completed`, false, String(e.stack || e).slice(0, 400));
  } finally {
    results.push({route, viewport, pass: ok});
    if (page) await closePage(cdp, page.targetId).catch(() => {});
  }
}

try {
  const started = await startServer();
  server = started.server;
  server.port = started.port;
  chrome = await launchChrome({userDataDir:PROFILE, resetProfile:true, removeProfileOnKill:false});
  cdp = new CDP(chrome.ws);
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) await drive(rung, viewport);
  }
} catch (e) {
  console.error(`FATAL ${String(e.stack || e)}`);
} finally {
  if (chrome) await chrome.kill({removeProfile:false});
  if (server) await new Promise((resolve) => server.close(resolve));
  rmSync(PROFILE, {recursive:true,force:true});
}
const succeeded = results.length === 10 && results.every((r) => r.pass) && checks === 52 && checks === passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["log", "-n1", "--format=%H", "HEAD", "--",
    "models/speech-commands", "scripts/validate-speech-commands.mjs"],
  {cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD, JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells: real keyword model on every rung; actual wake gate and Whisper transcription on multi-model. Visible linked JFK credit with clicked GitHub ledger hash, creator, licence and path; responsive/console/network checks; screenshots captured outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
