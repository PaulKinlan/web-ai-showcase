#!/usr/bin/env node
// Route-complete AST acceptance: five published routes on desktop and mobile.
// Real stages: Xenova/ast-finetuned-audioset-10-10-0.4593 everywhere, plus
// onnx-community/whisper-base_timestamped after AST detects JFK speech in multi-model.
// Only the Speech clip (sample-speech.wav) is JFK; tone/whistle are unrelated. Practical/wild
// have NO bundled JFK sample — this harness injects the JFK audio as fake microphone INPUT solely
// to drive their real live-listen path without relying on room sound or a physical microphone.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { alertWaitTerminal, hasSpeechAlert, isFreshMultiTerminal } from "./ast-acceptance-predicates.mjs";
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
const RECORD = join(repoRoot, "models/ast-audio-classification/acceptance-run.json");
const PROFILE = mkdtempSync(join(tmpdir(), "ast-audio-acceptance-"));
const EVIDENCE = process.env.AST_CREDIT_EVIDENCE_DIR || join(tmpdir(), "7nr-w3-ast-credit");
const JFK_WAV = join(repoRoot, "models/ast-audio-classification/sample-speech.wav");
mkdirSync(EVIDENCE, { recursive: true });
const CREDIT_URL = "https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES = {
  overview: "models/ast-audio-classification/",
  basics: "models/ast-audio-classification/basics/",
  practical: "models/ast-audio-classification/practical/",
  wild: "models/ast-audio-classification/wild/",
  multi: "models/ast-audio-classification/multi-model/",
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
  if (rung === "practical" || rung === "wild") await evaluate(sid,
    `(() => { document.querySelector('#readout')?.scrollIntoView({block:'center'}); return true; })()`);
  await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-${rung}.png`));
  if (rung === "practical" || rung === "wild") {
    mark("no false bundled JFK credit on mic-only route", !credit);
    return;
  }
  const visible = credit?.visible && credit.href.startsWith(CREDIT_URL) &&
    /John F\. Kennedy/.test(credit.context) && /public domain/.test(credit.context) &&
    /Speech clip \(not the tone or whistle\)/.test(credit.context);
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
      'John F. Kennedy','Public domain','models/ast-audio-classification/sample-speech.wav']
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
      await waitFor(sid, `!document.querySelector('#run')?.disabled`, 30_000, `${label} JFK sample decoded`);
      await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#readout')?.hidden &&
        document.querySelector('#scores')?.children.length > 0`, 120_000, `${label} real AST inference`);
      const result = await evaluate(sid, `({backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,top:document.querySelector('#rTop')?.textContent,
        scores:document.querySelector('#scores')?.children.length,
        inside:document.querySelector('#insideRows')?.children.length||0,
        status:document.querySelector('#status')?.textContent})`);
      mark("real AST sound labels", result.backend === "WASM" && /^\d+ ms$/.test(result.ms) &&
        result.scores > 0 && !!result.top && result.top !== "–" &&
        !/failed/i.test(result.status), JSON.stringify(result));
      if (rung === "overview") mark("see-inside real score table", result.inside > 0);
    } else if (rung === "practical" || rung === "wild") {
      if (rung === "practical") {
        const defaults = await evaluate(sid, `({pressed:[...document.querySelectorAll('#targets button')]
          .filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.match),
          threshold:document.querySelector('#thresh')?.value})`);
        mark("default Speech target and sensitivity remain selected", defaults.pressed.length === 1 &&
          defaults.pressed[0] === "Speech" && defaults.threshold === "0.12", JSON.stringify(defaults));
      }
      await evaluate(sid, `(() => { document.querySelector('#listen').click(); return true; })()`);
      await waitFor(sid, `document.querySelector('#listen')?.textContent === 'Stop listening'`,
        25_000, `${label} fake-device mic stream started`);
      await waitFor(sid, `Number(document.querySelector('#rCount')?.textContent) > 0 ||
        /Classify error|Mic blocked/.test((document.querySelector('#status')?.textContent||'') +
          (document.querySelector('#listenState')?.textContent||''))`,
        90_000, `${label} real live AST audio window`);
      const result = await evaluate(sid, `({count:Number(document.querySelector('#rCount')?.textContent),
        backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,
        live:document.querySelector('#live')?.children.length,
        guess:document.querySelector('#guessLabel')?.textContent,
        score:document.querySelector('#guessScore')?.textContent,
        feed:document.querySelector('#feed')?.children.length,
        alert:document.querySelector('#alert')?.textContent,
        alertHidden:document.querySelector('#alert')?.hidden,
        status:document.querySelector('#status')?.textContent})`);
      mark("real live mic AST classification", result.count > 0 && result.backend === "WASM" &&
        /^\d+ ms$/.test(result.ms) && result.live > 0 && !/error|blocked/i.test(result.status),
        JSON.stringify(result));
      if (rung === "practical") {
        // First fake-mic window can start on a non-speech segment; keep the real stream running
        // until a matching speech window fires the actual alert, or fail after four windows.
        await waitFor(sid, `(${alertWaitTerminal.toString()})({
          hidden:document.querySelector('#alert')?.hidden,
          count:Number(document.querySelector('#rCount')?.textContent),
          status:document.querySelector('#status')?.textContent||''
        })`, 180_000, `${label} real speech alert from fake-mic stream`);
        const alert = await evaluate(sid, `({hidden:document.querySelector('#alert')?.hidden,
          text:document.querySelector('#alert')?.textContent,
          count:Number(document.querySelector('#rCount')?.textContent),
          labels:document.querySelector('#live')?.textContent})`);
        mark("speech monitor alerts visibly", hasSpeechAlert(alert), JSON.stringify(alert));
        if (!alert.hidden) {
          await evaluate(sid, `(() => { document.querySelector('#alert').scrollIntoView({block:'center'}); return true; })()`);
          await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-practical-alert.png`));
        }
      }
      if (rung === "wild") mark("real live best-guess feed", result.guess && result.guess !== "—" &&
        /% confident/.test(result.score) && result.feed > 0, JSON.stringify(result));
      await evaluate(sid, `(() => { document.querySelector('#listen').click(); return true; })()`);
      mark("mic stopped after inference", await evaluate(sid,
        `document.querySelector('#listen')?.textContent === 'Start listening' &&
          document.querySelector('#listenState')?.textContent === 'Stopped.'`));
    } else {
      await waitFor(sid, `!document.querySelector('#run')?.disabled`, 30_000, `${label} both stages ready`);
      // First prove non-speech audio does NOT start Whisper.
      await evaluate(sid, `(() => { document.querySelector('#samples button[data-src="../sample-tone.wav"]').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#run')?.disabled &&
        document.querySelector('#status')?.textContent === '' &&
        document.querySelector('#samples button[data-src="../sample-tone.wav"]')?.getAttribute('aria-pressed') === 'true'`,
        30_000, `${label} tone decoded`);
      await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#run')?.disabled &&
        (document.querySelector('#rRoute')?.textContent === 'tagged, no ASR' ||
        document.querySelector('#rRoute')?.textContent === '→ Whisper' ||
        /Pipeline failed/.test(document.querySelector('#route')?.textContent||''))`,
        120_000, `${label} real non-speech gate`);
      const tone = await evaluate(sid, `({route:document.querySelector('#rRoute')?.textContent,
        whisper:document.querySelector('#rWhisper')?.textContent,
        visible:!document.querySelector('#whisperOut')?.hidden,
        ast:document.querySelector('#rAst')?.textContent})`);
      mark("real tone classification skips Whisper", tone.route === "tagged, no ASR" &&
        tone.whisper === "skipped" && !tone.visible && !!tone.ast && tone.ast !== "–", JSON.stringify(tone));
      // Then drive the actual JFK clip through AST's speech gate AND Whisper's real decoder.
      await evaluate(sid, `(() => { document.querySelector('#samples button[data-src="../sample-speech.wav"]').click(); return true; })()`);
      await waitFor(sid, `!document.querySelector('#run')?.disabled &&
        document.querySelector('#status')?.textContent === '' &&
        document.querySelector('#route')?.textContent === 'Ready — hit Route.'`,
        30_000, `${label} JFK sample decoded`);
      await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
      await waitFor(sid, `(${isFreshMultiTerminal.toString()})({
        runDisabled:document.querySelector('#run')?.disabled,
        status:document.querySelector('#route')?.textContent||'',
        readoutVisible:!document.querySelector('#readout')?.hidden,
        scores:document.querySelector('#astScores')?.children.length||0,
        route:document.querySelector('#rRoute')?.textContent,
        whisperVisible:!document.querySelector('#whisperOut')?.hidden
      })`, 180_000, `${label} real AST speech → Whisper`);
      const result = await evaluate(sid, `({route:document.querySelector('#rRoute')?.textContent,
        ast:document.querySelector('#rAst')?.textContent,
        scores:document.querySelector('#astScores')?.children.length,
        whisper:document.querySelector('#rWhisper')?.textContent,
        text:document.querySelector('#transcript')?.textContent,
        visible:!document.querySelector('#whisperOut')?.hidden,
        status:document.querySelector('#route')?.textContent})`);
      mark("real AST JFK speech gate triggers Whisper transcription",
        result.route === "→ Whisper" && !!result.ast && result.ast !== "–" &&
        result.scores > 0 && result.visible && /country/i.test(result.text) &&
        /^\d+ ms$/.test(result.whisper), JSON.stringify(result));
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
  chrome = await launchChrome({userDataDir:PROFILE, resetProfile:true, removeProfileOnKill:false,
    extraArgs:["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${JFK_WAV}`]});
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
const succeeded = results.length === 10 && results.every((r) => r.pass) && checks === 64 && checks === passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["log", "-n1", "--format=%H", "HEAD", "--",
    "models/ast-audio-classification", "scripts/validate-ast-audio-classification.mjs",
    ":(exclude)models/ast-audio-classification/acceptance.json",
    ":(exclude)models/ast-audio-classification/acceptance-run.json"],
  {cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD, JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells: real AST classification on every rung, fake-mic live listening on practical/wild; Practical keeps the selected default Speech target and default 0.12 sensitivity, with an actual visible Speech alert on both desktop and mobile. Real AST speech gate and Whisper transcription on multi-model; non-speech tone skips Whisper. Visible linked JFK Speech clip credit on three JFK routes with clicked GitHub ledger hash, creator, licence and path; no false bundled credit on mic-only routes; responsive/console/network checks; screenshots outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
