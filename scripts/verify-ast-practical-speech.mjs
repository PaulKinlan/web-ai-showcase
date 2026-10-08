#!/usr/bin/env node
// Scoped functional verification for y8l: AST Practical ONLY, desktop and mobile.
// This deliberately does NOT mint the five-route AST portfolio acceptance record.
// Chrome feeds the real bundled JFK speech as fake microphone INPUT; the page runs the real
// Xenova/ast-finetuned-audioset-10-10-0.4593 q8/WASM model in its worker.
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer,
} from "./browser.mjs";

const PROFILE = mkdtempSync(join(tmpdir(), "y8l-practical-"));
const EVIDENCE = process.env.Y8L_EVIDENCE_DIR || join(tmpdir(), "y8l-ast-practical-evidence");
const RECORD = join(EVIDENCE, "result.json");
const ROUTE = "models/ast-audio-classification/practical/";
const WAV = join(repoRoot, "models/ast-audio-classification/sample-speech.wav");
const VIEWPORTS = {desktop:DESKTOP, mobile:MOBILE};
mkdirSync(EVIDENCE, {recursive:true});
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let cdp, chrome, server;
let checks = 0, passed = 0;
const results = [];
function check(label, ok, detail = "") {
  checks++;
  if (ok) passed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${JSON.stringify(detail).slice(0, 350)}` : ""}`);
  return ok;
}
async function evaluate(sid, expression) {
  const { result } = await cdp.send("Runtime.evaluate", {
    expression:`(async()=>{try{return (${expression});}catch(e){return {__error:String(e?.message||e)};}})()`,
    awaitPromise:true,returnByValue:true,
  }, sid, 45_000);
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(sid, expression, deadlineMs, label) {
  const start = Date.now();
  let nextLog = 0;
  while (Date.now() - start < deadlineMs) {
    try { if (await evaluate(sid, expression)) return; } catch (e) {
      if (Date.now() >= nextLog) console.log(`  [${label}] ${e.message.slice(0, 100)}`);
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - start) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(750);
  }
  throw new Error(`hard timeout ${deadlineMs}ms: ${label}`);
}
async function ready(sid, label) {
  const start = Date.now();
  let nextLog = 0;
  while (Date.now() - start < 12 * 60_000) {
    const state = await evaluate(sid, `(() => {
      const l=document.querySelector('.model-loader');
      return {state:l?.dataset.state,localCheckMs:Number(l?.dataset.localCheckMs),
        status:l?.querySelector('.status')?.textContent,
        buttons:[...l.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim())};
    })()`);
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] ${Math.round((Date.now() - start) / 1000)}s ${JSON.stringify(state)}`);
      nextLog = Date.now() + 10_000;
    }
    if (state.state === "ready") return state;
    await evaluate(sid, `(() => {
      const bs=[...document.querySelectorAll('.model-loader button')].filter(b=>!b.disabled &&
        /Download|Retry|Continue/i.test(b.textContent));
      const cont=bs.find(b=>/^Continue/i.test(b.textContent.trim()));
      setTimeout(() => (cont||bs[0])?.click(), 0); return bs.length;
    })()`);
    await sleep(1500);
  }
  throw new Error(`hard timeout 720000ms: ${label} model preparation`);
}
const alertState = `({hidden:document.querySelector('#alert').hidden,
  text:document.querySelector('#alert').textContent,
  count:Number(document.querySelector('#rCount').textContent),
  live:document.querySelector('#live').textContent,
  backend:document.querySelector('#rBackend').textContent,
  ms:document.querySelector('#rMs').textContent,
  selected:[...document.querySelectorAll('#targets button')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.match)})`;
async function runCell(viewport, dimensions) {
  const label = `practical@${viewport}`;
  let page, ok = true;
  const mark = (name, value, detail = "") => { ok = check(`${label} ${name}`, !!value, detail) && ok; };
  try {
    console.log(`\n=== ${label} ${ROUTE} ===`);
    page = await openPage(cdp, `http://127.0.0.1:${server.port}/web-ai-showcase/${ROUTE}`);
    const sid = page.sessionId;
    await setViewport(cdp, sid, dimensions);
    const model = await ready(sid, label);
    mark("real model ready with bounded local check", Number.isFinite(model.localCheckMs) &&
      model.localCheckMs <= 450, model);
    const start = await evaluate(sid, alertState);
    mark("default Speech chip selected, alert initially hidden", start.selected.length === 1 &&
      start.selected[0] === "Speech" && start.hidden, start);
    await evaluate(sid, `(() => { document.querySelector('#listen').click(); return true; })()`);
    await waitFor(sid, `document.querySelector('#listen').textContent==='Stop listening'`,
      25_000, `${label} microphone started`);
    // Fail closed after four real model windows, not after any timer or DOM-only toggle.
    await waitFor(sid, `!document.querySelector('#alert').hidden ||
      Number(document.querySelector('#rCount').textContent)>=4 ||
      /Classify error|Mic blocked/.test(document.querySelector('#status').textContent+document.querySelector('#listenState').textContent)`,
      220_000, `${label} default Speech alert`);
    const initial = await evaluate(sid, alertState);
    mark("actual default Speech alert from AST inference", initial.hidden === false &&
      /Heard it — Speech/i.test(initial.text) && initial.count > 0 &&
      /Speech/.test(initial.live) && initial.backend === "WASM" && /^\d+ ms$/.test(initial.ms), initial);
    await evaluate(sid, `(() => { document.querySelector('#alert').scrollIntoView({block:'center'});return true; })()`);
    await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-default-speech-alert.png`));

    // Change to Music. The JFK fake microphone remains active and must not spuriously alert.
    await evaluate(sid, `(() => { document.querySelector('#targets button[data-match="Music"]').click();return true; })()`);
    const selectedMusic = await evaluate(sid, alertState);
    mark("Music target selected with native pressed state", selectedMusic.selected.length === 1 &&
      selectedMusic.selected[0] === "Music", selectedMusic);
    await waitFor(sid, `document.querySelector('#alert').hidden`, 10_000, `${label} previous toast expires`);
    const musicStart = await evaluate(sid, alertState);
    await waitFor(sid, `Number(document.querySelector('#rCount').textContent)>${musicStart.count} ||
      /Classify error/.test(document.querySelector('#status').textContent)`,
      120_000, `${label} fresh Music-target audio window`);
    const music = await evaluate(sid, alertState);
    mark("Speech under Music target does not alert", music.hidden &&
      music.count > musicStart.count && music.selected.length === 1 && music.selected[0] === "Music" &&
      /Speech/.test(music.live) && music.backend === "WASM" && /^\d+ ms$/.test(music.ms), music);
    await evaluate(sid, `(() => { document.querySelector('#readout').scrollIntoView({block:'center'});return true; })()`);
    await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-music-no-speech-alert.png`));

    // Re-select Speech: a fresh AST window must fire the actual visible alert again.
    await evaluate(sid, `(() => { document.querySelector('#targets button[data-match="Speech"]').click();return true; })()`);
    const selectedSpeech = await evaluate(sid, alertState);
    mark("Speech target reselected with native pressed state", selectedSpeech.selected.length === 1 &&
      selectedSpeech.selected[0] === "Speech", selectedSpeech);
    await waitFor(sid, `!document.querySelector('#alert').hidden ||
      Number(document.querySelector('#rCount').textContent)>=${selectedSpeech.count + 4} ||
      /Classify error/.test(document.querySelector('#status').textContent)`,
      220_000, `${label} Speech alert after target change`);
    const again = await evaluate(sid, alertState);
    mark("real Speech alert returns after chip change", again.hidden === false &&
      again.count > selectedSpeech.count && /Heard it — Speech/i.test(again.text) &&
      again.selected.length === 1 && again.selected[0] === "Speech", again);
    await evaluate(sid, `(() => { document.querySelector('#alert').scrollIntoView({block:'center'});return true; })()`);
    await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-reselected-speech-alert.png`));
    await evaluate(sid, `(() => { document.querySelector('#listen').click();return true; })()`);
    mark("microphone stopped", await evaluate(sid,
      `document.querySelector('#listen').textContent==='Start listening' &&
       document.querySelector('#listenState').textContent==='Stopped.'`));
    const hygiene = await evaluate(sid, `({overflow:document.documentElement.scrollWidth-innerWidth,
      names:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim())})`);
    mark("responsive and named controls", hygiene.overflow <= 1 && hygiene.names, hygiene);
    mark("console/network clean", page.errors.length === 0 && page.netFailures.length === 0,
      {errors:page.errors,network:page.netFailures});
  } catch (error) {
    ok = false;
    check(`${label} completed`, false, String(error.stack || error).slice(0, 500));
  } finally {
    results.push({route:ROUTE,viewport,pass:ok});
    if (page) await closePage(cdp, page.targetId).catch(() => {});
  }
}
try {
  const started = await startServer(); server = started.server; server.port = started.port;
  chrome = await launchChrome({userDataDir:PROFILE,resetProfile:true,removeProfileOnKill:false,
    extraArgs:["--use-fake-ui-for-media-stream","--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${WAV}`]});
  cdp = new CDP(chrome.ws);
  for (const [viewport, dimensions] of Object.entries(VIEWPORTS)) await runCell(viewport, dimensions);
} catch (error) {
  console.error(`FATAL ${String(error.stack || error)}`);
} finally {
  if (chrome) await chrome.kill({removeProfile:false});
  if (server) await new Promise((resolve) => server.close(resolve));
  rmSync(PROFILE, {recursive:true,force:true});
}
const success = results.length === 2 && results.every((r) => r.pass) && checks === 20 && passed === 20;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/2 practical route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
writeFileSync(RECORD, JSON.stringify({ok:success,checks,passed,results,
  source:"bundled JFK microphone input; actual AST outputs only, no fabricated inference",
  screenshots:6,fromCommit:process.env.FLEET_VERIFY_SHA||null},null,2)+"\n");
console.log(`EVIDENCE ${RECORD}`);
process.exit(success ? 0 : 1);
