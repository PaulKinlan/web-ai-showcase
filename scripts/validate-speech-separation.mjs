#!/usr/bin/env node
// Route-complete speech-separation acceptance. The sole published route is driven at desktop and
// mobile with audio-capture permission denied. Each cell owns a fresh Chrome process tree while a
// validator-owned profile preserves the downloaded model. Real inference uses a lawful, temporary,
// validator-generated two-source waveform; those bytes are never bundled or presented in the UI.
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
const RUN_RECORD = join(repoRoot, "models/speech-separation/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "speech-separation-acceptance-profile-"));
const FIXTURE_DIR = mkdtempSync(join(tmpdir(), "speech-separation-lawful-test-audio-"));
const FIXTURE = join(FIXTURE_DIR, "validator-generated-two-source.wav");
const MODEL_ID = "welcomyou/convtasnet-libri2mix-16k-onnx";
const ROUTE = "models/speech-separation/";
const SR = 16_000;
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

function writeSyntheticMixture(path) {
  const seconds = 2;
  const samples = SR * seconds;
  const bytes = Buffer.alloc(44 + samples * 2);
  bytes.write("RIFF", 0);
  bytes.writeUInt32LE(36 + samples * 2, 4);
  bytes.write("WAVEfmt ", 8);
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(SR, 24);
  bytes.writeUInt32LE(SR * 2, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36);
  bytes.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) {
    const t = i / SR;
    const fade = Math.min(1, i / 400, (samples - i) / 400);
    const sourceA = (Math.sin(2 * Math.PI * 173 * t) + .35 * Math.sin(2 * Math.PI * 346 * t)) *
      (.55 + .35 * Math.sin(2 * Math.PI * 2.3 * t));
    const sourceB = (Math.sin(2 * Math.PI * 281 * t) + .3 * Math.sin(2 * Math.PI * 843 * t)) *
      (.55 + .35 * Math.sin(2 * Math.PI * 3.7 * t + .8));
    const value = Math.max(-1, Math.min(1, (sourceA + sourceB) * .28 * fade));
    bytes.writeInt16LE(Math.round(value * 32767), 44 + i * 2);
  }
  writeFileSync(path, bytes);
}
writeSyntheticMixture(FIXTURE);

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
    `${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 300)}` : ""}`,
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

async function waitFor(cdp, sessionId, expression, deadlineMs, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < deadlineMs) {
    try {
      if (await evaluate(cdp, sessionId, expression)) return;
    } catch (error) {
      if (Date.now() >= nextLog) {
        console.log(`  [${label}] poll: ${String(error.message).slice(0, 120)}`);
      }
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(1_000);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

async function ensureReady(cdp, sessionId, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 12 * 60_000) {
    const state = await evaluate(
      cdp,
      sessionId,
      `({state:document.querySelector('.model-loader')?.dataset.state||'', status:document.querySelector('.model-loader .status')?.textContent||'', buttons:[...document.querySelectorAll('.model-loader button')].map(b=>b.textContent.trim())})`,
    );
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(state)}`,
      );
      nextLog = Date.now() + 8_000;
    }
    if (state.state === "ready") return;
    await evaluate(
      cdp,
      sessionId,
      `(()=>{const b=[...document.querySelectorAll('.model-loader button')].find(x=>/Download|Retry|Re-download/i.test(x.textContent)&&!x.disabled);if(b)setTimeout(()=>b.click(),0);return !!b;})()`,
    );
    await sleep(2_000);
  }
  throw new Error(`hard timeout after 720000ms: ${label} ${MODEL_ID} download/init`);
}

async function attachFixture(cdp, sessionId) {
  const doc = await cdp.send("DOM.getDocument", {}, sessionId, 10_000);
  const node = await cdp.send(
    "DOM.querySelector",
    { nodeId: doc.root.nodeId, selector: "#file" },
    sessionId,
    10_000,
  );
  await cdp.send(
    "DOM.setFileInputFiles",
    { files: [FIXTURE], nodeId: node.nodeId },
    sessionId,
    10_000,
  );
  await evaluate(
    cdp,
    sessionId,
    `document.getElementById('file').dispatchEvent(new Event('change'))`,
  );
}

async function exercise(cdp, page, viewport) {
  const sid = page.sessionId;
  await setViewport(cdp, sid, viewport === "mobile" ? MOBILE : DESKTOP);
  const initial = await evaluate(
    cdp,
    sid,
    `({
    instructions:/does not request camera or microphone permission/i.test(document.body.innerText),
    noBundledControl:!document.querySelector('#chips,[data-bundled-audio]'),
    status:document.getElementById('status')?.textContent||'',
    runDisabled:document.getElementById('separateBtn')?.disabled===true,
    outputsHidden:document.getElementById('sepOut')?.hidden===true,
    inputName:document.querySelector('label[for="file"]')?.textContent.trim()||'',
    permission:(await navigator.permissions.query({name:'microphone'})).state
  })`,
  );
  check(
    `${viewport}: useful accessible no-permission default`,
    initial.instructions && initial.noBundledControl &&
      /No audio selected|ready.*Choose/i.test(initial.status) && initial.runDisabled &&
      initial.outputsHidden &&
      /overlapping voices/i.test(initial.inputName) && initial.permission === "denied",
    JSON.stringify(initial),
  );

  await ensureReady(cdp, sid, `${viewport} overview`);
  await attachFixture(cdp, sid);
  await waitFor(
    cdp,
    sid,
    `document.getElementById('separateBtn')?.disabled===false`,
    30_000,
    `${viewport} upload decode`,
  );
  await evaluate(cdp, sid, `(()=>{document.getElementById('separateBtn').click();return true;})()`);
  await waitFor(
    cdp,
    sid,
    `!document.getElementById('sepOut')?.hidden`,
    180_000,
    `${viewport} real separation inference`,
  );

  const evidence = await evaluate(
    cdp,
    sid,
    `(async()=>{
    const M=await import('./separation.js');
    const decode=async(id)=>M.decodeTo16kMono(await (await fetch(document.getElementById(id).src)).arrayBuffer());
    const [mix,a,b]=await Promise.all([decode('mixAudio'),decode('s1Audio'),decode('s2Audio')]);
    const rms=x=>Math.sqrt(x.reduce((s,v)=>s+v*v,0)/x.length);
    const diff=Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0)/a.length);
    return {mix:mix.length,a:a.length,b:b.length,rmsA:rms(a),rmsB:rms(b),diff,status:document.getElementById('status').textContent,inside:document.getElementById('mixWave').width>0&&document.getElementById('s1Wave').width>0};
  })()`,
    45_000,
  );
  check(
    `${viewport}: real ${MODEL_ID} two-track inference + inside`,
    evidence.mix === SR * 2 && evidence.a === evidence.mix && evidence.b === evidence.mix &&
      evidence.rmsA > 0.001 && evidence.rmsB > 0.001 && evidence.diff > 0.001 &&
      /Separated into 2 speaker tracks/.test(evidence.status) && evidence.inside,
    JSON.stringify(evidence),
  );

  const hygiene = await evaluate(
    cdp,
    sid,
    `({overflow:document.documentElement.scrollWidth-window.innerWidth,named:[...document.querySelectorAll('button,input')].every(el=>el.type==='file'?!!document.querySelector('label[for="'+el.id+'"]'):(el.textContent||el.getAttribute('aria-label')||'').trim().length>0),live:document.getElementById('status')?.getAttribute('aria-live')})`,
  );
  check(
    `${viewport}: responsive and keyboard/screen-reader semantics`,
    hygiene.overflow <= 1 && hygiene.named && hygiene.live === "polite",
    JSON.stringify(hygiene),
  );
  check(
    `${viewport}: console/network clean`,
    page.errors.length === 0 && page.netFailures.length === 0,
    JSON.stringify({ errors: page.errors, network: page.netFailures }),
  );
}

try {
  const started = await startServer();
  server = started.server;
  const origin = `http://127.0.0.1:${started.port}`;
  for (const viewport of ["desktop", "mobile"]) {
    const cell = { route: ROUTE, viewport, pass: false };
    results.push(cell);
    let cdp;
    let page;
    const before = passed;
    try {
      console.log(`\n=== ${viewport} × overview: ${ROUTE} ===`);
      chrome = await launchChrome({
        userDataDir: PROFILE_DIR,
        resetProfile: false,
        removeProfileOnKill: false,
      });
      cdp = new CDP(chrome.ws);
      await cdp.send(
        "Browser.setPermission",
        { permission: { name: "microphone" }, setting: "denied", origin },
        undefined,
        15_000,
      );
      page = await openPage(cdp, `${origin}/web-ai-showcase/${ROUTE}`);
      await exercise(cdp, page, viewport);
      cell.pass = passed - before === 4;
    } catch (error) {
      console.log(`FAIL  ${viewport} overview: ${String(error.stack || error).slice(0, 800)}`);
    } finally {
      if (page) await closePage(cdp, page.targetId);
      if (chrome) await chrome.kill({ removeProfile: false });
      chrome = null;
      await sleep(1_000);
    }
  }
} finally {
  console.log(`\n${passed}/${checks} checks passed`);
  console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
  if (chrome) await chrome.kill({ removeProfile: false });
  if (server) await new Promise((resolve) => server.close(resolve));
  rmSync(PROFILE_DIR, { recursive: true, force: true });
  rmSync(FIXTURE_DIR, { recursive: true, force: true });
}

const succeeded = checks === 8 && passed === checks && results.length === 2 &&
  results.every((item) => item.pass);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
    .trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify({ commit, ranAt: new Date().toISOString(), exitCode: 0, results }, null, 2) +
      "\n",
  );
  console.log(`WROTE ${RUN_RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
