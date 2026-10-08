#!/usr/bin/env node
// Five published GTCRN routes × desktop/mobile: real WASM denoising, a five-level
// SNR sweep, practical user-upload path and GTCRN → Whisper transcription.
// JFK is bundled only on overview/basics/wild/multi; the practical file is supplied
// through the real upload control solely to exercise that route (not a bundled credit).
// The unrelated TED sample is neither selected nor modified.
// Advertised stages: bitsydarel/gtcrn-onnx (WASM raw ORT), Xenova/whisper-tiny.en
// (WASM Transformers.js on the multi-model route).
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer } from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const FAMILY = "models/gtcrn-speech-enhancement";
const RECORD = join(repoRoot, FAMILY, "acceptance-run.json");
const PROFILE = mkdtempSync(join(tmpdir(), "gtcrn-acceptance-"));
const EVIDENCE = process.env.GTCRN_CREDIT_EVIDENCE_DIR || join(tmpdir(), "7nr-w4-gtcrn-credit");
const JFK_WAV = join(repoRoot, FAMILY, "jfk.wav");
const CREDIT_URL = "https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES = {
  overview: "models/gtcrn-speech-enhancement/",
  basics: "models/gtcrn-speech-enhancement/basics/",
  practical: "models/gtcrn-speech-enhancement/practical/",
  wild: "models/gtcrn-speech-enhancement/wild/",
  multi: "models/gtcrn-speech-enhancement/multi-model/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
mkdirSync(EVIDENCE, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
let checks = 0, passed = 0, server, chrome, cdp;
function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 260)}` : ""}`);
  return condition;
}
async function evaluate(sid, expression, timeoutMs = 45_000) {
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: `(async()=>{try{return (${expression});}catch(e){return {__error:String(e?.message||e)};}})()`,
    awaitPromise: true, returnByValue: true,
  }, sid, timeoutMs);
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(sid, expression, deadlineMs, label, intervalMs = 1_000) {
  const start = Date.now();
  let nextLog = 0;
  while (Date.now() - start < deadlineMs) {
    try { if (await evaluate(sid, expression)) return; }
    catch (e) { if (Date.now() >= nextLog) console.log(`  [${label}] evaluation: ${String(e.message).slice(0, 120)}`); }
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
      return {states:loaders.map(x=>x.dataset.state),checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim()),
        buttons:loaders.flatMap(x=>[...x.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim()))};
    })()`);
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] ${Math.round((Date.now() - start) / 1000)}s ${JSON.stringify(snapshot)}`);
      nextLog = Date.now() + 10_000;
    }
    if (snapshot.states.length === 1 && snapshot.states[0] === "ready") return snapshot;
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
    return {href:a.href,visible:!!a.getClientRects().length,context:a.parentElement.textContent.trim()};
  })()`);
  if (rung === "practical") await evaluate(sid,
    `(() => { document.querySelector('#readout')?.scrollIntoView({block:'center'}); return true; })()`);
  await screenshot(cdp, sid, join(EVIDENCE, `${viewport}-${rung}.png`));
  if (rung === "practical") { mark("no false bundled JFK credit on upload-only route", !credit); return; }
  const visible = credit?.visible && credit.href.startsWith(CREDIT_URL) &&
    /John F\. Kennedy/.test(credit.context) && /public domain/.test(credit.context) &&
    /bundled JFK speech clip/.test(credit.context);
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
      'John F. Kennedy','Public domain','models/gtcrn-speech-enhancement/jfk.wav']
      .map(text=>body.includes(text));
  })()`);
  mark("visible JFK credit click reaches content-verified ledger",
    visible && responseStatus >= 200 && responseStatus < 400 && content?.every(Boolean),
    JSON.stringify({credit,responseStatus,content}));
}
async function attachPracticalFile(sid) {
  const doc = await cdp.send("DOM.getDocument", {}, sid, 10_000);
  const node = await cdp.send("DOM.querySelector",
    {nodeId:doc.root.nodeId,selector:"#file"}, sid, 10_000);
  if (!node.nodeId) throw new Error("practical upload control missing");
  await cdp.send("DOM.setFileInputFiles", {files:[JFK_WAV],nodeId:node.nodeId}, sid, 10_000);
  await evaluate(sid, `(() => { document.querySelector('#file').dispatchEvent(new Event('change')); return true; })()`);
}
async function drive(rung, viewport) {
  const route = ROUTES[rung], label = `${rung}@${viewport}`;
  let page, ok = true;
  const mark = (name, condition, detail = "") => { ok = check(`${label} ${name}`, condition, detail) && ok; };
  try {
    console.log(`\n=== ${label}: ${route} ===`);
    page = await openPage(cdp, `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
    const sid = page.sessionId;
    await setViewport(cdp, sid, VIEWPORTS[viewport]);
    const loader = await ensureReady(sid, label);
    mark("bounded local cache check", loader.checks.length === 1 &&
      Number.isFinite(loader.checks[0]) && loader.checks[0] <= 450, JSON.stringify(loader.checks));
    if (rung === "overview" || rung === "basics" || rung === "wild") {
      const sample = await evaluate(sid, `({selected:document.querySelector('#samples button.active')?.textContent,
        candidates:[...document.querySelectorAll('#samples button')].map(b=>b.textContent.trim())})`);
      mark("default JFK clip selected without using unrelated sample",
        /JFK speech/.test(sample.selected) && sample.candidates.length >= 2, JSON.stringify(sample));
    }
    if (rung === "practical") {
      await attachPracticalFile(sid);
      await waitFor(sid, `!document.querySelector('#run')?.disabled &&
        document.querySelector('#srcNote')?.textContent === 'jfk.wav'`,
        30_000, `${label} real user file decoded`);
      mark("real WAV loaded through native upload", await evaluate(sid,
        `document.querySelector('#file')?.files[0]?.name === 'jfk.wav' &&
          document.querySelector('#srcNote')?.textContent === 'jfk.wav'`));
    } else if (rung === "multi") {
      await waitFor(sid, `!document.querySelector('#run')?.disabled &&
        /fellow Americans/.test(document.querySelector('#refTxt')?.textContent||'')`,
        35_000, `${label} JFK reference decoded`);
      mark("JFK speech decoded as actual multi-model input", await evaluate(sid,
        `document.querySelector('#refTxt')?.textContent?.includes('fellow Americans') &&
          document.querySelector('#status')?.textContent === 'Both models ready.'`));
      // The signal remains genuinely noisy; 5 dB keeps tiny Whisper intelligible.
      await evaluate(sid, `(() => { const s=document.querySelector('#snr'); s.value='5';
        s.dispatchEvent(new Event('input',{bubbles:true})); return true; })()`);
    } else await waitFor(sid, `!document.querySelector('#run')?.disabled`,
      35_000, `${label} bundled JFK sample decoded`);
    await evaluate(sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    if (rung === "multi") {
      await waitFor(sid, `document.querySelector('#done')?.textContent === '1' &&
        !document.querySelector('#run')?.disabled`, 180_000, `${label} denoise → Whisper pair`);
      const result = await evaluate(sid, `({backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,visible:!document.querySelector('#abGrid')?.hidden,
        noisy:document.querySelector('#noisyTxt')?.textContent,
        enhanced:document.querySelector('#enhTxt')?.textContent,
        werNoisy:document.querySelector('#werNoisy')?.textContent,
        werEnhanced:document.querySelector('#werEnh')?.textContent,
        status:document.querySelector('#status')?.textContent,
        audio:[...document.querySelectorAll('#abGrid audio')].map(a=>a.src.startsWith('blob:'))})`);
      mark("real GTCRN→Whisper chain returns two playable transcriptions",
        result.backend === "WASM" && /^\d+ ms$/.test(result.ms) && result.visible &&
        result.audio.length === 2 && result.audio.every(Boolean) &&
        /^WER \d+%$/.test(result.werNoisy) && /^WER \d+%$/.test(result.werEnhanced) &&
        !/failed/i.test(result.status), JSON.stringify(result));
      mark("enhanced JFK produces intelligible real Whisper text",
        /fellow|american|country|ask/i.test(result.enhanced) &&
        result.enhanced.trim().length > 10, JSON.stringify(result.enhanced));
    } else if (rung === "wild") {
      await waitFor(sid, `!document.querySelector('#readout')?.hidden &&
        document.querySelector('#rows')?.children.length === 5`,
        180_000, `${label} five-level real GTCRN sweep`);
      const sweep = await evaluate(sid, `({backend:document.querySelector('#rBackend')?.textContent,
        levels:document.querySelector('#rLevels')?.textContent,
        gains:[...document.querySelectorAll('#rows tr')].map(r=>Number.parseFloat(r.cells[2]?.textContent)),
        audios:[...document.querySelectorAll('#rows audio')].map(a=>a.src.startsWith('blob:')),
        status:document.querySelector('#status')?.textContent})`);
      mark("five real enhancement levels and playable outputs",
        sweep.backend === "WASM" && sweep.levels === "5" && sweep.gains.length === 5 &&
        sweep.gains.every(Number.isFinite) && sweep.audios.length === 5 && sweep.audios.every(Boolean) &&
        !/failed/i.test(sweep.status), JSON.stringify(sweep));
      mark("measurable SNR gain across at least three noisy levels",
        sweep.gains.filter((gain)=>gain>0).length >= 3, JSON.stringify(sweep.gains));
    } else {
      await waitFor(sid, `!document.querySelector('#readout')?.hidden &&
        !document.querySelector('#abGrid')?.hidden`, 180_000, `${label} real GTCRN denoising`);
      const result = await evaluate(sid, `({backend:document.querySelector('#rBackend')?.textContent,
        frames:document.querySelector('#rFrames')?.textContent,
        rtf:document.querySelector('#rRtf')?.textContent,
        gain:document.querySelector('#metrics')?.textContent,
        audio:[...document.querySelectorAll('#abGrid audio')].map(a=>a.src.startsWith('blob:')),
        inside:document.querySelector('#insideRows')?.children.length||0,
        status:document.querySelector('#status')?.textContent,
        download:document.querySelector('#dl')?.href.startsWith('blob:')})`);
      mark("real WASM denoising and two playable audio waveforms",
        result.backend === "WASM" && result.audio.length === 2 && result.audio.every(Boolean) &&
        Number.parseFloat(result.rtf) > 0 && result.gain?.length > 0 &&
        (rung === "practical" || Number(result.frames) > 0) && !/failed/i.test(result.status),
        JSON.stringify(result));
      if (rung === "practical") mark("enhanced WAV downloadable from upload-only route",
        result.download && /Noise floor drop/.test(result.gain), JSON.stringify(result));
      else {
        mark("measurable positive JFK SNR improvement", /SNR improvement/.test(result.gain) &&
          /\+\d+(?:\.\d+)? dB/.test(result.gain) && !/\+0\.0 dB/.test(result.gain), result.gain);
        if (rung === "overview") mark("see-inside real noise-floor measurement table", result.inside >= 3);
      }
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
    results.push({route,viewport,pass:ok});
    if (page) await closePage(cdp, page.targetId).catch(()=>{});
  }
}
try {
  const started = await startServer(); server = started.server; server.port = started.port;
  chrome = await launchChrome({userDataDir:PROFILE,resetProfile:true,removeProfileOnKill:false});
  cdp = new CDP(chrome.ws);
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) await drive(rung, viewport);
  }
} catch (e) {
  console.error(`FATAL ${String(e.stack||e)}`);
} finally {
  try { if (chrome) await chrome.kill({removeProfile:false}); }
  finally {
    try { if (server) await new Promise((resolve)=>server.close(resolve)); }
    finally { rmSync(PROFILE,{recursive:true,force:true}); }
  }
}
const succeeded = results.length === 10 && results.every(r=>r.pass) && checks === 72 && checks === passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["log","-n1","--format=%H","HEAD","--",FAMILY,
    "scripts/validate-gtcrn-speech-enhancement.mjs",`:(exclude)${FAMILY}/acceptance.json`,
    `:(exclude)${FAMILY}/acceptance-run.json`], {cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD,JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells: real GTCRN WASM speech denoising on every rung; positive SNR gain on bundled JFK overview/basics and five-level wild sweep; overview see-inside measurements; real user WAV upload and downloadable enhancement on practical (no bundled JFK credit); GTCRN→Whisper-tiny.en returns two playable transcriptions and intelligible enhanced JFK on multi-model. Four visible JFK speech credits per viewport click through to pinned GitHub ledger hash, creator, public-domain licence and GTCRN path. TED never selected. Responsive/console/network checks; screenshots outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
