#!/usr/bin/env node
// Five public Silero VAD routes × desktop/mobile. The bundled Speech clip is
// JFK only on overview/basics/practical/multi; wild is microphone-only. A JFK
// fake microphone drives Wild, not a bundled Wild credit. Never select TED.
// Real stages: onnx-community/silero-vad (raw ONNX Runtime WASM), then
// onnx-community/whisper-base (Transformers.js in multi-model).
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer } from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const FAMILY = "models/silero-vad";
const RECORD = join(repoRoot, FAMILY, "acceptance-run.json");
const PROFILE = mkdtempSync(join(tmpdir(), "silero-acceptance-"));
const EVIDENCE = process.env.SILERO_CREDIT_EVIDENCE_DIR || join(tmpdir(), "7nr-w5-silero-credit");
const JFK_WAV = join(repoRoot, FAMILY, "speech.wav");
const CREDIT_URL = "https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES = {
  overview: "models/silero-vad/", basics: "models/silero-vad/basics/",
  practical: "models/silero-vad/practical/", wild: "models/silero-vad/wild/",
  multi: "models/silero-vad/multi-model/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
mkdirSync(EVIDENCE, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
let checks = 0, passed = 0, server, chrome, cdp;
function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0,260)}` : ""}`);
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
  const start = Date.now(); let nextLog = 0;
  while (Date.now() - start < deadlineMs) {
    try { if (await evaluate(sid, expression)) return; }
    catch (e) { if (Date.now() >= nextLog) console.log(`  [${label}] evaluation: ${String(e.message).slice(0,120)}`); }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now()-start)/1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}
async function ensureReady(sid, label, rung) {
  const start = Date.now(); let nextLog = 0;
  while (Date.now() - start < 12*60_000) {
    const snapshot = await evaluate(sid, `(() => {
      const loaders=[...document.querySelectorAll('.model-loader')];
      return {states:loaders.map(x=>x.dataset.state),checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim()),
        buttons:loaders.flatMap(x=>[...x.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim()))};
    })()`);
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] ${Math.round((Date.now()-start)/1000)}s ${JSON.stringify(snapshot)}`);
      nextLog = Date.now() + 10_000;
    }
    if (snapshot.states.length === (rung === "multi" ? 2 : 1) &&
        snapshot.states.every(x=>x === "ready")) return snapshot;
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
  if (rung !== "wild") await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-${rung}.png`));
  if (rung === "wild") { mark("no false bundled JFK credit on mic-only route", !credit); return; }
  const visible = credit?.visible && credit.href.startsWith(CREDIT_URL) &&
    /John F\. Kennedy/.test(credit.context) && /public domain/.test(credit.context) &&
    /Speech clip alone \(not the other samples\)/.test(credit.context);
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
    await waitFor(sid,`location.href.startsWith(${JSON.stringify(CREDIT_URL)})`,
      30_000, `${label} credit navigation`, 500);
    await waitFor(sid,
      `document.body?.innerText.includes('627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b')`,
      30_000, `${label} credit content`, 500);
  }
  const content = visible && await evaluate(sid, `(() => {
    const body=document.body?.innerText||'';
    return ['627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b',
      'John F. Kennedy','Public domain','models/silero-vad/speech.wav']
      .map(text=>body.includes(text));
  })()`);
  mark("visible JFK credit click reaches content-verified ledger",
    visible && responseStatus>=200 && responseStatus<400 && content?.every(Boolean),
    JSON.stringify({credit,responseStatus,content}));
}
async function drive(rung, viewport) {
  const route = ROUTES[rung], label = `${rung}@${viewport}`;
  let page, ok = true;
  const mark = (name,condition,detail="") => { ok = check(`${label} ${name}`,condition,detail) && ok; };
  try {
    console.log(`\n=== ${label}: ${route} ===`);
    page = await openPage(cdp,`http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
    const sid = page.sessionId;
    await setViewport(cdp,sid,VIEWPORTS[viewport]);
    const loader = await ensureReady(sid,label,rung);
    mark("bounded local cache checks", loader.checks.length === (rung === "multi" ? 2 : 1) &&
      loader.checks.every(n=>Number.isFinite(n) && n<=450),JSON.stringify(loader.checks));
    if (rung !== "wild") {
      const sample = await evaluate(sid,`({selected:document.querySelector('#samples button[aria-pressed="true"]')?.dataset.src,
        others:[...document.querySelectorAll('#samples button[aria-pressed="false"]')].map(b=>b.dataset.src)})`);
      mark("default JFK Speech sample selected without using unrelated clips",
        /(?:^|\/)speech\.wav$/.test(sample.selected) && sample.others.length >= 1,
        JSON.stringify(sample));
      await waitFor(sid,`!document.querySelector('#run')?.disabled`,30_000,`${label} JFK WAV decoded`);
      await evaluate(sid,`(() => { document.querySelector('#run').click(); return true; })()`);
    } else {
      await waitFor(sid,`!document.querySelector('#micBtn')?.disabled`,25_000,`${label} mic ready`);
      await evaluate(sid,`(() => {
        globalThis.__sawSpeaking = false;
        const verdict=document.querySelector('#verdict');
        new MutationObserver(()=>{if(verdict.textContent.includes('Speaking')) globalThis.__sawSpeaking=true;})
          .observe(verdict,{childList:true,characterData:true,subtree:true});
        document.querySelector('#micBtn').click(); return true;
      })()`);
    }
    if (rung === "overview" || rung === "basics") {
      await waitFor(sid,`!document.querySelector('#readout')?.hidden &&
        !document.querySelector('#verdict')?.hidden && !document.querySelector('#run')?.disabled`,
        120_000,`${label} real VAD inference`);
      const result = await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,ratio:document.querySelector('#rRatio')?.textContent,
        segs:document.querySelector('#rSegs')?.textContent,
        verdict:document.querySelector('#verdict')?.textContent,
        table:document.querySelectorAll('#segTable tbody tr').length,
        status:document.querySelector('#status')?.textContent})`);
      mark("real WASM VAD finds JFK speech frames",
        result.backend === "wasm (onnxruntime-web)" && /^\d+ ms$/.test(result.ms) &&
        Number.parseFloat(result.ratio)>0 && /Speech detected/.test(result.verdict) &&
        (rung === "basics" || Number(result.segs)>0) && !/failed/i.test(result.status),
        JSON.stringify(result));
      if (rung === "overview") mark("see-inside real probability and segment table",
        result.table >= 1 && Number(result.segs) === result.table,JSON.stringify(result));
    } else if (rung === "practical") {
      await waitFor(sid,`!document.querySelector('#readout')?.hidden &&
        !document.querySelector('#run')?.disabled`,120_000,`${label} real JFK segmentation`);
      const segmented = await evaluate(sid,`({segs:Number(document.querySelector('#rSegs')?.textContent),
        duration:Number.parseFloat(document.querySelector('#rDur')?.textContent),
        trimmed:Number.parseFloat(document.querySelector('#rTrim')?.textContent),
        cut:Number.parseFloat(document.querySelector('#rCut')?.textContent),
        table:document.querySelectorAll('#segTable tbody tr').length,
        exportVisible:!document.querySelector('#exportRow')?.hidden,
        status:document.querySelector('#status')?.textContent})`);
      mark("real speech segmentation trims JFK pauses",
        segmented.segs >= 1 && segmented.table === segmented.segs &&
        segmented.trimmed > 0 && segmented.cut > 0 && segmented.duration > segmented.trimmed &&
        segmented.exportVisible && !/failed/i.test(segmented.status),JSON.stringify(segmented));
      await evaluate(sid,`(() => { document.querySelector('#exportBtn').click(); return true; })()`);
      const exported = await evaluate(sid,`(async() => {
        const a=document.querySelector('#dl');
        if (!a?.href.startsWith('blob:')) return null;
        const buf=await (await fetch(a.href)).arrayBuffer();
        return {href:a.href,download:a.download,bytes:buf.byteLength,
          riff:new TextDecoder().decode(buf.slice(0,4)),status:document.querySelector('#status')?.textContent};
      })()`);
      mark("trimmed WAV exported with actual RIFF audio bytes",
        exported?.download === "trimmed-speech.wav" && exported.bytes > 44 &&
        exported.riff === "RIFF" && exported.status === "Exported trimmed-speech.wav.",
        JSON.stringify(exported));
    } else if (rung === "wild") {
      await waitFor(sid,`document.querySelector('#micBtn')?.textContent?.includes('Stop listening') &&
        Number(document.querySelector('#rFrames')?.textContent)>0 &&
        /Live\./.test(document.querySelector('#status')?.textContent||'')`,
        60_000,`${label} live fake-device JFK mic`);
      const live = await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
        frames:Number(document.querySelector('#rFrames')?.textContent),
        prob:document.querySelector('#probReadout')?.textContent,
        state:document.querySelector('#micState')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real live WASM microphone VAD analyses JFK frames",
        live.backend === "wasm (onnxruntime-web)" && live.frames>0 &&
        /speech probability: [01]\.\d+/.test(live.prob) && /Listening/.test(live.state) &&
        !/error|blocked/i.test(live.status),JSON.stringify(live));
      await waitFor(sid,`globalThis.__sawSpeaking &&
        Number.parseFloat(document.querySelector('#rTalk')?.textContent)>0 &&
        Number(document.querySelector('#rFrames')?.textContent)>10`,
        60_000,`${label} real JFK voice seen by live VAD`);
      const talking = await evaluate(sid,`({frames:Number(document.querySelector('#rFrames')?.textContent),
        share:document.querySelector('#rTalk')?.textContent,
        prob:document.querySelector('#probReadout')?.textContent,
        verdict:document.querySelector('#verdict')?.textContent,
        sawSpeaking:globalThis.__sawSpeaking,
        meter:document.querySelector('#meterFill')?.style.inlineSize})`);
      mark("live JFK speech probability exceeds threshold in real frame stream",
        talking.frames>10 && talking.sawSpeaking && Number.parseFloat(talking.share)>0 &&
        Number.parseFloat(talking.prob?.split(':').pop())>=0 &&
        Number.parseFloat(talking.meter)>=0,JSON.stringify(talking));
      await waitFor(sid,`document.querySelector('#verdict')?.textContent?.includes('Speaking')`,
        25_000,`${label} visible live Speaking state`);
      await evaluate(sid,`(() => { document.querySelector('#verdict').scrollIntoView({block:'center'}); return true; })()`);
      await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-wild.png`));
      await evaluate(sid,`(() => { document.querySelector('#micBtn').click(); return true; })()`);
      mark("fake microphone stream stopped after inference",await evaluate(sid,
        `document.querySelector('#micBtn')?.textContent?.includes('Start listening') &&
          document.querySelector('#micState')?.textContent === 'Stopped.'`));
    } else {
      await waitFor(sid,`!document.querySelector('#readout')?.hidden &&
        !document.querySelector('#run')?.disabled &&
        /^(Done — transcribed|No speech found|Failed:)/.test(document.querySelector('#status')?.textContent||'')`,
        300_000,`${label} real VAD→Whisper-base`);
      const chain = await evaluate(sid,`({segs:Number(document.querySelector('#rSegs')?.textContent),
        sent:Number.parseFloat(document.querySelector('#rSent')?.textContent),
        skip:Number.parseFloat(document.querySelector('#rSkip')?.textContent),
        duration:Number.parseFloat(document.querySelector('#rDur')?.textContent),
        rows:document.querySelectorAll('#transcript .seg-line').length,
        text:document.querySelector('#transcript')?.textContent,
        status:document.querySelector('#status')?.textContent,
        asrErrors:document.querySelectorAll('#transcript .status.err').length})`);
      mark("real Silero JFK segments are sent to Whisper while silence is skipped",
        chain.segs >= 1 && chain.sent > 0 && chain.skip > 0 &&
        chain.duration > chain.sent && chain.rows === chain.segs && chain.asrErrors === 0 &&
        /^Done — transcribed \d+ speech segment\(s\), skipped [\d.]+ s of silence\.$/.test(chain.status),
        JSON.stringify(chain));
      mark("real Whisper-base transcribes JFK words, no pending placeholder",
        /fellow|american|country|ask/i.test(chain.text) &&
        !/transcribing…|ASR failed/.test(chain.text),JSON.stringify(chain.text));
      await evaluate(sid,`(() => { document.querySelector('#transcript').scrollIntoView({block:'center'}); return true; })()`);
      await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-multi-transcript.png`));
    }
    const hygiene = await evaluate(sid,`({overflow:document.documentElement.scrollWidth-innerWidth,
      named:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim())})`);
    mark("responsive named controls",hygiene.overflow<=1 && hygiene.named,JSON.stringify(hygiene));
    mark("console/network clean",page.errors.length===0 && page.netFailures.length===0,
      JSON.stringify({errors:page.errors,network:page.netFailures}));
    await verifyCredit(sid,label,rung,viewport,mark);
  } catch(e) {
    ok = false;
    check(`${label} completed`,false,String(e.stack||e).slice(0,400));
  } finally {
    results.push({route,viewport,pass:ok});
    if (page) await closePage(cdp,page.targetId).catch(()=>{});
  }
}
try {
  const started = await startServer(); server=started.server; server.port=started.port;
  chrome = await launchChrome({userDataDir:PROFILE,resetProfile:true,removeProfileOnKill:false,
    extraArgs:["--use-fake-ui-for-media-stream","--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${JFK_WAV}`]});
  cdp = new CDP(chrome.ws);
  mkdirSync(join(PROFILE,"downloads"),{recursive:true});
  await cdp.send("Browser.setDownloadBehavior",{behavior:"allow",downloadPath:join(PROFILE,"downloads")});
  for(const rung of Object.keys(ROUTES)) for(const viewport of Object.keys(VIEWPORTS)) await drive(rung,viewport);
} catch(e) {
  console.error(`FATAL ${String(e.stack||e)}`);
} finally {
  try { if(chrome) await chrome.kill({removeProfile:false}); }
  finally {
    try { if(server) await new Promise(resolve=>server.close(resolve)); }
    finally { rmSync(PROFILE,{recursive:true,force:true}); }
  }
}
const succeeded = results.length===10 && results.every(r=>r.pass) && checks===68 && checks===passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if(WRITE_RUN && succeeded) {
  const commit=execFileSync("git",["log","-n1","--format=%H","HEAD","--",FAMILY,
    "scripts/validate-silero-vad.mjs",`:(exclude)${FAMILY}/acceptance.json`,
    `:(exclude)${FAMILY}/acceptance-run.json`],{cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD,JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells: real on-device Silero WASM VAD inference finds JFK speech segments and per-frame probabilities on overview/basics; overview see-inside segment table; practical trims pauses and exports a real RIFF WAV; wild fake-device live microphone consumes JFK input with positive speaking frames, then stops (no bundled JFK credit); multi-model sends actual speech segments (not silence) into Whisper-base yielding JFK words. Four visible JFK-only sample credits per viewport click through to pinned GitHub ledger hash, creator, public-domain licence and path. TED never selected. Responsive/console/network checks; screenshots outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded?0:1);
