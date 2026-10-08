#!/usr/bin/env node
// Five Moonshine routes × desktop/mobile, real ASR and multi-model Qwen answer.
// JFK jfk.wav is bundled on overview/basics/wild/multi, never practical (upload-only).
// The unrelated TED sample is neither selected nor modified.
// Stages: onnx-community/moonshine-base-ONNX (65MB primary ASR),
// onnx-community/Qwen2.5-0.5B-Instruct (400MB q4 multi-model second stage).
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer } from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const FAMILY = "models/moonshine-asr";
const RECORD = join(repoRoot,FAMILY,"acceptance-run.json");
const PROFILE = mkdtempSync(join(tmpdir(),"moonshine-acceptance-"));
const EVIDENCE = process.env.MOONSHINE_CREDIT_EVIDENCE_DIR || join(tmpdir(),"7nr-w6-moonshine-credit");
const JFK_WAV = join(repoRoot,FAMILY,"jfk.wav");
const CREDIT_URL = "https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES = {
  overview:"models/moonshine-asr/", basics:"models/moonshine-asr/basics/",
  practical:"models/moonshine-asr/practical/", wild:"models/moonshine-asr/wild/",
  multi:"models/moonshine-asr/multi-model/",
};
const VIEWPORTS = {desktop:DESKTOP,mobile:MOBILE};
mkdirSync(EVIDENCE,{recursive:true});
const sleep = ms => new Promise(resolve=>setTimeout(resolve,ms));
const results = [];
let checks=0,passed=0,server,chrome,cdp;
function check(label,condition,detail="") {
  checks++;
  if(condition) passed++;
  console.log(`${condition?"PASS":"FAIL"}  ${label}${detail?` — ${String(detail).slice(0,260)}`:""}`);
  return condition;
}
async function evaluate(sid,expression,timeoutMs=45_000) {
  const {result}=await cdp.send("Runtime.evaluate",{
    expression:`(async()=>{try{return (${expression});}catch(e){return {__error:String(e?.message||e)};}})()`,
    awaitPromise:true,returnByValue:true,
  },sid,timeoutMs);
  if(result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(sid,expression,deadlineMs,label,intervalMs=1_000) {
  const start=Date.now();let nextLog=0;
  while(Date.now()-start<deadlineMs) {
    try {if(await evaluate(sid,expression)) return;}
    catch(e){if(Date.now()>=nextLog)console.log(`  [${label}] evaluation: ${String(e.message).slice(0,120)}`);}
    if(Date.now()>=nextLog){
      console.log(`  [${label}] waiting ${Math.round((Date.now()-start)/1000)}s`);
      nextLog=Date.now()+10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}
async function ensureReady(sid,label,rung) {
  const start=Date.now();let nextLog=0;
  while(Date.now()-start<12*60_000) {
    const snapshot=await evaluate(sid,`(() => {
      const loaders=[...document.querySelectorAll('.model-loader')];
      return {states:loaders.map(x=>x.dataset.state),checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim()),
        buttons:loaders.flatMap(x=>[...x.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim()))};
    })()`);
    if(Date.now()>=nextLog){
      console.log(`  [${label}] ${Math.round((Date.now()-start)/1000)}s ${JSON.stringify(snapshot)}`);
      nextLog=Date.now()+10_000;
    }
    if(snapshot.states.length===(rung==="multi"?2:1) &&
      snapshot.states.every(x=>x==="ready")) return snapshot;
    await evaluate(sid,`(() => {
      const buttons=[...document.querySelectorAll('.model-loader button')].filter(b=>
        /Download|Retry|Re-download|Continue/i.test(b.textContent) && !b.disabled);
      buttons.forEach(b=>b.click());return buttons.length;
    })()`);
    await sleep(1_500);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model preparation`);
}
async function verifyCredit(sid,label,rung,viewport,mark) {
  const credit=await evaluate(sid,`(() => {
    const a=[...document.querySelectorAll('a')].find(x=>x.textContent.trim()===
      'JFK audio source and attribution record');
    if(!a)return null;
    a.scrollIntoView({block:'center'});
    return {href:a.href,visible:!!a.getClientRects().length,context:a.parentElement.textContent.trim()};
  })()`);
  if(rung==="practical") await evaluate(sid,
    `(() => {document.querySelector('#readout')?.scrollIntoView({block:'center'});return true;})()`);
  await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-${rung}.png`));
  if(rung==="practical") {mark("no false bundled JFK credit on upload-only route",!credit);return;}
  const visible=credit?.visible && credit.href.startsWith(CREDIT_URL) &&
    /John F\. Kennedy/.test(credit.context) && /public domain/.test(credit.context) &&
    /bundled JFK speech clip alone/.test(credit.context);
  let responseStatus=0;
  cdp.on(msg=>{
    if(msg.sessionId===sid && msg.method==="Network.responseReceived" &&
      msg.params.type==="Document" && msg.params.response.url.startsWith(CREDIT_URL)) {
      responseStatus=msg.params.response.status;
    }
  });
  if(visible) {
    await evaluate(sid,`(() => {
      const a=[...document.querySelectorAll('a')].find(x=>x.textContent.trim()===
        'JFK audio source and attribution record');
      setTimeout(()=>a.click(),0);return true;
    })()`);
    await waitFor(sid,`location.href.startsWith(${JSON.stringify(CREDIT_URL)})`,
      30_000,`${label} credit navigation`,500);
    await waitFor(sid,
      `document.body?.innerText.includes('627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b')`,
      30_000,`${label} credit content`,500);
  }
  const content=visible && await evaluate(sid,`(() => {
    const body=document.body?.innerText||'';
    return ['627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b',
      'John F. Kennedy','Public domain','models/moonshine-asr/jfk.wav']
      .map(text=>body.includes(text));
  })()`);
  mark("visible JFK credit click reaches content-verified ledger",
    visible && responseStatus>=200 && responseStatus<400 && content?.every(Boolean),
    JSON.stringify({credit,responseStatus,content}));
}
async function attachPracticalFile(sid) {
  const doc=await cdp.send("DOM.getDocument",{},sid,10_000);
  const node=await cdp.send("DOM.querySelector",{nodeId:doc.root.nodeId,selector:"#file"},sid,10_000);
  if(!node.nodeId)throw new Error("practical upload input missing");
  await cdp.send("DOM.setFileInputFiles",{files:[JFK_WAV],nodeId:node.nodeId},sid,10_000);
  await evaluate(sid,`(() => {document.querySelector('#file').dispatchEvent(new Event('change'));return true;})()`);
}
const jfkspeech=/fellow|american|country|ask/i;
const backend=/^(WASM|WEBGPU)$/;
async function drive(rung,viewport) {
  const route=ROUTES[rung],label=`${rung}@${viewport}`;
  let page,ok=true;
  const mark=(name,condition,detail="")=>{ok=check(`${label} ${name}`,condition,detail)&&ok;};
  try {
    console.log(`\n=== ${label}: ${route} ===`);
    page=await openPage(cdp,`http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
    const sid=page.sessionId;
    await setViewport(cdp,sid,VIEWPORTS[viewport]);
    const loader=await ensureReady(sid,label,rung);
    mark("bounded local cache checks",loader.checks.length===(rung==="multi"?2:1) &&
      loader.checks.every(n=>Number.isFinite(n)&&n<=450),JSON.stringify(loader.checks));
    if(rung==="practical") {
      await attachPracticalFile(sid);
      await waitFor(sid,`!document.querySelector('#run')?.disabled &&
        document.querySelector('#file')?.files[0]?.name==='jfk.wav' &&
        document.querySelector('#transcript')?.textContent==='Ready — hit Transcribe & match.'`,
        30_000,`${label} real user file decoded`);
      mark("native WAV user upload produces active practical inference control",
        await evaluate(sid,`document.querySelector('#player')?.src.startsWith('blob:') &&
          document.querySelector('#file')?.files[0]?.name==='jfk.wav'`));
    } else if(rung==="wild" || rung==="multi") {
      const chosen=await evaluate(sid,`(() => {
        const b=[...document.querySelectorAll('#samples button')].find(x=>x.dataset.src?.endsWith('/jfk.wav'));
        if(!b)return null;b.click();return b.dataset.src;
      })()`);
      mark("selects bundled JFK sample without selecting TED",/jfk\.wav$/.test(chosen),String(chosen));
    } else {
      const sample=await evaluate(sid,`({src:document.querySelector('#player')?.getAttribute('src'),
        label:document.querySelector('#clipLabel')?.textContent})`);
      mark("default JFK speech clip selected without choosing TED",
        /(?:^|\/)jfk\.wav$/.test(sample.src) && /JFK/.test(sample.label),JSON.stringify(sample));
    }
    if(rung==="wild") {
      await waitFor(sid,`document.querySelector('#status')?.textContent==='Done streaming the clip.' &&
        Number(document.querySelector('#rWins')?.textContent)>0`,180_000,
        `${label} real rolling JFK sample captions`);
      const streamed=await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
        wins:Number(document.querySelector('#rWins')?.textContent),
        avg:document.querySelector('#rAvg')?.textContent,
        speed:document.querySelector('#rSpeed')?.textContent,
        caption:document.querySelector('#caption')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real multi-window Moonshine JFK sample captions",backend.test(streamed.backend) &&
        streamed.wins>=2 && /^\d+(?:\.\d+)? s$/.test(streamed.avg) &&
        jfkspeech.test(streamed.caption) && streamed.status==='Done streaming the clip.',
        JSON.stringify(streamed));
      await evaluate(sid,`(() => {document.querySelector('#win').value='2';
        document.querySelector('#mic').click();return true;})()`);
      await waitFor(sid,`document.querySelector('#mic')?.textContent?.includes('Stop') &&
        Number(document.querySelector('#rWins')?.textContent)>=1 &&
        Number(document.querySelector('#rWins')?.textContent)<${streamed.wins} &&
        /fellow|american|country|ask/i.test(document.querySelector('#caption')?.textContent||'')`,
        90_000,`${label} fresh fake-mic JFK rolling caption`);
      const live=await evaluate(sid,`({mic:document.querySelector('#mic')?.textContent,
        wins:document.querySelector('#rWins')?.textContent,
        caption:document.querySelector('#caption')?.textContent,
        backend:document.querySelector('#rBackend')?.textContent})`);
      mark("live microphone records and transcribes a fresh JFK window",
        /Stop/.test(live.mic)&&Number(live.wins)>=1&&Number(live.wins)<streamed.wins&&
        backend.test(live.backend)&&jfkspeech.test(live.caption),
        JSON.stringify(live));
      await evaluate(sid,`(() => {document.querySelector('#caption').scrollIntoView({block:'center'});return true;})()`);
      await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-wild-live.png`));
      await evaluate(sid,`(() => {document.querySelector('#mic').click();return true;})()`);
      mark("fake microphone stopped after real captions",await evaluate(sid,
        `document.querySelector('#mic')?.textContent?.includes('Live mic captions')`));
    } else {
      await waitFor(sid,`!document.querySelector('#run')?.disabled`,45_000,`${label} audio decoded`);
      await evaluate(sid,`(() => {document.querySelector('#run').click();return true;})()`);
      if(rung==="multi") {
        await waitFor(sid,`!document.querySelector('#readout')?.hidden &&
          !document.querySelector('#run')?.disabled &&
          /^(Done\.|Failed:)/.test(document.querySelector('#status')?.textContent||'')`,
          600_000,`${label} Moonshine→Qwen 465MB real chain`);
        const answer=await evaluate(sid,`({heard:document.querySelector('#heard')?.textContent,
          answer:document.querySelector('#answer')?.textContent,
          asr:document.querySelector('#rAsr')?.textContent,
          asrMs:document.querySelector('#rAsrMs')?.textContent,
          llm:document.querySelector('#rLlm')?.textContent,
          llmMs:document.querySelector('#rLlmMs')?.textContent,
          status:document.querySelector('#status')?.textContent})`);
        mark("real Moonshine JFK speech decoded before second-stage Qwen answer",
          jfkspeech.test(answer.heard)&&backend.test(answer.asr)&&/^\d+(?:\.\d+)? s$/.test(answer.asrMs),
          JSON.stringify(answer));
        mark("400MB q4 Qwen stage actually generates a non-placeholder answer",
          backend.test(answer.llm)&&/^\d+(?:\.\d+)? s$/.test(answer.llmMs)&&
          !!answer.answer?.trim()&&!/^(…|\(no speech detected\))$/.test(answer.answer)&&
          answer.status==='Done.',JSON.stringify(answer));
        await evaluate(sid,`(() => {document.querySelector('#answer').scrollIntoView({block:'center'});return true;})()`);
        await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-multi-answer.png`));
      } else {
        await waitFor(sid,`!document.querySelector('#readout')?.hidden &&
          !document.querySelector('#run')?.disabled &&
          /^(Done\.|Transcription failed:|Failed:)/.test(document.querySelector('#status')?.textContent||'')`,
          180_000,`${label} real Moonshine ASR`);
        const result=await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
          ms:document.querySelector('#rMs')?.textContent,
          speed:document.querySelector('#rSpeed')?.textContent,
          tokens:document.querySelector('#rTok')?.textContent,
          text:document.querySelector('#transcript')?.textContent,
          inside:!document.querySelector('#insideStats')?.hidden,
          factor:document.querySelector('#iRtf')?.textContent,
          match:document.querySelector('#matchLine')?.textContent,
          status:document.querySelector('#status')?.textContent})`);
        mark("real Moonshine ASR transcribes JFK audio on this route",
          backend.test(result.backend)&&/^\d+(?:\.\d+)? s$/.test(result.ms)&&
          jfkspeech.test(result.text)&&/× real time/.test(result.speed)&&result.status==='Done.',
          JSON.stringify(result));
        if(rung==="overview")mark("see-inside real audio speed profile",
          result.inside&&Number.parseFloat(result.factor)>0&&Number(result.tokens)>0,
          JSON.stringify(result));
        if(rung==="practical")mark("JFK user upload does not falsely trigger unrelated voice command",
          result.match==='No command matched — try one of the phrases above.',
          JSON.stringify({text:result.text,match:result.match}));
      }
    }
    const hygiene=await evaluate(sid,`({overflow:document.documentElement.scrollWidth-innerWidth,
      named:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim())})`);
    mark("responsive named controls",hygiene.overflow<=1&&hygiene.named,JSON.stringify(hygiene));
    mark("console/network clean",page.errors.length===0&&page.netFailures.length===0,
      JSON.stringify({errors:page.errors,network:page.netFailures}));
    await verifyCredit(sid,label,rung,viewport,mark);
  } catch(e) {
    ok=false;
    check(`${label} completed`,false,String(e.stack||e).slice(0,400));
  } finally {
    results.push({route,viewport,pass:ok});
    if(page)await closePage(cdp,page.targetId).catch(()=>{});
  }
}
try {
  const started=await startServer();server=started.server;server.port=started.port;
  chrome=await launchChrome({userDataDir:PROFILE,resetProfile:true,removeProfileOnKill:false,
    extraArgs:["--use-fake-ui-for-media-stream","--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${JFK_WAV}`]});
  cdp=new CDP(chrome.ws);
  for(const rung of Object.keys(ROUTES))for(const viewport of Object.keys(VIEWPORTS))await drive(rung,viewport);
} catch(e) {
  console.error(`FATAL ${String(e.stack||e)}`);
} finally {
  try {if(chrome)await chrome.kill({removeProfile:false});}
  finally {
    try {if(server)await new Promise(resolve=>server.close(resolve));}
    finally {rmSync(PROFILE,{recursive:true,force:true});}
  }
}
const succeeded=results.length===10&&results.every(r=>r.pass)&&checks===70&&checks===passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if(WRITE_RUN&&succeeded) {
  const commit=execFileSync("git",["log","-n1","--format=%H","HEAD","--",FAMILY,
    "scripts/validate-moonshine-asr.mjs",`:(exclude)${FAMILY}/acceptance.json`,
    `:(exclude)${FAMILY}/acceptance-run.json`],{cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD,JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells: real Moonshine ASR JFK transcriptions on overview/basics, measured speed profile on overview, practical native JFK WAV user upload with no bundled JFK credit and no false voice command, wild real multi-window JFK sample captions plus fake-device JFK live microphone captions then stop; multi-model loads BOTH 65MB Moonshine and 400MB q4 Qwen second stage (~465MB total) and produces JFK transcript plus a generated answer (not evaluated for factual correctness). Four JFK-only sample credits per viewport click through to pinned GitHub ledger hash, creator, public-domain licence and path; TED never selected. Responsive/console/network checks; screenshots outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded?0:1);
