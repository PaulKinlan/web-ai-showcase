#!/usr/bin/env node
// W8: real 126-way language ID on five routes × desktop/mobile, JFK English
// source attribution verified by CLICKED destination content. No TED audio exists
// in this family. Xenova/mms-lid-126 q8 ~974MB; multi's actual second stage
// onnx-community/whisper-base_timestamped q8 ~120MB (~1.09GB TOTAL weights).
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer } from "./browser.mjs";

const WRITE_RUN=process.argv.includes("--write-run");
const FAMILY="models/spoken-language-id";
const RECORD=join(repoRoot,FAMILY,"acceptance-run.json");
const PROFILE=mkdtempSync(join(tmpdir(),"spoken-lid-acceptance-"));
const EVIDENCE=process.env.SPOKEN_LID_EVIDENCE_DIR||join(tmpdir(),"7nr-w8-spoken-lid-credit");
const JFK=join(repoRoot,FAMILY,"sample-eng.wav");
const CREDIT_URL="https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES={overview:"models/spoken-language-id/",basics:"models/spoken-language-id/basics/",
  practical:"models/spoken-language-id/practical/",wild:"models/spoken-language-id/wild/",
  multi:"models/spoken-language-id/multi-model/"};
const VIEWPORTS={desktop:DESKTOP,mobile:MOBILE};
mkdirSync(EVIDENCE,{recursive:true});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const backend=/^(WASM|WEBGPU)$/, jfkSpeech=/fellow|american|country|ask/i;
let checks=0,passed=0,server,chrome,cdp;
const results=[];
function check(label,condition,detail=""){
  checks++;if(condition)passed++;
  console.log(`${condition?"PASS":"FAIL"}  ${label}${detail?` — ${String(detail).slice(0,370)}`:""}`);
  return condition;
}
async function evaluate(sid,expression,timeoutMs=45_000){
  const {result}=await cdp.send("Runtime.evaluate",{
    expression:`(async()=>{try{return (${expression});}catch(e){return {__error:String(e?.message||e)};}})()`,
    awaitPromise:true,returnByValue:true,
  },sid,timeoutMs);
  if(result?.value?.__error)throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(sid,expression,deadlineMs,label,intervalMs=1_000){
  const started=Date.now();let nextLog=0;
  while(Date.now()-started<deadlineMs){
    try{if(await evaluate(sid,expression))return;}
    catch(e){if(Date.now()>=nextLog)console.log(`  [${label}] evaluation: ${String(e.message).slice(0,120)}`);}
    if(Date.now()>=nextLog){console.log(`  [${label}] waiting ${Math.round((Date.now()-started)/1000)}s`);nextLog=Date.now()+10_000;}
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}
async function ensureReady(sid,label){
  const started=Date.now();let nextLog=0;
  while(Date.now()-started<12*60_000){
    const info=await evaluate(sid,`(() => {const loaders=[...document.querySelectorAll('.model-loader')];
      return {states:loaders.map(x=>x.dataset.state),checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim())};})()`);
    if(Date.now()>=nextLog){console.log(`  [${label}] ${Math.round((Date.now()-started)/1000)}s ${JSON.stringify(info)}`);nextLog=Date.now()+10_000;}
    if(info.states.length===1&&info.states[0]==="ready")return info;
    await evaluate(sid,`(() => {const buttons=[...document.querySelectorAll('.model-loader button')]
      .filter(b=>/Download|Retry|Re-download|Continue/i.test(b.textContent)&&!b.disabled);
      buttons.forEach(b=>b.click());return buttons.length;})()`);
    await sleep(1_500);
  }
  throw new Error(`hard timeout after 720000ms: ${label} 974MB model preparation`);
}
async function verifyCredit(sid,label,mark){
  const credit=await evaluate(sid,`(() => {const a=[...document.querySelectorAll('a')]
    .find(x=>x.textContent.trim()==='JFK audio source and attribution record');
    if(!a)return null;a.scrollIntoView({block:'center'});
    return {href:a.href,visible:!!a.getClientRects().length,context:a.parentElement.textContent.trim()};})()`);
  await screenshot(cdp,sid,join(EVIDENCE,`${label.replace('@','-')}.png`));
  const visible=credit?.visible&&credit.href.startsWith(CREDIT_URL)&&
    /John F\. Kennedy/.test(credit.context)&&/public domain/.test(credit.context)&&
    /English clip alone/.test(credit.context)&&/Other language clips have separate sources/.test(credit.context);
  let responseStatus=0;
  cdp.on(msg=>{
    if(msg.sessionId===sid&&msg.method==="Network.responseReceived"&&msg.params.type==="Document"&&
      msg.params.response.url.startsWith(CREDIT_URL))responseStatus=msg.params.response.status;
  });
  if(visible){
    await evaluate(sid,`(() => {const a=[...document.querySelectorAll('a')]
      .find(x=>x.textContent.trim()==='JFK audio source and attribution record');
      setTimeout(()=>a.click(),0);return true;})()`);
    await waitFor(sid,`location.href.startsWith(${JSON.stringify(CREDIT_URL)})`,30_000,`${label} credit click`,500);
    await waitFor(sid,`document.body?.innerText.includes('627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b')`,
      30_000,`${label} ledger destination content`,500);
  }
  const content=visible&&await evaluate(sid,`(() => {const body=document.body?.innerText||'';
    return ['627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b',
      'John F. Kennedy','Public domain','models/spoken-language-id/sample-eng.wav']
      .map(text=>body.includes(text));})()`);
  mark("visible JFK credit CLICK reaches content-verified pinned ledger",
    visible&&responseStatus>=200&&responseStatus<400&&content?.every(Boolean),
    JSON.stringify({credit,responseStatus,content}));
}
async function attachFile(sid){
  const doc=await cdp.send("DOM.getDocument",{},sid,10_000);
  const node=await cdp.send("DOM.querySelector",{nodeId:doc.root.nodeId,selector:"#file"},sid,10_000);
  if(!node.nodeId)throw new Error("Practical native upload input missing");
  await cdp.send("DOM.setFileInputFiles",{files:[JFK],nodeId:node.nodeId},sid,10_000);
  await evaluate(sid,`(() => {document.querySelector('#file').dispatchEvent(new Event('change'));return true;})()`);
}
async function drive(rung,viewport){
  const route=ROUTES[rung],label=`${rung}@${viewport}`;
  let page,ok=true;
  const mark=(name,condition,detail="")=>{ok=check(`${label} ${name}`,condition,detail)&&ok;};
  try{
    console.log(`\n=== ${label}: ${route} ===`);
    page=await openPage(cdp,`http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
    const sid=page.sessionId;
    await setViewport(cdp,sid,VIEWPORTS[viewport]);
    const initial=await evaluate(sid,`(() => {const b=document.querySelector('#samples button[aria-pressed="true"]');
      return {src:b?.dataset.src,english:!!document.querySelector('#samples button[data-src$="sample-eng.wav"]')};})()`);
    if(rung==="wild"){
      // Wild intentionally defaults Spanish. Select the distinct bundled JFK English
      // clip explicitly; never attribute Spanish to JFK or change its default.
      mark("Wild defaults to non-JFK Spanish, with JFK English chip available",
        initial.src?.endsWith("sample-spa.wav")&&initial.english,JSON.stringify(initial));
      // Let the initial Spanish decode settle before selecting JFK, or its
      // asynchronous completion could overwrite the newly selected English PCM.
      await waitFor(sid,`document.querySelector('#status')?.textContent==='Ready — set a length and Identify.'`,
        30_000,`${label} initial non-TED Spanish decoded`);
      await evaluate(sid,`(() => {document.querySelector('#samples button[data-src$="sample-eng.wav"]').click();return true;})()`);
    }
    const ready=await ensureReady(sid,label);
    mark("bounded local cache check for 974MB LID model",
      ready.checks.length===1&&Number.isFinite(ready.checks[0])&&ready.checks[0]<=450,JSON.stringify(ready.checks));
    await waitFor(sid,`!document.querySelector('#run')?.disabled&&
      document.querySelector('#samples button[aria-pressed="true"]')?.dataset.src?.endsWith('sample-eng.wav')`,
      45_000,`${label} JFK English decoded`);
    if(rung!=="wild")mark("bundled JFK English is default source, no TED",
      initial.src?.endsWith("sample-eng.wav")&&initial.english,JSON.stringify(initial));
    // Wild's initial-state check is the same universal source-selection check.
    await evaluate(sid,`(() => {document.querySelector('#run').click();return true;})()`);
    if(rung==="multi"){
      await waitFor(sid,`!document.querySelector('#pipe')?.hidden&&
        !document.querySelector('#readout')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Pipeline done:','Pipeline failed:'].some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        600_000,`${label} real 974MB LID + ~120MB timestamped Whisper chain`);
      const r=await evaluate(sid,`({lid:document.querySelector('#pLang')?.textContent,
        transcript:document.querySelector('#pText')?.textContent,
        lidMs:document.querySelector('#rLid')?.textContent,
        asrMs:document.querySelector('#rAsr')?.textContent,
        backend:document.querySelector('#rBackend')?.textContent,
        whisper:document.querySelector('#wStatus')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real LID stage identifies bundled JFK English BEFORE Whisper",
        /English \(eng\)/.test(r.lid)&&/^\d+ ms$/.test(r.lidMs)&&
        /^Pipeline done: English → transcript\.$/.test(r.status),JSON.stringify(r));
      mark("timestamped Whisper second stage truly transcribes JFK speech",
        jfkSpeech.test(r.transcript)&&/^\d+ ms$/.test(r.asrMs)&&
        backend.test(r.backend)&&r.whisper==='Whisper ready.',JSON.stringify(r));
      await evaluate(sid,`(() => {document.querySelector('#pText').scrollIntoView({block:'center'});return true;})()`);
      await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-multi-transcript.png`));
    }else if(rung==="wild"){
      await waitFor(sid,`!document.querySelector('#readout')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        !document.querySelector('#status')?.classList.contains('err')`,
        180_000,`${label} real JFK 3-second English classification`);
      const first=await evaluate(sid,`({code:document.querySelector('#vCode')?.textContent,
        dur:document.querySelector('#rDur')?.textContent,
        entropy:document.querySelector('#rEnt')?.textContent,
        margin:document.querySelector('#rMargin')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real first 3s of JFK classified English with 126-way uncertainty",
        first.code==='eng'&&first.dur==='3.0 s'&&Number.isFinite(Number(first.entropy))&&
        /^\d+ pts$/.test(first.margin)&&/English/.test(first.status),JSON.stringify(first));
      await evaluate(sid,`(() => {document.querySelector('#mix2').value='../sample-rus.wav';
        document.querySelector('#mix').click();return true;})()`);
      await waitFor(sid,`document.querySelector('#mix')?.checked&&
        document.querySelector('#mix2')?.value==='../sample-rus.wav'&&
        performance.getEntriesByType('resource').some(x=>x.name.endsWith('/sample-rus.wav'))`,
        30_000,`${label} non-TED JFK→Russian splice`);
      await evaluate(sid,`(() => {document.querySelector('#run').click();return true;})()`);
      await waitFor(sid,`!document.querySelector('#run')?.disabled&&
        document.querySelector('#rDur')?.textContent==='6.0 s'&&
        !document.querySelector('#status')?.classList.contains('err')`,180_000,
        `${label} genuine mixed-language inference`);
      const mixed=await evaluate(sid,`({dur:document.querySelector('#rDur')?.textContent,
        entropy:document.querySelector('#rEnt')?.textContent,
        margin:document.querySelector('#rMargin')?.textContent,
        code:document.querySelector('#vCode')?.textContent,
        mix:document.querySelector('#mix')?.checked,
        second:document.querySelector('#mix2')?.value})`);
      mark("real 6s JFK-English→Russian code-switch runs distinct inference",
        mixed.dur==='6.0 s'&&mixed.mix&&mixed.second==='../sample-rus.wav'&&
        Number.isFinite(Number(mixed.entropy))&&/^[a-z]{3}$/.test(mixed.code)&&
        /^\d+ pts$/.test(mixed.margin),JSON.stringify(mixed));
      await evaluate(sid,`(() => {document.querySelector('#mix').click();return true;})()`);
    }else if(rung==="practical"){
      await waitFor(sid,`!document.querySelector('#route')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        document.querySelector('#status')?.textContent?.startsWith('Routing as ')`,180_000,
        `${label} real JFK language routing`);
      const r=await evaluate(sid,`({lang:document.querySelector('#rLang')?.textContent,
        asr:document.querySelector('#rRoute')?.textContent,
        translate:document.querySelector('#rTrans')?.textContent,
        confidence:document.querySelector('#confNote')?.textContent,
        bars:document.querySelectorAll('#langs .lang-row').length})`);
      mark("real JFK English classification recommends Whisper route, not actual ASR",
        /English \(eng\)/.test(r.lang)&&/Whisper \(multilingual\)/.test(r.asr)&&
        /source = English/.test(r.translate)&&r.bars>=5,JSON.stringify(r));
      mark("confidence note reflects measured model distribution",
        /confidence/.test(r.confidence)&&r.confidence.length>24,JSON.stringify(r.confidence));
      await attachFile(sid);
      await waitFor(sid,`document.querySelector('#file')?.files[0]?.name==='sample-eng.wav'&&
        document.querySelector('#samples button[aria-pressed="true"]')===null&&
        !document.querySelector('#run')?.disabled`,45_000,`${label} native JFK user upload decoded`);
      await evaluate(sid,`(() => {document.querySelector('#run').click();return true;})()`);
      await waitFor(sid,`!document.querySelector('#route')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        document.querySelector('#status')?.textContent==='Routing as English.'`,180_000,
        `${label} user-upload JFK reclassified`);
      const upload=await evaluate(sid,`({file:document.querySelector('#file')?.files[0]?.name,
        lang:document.querySelector('#rLang')?.textContent,
        route:document.querySelector('#rRoute')?.textContent})`);
      mark("native WAV user-upload also runs real English route without source miscredit",
        upload.file==='sample-eng.wav'&&/English \(eng\)/.test(upload.lang)&&
        /Whisper \(multilingual\)/.test(upload.route),JSON.stringify(upload));
    }else{
      await waitFor(sid,`!document.querySelector('#verdict')?.hidden&&
        !document.querySelector('#readout')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Detected:','Identification failed:','Failed:'].some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        180_000,`${label} real JFK language ID`);
      const r=await evaluate(sid,`({code:document.querySelector('#vCode')?.textContent,
        name:document.querySelector('#vName')?.textContent,
        backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,
        bars:document.querySelectorAll('#langs .lang-row').length,
        inside:!document.querySelector('#insideWrap')?.hidden,
        rows:document.querySelectorAll('#insideRows tr').length,
        margin:document.querySelector('#iMargin')?.textContent,
        entropy:document.querySelector('#iEntropy')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real 126-way LID identifies JFK English with backend/latency/bars",
        r.code==='eng'&&/English/.test(r.name)&&backend.test(r.backend)&&
        /^\d+ ms$/.test(r.ms)&&r.bars===6&&r.status==='Detected: English.',JSON.stringify(r));
      if(rung==="overview")mark("see-inside 10 actual ranked language scores/margin/entropy",
        r.inside&&r.rows===10&&Number.parseFloat(r.margin)>=0&&
        Number.isFinite(Number(r.entropy))&&Number(r.entropy)>=0&&Number(r.entropy)<=1,
        JSON.stringify(r));
    }
    const hygiene=await evaluate(sid,`({overflow:document.documentElement.scrollWidth-innerWidth,
      named:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim()),
      tedLoaded:performance.getEntriesByType('resource').some(x=>x.name.endsWith('/ted.wav'))})`);
    mark("responsive named controls, TED never loaded",hygiene.overflow<=1&&hygiene.named&&
      !hygiene.tedLoaded,JSON.stringify(hygiene));
    mark("console/network clean",page.errors.length===0&&page.netFailures.length===0,
      JSON.stringify({errors:page.errors,network:page.netFailures}));
    await verifyCredit(sid,label,mark);
  }catch(e){ok=false;check(`${label} completed`,false,String(e.stack||e).slice(0,500));}
  finally{results.push({route,viewport,pass:ok});if(page)await closePage(cdp,page.targetId).catch(()=>{});}
}
try{
  const started=await startServer();server=started.server;server.port=started.port;
  chrome=await launchChrome({userDataDir:PROFILE,resetProfile:true,removeProfileOnKill:false});
  cdp=new CDP(chrome.ws);
  for(const rung of Object.keys(ROUTES))for(const viewport of Object.keys(VIEWPORTS))await drive(rung,viewport);
}catch(e){console.error(`FATAL ${String(e.stack||e)}`);}
finally{
  try{if(chrome)await chrome.kill({removeProfile:false});}
  finally{try{if(server)await new Promise(resolve=>server.close(resolve));}
    finally{rmSync(PROFILE,{recursive:true,force:true});}}
}
const succeeded=results.length===10&&results.every(x=>x.pass)&&checks===70&&checks===passed;
console.log(`\n${passed}/${checks} checks passed across ${results.length}/10 route cells.`);
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if(WRITE_RUN&&succeeded){
  const commit=execFileSync("git",["log","-n1","--format=%H","HEAD","--",FAMILY,
    "scripts/validate-spoken-language-id.mjs",`:(exclude)${FAMILY}/acceptance.json`,
    `:(exclude)${FAMILY}/acceptance-run.json`],{cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD,JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport real 126-way LID cells: JFK English on all five; Wild starts non-JFK Spanish by design, then selects JFK and measures a 3s JFK English slice plus 6s English→Russian code-switch; Practical routes English to a suggested Whisper ASR/translation (not executed) plus native JFK user WAV upload reclassification; multi runs actual 974MB MMS-LID then timestamped Whisper q8 ~120MB (~1.09GB total) and checks real JFK transcript. Ten clicked visible JFK-only links reach pinned GitHub ledger hash/creator/public-domain/asset path. TED does not exist in family and is never loaded. Responsive/console/network assertions; twelve screenshots outside repo. Model may misrecognize audio; no perfect-transcript claim."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded?0:1);
