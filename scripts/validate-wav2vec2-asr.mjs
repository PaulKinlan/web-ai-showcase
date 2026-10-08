#!/usr/bin/env node
// W7: five wav2vec2 CTC routes × desktop/mobile; real JFK ASR, aligned words,
// frame scrubbing and dual-stage Qwen punctuation. TED is never selected.
// Primary Xenova/wav2vec2-base-960h q8 95MB + multi-model
// onnx-community/Qwen2.5-0.5B-Instruct q4 400MB
// (~495MB TOTAL weights, not 95MB overall). All FIVE rungs bundle the JFK clip.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CDP, closePage, DESKTOP, launchChrome, MOBILE, openPage, repoRoot,
  screenshot, setViewport, startServer } from "./browser.mjs";
import { evaluateCleanup } from "../models/wav2vec2-asr/multi-model/cleanup-quality.mjs";

const WRITE_RUN=process.argv.includes("--write-run");
const FAMILY="models/wav2vec2-asr";
const RECORD=join(repoRoot,FAMILY,"acceptance-run.json");
const PROFILE=mkdtempSync(join(tmpdir(),"wav2vec2-acceptance-"));
const EVIDENCE=process.env.WAV2VEC2_CREDIT_EVIDENCE_DIR||join(tmpdir(),"7nr-w7-wav2vec2-credit");
const CREDIT_URL="https://github.com/PaulKinlan/web-ai-showcase/blob/ab33435d6c4a50ca5d02184e8445b61998eb88fc/audio-provenance/ledger.json";
const ROUTES={overview:"models/wav2vec2-asr/",basics:"models/wav2vec2-asr/basics/",
  practical:"models/wav2vec2-asr/practical/",wild:"models/wav2vec2-asr/wild/",
  multi:"models/wav2vec2-asr/multi-model/"};
const VIEWPORTS={desktop:DESKTOP,mobile:MOBILE};
mkdirSync(EVIDENCE,{recursive:true});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const backend=/^(WASM|WEBGPU)$/, speech=/fellow|american|country|ask/i;
let checks=0,passed=0,server,chrome,cdp;
const results=[];
function check(label,condition,detail=""){
  checks++;if(condition)passed++;
  console.log(`${condition?"PASS":"FAIL"}  ${label}${detail?` — ${String(detail).slice(0,300)}`:""}`);
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
    if(Date.now()>=nextLog){
      console.log(`  [${label}] waiting ${Math.round((Date.now()-started)/1000)}s`);
      nextLog=Date.now()+10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}
async function ensureReady(sid,label,rung){
  const started=Date.now();let nextLog=0;
  while(Date.now()-started<12*60_000){
    const info=await evaluate(sid,`(() => {
      const loaders=[...document.querySelectorAll('.model-loader')];
      return {states:loaders.map(x=>x.dataset.state),checks:loaders.map(x=>Number(x.dataset.localCheckMs)),
        statuses:loaders.map(x=>x.querySelector('.status')?.textContent?.trim())};
    })()`);
    if(Date.now()>=nextLog){console.log(`  [${label}] ${Math.round((Date.now()-started)/1000)}s ${JSON.stringify(info)}`);nextLog=Date.now()+10_000;}
    if(info.states.length===(rung==="multi"?2:1)&&info.states.every(x=>x==="ready"))return info;
    await evaluate(sid,`(() => {const b=[...document.querySelectorAll('.model-loader button')]
      .filter(x=>/Download|Retry|Re-download|Continue/i.test(x.textContent)&&!x.disabled);
      b.forEach(x=>x.click());return b.length;})()`);
    await sleep(1_500);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model preparation`);
}
async function verifyCredit(sid,label,mark){
  const credit=await evaluate(sid,`(() => {const a=[...document.querySelectorAll('a')]
    .find(x=>x.textContent.trim()==='JFK audio source and attribution record');
    if(!a)return null;a.scrollIntoView({block:'center'});
    return {href:a.href,visible:!!a.getClientRects().length,context:a.parentElement.textContent.trim()};})()`);
  await screenshot(cdp,sid,join(EVIDENCE,`${label.replace('@','-')}.png`));
  const visible=credit?.visible&&credit.href.startsWith(CREDIT_URL)&&
    /John F\. Kennedy/.test(credit.context)&&/public domain/.test(credit.context)&&
    /bundled JFK speech clip alone/.test(credit.context);
  let responseStatus=0;
  cdp.on(msg=>{
    if(msg.sessionId===sid&&msg.method==="Network.responseReceived"&&
      msg.params.type==="Document"&&msg.params.response.url.startsWith(CREDIT_URL))
      responseStatus=msg.params.response.status;
  });
  if(visible){
    await evaluate(sid,`(() => {const a=[...document.querySelectorAll('a')]
      .find(x=>x.textContent.trim()==='JFK audio source and attribution record');
      setTimeout(()=>a.click(),0);return true;})()`);
    await waitFor(sid,`location.href.startsWith(${JSON.stringify(CREDIT_URL)})`,30_000,`${label} credit click`,500);
    await waitFor(sid,`document.body?.innerText.includes('627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b')`,
      30_000,`${label} credit destination content`,500);
  }
  const content=visible&&await evaluate(sid,`(() => {const body=document.body?.innerText||'';
    return ['627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b',
      'John F. Kennedy','Public domain','models/wav2vec2-asr/jfk.wav']
      .map(text=>body.includes(text));})()`);
  mark("visible JFK link clicked through to CONTENT-verified pinned ledger",
    visible&&responseStatus>=200&&responseStatus<400&&content?.every(Boolean),
    JSON.stringify({credit,responseStatus,content}));
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
    const ready=await ensureReady(sid,label,rung);
    mark("bounded local cache checks",ready.checks.length===(rung==="multi"?2:1)&&
      ready.checks.every(x=>Number.isFinite(x)&&x<=450),JSON.stringify(ready.checks));
    const selected=await evaluate(sid,`({src:document.querySelector('#player')?.getAttribute('src'),
      label:document.querySelector('#clipLabel')?.textContent})`);
    mark("JFK is default source; TED was never selected",rung==="wild"?
      /JFK/.test(selected.label)&&await evaluate(sid,`document.querySelector('#samples button[data-src="../jfk.wav"]')!==null`):
      /(?:^|\/)jfk\.wav$/.test(selected.src)&&/JFK/.test(selected.label),JSON.stringify(selected));
    await waitFor(sid,`!document.querySelector('#run')?.disabled`,45_000,`${label} JFK decoded`);
    await evaluate(sid,`(() => {document.querySelector('#run').click();return true;})()`);
    if(rung==="multi"){
      await waitFor(sid,`!document.querySelector('#readout')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Done.','Done (basic fallback; Qwen cleanup failed).','Failed:']
          .some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        600_000,`${label} real CTC+Qwen 495MB chain`);
      const r=await evaluate(sid,`({raw:document.querySelector('#raw')?.textContent,
        clean:document.querySelector('#clean')?.textContent,
        asr:document.querySelector('#rAsr')?.textContent,
        asrMs:document.querySelector('#rAsrMs')?.textContent,
        llm:document.querySelector('#rLlm')?.textContent,
        llmMs:document.querySelector('#rLlmMs')?.textContent,
        fallback:!document.querySelector('#cleanupNote')?.hidden,
        modelText:document.querySelector('#qwenRaw')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real wav2vec2 CTC decoded JFK text before Qwen cleanup",
        speech.test(r.raw)&&backend.test(r.asr)&&/^\d+(?:\.\d+)? s$/.test(r.asrMs),JSON.stringify(r));
      const quality=evaluateCleanup(r.raw,r.clean);
      mark("Qwen q4 400MB truly restores punctuation/casing WITHOUT changing CTC words",
        backend.test(r.llm)&&/^\d+(?:\.\d+)? s$/.test(r.llmMs)&&
        !r.fallback&&quality.valid&&r.status==='Done.',JSON.stringify({...r,quality}));
      await evaluate(sid,`(() => {document.querySelector('#clean').scrollIntoView({block:'center'});return true;})()`);
      await screenshot(cdp,sid,join(EVIDENCE,`${viewport}-multi-clean.png`));
    }else if(rung==="wild"){
      await waitFor(sid,`!document.querySelector('#scrubWrap')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Done — scrub the frames.','Failed:'].some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        180_000,`${label} real CTC frame strip`);
      const r=await evaluate(sid,`({text:document.querySelector('#transcript')?.textContent,
        total:Number(document.querySelector('#fTot')?.textContent),
        idx:Number(document.querySelector('#fIdx')?.textContent),
        strip:document.querySelectorAll('#context .ctc-cell').length,
        status:document.querySelector('#status')?.textContent})`);
      mark("real JFK CTC transcription exposes per-frame predictions",
        speech.test(r.text)&&r.total>100&&r.strip>0&&r.status==='Done — scrub the frames.',JSON.stringify(r));
      const scrub=await evaluate(sid,`(() => {const el=document.querySelector('#scrub');
        const target=Math.min(Number(el.max),Number(el.value)+83);
        el.value=String(target);el.dispatchEvent(new Event('input',{bubbles:true}));
        return {target,index:Number(document.querySelector('#fIdx')?.textContent),
          time:Number(document.querySelector('#fTime')?.textContent),
          character:document.querySelector('#bigChar')?.textContent,
          focused:document.querySelectorAll('#context .ctc-cell[style*="outline"]').length};})()`);
      mark("dragging frame scrub updates real character/time/context",
        scrub.target!==r.idx&&scrub.index===scrub.target&&scrub.time>0&&
        !!scrub.character&&scrub.focused===1,JSON.stringify(scrub));
    }else if(rung==="practical"){
      await waitFor(sid,`!document.querySelector('#tableWrap')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Done — click a word.','No words detected.','Failed:'].some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        180_000,`${label} JFK CTC forced alignment`);
      const r=await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
        words:Number(document.querySelector('#rWords')?.textContent),
        frames:Number(document.querySelector('#rFrames')?.textContent),
        frameMs:document.querySelector('#rFrame')?.textContent,
        rows:[...document.querySelectorAll('#rows tr')].map(x=>[...x.cells].map(y=>y.textContent)),
        buttons:document.querySelectorAll('#words button.align-word').length,
        status:document.querySelector('#status')?.textContent})`);
      mark("real JFK CTC alignment measures positive words/frames and backend",
        backend.test(r.backend)&&r.words>=8&&r.frames>100&&/^\d+ ms$/.test(r.frameMs)&&
        r.status==='Done — click a word.',JSON.stringify(r).slice(0,600));
      mark("every rendered clickable word has a populated timing table row",
        r.rows.length===r.words&&r.buttons===r.words&&
        r.rows.every(x=>x.length===3&&!!x[0]?.trim()&&x[1]!==x[2]),
        JSON.stringify({words:r.words,first:r.rows.slice(0,3)}));
      const target=await evaluate(sid,`(() => {
        const buttons=[...document.querySelectorAll('#words button.align-word')];
        const row=Math.min(4,buttons.length-1),button=buttons[row];
        const raw=document.querySelector('#rows tr:nth-child('+(row+1)+') td:nth-child(2)')?.textContent||'';
        const [minutes,seconds]=raw.split(':');
        const start=seconds!=null?Number(minutes)*60+Number(seconds):Number.parseFloat(raw);
        const player=document.querySelector('#player');player.pause();player.currentTime=0;
        button.scrollIntoView({block:'center'});
        const rect=button.getBoundingClientRect();
        return {start,x:rect.x+rect.width/2,y:rect.y+rect.height/2,word:button.textContent};})()`);
      await cdp.send('Input.dispatchMouseEvent',
        {type:'mousePressed',x:target.x,y:target.y,button:'left',clickCount:1},sid);
      await cdp.send('Input.dispatchMouseEvent',
        {type:'mouseReleased',x:target.x,y:target.y,button:'left',clickCount:1},sid);
      const seek={...target,now:await evaluate(sid,`document.querySelector('#player')?.currentTime`)};
      mark("trusted click on JFK word seeks real audio to measured start",
        seek.start>0.1&&Math.abs(seek.now-seek.start)<0.25,JSON.stringify(seek));
    }else{
      await waitFor(sid,`!document.querySelector('#readout')?.hidden&&
        !document.querySelector('#run')?.disabled&&
        ['Done.','Transcription failed:','Failed:'].some(x=>document.querySelector('#status')?.textContent?.startsWith(x))`,
        180_000,`${label} real CTC ASR`);
      const r=await evaluate(sid,`({backend:document.querySelector('#rBackend')?.textContent,
        ms:document.querySelector('#rMs')?.textContent,
        frames:Number(document.querySelector('#rFrames')?.textContent),
        text:document.querySelector('#transcript')?.textContent,
        inside:!document.querySelector('#insideWrap')?.hidden,
        strip:document.querySelectorAll('#strip .ctc-cell').length,
        collapsed:document.querySelector('#collapsed')?.textContent,
        status:document.querySelector('#status')?.textContent})`);
      mark("real wav2vec2 ASR transcribes JFK with measured CTC frames",
        backend.test(r.backend)&&/^\d+(?:\.\d+)? s$/.test(r.ms)&&
        r.frames>100&&speech.test(r.text)&&r.status==='Done.',JSON.stringify(r));
      if(rung==="overview")mark("see-inside real per-frame CTC collapse shown",
        r.inside&&r.strip===r.frames&&r.collapsed===r.text,
        JSON.stringify({frames:r.frames,strip:r.strip,collapsed:r.collapsed}));
    }
    const hygiene=await evaluate(sid,`({overflow:document.documentElement.scrollWidth-innerWidth,
      named:[...document.querySelectorAll('button')].every(b=>(b.textContent||b.getAttribute('aria-label')||'').trim()),
      tedLoaded:performance.getEntriesByType('resource').some(x=>x.name.endsWith('/ted.wav'))})`);
    mark("responsive named controls, TED not loaded",hygiene.overflow<=1&&hygiene.named&&
      !hygiene.tedLoaded,JSON.stringify(hygiene));
    mark("console/network clean",page.errors.length===0&&page.netFailures.length===0,
      JSON.stringify({errors:page.errors,network:page.netFailures}));
    await verifyCredit(sid,label,mark);
  }catch(e){ok=false;check(`${label} completed`,false,String(e.stack||e).slice(0,400));}
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
    "scripts/validate-wav2vec2-asr.mjs",`:(exclude)${FAMILY}/acceptance.json`,
    `:(exclude)${FAMILY}/acceptance-run.json`],{cwd:repoRoot,encoding:"utf8"}).trim();
  writeFileSync(RECORD,JSON.stringify({commit,ranAt:new Date().toISOString(),exitCode:0,
    results,notes:"Ten route×viewport cells; real JFK wav2vec2 CTC transcripts and per-frame collapse, Practical per-word timing and clickable audio seek, Wild frame scrub, multi BOTH 95MB Wav2Vec2 and 400MB q4 Qwen second stage (~495MB total) with actual JFK raw transcript and generated punctuation cleanup (factual accuracy not evaluated). Five JFK-only bundled sample credits per viewport click through to pinned GitHub ledger hash/creator/public-domain/path; TED never selected. Responsive/console/network checks; screenshots outside repo."},null,2)+"\n");
  console.log(`WROTE ${RECORD} for ${commit}`);
}
process.exit(succeeded?0:1);
