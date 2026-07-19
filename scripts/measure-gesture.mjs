#!/usr/bin/env node
// Ad-hoc perf measurement for the gesture-recognizer retrofit.
// Loads the page, triggers the model Download/init (fresh profile ⇒ loader shows Download),
// waits for ready, installs a longtask PerformanceObserver, then runs recognize on the bundled
// sample and reports the longest main-thread long task attributable to the recognize.
// Usage: node scripts/measure-gesture.mjs [burst]
import { CDP, closePage, launchChrome, openPage, startServer } from "./browser.mjs";

const BASE = "/web-ai-showcase/";
const PATH = "models/gesture-recognizer/";
const burst = Number(process.argv[2] || 1);

async function ev(cdp, sessionId, expr, timeoutMs = 120000) {
  const wrapped =
    `(async()=>{try{return (${expr});}catch(e){return {__err:String(e&&e.message||e)};}})()`;
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: wrapped,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const url = `http://127.0.0.1:${port}${BASE}${PATH}`;
console.log("open", url);
const ctx = await openPage(cdp, url);
const { sessionId } = ctx;

// Install a longtask observer early so we can zero it right before the run.
await ev(
  cdp,
  sessionId,
  `(()=>{window.__lt=[];try{new PerformanceObserver(l=>{for(const e of l.getEntries())window.__lt.push(Math.round(e.duration))}).observe({entryTypes:['longtask']});}catch(e){}return true;})()`,
);

// Trigger download/init: the loader shows a Download button on a fresh profile. Click it.
const clicked = await ev(
  cdp,
  sessionId,
  `(()=>{const b=[...document.querySelectorAll('.loader-actions button')].find(x=>/download/i.test(x.textContent));if(b){b.click();return b.textContent.trim();}return 'no-download-button(auto-init)';})()`,
);
console.log("loader:", clicked);

// Wait for the model to be ready — the run button becomes enabled (rec ready + sample loaded).
const ready = await ev(
  cdp,
  sessionId,
  `await (async()=>{for(let i=0;i<600;i++){const r=document.getElementById('run');const st=document.getElementById('status');if(r&&!r.disabled)return {ready:true,status:st&&st.textContent};if(st&&/failed|error|blocked/i.test(st.textContent||''))return {ready:false,status:st.textContent};await new Promise(z=>setTimeout(z,250));}return {ready:false,status:(document.getElementById('status')||{}).textContent};})()`,
  180000,
);
console.log("ready:", JSON.stringify(ready));
if (!ready?.ready) {
  console.log("MODEL NOT READY — aborting measurement");
  await closePage(cdp, ctx.targetId);
  chrome.kill();
  server.close();
  process.exit(2);
}

// Warm-up single run (WASM graph warmup), then measure.
const runOnce = async () => {
  await ev(cdp, sessionId, `(()=>{window.__lt=[];return true;})()`);
  const t = await ev(
    cdp,
    sessionId,
    `await (async()=>{const btn=document.getElementById('run');const t0=performance.now();btn.click();
      // wait until the readout shows a fresh latency (run re-enables the button when done)
      for(let i=0;i<400;i++){if(!btn.disabled && !document.getElementById('readout').hidden)break;await new Promise(z=>setTimeout(z,20));}
      await new Promise(z=>setTimeout(z,120));
      const ms=document.getElementById('rMs')?document.getElementById('rMs').textContent:'?';
      const maxLt=window.__lt.length?Math.max(...window.__lt):0;
      return {wallMs:Math.round(performance.now()-t0),reportedLatency:ms,maxLongTaskMs:maxLt,longTasks:window.__lt.slice(0,10)};})()`,
    60000,
  );
  return t;
};

console.log("warmup:", JSON.stringify(await runOnce()));
const samples = [];
for (let i = 0; i < burst; i++) samples.push(await runOnce());
for (const s of samples) console.log("RUN:", JSON.stringify(s));
const maxLts = samples.map((s) => s.maxLongTaskMs);
console.log(
  "SUMMARY maxLongTask over",
  burst,
  "runs → min",
  Math.min(...maxLts),
  "max",
  Math.max(...maxLts),
  "| console errors:",
  ctx.errors.length,
);
if (ctx.errors.length) console.log("ERRORS:", ctx.errors.slice(0, 5));

await closePage(cdp, ctx.targetId);
chrome.kill();
server.close();
