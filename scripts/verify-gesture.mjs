import { mkdirSync } from "node:fs";
import {
  CDP,
  closePage,
  launchChrome,
  MOBILE,
  openPage,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const BASE = "/web-ai-showcase/";
const outDir = new URL("../reports/gesture-retrofit/", import.meta.url);
mkdirSync(outDir, { recursive: true });
const shot = (n) => new URL(n, outDir).pathname;

async function ev(cdp, sid, expr, t = 120000) {
  const wrapped =
    `(async()=>{try{return (${expr});}catch(e){return {__err:String(e&&e.stack||e)};}})()`;
  const { result } = await cdp.send(
    "Runtime.evaluate",
    { expression: wrapped, awaitPromise: true, returnByValue: true },
    sid,
    t,
  );
  return result?.value;
}
async function setTheme(cdp, sid, theme) {
  await cdp.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: theme }],
  }, sid);
  await new Promise((r) => setTimeout(r, 250));
}
async function downloadAndReady(cdp, sid) {
  await ev(
    cdp,
    sid,
    `(()=>{const b=[...document.querySelectorAll('.loader-actions button')].find(x=>/download/i.test(x.textContent));if(b)b.click();return true;})()`,
  );
  return ev(
    cdp,
    sid,
    `await (async()=>{for(let i=0;i<400;i++){const r=document.getElementById('run');if(r&&!r.disabled)return true;const ldr=document.querySelector('.model-loader');if(ldr&&(ldr.dataset.state==='error'||ldr.dataset.state==='unsupported'))return {err:ldr.querySelector('.status').textContent};await new Promise(z=>setTimeout(z,300));}return {err:'timeout'};})()`,
    150000,
  );
}

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const results = {};

// ---- Desktop: main page, run recognize on the thumbs-up sample, verify correct gesture + console clean.
{
  const ctx = await openPage(cdp, `http://127.0.0.1:${port}${BASE}models/gesture-recognizer/`);
  const sid = ctx.sessionId;
  const ready = await downloadAndReady(cdp, sid);
  results.desktopReady = ready;
  // Run recognize (sample 1 is a thumbs-up).
  const out = await ev(
    cdp,
    sid,
    `await (async()=>{const btn=document.getElementById('run');btn.click();for(let i=0;i<300;i++){if(!btn.disabled&&!document.getElementById('readout').hidden)break;await new Promise(z=>setTimeout(z,20));}return {label:document.getElementById('gLabel').textContent,score:document.getElementById('gScore').textContent,delegate:document.getElementById('rDelegate').textContent,hands:document.getElementById('rHands').textContent,insideRows:document.querySelectorAll('#insideRows tr').length};})()`,
  );
  results.desktopRun = out;
  await setTheme(cdp, sid, "light");
  await screenshot(cdp, sid, shot("main-desktop-light.png"));
  await setTheme(cdp, sid, "dark");
  await screenshot(cdp, sid, shot("main-desktop-dark.png"));
  // No-camera fallback: click webcam (headless has no real camera) → expect a blocked message.
  const cam = await ev(
    cdp,
    sid,
    `await (async()=>{const b=document.getElementById('camBtn');b.click();for(let i=0;i<60;i++){const s=document.getElementById('camState').textContent;if(s&&/block|denied|error|not|allow/i.test(s))return s;await new Promise(z=>setTimeout(z,100));}return document.getElementById('camState').textContent||'(no message)';})()`,
    20000,
  );
  results.noCameraFallback = cam;
  results.desktopConsoleErrors = ctx.errors.slice(0, 5);
  results.desktopNetFailures = ctx.netFailures.filter((n) => !/favicon/i.test(n)).slice(0, 5);
  await closePage(cdp, ctx.targetId);
}

// ---- Mobile: layout + no horizontal overflow + screenshot.
{
  const ctx = await openPage(cdp, `http://127.0.0.1:${port}${BASE}models/gesture-recognizer/`);
  const sid = ctx.sessionId;
  await setViewport(cdp, sid, MOBILE);
  const overflow = await ev(
    cdp,
    sid,
    `(()=>({scrollW:document.documentElement.scrollWidth,clientW:document.documentElement.clientWidth,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+2}))()`,
  );
  results.mobileOverflow = overflow;
  await setTheme(cdp, sid, "light");
  await screenshot(cdp, sid, shot("main-mobile-light.png"));
  await setTheme(cdp, sid, "dark");
  await screenshot(cdp, sid, shot("main-mobile-dark.png"));
  results.mobileConsoleErrors = ctx.errors.slice(0, 5);
  await closePage(cdp, ctx.targetId);
}

// ---- A ladder page (wild) sanity: loads clean, worker path present.
{
  const ctx = await openPage(cdp, `http://127.0.0.1:${port}${BASE}models/gesture-recognizer/wild/`);
  results.wildConsoleErrors = ctx.errors.slice(0, 5);
  await screenshot(cdp, ctx.sessionId, shot("wild-desktop.png"));
  await closePage(cdp, ctx.targetId);
}

console.log(JSON.stringify(results, null, 2));
chrome.kill();
server.close();
