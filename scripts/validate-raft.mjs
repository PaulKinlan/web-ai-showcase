#!/usr/bin/env node
// End-to-end validation for the RAFT optical-flow demo (real inference in headless Chrome). Downloads the
// 64 MB fp32 model, auto-inits, applies a KNOWN pan to a licensed photo, and asserts RAFT recovers it —
// the model's honesty check. Also checks no console errors and no horizontal overflow (desktop + mobile).
// Heavy (~40 s: 64 MB download + ~6 s WASM inference); run deliberately, not on every push.
import { closePage, launchChrome, openPage, setViewport, startServer } from "./browser.mjs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { server, port } = await startServer();
const { CDP } = await import("./browser.mjs");
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const results = [];
const rec = (
  n,
  p,
  d,
) => (results.push(!!p), console.log(`${p ? "PASS" : "FAIL"}  ${n}${d ? " — " + d : ""}`));
// evalValue caps at 15 s; use cdp.send directly for the long download+inference evals.
const evalL = (sid, expr, ms = 280000) =>
  cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expr});}catch(e){return {__err:String(e&&e.message||e).slice(0,200)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    ms,
  ).then((r) => r.result?.value);

try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/raft-optical-flow/`,
  );
  await sleep(1500);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"), download:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent)), samples:document.querySelectorAll("#samples img").length}))()`,
    15000,
  );
  rec(
    "loads: shared loader + Download affordance (auto-init: absent → Download) + samples",
    s0?.loader && s0?.download && s0?.samples >= 3,
    JSON.stringify(s0),
  );

  // click Download → 64 MB fetch + session create
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
    15000,
  );
  let ready = false;
  for (let i = 0; i < 70 && !ready; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("run").disabled`, 10000);
    if (!ready) await sleep(2000);
  }
  rec("downloads + auto-inits to ready (run enabled)", ready);
  if (!ready) throw new Error("model never became ready");

  // known 14 px right pan → RAFT must recover u≈14, v≈0
  await evalL(
    pg.sessionId,
    `(()=>{document.getElementById("dir").value="1,0";document.getElementById("amt").value="14";document.getElementById("run").click();return true;})()`,
    15000,
  );
  let readout = "";
  for (let i = 0; i < 40 && !/recovered/.test(readout); i++) {
    readout = await evalL(pg.sessionId, `document.getElementById("readout").textContent`, 10000) ||
      "";
    if (!/recovered/.test(readout)) await sleep(1500);
  }
  const m = readout.match(/recovered \(centre\): u=([\-0-9.]+), v=([\-0-9.]+)/);
  const okMotion = m && Math.abs(parseFloat(m[1]) - 14) < 5 && Math.abs(parseFloat(m[2])) < 5;
  rec("REAL inference recovers a known 14 px pan (u≈14, v≈0)", okMotion, readout.slice(0, 110));

  const flowRendered = await evalL(
    pg.sessionId,
    `(()=>{const c=document.getElementById("cf");const d=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let nz=0;for(let i=0;i<d.length;i+=4)if(d[i]||d[i+1]||d[i+2])nz++;return nz>1000;})()`,
    15000,
  );
  rec("flow field colour-rendered to canvas", flowRendered === true);
  rec("no console errors", pg.errors.length === 0, pg.errors.slice(0, 2).join(" | "));

  const ovD = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
    10000,
  );
  rec("no horizontal overflow (desktop)", ovD <= 2, `overflow=${ovD}`);
  await setViewport(cdp, pg.sessionId, {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await sleep(400);
  const ovM = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
    10000,
  );
  rec("no horizontal overflow (mobile 390px)", ovM <= 2, `overflow=${ovM}`);
  await closePage(cdp, pg.targetId);
} finally {
  chrome.kill();
  server.close();
}
const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} RAFT checks passed.`);
process.exit(passed === results.length ? 0 : 1);
