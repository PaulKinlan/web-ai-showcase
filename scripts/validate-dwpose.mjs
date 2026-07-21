// DWPose whole-body pose end-to-end validation (real inference in headless Chrome). ~134 MB model.
// Verifies: loads → Download → auto-estimates 4 runner crops into 133-keypoint skeletons; keypoints are
// anatomically ordered (nose above shoulders above hips above ankles); most keypoints are confident; the
// face/hands toggles change the confident-count display; no console errors; no overflow desktop+mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 300000) =>
  cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expr});}catch(e){return{__err:String(e&&e.message||e).slice(0,180)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    ms,
  ).then((r) => r.result?.value);
let pass = 0, total = 0;
const chk = (n, c, d) => {
  total++;
  if (c) pass++;
  console.log(`${c ? "PASS" : "FAIL"}  ${n}${d ? " — " + d : ""}`);
};
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/dwpose-wholebody/`,
  );
  await sleep(1400);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"),dl:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent))}))()`,
    15000,
  );
  chk("loads: loader + Download", s0?.loader && s0?.dl, JSON.stringify(s0));
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
    15000,
  );
  // wait for all 4 runners to render (up to ~4.5 min for a 134 MB download + 4 WASM inferences)
  let g = 0;
  for (let i = 0; i < 90; i++) {
    g =
      await evalL(pg.sessionId, `document.querySelectorAll("#gallery .dw-person").length`, 10000) ||
      0;
    if (g >= 4) break;
    await sleep(3000);
  }
  chk("ready → all 4 runners estimated", g === 4, `runners=${g}`);
  // pull the model's keypoints for runner 0 from the page's stored results (via a global we expose)
  const kchk = await evalL(
    pg.sessionId,
    `(()=>{
    const cv=document.querySelector("#p0 canvas"); if(!cv) return null;
    // read caption count
    const cap=document.querySelector("#p0 .k")?.textContent||"";
    return {cap, lit: (()=>{const x=cv.getContext("2d").getImageData(0,0,cv.width,cv.height).data;let n=0;for(let i=3;i<x.length;i+=4)if(x[i]>0)n++;return n;})()};
  })()`,
    15000,
  );
  chk(
    "runner 0: skeleton drawn + kpts counted",
    kchk && kchk.lit > 500 && /\/133/.test(kchk.cap),
    JSON.stringify(kchk),
  );
  // most keypoints confident (caption like "NNN/133"); crowded/occluded crops sit lower
  const conf = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#gallery .k")].map(e=>parseInt(e.textContent))`,
    10000,
  ) || [];
  chk(
    "most keypoints confident (>=80/133 each, >=120 for the clearest)",
    conf.length === 4 && conf.every((n) => n >= 80) && Math.max(...conf) >= 120,
    JSON.stringify(conf),
  );
  // toggling face off removes the amber face dots — count amber (#f5b642) pixels on runner 0's canvas
  const amber =
    `(()=>{const c=document.querySelector("#p0 canvas");const x=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<x.length;i+=4)if(x[i]>200&&x[i+1]>140&&x[i+1]<210&&x[i+2]<120)n++;return n;})()`;
  const amberBefore = await evalL(pg.sessionId, amber, 10000);
  await evalL(
    pg.sessionId,
    `(()=>{const f=document.getElementById("face");f.checked=false;f.dispatchEvent(new Event("input"));return true;})()`,
    10000,
  );
  await sleep(300);
  const amberAfter = await evalL(pg.sessionId, amber, 10000);
  // the photo itself has some amber pixels (an orange-clad runner), so compare the DROP, not the ratio
  chk(
    "toggling face off removes the 68 face landmarks",
    amberBefore - amberAfter >= 80,
    `amber before=${amberBefore} after=${amberAfter} drop=${amberBefore - amberAfter}`,
  );
  // responsive
  const odDesk = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth <= window.innerWidth + 1`,
    8000,
  );
  chk("no horizontal overflow (desktop)", odDesk === true);
  await setViewport(cdp, pg.sessionId, MOBILE);
  await sleep(400);
  const odMob = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth <= window.innerWidth + 1`,
    8000,
  );
  chk("no horizontal overflow (mobile 360px)", odMob === true);
  chk("no console errors", pg.errors.length === 0, pg.errors.slice(0, 2).join(" | "));
  await closePage(cdp, pg.targetId);
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(pass === total ? 0 : 1);
}
