// Speech-separation end-to-end validation (real inference in headless Chrome). ~20 MB ConvTasNet.
// Verifies: loads → Download → auto-mixes two sample speakers, separates into 2 tracks; each output track
// correlates with a DIFFERENT original speaker (a clean one-speaker-per-output split); no console errors;
// no overflow desktop+mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 240000) =>
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/speech-separation/`,
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
  // wait for the auto sample separation (status reports "Separated")
  let st = "";
  for (let i = 0; i < 60; i++) {
    st = await evalL(pg.sessionId, `document.getElementById("status").textContent`, 10000) || "";
    if (/Separated/.test(st)) break;
    await sleep(2500);
  }
  chk("ready → sample mix separated", /Separated into 2/.test(st), st);
  // both output audio players + waveforms present
  const ui = await evalL(
    pg.sessionId,
    `(()=>({out:!document.getElementById("sepOut").hidden, s1:!!document.getElementById("s1Audio").src, s2:!!document.getElementById("s2Audio").src}))()`,
    10000,
  );
  chk("two separated speaker tracks produced", ui?.out && ui?.s1 && ui?.s2, JSON.stringify(ui));
  // CORRECTNESS: re-run the model in-page on a fresh jfk+ted mix and check each output maps to a different speaker
  const corr = await evalL(
    pg.sessionId,
    `(async()=>{
    const M = await import("./separation.js");
    const A = "../whisper-speech-to-text/";
    const clip = async (u) => { const a = await M.decodeTo16kMono(await (await fetch(A+u)).arrayBuffer()); return a.subarray(0, M.SR*3); };
    const jfk = await clip("jfk.wav"), ted = await clip("ted.wav");
    const mix = M.mixClips(jfk, ted);
    const eng = new M.SeparationEngine(); await eng.load(); const { s1, s2 } = await eng.separate(mix);
    const corr = (x,y)=>{let sx=0,sy=0,sxy=0,sxx=0,syy=0,n=Math.min(x.length,y.length);for(let i=0;i<n;i++){sx+=x[i];sy+=y[i];}sx/=n;sy/=n;for(let i=0;i<n;i++){const dx=x[i]-sx,dy=y[i]-sy;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy;}return Math.abs(sxy/Math.sqrt(sxx*syy+1e-12));};
    return { s1jfk:corr(s1,jfk), s1ted:corr(s1,ted), s2jfk:corr(s2,jfk), s2ted:corr(s2,ted) };
  })()`,
    120000,
  );
  // each output should map to a different speaker: (s1→jfk & s2→ted) or (s1→ted & s2→jfk), each dominant
  const permA = corr && corr.s1jfk > 0.5 && corr.s2ted > 0.5 && corr.s1ted < 0.4 &&
    corr.s2jfk < 0.4;
  const permB = corr && corr.s1ted > 0.5 && corr.s2jfk > 0.5 && corr.s1jfk < 0.4 &&
    corr.s2ted < 0.4;
  chk("each output = a DIFFERENT speaker (clean separation)", permA || permB, JSON.stringify(corr));
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
