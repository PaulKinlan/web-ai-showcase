// QuickDraw sketch-recognition end-to-end validation (real inference in headless Chrome). ~21 MB.
// Verifies: loads → Download → ready; a programmatically-drawn ladder classifies as "ladder" and a star as
// "star" (real sketch recognition via draw.js); drawing on the pad with pointer events updates the guess;
// no console errors; no overflow desktop+mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 200000) =>
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/quickdraw-sketch-recognition/`,
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
  // wait for ready
  let g = "";
  for (let i = 0; i < 60; i++) {
    g = await evalL(pg.sessionId, `document.getElementById("guess").textContent`, 10000) || "";
    if (/Ready/.test(g)) break;
    await sleep(2000);
  }
  chk("ready", /Ready/.test(g), g);
  // CORRECTNESS: draw a ladder + a star programmatically, classify via draw.js
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
    const M = await import("./draw.js");
    const eng = new M.SketchEngine(); await eng.load();
    const drawShape = (fn) => {
      const c = document.createElement("canvas"); c.width = 280; c.height = 280;
      const x = c.getContext("2d"); x.fillStyle="#0b0f14"; x.fillRect(0,0,280,280);
      x.strokeStyle="#fff"; x.lineWidth=12; x.lineCap="round"; x.lineJoin="round"; fn(x); return c;
    };
    const ladder = drawShape((x)=>{ x.beginPath(); x.moveTo(90,30); x.lineTo(90,250); x.moveTo(190,30); x.lineTo(190,250); for(let y=55;y<250;y+=40){x.moveTo(90,y);x.lineTo(190,y);} x.stroke(); });
    const square = drawShape((x)=>{ x.strokeRect(55,55,170,170); });
    const l = await eng.classify(M.canvasToGray28(ladder), 3);
    const s = await eng.classify(M.canvasToGray28(square), 3);
    return { ladder: l.top[0].label, square: s.top[0].label, ladderTop3: l.top.map(t=>t.label), squareTop3: s.top.map(t=>t.label) };
  })()`,
    120000,
  );
  chk("drawn ladder → 'ladder'", rec && rec.ladder === "ladder", JSON.stringify(rec?.ladderTop3));
  chk("drawn square → 'square'", rec && rec.square === "square", JSON.stringify(rec?.squareTop3));
  // UI: draw a diagonal line on the pad with pointer events → guess updates
  await evalL(
    pg.sessionId,
    `(async()=>{
    const pad = document.getElementById("pad"); const r = pad.getBoundingClientRect();
    const ev = (type,x,y)=>pad.dispatchEvent(new PointerEvent(type,{pointerId:1,clientX:r.left+x,clientY:r.top+y,bubbles:true}));
    ev("pointerdown",40,240);
    for(let i=1;i<=8;i++){ ev("pointermove",40+i*24,240-i*24); }
    ev("pointerup",232,48);
    return true;
  })()`,
    15000,
  );
  let g2 = "";
  for (let i = 0; i < 20; i++) {
    await sleep(800);
    g2 = await evalL(pg.sessionId, `document.getElementById("guess").textContent`, 8000) || "";
    if (/I think it's/.test(g2)) break;
  }
  const bars = await evalL(pg.sessionId, `document.querySelectorAll("#bars .qd-bar").length`, 8000);
  chk(
    "drawing on the pad updates the guess + bars",
    /I think it's/.test(g2) && bars >= 3,
    `guess=${g2} bars=${bars}`,
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
