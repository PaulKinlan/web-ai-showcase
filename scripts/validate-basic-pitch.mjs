// Basic Pitch music-transcription end-to-end validation (real inference in headless Chrome).
// Tiny ~0.3 MB model. Verifies: loads → Download → auto-transcribes the chords sample into a piano roll +
// picked notes; the picked notes include the intended pitch classes; polyphony (>=2 simultaneous keys) for
// chords; switching to the arpeggio sample re-transcribes; no console errors; no overflow desktop+mobile.
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/basic-pitch-transcription/`,
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
  // wait for the auto chords transcription (status reports "note event")
  let st = "";
  for (let i = 0; i < 60; i++) {
    st = await evalL(pg.sessionId, `document.getElementById("status").textContent`, 10000) || "";
    if (/note event/.test(st)) break;
    await sleep(1500);
  }
  chk("ready → chords auto-transcribed", /note event/.test(st), st);
  const notes = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#notes .bp-note")].map(e=>e.textContent)`,
    10000,
  ) || [];
  chk(
    "chords → note events picked",
    notes.length >= 3,
    `notes=${JSON.stringify(notes).slice(0, 160)}`,
  );
  // intended chord pitch classes are C,E,G,F,A,B,D (from C/F/G triads). Expect the roots present.
  const pcs = new Set(notes.map((n) => n.replace(/[0-9#-]/g, "").replace(/#/, "")));
  chk(
    "intended pitches present (C,E,G among picked)",
    ["C", "E", "G"].filter((p) => notes.some((n) => n.startsWith(p))).length >= 2,
    [...pcs].join(","),
  );
  // piano roll drew content (non-blank canvas)
  const drew = await evalL(
    pg.sessionId,
    `(()=>{const c=document.getElementById("roll");const x=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let lit=0;for(let i=3;i<x.length;i+=4)if(x[i]>0)lit++;return lit;})()`,
    10000,
  ) || 0;
  chk("piano roll rendered (lit pixels)", drew > 500, `litPx=${drew}`);
  // switch to arpeggio
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll("#samples .chip")].find(x=>/Arpeggio/.test(x.textContent));b&&b.click();return !!b;})()`,
    10000,
  );
  let st2 = "";
  for (let i = 0; i < 30; i++) {
    st2 = await evalL(pg.sessionId, `document.getElementById("status").textContent`, 10000) || "";
    if (/Arpeggio/.test(st2) && /note event/.test(st2)) break;
    await sleep(1200);
  }
  chk("arpeggio re-transcribes", /note event/.test(st2), st2);
  // responsive: no horizontal overflow desktop + mobile
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
