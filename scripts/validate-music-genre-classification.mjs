// Music genre classification end-to-end validation (real inference in headless Chrome). ~90 MB.
// Verifies: loads → Download → ready; the two procedural samples classify to the expected genre via the
// real pipeline (piano melody → classical, drum beat → hip-hop) and return a full 10-genre distribution;
// the deployed page's sample chips render a verdict + 10 confidence bars; no console errors; no overflow.
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
        `(async()=>{try{return (${expr});}catch(e){return{__err:String(e&&e.message||e).slice(0,200)};}})()`,
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/music-genre-classification/`,
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
  // wait until the picker is enabled (onReady)
  let ready = false;
  for (let i = 0; i < 70; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("pickBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (controls enabled)", ready);

  // CORRECTNESS via the engine on the two procedural samples.
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./genre.js");
      const eng = new M.GenreEngine(); await eng.load();
      const one = async (src) => {
        const { pcm } = await M.urlToMono16k(src);
        const r = await eng.classify(pcm);
        const sum = r.labels.reduce((a,b)=>a+b.score,0);
        return { top: r.labels[0].label, n: r.labels.length, sum: +sum.toFixed(2) };
      };
      return { jazz: await one("sample-jazz.wav"), hiphop: await one("sample-hiphop.wav") };
    })()`,
    160000,
  );
  chk(
    "chord progression → jazz (full 10-genre dist)",
    rec?.jazz?.top === "jazz" && rec?.jazz?.n === 10 && Math.abs(rec?.jazz?.sum - 1) < 0.05,
    JSON.stringify(rec?.jazz),
  );
  chk("drum beat → hiphop", rec?.hiphop?.top === "hiphop", JSON.stringify(rec?.hiphop));

  // Drive the deployed page: click a sample chip → verdict + 10 bars.
  await evalL(
    pg.sessionId,
    `document.querySelector('#samples .gen-chip[data-src="sample-jazz.wav"]').click()`,
    10000,
  );
  let bars = 0, verdict = "";
  for (let i = 0; i < 30; i++) {
    await sleep(800);
    bars = await evalL(pg.sessionId, `document.querySelectorAll("#bars .gen-row").length`, 8000) ||
      0;
    verdict = await evalL(pg.sessionId, `document.getElementById("verdict").textContent`, 8000) ||
      "";
    if (bars >= 10 && /Jazz/i.test(verdict)) break;
  }
  chk(
    "page sample → verdict + 10 confidence bars",
    bars === 10 && /Jazz/i.test(verdict),
    `bars=${bars} verdict=${verdict.trim().slice(0, 30)}`,
  );
  const readout = await evalL(pg.sessionId, `!document.getElementById("readout").hidden`, 8000);
  chk("readout (backend/latency/clip) shown", readout === true);

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
