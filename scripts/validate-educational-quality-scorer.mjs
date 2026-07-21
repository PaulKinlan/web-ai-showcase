// Educational-quality scorer end-to-end validation (real inference in headless Chrome). ~110 MB.
// Verifies: loads → Download → ready; the real FineWeb-Edu regression model scores educational passages
// (science / history / code tutorial) markedly higher than casual chat / promotional spam; the deployed
// page renders the 0-5 gauge + readout for the pre-filled sample and reacts to the sample chips; no console
// errors; no overflow desktop + mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 150000) =>
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/educational-quality-scorer/`,
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
  let ready = false;
  for (let i = 0; i < 70; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("text").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (editor enabled)", ready);

  // CORRECTNESS via the engine on the sample passages.
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./edu.js");
      const eng = new M.EduEngine(); await eng.load();
      const s = {};
      for (const k of ["science","history","tutorial","casual","promo"]) s[k] = (await eng.score(M.SAMPLES[k])).raw;
      return s;
    })()`,
    140000,
  );
  const eduMin = Math.min(rec?.science, rec?.history, rec?.tutorial);
  const junkMax = Math.max(rec?.casual, rec?.promo);
  chk(
    "educational passages score high (>=2.5)",
    rec && rec.science >= 2.5 && rec.history >= 2.5 && rec.tutorial >= 2.0,
    JSON.stringify(rec),
  );
  chk(
    "casual chat + spam score low (<1.5)",
    rec && rec.casual < 1.5 && rec.promo < 1.5,
    `casual=${rec?.casual} promo=${rec?.promo}`,
  );
  chk(
    "clear separation (edu min > junk max)",
    eduMin > junkMax + 1,
    `eduMin=${eduMin} junkMax=${junkMax}`,
  );

  // Drive the page: pre-filled science → gauge + readout.
  let num = "";
  for (let i = 0; i < 30; i++) {
    await sleep(800);
    num = await evalL(
      pg.sessionId,
      `(document.querySelector("#out .edu-num")||{}).textContent||""`,
      8000,
    ) || "";
    if (num) break;
  }
  const readout = await evalL(pg.sessionId, `!document.getElementById("readout").hidden`, 8000);
  chk(
    "page renders 0-5 gauge + readout for pre-filled science",
    parseFloat(num) >= 2.5 && readout === true,
    `num=${num}`,
  );

  // click the spam chip → score drops
  await evalL(
    pg.sessionId,
    `document.querySelector('#chips .edu-chip[data-key="promo"]').click()`,
    8000,
  );
  let promoNum = "";
  for (let i = 0; i < 25; i++) {
    await sleep(800);
    promoNum = await evalL(
      pg.sessionId,
      `(document.querySelector("#out .edu-num")||{}).textContent||""`,
      8000,
    ) || "";
    if (promoNum && parseFloat(promoNum) < 1.5) break;
  }
  chk("spam sample chip drops the score (<1.5)", parseFloat(promoNum) < 1.5, `num=${promoNum}`);

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
