// Protein mutation oracle end-to-end validation (real inference in headless Chrome). ~35 MB.
// Verifies: loads → Download → ready; ESM-2 masked-LM reconstructs the wild-type residue at hemoglobin β6
// (E ranks top) and scores the sickle-cell mutation E6V as strongly negative (and more damaging than the
// conservative E6D); the deployed page auto-analyses the sickle-cell position and renders the prediction +
// mutation bars + readout, and reacts to clicking another residue; no console errors; no overflow.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(B);
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/protein-mutation-oracle/`,
  );
  await sleep(1400);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"),dl:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent)),grid:document.querySelectorAll("#grid .pg-res").length}))()`,
    15000,
  );
  chk("loads: loader + Download + residue grid", s0?.loader && s0?.dl && s0?.grid > 100, JSON.stringify(s0));
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
    15000,
  );
  let ready = false;
  for (let i = 0; i < 70; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("analyseBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (Analyse enabled)", ready);

  // CORRECTNESS via the engine: hemoglobin β6 (index 5).
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./protein.js");
      const eng = new M.ProteinEngine(); await eng.load();
      const seq = M.SAMPLES.hbb.seq;
      const r = await eng.scan(seq, 5);           // β6, the sickle-cell position
      const ranked = Object.keys(r.probs).sort((a,b)=>r.probs[b]-r.probs[a]);
      const rows = M.mutationRatios(r.probs, r.wtAA);
      const llr = (aa)=> rows.find(x=>x.aa===aa).llr;
      return { wtAA:r.wtAA, wtRank:ranked.indexOf(r.wtAA), llrV:+llr("V").toFixed(3), llrD:+llr("D").toFixed(3), n:ranked.length };
    })()`,
    150000,
  );
  chk("wild-type is E and reconstructed (rank ≤ 2)", rec && rec.wtAA === "E" && rec.wtRank <= 2, JSON.stringify(rec));
  chk("sickle-cell E6V scores damaging (LLR < −1)", rec && rec.llrV < -1, `llrV=${rec?.llrV}`);
  chk("E6V more damaging than conservative E6D", rec && rec.llrV < rec.llrD, `V=${rec?.llrV} D=${rec?.llrD}`);
  chk("distribution over all 20 amino acids", rec && rec.n === 20, `n=${rec?.n}`);

  // Page auto-analysed the sickle-cell position → prediction bars + mutation bars + readout.
  let preds = 0, muts = 0, rpos = "";
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    preds = await evalL(pg.sessionId, `document.querySelectorAll("#preds .pg-bar-row").length`, 8000) || 0;
    muts = await evalL(pg.sessionId, `document.querySelectorAll("#muts .pg-mut-row").length`, 8000) || 0;
    rpos = await evalL(pg.sessionId, `document.getElementById("rPos").textContent||""`, 8000) || "";
    if (preds >= 1 && muts >= 1) break;
  }
  chk("page renders predicted-residue bars", preds >= 5, `rows=${preds}`);
  chk("page renders 20 mutation-effect bars", muts === 20, `rows=${muts}`);
  chk("readout shows the selected position (E6)", rpos === "E6", `rPos="${rpos}"`);

  // Click a different residue → panel updates the position.
  await evalL(pg.sessionId, `document.querySelectorAll("#grid .pg-res")[20].click()`, 8000);
  let rpos2 = "";
  for (let i = 0; i < 25; i++) {
    await sleep(800);
    rpos2 = await evalL(pg.sessionId, `document.getElementById("rPos").textContent||""`, 8000) || "";
    if (rpos2 && rpos2 !== "E6") break;
  }
  chk("clicking another residue re-scans (position 21)", /21$/.test(rpos2), `rPos="${rpos2}"`);

  const odDesk = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth <= window.innerWidth + 1`,
    8000,
  );
  chk("no horizontal overflow (desktop)", odDesk === true);
  await setViewport(cdp, pg.sessionId, MOBILE);
  await sleep(500);
  const odMob = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth <= window.innerWidth + 1`,
    8000,
  );
  chk("no horizontal overflow (mobile 360px)", odMob === true);
  chk("no console errors", pg.errors.length === 0, pg.errors.slice(0, 2).join(" | "));
  await closePage(cdp, pg.targetId);
} catch (e) {
  console.log("THREW: " + (e && e.stack || e));
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(pass === total && total > 0 ? 0 : 1);
}
