// ESM-2 protein language model end-to-end validation (real inference in headless Chrome). ~8 MB q8.
// Verifies: loads → Download → auto-scans ubiquitin into a conservation map (varying per-residue fit);
// masking the conserved C-terminal glycine recovers G; masking a conserved lysine recovers K; no console
// errors; no overflow desktop+mobile.
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
  const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/models/esm-protein/`);
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
  // wait for the auto conservation scan to finish
  let st = "";
  for (let i = 0; i < 70; i++) {
    st = await evalL(pg.sessionId, `document.getElementById("status").textContent`, 10000) || "";
    if (/Conservation map/.test(st)) break;
    await sleep(2500);
  }
  chk("ready → ubiquitin conservation scan", /Conservation map/.test(st), st);
  // the residue track is coloured (cells have background set)
  const tinted = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#seqView .esm-res")].filter(e=>e.getAttribute("style")&&/background/.test(e.getAttribute("style"))).length`,
    10000,
  );
  const nres = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#seqView .esm-res").length`,
    10000,
  );
  chk(
    "residues rendered + conservation-tinted",
    nres > 50 && tinted === nres,
    `res=${nres} tinted=${tinted}`,
  );
  // correctness: run predictions in-page for two conserved positions (recovers G and K)
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
    const M = await import("./esm.js");
    const eng = new M.EsmEngine(); await eng.load();
    const ubi = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
    const g = await eng.predict(ubi, ubi.length-2); // conserved di-glycine → G
    const k = await eng.predict(ubi, 5);            // conserved K → K
    return { g_truth:g.truth, g_top:g.top[0].aa, k_truth:k.truth, k_top:k.top[0].aa };
  })()`,
    120000,
  );
  chk(
    "masked conserved glycine → recovers G",
    rec && rec.g_truth === "G" && rec.g_top === "G",
    JSON.stringify(rec),
  );
  chk(
    "masked conserved lysine → recovers K",
    rec && rec.k_truth === "K" && rec.k_top === "K",
    JSON.stringify(rec),
  );
  // click a residue in the UI → prediction panel populates
  await evalL(
    pg.sessionId,
    `(()=>{const b=document.querySelector('#seqView .esm-res[data-i="5"]');b&&b.click();return !!b;})()`,
    10000,
  );
  let pred = 0;
  for (let i = 0; i < 20; i++) {
    await sleep(800);
    pred = await evalL(pg.sessionId, `document.querySelectorAll("#pred .esm-pbar").length`, 8000) ||
      0;
    if (pred > 0) break;
  }
  chk("clicking a residue shows amino-acid predictions", pred >= 3, `bars=${pred}`);
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
