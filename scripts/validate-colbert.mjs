// ColBERT late-interaction end-to-end validation (real inference in headless Chrome). ~133 MB fp32.
// Verifies: loads → Download → auto-matches "who wrote Hamlet" against a Shakespeare doc; the key query
// tokens align to the right document words (Hamlet→Hamlet, wrote→wrote) with high similarity; a relevant
// document out-scores an irrelevant one (MaxSim ranking); no console errors; no overflow desktop+mobile.
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/colbert-late-interaction/`,
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
  // wait for the auto Hamlet example → alignment rendered
  let a = 0;
  for (let i = 0; i < 70; i++) {
    a = await evalL(pg.sessionId, `document.querySelectorAll("#align .cb-pair").length`, 10000) ||
      0;
    if (a >= 3) break;
    await sleep(2500);
  }
  chk("ready → query aligned to document (per-token)", a >= 3, `pairs=${a}`);
  // the alignment maps key query words to the right doc words with high similarity
  const pairs = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#align .cb-pair")].map(p=>({q:p.querySelector(".cb-q").textContent, d:p.querySelector(".cb-d").textContent, sim:parseInt(p.querySelector(".cb-sim").textContent)}))`,
    10000,
  ) || [];
  const findPair = (qw) => pairs.find((p) => new RegExp(qw, "i").test(p.q));
  const hamlet = findPair("hamlet"), wrote = findPair("wrote");
  chk(
    "query 'Hamlet' → document 'Hamlet' (high sim)",
    hamlet && /hamlet/i.test(hamlet.d) && hamlet.sim >= 85,
    JSON.stringify(hamlet),
  );
  chk(
    "query 'wrote' → document 'wrote' (high sim)",
    wrote && /wrote/i.test(wrote.d) && wrote.sim >= 85,
    JSON.stringify(wrote),
  );
  // CORRECTNESS: relevant doc out-scores an irrelevant one (MaxSim ranking), computed in-page
  const rank = await evalL(
    pg.sessionId,
    `(async()=>{
    const M = await import("./colbert.js");
    const eng = new M.ColbertEngine(); await eng.load();
    const q = "who wrote the play Hamlet";
    const rel = await eng.score(q, "William Shakespeare wrote the tragedy Hamlet around 1600.");
    const irr = await eng.score(q, "The recipe calls for two cups of flour and a pinch of salt.");
    return { rel: rel.score, irr: irr.score };
  })()`,
    120000,
  );
  chk(
    "relevant document out-scores irrelevant (MaxSim ranking)",
    rank && rank.rel > rank.irr,
    JSON.stringify(rank),
  );
  // document heatmap tokens rendered
  const dtok = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#docView .cb-dtok").length`,
    10000,
  );
  chk("document rendered as a match heatmap", dtok >= 5, `docTokens=${dtok}`);
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
