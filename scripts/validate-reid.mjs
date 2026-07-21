const B = "./browser.mjs";
// Person re-ID end-to-end validation (real inference in headless Chrome). Heavy (~40s, 107 MB download).
const { closePage, launchChrome, openPage, setViewport, startServer } = await import(B);
const { CDP } = await import(B);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 280000) =>
  cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expr});}catch(e){return{__err:String(e&&e.message||e).slice(0,150)};}})()`,
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
  const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/models/person-reid/`);
  await sleep(1400);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"),dl:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent)),gallery:document.querySelectorAll("#gallery .reid-person").length}))()`,
    15000,
  );
  chk(
    "loads: loader + Download + 5 person crops",
    s0?.loader && s0?.dl && s0?.gallery === 5,
    JSON.stringify(s0),
  );
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
    15000,
  );
  // wait for the preselected same-person result to appear
  let res = "";
  for (let i = 0; i < 80 && !/similarity/.test(res); i++) {
    res = await evalL(pg.sessionId, `document.getElementById("result").textContent`, 10000) || "";
    if (!/similarity/.test(res)) await sleep(2000);
  }
  const mSame = res.match(/similarity ([\d.]+)/);
  const same = mSame ? parseFloat(mSame[1]) : 0;
  chk(
    "SAME person (runner1 vs re-framed runner1) → high similarity + SAME verdict",
    /SAME/.test(res) && same >= 0.85,
    `sim=${same} · ${res.slice(0, 60)}`,
  );
  // now select two DIFFERENT runners (index 0 and 3)
  await evalL(
    pg.sessionId,
    `(()=>{const g=document.querySelectorAll("#gallery .reid-person"); [...g].forEach(c=>c.classList.remove("sel")); return true;})()`,
    10000,
  );
  await evalL(pg.sessionId, `document.querySelectorAll("#gallery .reid-person")[0].click()`, 10000);
  await evalL(pg.sessionId, `document.querySelectorAll("#gallery .reid-person")[3].click()`, 10000);
  let res2 = "";
  for (let i = 0; i < 20 && !/similarity/.test(res2); i++) {
    res2 = await evalL(pg.sessionId, `document.getElementById("result").textContent`, 10000) || "";
    if (!/similarity/.test(res2)) await sleep(1000);
  }
  const mDiff = res2.match(/similarity ([\d.]+)/);
  const diff = mDiff ? parseFloat(mDiff[1]) : 1;
  chk(
    "DIFFERENT people (runner1 vs runner5) → lower similarity than same-person",
    diff < same,
    `same=${same} diff=${diff}`,
  );
  chk("no console errors", pg.errors.length === 0, pg.errors.slice(0, 1).join("|"));
  const ovD = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth-window.innerWidth`,
    10000,
  );
  await setViewport(cdp, pg.sessionId, {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await sleep(300);
  const ovM = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth-window.innerWidth`,
    10000,
  );
  chk("no overflow desktop+mobile", ovD <= 2 && ovM <= 2, `D=${ovD} M=${ovM}`);
  await closePage(cdp, pg.targetId);
} finally {
  chrome.kill();
  server.close();
}
console.log(`\n${pass}/${total} person-reid checks passed.`);
process.exit(pass === total ? 0 : 1);
