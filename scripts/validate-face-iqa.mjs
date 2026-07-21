const B = "./browser.mjs";
// Face image-quality end-to-end validation (real inference in headless Chrome). ~7 MB model, fast.
const { closePage, launchChrome, openPage, setViewport, startServer } = await import(B);
const { CDP } = await import(B);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 200000) =>
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
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/face-image-quality/`,
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
  let g = 0;
  for (let i = 0; i < 40 && g < 6; i++) {
    g =
      await evalL(pg.sessionId, `document.querySelectorAll("#gallery .iqa-face").length`, 10000) ||
      0;
    if (g < 6) await sleep(1500);
  }
  chk("ready → all 6 licensed faces scored + ranked", g === 6, `faces=${g}`);
  // ranked descending?
  const ranked = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#gallery .iqa-face .q")].map(e=>parseFloat(e.textContent))`,
    10000,
  ) || [];
  chk(
    "faces ranked by quality (descending)",
    ranked.length === 6 && ranked.every((v, i) => i === 0 || ranked[i - 1] >= v - 1e-6),
    JSON.stringify(ranked.map((v) => +v.toFixed(3))),
  );
  // degrade: set blur 8 → score should drop below sharp original
  await evalL(
    pg.sessionId,
    `(()=>{const b=document.getElementById("blur");b.value="8";b.dispatchEvent(new Event("input"));return true;})()`,
    10000,
  );
  let res = "";
  for (let i = 0; i < 20 && !/Quality/.test(res); i++) {
    res = await evalL(pg.sessionId, `document.getElementById("result").textContent`, 10000) || "";
    if (!/Quality/.test(res)) await sleep(1000);
  }
  const m = res.match(/Quality ([\d.]+) \(sharp original ([\d.]+)/);
  chk(
    "degradation (blur 8) lowers the quality score below the sharp original",
    m && parseFloat(m[1]) < parseFloat(m[2]),
    res.slice(0, 80),
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
console.log(`\n${pass}/${total} face-iqa checks passed.`);
process.exit(pass === total ? 0 : 1);
