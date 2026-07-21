// Human action-recognition end-to-end validation (real inference in headless Chrome). ~87 MB q8 model.
// Verifies: loads → Download → auto-classifies the licensed gallery; the marathon photo reads as Running
// and the woman-with-a-cup as Drinking (real activity recognition, not canned); each card shows top-k bars;
// no console errors; no overflow desktop+mobile.
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/human-action-recognition/`,
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
  // wait for all 6 gallery cards to get a top-action label
  let labeled = 0;
  for (let i = 0; i < 80; i++) {
    labeled = await evalL(
      pg.sessionId,
      `[...document.querySelectorAll("#gallery .har-top")].filter(e=>e.textContent&&e.textContent!=="…").length`,
      10000,
    ) || 0;
    if (labeled >= 6) break;
    await sleep(2500);
  }
  chk("ready → all 6 gallery photos classified", labeled === 6, `labeled=${labeled}`);
  const tops = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#gallery .har-top")].map(e=>e.textContent)`,
    10000,
  ) || [];
  // marathon (card 0) → Running; woman-with-cup (card 1) → Drinking
  chk("marathon → Running", /Running/.test(tops[0] || ""), tops[0]);
  chk("woman with a cup → Drinking", /Drinking/.test(tops[1] || ""), tops[1]);
  // each card shows top-k bars
  const bars = await evalL(
    pg.sessionId,
    `[...document.querySelectorAll("#gallery .har-card")].map(c=>c.querySelectorAll(".har-bar").length)`,
    10000,
  ) || [];
  chk(
    "each photo shows top-k probability bars",
    bars.length === 6 && bars.every((n) => n >= 3),
    JSON.stringify(bars),
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
