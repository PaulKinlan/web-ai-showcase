// Code language ID end-to-end validation (real inference in headless Chrome). ~84 MB.
// Verifies: loads → Download → ready; the pre-filled Python sample auto-classifies as Python; loading the
// Go / Ruby / PHP / Java samples classifies each correctly via the real CodeBERTa model, with a full
// 6-language ranking + readout; no console errors; no overflow desktop + mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 120000) =>
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
async function waitVerdict(sid, re) {
  let v = "";
  for (let i = 0; i < 25; i++) {
    await sleep(700);
    v = await evalL(sid, `document.getElementById("verdict").textContent`, 8000) || "";
    if (re.test(v)) break;
  }
  return v.trim();
}
async function loadSample(sid, lang, re) {
  await evalL(sid, `document.querySelector('#chips .cl-chip[data-lang="${lang}"]').click()`, 8000);
  return waitVerdict(sid, re);
}
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/code-language-id/`,
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
    ready = await evalL(pg.sessionId, `!document.getElementById("code").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (editor enabled)", ready);

  // pre-filled Python auto-classifies
  const py = await waitVerdict(pg.sessionId, /Python/i);
  chk("pre-filled Python sample → Python", /Python/i.test(py), JSON.stringify(py));
  const bars = await evalL(pg.sessionId, `document.querySelectorAll("#bars .cl-bar").length`, 8000);
  const readout = await evalL(pg.sessionId, `!document.getElementById("readout").hidden`, 8000);
  chk("full 6-language ranking + readout", bars === 6 && readout === true, `bars=${bars}`);

  const go = await loadSample(pg.sessionId, "go", /Go/);
  chk("Go sample → Go", /^Go\b/.test(go), JSON.stringify(go));
  const ruby = await loadSample(pg.sessionId, "ruby", /Ruby/i);
  chk("Ruby sample → Ruby", /Ruby/i.test(ruby), JSON.stringify(ruby));
  const php = await loadSample(pg.sessionId, "php", /PHP/i);
  chk("PHP sample → PHP", /PHP/i.test(php), JSON.stringify(php));
  const java = await loadSample(pg.sessionId, "java", /Java\b/);
  chk("Java sample → Java", /Java\b/.test(java) && !/JavaScript/i.test(java), JSON.stringify(java));

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
