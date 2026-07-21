// Prompt-injection detection end-to-end validation (real inference in headless Chrome). ~739 MB fp32.
// Verifies: loads → Download → auto-screens the ignore-instructions example → INJECTION; a benign example
// chip → SAFE; a jailbreak (DAN) chip → INJECTION; verdict badge + confidence render; no console errors;
// no overflow desktop+mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 360000) =>
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
const verdict = (sid) => evalL(sid, `document.getElementById("verdict").textContent`, 10000);
const clickChipContaining = (sid, frag) =>
  evalL(
    sid,
    `(()=>{const b=[...document.querySelectorAll("#chips .chip")].find(x=>/${frag}/i.test(x.textContent));b&&b.click();return !!b;})()`,
    10000,
  );
async function waitVerdict(sid, re, prev) {
  let v = "";
  for (let i = 0; i < 30; i++) {
    await sleep(1200);
    v = await verdict(sid) || "";
    if (re.test(v) && v !== prev) return v;
  }
  return v;
}
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/prompt-injection-detection/`,
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
  // wait for the auto ignore-instructions example → INJECTION
  let v1 = "";
  for (let i = 0; i < 90; i++) {
    v1 = await verdict(pg.sessionId) || "";
    if (/INJECTION|SAFE/.test(v1)) break;
    await sleep(3000);
  }
  chk("ready → ignore-instructions example → INJECTION", /INJECTION/.test(v1), v1);
  // benign chip → SAFE
  await clickChipContaining(pg.sessionId, "recipe|Summarize|Python");
  const v2 = await waitVerdict(pg.sessionId, /SAFE|INJECTION/, v1);
  chk("benign prompt → SAFE", /SAFE/.test(v2), v2);
  // DAN jailbreak chip → INJECTION
  await clickChipContaining(pg.sessionId, "DAN");
  const v3 = await waitVerdict(pg.sessionId, /INJECTION|SAFE/, v2);
  chk("DAN jailbreak → INJECTION", /INJECTION/.test(v3), v3);
  // verdict badge + confidence bar rendered
  const ui = await evalL(
    pg.sessionId,
    `(()=>({badge:!!document.querySelector(".pg-badge"), bar:!!document.querySelector("#bar > i")}))()`,
    10000,
  );
  chk("verdict badge + confidence bar rendered", ui?.badge && ui?.bar, JSON.stringify(ui));
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
