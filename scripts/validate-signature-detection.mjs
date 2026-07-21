// Signature detection end-to-end validation (real inference in headless Chrome).
// Verifies: loads → Download → ready; the real YOLOS signature detector locates ≥1 signature box in each
// sample document (contract + letter) with a plausible score; the deployed page draws the overlay box,
// shows the verdict + readout when a sample is clicked; no console errors; no overflow desktop + mobile.
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/signature-detection/`,
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
  for (let i = 0; i < 90; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("pickBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (Choose image enabled)", ready);

  // CORRECTNESS via the engine on both sample documents.
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./signature.js");
      const eng = new M.SignatureEngine(); await eng.load();
      const out = {};
      for (const s of ["sample-contract.png","sample-letter.png"]) {
        const r = await eng.detect(new URL("./"+s, location.href).href, 0.3);
        out[s] = r.dets.map(d=>({label:d.label,score:+d.score.toFixed(2),w:Math.round(d.xmax-d.xmin),h:Math.round(d.ymax-d.ymin)}));
      }
      return out;
    })()`,
    180000,
  );
  const c = rec?.["sample-contract.png"] || [];
  const l = rec?.["sample-letter.png"] || [];
  chk(
    "contract: ≥1 signature box (score ≥ .5)",
    c.length >= 1 && c[0].label === "signature" && c[0].score >= 0.5,
    JSON.stringify(c),
  );
  chk(
    "letter: ≥1 signature box (score ≥ .5)",
    l.length >= 1 && l[0].label === "signature" && l[0].score >= 0.5,
    JSON.stringify(l),
  );
  chk(
    "boxes have real extent (>20px each side)",
    c.every((d) => d.w > 20 && d.h > 20) && l.every((d) => d.w > 20 && d.h > 20),
    JSON.stringify({ c: c[0], l: l[0] }),
  );

  // Drive the page: click the contract sample → verdict + overlay box + readout.
  await evalL(
    pg.sessionId,
    `document.querySelector('#samples .sig-sample[data-src="sample-contract.png"]').click()`,
    8000,
  );
  let verdict = "", boxes = 0, readout = false;
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    verdict = await evalL(pg.sessionId, `document.getElementById("verdict").textContent||""`, 8000) || "";
    boxes = await evalL(pg.sessionId, `document.querySelectorAll("#overlay .sig-box").length`, 8000) || 0;
    readout = await evalL(pg.sessionId, `!document.getElementById("readout").hidden`, 8000);
    if (/found/.test(verdict) && boxes >= 1) break;
  }
  chk(
    "page: verdict reports signature(s) found",
    /signature/.test(verdict) && /found/.test(verdict),
    `verdict="${verdict}"`,
  );
  chk("page: overlay box drawn on the document", boxes >= 1, `boxes=${boxes}`);
  chk("page: readout visible (backend/latency/count)", readout === true);

  // click the letter sample → still detects
  await evalL(
    pg.sessionId,
    `document.querySelector('#samples .sig-sample[data-src="sample-letter.png"]').click()`,
    8000,
  );
  let boxes2 = 0;
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    boxes2 = await evalL(pg.sessionId, `document.querySelectorAll("#overlay .sig-box").length`, 8000) || 0;
    if (boxes2 >= 1) break;
  }
  chk("page: letter sample also detects (box drawn)", boxes2 >= 1, `boxes=${boxes2}`);

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
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(pass === total ? 0 : 1);
}
