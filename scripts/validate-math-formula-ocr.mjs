// Math-formula OCR end-to-end validation (real inference in headless Chrome). ~320 MB.
// Verifies: loads → Download → ready; clicking sample equations runs the real Donut/texify model and
// generates the correct LaTeX structure (Pythagoras → a^2+b^2=c^2, an integral → \int…\frac, the quadratic
// formula → \frac…\sqrt…\pm); the image preview, copyable LaTeX and readout appear; no console errors;
// no overflow desktop + mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 200000) =>
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
async function ocrSample(sid, src) {
  await evalL(
    sid,
    `document.querySelector('#samples .mf-sample[data-src="${src}"]').click()`,
    10000,
  );
  let latex = "";
  for (let i = 0; i < 45; i++) {
    await sleep(2000);
    latex = await evalL(sid, `document.getElementById("latex").textContent`, 8000) || "";
    const busy = await evalL(
      sid,
      `!document.getElementById("runStatus").hidden && /Reading/.test(document.getElementById("runStatus").textContent)`,
      8000,
    );
    if (latex && !busy) break;
  }
  return latex.trim();
}
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/math-formula-ocr/`,
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
  for (let i = 0; i < 80; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("pickBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (controls enabled)", ready);

  const pyth = await ocrSample(pg.sessionId, "sample-pythagoras.png");
  chk(
    "Pythagoras → a^2+b^2=c^2 LaTeX",
    /a\^?\{?2\}?\s*\+\s*b\^?\{?2\}?\s*=\s*c\^?\{?2\}?/.test(pyth),
    JSON.stringify(pyth.slice(0, 60)),
  );
  // preview + readout appeared after a run
  const ui = await evalL(
    pg.sessionId,
    `({preview:!document.getElementById("preview").hidden, readout:!document.getElementById("readout").hidden, copy:!!document.getElementById("copyBtn")})`,
    8000,
  );
  chk("image preview + readout shown", ui?.preview && ui?.readout, JSON.stringify(ui));

  const integ = await ocrSample(pg.sessionId, "sample-integral.png");
  chk(
    "Integral → \\int + \\frac LaTeX",
    /\\int/.test(integ) && /\\frac/.test(integ),
    JSON.stringify(integ.slice(0, 60)),
  );

  const quad = await ocrSample(pg.sessionId, "sample-quadratic.png");
  chk(
    "Quadratic formula → \\frac + \\sqrt + \\pm",
    /\\frac/.test(quad) && /\\sqrt/.test(quad) && /\\pm/.test(quad),
    JSON.stringify(quad.slice(0, 70)),
  );

  // copy path works (writes cleaned latex to clipboard, button feedback)
  const copied = await evalL(
    pg.sessionId,
    `(()=>{document.getElementById("copyBtn").click();return true;})()`,
    8000,
  );
  chk("copy button operable", copied === true);

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
