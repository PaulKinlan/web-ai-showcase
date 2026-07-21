// Bird species classification end-to-end validation (real inference in headless Chrome). ~34 MB.
// Verifies: loads → Download → ready; the real EfficientNet model names the right species from the sample
// photos (flamingo → American Flamingo, bald eagle → Bald Eagle, puffin → Puffin) with a top-5 ranking; the
// preview + confidence bars + readout appear; the shared credit annotator adds attribution to the licensed
// sample photos; no console errors; no overflow desktop + mobile.
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
async function classifySample(sid, src) {
  await evalL(
    sid,
    `document.querySelector('#samples .bird-sample[data-src="${src}"]').click()`,
    10000,
  );
  let guess = "";
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    guess = await evalL(sid, `document.getElementById("guess").textContent`, 8000) || "";
    if (guess && !/Looking/.test(guess)) break;
  }
  return guess.trim();
}
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/bird-species-classification/`,
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
  for (let i = 0; i < 60; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("pickBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (controls enabled)", ready);

  const flamingo = await classifySample(pg.sessionId, "sample-flamingo.jpg");
  chk("flamingo photo → American Flamingo", /Flamingo/i.test(flamingo), JSON.stringify(flamingo));
  const bars = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#bars .bird-bar").length`,
    8000,
  );
  const ui = await evalL(
    pg.sessionId,
    `({preview:!document.getElementById("result").hidden, readout:!document.getElementById("readout").hidden})`,
    8000,
  );
  chk(
    "top-5 bars + preview + readout",
    bars === 5 && ui?.preview && ui?.readout,
    `bars=${bars} ${JSON.stringify(ui)}`,
  );

  const eagle = await classifySample(pg.sessionId, "sample-bald-eagle.jpg");
  chk("bald eagle photo → Bald Eagle", /Eagle/i.test(eagle), JSON.stringify(eagle));
  const puffin = await classifySample(pg.sessionId, "sample-puffin.jpg");
  chk("puffin photo → Puffin", /Puffin/i.test(puffin), JSON.stringify(puffin));

  // credit annotator: licensed sample photos carry attribution (creator name) after annotation.
  await sleep(1200);
  const credited = await evalL(
    pg.sessionId,
    `(()=>{const t=document.body.innerText+[...document.images].map(i=>i.title||"").join(" ")+[...document.querySelectorAll('[data-credit],.img-credit,figcaption')].map(e=>e.textContent).join(" ");return /Wilfredor|Paul Friel|Boaworm|M\\. Betley|Wikimedia/.test(t);})()`,
    8000,
  );
  chk("credit annotator adds attribution to licensed samples", credited === true);

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
