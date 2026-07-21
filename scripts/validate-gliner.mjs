// GLiNER zero-shot NER end-to-end validation (real inference in headless Chrome). ~183 MB q8.
// Verifies: loads → Download → auto-extracts the News example (Apple→company, Steve Jobs→person, 1976→date);
// switching to the Sci-fi example extracts ARBITRARY types (spaceship: USS Enterprise, planet: Vulcan) —
// the zero-shot property; no console errors; no overflow desktop+mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 300000) =>
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
// read the highlighted entities as "type:text" pairs
const ents = (sid) =>
  evalL(
    sid,
    `[...document.querySelectorAll("#out .gl-ent")].map(m=>({type:m.querySelector("sub")?.textContent, text:m.textContent.replace(m.querySelector("sub")?.textContent||"","").trim()}))`,
    10000,
  );
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/gliner-zero-shot-ner/`,
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
  // wait for the auto News extraction
  let e1 = [];
  for (let i = 0; i < 80; i++) {
    e1 = await ents(pg.sessionId) || [];
    if (e1.length >= 3) break;
    await sleep(2500);
  }
  const find = (list, type, text) =>
    list.some((e) => e.type === type && new RegExp(text, "i").test(e.text));
  chk("ready → News example extracted", e1.length >= 3, JSON.stringify(e1).slice(0, 200));
  chk("Apple → company", find(e1, "company", "Apple"), "");
  chk("Steve Jobs → person", find(e1, "person", "Steve Jobs"), "");
  chk("1976 → date", find(e1, "date", "1976"), "");
  // switch to Sci-fi example → ARBITRARY types (the zero-shot proof)
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll("#chips .chip")].find(x=>/Sci-fi/.test(x.textContent));b&&b.click();return !!b;})()`,
    10000,
  );
  let e2 = [];
  for (let i = 0; i < 30; i++) {
    await sleep(1200);
    e2 = await ents(pg.sessionId) || [];
    if (e2.some((e) => e.type === "spaceship")) break;
  }
  chk(
    "zero-shot: spaceship → USS Enterprise",
    find(e2, "spaceship", "Enterprise"),
    JSON.stringify(e2).slice(0, 160),
  );
  chk("zero-shot: planet → Vulcan", find(e2, "planet", "Vulcan"), "");
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
