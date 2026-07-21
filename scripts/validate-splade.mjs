// SPLADE sparse-retrieval end-to-end validation (real inference in headless Chrome). ~532 MB fp32.
// Verifies: loads → Download → auto-encodes the pasta example (weights pasta/italian/restaurant AND shows
// expansion terms not in the text); a second text produces a match score with shared terms; no console
// errors; no overflow desktop+mobile.
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
const terms = (sid) =>
  evalL(
    sid,
    `[...document.querySelectorAll("#cloud .sp-term")].map(e=>({t:e.textContent.replace(/\\+$/,"").trim(), exp:e.classList.contains("sp-exp")}))`,
    10000,
  );
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/splade-sparse-retrieval/`,
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
  // wait for the auto pasta example
  let ts = [];
  for (let i = 0; i < 100; i++) {
    ts = await terms(pg.sessionId) || [];
    if (ts.length >= 5) break;
    await sleep(3000);
  }
  const words = ts.map((x) => x.t.toLowerCase());
  chk("ready → pasta example encoded to weighted terms", ts.length >= 5, `${ts.length} terms`);
  chk(
    "weights input terms (pasta/italian/restaurant)",
    ["pasta", "italian", "restaurant"].filter((w) => words.includes(w)).length >= 2,
    words.slice(0, 10).join(","),
  );
  chk(
    "adds expansion terms (not in the input)",
    ts.some((x) => x.exp),
    `expansions=${ts.filter((x) => x.exp).map((x) => x.t).slice(0, 6).join(",")}`,
  );
  // a related second text → a positive match score + shared terms
  await evalL(
    pg.sessionId,
    `(()=>{const i=document.getElementById("text2");i.value="Where can I find good spaghetti and lasagna?";i.dispatchEvent(new Event("input"));return true;})()`,
    10000,
  );
  let matched = "";
  for (let i = 0; i < 30; i++) {
    await sleep(1500);
    matched = await evalL(pg.sessionId, `document.getElementById("match").textContent`, 8000) || "";
    if (/Relevance score/.test(matched)) break;
  }
  const score = parseFloat((matched.match(/([0-9.]+)/) || [])[1] || "0");
  chk(
    "second text → positive relevance score",
    /Relevance score/.test(matched) && score > 0,
    matched,
  );
  const shared = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#shared span").length`,
    8000,
  );
  chk("shows shared contributing terms", shared >= 1, `shared=${shared}`);
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
