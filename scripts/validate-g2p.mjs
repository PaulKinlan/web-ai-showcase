// Grapheme-to-phoneme end-to-end validation (real inference in headless Chrome). ~84 MB byT5.
// Verifies: loads → Download → auto-phonemizes "hello" to IPA (ˈhɛɫoʊ); an example chip (bonjour) switches
// language + phonemizes to French IPA; a non-Latin script (Russian спасибо) phonemizes; no console errors;
// no overflow desktop+mobile.
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
// crude: does a string contain IPA-ish characters (non-ASCII-letter phonetic symbols or stress marks)?
const looksIPA = (s) => typeof s === "string" && s.length > 0 && /[ˈˌəɛɪʊʌɔæŋʃʒθðɡɾʁʲːɐ-ʯ]/.test(s);
try {
  const pg = await openPage(
    cdp,
    `http://127.0.0.1:${port}/web-ai-showcase/models/grapheme-to-phoneme/`,
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
  // wait for the auto "hello" phonemization
  let out = "";
  for (let i = 0; i < 70; i++) {
    out = await evalL(pg.sessionId, `document.getElementById("out").textContent`, 10000) || "";
    if (looksIPA(out)) break;
    await sleep(2500);
  }
  chk("ready → 'hello' phonemized to IPA", looksIPA(out), out);
  chk("hello → ˈhɛɫoʊ (English)", /h/.test(out) && /[ɛɫoʊ]/.test(out), out);
  // click the bonjour example chip → French IPA
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll("#chips .chip")].find(x=>/bonjour/.test(x.textContent));b&&b.click();return !!b;})()`,
    10000,
  );
  let fr = "";
  for (let i = 0; i < 25; i++) {
    await sleep(1200);
    const w = await evalL(pg.sessionId, `document.getElementById("word").value`, 8000);
    fr = await evalL(pg.sessionId, `document.getElementById("out").textContent`, 8000) || "";
    if (w === "bonjour" && looksIPA(fr) && fr !== out) break;
  }
  chk("bonjour → French IPA (ʒ/ʁ)", looksIPA(fr) && /[ʒʁ]/.test(fr), fr);
  // non-Latin script: Russian спасибо via the chip
  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll("#chips .chip")].find(x=>/спасибо/.test(x.textContent));b&&b.click();return !!b;})()`,
    10000,
  );
  let ru = "";
  for (let i = 0; i < 25; i++) {
    await sleep(1200);
    ru = await evalL(pg.sessionId, `document.getElementById("out").textContent`, 8000) || "";
    if (looksIPA(ru) && ru !== fr) break;
  }
  chk("Cyrillic спасибо phonemizes (byte-level, any script)", looksIPA(ru), ru);
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
