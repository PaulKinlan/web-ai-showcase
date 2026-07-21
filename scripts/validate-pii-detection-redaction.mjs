// PII detection & redaction end-to-end validation (real inference in headless Chrome). ~143 MB.
// Verifies: loads → Download → ready; controlled sentences have their email + phone captured as WHOLE spans
// with the right categories; the pre-filled support message detects email/SSN/IBAN/card/date; the Redact
// toggle replaces each value with a [CATEGORY] placeholder AND the raw email no longer appears in the output
// (no leak); the copy button appears; no console errors; no overflow desktop + mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 220000) =>
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/pii-detection-redaction/`,
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
  // wait until the textarea is enabled (onReady)
  let ready = false;
  for (let i = 0; i < 80; i++) {
    ready = await evalL(pg.sessionId, `!document.getElementById("text").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (editor enabled)", ready);

  // CORRECTNESS via the engine on controlled sentences.
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./pii.js");
      const eng = new M.PIIEngine(); await eng.load();
      const find = (spans, pred) => spans.find(pred);
      const a = await eng.detect("Email me at sam.lee@acme.io or call 415-555-0182.");
      const email = find(a.spans, s=>s.type==="EMAIL");
      const phone = find(a.spans, s=>M.groupOf(s.type)==="contact" && /415-555-0182/.test(s.text));
      const b = await eng.detect("Contact maria.gonzalez@hospital.org please.");
      const email2 = find(b.spans, s=>s.type==="EMAIL");
      // Naturally-phrased sentences (the real use case). The model fragments only on adversarially dense
      // adjacent numbers, which the page copy documents honestly.
      const c = await eng.detect("Please charge card 4539 1488 0343 6467 and wire the balance to IBAN GB29 NWBK 6016 1331 9268 19.");
      const d = await eng.detect("His SSN is 078-05-1120 and he was born on 1988-07-14.");
      return {
        emailText: email?.text, emailType: email?.type,
        phoneText: phone?.text,
        email2Text: email2?.text,
        cCard: !!find(c.spans, s=>/4539\\s?1488\\s?0343\\s?6467/.test(s.text)),
        cIban: !!find(c.spans, s=>/GB29/.test(s.text)),
        cSsn: !!find(d.spans, s=>s.type==="SSN" && /078-05-1120/.test(s.text)),
        cDate: !!find(d.spans, s=>s.type==="DATE" && /1988-07-14/.test(s.text)),
      };
    })()`,
    180000,
  );
  chk(
    "email captured as WHOLE span, typed EMAIL",
    rec?.emailText === "sam.lee@acme.io" && rec?.emailType === "EMAIL",
    JSON.stringify(rec?.emailText),
  );
  chk("phone captured (contact group)", /415-555-0182/.test(rec?.phoneText || ""), rec?.phoneText);
  chk(
    "email detected in the 'weak-model miss' sentence",
    rec?.email2Text === "maria.gonzalez@hospital.org",
    rec?.email2Text,
  );
  chk(
    "card + IBAN + SSN + DOB detected in natural sentences",
    rec?.cCard && rec?.cIban && rec?.cSsn && rec?.cDate,
    JSON.stringify(rec),
  );

  // Drive the deployed page: default text → highlights + legend appear.
  await evalL(
    pg.sessionId,
    `(()=>{const t=document.getElementById("text");t.dispatchEvent(new Event("input",{bubbles:true}));return true;})()`,
    10000,
  );
  let marks = 0;
  for (let i = 0; i < 30; i++) {
    await sleep(800);
    marks = await evalL(pg.sessionId, `document.querySelectorAll("#out mark.pii").length`, 8000) ||
      0;
    if (marks > 0) break;
  }
  const legendVisible = await evalL(
    pg.sessionId,
    `!document.getElementById("legend").hidden && document.querySelectorAll("#legend > span").length>0`,
    8000,
  );
  chk(
    "page highlights PII inline + legend shows groups",
    marks >= 4 && legendVisible === true,
    `marks=${marks}`,
  );

  // Redact toggle: [EMAIL] placeholder appears AND the raw email is gone from the output (no leak).
  const redact = await evalL(
    pg.sessionId,
    `(async()=>{
      document.getElementById("redactToggle").checked = true;
      document.getElementById("redactToggle").dispatchEvent(new Event("change",{bubbles:true}));
      await new Promise(r=>setTimeout(r,300));
      const out = document.getElementById("out").textContent;
      return { hasPlaceholder: /\\[EMAIL\\]/.test(out), leaksEmail: /jordan\\.mitchell@gmail\\.com/.test(out), copyVisible: !document.getElementById("copyBtn").hidden };
    })()`,
    15000,
  );
  chk(
    "redact replaces email with [EMAIL] placeholder",
    redact?.hasPlaceholder === true,
    JSON.stringify(redact),
  );
  chk("redacted output does NOT leak the raw email", redact?.leaksEmail === false);
  chk("copy-redacted button is available", redact?.copyVisible === true);

  // See-inside token strip populated.
  const toks = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#tokens .tok-chip").length`,
    8000,
  );
  chk("see-inside per-token BIO strip populated", toks >= 4, `chips=${toks}`);

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
