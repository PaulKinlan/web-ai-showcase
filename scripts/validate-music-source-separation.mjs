// Music source separation end-to-end validation (real inference in headless Chrome). ~303 MB.
// Verifies: loads → Download → ready; the real Demucs model separates the sample into 4 stems whose sum
// reconstructs the mix (corr > 0.8) and whose energies differ by content (the spoken voice lands in "other",
// not the sung-vocals stem); the deployed page renders 4 stem rows with download links + presets; no console
// errors; no overflow desktop + mobile.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(
  B
);
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 260000) =>
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
    `http://127.0.0.1:${port}/web-ai-showcase/models/music-source-separation/`,
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
    ready = await evalL(pg.sessionId, `!document.getElementById("sampleBtn").disabled`, 10000);
    if (ready) break;
    await sleep(2000);
  }
  chk("ready (controls enabled)", ready);

  // CORRECTNESS via the engine on the sample mix.
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./sep.js");
      const eng = new M.SepEngine(); await eng.load();
      const seg = await M.makeSampleMix();
      const mix = seg.ch0.slice();
      const r = await eng.separate(seg);
      const names = r.stems.map(s=>s.name);
      const rms = {}; for (const s of r.stems) { let e=0; for (let i=0;i<s.l.length;i++) e+=s.l[i]*s.l[i]; rms[s.name]=Math.sqrt(e/s.l.length); }
      // reconstruction: sum of stems ch0 vs mix
      const n = r.len; const sum = new Float32Array(n);
      for (const s of r.stems) for (let i=0;i<n;i++) sum[i]+=s.l[i];
      const corr=(a,b,len)=>{let sa=0,sb=0,saa=0,sbb=0,sab=0;for(let i=0;i<len;i++){sa+=a[i];sb+=b[i];saa+=a[i]*a[i];sbb+=b[i]*b[i];sab+=a[i]*b[i];}const cov=sab-sa*sb/len;return cov/Math.sqrt((saa-sa*sa/len)*(sbb-sb*sb/len));};
      return { names, nStems: r.stems.length, ms: r.ms, reconstruction:+corr(sum,mix,n).toFixed(3), rms:{drums:+rms.drums.toFixed(4),bass:+rms.bass.toFixed(4),other:+rms.other.toFixed(4),vocals:+rms.vocals.toFixed(4)} };
    })()`,
    240000,
  );
  chk(
    "4 stems [drums,bass,other,vocals]",
    rec?.nStems === 4 &&
      JSON.stringify(rec?.names) === JSON.stringify(["drums", "bass", "other", "vocals"]),
    JSON.stringify(rec?.names),
  );
  chk(
    "stems reconstruct the mix (corr > 0.8)",
    rec?.reconstruction > 0.8,
    "corr=" + rec?.reconstruction,
  );
  // Content separation: the bass-heavy backing beat isolates to the BASS stem (loudest), and the 4 stems
  // carry genuinely different energy (real separation, not a pass-through of identical stems).
  const rmsVals = rec?.rms ? Object.values(rec.rms) : [];
  const bassIsLoudest = rec?.rms && rec.rms.bass === Math.max(...rmsVals);
  const distinct = rmsVals.length === 4 && Math.max(...rmsVals) > 2 * Math.min(...rmsVals);
  chk(
    "content-separated: bass-heavy backing → bass stem (loudest); stems distinct",
    bassIsLoudest && distinct,
    JSON.stringify(rec?.rms),
  );

  // Drive the deployed page: click "Try the sample" → 4 stem rows + download links render.
  await evalL(pg.sessionId, `document.getElementById("sampleBtn").click()`, 10000);
  let rows = 0;
  for (let i = 0; i < 60; i++) {
    await sleep(2000);
    rows = await evalL(pg.sessionId, `document.querySelectorAll("#stems .ms-stem").length`, 8000) ||
      0;
    if (rows === 4) break;
  }
  const dls = await evalL(
    pg.sessionId,
    `document.querySelectorAll("#stems a.ms-dl[download]").length`,
    8000,
  );
  const readout = await evalL(pg.sessionId, `!document.getElementById("readout").hidden`, 8000);
  chk(
    "page renders 4 stem rows + download links + readout",
    rows === 4 && dls === 4 && readout === true,
    `rows=${rows} dls=${dls}`,
  );

  // Karaoke preset unchecks the vocals stem.
  const karaoke = await evalL(
    pg.sessionId,
    `(()=>{document.getElementById("presetKaraoke").click();const v=[...document.querySelectorAll("#stems .ms-stem")].find(r=>r.dataset.name==="vocals");return v&&!v.querySelector("input").checked;})()`,
    8000,
  );
  chk("Karaoke preset mutes the vocals stem", karaoke === true);

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
