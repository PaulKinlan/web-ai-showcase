// Music source separation end-to-end validation (real inference in headless Chrome). ~303 MB + ~40 MB + ~95 MB.
// Stages: MrCitron/demucs-v4-onnx, onnx-community/whisper-tiny.en, onnx-community/Musical-Instrument-Classification-ONNX.
// Route-complete: drives the OVERVIEW plus all four ladder routes (basics / practical / wild / multi-model)
// at desktop and mobile, exercising the visible controls on each. Verifies: loads → Download → ready; the
// real Demucs model separates the bundled OPENLY LICENSED sample song ("The CC BY Song" by loveshadow,
// additional lyrics by Victor Stone, CC BY 3.0, via ccMixter — the opening 0.0-7.8 s window; provenance +
// sha256 in CREDITS.md) into 4 stems whose sum reconstructs the mix (corr > 0.8), with the sung vocal
// carried by the VOCALS stem; the bundled file is pinned by exact size + SHA-256; visible attribution;
// overview upload path via CDP DOM.setFileInputFiles; basics renders 4 playable
// per-stem GainNodes (faders measurably drive the gains; presets set faders + mutes; the downloaded
// remix WAV is recomputed from the real stems); multi-model chains the vocals stem to a whisper-tiny.en
// ASR worker — the editable transcript is real (empty for this song: a tiny speech model vs singing,
// measured + explained, with the original-mix control), the chain is health-proven on real speech
// (the family-local LibriSpeech fixture fixtures/libri-61-70968-0000-16k-mono.wav — OpenSLR SLR12
// test-clean 61-70968-0000, CC BY 4.0, see CREDITS.md/provenance.json — a neutral ASR validation
// asset with no cross-family dependency), and
// whisper NEVER auto-downloads (0 whisper fetches before an explicit click); no console errors; no
// overflow at desktop + mobile on every route.
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, DESKTOP, MOBILE, CDP } =
  await import(
    B
  );
const { server, port } = await startServer();
// Family-unique Chrome profile: cleanup (ours and others') is scoped to THIS exact --user-data-dir,
// so no kill or wipe can ever touch another lane's browser.
const chrome = await launchChrome({
  userDataDir: new URL("../.chrome-profile-music-source-separation", import.meta.url).pathname,
});
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

// No generated audio anywhere: the sample path loads/decodes the bundled song, and the upload path is
// exercised by attaching that SAME real bundled MP3 (a file on disk) to the page's REAL upload control
// via CDP DOM.setFileInputFiles.
const { readFileSync } = await import("node:fs");
const { createHash } = await import("node:crypto");
const UPLOAD_FILE = new URL(
  "../models/music-source-separation/song.mp3",
  import.meta.url,
).pathname;

// Attribution probe, run in-page on every route: the complete credit (title link, artist, additional
// lyricist, exact CC BY 3.0 deed link, source page link, unmodified + first-~7.8 s statements,
// route-local CREDITS.md link) must sit beside the route's sample control (#sampleBtn / #runBtn).
const ATTRIB_JS = `(()=>{
  const btn = document.getElementById("sampleBtn") || document.getElementById("runBtn");
  const links = [...document.querySelectorAll("a")];
  const title = links.find(a => a.href === "https://ccmixter.org/files/Loveshadow/29635" && a.textContent.includes("The CC BY Song"));
  const credit = title && title.closest("p");
  let near = false;
  if (btn && credit) {
    const b = btn.getBoundingClientRect(), c = credit.getBoundingClientRect();
    near = Math.abs(c.top - b.top) < 220;
  }
  const txt = document.body.textContent;
  return {
    titleLink: !!title,
    sourceLinks: links.filter(a => a.href === "https://ccmixter.org/files/Loveshadow/29635").length,
    licenseExact: links.some(a => a.href === "https://creativecommons.org/licenses/by/3.0/"),
    artist: txt.includes("loveshadow"),
    lyricist: txt.includes("Victor Stone"),
    unmodified: /unmodified/.test(txt),
    window: /~7\\.8 s/.test(txt),
    creditsLink: !!(credit && [...credit.querySelectorAll("a")].some(a => a.getAttribute("href") === "CREDITS.md")),
    nearControls: near,
  };
})()`;

async function attribCheck(sessionId, name) {
  const a = await evalL(sessionId, ATTRIB_JS, 10000);
  chk(
    `${name}: attribution complete beside sample controls`,
    !!a && a.titleLink && a.sourceLinks >= 2 && a.licenseExact && a.artist && a.lyricist &&
      a.unmodified && a.window && a.creditsLink && a.nearControls,
    JSON.stringify(a),
  );
}

// Disposal + fresh-reload probe, run LAST on each route (it tears the page's engine down and reloads).
// Genuine on both ends: disposal must make the engine unusable (isDisposed + worker null + separate()
// rejects), and the Reset button must produce a brand-new JS context (a marker we set beforehand is
// gone) with the results cleared.
async function resetCheck(pgX, name) {
  const pre = await evalL(
    pgX.sessionId,
    `(async()=>{
      const eng = window.__engine;
      if (!eng || typeof eng.dispose !== "function") return { api: false };
      await eng.dispose();
      let rejected = false;
      try { await eng.separate({ ch0: new Float32Array(8), ch1: new Float32Array(8), len: 8 }); }
      catch (e) { rejected = /disposed/.test(String(e && e.message || e)); }
      window.__resetMarker = 1;
      return { api: true, disposed: eng.isDisposed(), workerGone: eng.worker === null, rejected };
    })()`,
    20000,
  );
  chk(
    `${name}: engine disposal terminates the worker for real`,
    !!pre && pre.api === true && pre.disposed === true && pre.workerGone === true &&
      pre.rejected === true,
    JSON.stringify(pre),
  );
  await evalL(pgX.sessionId, `document.getElementById("resetBtn").click()`, 10000);
  let fresh = null;
  for (let i = 0; i < 30; i++) {
    await sleep(1000);
    try {
      fresh = await evalL(
        pgX.sessionId,
        `(()=>({
          markerGone: typeof window.__resetMarker === "undefined",
          resetBtn: !!document.getElementById("resetBtn"),
          stemsCleared: !document.querySelector("#stems .ms-stem") && !document.querySelector("#chainTable tr td"),
        }))()`,
        8000,
      );
    } catch {
      fresh = null; // context mid-reload
    }
    if (fresh && fresh.markerGone && fresh.resetBtn) break;
  }
  chk(
    `${name}: Reset button reloads to a genuinely fresh page`,
    !!fresh && fresh.markerGone === true && fresh.resetBtn === true && fresh.stemsCleared === true,
    JSON.stringify(fresh),
  );
}
let fatalErr = null;
try {
  // Machine-readable provenance record — fail-closed: every fact must match the canonical values,
  // and the recorded bytes/SHA-256 must match the bundled file on disk.
  const prov = JSON.parse(
    readFileSync("models/music-source-separation/provenance.json", "utf8"),
  );
  const songAsset = (prov.assets || []).find((a) => a.id === "sample-song");
  const songBytes = readFileSync(
    songAsset?.localPath || "models/music-source-separation/song.mp3",
  );
  const songSha = createHash("sha256").update(songBytes).digest("hex");
  chk(
    "provenance: canonical source/API/download URLs + retrieval date + SPDX-ish license",
    !!songAsset &&
      songAsset.sourceUrl === "https://ccmixter.org/files/Loveshadow/29635" &&
      songAsset.apiUrl === "https://ccmixter.org/api/query?f=json&ids=29635" &&
      songAsset.downloadUrl ===
        "https://ccmixter.org/content/Loveshadow/Loveshadow_-_The_CC_BY_Song.mp3" &&
      /^\d{4}-\d{2}-\d{2}$/.test(songAsset.retrievedAt || "") &&
      songAsset.license === "CC-BY-3.0" &&
      songAsset.licenseUrl === "https://creativecommons.org/licenses/by/3.0/",
    songAsset ? `${songAsset.sourceUrl} | ${songAsset.license}` : "missing sample-song asset",
  );
  chk(
    "provenance: bytes + SHA-256 match the bundled file on disk",
    !!songAsset && songAsset.bytes === 4625972 && songBytes.length === songAsset.bytes &&
      songAsset.sha256 === songSha,
    `${songBytes.length} B, sha256 ${songSha.slice(0, 12)}…`,
  );
  chk(
    "provenance: creator + contributor + unmodified + first-7.8s window + local path",
    !!songAsset && songAsset.creator === "loveshadow" &&
      (songAsset.contributors || []).join(" ").includes("Victor Stone") &&
      songAsset.unmodified === true &&
      Array.isArray(songAsset.analyzedWindowSec) && songAsset.analyzedWindowSec[0] === 0 &&
      Math.abs(songAsset.analyzedWindowSec[1] - 7.8) < 0.2 &&
      songAsset.localPath === "models/music-source-separation/song.mp3",
    songAsset ? `${songAsset.creator} | ${(songAsset.contributors || []).join(",")}` : "missing",
  );

  // Route-local CREDITS.md files (overview root + one per ladder route).
  for (const route of ["", "basics", "practical", "wild", "multi-model"]) {
    const p = `models/music-source-separation/${route ? route + "/" : ""}CREDITS.md`;
    let t = "";
    try {
      t = readFileSync(p, "utf8");
    } catch { /* fail closed below */ }
    chk(
      `CREDITS.md complete (${route || "overview"})`,
      t.includes("The CC BY Song") && t.includes("loveshadow") &&
        t.includes("Victor Stone") &&
        t.includes("https://creativecommons.org/licenses/by/3.0/") &&
        t.includes("https://ccmixter.org/files/Loveshadow/29635"),
      p,
    );
  }

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

  // The deployed page names the advertised model stage (the loader config carries the literal HF id).
  const stage = await evalL(
    pg.sessionId,
    `document.documentElement.innerHTML.includes("MrCitron/demucs-v4-onnx")`,
    10000,
  );
  chk("names advertised stage MrCitron/demucs-v4-onnx", stage === true);

  // See-inside surface on the overview route: the at-a-glance table names the real pipeline task.
  const inside = await evalL(
    pg.sessionId,
    `(()=>{const t=document.querySelector(".inside-table");return !!t && /audio source separation/.test(t.textContent);})()`,
    10000,
  );
  chk("see-inside: at-a-glance table names the real task", inside === true);

  // The four use-case demos must be discoverable from the overview as real cards.
  const cards = await evalL(
    pg.sessionId,
    `JSON.stringify([...document.querySelectorAll(".model-grid a.model-card")].map(a => [
      a.getAttribute("href"),
      (a.querySelector("h3")||{}).textContent || "",
      ((a.querySelector("p")||{}).textContent || "").length > 40,
    ]))`,
    10000,
  );
  const cardArr = JSON.parse(cards || "[]");
  chk(
    "overview: the four use-case demos are discoverable as cards",
    cardArr.length === 4 &&
      ["basics/", "practical/", "wild/", "multi-model/"].every((r, i) =>
        cardArr[i] && cardArr[i][0] === r && cardArr[i][1].length > 2 && cardArr[i][2] === true
      ),
    cards,
  );
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
  chk("ready (controls enabled)", ready);

  // Bundled song integrity: the exact unmodified upload, pinned by size + SHA-256 (confirmed 2026-07-27;
  // provenance chain in CREDITS.md).
  const integrity = await evalL(
    pg.sessionId,
    `(async()=>{
      const res = await fetch("/web-ai-showcase/models/music-source-separation/song.mp3");
      const buf = await res.arrayBuffer();
      const hex = [...new Uint8Array(await crypto.subtle.digest("SHA-256", buf))].map(b=>b.toString(16).padStart(2,"0")).join("");
      return { bytes: buf.byteLength, sha256: hex };
    })()`,
    30000,
  );
  chk(
    "bundled song integrity: exact size (4,625,972 B) + SHA-256",
    integrity?.bytes === 4625972 &&
      integrity?.sha256 === "9ae38593020674f9f89879f79163031392f00a937b5332365f504be24b2e91aa",
    JSON.stringify(integrity),
  );

  // CORRECTNESS via the engine on the bundled sample song — "The CC BY Song" by loveshadow (CC BY 3.0,
  // additional lyrics by Victor Stone). The documented window is the opening 0.0-7.8 s: the song sings
  // from the first bar (measured vocals-stem RMS 0.064 at 0 s, strong throughout: 0.064-0.098).
  const rec = await evalL(
    pg.sessionId,
    `(async()=>{
      const M = await import("./sep.js");
      const eng = new M.SepEngine(); await eng.load();
      const seg = await M.songSegment(0.0);
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
  // Content separation on the real song: the SUNG vocal lands in the VOCALS stem (clearly present,
  // stronger than drums), and the 4 stems carry genuinely different
  // energy (real separation, not a pass-through). The acoustic BASS stem being loudest is musically
  // honest for this track, so vocals are not required to dominate it.
  const rmsVals = rec?.rms ? Object.values(rec.rms) : [];
  const vocalsCarried = rec?.rms && rec.rms.vocals > rec.rms.drums && rec.rms.vocals > 0.05;
  const distinct = rmsVals.length === 4 && Math.max(...rmsVals) > 2 * Math.min(...rmsVals);
  chk(
    "content-separated: sung vocal → vocals stem; stems distinct",
    vocalsCarried && distinct,
    JSON.stringify(rec?.rms),
  );

  // Attribution is visible, complete, and beside the sample control (CC BY 3.0 requires it).
  await attribCheck(pg.sessionId, "overview");

  // Drive the deployed page: click "Try the sample song" → 4 stem rows + download links render.
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
    "page renders 4 stem rows + download links + readout (sample song)",
    rows === 4 && dls === 4 && readout === true,
    `rows=${rows} dls=${dls}`,
  );

  // The upload path also works: attach the REAL bundled song MP3 to the hidden file input via CDP,
  // fire change, and expect the stems to re-render without an error status.
  const doc = await cdp.send("DOM.getDocument", {}, pg.sessionId, 10000);
  const fileNode = await cdp.send(
    "DOM.querySelector",
    { nodeId: doc.root.nodeId, selector: "#file" },
    pg.sessionId,
    10000,
  );
  await cdp.send(
    "DOM.setFileInputFiles",
    { files: [UPLOAD_FILE], nodeId: fileNode.nodeId },
    pg.sessionId,
    10000,
  );
  await evalL(
    pg.sessionId,
    `document.getElementById("file").dispatchEvent(new Event("change"))`,
    10000,
  );
  let upRows = 0;
  for (let i = 0; i < 60; i++) {
    await sleep(2000);
    upRows =
      await evalL(pg.sessionId, `document.querySelectorAll("#stems .ms-stem").length`, 8000) ||
      0;
    const busy = await evalL(
      pg.sessionId,
      `/Separating/.test((document.getElementById("runStatus")||{}).textContent||"")`,
      8000,
    );
    if (upRows === 4 && !busy) break;
  }
  const upErr = await evalL(
    pg.sessionId,
    `document.getElementById("runStatus").classList.contains("err")`,
    8000,
  );
  chk("upload path separates an uploaded file", upRows === 4 && upErr === false, `rows=${upRows}`);

  // Karaoke preset unchecks the vocals stem.
  const karaoke = await evalL(
    pg.sessionId,
    `(()=>{document.getElementById("presetKaraoke").click();const v=[...document.querySelectorAll("#stems .ms-stem")].find(r=>r.dataset.name==="vocals");return v&&!v.querySelector("input").checked;})()`,
    8000,
  );
  chk("Karaoke preset mutes the vocals stem", karaoke === true);

  // responsive
  await setViewport(cdp, pg.sessionId, DESKTOP);
  await sleep(400);
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
  await resetCheck(pg, "overview");
  await closePage(cdp, pg.targetId);

  // ── LADDER ROUTES ── basics / practical / wild / multi-model: every route is opened for real, its
  // visible controls are exercised, and overflow is checked at DESKTOP and MOBILE. The Demucs model is
  // already in this profile's cache from the overview run; the AST tagger downloads once (~90 MB).
  const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
  const overflow = async (pg2, label) => {
    await setViewport(cdp, pg2.sessionId, DESKTOP);
    await sleep(400);
    const d = await evalL(
      pg2.sessionId,
      `document.documentElement.scrollWidth <= window.innerWidth + 1`,
      8000,
    );
    chk(`${label}: no horizontal overflow (desktop)`, d === true);
    await setViewport(cdp, pg2.sessionId, MOBILE);
    await sleep(400);
    const m = await evalL(
      pg2.sessionId,
      `document.documentElement.scrollWidth <= window.innerWidth + 1`,
      8000,
    );
    chk(`${label}: no horizontal overflow (mobile)`, m === true);
    await setViewport(cdp, pg2.sessionId, DESKTOP);
    await sleep(300);
  };
  const clickDownloadIfAny = async (pg2, idx = 0) => {
    await evalL(
      pg2.sessionId,
      `(()=>{const l=[...document.querySelectorAll(".model-loader")][${idx}];const b=l&&l.querySelector(".loader-actions button");if(b&&/Download/.test(b.textContent)&&!b.disabled){b.click();return true;}return false;})()`,
      15000,
    );
  };
  const waitFor = async (pg2, expr, tries = 90) => {
    for (let i = 0; i < tries; i++) {
      const v = await evalL(pg2.sessionId, expr, 10000);
      if (v) return v;
      await sleep(2000);
    }
    return null;
  };

  // BASICS — one button → four playable stems.
  {
    const pg2 = await openPage(cdp, base + "models/music-source-separation/basics/");
    await sleep(1400);
    await clickDownloadIfAny(pg2);
    const rdy = await waitFor(pg2, `!document.getElementById("sampleBtn").disabled`);
    chk("basics: ready (model from cache)", !!rdy);
    await evalL(pg2.sessionId, `document.getElementById("sampleBtn").click()`, 10000);
    const rows = await waitFor(
      pg2,
      `document.querySelectorAll("#stems .ms-stem").length === 4 && !document.getElementById("result").hidden`,
      60,
    );
    chk("basics: 4 playable stem rows render", !!rows);
    const play = await evalL(
      pg2.sessionId,
      `(()=>{const b=document.querySelector("#stems .ms-stem button");b.click();return new Promise(r=>setTimeout(()=>r(b.textContent),800));})()`,
      15000,
    );
    chk("basics: a stem's play button actually plays it", play === "■", String(play));
    await evalL(
      pg2.sessionId,
      `(()=>{const b=document.querySelector("#stems .ms-stem button");if(b.textContent==="■")b.click();return 1;})()`,
      8000,
    );
    // Microscope checks: the displayed measurements must be REAL — the vocals RMS shown on the card is
    // recomputed here from the actual stem waveform (window.__lastStems), failing closed on authored text.
    const stats = await evalL(
      pg2.sessionId,
      `(()=>{
        const rows = [...document.querySelectorAll("#stems .ms-stem")];
        const stems = window.__lastStems || [];
        const rms = (s) => { let a = 0; for (let i = 0; i < s.l.length; i++) a += s.l[i]*s.l[i] + s.r[i]*s.r[i]; return Math.sqrt(a / (2*s.l.length)); };
        const texts = rows.map(r => (r.querySelector(".ms-stat") || {}).textContent || "");
        const v = stems.find(s => s.name === "vocals");
        const vRow = rows.find(r => r.dataset.name === "vocals");
        const m = vRow && ((vRow.querySelector(".ms-stat").textContent).match(/RMS ([0-9.]+)/) || [])[1];
        return {
          n: texts.length,
          allRms: texts.every(t => /RMS [0-9.]+/.test(t)),
          allPeak: texts.every(t => /peak (-?[0-9.]+|-∞) dB/.test(t)),
          allShare: texts.every(t => /[0-9]+% of the window's energy/.test(t)),
          allTiming: texts.every(t => /window [0-9]+\\.[0-9]+–[0-9]+\\.[0-9]+ s of the 2:22 song/.test(t)),
          vocalsShown: m, vocalsActual: v ? rms(v).toFixed(3) : null,
          vocalsMatch: !!(v && m && Math.abs(parseFloat(m) - rms(v)) < 0.001),
        };
      })()`,
      10000,
    );
    chk(
      "basics: stem cards show REAL measured energy + timing (vocals RMS recomputed from stem data)",
      !!stats && stats.n === 4 && stats.allRms && stats.allPeak && stats.allShare &&
        stats.allTiming &&
        stats.vocalsMatch,
      JSON.stringify(stats),
    );
    const why = await evalL(
      pg2.sessionId,
      `(()=>{
        const q = (n) => ((document.querySelector("#stems .ms-stem[data-name='" + n + "'] .ms-why") || {}).textContent || "");
        return {
          drums: /drum kit|percussion/.test(q("drums")),
          bass: /low melodic foundation|bass guitar|synth bass/.test(q("bass")),
          other: /catch-all|everything harmonic/.test(q("other")),
          vocals: /SUNG|singing/.test(q("vocals")),
          prefix: [...document.querySelectorAll("#stems .ms-why")].every(el => el.textContent.includes("What the model assigned here:")),
        };
      })()`,
      8000,
    );
    chk(
      "basics: each stem explains what the model assigned",
      !!why && why.drums && why.bass && why.other && why.vocals && why.prefix,
      JSON.stringify(why),
    );
    const dls = await evalL(
      pg2.sessionId,
      `JSON.stringify([...document.querySelectorAll("#stems .ms-stem a.ms-dl")].map(a => [a.download, a.href.startsWith("blob:")]))`,
      8000,
    );
    const dlArr = JSON.parse(dls || "[]");
    chk(
      "basics: every stem has a real WAV download (blob URL + .wav name)",
      dlArr.length === 4 && dlArr.every(([nm, isBlob]) => /\.wav$/.test(nm) && isBlob === true),
      dls,
    );
    // Example 2 (reconstruction check): the original-vs-sum A/B is real, and the displayed residual
    // is recomputed independently from the original segment + the returned stems (never authored text).
    const recon = await evalL(
      pg2.sessionId,
      `(()=>{
        const seg = window.__segment, stems = window.__lastStems;
        if (!seg || !stems) return null;
        const len = seg.l.length;
        const sl = new Float32Array(len), sr = new Float32Array(len);
        for (const s of stems) {
          for (let i = 0; i < len; i++) {
            sl[i] += s.l[i];
            sr[i] += s.r[i];
          }
        }
        let sum = 0;
        for (let i = 0; i < len; i++) sum += (seg.l[i] - sl[i]) ** 2 + (seg.r[i] - sr[i]) ** 2;
        const cap = (document.getElementById("reconCap")||{}).textContent || "";
        const m = cap.match(/residual RMS ([0-9.]+)/);
        return { recomputed: Math.sqrt(sum / (2 * len)), shown: m ? Number(m[1]) : null, cap };
      })()`,
      15000,
    );
    chk(
      "basics example 2 (reconstruction check): measured residual shown + independently recomputed",
      !!recon && recon.recomputed > 0 && recon.shown !== null &&
        Math.abs(recon.shown - recon.recomputed) < 5e-4,
      JSON.stringify(recon),
    );
    await evalL(pg2.sessionId, `document.getElementById("sumBtn").click()`, 8000);
    await sleep(500);
    const sumPlayed = await evalL(
      pg2.sessionId,
      `JSON.stringify(window.__lastPlayed || null)`,
      5000,
    );
    chk(
      "basics example 2: sum-of-stems playback is real",
      sumPlayed === '["__sum"]',
      String(sumPlayed),
    );

    await overflow(pg2, "basics");
    await attribCheck(pg2.sessionId, "basics");
    chk("basics: no console errors", pg2.errors.length === 0, pg2.errors.slice(0, 2).join(" | "));
    await resetCheck(pg2, "basics");
    await closePage(cdp, pg2.targetId);
  }

  // PRACTICAL — karaoke / acapella / stem WAV exports.
  {
    const pg2 = await openPage(cdp, base + "models/music-source-separation/practical/");
    await sleep(1400);
    await clickDownloadIfAny(pg2);
    const rdy = await waitFor(pg2, `!document.getElementById("sampleBtn").disabled`);
    chk("practical: ready (model from cache)", !!rdy);
    await evalL(pg2.sessionId, `document.getElementById("sampleBtn").click()`, 10000);
    const ex = await waitFor(
      pg2,
      `document.querySelectorAll("#exports a.ms-dl").length === 6 && !document.getElementById("result").hidden`,
      60,
    );
    chk("practical: karaoke + acapella + 4 stem downloads render", !!ex);
    const wavs = await evalL(
      pg2.sessionId,
      `(async()=>{const out=[];for(const a of document.querySelectorAll("#exports a.ms-dl")){const r=await fetch(a.href);const b=await r.arrayBuffer();out.push({name:a.download,bytes:b.byteLength,riff:String.fromCharCode(...new Uint8Array(b.slice(0,4)))});}return out;})()`,
      30000,
    );
    chk(
      "practical: every export is a real WAV blob (RIFF, >100 KB)",
      !!wavs && wavs.length === 6 && wavs.every((w) => w.bytes > 100000 && w.riff === "RIFF"),
      JSON.stringify(wavs?.map((w) => w.name + ":" + w.bytes)),
    );
    // Karaoke-maker compare player: it must PLAY, and the A/B modes must flip the REAL per-stem
    // GainNodes (the same mechanism the wild decks use) — verified from the mixer's live gain state.
    await evalL(pg2.sessionId, `document.getElementById("abPlay").click()`, 8000);
    const playing = await waitFor(pg2, `window.__mixer && window.__mixer.playing === true`, 15);
    chk("practical: the compare player actually plays the separated window", !!playing);
    await evalL(pg2.sessionId, `document.getElementById("abBacking").click()`, 8000);
    const backing = await waitFor(
      pg2,
      `window.__mixer.gains.vocals && window.__mixer.gains.vocals.gain.value < 0.1 && window.__mixer.gains.drums.gain.value > 0.9`,
      15,
    );
    chk(
      "practical: backing mode removes the vocal for real (vocals GainNode → 0, band stays)",
      !!backing,
    );
    await evalL(pg2.sessionId, `document.getElementById("abVocal").click()`, 8000);
    const vocal = await waitFor(
      pg2,
      `window.__mixer.gains.drums && window.__mixer.gains.drums.gain.value < 0.1 && window.__mixer.gains.vocals.gain.value > 0.9`,
      15,
    );
    chk(
      "practical: vocal mode isolates the acapella for real (band GainNodes → 0)",
      !!vocal,
    );
    await evalL(
      pg2.sessionId,
      `(()=>{const b=document.getElementById("abPlay");if(/Stop/.test(b.textContent))b.click();return 1;})()`,
      8000,
    );
    // The energy captions must be REAL — recomputed here from the actual separated stems.
    const caps = await evalL(
      pg2.sessionId,
      `(()=>{
        const stems = window.__lastStems || [];
        const rms = (l, r) => { let a = 0; for (let i = 0; i < l.length; i++) a += l[i]*l[i] + r[i]*r[i]; return Math.sqrt(a / (2*l.length)); };
        const stem = (n) => stems.find(s => s.name === n);
        const kl = new Float32Array(stem("drums").l.length), kr = new Float32Array(kl.length);
        for (const n of ["drums", "bass", "other"]) {
          for (let i = 0; i < kl.length; i++) { kl[i] += stem(n).l[i]; kr[i] += stem(n).r[i]; }
        }
        const kActual = rms(kl, kr), vActual = rms(stem("vocals").l, stem("vocals").r);
        const kShown = ((document.getElementById("statBacking").textContent).match(/RMS ([0-9.]+)/) || [])[1];
        const vShown = ((document.getElementById("statVocal").textContent).match(/RMS ([0-9.]+)/) || [])[1];
        return {
          kShown, kActual: kActual.toFixed(3), vShown, vActual: vActual.toFixed(3),
          kMatch: !!(kShown && Math.abs(parseFloat(kShown) - kActual) < 0.001),
          vMatch: !!(vShown && Math.abs(parseFloat(vShown) - vActual) < 0.001),
          karaokeClaim: /drums \\+ bass \\+ other/.test(document.getElementById("statBacking").textContent),
          removesClaim: /the part karaoke removes/.test(document.getElementById("statVocal").textContent),
        };
      })()`,
      10000,
    );
    chk(
      "practical: backing/vocal energy captions recomputed from the real stems",
      !!caps && caps.kMatch && caps.vMatch && caps.karaokeClaim && caps.removesClaim,
      JSON.stringify(caps),
    );
    // Example 2 (practice mode): the speed control drives the real playbackRate of the solo stem.
    await evalL(
      pg2.sessionId,
      `document.querySelector('#pracRate [data-rate="0.5"]').click()`,
      8000,
    );
    await evalL(pg2.sessionId, `document.getElementById("pracBtn").click()`, 8000);
    await sleep(600);
    const prac = await evalL(
      pg2.sessionId,
      `JSON.stringify({ state: window.__practice || null, btn: (document.getElementById("pracBtn")||{}).textContent || "" })`,
      8000,
    );
    const pracObj = typeof prac === "string" ? JSON.parse(prac) : {};
    chk(
      "practical example 2 (practice mode): rate control drives real slowed playback of the solo stem",
      pracObj.state && pracObj.state.stem === "vocals" && pracObj.state.rate === 0.5 &&
        pracObj.btn.includes("Stop"),
      String(prac),
    );
    await evalL(pg2.sessionId, `document.getElementById("pracBtn").click()`, 8000); // stop
    await sleep(300);

    await overflow(pg2, "practical");
    await attribCheck(pg2.sessionId, "practical");
    chk(
      "practical: no console errors",
      pg2.errors.length === 0,
      pg2.errors.slice(0, 2).join(" | "),
    );
    await resetCheck(pg2, "practical");
    await closePage(cdp, pg2.targetId);
  }

  // WILD — live stem-DJ pads riding real GainNodes.
  {
    const pg2 = await openPage(cdp, base + "models/music-source-separation/wild/");
    await sleep(1400);
    await clickDownloadIfAny(pg2);
    const rdy = await waitFor(pg2, `!document.getElementById("sampleBtn").disabled`);
    chk("wild: ready (model from cache)", !!rdy);
    await evalL(pg2.sessionId, `document.getElementById("sampleBtn").click()`, 10000);
    const pads = await waitFor(
      pg2,
      `document.querySelectorAll("#pads .pad").length === 4 && !document.getElementById("result").hidden`,
      60,
    );
    chk("wild: 4 live pads render", !!pads);
    await evalL(pg2.sessionId, `document.getElementById("playBtn").click()`, 10000);
    const playing = await waitFor(pg2, `window.__mixer && window.__mixer.playing === true`, 15);
    chk("wild: the loop plays through per-stem GainNodes", !!playing);
    await evalL(
      pg2.sessionId,
      `[...document.querySelectorAll(".pad")].find(p=>p.dataset.name==="vocals").click()`,
      8000,
    );
    const gainDown = await waitFor(
      pg2,
      `window.__mixer.gains.vocals && window.__mixer.gains.vocals.gain.value < 0.1`,
      15,
    );
    const pressed = await evalL(
      pg2.sessionId,
      `[...document.querySelectorAll(".pad")].find(p=>p.dataset.name==="vocals").getAttribute("aria-pressed")`,
      8000,
    );
    chk(
      "wild: vocals pad mutes the live stem for real (GainNode → 0)",
      !!gainDown && pressed === "false",
      `pressed=${pressed}`,
    );
    await evalL(pg2.sessionId, `document.getElementById("soloBtn").click()`, 8000);
    const solo = await evalL(
      pg2.sessionId,
      `JSON.stringify([...document.querySelectorAll(".pad")].map(p=>[p.dataset.name,p.getAttribute("aria-pressed")]))`,
      8000,
    );
    chk(
      "wild: solo-vocals leaves only the vocals pad on",
      solo ===
        JSON.stringify([["drums", "false"], ["bass", "false"], ["other", "false"], [
          "vocals",
          "true",
        ]]),
      solo,
    );
    // Remix deck: the faders must drive the REAL GainNodes (no decorative sliders).
    await evalL(pg2.sessionId, `document.getElementById("presetFlat").click()`, 8000);
    await waitFor(
      pg2,
      `window.__mixer.gains.drums && window.__mixer.gains.drums.gain.value > 0.9`,
      15,
    );
    await evalL(
      pg2.sessionId,
      `(()=>{const f=document.querySelector(".ms-gain[data-name='drums']");f.value="0.35";f.dispatchEvent(new Event("input"));return 1;})()`,
      8000,
    );
    const fader = await waitFor(
      pg2,
      `window.__mixer.gains.drums.gain.value > 0.3 && window.__mixer.gains.drums.gain.value < 0.4 && window.__mixer.gains.bass.gain.value > 0.9`,
      15,
    );
    const faderLabel = await evalL(
      pg2.sessionId,
      `(document.querySelector(".ms-gain[data-name='drums'] + .ms-gainval")||{}).textContent`,
      8000,
    );
    chk(
      "wild: gain faders drive the REAL GainNodes (not decorative)",
      !!fader && faderLabel === "35%",
      JSON.stringify({ gainMoved: !!fader, faderLabel }),
    );
    // Presets must set BOTH the fader positions and the gains — including a boost ABOVE unity (which a
    // mute-only deck could never produce).
    await evalL(pg2.sessionId, `document.getElementById("presetClub").click()`, 8000);
    const club = await waitFor(
      pg2,
      `window.__mixer.gains.drums.gain.value > 1.1 && window.__mixer.gains.other.gain.value < 0.85 && window.__mixer.gains.other.gain.value > 0.65`,
      15,
    );
    const clubFader = await evalL(
      pg2.sessionId,
      `(document.querySelector(".ms-gain[data-name='drums']")||{}).value`,
      8000,
    );
    await evalL(pg2.sessionId, `document.getElementById("presetKaraoke").click()`, 8000);
    const pk = await waitFor(
      pg2,
      `window.__mixer.gains.vocals.gain.value < 0.1 && window.__mixer.gains.drums.gain.value > 0.9`,
      15,
    );
    chk(
      "wild: presets set faders AND gains for real (club boost > unity; karaoke kills vocals)",
      !!club && clubFader === "1.25" && !!pk,
      JSON.stringify({ clubBoost: !!club, clubFader, karaoke: !!pk }),
    );
    // The downloaded remix must be rendered from the CURRENT deck state: set a known state (drums 50%,
    // vocals muted, rest unity), render, then independently parse the WAV and recompute the expected
    // mix from the real stems.
    await evalL(pg2.sessionId, `document.getElementById("presetFlat").click()`, 8000);
    await evalL(
      pg2.sessionId,
      `(()=>{const f=document.querySelector(".ms-gain[data-name='drums']");f.value="0.5";f.dispatchEvent(new Event("input"));return 1;})()`,
      8000,
    );
    await evalL(
      pg2.sessionId,
      `(()=>{const p=[...document.querySelectorAll("#pads .pad")].find(x=>x.dataset.name==="vocals");if(p.getAttribute("aria-pressed")==="true")p.click();return 1;})()`,
      8000,
    );
    await waitFor(pg2, `window.__mixer.gains.vocals.gain.value < 0.1`, 15);
    await evalL(pg2.sessionId, `document.getElementById("mixDlBtn").click()`, 8000);
    const mix = await evalL(
      pg2.sessionId,
      `(async()=>{
        const res = await fetch(window.__lastMixUrl);
        const buf = await res.arrayBuffer();
        const v = new DataView(buf);
        const tag = String.fromCharCode(v.getUint8(0),v.getUint8(1),v.getUint8(2),v.getUint8(3)) + String.fromCharCode(v.getUint8(8),v.getUint8(9),v.getUint8(10),v.getUint8(11));
        const n = (buf.byteLength - 44) / 4;
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const l = v.getInt16(44 + i*4, true) / 32768, r = v.getInt16(44 + i*4 + 2, true) / 32768;
          sum += l*l + r*r;
        }
        const got = Math.sqrt(sum / (2*n));
        const stems = window.__lastStems || [];
        const gains = { drums: 0.5, bass: 1, other: 1, vocals: 0 };
        const len = stems[0].l.length;
        let es = 0;
        for (let i = 0; i < len; i++) {
          let l = 0, r = 0;
          for (const s of stems) { l += s.l[i] * gains[s.name]; r += s.r[i] * gains[s.name]; }
          es += l*l + r*r;
        }
        const want = Math.sqrt(es / (2*len));
        return { tag, bytes: buf.byteLength, got, want, match: Math.abs(got - want) / want < 0.02 };
      })()`,
      20000,
    );
    chk(
      "wild: the downloaded remix WAV reflects the CURRENT fader/pad state (recomputed)",
      !!mix && mix.tag === "RIFFWAVE" && mix.bytes > 1300000 && mix.match === true,
      JSON.stringify(mix),
    );
    await evalL(pg2.sessionId, `document.getElementById("playBtn").click()`, 8000);
    // Example 2 (auto-duck): enabling it during playback measurably pulls the backing gains down —
    // a real envelope follower on the vocals stem (the vocal is loud in this window, so duck < 1).
    // Reset to the full mix first: earlier fader/preset checks may have left the vocals stem down,
    // and the follower taps the vocals gain (a muted vocal is honestly silent).
    await evalL(pg2.sessionId, `document.getElementById("presetFlat").click()`, 8000);
    await evalL(pg2.sessionId, `document.getElementById("playBtn").click()`, 8000);
    await sleep(700);
    await evalL(pg2.sessionId, `document.getElementById("duckBtn").click()`, 8000);
    // Poll one full 7.8 s loop: the window's opening is quiet, so the follower bites when the vocal
    // enters — sampling a single fixed instant would be timing luck.
    let duckObj = {};
    for (let k = 0; k < 24; k++) {
      await sleep(600);
      const duck = await evalL(
        pg2.sessionId,
        `JSON.stringify({ duck: window.__duck || null, live: (document.getElementById("duckLive")||{}).textContent || "" })`,
        8000,
      );
      duckObj = typeof duck === "string" ? JSON.parse(duck) : {};
      if (duckObj.duck && duckObj.duck.current < 0.9) break;
    }
    chk(
      "wild example 2 (auto-duck): the vocals envelope measurably ducks the backing, live",
      !!duckObj.duck && duckObj.duck.on === true && duckObj.duck.current < 0.9 &&
        duckObj.duck.ticks > 10 && /duck ×0\./.test(duckObj.live || ""),
      JSON.stringify(duckObj),
    );
    await evalL(pg2.sessionId, `document.getElementById("duckBtn").click()`, 8000); // off
    await evalL(pg2.sessionId, `document.getElementById("playBtn").click()`, 8000); // stop the loop
    await sleep(400);

    await overflow(pg2, "wild");
    await attribCheck(pg2.sessionId, "wild");
    chk("wild: no console errors", pg2.errors.length === 0, pg2.errors.slice(0, 2).join(" | "));
    await resetCheck(pg2, "wild");
    await closePage(cdp, pg2.targetId);
  }

  // MULTI-MODEL — Demucs vocals stem → Whisper tiny English ASR (vocals → lyrics).
  {
    const pg2 = await openPage(cdp, base + "models/music-source-separation/multi-model/");
    await sleep(1400);
    // NEVER auto-download: on arrival the whisper loader must show an explicit Download button, and NO
    // whisper weights may have been fetched before any click. (Demucs is already in this profile's cache
    // from the overview run, so it may auto-init from cache — no network. Whisper is fresh.)
    const auto = await evalL(
      pg2.sessionId,
      `(()=>{
        const loader2 = (document.getElementById("model-loader-2")||{}).textContent || "";
        const whisperFetches = performance.getEntriesByType("resource")
          .filter(e => /whisper-tiny/i.test(e.name)).length;
        return { whisperDownloadShown: /Download/i.test(loader2), whisperFetches };
      })()`,
      10000,
    );
    chk(
      "multi-model: whisper never auto-downloads (Download shown, 0 whisper fetches before click)",
      !!auto && auto.whisperDownloadShown === true && auto.whisperFetches === 0,
      JSON.stringify(auto),
    );
    const stage = await evalL(
      pg2.sessionId,
      `document.body.textContent.includes("onnx-community/whisper-tiny.en") && document.body.textContent.includes("MrCitron/demucs-v4-onnx")`,
      10000,
    );
    chk(
      "multi-model: names advertised stages MrCitron/demucs-v4-onnx + onnx-community/whisper-tiny.en",
      stage === true,
    );
    await clickDownloadIfAny(pg2, 0); // Demucs (auto-inits from this profile's cache)
    await clickDownloadIfAny(pg2, 1); // Whisper (one-time ~40 MB download)
    const rdy = await waitFor(pg2, `!document.getElementById("runBtn").disabled`, 120);
    chk("multi-model: both models ready (explicit downloads only)", !!rdy);
    await evalL(pg2.sessionId, `document.getElementById("runBtn").click()`, 10000);
    const chainReady = await waitFor(pg2, `!document.getElementById("result").hidden`, 90);
    chk("multi-model: the chain ran (separate → transcribe)", !!chainReady);
    const chain = await evalL(
      pg2.sessionId,
      `(()=>{
        const ta = document.getElementById("transcript");
        return {
          editable: !!ta && !ta.readOnly && !ta.disabled && ta.tagName === "TEXTAREA",
          transcript: ta ? ta.value : null,
          sepMs: /[0-9]/.test(document.getElementById("rSepMs").textContent),
          asrMs: /[0-9]/.test(document.getElementById("rAsrMs").textContent),
          words: document.getElementById("rWords").textContent,
          mixVerdict: document.getElementById("mixVerdict").textContent,
          verdict: document.getElementById("verdictText").textContent,
          vocalPlay: !!document.getElementById("vocalPlay"),
        };
      })()`,
      10000,
    );
    chk(
      "multi-model: editable transcript + real timings + honest verdict shown",
      !!chain && chain.editable === true && typeof chain.transcript === "string" &&
        chain.sepMs && chain.asrMs && chain.vocalPlay &&
        /no words|lossier|speech model|singing/i.test(chain.verdict) &&
        (chain.transcript.trim() === ""
          ? /also hears no words/.test(chain.mixVerdict) && chain.words === "0"
          : chain.words !== "0"),
      JSON.stringify({
        transcript: chain?.transcript,
        words: chain?.words,
        verdict: chain?.verdict?.slice(0, 100),
        mixVerdict: chain?.mixVerdict?.slice(0, 80),
      }),
    );
    // Anti-stub proof: the SAME ASR worker must transcribe real speech — an always-empty stub fails.
    // The speech clip is the FAMILY-LOCAL neutral ASR validation fixture
    // models/music-source-separation/fixtures/libri-61-70968-0000-16k-mono.wav: LibriSpeech (OpenSLR
    // SLR12) test-clean utterance 61-70968-0000 (reader 61, Paul-Gabriel Wiener; LibriVox 'Robin Hood'
    // chapter 4), CC BY 4.0, derived to 16 kHz mono WAV; reference transcript "HE BEGAN A CONFUSED
    // COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT". Full source /
    // creator / license / retrieval / derivation + SHA-256 recorded in provenance.json + CREDITS.md.
    const health = await evalL(
      pg2.sessionId,
      `(async()=>{
        const buf = await (await fetch("/web-ai-showcase/models/music-source-separation/fixtures/libri-61-70968-0000-16k-mono.wav")).arrayBuffer();
        const dec = await new AudioContext().decodeAudioData(buf.slice(0));
        const off = new OfflineAudioContext(1, Math.ceil(dec.duration * 16000), 16000);
        const s = off.createBufferSource(); s.buffer = dec; s.connect(off.destination); s.start();
        const pcm = (await off.startRendering()).getChannelData(0);
        const r = await window.__asr.transcribe(pcm);
        return { text: (r.text || "").slice(0, 120), ms: r.ms };
      })()`,
      120000,
    );
    chk(
      "multi-model: the ASR chain genuinely works (same worker transcribes real LibriSpeech speech)",
      !!health && /wizard|vanished|curtain|confused|complaint/i.test(health.text || ""),
      JSON.stringify(health),
    );

    // Example 2 (instrument specialist): explicit-only download, then real measured verdicts.
    const instPre = await evalL(
      pg2.sessionId,
      `(()=>{
        const loader = (document.getElementById("model-loader-inst")||{}).textContent || "";
        const fetches = performance.getEntriesByType("resource")
          .filter(e => /musical-instrument/i.test(e.name)).length;
        return { downloadShown: /Download/i.test(loader), fetches };
      })()`,
      10000,
    );
    chk(
      "multi-model example 2 (specialist): never auto-downloads (Download shown, 0 fetches before click)",
      !!instPre && instPre.downloadShown === true && instPre.fetches === 0,
      JSON.stringify(instPre),
    );
    await evalL(
      pg2.sessionId,
      `document.querySelector("#model-loader-inst button").click()`,
      8000,
    );
    const instReady = await waitFor(
      pg2,
      `document.getElementById("run2Btn").disabled === false`,
      120,
    );
    chk("multi-model example 2 (specialist): model ready after explicit download", !!instReady);
    await evalL(pg2.sessionId, `document.getElementById("run2Btn").click()`, 8000);
    const xv = await waitFor(
      pg2,
      `Array.isArray(window.__xverdicts) && window.__xverdicts.length === 4`,
      60,
    );
    const xvData = await evalL(pg2.sessionId, `JSON.stringify(window.__xverdicts || [])`, 8000);
    const xva = typeof xvData === "string" ? JSON.parse(xvData) : [];
    const xdrums = xva.find((x) => x.name === "drums");
    const xvox = xva.find((x) => x.name === "vocals");
    chk(
      "multi-model example 2 (specialist): real measured verdicts - drums confirmed, vocals honestly N/A",
      !!xv && !!xdrums && xdrums.cls === "ok" && /hi.?hats|drums/i.test(xdrums.top || "") &&
        !!xvox && xvox.cls === "na",
      String(xvData),
    );

    await overflow(pg2, "multi-model");
    await attribCheck(pg2.sessionId, "multi-model");
    chk(
      "multi-model: no console errors",
      pg2.errors.length === 0,
      pg2.errors.slice(0, 2).join(" | "),
    );
    await resetCheck(pg2, "multi-model");
    await closePage(cdp, pg2.targetId);
  }
} catch (e) {
  fatalErr = e;
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  if (fatalErr) {
    console.log(
      `FATAL  run aborted before completing all sections: ${fatalErr?.stack || fatalErr}`,
    );
  }
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(!fatalErr && pass === total ? 0 : 1);
}
