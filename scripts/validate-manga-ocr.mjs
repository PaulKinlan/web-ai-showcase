// Manga OCR end-to-end validation (real inference in headless Chrome). ~117 MB download.
// Verifies: loads → Download → per-file progress → ready; each bundled sample is recognised by the
// REAL model (encoder uint8 + decoder q8, WASM worker) with per-character probabilities streamed into
// the "see inside" trace; release-from-memory → reload from cache works; no console errors; no
// horizontal overflow desktop + mobile.
//
// Modes:
//   node scripts/validate-manga-ocr.mjs           → full assertion suite (exit 1 on any FAIL)
//   node scripts/validate-manga-ocr.mjs --report  → print true outputs, skip text assertions
const B = "./browser.mjs";
const { closePage, launchChrome, openPage, setViewport, startServer, MOBILE, CDP } = await import(B);
const REPORT = process.argv.includes("--report");
// Clean up stale harness Chromes/profiles from interrupted runs (they lock the profile dir).
const { execFileSync } = await import("node:child_process");
try { execFileSync("pkill", ["-f", "conformance-chrome-profile"]); } catch { /* none */ }
try {
  const { rmSync } = await import("node:fs");
  rmSync(new URL("../.conformance-chrome-profile", import.meta.url), { recursive: true, force: true });
} catch { /* ignore */ }
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const evalL = (sid, expr, ms = 240000) =>
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
const note = (n, d) => console.log(`NOTE  ${n}${d ? " — " + d : ""}`);

// The bundled samples are first-party renders (Noto Sans CJK, OFL) with known ground truth.
const SAMPLES = [
  { src: "sample-horizontal.png", want: "吾輩は猫である" },
  { src: "sample-vertical.png", want: "縦書きの文章も読めます" },
  { src: "sample-bubble.png", want: "これは複数行の吹き出しです。まとめて読みます。" },
  { src: "sample-menu.png", want: "ラーメン８５０円餃子４００円炒飯７００円" },
  { src: "sample-noisy.png", want: "少し汚れた印刷でも読めます" },
  { src: "sample-sentence.png", want: "ブラウザの中だけで日本語を読み取ります" },
  { src: "sample-vbubble.png", want: "縦書きの吹き出しも一度に読む" },
];

async function runSample(sid, src) {
  const kicked = await evalL(
    sid,
    `(()=>{const b=document.querySelector('#samples .sample-thumb[data-src="${src}"]');b.click();
      const run=document.getElementById("run");if(run.disabled)return false;run.click();return true;})()`,
    15000,
  );
  if (!kicked) return { out: "", err: "run disabled (model not ready)" };
  for (let i = 0; i < 90; i++) {
    await sleep(2000);
    const st = await evalL(
      sid,
      `({busy:!document.getElementById("cancel").disabled, out:document.getElementById("out").textContent,
         err:!document.getElementById("runStatus").hidden && document.getElementById("runStatus").classList.contains("err")
           ? document.getElementById("runStatus").textContent : null})`,
      15000,
    );
    if (st?.err) return { out: "", err: st.err };
    if (st && !st.busy && st.out !== "") return { out: st.out };
  }
  return { out: "", err: "timeout" };
}

try {
  const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/models/manga-ocr/`);
  await sleep(1500);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"),
      state:document.querySelector(".model-loader")?.dataset.state,
      dl:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent))}))()`,
    15000,
  );
  chk("loads: loader + Download (fresh profile = absent)", s0?.loader && s0?.dl, JSON.stringify(s0));

  await evalL(
    pg.sessionId,
    `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
    15000,
  );
  // Load with up to 3 attempts: a CDN stall surfaces as an honest error + Retry (worker watchdog).
  let ready = false, loadErr = null;
  for (let attempt = 0; attempt < 3 && !ready; attempt++) {
    if (attempt > 0) {
      console.log(`  retry attempt ${attempt + 1} after: ${loadErr}`);
      await evalL(
        pg.sessionId,
        `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Retry|Download/.test(x.textContent));if(b)b.click();return !!b;})()`,
        15000,
      );
    }
    for (let i = 0; i < 120; i++) {
      const st = await evalL(
        pg.sessionId,
        `({state:document.querySelector(".model-loader")?.dataset.state,
          status:document.querySelector(".model-loader .status")?.textContent,
          ready:!document.getElementById("run").disabled})`,
        15000,
      );
      if (st?.state === "error") { loadErr = st.status; break; }
      if (st?.state === "ready" && st?.ready) { ready = true; break; }
      await sleep(2000);
    }
  }
  chk("download ~117 MB → ready, controls enabled", ready, loadErr || undefined);

  // first sample is preloaded; run each sample through the real model
  const results = {};
  for (const s of SAMPLES) {
    const r = await runSample(pg.sessionId, s.src);
    results[s.src] = r;
    if (REPORT) note(s.src, JSON.stringify(r));
    else if (r.err) chk(`recognise ${s.src}`, false, r.err);
    else {
      const norm = (t) => t.replace(/\s+/g, "");
      const want = norm(s.want);
      chk(
        `recognise ${s.src}`,
        norm(r.out) === want,
        norm(r.out) === want ? "" : `got "${r.out}" want "${want}"`,
      );
    }
  }

  if (!REPORT) {
    // see-inside trace has per-character probability chips after a run
    const trace = await evalL(
      pg.sessionId,
      `({chips:document.querySelectorAll("#trace .tok").length,
         title:(document.querySelector("#trace .tok")||{}).title || ""})`,
      10000,
    );
    chk("see-inside: per-char probability chips", trace?.chips > 3, JSON.stringify(trace));

    // raw toggle flips between post-processed and raw output
    const rawToggle = await evalL(
      pg.sessionId,
      `(()=>{const b=document.getElementById("rawBtn");if(b.disabled)return {skip:true};
        const before=document.getElementById("out").textContent;b.click();
        const raw=document.getElementById("out").textContent;b.click();
        return {before, raw, back:document.getElementById("out").textContent};})()`,
      10000,
    );
    chk(
      "raw model output toggle",
      rawToggle?.skip || (rawToggle?.back === rawToggle?.before),
      JSON.stringify(rawToggle).slice(0, 120),
    );

    // release from memory → released state → load again (from cache) → ready
    await evalL(
      pg.sessionId,
      `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Release/.test(x.textContent));if(b)b.click();return !!b;})()`,
      15000,
    );
    let released = false;
    for (let i = 0; i < 30; i++) {
      released = await evalL(
        pg.sessionId,
        `document.querySelector(".model-loader")?.dataset.state === "released"`,
        10000,
      );
      if (released) break;
      await sleep(1000);
    }
    chk("release from memory → released state", released);
    await evalL(
      pg.sessionId,
      `(()=>{const b=[...document.querySelectorAll(".loader-actions button")].find(x=>/Load model into memory/.test(x.textContent));if(b)b.click();return !!b;})()`,
      15000,
    );
    let ready2 = false;
    for (let i = 0; i < 90; i++) {
      ready2 = await evalL(
        pg.sessionId,
        `document.querySelector(".model-loader")?.dataset.state === "ready"`,
        15000,
      );
      if (ready2) break;
      await sleep(2000);
    }
    chk("reload after release → ready (from cache, no re-download)", ready2);

    // responsive
    const odDesk = await evalL(
      pg.sessionId,
      `document.documentElement.scrollWidth <= window.innerWidth + 1`,
      10000,
    );
    chk("no horizontal overflow (desktop)", odDesk === true);
    await setViewport(cdp, pg.sessionId, MOBILE);
    await sleep(500);
    const odMob = await evalL(
      pg.sessionId,
      `document.documentElement.scrollWidth <= window.innerWidth + 1`,
      10000,
    );
    chk("no horizontal overflow (mobile 360px)", odMob === true);
  }
  chk("no console errors", pg.errors.length === 0, pg.errors.slice(0, 3).join(" | "));
  await closePage(cdp, pg.targetId);
} catch (e) {
  console.log("ABORT", String(e?.stack || e).slice(0, 400));
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(REPORT ? 0 : pass === total ? 0 : 1);
}
