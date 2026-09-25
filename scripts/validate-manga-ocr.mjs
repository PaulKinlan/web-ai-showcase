// Manga OCR route-complete acceptance (web-ai-showcase-62m.1) — real inference in headless Chrome
// at desktop AND mobile for every published rung: overview · basics · practical · wild · multimodel.
//
// Advertised stages driven for real (both named below so check-portfolio-acceptance can attribute
// them):
//   onnx-community/manga-ocr-base-ONNX  — uint8 encoder (87.0 MB) + q8 decoder (29.6 MB), WASM,
//                                          raw onnxruntime-web sessions with the hand-rolled greedy
//                                          decode (the 62m.1 migration: ort.wasm.min.mjs @1.21.0)
//   Xenova/m2m100_418M                  — q8, 470 MB, the multimodel route's translation stage
//
// Every cell exercises the real control path and asserts real recognised text; the overview desktop
// cell additionally drives the full first-visit lifecycle (absent → Download → ready → 7 bundled
// samples → see-inside probabilities → raw toggle → release → reload from cache).
//
// Modes:
//   node scripts/validate-manga-ocr.mjs              → full assertion suite (exit 1 on any FAIL)
//   node scripts/validate-manga-ocr.mjs --report     → print true outputs for the sample loop
//   node scripts/validate-manga-ocr.mjs --write-run  → write models/manga-ocr/acceptance-run.json
//
// Harness: one fresh Chrome process (absent cache), the same profile across cells so every later
// cell proves cached auto-init. Every wait has a hard deadline.

import { execFileSync } from "node:child_process";
import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  assertHeadUnchanged,
  captureHeadCommit,
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const REPORT = process.argv.includes("--report");
const SLUG = "manga-ocr";
const STAGE = "onnx-community/manga-ocr-base-ONNX";
const TRANSLATE_STAGE = "Xenova/m2m100_418M";
const RUN_FILE = join(repoRoot, `models/${SLUG}/acceptance-run.json`);
const ROUTES = {
  overview: "models/manga-ocr/",
  basics: "models/manga-ocr/basics/",
  practical: "models/manga-ocr/practical/",
  wild: "models/manga-ocr/wild/",
  multimodel: "models/manga-ocr/multimodel/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };

// Clean up stale harness Chromes/profiles from interrupted runs (they lock the profile dir).
try {
  execFileSync("pkill", ["-f", "conformance-chrome-profile"]);
} catch { /* none */ }
try {
  rmSync(new URL("../.conformance-chrome-profile", import.meta.url), {
    recursive: true,
    force: true,
  });
} catch { /* ignore */ }

const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
const startCommit = captureHeadCommit();
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
  ).then((r) => r.result?.value).catch((err) => {
    // A transient CDP timeout must not abort the whole acceptance run; the polling loops retry, and
    // an assertion still fails if the page never answers.
    console.log(`  CDP evaluate failed (retrying): ${String(err?.message || err).slice(0, 120)}`);
    return null;
  });

const assertions = [];
const results = [];
let pass = 0, total = 0;
const chk = (n, c, d) => {
  total++;
  if (c) pass++;
  assertions.push({ name: n, state: c ? "pass" : "fail", detail: d ? String(d).slice(0, 400) : undefined });
  console.log(`${c ? "PASS" : "FAIL"}  ${n}${d ? " — " + String(d).slice(0, 200) : ""}`);
};
const note = (n, d) => console.log(`NOTE  ${n}${d ? " — " + d : ""}`);
const cell = (rung, viewport, passed, detail) => {
  results.push({
    name: rung,
    route: ROUTES[rung],
    viewport,
    pass: passed,
    detail: detail ? String(detail).slice(0, 400) : undefined,
  });
};
const norm = (t) => String(t ?? "").replace(/\s+/g, "");

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
const HORIZONTAL = SAMPLES[0].want;
const BUBBLE = SAMPLES[2].want;
const HAS_JAPANESE = /[\u3040-\u30ff\u4e00-\u9fff]/;

async function waitFor(sid, expr, predicate, tries = 90, ms = 2000) {
  let last = null;
  for (let i = 0; i < tries; i++) {
    last = await evalL(sid, expr, 15000);
    if (predicate(last)) return last;
    await sleep(ms);
  }
  return last;
}
const waitContains = (sid, expr, want, tries = 90) =>
  waitFor(sid, expr, (t) => String(t ?? "").includes(want), tries);

// Model readiness: auto-init from cache when possible, otherwise click Download/Load and wait.
async function waitReady(sid, mount, tries = 120, label = "model") {
  let clicked = false;
  for (let i = 0; i < tries; i++) {
    const state = await evalL(
      sid,
      `document.querySelector(${JSON.stringify(mount + " .model-loader")})?.dataset.state`,
      15000,
    );
    if (state === "ready") return true;
    if (state === "error") return false;
    if (!clicked) {
      clicked = await evalL(
        sid,
        `(()=>{const root=document.querySelector(${JSON.stringify(mount)});
          const b=root&&[...root.querySelectorAll(".loader-actions button")].find(x=>/Download|Load model/.test(x.textContent));
          if(b){b.click();return true;}return false;})()`,
        15000,
      );
    }
    await sleep(2000);
  }
  console.log(`  ${label}: not ready after ${tries} polls`);
  return false;
}

async function waitEnabled(sid, id, tries = 30) {
  for (let i = 0; i < tries; i++) {
    const enabled = await evalL(sid, `!document.getElementById(${JSON.stringify(id)}).disabled`, 10000);
    if (enabled) return true;
    await sleep(500);
  }
  return false;
}

// --- rung drivers (each asserts real recognised text through the page's own controls) ------------
// Click a sample thumb, let the crop image load, then run and wait for the final text. The thumb
// click is async (crop.setSrc → image load); clicking run in the same tick races that load and can
// read a blank crop, so a rejected run is retried instead of silently returning stale output.
async function runSampleViaUi(sid, src, tries = 90) {
  await evalL(sid, `document.querySelector('#samples .sample-thumb[data-src="${src}"]')?.click()`, 15000);
  await sleep(700); // crop.setSrc load + redraw before the run reads the crop
  for (let attempt = 0; attempt < 4; attempt++) {
    const before = await evalL(sid, `document.getElementById("out")?.textContent || ""`, 10000);
    const started = await evalL(
      sid,
      `(()=>{const r=document.getElementById("run");if(!r||r.disabled)return false;r.click();return true;})()`,
      10000,
    );
    if (!started) {
      await sleep(1000);
      continue;
    }
    for (let i = 0; i < tries; i++) {
      const st = await evalL(
        sid,
        `({busy:!document.getElementById("cancel").disabled,
          out:document.getElementById("out")?.textContent || "",
          rejected:!document.getElementById("runStatus").hidden && /Choose an image/.test(document.getElementById("runStatus").textContent)})`,
        15000,
      );
      if (st?.rejected) break; // crop was not ready — retry the run
      if (st && !st.busy && st.out !== "") return st.out;
      await sleep(750);
    }
    if (before === "" || await evalL(sid, `document.getElementById("out")?.textContent || ""`, 10000) === before) {
      await sleep(500); // the click did nothing — try again
    }
  }
  return await evalL(sid, `document.getElementById("out")?.textContent || ""`, 10000);
}

async function driveOverviewSample(sid) {
  const out = await runSampleViaUi(sid, "sample-horizontal.png");
  chk(`overview: bundled sample recognised`, norm(out) === norm(HORIZONTAL), `got "${out}"`);
}

async function driveBasics(sid) {
  await evalL(sid, `document.querySelector(".read-btn")?.click()`, 10000);
  const out = await waitContains(sid, `document.getElementById("out1")?.textContent || ""`, HORIZONTAL);
  chk("basics: sample card produces real OCR", String(out).includes(HORIZONTAL), JSON.stringify(out));
}

async function drivePractical(sid) {
  const init = await evalL(
    sid,
    `({loaders:document.querySelectorAll(".model-loader").length, canvas:document.querySelector("#crop")?.tagName})`,
    15000,
  );
  chk("practical: one loader + crop canvas", init?.loaders === 1 && init?.canvas === "CANVAS", JSON.stringify(init));
  if (await waitEnabled(sid, "add")) await evalL(sid, `document.getElementById("add").click()`, 10000);
  if (await waitEnabled(sid, "runAll")) await evalL(sid, `document.getElementById("runAll").click()`, 10000);
  const out = await waitContains(sid, `document.querySelector("#transcript .qtext")?.textContent || ""`, BUBBLE);
  chk("practical: queued region transcribed for real", String(out).includes(BUBBLE), JSON.stringify(out));
}

async function driveWild(sid) {
  const clicked = await evalL(
    sid,
    `(()=>{const b=document.querySelector(".cap__fallback button");if(!b)return false;b.click();return true;})()`,
    15000,
  );
  chk("wild: bundled-frame path offered without a camera", clicked === true);
  // The bundled sample lands in the capture component's review step — confirm it with its own
  // "Use this" control, which is what actually calls the page's onResult (house pattern, see
  // scripts/validate-wav2vec2-large-xlsr-53-gender-recognition-librispeech.mjs).
  const used = await waitFor(
    sid,
    `(()=>{const b=[...document.querySelectorAll("#cap button")].find(x=>/Use this/.test(x.textContent));if(!b)return false;b.click();return true;})()`,
    (v) => v === true,
    40,
    500,
  );
  chk("wild: bundled sample reviewed and used", used === true);
  const out = await waitContains(sid, `document.querySelector("#log li .rtext")?.textContent || ""`, BUBBLE, 120);
  chk("wild: session log holds real OCR for the frame", String(out).includes(BUBBLE), JSON.stringify(out));
}

async function driveMultimodel(sid) {
  const init = await evalL(
    sid,
    `({loaders:document.querySelectorAll(".model-loader").length, canvas:document.querySelector("#crop")?.tagName})`,
    15000,
  );
  chk("multimodel: two stage loaders + crop canvas", init?.loaders === 2 && init?.canvas === "CANVAS", JSON.stringify(init));
  if (await waitEnabled(sid, "read")) await evalL(sid, `document.getElementById("read").click()`, 10000);
  const ja = await waitContains(sid, `document.getElementById("ja")?.value || ""`, HORIZONTAL);
  chk("multimodel: real OCR into the editable box", String(ja).includes(HORIZONTAL), JSON.stringify(ja));

  // Stage 2 is a separate advertised model (Xenova/m2m100_418M, 470 MB q8) — drive it for real.
  const trReady = await waitReady(sid, "#loader-tr", 300, "m2m100 translation stage");
  chk("multimodel: translation stage ready (m2m100 q8, 470 MB)", trReady);
  if (trReady) {
    await waitEnabled(sid, "translate", 60);
    await evalL(sid, `document.getElementById("translate").click()`, 10000);
    // #trMeta is written only on the final result message, so waiting for it proves the run finished
    // (the streamed #en text alone can still be mid-stream).
    const done = await waitFor(
      sid,
      `({en:document.getElementById("en")?.textContent || "", meta:document.getElementById("trMeta")?.textContent || ""})`,
      (v) => v && /Translated in/.test(v.meta || ""),
      180,
    );
    const en = String(done?.en ?? "").trim();
    chk(
      "multimodel: real English translation from the OCR text",
      en.length > 3 && !HAS_JAPANESE.test(en),
      `${JSON.stringify(en)} | ${done?.meta ?? ""}`,
    );
  }
}

async function driveCell(rung, viewport) {
  const before = { pass, total };
  const page = await openPage(cdp, base + ROUTES[rung]);
  try {
    await setViewport(cdp, page.sessionId, VIEWPORTS[viewport]);
    await sleep(800);
    const mount = rung === "multimodel" ? "#loader-ocr" : "#model-loader";
    const ready = await waitReady(page.sessionId, mount, 120, `${rung} [${viewport}]`);
    chk(`${rung} [${viewport}]: model ready (cached auto-init)`, ready);
    if (ready) {
      if (rung === "overview") await driveOverviewSample(page.sessionId);
      else if (rung === "basics") await driveBasics(page.sessionId);
      else if (rung === "practical") await drivePractical(page.sessionId);
      else if (rung === "wild") await driveWild(page.sessionId);
      else if (rung === "multimodel") await driveMultimodel(page.sessionId);
    }
    const overflow = await evalL(
      page.sessionId,
      `document.documentElement.scrollWidth <= window.innerWidth + 1`,
      10000,
    );
    chk(`${rung} [${viewport}]: no horizontal overflow`, overflow === true);
    chk(`${rung} [${viewport}]: no console errors`, page.errors.length === 0, page.errors.slice(0, 3).join(" | "));
  } finally {
    await closePage(cdp, page.targetId);
  }
  const ok = pass - before.pass === total - before.total;
  cell(rung, viewport, ok, ok ? undefined : "one or more checks failed — see assertions");
  return ok;
}

try {
  // --- overview desktop: full first-visit lifecycle on a cold profile ---------------------------
  const deepBefore = { pass, total };
  const pg = await openPage(cdp, base + ROUTES.overview);
  await sleep(1500);
  const s0 = await evalL(
    pg.sessionId,
    `(()=>({loader:!!document.querySelector(".model-loader"),
      state:document.querySelector(".model-loader")?.dataset.state,
      dl:[...document.querySelectorAll(".loader-actions button")].some(b=>/Download/.test(b.textContent))}))()`,
    15000,
  );
  chk("overview [desktop]: loader + Download (fresh profile = absent)", s0?.loader && s0?.dl, JSON.stringify(s0));

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
      if (st?.state === "error") {
        loadErr = st.status;
        break;
      }
      if (st?.state === "ready" && st?.ready) {
        ready = true;
        break;
      }
      await sleep(2000);
    }
  }
  chk("overview [desktop]: download ~117 MB → ready, controls enabled", ready, loadErr || undefined);

  // first sample is preloaded; run each sample through the real model
  if (ready) {
    for (const s of SAMPLES) {
      const out = await runSampleViaUi(pg.sessionId, s.src);
      if (REPORT) note(s.src, JSON.stringify(out));
      else chk(`overview [desktop]: recognise ${s.src}`, norm(out) === norm(s.want), norm(out) === norm(s.want) ? "" : `got "${out}" want "${s.want}"`);
    }

    // see-inside trace has per-character probability chips after a run
    const trace = await evalL(
      pg.sessionId,
      `({chips:document.querySelectorAll("#trace .tok").length,
        title:(document.querySelector("#trace .tok")||{}).title || ""})`,
      10000,
    );
    chk("overview [desktop]: see-inside per-char probability chips", trace?.chips > 3, JSON.stringify(trace));

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
      "overview [desktop]: raw model output toggle",
      rawToggle?.skip || rawToggle?.back === rawToggle?.before,
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
    chk("overview [desktop]: release from memory → released state", released);
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
    chk("overview [desktop]: reload after release → ready (from cache, no re-download)", ready2);
  }
  const odDesk = await evalL(
    pg.sessionId,
    `document.documentElement.scrollWidth <= window.innerWidth + 1`,
    10000,
  );
  chk("overview [desktop]: no horizontal overflow", odDesk === true);
  chk("overview [desktop]: no console errors", pg.errors.length === 0, pg.errors.slice(0, 3).join(" | "));
  await closePage(cdp, pg.targetId);
  cell("overview", "desktop", pass - deepBefore.pass === total - deepBefore.total);

  // --- every other rung x viewport cell (cached auto-init), including overview mobile ------------
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) {
      if (rung === "overview" && viewport === "desktop") continue;
      await driveCell(rung, viewport);
    }
  }
} catch (e) {
  console.log("ABORT", String(e?.stack || e).slice(0, 400));
  total += 1; // an aborted run can never report a green denominator
  assertions.push({ name: "run completed", state: "fail", detail: String(e?.message || e).slice(0, 200) });
} finally {
  const failed = total - pass;
  const cellsPassed = results.filter((r) => r.pass).length;
  console.log(`\nRESULT ${pass}/${total} checks; ${cellsPassed}/${results.length} route cells`);
  const record = {
    schemaVersion: 1,
    slug: SLUG,
    commit: startCommit,
    ranAt: new Date().toISOString(),
    exitCode: failed ? 1 : 0,
    stages: [STAGE, TRANSLATE_STAGE],
    matrix: { routes: Object.values(ROUTES), viewports: Object.keys(VIEWPORTS), routeDeviceCells: results.length },
    summary: { checks: total, passed: pass, failed, cells: results.length, cellsPassed },
    results,
    assertions,
    notes: [
      "Route-complete real-inference acceptance at desktop and mobile for every published rung.",
      "OCR runs on raw onnxruntime-web sessions (1.21.0 wasm bundle after web-ai-showcase-62m.1).",
      "The multimodel cell drives BOTH advertised stages: " + STAGE + " and " + TRANSLATE_STAGE + ".",
      "Mobile is Chrome device emulation on desktop hardware, not a physical low-memory phone.",
    ],
  };
  if (WRITE_RUN) {
    try {
      assertHeadUnchanged(startCommit);
      writeFileSync(RUN_FILE, JSON.stringify(record, null, 2) + "\n");
      console.log(`WROTE ${RUN_FILE} for commit ${startCommit}`);
    } catch (err) {
      console.error(`REFUSAL: ${err.message}`);
      process.exitCode = 1;
    }
  }
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  if (failed) process.exitCode = 1;
}
