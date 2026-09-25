// Model2Vec static-embeddings route-complete acceptance (bead web-ai-showcase-62m.4) — real inference
// in headless Chrome at desktop AND mobile for every published rung: overview · basics · practical ·
// wild · multi-model.
//
// Advertised stages driven for real (both named below so check-portfolio-acceptance can attribute
// them):
//   minishlab/potion-base-8M          — fp32 ONNX via a raw onnxruntime-web session (the 62m.4
//                                       migration: ort.wasm.min.mjs @1.21.0), tokenised by
//                                       transformers.js AutoTokenizer
//   jinaai/jina-reranker-v1-tiny-en   — q8 cross-encoder via transformers.js, the multi-model route's
//                                       stage 2 (prefilter = Model2Vec, rerank = Jina)
//
// Every cell drives the page's own controls and asserts real observable output; nothing is faked.
// Modes:
//   node scripts/validate-model2vec-static-embeddings.mjs             → full suite (exit 1 on FAIL)
//   node scripts/validate-model2vec-static-embeddings.mjs --write-run → write the run record

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
const SLUG = "model2vec-static-embeddings";
const RUN_FILE = join(repoRoot, `models/${SLUG}/acceptance-run.json`);
const ROUTES = {
  overview: "models/model2vec-static-embeddings/",
  basics: "models/model2vec-static-embeddings/basics/",
  practical: "models/model2vec-static-embeddings/practical/",
  wild: "models/model2vec-static-embeddings/wild/",
  multimodel: "models/model2vec-static-embeddings/multi-model/",
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
  ).then((r) => r.result?.value).catch((err) => {
    console.log(`  CDP evaluate failed (retrying): ${String(err?.message || err).slice(0, 120)}`);
    return null;
  });

const assertions = [];
const results = [];
let pass = 0, total = 0;
const chk = (n, c, d) => {
  total++;
  if (c) pass++;
  assertions.push({
    name: n,
    state: c ? "pass" : "fail",
    detail: d ? String(d).slice(0, 400) : undefined,
  });
  console.log(`${c ? "PASS" : "FAIL"}  ${n}${d ? " — " + String(d).slice(0, 200) : ""}`);
};
const cell = (rung, viewport, passed, detail) => {
  results.push({
    name: rung,
    route: ROUTES[rung],
    viewport,
    pass: passed,
    detail: detail ? String(detail).slice(0, 400) : undefined,
  });
};

async function waitFor(sid, expr, predicate, tries = 90, ms = 1000) {
  let last = null;
  for (let i = 0; i < tries; i++) {
    last = await evalL(sid, expr, 15000);
    if (predicate(last)) return last;
    await sleep(ms);
  }
  return last;
}

// Model readiness: auto-init from cache when possible, otherwise click Download/Load; the loader's
// honest check-timeout state is nudged with its own recovery controls instead of waiting forever.
async function waitReady(sid, mount, tries = 240, label = "model") {
  let lastLog = "";
  for (let i = 0; i < tries; i++) {
    const snapshot = await evalL(
      sid,
      `[...document.querySelectorAll(${JSON.stringify(mount + " .model-loader")})].map(n=>({state:n.dataset.state,status:n.querySelector('.status')?.textContent||''}))`,
      15000,
    );
    if (snapshot && snapshot.length > 0) {
      const encoded = JSON.stringify(snapshot.map((s) => s.state));
      if (encoded !== lastLog) {
        console.log(`    [${label}] ${encoded}`);
        lastLog = encoded;
      }
      if (snapshot.every((s) => s.state === "ready")) return true;
      if (snapshot.some((s) => s.state === "error")) return false;
      if (snapshot.some((s) => /check-timeout/.test(s.state))) {
        await evalL(
          sid,
          `(()=>{let n=0;for(const b of document.querySelectorAll(${JSON.stringify(mount)} + " .model-loader button")){if(/Retry local check|Continue/i.test(b.textContent)&&!b.disabled){b.click();n++}}return n})()`,
          10000,
        );
      }
    }
    await evalL(
      sid,
      `(()=>{let n=0;for(const b of document.querySelectorAll(${JSON.stringify(mount)} + " .model-loader button")){if(/Download|Retry|Re-download|Load model/i.test(b.textContent)&&!b.disabled){b.click();n++}}return n})()`,
      10000,
    );
    await sleep(2000);
  }
  console.log(`  ${label}: not ready after ${tries} polls`);
  return false;
}

async function waitEnabled(sid, id, tries = 60) {
  for (let i = 0; i < tries; i++) {
    const enabled = await evalL(sid, `!document.getElementById(${JSON.stringify(id)}).disabled`, 10000);
    if (enabled) return true;
    await sleep(500);
  }
  return false;
}

const bodyText = /[\s\S]/;

// --- rung drivers -------------------------------------------------------------------------------
async function driveOverview(sid) {
  await evalL(sid, `document.getElementById("run").click()`, 10000);
  const readout = await waitFor(
    sid,
    `(()=>{const r=document.getElementById("readout");return {hidden:r?.hidden, n:document.getElementById("rN")?.textContent, dim:document.getElementById("rDim")?.textContent, backend:document.getElementById("rBackend")?.textContent, status:document.getElementById("rowStatus")?.textContent}})()`,
    (v) => v && v.hidden === false && /\d/.test(v.n || "") && /\d/.test(v.dim || ""),
    120,
  );
  chk(
    "overview: embeds a batch and reports dimension/backend",
    readout && Number(readout.n) >= 2 && Number(readout.dim) > 0 && /wasm/i.test(readout.backend || ""),
    JSON.stringify(readout),
  );
}

async function driveBasics(sid) {
  await evalL(sid, `document.getElementById("run").click()`, 10000);
  const basics = await waitFor(
    sid,
    `(()=>{const out=document.getElementById("out");return {hidden:out?.hidden, cos:document.getElementById("cos")?.textContent, verdict:document.getElementById("verdict")?.textContent, backend:document.getElementById("rBackend")?.textContent, status:document.getElementById("status")?.textContent}})()`,
    (v) => v && v.hidden === false && /\d/.test(v.cos || ""),
    120,
  );
  chk(
    "basics: cosine between two sentences is a real number with a reading",
    basics && Number(basics.cos) > 0 && bodyText.test(basics.verdict || "") && /wasm/i.test(basics.backend || ""),
    JSON.stringify(basics),
  );
}

async function drivePractical(sid) {
  await evalL(sid, `document.getElementById("run").click()`, 10000);
  const practical = await waitFor(
    sid,
    `(()=>{const r=document.getElementById("readout");return {hidden:r?.hidden, n:document.getElementById("rN")?.textContent, q:document.getElementById("rQ")?.textContent, backend:document.getElementById("rBackend")?.textContent}})()`,
    (v) => v && v.hidden === false && /\d/.test(v.n || "") && /ms/.test(v.q || ""),
    150,
  );
  chk(
    "practical: indexes the corpus and answers a query",
    practical && Number(practical.n) > 0 && /ms/.test(practical.q || "") && /wasm/i.test(practical.backend || ""),
    JSON.stringify(practical),
  );
}

async function driveWild(sid) {
  const typed = await evalL(
    sid,
    `(()=>{const q=document.getElementById("query");if(!q||q.disabled)return false;q.value="a cat sleeping in the sun";q.dispatchEvent(new Event("input",{bubbles:true}));return true;})()`,
    10000,
  );
  chk("wild: query input accepts live typing", typed === true);
  const wild = await waitFor(
    sid,
    `(()=>{const r=document.getElementById("readout");return {hidden:r?.hidden, k:document.getElementById("rK")?.textContent, ms:document.getElementById("rMs")?.textContent, results:document.querySelectorAll("#results > *").length, backend:document.getElementById("rBackend")?.textContent}})()`,
    (v) => v && v.hidden === false && Number(v.k) >= 1 && v.results > 0,
    150,
  );
  chk(
    "wild: live ranking renders results from real embeddings",
    wild && Number(wild.k) >= 1 && wild.results > 0 && /wasm/i.test(wild.backend || ""),
    JSON.stringify(wild),
  );
}

async function driveMultimodel(sid) {
  const potion = await waitReady(sid, "#model-loader", 240, "potion (stage 1)");
  const jina = await waitReady(sid, "#model-loader-2", 240, "jina reranker (stage 2)");
  chk("multimodel: both advertised stages ready", potion && jina, JSON.stringify({ potion, jina }));
  if (!potion || !jina) return;
  await waitEnabled(sid, "run");
  await evalL(sid, `document.getElementById("run").click()`, 10000);
  const mm = await waitFor(
    sid,
    `(()=>{const post=document.getElementById("post");return {status:document.getElementById("status")?.textContent, idx:document.getElementById("rIdx")?.textContent, rr:document.getElementById("rRr")?.textContent, rendered:post?post.children.length:0, sample:post?post.textContent.slice(0,80):""}})()`,
    (v) => v && /Done/.test(v.status || "") && Number(v.rendered) > 0,
    180,
  );
  chk(
    "multimodel: prefilter then real cross-encoder rerank renders a ranked list",
    mm && /Done/.test(mm.status || "") && Number(mm.rendered) > 0 && /ms/.test(mm.idx || "") && /ms/.test(mm.rr || ""),
    JSON.stringify(mm),
  );
}

const DRIVERS = {
  overview: driveOverview,
  basics: driveBasics,
  practical: drivePractical,
  wild: driveWild,
};

async function driveCell(rung, viewport) {
  const before = { pass, total };
  const page = await openPage(cdp, base + ROUTES[rung]);
  try {
    await setViewport(cdp, page.sessionId, VIEWPORTS[viewport]);
    await sleep(600);
    const mount = "#model-loader";
    const ready = await waitReady(page.sessionId, mount, 240, `${rung} [${viewport}]`);
    chk(`${rung} [${viewport}]: model ready (cached auto-init)`, ready);
    if (ready) {
      if (rung === "multimodel") {
        // The multimodel driver loads stage 2 itself; stage 1 is already ready from the wait above.
        await driveMultimodel(page.sessionId);
      } else {
        await DRIVERS[rung](page.sessionId);
      }
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
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) {
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
    stages: ["minishlab/potion-base-8M", "jinaai/jina-reranker-v1-tiny-en"],
    matrix: { routes: Object.values(ROUTES), viewports: Object.keys(VIEWPORTS), routeDeviceCells: results.length },
    summary: { checks: total, passed: pass, failed, cells: results.length, cellsPassed },
    results,
    assertions,
    notes: [
      "Route-complete real-inference acceptance at desktop and mobile for every published rung.",
      "Stage 1 runs on a raw onnxruntime-web session (1.21.0 ort.wasm bundle after web-ai-showcase-62m.4).",
      "Stage 2 (Jina cross-encoder reranker, transformers.js q8) is driven for real on the multi-model rung.",
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
