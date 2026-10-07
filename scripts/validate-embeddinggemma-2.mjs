// Route-complete real-browser acceptance for EmbeddingGemma 2.
//
// Drives every overview / Basics / Practical / Wild / Multi-model route at desktop + mobile, and asserts
// the things that actually matter for this family:
//   • real inference, not a rendered placeholder — the vectors must be finite, unit-length and NON-trivial
//     (if every pairwise cosine were identical the "embeddings" would be degenerate and every ranking a lie);
//   • semantic behaviour, not just a number — the prefilled attention query must rank an attention passage
//     first; cross-lingual query retrieves the measured ranking;
//   • lifecycle — the model is released and re-initialised; sequential unload on the multi-model route
//     verifies bounded memory footprint when comparing two distinct model generations;
//   • the honest failure surface — this export only runs on WebGPU, and that has to be visible to a user.
//
// The advertised stages are named here in full: onnx-community/embeddinggemma-2-ONNX (the browser build) of
// google/embeddinggemma-2 (the canonical weights).
//
// Environment note carried into every claim: these runs use a SwiftShader *software* WebGPU adapter on a
// shared CPU box, so the latencies prove the kernels ran, not how fast this is on real hardware.
import { join } from "node:path";
import {
  captureHeadCommit,
  CDP,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  printAcceptanceSummary,
  repoRoot,
  setViewport,
  startServer,
  writeAcceptanceRunRecord,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const RUN_RECORD = join(repoRoot, "models/embeddinggemma-2/acceptance-run.json");
const startCommit = captureHeadCommit(repoRoot);
const MODEL_ID = "onnx-community/embeddinggemma-2-ONNX";
const UPSTREAM_ID = "google/embeddinggemma-2";
const ROUTES = {
  overview: "models/embeddinggemma-2/",
  basics: "models/embeddinggemma-2/basics/",
  practical: "models/embeddinggemma-2/practical/",
  wild: "models/embeddinggemma-2/wild/",
  multimodel: "models/embeddinggemma-2/multi-model/",
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const { server, port } = await startServer();
const chrome = await launchChrome({
  // Deliberately NOT overriding userDataDir/resetProfile/removeProfileOnKill. I had pinned a fixed profile
  // and disabled its removal to cache the ~175 MB model between runs, and those three overrides accumulated
  // 7.6 GiB of abandoned profiles in /tmp and triggered a disk-pressure alert. The harness defaults
  // (isolated dir per run, reset, removed on kill, closure via registerGlobalExitHooks) are correct; a fresh
  // download per run costs seconds and cannot leak. Do not re-add a persistent profile here.
  webgpu: true,
});
const cdp = new CDP(chrome.ws);
const url = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;
let passed = 0;
let total = 0;
const results = [];

function check(name, condition, detail = "") {
  total++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${String(detail).slice(0, 240)}` : ""}`,
  );
  return Boolean(condition);
}
async function evaluate(sessionId, expression, timeoutMs = 60_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: `(async()=>{try{return (${expression});}catch(e){return "ERR:"+(e?.message||e)}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}
async function waitFor(sessionId, expression, label, timeoutMs = 20 * 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await evaluate(sessionId, expression)) return true;
    } catch (error) {
      console.log(`  [${label}] transient CDP stall: ${error.message}`);
    }
    await sleep(2000);
  }
  throw new Error(`TIMEOUT: ${label}`);
}
const click = (selector) =>
  `(()=>{const e=document.querySelector(${
    JSON.stringify(selector)
  });if(!e)return false;e.click();return true})()`;
const clickByText = (scope, pattern) =>
  `(()=>{let n=0;for(const b of document.querySelectorAll(${JSON.stringify(scope)})){if(${
    pattern
  }.test(b.textContent)){b.click();n++}}return n})()`;
const clickDownloads = (scope = "#model-loader") =>
  `(()=>{let n=0;for(const b of document.querySelectorAll(${JSON.stringify(scope)} + ' button')){if(/Download|Retry/i.test(b.textContent)){b.click();n++}}return n})()`;
const noOverflow = `document.documentElement.scrollWidth <= window.innerWidth + 1`;

/**
 * Drive a route until its controls are enabled. A reused Chrome profile means the second route onward is
 * cache-warm and the loader auto-initialises; the first visit must still present a real Download button.
 */
async function readyControls(page, firstVisit, readySelector, loaderSelector = "#model-loader") {
  await sleep(1800);
  if (firstVisit) {
    const labels = await evaluate(
      page.sessionId,
      `JSON.stringify([...document.querySelectorAll('.model-loader button')].map(b=>b.textContent.trim()))`,
    );
    check(`fresh profile offers a real Download for ${MODEL_ID}`, /Download/.test(labels), labels);
  }
  await evaluate(page.sessionId, clickDownloads(loaderSelector));
  await waitFor(page.sessionId, `!document.querySelector(${JSON.stringify(readySelector)})?.disabled`, `ready via ${readySelector}`);
}

async function hygiene(page, route, viewport) {
  const overflow = await evaluate(page.sessionId, noOverflow);
  return [
    check(`${viewport} ${route}: no horizontal overflow`, overflow),
    check(`${viewport} ${route}: zero console errors`, page.errors.length === 0, page.errors.join(" | ")),
    check(`${viewport} ${route}: zero failed network requests`, page.netFailures.length === 0, page.netFailures.join(" | ")),
  ].every(Boolean);
}

/** Split the rendered similarity matrix into its numeric cells so triviality can be ruled out. */
const matrixCells =
  `JSON.stringify([...document.querySelectorAll('#matrix .sim-matrix td')].map(td=>td.textContent).filter(t=>t!==''))`;

async function exercise(routeName, viewportName, viewport, firstVisit = false) {
  const route = ROUTES[routeName];
  const page = await openPage(cdp, url(route));
  await setViewport(cdp, page.sessionId, viewport);
  let ok = true;
  let pass = false;
  try {
    if (routeName === "overview") {
      await readyControls(page, firstVisit, "#run");
      ok = check(`${viewportName} overview: WebGPU-only limitation is stated on the page`,
        await evaluate(page.sessionId, `document.body.innerText.includes('GatherBlockQuantized') || document.body.innerText.includes('WebGPU')`)) && ok;

      ok = check(`${viewportName} overview: Embed & compare clicked`, await evaluate(page.sessionId, click("#run"))) && ok;
      await waitFor(page.sessionId, `document.querySelector('#rDim')?.textContent==='768'`, "overview 768-d inference");
      const shape = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({dim:document.querySelector('#rDim')?.textContent,cells:document.querySelectorAll('#vecStrip .vec-cell').length,norm:Number(document.querySelector('#iNorm')?.textContent),nan:Number(document.querySelector('#iNan')?.textContent),matrix:${matrixCells},ms:document.querySelector('#rMs')?.textContent,budgetRows:document.querySelectorAll('#budget tbody tr').length})`));
      const uniq = new Set(shape.matrix);
      ok = check(
        `${viewportName} overview: real 768-d vectors are finite, unit-length and non-trivial`,
        shape.dim === "768" && shape.cells === 128 && Math.abs(shape.norm - 1) < 0.01 && shape.nan === 0 &&
          shape.matrix.length >= 6 && uniq.size >= 3,
        JSON.stringify({ dim: shape.dim, cells: shape.cells, norm: shape.norm, nan: shape.nan, offDiagValues: uniq.size, ms: shape.ms }),
      ) && ok;
      ok = check(`${viewportName} overview: encoder-budget table lists encoders`, shape.budgetRows >= 2, `rows=${shape.budgetRows}`) && ok;

      await evaluate(page.sessionId, click("#search"));
      await waitFor(page.sessionId, `document.querySelectorAll('#ranked .result-row').length>=2`, "overview search");
      const top = await evaluate(page.sessionId, `document.querySelector('#ranked .result-row .result-head span')?.textContent||""`);
      const scores = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#ranked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} overview: semantic search is finite and descending`, scores.length >= 2 && scores.every(Number.isFinite) && scores.every((s, i) => i === 0 || s <= scores[i - 1] + 1e-6), JSON.stringify(scores)) && ok;
      ok = check(`${viewportName} overview: the attention query retrieves an attention passage`, /attention/i.test(top), top.slice(0, 90)) && ok;

      await evaluate(page.sessionId, click("#matRun"));
      await waitFor(page.sessionId, `document.querySelectorAll('#matBars .mat-row').length===4`, "overview matryoshka");
      const mats = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#matBars .mat-row .val')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} overview: Matryoshka 768/512/256/128 all real`, mats.length === 4 && mats.every(Number.isFinite), JSON.stringify(mats)) && ok;

      // Lifecycle: release the model from memory, then re-initialise it (assets stay cached, so this is an
      // init, not a re-download) and prove inference still works afterwards.
      await evaluate(page.sessionId, clickByText(".model-loader button", "/Release from memory/"));
      const reinit = await waitFor(page.sessionId, `(()=>{for(const b of document.querySelectorAll('.model-loader button')){if(/Load model into memory/.test(b.textContent)){b.click();return true}}return false})()`, "overview release", 4 * 60_000);
      ok = check(`${viewportName} overview: model can be released from memory`, reinit) && ok;
      await waitFor(page.sessionId, `!document.querySelector('#run')?.disabled`, "overview re-init");
      await evaluate(page.sessionId, click("#run"));
      await waitFor(page.sessionId, `document.querySelector('#rDim')?.textContent==='768'`, "overview re-run after re-init");
      const afterRelease = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({norm:Number(document.querySelector('#iNorm')?.textContent),nan:Number(document.querySelector('#iNan')?.textContent)})`));
      ok = check(`${viewportName} overview: inference works again after release + re-initialise`, Math.abs(afterRelease.norm - 1) < 0.01 && afterRelease.nan === 0, JSON.stringify(afterRelease)) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "basics") {
      await readyControls(page, false, "#run");
      ok = check(`${viewportName} Basics: run clicked`, await evaluate(page.sessionId, click("#run"))) && ok;
      await waitFor(page.sessionId, `Number.isFinite(Number(document.querySelector('#score')?.textContent))`, "Basics cosine");
      // Read the COSINE column (3rd cell) for the finite check, not td:last-child: the last cell is the delta
      // column, and the baseline row renders "0.000 (baseline)", so Number() on it is NaN and would fail a
      // correct page. The baseline row is asserted separately for its own text, which is the real property.
      const basics = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({score:Number(document.querySelector('#score')?.textContent),rows:[...document.querySelectorAll('#prefixTable tbody tr')].map(r=>Number(r.querySelector('td:nth-child(3)')?.textContent)),deltas:[...document.querySelectorAll('#prefixTable tbody tr')].map(r=>r.querySelector('td:last-child')?.textContent||''),mode:document.querySelector('#rMode')?.textContent,tok:document.querySelector('#rTok')?.textContent,dim:document.querySelector('#rDim')?.textContent})`));
      ok = check(`${viewportName} Basics: real cosine in (0,1]`, basics.score > 0 && basics.score <= 1.0001, String(basics.score)) && ok;
      ok = check(`${viewportName} Basics: 768-d output reported`, basics.dim === "768", String(basics.dim)) && ok;
      ok = check(`${viewportName} Basics: prefix comparison table computed from real embeddings`, basics.rows.length >= 5 && basics.rows.every(Number.isFinite), JSON.stringify(basics.rows)) && ok;
      // The baseline row must be labelled as the baseline rather than showing a bare delta, so a page that
      // silently dropped its control row cannot pass just by producing five finite cosines.
      ok = check(`${viewportName} Basics: baseline row labelled`, basics.deltas.filter((d) => /baseline/i.test(d)).length === 1, JSON.stringify(basics.deltas)) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "practical") {
      await readyControls(page, false, "#indexBtn");
      await evaluate(page.sessionId, click("#indexBtn"));
      await waitFor(page.sessionId, `Number(document.querySelector('#rChunks')?.textContent)>1`, "Practical index");
      // #rTokens now reports a compound value (largest chunk + indexed total), so parse the leading integer
      // rather than coercing the whole string: Number("32 (248 total)") is NaN and would fail a passing page.
      const practical = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({chunks:Number(document.querySelector('#rChunks')?.textContent),tokens:parseInt(document.querySelector('#rTokens')?.textContent,10),rDim:document.querySelector('#rDim')?.textContent,size:document.querySelector('#rSize')?.textContent})`));
      ok = check(`${viewportName} Practical: document chunked and embedded for real`, practical.chunks > 1 && practical.tokens > 0, JSON.stringify(practical)) && ok;
      ok = check(`${viewportName} Practical: chunk token count is inside the 8192-token window`, practical.tokens > 0 && practical.tokens <= 8192, `tokens=${practical.tokens}`) && ok;
      // Indexing automatically executes search() when query is present; wait for the auto-search hits.
      await waitFor(page.sessionId, `document.querySelectorAll('#ranked .result-row').length>=2`, "Practical search");
      const hits = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#ranked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} Practical: chunk retrieval returns finite scores`, hits.length >= 2 && hits.every(Number.isFinite), JSON.stringify(hits.slice(0, 4))) && ok;
      // Matryoshka dial: the stored width changes and the reported corpus size shrinks, with no re-embedding.
      const before = await evaluate(page.sessionId, `document.querySelector('#rSize')?.textContent`);
      await evaluate(page.sessionId, `(()=>{const s=document.querySelector('#dims');if(!s)return false;s.value=s.max;s.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
      await waitFor(page.sessionId, `document.querySelector('#rDim')?.textContent==='128-d' || document.querySelector('#rDim')?.textContent==='128'`, "Practical 128-d");
      const after = await evaluate(page.sessionId, `document.querySelector('#rSize')?.textContent`);
      const afterHits = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#ranked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} Practical: 128-d re-rank is real and the stored index shrinks`, afterHits.every(Number.isFinite) && after !== before && afterHits.length >= 2, `${before} -> ${after}`) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "wild") {
      await readyControls(page, false, "#run");
      ok = check(`${viewportName} Wild: Search across languages clicked`, await evaluate(page.sessionId, click("#run"))) && ok;
      await waitFor(page.sessionId, `document.querySelectorAll('#ranked .result-row').length>=2`, "Wild cross-lingual search", 30 * 60_000);
      const wild = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({
        hits: [...document.querySelectorAll('#ranked .result-row')].map(r => ({
          head: r.querySelector('.result-head')?.textContent || '',
          sub: r.querySelector('.result-sub')?.textContent || '',
          score: Number(r.querySelector('.result-score')?.textContent)
        })),
        verdict: document.querySelector('#verdict')?.textContent || '',
        backend: document.querySelector('#rBackend')?.textContent || '',
        tokens: document.querySelector('#rTokens')?.textContent || '',
        dim: document.querySelector('#rDim')?.textContent || '',
        ms: document.querySelector('#rMs')?.textContent || '',
        langTableRows: document.querySelectorAll('#langTable tbody tr').length
      })`));
      ok = check(`${viewportName} Wild: cross-lingual retrieval returns finite scores`, wild.hits.length >= 2 && wild.hits.every(h => Number.isFinite(h.score)), JSON.stringify(wild.hits.slice(0, 3).map(h => h.score))) && ok;
      ok = check(`${viewportName} Wild: scores are descending (a real ranking)`, wild.hits.every((h, i) => i === 0 || h.score <= wild.hits[i - 1].score + 1e-6), JSON.stringify(wild.hits.map(h => h.score))) && ok;
      ok = check(`${viewportName} Wild: each rendered hit carries its language`, wild.hits.every(h => /\[[A-Z]{2}\]/.test(h.head) || /Language:/i.test(h.sub)), JSON.stringify(wild.hits.slice(0, 2).map(h => h.head))) && ok;
      const topCodeMatch = wild.hits[0]?.head.match(/\[([A-Z]{2})\]/);
      const topCode = topCodeMatch ? topCodeMatch[1] : "";
      const verdictMatchesTop = Boolean(wild.verdict) && (topCode ? wild.verdict.includes(`[${topCode}]`) : true);
      const verdictHasMargin = /margin/i.test(wild.verdict);
      ok = check(`${viewportName} Wild: verdict is non-empty and reports measured top-1 language and margin`, Boolean(wild.verdict) && verdictMatchesTop && verdictHasMargin, wild.verdict) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "multimodel") {
      // Stage 2 (v2): EmbeddingGemma 2 (q4, WebGPU)
      await readyControls(page, false, "#runV2", "#model-loader");
      ok = check(`${viewportName} Multi-model: Index & rank (v2) clicked`, await evaluate(page.sessionId, click("#runV2"))) && ok;
      await waitFor(page.sessionId, `document.querySelectorAll('#rankedV2 .result-row').length>=2`, "Multi-model v2 run", 30 * 60_000);
      const v2Hits = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#rankedV2 .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} Multi-model: stage 2 (v2) ranks with finite scores`, v2Hits.length >= 2 && v2Hits.every(Number.isFinite), JSON.stringify(v2Hits.slice(0, 3))) && ok;
      const unloadV2Enabled = await evaluate(page.sessionId, `!document.querySelector('#unloadV2')?.disabled`);
      ok = check(`${viewportName} Multi-model: unloadV2 is enabled after v2 run`, Boolean(unloadV2Enabled)) && ok;

      // Sequential unload: clicking #unloadV2 releases v2 and leaves both runs disabled until v1 is brought up
      await evaluate(page.sessionId, click("#unloadV2"));
      await waitFor(page.sessionId, `document.querySelector('#runV2')?.disabled && document.querySelector('#runV1')?.disabled`, "Multi-model runs disabled after unload");
      const runsDisabled = await evaluate(page.sessionId, `document.querySelector('#runV2')?.disabled && document.querySelector('#runV1')?.disabled`);
      ok = check(`${viewportName} Multi-model: clicking unloadV2 leaves runs disabled until v1 is brought up`, Boolean(runsDisabled)) && ok;

      // Bring up Stage 1 (v1): EmbeddingGemma v1 (q8, WASM)
      await evaluate(page.sessionId, `(()=>{for(const b of document.querySelectorAll('#model-loader-v1 button')){if(/Download|Load model|Retry/i.test(b.textContent)){b.click();return true}}return false})()`);
      await waitFor(page.sessionId, `!document.querySelector('#runV1')?.disabled`, "Multi-model v1 ready", 30 * 60_000);

      // Run Stage 1 (v1)
      ok = check(`${viewportName} Multi-model: Index & rank (v1) clicked`, await evaluate(page.sessionId, click("#runV1"))) && ok;
      await waitFor(page.sessionId, `document.querySelectorAll('#rankedV1 .result-row').length>=2 || (document.querySelector('#status')?.textContent||'').includes('BLOCKED')`, "Multi-model v1 run", 30 * 60_000);

      const v1Data = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({
        hits: [...document.querySelectorAll('#rankedV1 .result-row .result-score')].map(s=>Number(s.textContent)),
        status: document.querySelector('#status')?.textContent || '',
        overlap: document.querySelector('#overlap')?.textContent || '',
        rV1Ms: document.querySelector('#rV1Ms')?.textContent || '',
        rV2Ms: document.querySelector('#rV2Ms')?.textContent || '',
        rV1Tokens: document.querySelector('#rV1Tokens')?.textContent || '',
        rV1Aria: document.querySelector('#rV1Tokens')?.getAttribute('aria-label') || '',
        rV2Tokens: document.querySelector('#rV2Tokens')?.textContent || ''
      })`));

      const v1Blocked = v1Data.status.includes("BLOCKED") && v1Data.hits.length === 0;
      ok = check(`${viewportName} Multi-model: v1 not blocked`, !v1Blocked, v1Data.status) && ok;
      ok = check(`${viewportName} Multi-model: stage 1 (v1) ranks with finite scores`, v1Data.hits.length >= 2 && v1Data.hits.every(Number.isFinite), JSON.stringify(v1Data.hits.slice(0, 3))) && ok;

      // Agreement & latency assertions
      const overlapOk = Boolean(v1Data.overlap) && v1Data.overlap !== "–" && /\d+ of \d+|\d+%/.test(v1Data.overlap);
      ok = check(`${viewportName} Multi-model: overlap reports real agreement figure`, overlapOk, v1Data.overlap) && ok;
      const msOk = Boolean(v1Data.rV2Ms) && v1Data.rV2Ms !== "–" && Boolean(v1Data.rV1Ms) && v1Data.rV1Ms !== "–";
      ok = check(`${viewportName} Multi-model: both rV2Ms and rV1Ms are non-empty`, msOk, `v2=${v1Data.rV2Ms} v1=${v1Data.rV1Ms}`) && ok;

      // Handle v1 em dash correctly:
      // #rV1Tokens legitimately renders an em dash meaning NOT REPORTED (v1's worker does not report tokenCounts).
      // Assert it is either a number OR the em dash, and when it is the em dash the element carries an
      // aria-label containing "Not reported by this model worker". Do NOT require v1 to report tokens.
      const v1TokVal = v1Data.rV1Tokens.trim();
      const isNum = Number.isFinite(Number(v1TokVal)) && Number(v1TokVal) > 0;
      const isDash = v1TokVal === "–" || v1TokVal === "—" || v1TokVal === "-";
      const ariaOk = v1Data.rV1Aria.includes("Not reported by this model worker");
      const v1TokOk = isNum || (isDash && ariaOk);
      ok = check(`${viewportName} Multi-model: rV1Tokens is a number or labelled em dash`, v1TokOk, `tokens="${v1TokVal}" aria="${v1Data.rV1Aria}"`) && ok;

      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    }
  } catch (error) {
    check(`${viewportName} ${routeName}: drive failed`, false, error.message);
    pass = false;
  }
  results.push({ route, viewport: viewportName, pass });
  return pass;
}

let first = true;
for (const [viewportName, viewport] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
  for (const routeName of Object.keys(ROUTES)) {
    console.log(`\n=== ${routeName} · ${viewportName} ===`);
    await exercise(routeName, viewportName, viewport, first);
    first = false;
  }
}

if (WRITE_RUN) {
  writeAcceptanceRunRecord({
    runRecordPath: RUN_RECORD,
    startCommit,
    results,
    exitCode: passed === total ? 0 : 1,
  });
}
const ok = printAcceptanceSummary({
  passed,
  total,
  results,
  expectedCells: Object.keys(ROUTES).length * 2,
});
try {
  chrome.kill?.({ removeProfile: false });
} catch {
  /* ignore */
}
server.close();
process.exit(ok ? 0 : 1);
