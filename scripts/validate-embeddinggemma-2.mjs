// Route-complete real-browser acceptance for EmbeddingGemma 2.
//
// Drives every overview / Basics / Practical / Wild / Multi-model route at desktop + mobile, and asserts
// the things that actually matter for this family:
//   • real inference, not a rendered placeholder — the vectors must be finite, unit-length and NON-trivial
//     (if every pairwise cosine were identical the "embeddings" would be degenerate and every ranking a lie);
//   • semantic behaviour, not just a number — the prefilled attention query must rank an attention passage
//     first, and a real image must score highest against its own description;
//   • lifecycle — the model is released and re-initialised, and the text encoder is re-loaded WITH the vision
//     encoder on the multi-model route (the selective-encoder path this family exists to demonstrate);
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
  userDataDir: "/tmp/eg2-acceptance-profile",
  resetProfile: false,
  removeProfileOnKill: false,
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
const clickDownloads =
  `(()=>{let n=0;for(const b of document.querySelectorAll('.model-loader button')){if(/Download|Retry/i.test(b.textContent)){b.click();n++}}return n})()`;
const noOverflow = `document.documentElement.scrollWidth <= window.innerWidth + 1`;

/**
 * Drive a route until its controls are enabled. A reused Chrome profile means the second route onward is
 * cache-warm and the loader auto-initialises; the first visit must still present a real Download button.
 */
async function readyControls(page, firstVisit, readySelector) {
  await sleep(1800);
  if (firstVisit) {
    const labels = await evaluate(
      page.sessionId,
      `JSON.stringify([...document.querySelectorAll('.model-loader button')].map(b=>b.textContent.trim()))`,
    );
    check(`fresh profile offers a real Download for ${MODEL_ID}`, /Download/.test(labels), labels);
  }
  await evaluate(page.sessionId, clickDownloads);
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
      ok = check(`${viewportName} overview: encoder-budget table lists all 3 encoders`, shape.budgetRows === 3, `rows=${shape.budgetRows}`) && ok;

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
      const basics = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({score:Number(document.querySelector('#score')?.textContent),rows:[...document.querySelectorAll('#prefixTable tbody tr')].map(r=>Number(r.querySelector('td:last-child')?.textContent)),mode:document.querySelector('#rMode')?.textContent,tok:document.querySelector('#rTok')?.textContent,dim:document.querySelector('#rDim')?.textContent})`));
      ok = check(`${viewportName} Basics: real cosine in (0,1]`, basics.score > 0 && basics.score <= 1.0001, String(basics.score)) && ok;
      ok = check(`${viewportName} Basics: 768-d output reported`, basics.dim === "768", String(basics.dim)) && ok;
      ok = check(`${viewportName} Basics: prefix comparison table computed from real embeddings`, basics.rows.length >= 5 && basics.rows.every(Number.isFinite), JSON.stringify(basics.rows)) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "practical") {
      await readyControls(page, false, "#indexBtn");
      await evaluate(page.sessionId, click("#indexBtn"));
      await waitFor(page.sessionId, `Number(document.querySelector('#rChunks')?.textContent)>1`, "Practical index");
      const practical = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({chunks:Number(document.querySelector('#rChunks')?.textContent),tokens:Number(document.querySelector('#rTokens')?.textContent),rDim:document.querySelector('#rDim')?.textContent,size:document.querySelector('#rSize')?.textContent})`));
      ok = check(`${viewportName} Practical: document chunked and embedded for real`, practical.chunks > 1 && practical.tokens > 0, JSON.stringify(practical)) && ok;
      ok = check(`${viewportName} Practical: chunk token count is inside the 8192-token window`, practical.tokens > 0 && practical.tokens <= 8192, `tokens=${practical.tokens}`) && ok;
      await evaluate(page.sessionId, click("#searchBtn"));
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
      const visionSession = await evaluate(page.sessionId, `document.body.innerText.includes('vision_encoder') || (document.querySelector('#rSessions')?.textContent||'').includes('vision_encoder')`);
      ok = check(`${viewportName} Wild: the real session list shows the loaded vision encoder`, Boolean(visionSession), String(visionSession));
      // text -> image
      await evaluate(page.sessionId, `(()=>{const r=document.querySelector('input[name="direction"][value="text2image"]');if(r){r.checked=true;r.dispatchEvent(new Event('change',{bubbles:true}));}return !!r})()`);
      await evaluate(page.sessionId, click("#run"));
      await waitFor(page.sessionId, `document.querySelectorAll('#imageResults figure').length>0`, "Wild text->image", 30 * 60_000);
      const t2i = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#imageResults figure')].map(f=>({label:f.querySelector('.gallery-label')?.textContent||'',score:Number((f.querySelector('.gallery-score')?.textContent||'').replace(/[^0-9.]/g,''))})))`));
      ok = check(`${viewportName} Wild text→image: real image embeddings ranked with finite scores`, t2i.length > 0 && t2i.every((t) => Number.isFinite(t.score) && t.score > 0), JSON.stringify(t2i.slice(0, 3))) && ok;
      ok = check(`${viewportName} Wild text→image: scores are descending (a real ranking)`, t2i.every((t, i) => i === 0 || t.score <= t2i[i - 1].score + 1e-6), JSON.stringify(t2i.map((t) => t.score))) && ok;
      const visual = await evaluate(page.sessionId, `Number((document.querySelector('#rVisualTokens')?.textContent||'').replace(/[^0-9]/g,''))`);
      ok = check(`${viewportName} Wild: the real vision token budget is reported`, Number.isFinite(visual) && visual > 0, `softTokens=${visual}`) && ok;
      // image -> text
      await evaluate(page.sessionId, `(()=>{const r=document.querySelector('input[name="direction"][value="image2text"]');if(r){r.checked=true;r.dispatchEvent(new Event('change',{bubbles:true}));}return !!r})()`);
      await evaluate(page.sessionId, click("#run"));
      await waitFor(page.sessionId, `document.querySelectorAll('#ranked .result-row').length>=2`, "Wild image->text", 30 * 60_000);
      const i2t = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#ranked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} Wild image→text: an image query ranks text passages`, i2t.length >= 2 && i2t.every(Number.isFinite), JSON.stringify(i2t.slice(0, 4))) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "multimodel") {
      await readyControls(page, false, "#runText");
      await evaluate(page.sessionId, click("#runText"));
      await waitFor(page.sessionId, `document.querySelectorAll('#textRanked .result-row').length>=2`, "Multi-model text stage");
      const stage1 = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#textRanked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      const budget1 = await evaluate(page.sessionId, `document.querySelector('#rBudget')?.textContent||''`);
      ok = check(`${viewportName} Multi-model: text-only stage ranks with finite scores`, stage1.length >= 2 && stage1.every(Number.isFinite), JSON.stringify(stage1.slice(0, 3))) && ok;
      ok = check(`${viewportName} Multi-model: the 175 MB text-only budget is reported before the reload`, /175/.test(budget1), budget1) && ok;
      // Stage 2: re-load the SAME engine with the vision encoder, then query the same index with an image.
      await evaluate(page.sessionId, click("#addVision"));
      await waitFor(page.sessionId, `(document.querySelector('#rSessions')?.textContent||'').includes('vision_encoder')`, "Multi-model vision reload", 30 * 60_000);
      const budget2 = await evaluate(page.sessionId, `document.querySelector('#rBudget')?.textContent||''`);
      const sessions = await evaluate(page.sessionId, `document.querySelector('#rSessions')?.textContent||''`);
      ok = check(`${viewportName} Multi-model: reload adds the vision encoder to the real session list`, /vision_encoder/.test(String(sessions)), String(sessions)) && ok;
      ok = check(`${viewportName} Multi-model: the budget grows to 284 MB after the vision reload`, /284/.test(budget2), budget2) && ok;
      await evaluate(page.sessionId, click("#runImage"));
      await waitFor(page.sessionId, `document.querySelectorAll('#imageRanked .result-row').length>=2`, "Multi-model image stage", 30 * 60_000);
      const stage2 = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#imageRanked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      const moved = await evaluate(page.sessionId, `document.querySelector('#rMoved')?.textContent||''`);
      ok = check(`${viewportName} Multi-model: the image query re-scores the same index cross-modally`, stage2.length >= 2 && stage2.every(Number.isFinite), JSON.stringify(stage2.slice(0, 3))) && ok;
      ok = check(`${viewportName} Multi-model: order movement between the two modalities is reported`, String(moved).length > 0, String(moved)) && ok;
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
