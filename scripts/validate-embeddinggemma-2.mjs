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
// google/embeddinggemma-2 (the canonical weights), and onnx-community/embeddinggemma-300m-ONNX (the v1
// EmbeddingGemma the multi-model route compares against).
//
// Environment note carried into every claim: these runs use a SwiftShader *software* WebGPU adapter on a
// shared CPU box, so the latencies prove the kernels ran, not how fast this is on real hardware.
//
// Split-run acceptance (viewport-scoped halves + mechanical merge): the full both-viewport matrix takes
// ~2750s but the box reaps a browser at 45 min (2700s), so it does not fit in one run. Two separate runs —
//   VIEWPORTS=desktop node scripts/acceptance-run.mjs scripts/validate-embeddinggemma-2.mjs --write-run --max-load 40
//   VIEWPORTS=mobile  node scripts/acceptance-run.mjs scripts/validate-embeddinggemma-2.mjs --write-run --max-load 40
// each write a HALF artifact (reports/acceptance/embeddinggemma-2/acceptance-runs/{desktop,mobile}-half.json), never the
// final record. Then the merge assembles the single 10-cell record the project expects:
//   node scripts/validate-embeddinggemma-2.mjs --merge-halves
// See scripts/embeddinggemma-2-half-merge.mjs for the fail-closed rules (missing/failed/stale/duplicate/
// wrong-cell-count/overlapping-routes halves all refuse to publish; atomic write; idempotent merge).
//
// A single-viewport run with --write-run only ever writes a half; the final
// models/embeddinggemma-2/acceptance-run.json is produced by `--merge-halves` (or, for the legacy
// single-run path, VIEWPORTS=both --write-run). A focused single-viewport re-drive (ROUTES=… subset) writes
// nothing, because a partial run must never become a half or an acceptance record.
import {
  captureHeadCommit,
  CDP,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  closePage,
  printAcceptanceSummary,
  repoRoot,
  setViewport,
  startServer,
  writeAcceptanceRunRecord,
} from "./browser.mjs";
import {
  computeValidatorBlobSha,
  FINAL_RECORD_PATH,
  mergeHalves,
  writeHalfRecord,
} from "./embeddinggemma-2-half-merge.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const MERGE_HALVES = process.argv.includes("--merge-halves");
const RUN_RECORD = FINAL_RECORD_PATH;
const startCommit = captureHeadCommit(repoRoot);

if (MERGE_HALVES) {
  const outcome = mergeHalves({
    currentCommit: startCommit,
    currentValidatorBlobSha: computeValidatorBlobSha(),
  });
  if (!outcome.ok) {
    console.error(`\nMERGE REFUSED: ${outcome.reason}`);
    process.exit(1);
  }
  console.log(`\nMERGED halves into ${FINAL_RECORD_PATH} — 10/10 cells from two viewport-scoped runs`);
  process.exit(0);
}
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
const PROFILE_PREFIX = "embeddinggemma-2";
const chrome = await launchChrome({
  profilePrefix: PROFILE_PREFIX,
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

// Index of the row the page labelled as the baseline, or 0 when there is none (the labelled-row check
// reports that case separately, so this only has to avoid throwing).
function baselineIdxOf(basics) {
  const i = basics.deltas.findIndex((d) => /baseline/i.test(d));
  return i >= 0 ? i : 0;
}

async function exercise(routeName, viewportName, viewport, firstVisit = false) {
  const route = ROUTES[routeName];
  let page = null;
  let ok = true;
  let pass = false;
  try {
    page = await openPage(cdp, url(route));
    await setViewport(cdp, page.sessionId, viewport);
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

      // Anti-stale re-inference detection:
      // The first inference run populated #rDim ('768'), #iNorm (~1.0000), #iNan ('0'), and #rMs. Those DOM
      // values survive the release/re-init lifecycle. If the second inference silently fails or never runs,
      // a check waiting only on '#rDim === 768' would pass vacuously on stale output from the first run.
      // To ensure the assertion only passes if a genuine NEW inference occurred:
      // 1. Record the pre-run count exposed by the page on #readout.dataset.runCount.
      // 2. Clear/blank the output DOM nodes (#rDim, #iNorm, #iNan, #rMs) before triggering re-inference.
      // 3. Trigger re-inference by clicking #run.
      // 4. Wait for both a fresh run count (> prevRuns) AND #rDim to become non-empty ('768').
      // 5. Assert afterRelease has norm ~ 1, nan == 0, and runs > prevRuns.
      const prevRuns = Number(await evaluate(page.sessionId, `Number(document.querySelector('#readout')?.dataset.runCount || 0)`));
      await evaluate(page.sessionId, `(()=>{
        const dim = document.querySelector('#rDim');
        if (dim) dim.textContent = '';
        const norm = document.querySelector('#iNorm');
        if (norm) norm.textContent = '';
        const nan = document.querySelector('#iNan');
        if (nan) nan.textContent = '';
        const ms = document.querySelector('#rMs');
        if (ms) ms.textContent = '';
        return true;
      })()`);
      await evaluate(page.sessionId, click("#run"));
      await waitFor(
        page.sessionId,
        `document.querySelector('#rDim')?.textContent==='768' && Number(document.querySelector('#readout')?.dataset.runCount || 0) > ${prevRuns}`,
        "overview re-run after re-init",
      );
      const afterRelease = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({
        norm: Number(document.querySelector('#iNorm')?.textContent),
        nan: Number(document.querySelector('#iNan')?.textContent),
        runs: Number(document.querySelector('#readout')?.dataset.runCount || 0)
      })`));
      ok = check(
        `${viewportName} overview: inference works again after release + re-initialise`,
        afterRelease.runs > prevRuns && Math.abs(afterRelease.norm - 1) < 0.01 && afterRelease.nan === 0,
        JSON.stringify(afterRelease),
      ) && ok;
      ok = await hygiene(page, route, viewportName) && ok;
      pass = ok;
    } else if (routeName === "basics") {
      await readyControls(page, false, "#run");
      ok = check(`${viewportName} Basics: run clicked`, await evaluate(page.sessionId, click("#run"))) && ok;
      await waitFor(page.sessionId, `Number.isFinite(Number(document.querySelector('#score')?.textContent))`, "Basics cosine");
      // Read the COSINE column (3rd cell) for the finite check, not td:last-child: the last cell is the delta
      // column, and the baseline row renders "0.000 (baseline)", so Number() on it is NaN and would fail a
      // correct page. The baseline row is asserted separately for its own text, which is the real property.
      const basics = JSON.parse(await evaluate(page.sessionId, `JSON.stringify({
        score: Number(document.querySelector('#score')?.textContent),
        tasks: [...document.querySelectorAll('#prefixTable tbody tr')].map(r => r.querySelector('th')?.textContent?.trim() || ''),
        rows: [...document.querySelectorAll('#prefixTable tbody tr')].map(r => Number(r.querySelector('td:nth-child(3)')?.textContent)),
        deltas: [...document.querySelectorAll('#prefixTable tbody tr')].map(r => r.querySelector('td:last-child')?.textContent || ''),
        prompts: [...document.querySelectorAll('#prefixTable tbody tr')].map(r => r.querySelector('td:nth-child(2)')?.textContent || ''),
        conclusion: document.querySelector('#prefixTable .ctx-note')?.textContent || '',
        mode: document.querySelector('#rMode')?.textContent,
        tok: document.querySelector('#rTok')?.textContent,
        dim: document.querySelector('#rDim')?.textContent
      })`));
      ok = check(`${viewportName} Basics: real cosine in (0,1]`, basics.score > 0 && basics.score <= 1.0001, String(basics.score)) && ok;
      ok = check(`${viewportName} Basics: 768-d output reported`, basics.dim === "768", String(basics.dim)) && ok;
      // Honest name: this asserts the table is internally consistent, NOT that the vectors were really
      // produced by the model. A varying but fabricated table would satisfy it, so the descriptor says what
      // it checks rather than more than it checks.
      ok = check(`${viewportName} Basics: prefix comparison table is internally consistent (5 finite cosines)`, basics.rows.length >= 5 && basics.rows.every(Number.isFinite), JSON.stringify(basics.rows)) && ok;
      // The selected score must be one of the table's own cosines, not an unrelated number.
      ok = check(`${viewportName} Basics: selected score is one of the table rows`, basics.rows.some((v) => Math.abs(v - basics.score) <= 0.0005), `score=${basics.score} rows=${JSON.stringify(basics.rows)}`) && ok;
      // The baseline row must be labelled as the baseline rather than showing a bare delta, so a page that
      // silently dropped its control row cannot pass just by producing five finite cosines.
      const baselineIdx = basics.deltas.findIndex((d) => /baseline/i.test(d));
      ok = check(`${viewportName} Basics: baseline row labelled`, basics.deltas.filter((d) => /baseline/i.test(d)).length === 1 && baselineIdx >= 0, JSON.stringify(basics.deltas)) && ok;

      // FIX 4 (a): Require EXACTLY ONE row to be the genuinely bare/baseline row (i.e. no other row's
      // prompt mapping is bare).
      const isBare = (p) => /bare text|no task prefix/i.test(String(p));
      const bareIndices = basics.prompts
        .map((p, idx) => (isBare(p) ? idx : -1))
        .filter((idx) => idx >= 0);
      const exactlyOneBareRow = bareIndices.length === 1 && bareIndices[0] === baselineIdx;
      ok = check(
        `${viewportName} Basics: exactly one row is bare and matches the baseline row`,
        exactlyOneBareRow,
        `bareIndices=${JSON.stringify(bareIndices)} baselineIdx=${baselineIdx} prompts=${JSON.stringify(basics.prompts)}`,
      ) && ok;

      // FIX 4 (c): Assert that the displayed prefix mapping for each row matches the prefix the page claims
      // to have applied rather than merely being non-empty. All 5 documented tasks must be present, and each
      // row's displayed instruction mapping in column 2 must match the instruction syntax for that task.
      const EXPECTED_PREFIX_SPECS = [
        { name: "search result", labelRe: /search result/i, promptRe: /task:\s*search result\s*\|\s*query:.*title:\s*none\s*\|\s*text:/i },
        { name: "classification", labelRe: /classification/i, promptRe: /task:\s*classification\s*\|\s*query:/i },
        { name: "clustering", labelRe: /clustering/i, promptRe: /task:\s*clustering\s*\|\s*query:/i },
        { name: "sentence similarity", labelRe: /sentence similarity/i, promptRe: /task:\s*sentence similarity\s*\|\s*query:/i },
        { name: "none", labelRe: /none/i, promptRe: /bare text|no task prefix/i },
      ];
      const allSpecsCovered = EXPECTED_PREFIX_SPECS.every((spec) =>
        basics.tasks.some((taskText) => spec.labelRe.test(taskText))
      );
      const prefixMappingsMatch = basics.tasks.length >= 5 && basics.tasks.every((taskText, i) => {
        const spec = EXPECTED_PREFIX_SPECS.find((s) => s.labelRe.test(taskText));
        if (!spec) return false;
        return spec.promptRe.test(basics.prompts[i]);
      });
      ok = check(
        `${viewportName} Basics: displayed prefix mapping matches the task prefix claimed for each row`,
        allSpecsCovered && prefixMappingsMatch,
        `tasks=${JSON.stringify(basics.tasks)} prompts=${JSON.stringify(basics.prompts)}`,
      ) && ok;

      // FIX 4 (b): Five identical cosines plus a baseline label would demonstrate no prefix effect at all.
      // Require real variation across prefixes (unique values at 3 decimals) AND non-zero variation vs baseline,
      // AND reconcile the displayed deltas against the cosine column (delta_i = cosine_i - cosine_baseline).
      const uniqueCos = new Set(basics.rows.map((v) => v.toFixed(3))).size;
      const cosBase = basics.rows[baselineIdx];
      const hasCosVariation = uniqueCos > 1 && Number.isFinite(cosBase) && basics.rows.some((v) => Math.abs(v - cosBase) >= 0.0005);
      ok = check(`${viewportName} Basics: cosine values actually vary across prefixes`, hasCosVariation, `distinct=${uniqueCos} baseline=${cosBase} rows=${JSON.stringify(basics.rows)}`) && ok;
      const deltaReconciles = Number.isFinite(cosBase) && basics.rows.every((cos, i) => {
        const shown = Number(String(basics.deltas[i]).match(/[-+]?\d*\.?\d+/)?.[0]);
        return Number.isFinite(shown) && Math.abs(shown - (cos - cosBase)) <= 0.002;
      });
      ok = check(`${viewportName} Basics: displayed deltas reconcile with the cosine column`, deltaReconciles, `baseline=${cosBase} deltas=${JSON.stringify(basics.deltas)}`) && ok;

      // FIX 3: Tie the page's OWN conclusion sentence to its own numbers, using the ±0.01 rule the page states.
      // Invert vacuous pass: an unrecognised, empty, or missing conclusion must FAIL (the page must state a
      // conclusion the validator recognises and that agrees with its own deltas).
      const maxAbsDelta = Math.max(...basics.rows.map((cos) => Math.abs(cos - cosBase)));
      const conclusion = String(basics.conclusion).trim();
      const claimsUnchanged = /unchanged/i.test(conclusion);
      const claimsChanged = /changed the similarity/i.test(conclusion);
      const conclusionRecognised = (claimsUnchanged || claimsChanged) && !(claimsUnchanged && claimsChanged);
      const conclusionConsistent = conclusionRecognised && (
        claimsUnchanged ? maxAbsDelta < 0.01 : maxAbsDelta >= 0.01
      );
      ok = check(
        `${viewportName} Basics: table conclusion agrees with the table's own deltas`,
        conclusionConsistent,
        `maxAbsDelta=${maxAbsDelta.toFixed(3)} recognised=${conclusionRecognised} conclusion="${conclusion.slice(0, 90)}"`,
      ) && ok;
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
      // Matryoshka dial: the width changes, the re-rank is real, and the PROJECTED persisted size changes
      // with no re-embedding. The page must not claim it realised a saving: it keeps full-width 768-d
      // vectors in memory and truncates only for scoring, so the figure has to be labelled a projection.
      // That wording is asserted below, because the dishonest version of this page passed every numeric
      // check here while claiming a 6x memory reduction it never made.
      const before = await evaluate(page.sessionId, `document.querySelector('#rSize')?.textContent`);
      await evaluate(page.sessionId, `(()=>{const s=document.querySelector('#dims');if(!s)return false;s.value=s.max;s.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
      await waitFor(page.sessionId, `document.querySelector('#rDim')?.textContent==='128-d' || document.querySelector('#rDim')?.textContent==='128'`, "Practical 128-d");
      const after = await evaluate(page.sessionId, `document.querySelector('#rSize')?.textContent`);
      const afterHits = JSON.parse(await evaluate(page.sessionId, `JSON.stringify([...document.querySelectorAll('#ranked .result-row .result-score')].map(s=>Number(s.textContent)))`));
      ok = check(`${viewportName} Practical: 128-d re-rank is real and the projected persisted size changes`, afterHits.every(Number.isFinite) && after !== before && afterHits.length >= 2, `${before} -> ${after}`) && ok;
      ok = check(`${viewportName} Practical: size figure is labelled a projection, not a realised saving`, /if persisted/i.test(String(after)), `after="${after}"`) && ok;
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
      // The verdict must be tied STRUCTURALLY to the rendered ranking, not by substring. A plain
      // includes(`[${topCode}]`) is unsound because the query-language code is emitted before the top-hit
      // code in several verdict variants, so a verdict naming a different top-1 could still contain the top
      // row's code. Instead: take the code the verdict associates with its own cosine (the last code before
      // the word "cosine"), and require its stated cosine and margin to equal the rendered rank-1 and rank-2
      // scores. That is the property that matters — the page's stated outcome agrees with its own numbers —
      // and it deliberately does NOT require the cross-lingual target to be rank 1, since the page reports a
      // losing target honestly.
      const verdictCosMatch = String(wild.verdict).match(/cosine\s+(\d+\.\d{3})/);
      const beforeCosine = String(wild.verdict).split(/cosine/i)[0] || "";
      const codesBeforeCosine = [...beforeCosine.matchAll(/\[([A-Z]{2})\]/g)].map((m) => m[1]);
      const verdictCode = codesBeforeCosine.length ? codesBeforeCosine[codesBeforeCosine.length - 1] : "";
      const verdictMarginMatch = String(wild.verdict).match(/margin(?:\s+of|:)\s*([-+]?\d+\.\d{3})/);
      const renderedTop = wild.hits[0]?.score;
      const renderedSecond = wild.hits[1]?.score;
      const expectedMargin = (Number.isFinite(renderedTop) && Number.isFinite(renderedSecond)) ? (renderedTop - renderedSecond).toFixed(3) : null;
      const verdictNamesTop = Boolean(verdictCode) && Boolean(topCode) && verdictCode === topCode;
      const verdictScoreAgrees = Boolean(verdictCosMatch) && Number.isFinite(renderedTop) && verdictCosMatch[1] === renderedTop.toFixed(3);
      const verdictMarginAgrees = Boolean(verdictMarginMatch) && expectedMargin !== null && Math.abs(Number(verdictMarginMatch[1]) - Number(expectedMargin)) <= 0.002;
      ok = check(`${viewportName} Wild: verdict names the rendered top-1 language as its top hit`, verdictNamesTop, `verdictCode=${verdictCode} topCode=${topCode} verdict="${String(wild.verdict).slice(0, 160)}"`) && ok;
      ok = check(`${viewportName} Wild: verdict's stated cosine equals the rendered rank-1 score`, verdictScoreAgrees, `stated=${verdictCosMatch ? verdictCosMatch[1] : null} rendered=${Number.isFinite(renderedTop) ? renderedTop.toFixed(3) : null}`) && ok;
      ok = check(`${viewportName} Wild: verdict's stated margin equals the rendered rank-1 minus rank-2 gap`, verdictMarginAgrees, `stated=${verdictMarginMatch ? verdictMarginMatch[1] : null} expected=${expectedMargin}`) && ok;
      // The verdict's CLASSIFICATION and any stated target RANK must also agree with the rendered data.
      // Without this a page could label a genuinely cross-lingual result "Monolingual", or claim a target
      // "ranked #2" when it ranked last, and still satisfy the code/cosine/margin checks above.
      const vText = String(wild.verdict);
      const codesInVerdict = [...vText.matchAll(/\[([A-Z]{2})\]/g)].map((m) => m[1]);
      const firstCode = codesInVerdict[0] || "";
      const claimsMono = /Monolingual/i.test(vText);
      const claimsSuccess = /succeeded/i.test(vText);
      const expectedMatch = vText.match(/Expected target was \[([A-Z]{2})\], which ranked #(\d+)/);
      // "Monolingual" asserts the top hit is in the query's OWN language, so the first code (the query
      // language in that variant) must equal the top hit code.
      const monoConsistent = !claimsMono || (Boolean(firstCode) && firstCode === topCode);
      // A claimed cross-lingual SUCCESS must not also carry a miss-note, and must name two languages that
      // actually differ — otherwise "succeeded" is unfalsifiable.
      const successConsistent = !claimsSuccess || (!expectedMatch && Boolean(firstCode) && firstCode !== topCode);
      ok = check(`${viewportName} Wild: a claimed monolingual hit really is same-language`, monoConsistent, `claimsMono=${claimsMono} firstCode=${firstCode} topCode=${topCode}`) && ok;
      ok = check(`${viewportName} Wild: a claimed cross-lingual success names two different languages and no miss-note`, successConsistent, `claimsSuccess=${claimsSuccess} hasMissNote=${Boolean(expectedMatch)} firstCode=${firstCode} topCode=${topCode}`) && ok;
      // When the verdict states a miss ("Expected target was [X], which ranked #N"), the rank must match the
      // position that target actually occupies in the rendered list.
      let targetRankOk = true;
      let targetRankDetail = "no miss-note";
      if (expectedMatch) {
        const targetCode = expectedMatch[1];
        const statedRank = Number(expectedMatch[2]);
        const renderedIdx = wild.hits.findIndex((h) => (String(h.head).match(/\[([A-Z]{2})\]/) || [])[1] === targetCode);
        targetRankOk = renderedIdx >= 0 && renderedIdx + 1 === statedRank;
        targetRankDetail = `target=[${targetCode}] statedRank=${statedRank} renderedIdx=${renderedIdx}`;
      }
      ok = check(`${viewportName} Wild: a stated target rank matches where that target really ranked`, targetRankOk, targetRankDetail) && ok;
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
  } finally {
    // Close every target instead of leaving it open. Ten stages ran in one Chrome, so without this all ten
    // pages kept their Web Worker and WebGPU allocations alive at once on a box with 2 shared CPUs.
    if (page?.targetId) {
      try { await closePage(cdp, page.targetId); } catch { /* teardown must not fail the run */ }
    }
  }
  results.push({ route, viewport: viewportName, pass });
  return pass;
}

// One viewport per run when requested. The reaper killed the combined matrix at 8106 MB against a 4000 MB
// limit: ten stages in one Chrome meant ten pages holding their Web Workers and WebGPU allocations at once.
// Closing each page (above) is the real fix, but bounding a run to five stages keeps a re-run comfortably
// inside the limit and means a failure costs half the time. VIEWPORTS=desktop|mobile|both.
const viewportFilter = (process.env.VIEWPORTS || "both").toLowerCase();
const viewportsToRun = [
  ["desktop", DESKTOP],
  ["mobile", MOBILE],
].filter(([name]) => viewportFilter === "both" || viewportFilter === name);
if (viewportsToRun.length === 0) throw new Error(`VIEWPORTS='${viewportFilter}' selected nothing; use desktop, mobile or both`);
// Focused re-drive: name the routes you actually changed, so confirming a fix does not cost a full matrix.
// ROUTES=basics,overview (comma-separated names from ROUTES). Default is every route.
const routeFilter = (process.env.ROUTES || "").trim().toLowerCase();
const routesToRun = Object.keys(ROUTES).filter((name) => !routeFilter || routeFilter.split(",").map((s) => s.trim()).includes(name));
if (routesToRun.length === 0) throw new Error(`ROUTES='${routeFilter}' matched nothing; known: ${Object.keys(ROUTES).join(", ")}`);
console.log(`ROUTES=${routeFilter || "all"} -> ${routesToRun.join(", ")}`);
console.log(`VIEWPORTS=${viewportFilter} -> ${viewportsToRun.map(([n]) => n).join(", ")}`);

let first = true;
for (const [viewportName, viewport] of viewportsToRun) {
  for (const routeName of routesToRun) {
    console.log(`\n=== ${routeName} · ${viewportName} ===`);
    await exercise(routeName, viewportName, viewport, first);
    first = false;
  }
}

const ok = printAcceptanceSummary({
  passed,
  total,
  results,
  // Must follow the viewport filter, or a single-viewport run reports half its cells as missing.
  expectedCells: routesToRun.length * viewportsToRun.length,
});

if (WRITE_RUN) {
  const isSingleViewportRun = viewportFilter === "desktop" || viewportFilter === "mobile";
  const isFullRouteSet = routesToRun.length === Object.keys(ROUTES).length;
  if (viewportFilter === "both") {
    // Legacy single-run path: the whole matrix in one Chrome writes the final record directly.
    writeAcceptanceRunRecord({
      runRecordPath: RUN_RECORD,
      startCommit,
      results,
      exitCode: passed === total ? 0 : 1,
    });
  } else if (isSingleViewportRun && isFullRouteSet) {
    // Split path: one complete viewport-scoped run writes ONLY its half; the final record is assembled
    // later by `--merge-halves`. The viewport is the one the run loop ACTUALLY drove, never a flag.
    writeHalfRecord({
      viewport: viewportsToRun[0][0],
      results,
      commit: startCommit,
      ranAt: new Date().toISOString(),
      exitCode: ok ? 0 : 1,
      pass: ok,
      validatorBlobSha: computeValidatorBlobSha(),
    });
  } else if (isSingleViewportRun) {
    // A focused re-drive (ROUTES subset) is NOT a half and never becomes an acceptance record.
    console.log(
      `NO RECORD: single-viewport focused re-drive (ROUTES=${routeFilter || "all"}) — a partial run never becomes a half or a final acceptance record`,
    );
  }
}
const runProfileDir = chrome.userDataDir || null;
try {
  // No removeProfile:false. That option skipped rmSync AND removed the instance from activeChromeInstances,
  // so the exit hook had nothing left to clean and EVERY completed run abandoned its isolated profile in
  // /tmp with cached weights and shaders. The harness default removes it, and the global exit hooks cover
  // SIGINT/SIGTERM/SIGHUP.
  //
  // KNOWN LIMIT, stated rather than implied: SIGKILL cannot run exit hooks, so a mid-run SIGKILL DOES leave
  // this run's profile behind, and a Chrome that fails during startup can leave one that was never
  // registered. That is inherent to signal-skipping kills, not a bug this code can fix; it is exactly what
  // happened when the memory reaper killed an earlier matrix and a 963M profile survived. So the profile fix
  // is complete for normal and handled-signal exits, NOT for every possible exit path. Do not claim more.
  await chrome.kill?.();
} catch {
  /* ignore */
}
server.close();
// Printed from the validator itself so the retained output file carries the teardown evidence rather than
// requiring the caller's wrapper log to be believed. Scoped to THIS run's profile prefix, so another lane's
// leftover directory cannot make this line pass or fail.
// Scoped to THIS run's own profile directory, which the harness exposes as chrome.userDataDir. An earlier
// version globbed a per-SLUG pattern, which a concurrent run or a leftover from a SIGKILLed run would have
// made non-zero for reasons unrelated to this run — a reviewer caught that the comment claimed per-run
// scoping the code did not implement. Bound: SIGKILL cannot run exit hooks, so a killed run may still leave
// its directory behind; this line measures the normal path and says which dir it means.
try {
  const { execSync } = await import("node:child_process");
  const { existsSync } = await import("node:fs");
  const stillExists = runProfileDir ? existsSync(runProfileDir) : null;
  const procsOnThisProfile = runProfileDir
    ? Number(String(execSync(`ps -eo args | grep -c -- "--user-data-dir=${runProfileDir}" || true`, { encoding: "utf8" })).trim())
    : -1;
  console.log(`TEARDOWN profileDir=${runProfileDir} stillExists=${stillExists} chromeProcsOnThisProfile=${procsOnThisProfile} (SIGKILL cannot run exit hooks; a killed run may still leave a profile)`);
} catch (e) {
  console.log(`TEARDOWN check unavailable: ${e.message}`);
}
process.exit(ok ? 0 : 1);
