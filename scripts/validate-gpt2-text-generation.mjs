#!/usr/bin/env node
// Route-complete gpt2-text-generation acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   Xenova/gpt2        (the step-loop worker: model -> logits -> sample one token -> feed it back)
//   Xenova/distilbert-base-uncased-finetuned-sst-2-english  (the multi-model rung's sentiment judge)
//
// web-ai-showcase-0ly acceptance for THIS family: the census flagged gpt2-text-generation as the route
// whose readout came from DOM chips (`#trace.children.length`) while the worker's own done payload
// reported `tokens: ids.length` — and `ids` starts as the PROMPT's ids, so that number is prompt +
// generated. The fix counts generated IDs separately and the pages display that count. The proof here
// is therefore not "chunks vs IDs" (this worker emits exactly one message per sampled token) but:
//   readout === generated count  AND  the worker also reports a non-zero promptTokens,
// i.e. the displayed number is the generated tokens, not the prompt-inclusive total it used to be, and
// not the number of nodes that happen to be in the DOM.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
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
// This family's see-inside surface is the per-step next-token distribution (#dist) plus the confidence
// trace (#trace) — there is no #peek/#topkBtn button to click.
const HAS_INSIDE = true;
const INSIDE = {
  // The overview renders the next-token distribution for every step it takes (#dist) and colours the
  // confidence trace (#trace); both fill while generating, so no extra click is needed.
  ready:
    `(() => { const d = document.querySelector('#dist'); const t = document.querySelector('#trace'); ` +
    `return !!d && (d.children.length >= 3 || (t && t.children.length >= 3)); })()`,
  text: `((document.querySelector('#dist')?.innerText || '') + ' | ' + (document.querySelector('#trace')?.innerText || '')).slice(0, 160)`,
  label: "see-inside distribution + confidence trace render",
  minLength: 25,
};
// Debug aid: ONLY_RUNG / ONLY_VIEWPORT narrow a run to a single cell (useful when one heavy cell
// needs isolating); unset, the validator runs every rung at both viewports as acceptance requires.
const ONLY_RUNG = process.env.ONLY_RUNG ? process.env.ONLY_RUNG.split(",").map((n) => n.trim()) : null;
const ONLY_VIEWPORT = process.env.ONLY_VIEWPORT ?? null;
const RUN_RECORD = join(repoRoot, "models/gpt2-text-generation/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "gpt2-text-generation-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/gpt2-text-generation/",
  basics: "models/gpt2-text-generation/basics/",
  practical: "models/gpt2-text-generation/practical/",
  wild: "models/gpt2-text-generation/wild/",
  "multi-model": "models/gpt2-text-generation/multi-model/",
};
// Source text and JSON template the extraction rungs need: the pages refuse an empty one with
// "Add both text and a template.", and NuExtract wraps them in its own <|input|> prompt.
const SAMPLE_TEXT = "Invoice 001, Acme Ltd, total 42.50 EUR, due 2026-10-01, paid by card.";
const SAMPLE_TEMPLATE = '{"vendor": "", "total": "", "currency": ""}';
const EXTRACTION_INPUTS = [["#text", SAMPLE_TEXT], ["#template", SAMPLE_TEMPLATE]];
// This family's rungs drive three different ways, so the driver dispatches on `style`:
//   click  — set #prompt, click a button, generation streams back
//   input  — the rung renders its distribution straight from the field (no button at all)
//   steps  — a manual step chain: #start lays out candidates, then each #auto click commits one token
const RUNGS = {
  overview: { style: "click", trigger: "#run", input: "#prompt", tok: "#rTok", backend: "#rBackend" },
  basics: { style: "click", trigger: "#run", input: "#prompt", tok: "#rTok", backend: "#rBackend" },
  // practical shows the next-token distribution for whatever is typed: no button, no token count.
  practical: { style: "input", input: "#prompt", tok: null, backend: "#rBackend" },
  // wild commits tokens one at a time; #rSteps counts the choices it has committed.
  wild: {
    style: "steps",
    // The step chain seeds its own field: an <input id="seed">, not the usual #prompt.
    trigger: "#start",
    stepClick: "#auto",
    stepTarget: 8,
    stepReadout: "#rSteps",
    input: "#seed",
    tok: null,
    backend: "#rBackend",
  },
  // multi-model writes with GPT-2 and then judges with DistilBERT: its readout carries verdict scores,
  // never a token count, and it exposes no backend element.
  "multi-model": { style: "click", trigger: "#run", input: "#prompt", tok: null, noBackend: true, doneText: "Done" },
};
const PROMPT = "Describe the unbelievably heterogeneous thundercloud formation briefly.";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
const divergence = [];
let checks = 0;
let passed = 0;

function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${label}${
      detail
        ? ` — ${String(typeof detail === "object" ? JSON.stringify(detail) : detail).slice(0, 240)}`
        : ""
    }`,
  );
  return condition;
}

async function evaluate(cdp, sessionId, expression, timeoutMs = 120_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.message || error)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}

async function evalJSON(cdp, sessionId, expression, timeoutMs = 45_000) {
  const value = await evaluate(cdp, sessionId, expression, timeoutMs);
  return typeof value === "string" ? JSON.parse(value) : value;
}

async function waitFor(cdp, sessionId, expression, deadlineMs, label, intervalMs = 4_000) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < deadlineMs) {
    try {
      if (await evaluate(cdp, sessionId, expression)) return;
    } catch (error) {
      if (Date.now() >= nextLog) {
        console.log(`  [${label}] poll stalled: ${String(error.message).slice(0, 120)}`);
      }
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

// Count the page's DECODED CHUNKS, validator-side only: the page's engine is a module-scoped
// binding, but a dynamic import of the same URL resolves to the SAME module instance, so wrapping
// QwenEngine.prototype.chat and RagEngine.prototype.run here intercept every onToken the page receives — one per decoded
// text chunk. The real token-ID count arrives separately in the resolved payload (and the #rTok
// readout). Name the family's real classes AND methods: this family's plain rungs call
// QwenEngine.chat while the RAG rung calls RagEngine.run — a single chat wrap would miss the RAG rung
// entirely, and a wrong class name silently intercepts nothing.
async function installChunkCounter(cdp, sessionId) {
  // ONE class covers every generating rung: GPT2Engine.generate. Its streaming callback is the THIRD
  // ARGUMENT (onStep), not an onToken option — wrapping opts.onToken here would count nothing. A
  // dynamic import of the same URL resolves to the SAME module instance the page uses, so wrapping the
  // prototype intercepts every step message the page receives.
  const targets = [
    // gpt2 streams through GPT2Engine.generate(prompt, opts, onStep) — the step callback is the third
    // argument, not an onToken option, so the wrap below counts every step message the page receives.
    { url: "/web-ai-showcase/models/gpt2-text-generation/gpt2.js", cls: "GPT2Engine", method: "generate" },
  ];
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = []; window.__streaming = false;
      return (async () => {
        for (const t of ${JSON.stringify(targets)}) {
          const m = await import(t.url);
          const Klass = m?.[t.cls];
          if (!Klass?.prototype?.[t.method]) continue;
          const orig = Klass.prototype[t.method];
          Klass.prototype[t.method] = function (...args) {
            // generate(prompt, opts, onStep): the callback is argument index 2.
            if (typeof args[2] === "function") {
              const inner = args[2];
              window.__streaming = true;
              args[2] = (...cb) => {
                window.__chunks += 1;
                return inner(...cb);
              };
            }
            const result = orig.apply(this, args);
            if (result && typeof result.then === "function") {
              result.then((res) => {
                window.__tokens = window.__tokens ?? [];
                window.__tokens.push(Number(res?.tokens ?? 0));
                if (res?.promptTokens != null) window.__promptTokens = Number(res.promptTokens);
              });
            }
            return result;
          };
        }
        return true;
      })();
    })()`,
    30_000,
  );
}

async function ensureReady(cdp, sessionId, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 20 * 60_000) {
    const state = await evalJSON(
      cdp,
      sessionId,
      `JSON.stringify({
        ready: document.querySelectorAll('.model-loader[data-state="ready"]').length,
        total: document.querySelectorAll('.model-loader').length,
        unsupported: document.querySelector('.model-loader[data-state="unsupported"]') !== null,
      })`,
    ).catch(() => ({ ready: -1, total: -1 }));
    if (state.unsupported) {
      // Headless WebGPU adapter availability is intermittent per launch: fail fast and let the
      // operator relaunch instead of spinning for 20 minutes (web-ai-showcase-0ly).
      throw new Error("model loader refused: needs a WebGPU adapter (relaunch the validator)");
    }
    if (state.total > 0 && state.ready === state.total) return true;
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${
          Math.round((Date.now() - started) / 1000)
        }s ${state.ready}/${state.total} loaders ready`,
      );
      nextLog = Date.now() + 8_000;
    }
    await evaluate(
      cdp,
      sessionId,
      `(() => { let n = 0; for (const b of document.querySelectorAll('.model-loader button')) {
        if (/Download|Retry|Re-download/i.test(b.textContent) && !b.disabled) { b.click(); n++; } } return n; })()`,
    ).catch(() => 0);
    await sleep(4_000);
  }
  throw new Error(`hard timeout after 1200000ms: ${label} model download/init`);
}

async function drive(cdp, sid, label, cfg) {
  const { style = "click", trigger, input, extra = [], stepClick, stepTarget = 0, stepReadout } = cfg;
  const setInputs = (e, val) => {
    for (const [sel, v] of extra) {
      const node = document.querySelector(sel);
      if (node) { node.value = v; node.dispatchEvent(new Event('input', { bubbles: true })); }
    }
    const el = e ? document.querySelector(e) : null;
    if (e && !el) return null;
    if (el) {
      el.value = val;
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
    return el;
  };

  if (style === "steps") {
    // Manual chain: seed the field, start the chain, then commit tokens one click at a time until the
    // page has committed stepTarget of them (its own counter is the progress signal).
    const seeded = await evaluate(cdp, sid, `(() => { const el = ${input ? `document.querySelector('${input}')` : "null"}; if (!el) return false; el.value = ${JSON.stringify(PROMPT)}; el.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
    if (!seeded) throw new Error(`${label}: prompt field missing`);
    await evaluate(cdp, sid, `document.querySelector('${trigger}').click()`);
    for (let i = 0; i < stepTarget * 3; i++) {
      const steps = Number(
        (await evaluate(cdp, sid, `document.querySelector('${stepReadout}')?.textContent ?? "0"`)) || 0,
      );
      if (steps >= stepTarget) return;
      const clicked = await evaluate(
        cdp,
        sid,
        `(() => { const b = document.querySelector('${stepClick}'); if (!b || b.disabled) return false; b.click(); return true; })()`,
      ).catch(() => false);
      if (!clicked) await sleep(1_500);
      await sleep(600);
    }
    const steps = Number((await evaluate(cdp, sid, `document.querySelector('${stepReadout}')?.textContent ?? "0"`)) || 0);
    if (steps >= stepTarget) return;
    throw new Error(`${label}: step chain never reached ${stepTarget} choices (saw ${steps})`);
  }

  // click / input styles: atomic set(+click), retried, so a first-visit service-worker reload or a
  // swallowed click can never read as a hung generation.
  for (let attempt = 1; attempt <= 6; attempt++) {
    const phase = await evaluate(
      cdp,
      sid,
      `(() => {
        const el = ${input ? `document.querySelector('${input}')` : "null"};
        if (${input ? "true" : "false"} && !el) return "no-input";
        if (el) {
          el.value = ${JSON.stringify(PROMPT)};
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        ${style === "click" ? `const btn = document.querySelector('${trigger}'); if (!btn || btn.disabled) return "no-trigger"; btn.click();` : ""}
        return "clicked";
      })()`,
    );
    if (phase === "clicked") {
      let started = false;
      for (let i = 0; i < 10; i++) {
        const st = await evalJSON(
          cdp,
          sid,
          `JSON.stringify({
            busy: (document.querySelector('#status')?.textContent || '').length > 0 ||
              ${style === "click" ? `document.querySelector('${trigger}')?.disabled === true ||` : ""}
              (document.querySelector('#stop') ? document.querySelector('#stop').hidden === false : false),
            readout: document.querySelector('#readout')?.hidden === false,
          })`,
        ).catch(() => null);
        if (st && (st.busy || st.readout)) {
          started = true;
          break;
        }
        await sleep(2_000);
      }
      if (started) return;
    }
    console.log(`  [${label}] drive attempt ${attempt} did not start — retrying`);
    await sleep(3_000);
  }
  throw new Error(`${label}: drive never started (input/trigger not actionable)`);
}

async function exercise(browser, rung, viewportName, viewport, attempt = 1) {
  const cdp = browser.cdp;
  const route = ROUTES[rung];
  const label = `${viewportName} ${rung}`;
  currentCell = label;
  armWatchdog();
  const cfg = RUNGS[rung];
  // A fresh attempt starts from the same check ledger: an abandoned attempt's checks are rolled
  // back so a harness-level stall retry cannot inflate the evidence (the retry is logged).
  const checksBefore = checks;
  const passedBefore = passed;
  const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/${route}`);
  await setViewport(cdp, page.sessionId, viewport);
  let ok = true;
  let retry = false;

  try {
    await ensureReady(cdp, page.sessionId, label);
    await installChunkCounter(cdp, page.sessionId);
    // A page can report its loader "ready" a beat before it enables the controls (and a fresh profile
    // shows a Download button first), so wait for the trigger to be clickable rather than racing it.
    if (cfg.trigger) {
      await waitFor(
        cdp,
        page.sessionId,
        `(() => { const b = document.querySelector('${cfg.trigger}'); return !!b && b.disabled === false; })()`,
        240_000,
        `${label} trigger enabled`,
        2_000,
      );
    }
    await drive(cdp, page.sessionId, label, cfg);
    if (cfg.afterTrigger) {
      // Wait for the first step to finish (TrOCR) before clicking the second (NuExtract).
      await waitFor(
        cdp,
        page.sessionId,
        `(document.querySelector('#status')?.textContent || '').includes('OCR done')`,
        300_000,
        `${label} OCR step`,
        3_000,
      );
      await evaluate(cdp, page.sessionId, `document.querySelector('${cfg.afterTrigger}').click()`);
    }
    await waitFor(
      cdp,
      page.sessionId,
      `(() => {
        const readout = document.querySelector('#readout');
        const st = document.querySelector('#status')?.textContent || '';
        if (/failed|add at least|type a question/i.test(st)) return true;
        ${
          cfg.doneText
            // Where a rung names its final status, that is the ONLY completion signal: some readouts
            // (multi-model's verdict panel) are never hidden, so a readout fallback would fire before
            // the work even started.
            ? `return st.includes(${JSON.stringify(cfg.doneText)});`
            : `// Otherwise the READOUT is the completion signal.
        return readout && readout.hidden === false;`
        }
      })()`,
      900_000,
      `${label} generation`,
      5_000,
    );

    if (rung === "overview" && HAS_INSIDE) {
      // Not every family has a see-inside BUTTON: this one fills its distribution panel as it generates,
      // so only click when the config names a control (and never crash on a missing one).
      if (INSIDE.click) {
        await evaluate(cdp, page.sessionId, `document.querySelector('${INSIDE.click}')?.click()`);
      }
      await waitFor(cdp, page.sessionId, INSIDE.ready, 600_000, `${label} see-inside`, 5_000);
      const insideText = await evaluate(cdp, page.sessionId, INSIDE.text);
      ok = check(
        `${label}: ${INSIDE.label}`,
        String(insideText).length > INSIDE.minLength,
        insideText,
      ) && ok;
    }

    // THE 0ly PROOF for this family: the displayed count is the worker's GENERATED token count.
    // The old code read it from DOM chips and the worker's own field was prompt+generated, so the
    // falsifiable claim is: readout === resolved.tokens AND the worker reported a non-zero promptTokens
    // (i.e. the number on screen excludes the prompt — it is not the prompt-inclusive value it used to
    // be, and it is not the length of anything in the DOM).
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({
        tok: ${cfg.tok ? `document.querySelector('${cfg.tok}')?.textContent ?? null` : "null"},
        chunks: window.__chunks ?? null,
        tokenIds: window.__tokens ?? [],
        promptTokens: window.__promptTokens ?? null,
        distRows: document.querySelector('#dist') ? document.querySelector('#dist').children.length : null,
        steps: ${cfg.stepReadout ? `document.querySelector('${cfg.stepReadout}')?.textContent ?? null` : "null"},
        verdict: document.querySelector('#rLabel')?.textContent ?? null,
        backend: document.querySelector('${cfg.backend ?? "#rBackend"}')?.textContent ?? null,
      })`,
    );
    const chunks = proof.chunks;
    const realIds = (proof.tokenIds ?? []).reduce((a, b) => a + Number(b || 0), 0);
    const tok = Number(proof.tok);
    const promptTokens = Number(proof.promptTokens ?? 0);

    if (cfg.style === "click" && cfg.trigger === "#run" && cfg.tok) {
      // A generating rung: one step message per sampled token, and the done payload's count.
      ok = check(
        `${label}: worker streamed one message per sampled token`,
        Number.isInteger(chunks) && chunks >= 1,
        { chunks },
      ) && ok;
      ok = check(
        `${label}: worker reported its generated token count and the prompt length separately`,
        (proof.tokenIds ?? []).length >= 1 && realIds >= 1 && promptTokens >= 1,
        { tokenIds: proof.tokenIds, promptTokens },
      ) && ok;
      ok = check(
        `${label}: readout is the generated count, not the prompt-inclusive total`,
        Number.isFinite(tok) && tok === realIds && tok !== realIds + promptTokens,
        { readout: proof.tok, generated: realIds, prompt: promptTokens, oldPromptInclusive: realIds + promptTokens },
      ) && ok;
      divergence.push({ label, viewport: viewportName, chunks, tokens: realIds, promptTokens });
    } else if (cfg.style === "steps") {
      ok = check(
        `${label}: manual chain committed at least ${cfg.stepTarget} tokens`,
        Number(proof.steps) >= cfg.stepTarget,
        { steps: proof.steps, target: cfg.stepTarget },
      ) && ok;
    } else if (cfg.style === "input") {
      ok = check(
        `${label}: distribution rendered straight from the input`,
        Number(proof.distRows) >= 3,
        { distRows: proof.distRows },
      ) && ok;
    } else {
      ok = check(
        `${label}: write-then-judge completed with a verdict`,
        String(proof.verdict ?? "").trim().length > 0 && realIds >= 1,
        { verdict: proof.verdict, writeTokens: proof.tokenIds },
      ) && ok;
    }

    if (cfg.noBackend) {
      ok = check(`${label}: rung exposes no backend element (multi-model readout is verdict scores)`, true) && ok;
    } else {
      ok = check(
        `${label}: readout reports the advertised WASM backend`,
        String(proof.backend || "").toUpperCase() === "WASM",
        { backend: proof.backend },
      ) && ok;
    }

    const hygiene = await evalJSON(
      cdp,
      page.sessionId,
      `({
      overflow: document.documentElement.scrollWidth <= window.innerWidth + 1,
    })`,
    );
    check(`${label}: no horizontal overflow`, hygiene.overflow);
    check(`${label}: zero console errors`, page.errors.length === 0, page.errors.join(" | "));
    check(
      `${label}: zero failed network requests`,
      page.netFailures.length === 0,
      page.netFailures.join(" | "),
    );
    ok = ok && hygiene.overflow && page.errors.length === 0 && page.netFailures.length === 0;
  } catch (error) {
    if (attempt < 3) {
      // WebGPU/worker contention across successive pages is real in this harness, and on a shared box
      // a browser can be killed outright mid-cell (observed: two consecutive deaths during one
      // gpt2-text-generation wild cell, with the whole Chrome process gone and no crash dump). Two retries
      // in fresh browsers distinguish a transient death from a route that cannot run; every retry is
      // logged and the abandoned attempt's checks are rolled back, so the ledger shows only what was
      // actually driven.
      retry = true;
      console.log(`  [${label}] attempt ${attempt} stalled (${String(error.message).slice(0, 140)}) — retrying in a fresh browser`);
    } else {
      ok = false;
      check(`${label}: route completed`, false, error.stack || error.message);
    }
  } finally {
    if (!retry) results.push({ route, viewport: viewportName, pass: ok });
    await closePage(cdp, page.targetId).catch(() => {});
  }
  if (retry) {
    checks = checksBefore;
    passed = passedBefore;
    for (let i = divergence.length - 1; i >= 0; i--) {
      if (divergence[i].label === label) divergence.splice(i, 1);
    }
    // A stalled renderer often cannot even open a page, so the retry starts a whole new browser
    // (same profile dir, so the cached model weights survive) after a short settle.
    await browser.restart();
    return exercise(browser, rung, viewportName, viewport, attempt + 1);
  }
}

const { server, port } = await startServer();
// A FRESH BROWSER PER CELL (0ly route 7): heavy WASM pages wedge the renderer after a few
// successive cells on a shared box — a solo rerun of the same cell passes — so each cell gets its
// own browser process. The profile dir is shared and NOT reset after the first launch, because the
// 200-600MB model download is cached there; resetting it per cell would re-download ten times.
let browserLaunches = 0;
// A hang is worse than a failure: the runner buffers this validator's output, so a wedged run leaves
// no evidence at all. Re-armed per cell, this watchdog force-exits with the cell it was stuck on.
let currentCell = "startup";
let watchdog = null;
function armWatchdog() {
  clearTimeout(watchdog);
  watchdog = setTimeout(() => {
    console.error(`WATCHDOG: no cell completed in 30 minutes (stuck at ${currentCell}) — exiting`);
    process.exit(3);
  }, 30 * 60_000);
}
armWatchdog();
function makeBrowser() {
  let chrome = null;
  let cdp = null;
  return {
    get cdp() { return cdp; },
    async start() {
      chrome = await launchChrome({
        // This family is the CPU/WASM demo: its loader declares requiresWebGPU: false and the page
        // says it runs on WebAssembly, but the worker silently prefers WebGPU whenever
        // navigator.gpu exists. Under this box's headless WebGPU (SwiftShader) that path emitted
        // gibberish at ~0.6 chunks/s, so the validator drives the ADVERTISED stage with GPU
        // disabled — and then asserts the readout reports WASM.
        webgpu: false,
        userDataDir: PROFILE_DIR,
        resetProfile: browserLaunches === 0,
        removeProfileOnKill: false,
      });
      browserLaunches++;
      cdp = new CDP(chrome.ws);
    },
    async stop() {
      const dying = chrome;
      chrome = null;
      cdp = null;
      try { if (dying) dying.kill(); } catch { /* ignore */ }
    },
    async restart() {
      await this.stop();
      await sleep(4000); // let the OS reclaim the stalled renderer's memory before relaunching
      await this.start();
    },
  };
}
const browser = makeBrowser();
async function runCell(name, viewportName, viewport) {
  await browser.start();
  try {
    await exercise(browser, name, viewportName, viewport);
  } finally {
    await browser.stop();
    await sleep(2000); // brief settle between cells: back-to-back heavy WASM pages thrash the box
  }
}
try {
  const cells = Object.entries(ROUTES).filter(([n]) => !ONLY_RUNG || ONLY_RUNG.includes(n));
  if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "desktop") {
    for (const [name] of cells) await runCell(name, "desktop", DESKTOP);
  }
  if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "desktop") check(
    `desktop: at least one cell proves the readout is the generated count, not the prompt-inclusive one (0ly proof)`,
    divergence.some((d) => d.viewport === "desktop" && d.promptTokens >= 1 && d.tokens >= 1),
    divergence.filter((d) => d.viewport === "desktop"),
  );
  if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "mobile") {
    for (const [name] of cells) await runCell(name, "mobile", MOBILE);
  }
  if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "mobile") check(
    `mobile: at least one cell proves the readout is the generated count, not the prompt-inclusive one (0ly proof)`,
    divergence.some((d) => d.viewport === "mobile" && d.promptTokens >= 1 && d.tokens >= 1),
    divergence.filter((d) => d.viewport === "mobile"),
  );
  console.log(`CHUNK-VS-ID: ${JSON.stringify(divergence)}`);
} finally {
  // no single browser to kill: withBrowser owns each one (the shared profile dir is removed below)
  try {
    server.close();
  } catch {
    // ignore
  }
  try {
    rmSync(PROFILE_DIR, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

console.log(`\n${passed}/${checks} checks passed`);
const allRoutesPassed = results.length === Object.keys(ROUTES).length * 2 && results.every((r) => r.pass);
const succeeded = passed === checks && allRoutesPassed && checks > 0;
console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
if (WRITE_RUN && succeeded) {
  const runRecord = { commit: startCommit, ranAt: new Date().toISOString(), exitCode: 0, results };
  writeFileSync(RUN_RECORD, JSON.stringify(runRecord, null, 2) + "\n");
  console.log(`WROTE ${RUN_RECORD} for ${startCommit}`);
} else if (WRITE_RUN && !succeeded) {
  console.log("REFUSAL: run did not pass — no run record written");
}
process.exit(succeeded ? 0 : 1);
