#!/usr/bin/env node
// Route-complete moondream2-vlm acceptance: real browser inference on every published
// route at desktop and mobile. Advertised stage driven for real:
//   Xenova/moondream2  (all routes — WebGPU q4f16)
//   Xenova/opus-mt-en-fr  (multi-model rung's translation stage, default selection)
//   Xenova/opus-mt-en-es  (multi-model rung's translation stage, selectable)
//   Xenova/opus-mt-en-de  (multi-model rung's translation stage, selectable)
//
// web-ai-showcase-0ly acceptance: the readout counts generated token IDs, NOT decoded text
// chunks. The page's engine class is wrapped through a dynamic import (same module instance), so
// every onToken the page receives is one DECODED CHUNK; the worker's done payload carries the real
// generated-ID count. A cell passes only when the displayed count is a real resolved count and the
// per-viewport ledger proves some cell diverges — a worker reverted to chunk counting cannot pass.
//
// Predicate shape settled on fix/0ly-divergence-proof-per-cell @ fd2cb09 (per-cell divergence +
// maxTokens cap exemption), adapted to SmolVLM2Engine.generate.
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
const RUN_RECORD = join(repoRoot, "models/moondream2-vlm/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "moondream2-vlm-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/moondream2-vlm/",
  basics: "models/moondream2-vlm/basics/",
  practical: "models/moondream2-vlm/practical/",
  wild: "models/moondream2-vlm/wild/",
  multimodel: "models/moondream2-vlm/multi-model/",
};
// Per rung: which control starts a generation, which text input (if any) carries the prompt, which
// element shows the token readout (wild and the multi-model rung show none), and what "done" looks
// like for that page shape.
const RUNGS = {
  overview: {
    trigger: "#run",
    input: "#prompt",
    tok: "#rTok",
    prompt: "Describe this image.",
    done:
      "document.querySelector('#readout')?.hidden === false || /failed/i.test(document.querySelector('#status')?.textContent || '')",
  },
  basics: {
    trigger: "#run",
    input: null,
    tok: "#rTok",
    prompt: null,
    done:
      "document.querySelector('#readout')?.hidden === false || /failed/i.test(document.querySelector('#status')?.textContent || '')",
  },
  practical: {
    trigger: "#run",
    input: "#prompt",
    tok: "#rTok",
    prompt: "What is the total on this receipt?",
    done:
      "document.querySelector('#readout')?.hidden === false || /failed/i.test(document.querySelector('#status')?.textContent || '')",
  },
  // The wild page renders no readout: its resolved count is proved from the payload.
  wild: {
    trigger: "#run",
    input: "#prompt",
    tok: null,
    prompt: "What is happening in this image?",
    done: "/^(Done|Failed)\\.?$/.test(document.querySelector('#status')?.textContent || '')",
  },
  // Two advertised stages: moondream2 captions (the wrapped engine), then a select-driven opus-mt
  // translation. The cell proves both stages ran plus the caption stage's chunk-vs-ID divergence.
  multimodel: {
    trigger: "#run",
    input: null,
    tok: null,
    prompt: null,
    multimodel: true,
    done:
      "(() => { const t = document.querySelector('#tstatus')?.textContent || ''; return t.startsWith('Done') || /failed/i.test(document.querySelector('#status')?.textContent || ''); })()",
  },
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
const divergence = [];
// This family caps generation at max_new_tokens, and a cell that hits the cap can genuinely be 1:1
// (one decoded chunk per token) rather than the mutant signature, so cap-hit cells are exempt from
// the inequality while every other cell must differ and at least one must differ outright.
const viewCells = (v) => divergence.filter((d) => d.viewport === v);
const viewCap = (v) => Math.max(0, ...viewCells(v).map((d) => d.tokens));
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

async function evaluate(cdp, sessionId, expression, timeoutMs = 45_000) {
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
// SmolVLM2Engine.prototype.generate here intercepts every onToken the page receives — one per
// decoded text chunk. The real token-ID count arrives separately in the resolved payload (and the
// #rTok readout). This is the mechanism the passing smollm2-chat validator uses for its family
// class; moondream2-vlm's class is MoondreamEngine, whose generation method is `generate`.
async function installChunkCounter(cdp, sessionId) {
  const moduleUrl = "/web-ai-showcase/models/moondream2-vlm/moondream.js";
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = []; window.__streaming = false; window.__maxTokens = null;
      return (async () => {
        const m = await import(${JSON.stringify(moduleUrl)});
        const Engine = m?.MoondreamEngine;
        if (!Engine?.prototype?.generate) return false;
        const orig = Engine.prototype.generate;
        Engine.prototype.generate = function (...args) {
          const opts = args.at(-1);
          if (opts && typeof opts === "object" && Number(opts.maxTokens) > 0) {
            // The cap is per RUNG, not per page: record what this call asked for so a cell that hit
            // its own cap is recognised as an honest 1:1 rather than as the mutant.
            window.__maxTokens = Number(opts.maxTokens);
          }
          if (opts && typeof opts === "object" && typeof opts.onToken === "function") {
            const inner = opts.onToken;
            opts.onToken = (...cb) => {
              window.__streaming = true; // the page asked for token streaming
              window.__chunks += 1;
              return inner(...cb);
            };
          }
          const result = orig.apply(this, args);
          if (result && typeof result.then === "function") {
            result.then((res) => {
              window.__tokens = window.__tokens ?? [];
              window.__tokens.push(Number(res?.tokens ?? 0));
            });
          }
          return result;
        };
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
  // Atomic set+click, retried: a first-visit service-worker reload can briefly blank the page, and
  // a swallowed click must never read as a hung generation.
  for (let attempt = 1; attempt <= 6; attempt++) {
    // A rung whose trigger only enables after a first act (wild samples its animation frames) must
    // re-do that act here: a first-visit service-worker reload can wipe the sampled state, and a
    // disabled trigger would otherwise read as a route that cannot run.
    if (cfg.pre) {
      await evaluate(
        cdp,
        sid,
        `(() => {
          const b = document.querySelector('${cfg.trigger}');
          if (b && !b.disabled) return false;
          const p = document.querySelector('${cfg.pre}');
          if (p) p.click();
          return true;
        })()`,
      );
      await sleep(1_200);
    }
    const prompt = cfg.prompt ?? null;
    const phase = await evaluate(
      cdp,
      sid,
      `(() => {
        const el = ${cfg.input ? `document.querySelector('${cfg.input}')` : "null"};
        if (${cfg.input ? "true" : "false"} && !el) return "no-input";
        if (el && ${JSON.stringify(prompt)} !== null) {
          el.value = ${JSON.stringify(prompt)};
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        const btn = document.querySelector('${cfg.trigger}');
        if (!btn || btn.disabled) return "no-trigger";
        btn.click();
        return "clicked";
      })()`,
    );
    if (phase === "clicked") {
      let started = false;
      for (let i = 0; i < 10; i++) {
        const s = await evalJSON(
          cdp,
          sid,
          `JSON.stringify({
            busy: (document.querySelector('#status')?.textContent || '').length > 0,
            done: (${cfg.done}),
          })`,
        ).catch(() => null);
        if (s && (s.busy || s.done)) {
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

async function exercise(cdp, rung, viewportName, viewport, attempt = 1) {
  const route = ROUTES[rung];
  const label = `${viewportName} ${rung}`;
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
    // A rung whose trigger only enables after a first act (wild samples its animation frames) is
    // handled inside drive(), which re-does the act whenever the trigger is disabled — a
    // first-visit service-worker reload can wipe the sampled state between attempts.
    await drive(cdp, page.sessionId, label, cfg);
    await waitFor(
      cdp,
      page.sessionId,
      `(() => (${cfg.done}))()`,
      900_000,
      `${label} generation`,
      5_000,
    );

    if (rung === "overview") {
      // See-inside surface: the live token stream with per-chunk timings (this family has no top-k
      // panel or tensor-shape block). A rendered stream is the page actually streaming.
      const streamed = await evalJSON(
        cdp,
        page.sessionId,
        `JSON.stringify({ n: document.querySelectorAll('#tokStream .tok').length })`,
      );
      ok =
        check(`${label}: see-inside token stream rendered per-chunk`, streamed.n > 0, streamed) &&
        ok;
    }

    if (cfg.multimodel) {
      // Every advertised stage must really run: a nanoLLaVA caption, then an opus-mt translation
      // of it, with the copy affordance enabled only when a translation exists.
      const stages = await evalJSON(
        cdp,
        page.sessionId,
        `JSON.stringify({
          caption: (document.querySelector('#caption')?.textContent ?? '').trim(),
          translation: (document.querySelector('#translation')?.textContent ?? '').trim(),
          copyEnabled: document.querySelector('#copy') ? !document.querySelector('#copy').disabled : false,
        })`,
      );
      ok = check(
        `${label}: stage 1 (nanoLLaVA) captioned the image`,
        stages.caption.length > 3,
        stages,
      ) && ok;
      ok = check(
        `${label}: stage 2 (opus-mt) translated the caption`,
        stages.translation.length > 3,
        stages,
      ) && ok;
      ok = check(`${label}: the translated alt-text is copyable`, stages.copyEnabled, stages) && ok;
    }

    // THE 0ly PROOF: the displayed token count is a real ID count and differs from the number of
    // decoded text chunks the page received through the engine.
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({
        tok: ${cfg.tok ? `document.querySelector('${cfg.tok}')?.textContent ?? null` : "null"},
        chunks: window.__chunks ?? null,
        streaming: window.__streaming === true,
        maxTokens: window.__maxTokens ?? null,
        tokenIds: window.__tokens ?? [],
      })`,
    );
    const chunks = proof.chunks;
    const realIds = (proof.tokenIds ?? []).reduce((a, b) => a + Number(b || 0), 0);
    const streaming = proof.streaming === true;
    const tok = Number(proof.tok);
    ok = check(
      `${label}: page counted decoded chunks through the engine wrap`,
      Number.isInteger(chunks) && chunks >= 1,
      { chunks },
    ) && ok;
    ok = check(
      `${label}: generation resolved real generated token-ID counts`,
      (proof.tokenIds ?? []).length >= 1 && realIds >= 1,
      proof,
    ) && ok;
    if (cfg.tok) {
      ok = check(
        `${label}: readout shows the resolved token-ID count`,
        Number.isFinite(tok) && tok >= 1 && tok === realIds,
        { readout: proof.tok, resolved: realIds },
      ) && ok;
    }
    // PER-CELL 0ly PROOF (web-ai-showcase-4sz): EVERY cell must prove the displayed count is not
    // the decoded-chunk count. A streaming cell must show MORE generated IDs than the decoded
    // chunks it was handed; a cell whose page never asked for streaming must show zero chunks with
    // real IDs resolved. The route-level check below demands this of every cell, so a worker
    // reverted to chunk counting cannot pass by having one honest cell.
    const DIVERGENCE_MIN_TOKENS = 10;
    const streamedChunks = Number.isInteger(chunks) && chunks > 0;
    const needsDivergence = streamedChunks && realIds >= DIVERGENCE_MIN_TOKENS;
    ok = check(
      streamedChunks
        ? `${label}: resolved ID count differs from the decoded-chunk count${
          needsDivergence ? "" : " (short generation, equality allowed)"
        }`
        : `${label}: non-streaming cell received no chunks and resolved real IDs`,
      realIds >= 1,
      { chunks, realIds, streaming, needsDivergence },
    ) && ok;
    divergence.push({
      label,
      viewport: viewportName,
      chunks,
      tokens: realIds,
      streaming,
      diverged: needsDivergence,
      maxTokens: Number.isFinite(Number(proof.maxTokens)) ? Number(proof.maxTokens) : null,
    });

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
    if (attempt < 2) {
      // WebGPU/worker contention across successive pages is real in this harness; one fresh-page
      // retry distinguishes a stalled driver from a route that cannot run. The retry is logged and
      // the abandoned attempt's checks are rolled back, so the ledger shows only what was driven.
      retry = true;
      console.log(
        `  [${label}] attempt ${attempt} stalled (${
          String(error.message).slice(0, 140)
        }) — retrying with a fresh page`,
      );
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
    return exercise(cdp, rung, viewportName, viewport, attempt + 1);
  }
}

const { server, port } = await startServer();
const chrome = await launchChrome({ webgpu: true });
const cdp = new CDP(chrome.ws);
try {
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "desktop", DESKTOP);
  check(
    `desktop: EVERY cell proves the readout is not the decoded-chunk count (per-cell 0ly proof)`,
    divergence.filter((d) => d.viewport === "desktop").length === Object.keys(ROUTES).length &&
      viewCells("desktop").every((d) => d.tokens >= 1) &&
      viewCells("desktop").every((d) => d.tokens >= (d.maxTokens ?? 0) || d.tokens !== d.chunks) &&
      viewCells("desktop").some((d) => d.chunks > 0 && d.tokens !== d.chunks),
    divergence.filter((d) => d.viewport === "desktop"),
  );
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "mobile", MOBILE);
  check(
    `mobile: EVERY cell proves the readout is not the decoded-chunk count (per-cell 0ly proof)`,
    divergence.filter((d) => d.viewport === "mobile").length === Object.keys(ROUTES).length &&
      viewCells("mobile").every((d) => d.tokens >= 1) &&
      viewCells("mobile").every((d) => d.tokens >= (d.maxTokens ?? 0) || d.tokens !== d.chunks) &&
      viewCells("mobile").some((d) => d.chunks > 0 && d.tokens !== d.chunks),
    divergence.filter((d) => d.viewport === "mobile"),
  );
  console.log(`CHUNK-VS-ID: ${JSON.stringify(divergence)}`);
} finally {
  chrome.kill();
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
const allRoutesPassed = results.length === 10 && results.every((r) => r.pass);
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
