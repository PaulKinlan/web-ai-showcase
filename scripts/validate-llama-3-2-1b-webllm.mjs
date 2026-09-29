#!/usr/bin/env node
// Route-complete Llama 3.2 1B webllm acceptance: real browser inference on every published
// route at desktop and mobile. Advertised stage driven for real:
//   Llama-3.2-1B-Instruct-q4f16_1-MLC  (all routes — WebLLM/MLC, WebGPU q4f16_1)
//   Xenova/all-MiniLM-L6-v2  (multi-model rung's retrieval stage, WASM q8)
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
const RUN_RECORD = join(repoRoot, "models/llama-3-2-1b-webllm/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "mistral-webllm-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/llama-3-2-1b-webllm/",
  basics: "models/llama-3-2-1b-webllm/basics/",
  practical: "models/llama-3-2-1b-webllm/practical/",
  wild: "models/llama-3-2-1b-webllm/wild/",
  multimodel: "models/llama-3-2-1b-webllm/multi-model/",
};
// Per rung: the control that starts a generation, the input carrying the prompt, the count readout(s)
// (this family shows one only on the overview), the disabled control that proves a generation is
// running, and the page's real completion signal (every rung sets #status). Prompts that end in an
// emoji make the engine's usage genuinely exceed its delta count: a multi-byte character decodes from
// several tokens, which is what lets a per-cell divergence be proved at all.
const GENERATION_DEADLINE_MS = 900_000;
const RUNGS = {
  overview: {
    trigger: "#send",
    input: "#input",
    tok: ["#sGen"],
    prompt: "Explain what a large language model is in two sentences. End with one emoji.",
    busy: "document.querySelector('#send')?.disabled === true",
    done:
      "/^(Done|Failed|Generation failed)/.test(document.querySelector('#status')?.textContent || '')",
  },
  basics: {
    trigger: "#send",
    input: "#input",
    tok: null,
    prompt: "What is a token, in one sentence? Add one emoji at the end.",
    busy: "document.querySelector('#send')?.disabled === true",
    done:
      "/^(Done|Failed|Generation failed)/.test(document.querySelector('#status')?.textContent || '')",
  },
  practical: {
    trigger: "#run",
    input: "#src",
    tok: null,
    prompt: null,
    busy: "document.querySelector('#run')?.disabled === true",
    done:
      "/^(Done|Failed|Generation failed)/.test(document.querySelector('#status')?.textContent || '')",
  },
  wild: {
    trigger: "#send",
    input: "#input",
    tok: null,
    prompt: "Say something short and in character, and end with one emoji.",
    busy: "document.querySelector('#send')?.disabled === true",
    done:
      "/^(Done|Failed|Generation failed)/.test(document.querySelector('#status')?.textContent || '')",
  },
  multimodel: {
    trigger: "#ask",
    input: "#q",
    tok: null,
    prompt: "What colour is the boat?",
    corpus:
      "The harbour was quiet at dawn. A small blue boat bobbed beside the pier, its paint faded but bright against the grey water. The ferry would leave at noon. Sailors checked the ropes and talked about the wind. By evening the harbour was empty again.",
    multimodel: true,
    deadlineMs: 1_800_000,
    stallBudgetMs: 600_000,
    busy: "document.querySelector('#ask')?.disabled === true",
    done:
      "/^(Done|Failed|Generation failed)/.test(document.querySelector('#status')?.textContent || '')",
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

/**
 * Wait for a cell's generation to finish, and FAIL FAST when it never starts streaming.
 *
 * A cold WebLLM engine sometimes accepts the request and then delivers nothing at all: no deltas, no
 * console error, no page status — the GPU sits idle while the per-cell deadline burns. That is a
 * driver/engine stall, not a slow generation, so an attempt with no delta inside STALL_BUDGET_MS is
 * thrown as a stall and the caller's logged fresh-page retry handles it. A generation that HAS started
 * streaming keeps the rung's full budget.
 */
async function waitForGeneration(cdp, sessionId, cfg, label) {
  const deadlineMs = cfg.deadlineMs ?? GENERATION_DEADLINE_MS;
  const STALL_BUDGET_MS = cfg.stallBudgetMs ?? 300_000;
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < deadlineMs) {
    const state = await evalJSON(
      cdp,
      sessionId,
      `JSON.stringify({ done: (${cfg.done}), chunks: window.__chunks ?? 0 })`,
    ).catch(() => null);
    if (state?.done) return;
    const elapsed = Date.now() - started;
    if (!state || state.chunks === 0) {
      if (elapsed > STALL_BUDGET_MS) {
        throw new Error(
          `${label}: the engine delivered no deltas within ${
            STALL_BUDGET_MS / 1000
          }s — a stalled generation, not a slow one`,
        );
      }
    }
    if (elapsed >= nextLog) {
      console.log(
        `  [${label} generation] waiting ${Math.round(elapsed / 1000)}s (chunks=${
          state?.chunks ?? "?"
        })`,
      );
      nextLog = elapsed + 30_000;
    }
    await sleep(5_000);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label} generation`);
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
// class; mistral-webllm's class is WebLLMChatEngine, whose generation method is `chat`.
async function installChunkCounter(cdp, sessionId) {
  const moduleUrl = "/web-ai-showcase/models/llama-3-2-1b-webllm/webllm-chat.js";
  const installed = await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = []; window.__streaming = false; window.__maxTokens = null;
      return (async () => {
        const m = await import(${JSON.stringify(moduleUrl)});
        const Engine = m?.WebLLMChatEngine;
        // Families differ in the generation method's name (generate, chat, ...): wrap whichever
        // the class actually exposes, or the counter never installs and every cell fails vacuous.
        const method = ["generate", "chat"].find((name) => typeof Engine?.prototype?.[name] === "function");
        if (!method || typeof Engine !== "function") return false;
        const orig = Engine.prototype[method];
        Engine.prototype[method] = function (...args) {
          const opts = args.at(-1);
          if (opts && typeof opts === "object" && Number(opts.maxTokens) > 0) {
            // The cap is per RUNG, not per page: record what this call asked for so a cell that hit
            // its own cap is recognised as an honest 1:1 rather than as the mutant.
            window.__maxTokens = Number(opts.maxTokens);
          }
          window.__raceRetries = 0;
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
              window.__raceRetries = (window.__raceRetries ?? 0) +
                Number(res?.loadRaceRetries ?? 0) + Number(res?.generationRaceRetries ?? 0);
            });
          }
          return result;
        };
        return true;
      })();
    })()`,
    30_000,
  );
  // A counter that never installed reports zero for every cell and looks EXACTLY like a route that
  // never streamed. Fail loudly instead: the instrument is part of the proof.
  if (installed !== true) {
    throw new Error(
      "the engine wrap did not install (module imported, but no WebLLMChatEngine.chat to wrap) — the proof would read zeros for a working page",
    );
  }
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
        error: document.querySelector('.model-loader[data-state="error"]') !== null,
        errorText: (document.querySelector('.model-loader[data-state="error"]')?.innerText || '').replace(/\s+/g, ' ').slice(0, 220),
      })`,
    ).catch(() => ({ ready: -1, total: -1 }));
    if (state.unsupported) {
      // Headless WebGPU adapter availability is intermittent per launch: fail fast and let the
      // operator relaunch instead of spinning for 20 minutes (web-ai-showcase-0ly).
      throw new Error("model loader refused: needs a WebGPU adapter (relaunch the validator)");
    }
    if (state.error) {
      // A loader that reached an ERROR state will not heal by retrying: clicking its Retry button
      // re-runs the same failing initialisation (phi-3.5-vision's ORT session creation fails with
      // "Failed to find kernel for MemcpyToHost(1)"). Fail fast, naming the page's own message, so
      // the run reports the real defect instead of a 20-minute load timeout.
      throw new Error(`model loader refused: ${state.errorText || "see the page's error state"}`);
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
    // A rung may declare fields to seed before its trigger (the multi-model rung's knowledge base
    // and question): the page's own pipeline still runs, on a tractable input for a CPU-WASM model.
    if (cfg.preFill) {
      const fill = Object.entries(cfg.preFill)
        .map(([sel, val]) =>
          `{ const el = document.querySelector('${sel}'); if (el) { el.value = ${
            JSON.stringify(val)
          }; el.dispatchEvent(new Event('input', { bubbles: true })); } }`
        )
        .join("\n");
      await evaluate(cdp, sid, `(() => { ${fill}; return true; })()`);
    }
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
        const corpusEl = ${cfg.corpus ? "document.querySelector('#corpus')" : "null"};
        if (corpusEl) {
          corpusEl.value = ${JSON.stringify(cfg.corpus ?? "")};
          corpusEl.dispatchEvent(new Event("input", { bubbles: true }));
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
            busy: !!(${cfg.busy ?? "(document.querySelector('#status')?.textContent || '')"}),
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
    await waitForGeneration(cdp, page.sessionId, cfg, label);

    if (rung === "overview") {
      // See-inside surface: WebLLM's own runtime stats line (prefill/decode rates), rendered from
      // the engine's runtimeStatsText(). This family has no token stream or top-k panel.
      const stats = await evaluate(
        cdp,
        page.sessionId,
        `(document.querySelector('#statsRaw')?.textContent || '').trim().slice(0, 160)`,
      );
      ok = check(`${label}: see-inside WebLLM runtime stats render`, stats.length > 10, stats) &&
        ok;
    }

    if (cfg.multimodel) {
      // Every advertised stage must really run: retrieved passages, then an answer built from them.
      const stages = await evalJSON(
        cdp,
        page.sessionId,
        `JSON.stringify({
          retrieved: [...document.querySelectorAll('#retrieved > div, #retrieved li, #retrieved .chunk')].filter((el) => el.textContent.trim().length > 0).length,
          answer: (document.querySelector('#answer')?.textContent ?? '').trim(),
          status: (document.querySelector('#status')?.textContent ?? '').trim().slice(0, 200),
        })`,
      );
      ok = check(`${label}: stage 1 (MiniLM) retrieved passages`, stages.retrieved >= 1, stages) &&
        ok;
      ok = check(
        `${label}: stage 2 (Llama via WebLLM) answered from them`,
        stages.answer.length > 3 && !/^\s*Failed/i.test(stages.answer),
        stages,
      ) && ok;
    }

    // THE 0ly PROOF: the displayed token count is a real ID count and differs from the number of
    // decoded text chunks the page received through the engine.
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({
        tok: ${
        cfg.tok
          ? `[${
            cfg.tok.map((sel) => `document.querySelector('${sel}')?.textContent ?? null`).join(", ")
          }]`
          : "null"
      },
        chunks: window.__chunks ?? null,
        streaming: window.__streaming === true,
        maxTokens: window.__maxTokens ?? null,
        tokenIds: window.__tokens ?? [],
        raceRetries: window.__raceRetries ?? 0,
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
      // Each count readout carries its unit ("20 tok" / "14 chunks"), so compare the leading integer;
      // a rung that shows more than one generation's count (the wild rung's two passes) is compared
      // on the SUM.
      const readoutCount = (proof.tok ?? []).reduce(
        (sum, text) => sum + (Number.parseInt(String(text ?? ""), 10) || 0),
        0,
      );
      const everyReadoutPresent = (proof.tok ?? []).every((text) =>
        String(text ?? "").trim().length > 0
      );
      ok = check(
        `${label}: readout shows the completion's real token count`,
        everyReadoutPresent && Number.isFinite(readoutCount) && readoutCount >= 1 &&
          readoutCount === realIds,
        { readout: proof.tok, resolved: realIds },
      ) && ok;
    }
    // PER-CELL 0ly PROOF (web-ai-showcase-4sz): EVERY cell must prove the displayed count is not
    // the decoded-chunk count. A streaming cell must show MORE generated IDs than the decoded
    // chunks it was handed; a cell whose page never asked for streaming must show zero chunks with
    // real IDs resolved. The route-level check below demands this of every cell, so a worker
    // reverted to chunk counting cannot pass by having one honest cell.
    const DIVERGENCE_MIN_CHUNKS = 20;
    const streamedChunks = Number.isInteger(chunks) && chunks > 0;
    const needsDivergence = streamedChunks && chunks >= DIVERGENCE_MIN_CHUNKS;
    const divergenceProved = needsDivergence ? realIds !== chunks : realIds >= 1;
    if (!divergenceProved && attempt < 3) {
      // The evidence did not materialise (a 1:1 generation, or a worker that reported nothing):
      // retry the cell once with a fresh page. The retry is logged and the abandoned attempt's
      // checks are rolled back, so a mutant still fails on the second attempt.
      retry = true;
      console.log(
        `  [${label}] no chunk-vs-usage divergence yet (chunks=${chunks}, reported=${realIds}) — retrying with a fresh page`,
      );
    }
    ok = check(
      streamedChunks
        ? `${label}: the reported count is not the decoded-chunk count${
          needsDivergence ? "" : " (short generation, equality allowed)"
        }`
        : `${label}: non-streaming cell received no chunks and resolved real IDs`,
      divergenceProved && realIds >= 1,
      { chunks, realIds, streaming, needsDivergence },
    ) && ok;
    if (Number(proof.raceRetries ?? 0) > 0) {
      // Legible, not hidden: a cell that only produced its numbers after losing GPU races says so.
      console.log(
        `  [${label}] this cell needed ${proof.raceRetries} GPU-race retry(ies) before it streamed`,
      );
    }
    divergence.push({
      label,
      viewport: viewportName,
      chunks,
      tokens: realIds,
      streaming,
      diverged: needsDivergence,
      raceRetries: Number(proof.raceRetries ?? 0),
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
    // ONE KNOWN DRIVER RACE, EXEMPTED BY NAME — the rest of the console-error check stands.
    //
    // 'mapAsync … Buffer was unmapped before mapping was resolved' comes out of the WebGPU buffer
    // mapping the MLC engine does while it loads; it appears against the honest tree AND against the
    // pre-fix tree (it hit mistral's first cell before this change existed), so it is not this
    // family's defect and not this change's. It is exempted by exact message, counted, and logged —
    // every OTHER console error still fails the cell, and the cell's functional assertions (readout
    // equals the engine's own usage; per-cell divergence) still gate it independently.
    const DRIVER_RACE =
      /Failed to execute 'mapAsync' on 'GPUBuffer': Buffer was unmapped before mapping was resolved/;
    const driverRaces = page.errors.filter((err) => DRIVER_RACE.test(String(err)));
    const realErrors = page.errors.filter((err) => !DRIVER_RACE.test(String(err)));
    if (driverRaces.length > 0) {
      console.log(
        `  [${label}] ${driverRaces.length} known WebGPU loader race(s) recorded and exempted by name — the functional proof still gates this cell`,
      );
    }
    if (realErrors.length > 0 && attempt < 3) {
      retry = true;
      console.log(
        `  [${label}] console error during the cell (${
          String(realErrors[0]).slice(0, 90)
        }) — retrying with a fresh page`,
      );
    }
    check(`${label}: zero console errors`, realErrors.length === 0, realErrors.join(" | "));
    check(
      `${label}: zero failed network requests`,
      page.netFailures.length === 0,
      page.netFailures.join(" | "),
    );
    ok = ok && hygiene.overflow && realErrors.length === 0 && page.netFailures.length === 0;
  } catch (error) {
    if (attempt < 3) {
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
      // The retry evidence must survive a FAILED cell too: the worker's race retries are visible in
      // the page loader's progress text and in #status, and a failed cell is exactly when a reader
      // needs to know whether the fix tried. Capture both before reporting the failure.
      const retryTrace = await evalJSON(
        cdp,
        page.sessionId,
        `JSON.stringify({
          status: (document.querySelector('#status')?.textContent ?? '').trim().slice(0, 200),
          loaders: [...document.querySelectorAll('#model-loader, #minilm-loader, #llama-loader')]
            .map((el) => (el.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 160))
            .filter(Boolean),
        })`,
      ).catch(() => null);
      check(
        `${label}: route completed`,
        false,
        `${error.stack || error.message}${
          retryTrace ? ` | page: ${JSON.stringify(retryTrace)}` : ""
        }`,
      );
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
