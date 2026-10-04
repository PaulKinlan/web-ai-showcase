#!/usr/bin/env node
// Route-complete granite-4-350m acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   onnx-community/granite-4.0-350m-ONNX-web  (all routes — WASM q4; multi-model also
//   retrieves with Xenova/all-MiniLM-L6-v2)
//
// web-ai-showcase-0ly acceptance: the readout counts generated token IDs, NOT decoded text
// chunks. The validator attaches to the family worker's CDP target, counts the worker's decoded
// "token" posts (chunks) and its "done" post (real IDs), and asserts the displayed count equals
// the done payload AND differs from the chunk count — that difference is the proof.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CDP,
  checkPairwiseDivergence,
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
const RUN_RECORD = join(repoRoot, "models/granite-4-350m/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "granite-4-350m-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/granite-4-350m/",
  basics: "models/granite-4-350m/basics/",
  practical: "models/granite-4-350m/practical/",
  wild: "models/granite-4-350m/wild/",
  "multi-model": "models/granite-4-350m/multi-model/",
};
const RUNGS = {
  overview: { trigger: "#send", input: "#prompt", tok: "#rTokens" },
  basics: { trigger: "#send", input: "#prompt", tok: "#rTokens" },
  practical: { trigger: "#run", input: "#text", tok: "#rTokens" },
  // Wild runs the same prompt at three temperatures and its readout has no token count; the
  // multi-model rung retrieves first and its readout shows timings only. Both prove the resolved
  // ID count from the payload and the chunk divergence.
  wild: { trigger: "#run", input: "#prompt", tok: null, doneText: "Compare the three" },
  "multi-model": {
    trigger: "#run",
    input: "#query",
    tok: null,
    doneText: "Answered from the retrieved note",
    // The page's own five default notes are the retrieval corpus; the drive only replaces the
    // question. (A prefilled single-line corpus would fail the page's "at least two notes" check.)
    extra: [],
  },
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
// QwenEngine.prototype.chat here intercepts every onToken the page receives — one per decoded text
// chunk. The real token-ID count arrives separately in the resolved payload (and the #rTok
// readout). This is the mechanism the passing smollm2-chat validator uses for its family class;
// qwen's class is QwenEngine, which is why an `Engine` wrap never intercepted here.
async function installChunkCounter(cdp, sessionId) {
  // This family has TWO generation paths: the chat engine (GraniteEngine.chat, used by
  // overview/basics/practical/wild) and the RAG engine (RagEngine.run, used by multi-model).
  // A dynamic import of each module URL resolves to the SAME instance the page uses, so wrapping
  // the class prototype intercepts every onToken the page receives — one per decoded text chunk —
  // while the resolved payload carries the real generated-ID count.
  const targets = [
    { url: "/web-ai-showcase/models/granite-4-350m/engine.js", cls: "GraniteEngine", method: "chat" },
    { url: "/web-ai-showcase/models/granite-4-350m/mm.js", cls: "RagEngine", method: "run" },
  ];
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = []; window.__calls = [];
      return (async () => {
        for (const t of ${JSON.stringify(targets)}) {
          const m = await import(t.url);
          const Klass = m?.[t.cls];
          if (!Klass?.prototype?.[t.method]) continue;
          const orig = Klass.prototype[t.method];
          Klass.prototype[t.method] = function (...args) {
            let callChunks = 0;
            const opts = args.at(-1);
            if (opts && typeof opts === "object" && typeof opts.onToken === "function") {
              const inner = opts.onToken;
              opts.onToken = (...cb) => {
                callChunks += 1;
                window.__chunks += 1;
                return inner(...cb);
              };
            }
            const result = orig.apply(this, args);
            if (result && typeof result.then === "function") {
              result.then((res) => {
                const count = Number(res?.tokens ?? 0);
                window.__tokens = window.__tokens ?? [];
                window.__tokens.push(count);
                window.__calls = window.__calls ?? [];
                window.__calls.push({ chunks: callChunks, tokens: count });
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

async function drive(cdp, sid, label, { trigger, input, extra = [] }) {
  // Atomic set+click, retried: a first-visit service-worker reload can briefly blank the page,
  // and a swallowed click must never read as a hung generation.
  for (let attempt = 1; attempt <= 6; attempt++) {
    const phase = await evaluate(
      cdp,
      sid,
      `(() => {
        for (const [sel, val] of ${JSON.stringify(extra)}) {
          const e = document.querySelector(sel);
          if (e) { e.value = val; e.dispatchEvent(new Event('input', { bubbles: true })); }
        }
        const el = ${input ? `document.querySelector('${input}')` : "null"};
        if (${input ? "true" : "false"} && !el) return "no-input";
        if (el) {
          el.value = ${JSON.stringify(PROMPT)};
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        const btn = document.querySelector('${trigger}');
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
            readout: document.querySelector('#readout')?.hidden === false,
          })`,
        ).catch(() => null);
        if (s && (s.busy || s.readout)) {
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
    await drive(cdp, page.sessionId, label, cfg);
    await waitFor(
      cdp,
      page.sessionId,
      `(() => {
        const readout = document.querySelector('#readout');
        const st = document.querySelector('#status')?.textContent || '';
        if (/failed|add at least|type a question/i.test(st)) return true;
        ${cfg.doneText ? `if (st.includes(${JSON.stringify(cfg.doneText)})) return true;` : ""}
        // The READOUT is the completion signal where a rung has no final status line.
        return readout && readout.hidden === false;
      })()`,
      900_000,
      `${label} generation`,
      5_000,
    );

    if (rung === "overview") {
      // See-inside surface: real next-token distribution.
      await evaluate(cdp, page.sessionId, `document.querySelector('#topkBtn').click()`);
      await waitFor(
        cdp,
        page.sessionId,
        `(() => {
          const t = document.querySelector('#topk');
          return t && (t.innerText || '').length > 25 && !/Computing/.test(t.innerText);
        })()`,
        600_000,
        `${label} next-token distribution`,
        5_000,
      );
      const dist = await evaluate(
        cdp,
        page.sessionId,
        `(document.querySelector('#topk')?.innerText || '').slice(0, 120)`,
      );
      ok = check(`${label}: see-inside top-k distribution renders`, dist.length > 20, dist) && ok;
    }

    // THE 0ly PROOF: the displayed token count is a real ID count and differs from the number of
    // decoded text chunks the page received through the engine.
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({
        tok: document.querySelector('${cfg.tok}')?.textContent ?? null,
        chunks: window.__chunks ?? null,
        tokenIds: window.__tokens ?? [],
        calls: window.__calls ?? [],
      })`,
    );
    const chunks = proof.chunks;
    const realIds = (proof.tokenIds ?? []).reduce((a, b) => a + Number(b || 0), 0);
    const calls = Array.isArray(proof.calls) && proof.calls.length > 0
      ? proof.calls
      : [{ chunks, tokens: realIds }];
    const DIVERGENCE_MIN_TOKENS = 10;
    const divResult = checkPairwiseDivergence(calls, { minTokens: DIVERGENCE_MIN_TOKENS });
    const streamedChunks = divResult.anyStreamed;
    const needsDivergence = divResult.needsDivergence;
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
    // PER-CELL 0ly PROOF (web-ai-showcase-4sz, 6b5): EVERY cell must prove the displayed count is not
    // the decoded-chunk count, checked pairwise per generation on multi-generation rungs.
    ok = check(
      streamedChunks
        ? `${label}: resolved ID count differs from the decoded-chunk count${
          needsDivergence ? "" : " (short generation, equality allowed)"
        }`
        : `${label}: non-streaming cell received no chunks and resolved real IDs`,
      streamedChunks ? divResult.pass : realIds >= 1,
      { chunks, realIds, calls, streaming, needsDivergence },
    ) && ok;
    divergence.push({ label, viewport: viewportName, chunks, tokens: realIds, streaming, diverged: needsDivergence });

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
      console.log(`  [${label}] attempt ${attempt} stalled (${String(error.message).slice(0, 140)}) — retrying with a fresh page`);
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
    divergence.filter((d) => d.viewport === "desktop").every((d) =>
      d.chunks > 0 ? d.tokens !== d.chunks : d.tokens >= 1),
    divergence.filter((d) => d.viewport === "desktop"),
  );
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "mobile", MOBILE);
  check(
    `mobile: EVERY cell proves the readout is not the decoded-chunk count (per-cell 0ly proof)`,
    divergence.filter((d) => d.viewport === "mobile").length === Object.keys(ROUTES).length &&
    divergence.filter((d) => d.viewport === "mobile").every((d) =>
      d.chunks > 0 ? d.tokens !== d.chunks : d.tokens >= 1),
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
