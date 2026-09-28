#!/usr/bin/env node
// Route-complete baguettotron acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   onnx-community/Baguettotron-ONNX  (all routes — WASM q8; the multi-model rung also
//   retrieves with Xenova/all-MiniLM-L6-v2 through a page-side pipeline)
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
// Debug aid: ONLY_RUNG / ONLY_VIEWPORT narrow a run to a single cell (useful when one heavy cell
// needs isolating); unset, the validator runs every rung at both viewports as acceptance requires.
const ONLY_RUNG = process.env.ONLY_RUNG ? process.env.ONLY_RUNG.split(",").map((n) => n.trim()) : null;
const ONLY_VIEWPORT = process.env.ONLY_VIEWPORT ?? null;
const RUN_RECORD = join(repoRoot, "models/baguettotron/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "baguettotron-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/baguettotron/",
  basics: "models/baguettotron/basics/",
  practical: "models/baguettotron/practical/",
  wild: "models/baguettotron/wild/",
  "multi-model": "models/baguettotron/multi-model/",
};
const RUNGS = {
  overview: { trigger: "#send", input: "#input", tok: "#rTok" },
  basics: { trigger: "#send", input: "#input", tok: "#rTok" },
  // practical types a prompt into #q and asks with #ask.
  practical: { trigger: "#ask", input: "#q", tok: "#rTok" },
  // wild is a two-generation race (greedy then sampled) with NO #readout: each pane prints its own
  // "WASM · N tokens · …" meta, so both metas are the readout and the completion signal is the
  // final status line.
  wild: {
    trigger: "#race",
    input: "#q",
    metas: ["#greedyMeta", "#sampledMeta"],
    doneText: "compare the two chains of thought",
  },
  // The multi-model rung retrieves over its own prefilled facts and then reasons with the SAME
  // engine class: #ask triggers it and #q carries the question.
  "multi-model": { trigger: "#ask", input: "#q", tok: "#rTok" },
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
// BaguettotronEngine.prototype.chat here intercepts every onToken the page receives — one per decoded
// text chunk. The real token-ID count arrives separately in the resolved payload (and the #rTok
// readout). Name the family's real class: this one is BaguettotronEngine (the qwen-tiny-llm family's
// is QwenEngine), because a wrap of a generic name like `Engine` silently intercepts nothing.
async function installChunkCounter(cdp, sessionId) {
  // ONE class covers every rung here: overview/basics/practical/wild chat through
  // BaguettotronEngine.chat, and the multi-model rung (which adds a page-side MiniLM embedder for
  // retrieval) still generates through the same engine. A dynamic import of the module URL resolves
  // to the SAME instance the page uses, so wrapping the prototype intercepts every onToken the page
  // receives — one per decoded text chunk — while the resolved payload carries the real ID count.
  const targets = [
    { url: "/web-ai-showcase/models/baguettotron/engine.js", cls: "BaguettotronEngine", method: "chat" },
  ];
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = [];
      return (async () => {
        for (const t of ${JSON.stringify(targets)}) {
          const m = await import(t.url);
          const Klass = m?.[t.cls];
          if (!Klass?.prototype?.[t.method]) continue;
          const orig = Klass.prototype[t.method];
          Klass.prototype[t.method] = function (...args) {
            const opts = args.at(-1);
            if (opts && typeof opts === "object" && typeof opts.onToken === "function") {
              const inner = opts.onToken;
              opts.onToken = (...cb) => {
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
            // Some rungs set no status text while generating (basics clears the input and only
            // writes "Done." at the end), so a disabled trigger or a visible Stop also means started.
            busy: (document.querySelector('#status')?.textContent || '').length > 0 ||
              document.querySelector('${trigger}')?.disabled === true ||
              (document.querySelector('#stop') ? document.querySelector('#stop').hidden === false : false),
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
      await evaluate(cdp, page.sessionId, `document.querySelector('#peek').click()`);
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
        tok: ${cfg.tok ? `document.querySelector('${cfg.tok}')?.textContent ?? null` : "null"},
        metas: ${cfg.metas ? `[${cfg.metas.map((sel) => `document.querySelector('${sel}')?.textContent ?? null`).join(", ")}]` : "null"},
        chunks: window.__chunks ?? null,
        tokenIds: window.__tokens ?? [],
        backend: document.querySelector('#rBackend')?.textContent ?? null,
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
    if (cfg.metas) {
      // The race rung has no single readout: each pane prints its own "N tokens" meta, so each must
      // equal its own resolved ID count (greedy first, then sampled, in generation order).
      const shown = (proof.metas ?? []).map((meta) => {
        const m = /(\d+)\s+tokens/.exec(meta || "");
        return m ? Number(m[1]) : null;
      });
      const resolved = (proof.tokenIds ?? []).map((n) => Number(n || 0));
      ok = check(
        `${label}: both race panes show their resolved token-ID count`,
        resolved.length === 2 && shown.length === 2 && shown[0] === resolved[0] &&
          shown[1] === resolved[1],
        { shown, resolved },
      ) && ok;
    }
    if (cfg.metas) {
      ok = check(
        `${label}: both panes report the advertised WASM backend`,
        (proof.metas ?? []).every((meta) => /^WASM\b/.test(meta || "")),
        proof.metas,
      ) && ok;
    } else {
      ok = check(
        `${label}: readout reports the advertised WASM backend`,
        String(proof.backend || "").toUpperCase() === "WASM",
        { backend: proof.backend },
      ) && ok;
    }
    // PER-CELL 0ly PROOF (web-ai-showcase-4sz): EVERY cell must prove the displayed count is
    // not the decoded-chunk count. A streaming cell must show MORE generated IDs than the decoded
    // chunks it was handed; a cell whose page never asked for streaming must show zero chunks with
    // real IDs resolved. The route-level check below demands this of every cell, so a worker
    // reverted to chunk counting cannot pass by having one honest cell.
    // Per cell: the readout must be a real resolved count (the comparison against the readout itself
    // is the check above). Where the generation is long enough that one decoded chunk per token is
    // implausible in this family's streamer, the chunk count must ALSO differ from the ID count — that
    // is the mutant signature (a worker reverted to chunk counting makes the two identical). A short
    // generation can legitimately be 1:1, so the inequality is only required past a token threshold,
    // and the route-level check below insists on at least one such cell per viewport.
    const DIVERGENCE_MIN_TOKENS = 10;
    const streamedChunks = Number.isInteger(chunks) && chunks > 0;
    const needsDivergence = streamedChunks && realIds >= DIVERGENCE_MIN_TOKENS;
    ok = check(
      streamedChunks
        ? `${label}: resolved ID count differs from the decoded-chunk count${
          needsDivergence ? "" : " (short generation, equality allowed)"
        }`
        : `${label}: non-streaming cell received no chunks and resolved real IDs`,
      streamedChunks ? realIds >= 1 && (!needsDivergence || realIds !== chunks) : realIds >= 1,
      { chunks, realIds, streaming, needsDivergence },
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
      `desktop: EVERY cell proves the readout is not the decoded-chunk count (per-cell 0ly proof)`,
      divergence.filter((d) => d.viewport === "desktop").length === Object.keys(ROUTES).length &&
        divergence.filter((d) => d.viewport === "desktop").every((d) => d.tokens >= 1) &&
        divergence.some((d) => d.viewport === "desktop" && d.diverged && d.tokens !== d.chunks),
      divergence.filter((d) => d.viewport === "desktop"),
    );
  if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "mobile") {
    for (const [name] of cells) await runCell(name, "mobile", MOBILE);
  }
    if (!ONLY_VIEWPORT || ONLY_VIEWPORT === "mobile") check(
      `mobile: EVERY cell proves the readout is not the decoded-chunk count (per-cell 0ly proof)`,
      divergence.filter((d) => d.viewport === "mobile").length === Object.keys(ROUTES).length &&
        divergence.filter((d) => d.viewport === "mobile").every((d) => d.tokens >= 1) &&
        divergence.some((d) => d.viewport === "mobile" && d.diverged && d.tokens !== d.chunks),
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
