#!/usr/bin/env node
// Route-complete smollm2-chat acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   HuggingFaceTB/SmolLM2-360M-Instruct  (all routes; multi-model adds the retriever below)
//   Xenova/all-MiniLM-L6-v2              (multi-model RAG retriever)
//
// web-ai-showcase-0ly acceptance: the readout must count generated token IDs, NOT decoded text
// chunks. Each cell intercepts the worker's messages (validator-side only) and counts the decoded
// chunks, then asserts the displayed/returned token count DIFFERS from that chunk count — that
// difference is the proof the streamer counts IDs now.
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
const RUN_RECORD = join(repoRoot, "models/smollm2-chat/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "smollm2-chat-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct";
const ROUTES = {
  overview: "models/smollm2-chat/",
  basics: "models/smollm2-chat/basics/",
  practical: "models/smollm2-chat/practical/",
  wild: "models/smollm2-chat/wild/",
  multimodel: "models/smollm2-chat/multi-model/",
};
const PROMPT = "Describe the unbelievably heterogeneous thundercloud formation briefly.";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
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
      const state = await evalJSON(
        cdp,
        sessionId,
        `JSON.stringify({
        status: (document.querySelector('#status')?.textContent || '').slice(0, 80),
        inputLen: (document.querySelector('#input, #q')?.value || '').length,
        send: document.querySelector('#send')?.disabled ?? null,
        run: document.querySelector('#run')?.disabled ?? null,
        ask: document.querySelector('#ask')?.disabled ?? null,
        readout: document.querySelector('#readout')?.hidden ?? null,
        tok: document.querySelector('#rTok')?.textContent || null,
        chunks: window.__chunks ?? null,
      })`,
      ).catch(() => "(state eval failed)");
      console.log(
        `  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s ${
          JSON.stringify(state)
        }`,
      );
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

// Count the page's DECODED CHUNKS, validator-side only: the page's engine is a module-scoped
// binding, but a dynamic import of the same URL resolves to the SAME module instance, so wrapping
// engine.chat here intercepts every onToken the page receives — one per decoded text chunk. The
// real token-ID count arrives separately in the done payload (and the #rTok readout).
async function installChunkCounter(cdp, sessionId, route) {
  // This family's engine module is smollm2.js (not engine.js); every rung imports it, so the
  // dynamic import below resolves to the SAME module instance the page already uses.
  const moduleUrl = "/web-ai-showcase/models/smollm2-chat/smollm2.js";
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      window.__chunks = 0; window.__tokens = [];
      return (async () => {
        const m = await import(${JSON.stringify(moduleUrl)});
        const Engine = m?.SmolLMEngine;
        if (!Engine?.prototype?.chat) return false;
        const orig = Engine.prototype.chat;
        Engine.prototype.chat = function (...args) {
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
      })`,
    ).catch(() => ({ ready: -1, total: -1 }));
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

/** Fill an input, fire the trigger, wait for the button to re-enable (or the readout to open). */
async function drive(cdp, sid, label, { trigger, input, text, extra = "" }) {
  await evaluate(
    cdp,
    sid,
    `(() => {
    const el = document.querySelector('${input}');
    el.value = ${JSON.stringify(text ?? PROMPT)};
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  })()`,
  );
  await evaluate(
    cdp,
    sid,
    `(() => { document.querySelector('${trigger}').click(); return true; })()`,
  );
  await waitFor(
    cdp,
    sid,
    `(() => {
      const btn = document.querySelector('${trigger}');
      const readout = document.querySelector('#readout');
      const st = document.querySelector('#status')?.textContent || '';
      if (/failed/i.test(st)) return true;
      return btn && !btn.disabled && readout && readout.hidden === false;
    })()`,
    900_000,
    `${label} generation`,
    5_000,
  );
  const termState = await evalJSON(
    cdp,
    sid,
    `JSON.stringify({
    status: document.querySelector('#status')?.textContent?.slice(0, 120),
    readout: document.querySelector('#readout')?.hidden === false,
    chunks: window.__chunks ?? null,
  })`,
  ).catch(() => ({ error: "state eval failed" }));
  if (/failed/i.test(termState.status ?? "")) {
    throw new Error(`generation failed on-page: ${termState.status}`);
  }
}

const RUNGS = {
  overview: { trigger: "#send", input: "#input", tok: "#rTok" },
  basics: { trigger: "#send", input: "#input", tok: "#rTok" },
  practical: { trigger: "#run", input: "#src", tok: "#rTok" },
  // The adventure page has no #q/#run: Begin starts the scene, #act continues it. It does render
  // its own real token count + tok/s (#rTok/#rTps), unlike the 135M sibling's cold/warm pair.
  wild: { trigger: "#begin", input: "#setting", tok: "#rTok" },
  // No #rTok on this rung: the resolved count is proven from the done payload + the 0ly delta.
  multimodel: { trigger: "#ask", input: "#q", tok: null },
};

async function exercise(cdp, rung, viewportName, viewport) {
  const route = ROUTES[rung];
  const label = `${viewportName} ${rung}`;
  const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/${route}`);
  await setViewport(cdp, page.sessionId, viewport);
  let ok = true;

  try {
    await ensureReady(cdp, page.sessionId, label);
    await installChunkCounter(cdp, page.sessionId, route);
    const cfg = RUNGS[rung];

    if (rung === "overview") {
      // See-inside surface: real next-token distribution (overview is the inside rung).
      await drive(cdp, page.sessionId, label, cfg);
      await evaluate(cdp, page.sessionId, `document.querySelector('#peek')?.click()`);
      await waitFor(
        cdp,
        page.sessionId,
        `(() => {
          const t = document.querySelector('#topk');
          return t && (t.innerText || '').length > 20 && !/Computing/.test(t.innerText);
        })()`,
        300_000,
        `${label} next-token distribution`,
        5_000,
      );
      const dist = await evaluate(
        cdp,
        page.sessionId,
        `(document.querySelector('#topk')?.innerText || '').slice(0, 120)`,
      );
      ok = check(`${label}: see-inside top-k distribution renders`, dist.length > 20, dist) && ok;
    } else if (rung === "wild") {
      // The adventure page reports its own real token count and tok/s after a turn.
      await drive(cdp, page.sessionId, label, cfg);
      const done = await evalJSON(
        cdp,
        page.sessionId,
        `JSON.stringify({
        tok: document.querySelector('#rTok')?.textContent || '',
        tps: document.querySelector('#rTps')?.textContent || '',
      })`,
      );
      ok = check(
        `${label}: adventure reports a real token count + tok/s`,
        /^[\d.]+$/.test(done.tok) && /^[\d.]+$/.test(done.tps),
        done,
      ) && ok;
    } else {
      await drive(cdp, page.sessionId, label, cfg);
    }

    // THE 0ly PROOF: the displayed token count is a real ID count and differs from the number of
    // decoded text chunks the worker streamed to the page.
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({
      tok: ${cfg.tok ? `document.querySelector('${cfg.tok}')?.textContent || ''` : "null"},
      chunks: window.__chunks ?? null,
      tokenIds: window.__tokens ?? [],
    })`,
    );
    const realIds = proof.tokenIds.reduce((a, b) => a + Number(b || 0), 0);
    ok = check(
      `${label}: worker resolved real generated token-ID counts`,
      proof.tokenIds.length >= 1 && realIds >= 1,
      proof,
    ) && ok;
    if (cfg.tok) {
      const tok = Number(proof.tok);
      ok = check(
        `${label}: readout shows the resolved token-ID count`,
        Number.isFinite(tok) && tok === realIds,
        { readout: proof.tok, resolved: realIds },
      ) && ok;
    }
    ok = check(
      `${label}: token-ID count differs from decoded-chunk count (0ly proof)`,
      Number.isInteger(proof.chunks) && proof.chunks >= 1 && proof.chunks !== realIds,
      { chunks: proof.chunks, tokens: realIds },
    ) && ok;

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
    ok = false;
    check(`${label}: route completed`, false, error.stack || error.message);
  } finally {
    results.push({ route, viewport: viewportName, pass: ok });
    await closePage(cdp, page.targetId).catch(() => {});
  }
}

const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
try {
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "desktop", DESKTOP);
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "mobile", MOBILE);
} finally {
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  try {
    rmSync(PROFILE_DIR, { recursive: true, force: true });
  } catch { /* ignore */ }
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
