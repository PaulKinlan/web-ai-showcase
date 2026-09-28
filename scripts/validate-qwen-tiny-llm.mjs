#!/usr/bin/env node
// Route-complete qwen-tiny-llm acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   onnx-community/Qwen2.5-0.5B-Instruct  (all routes — WebGPU q4f16 / WASM q4)
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
const RUN_RECORD = join(repoRoot, "models/qwen-tiny-llm/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "qwen-tiny-llm-acceptance-"));
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/qwen-tiny-llm/",
  basics: "models/qwen-tiny-llm/basics/",
  practical: "models/qwen-tiny-llm/practical/",
  wild: "models/qwen-tiny-llm/wild/",
};
const RUNGS = {
  overview: { trigger: "#send", input: "#input", tok: "#rTok" },
  basics: { trigger: "#send", input: "#input", tok: "#rTok" },
  practical: { trigger: "#run", input: "#input", tok: "#rTok" },
  wild: { trigger: "#send", input: "#input", tok: "#rTok" },
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
      console.log(`  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

// The family worker is observable through the CDP Target domain: its targetInfo url is empty,
// so filter by type and attach on demand (deterministic — no attach-event racing).
async function familyWorkerTarget(cdp) {
  const t = await cdp.send("Target.getTargets", {}, undefined, 10_000).catch(() => null);
  if (!t) {
    console.log("  [worker-counter] getTargets failed");
    return null;
  }
  const infos = t?.targetInfos ?? [];
  const worker = infos.find((i) => i.type === "worker");
  if (!worker) {
    console.log(
      `  [worker-counter] no worker target; seen: ${infos.map((i) => i.type).join(",")}`,
    );
  }
  return worker ?? null;
}

const WRAP_WORKER = `(() => {
  if (self.__hooked) return "already";
  self.__hooked = true;
  self.__chunks = 0;
  self.__doneTokens = null;
  const orig = self.postMessage.bind(self);
  self.postMessage = (msg) => {
    if (msg && typeof msg === "object") {
      if (msg.type === "token") self.__chunks += 1;
      if (msg.type === "done") self.__doneTokens = Number(msg.tokens ?? 0);
    }
    return orig(msg);
  };
  return "wrapped";
})()`;

async function attachWorkerCounter(cdp) {
  const worker = await familyWorkerTarget(cdp);
  if (!worker) return null;
  const att = await cdp.send(
    "Target.attachToTarget",
    { targetId: worker.targetId, flatten: true },
    undefined,
    10_000,
  ).catch((e) => {
    console.log(`  [worker-counter] attach failed: ${e.message}`);
    return null;
  });
  if (!att?.sessionId) {
    console.log(`  [worker-counter] attach returned no session (${worker.type})`);
    return null;
  }
  await cdp.send("Runtime.enable", {}, att.sessionId).catch(() => {});
  // Idempotent in the worker realm: a second attach's wrap is a no-op.
  await cdp.send(
    "Runtime.evaluate",
    { expression: WRAP_WORKER, returnByValue: true },
    att.sessionId,
    10_000,
  ).catch(() => {});
  return att.sessionId;
}

async function readWorkerCounts(cdp, sid) {
  const r = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `JSON.stringify({ chunks: self.__chunks ?? null, hooked: self.__hooked ?? null, doneTokens: self.__doneTokens ?? null })`,
      returnByValue: true,
    },
    sid,
    10_000,
  ).catch((e) => {
    console.log(`  [worker-counter] read failed: ${e.message}`);
    return null;
  });
  if (!r) return null;
  if (r.exceptionDetails) {
    console.log(`  [worker-counter] read exception: ${JSON.stringify(r.exceptionDetails).slice(0, 200)}`);
  }
  console.log(`  [worker-counter] raw read: ${JSON.stringify(r.result).slice(0, 200)}`);
  if (r?.result?.value) {
    try {
      return JSON.parse(r.result.value);
    } catch {
      // fall through
    }
  }
  return null;
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

async function drive(cdp, sid, label, { trigger, input }) {
  // Atomic set+click, retried: a first-visit service-worker reload can briefly blank the page,
  // and a swallowed click must never read as a hung generation.
  for (let attempt = 1; attempt <= 6; attempt++) {
    const phase = await evaluate(
      cdp,
      sid,
      `(() => {
        const el = document.querySelector('${input}');
        if (!el) return "no-input";
        el.value = ${JSON.stringify(PROMPT)};
        el.dispatchEvent(new Event('input', { bubbles: true }));
        const btn = document.querySelector('${trigger}');
        if (!btn) return "no-trigger";
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

async function exercise(cdp, rung, viewportName, viewport) {
  const route = ROUTES[rung];
  const label = `${viewportName} ${rung}`;
  const cfg = RUNGS[rung];
  const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/${route}`);
  await setViewport(cdp, page.sessionId, viewport);
  let ok = true;

  try {
    await ensureReady(cdp, page.sessionId, label);
    const workerSid = await attachWorkerCounter(cdp);
    await drive(cdp, page.sessionId, label, cfg);
    await waitFor(
      cdp,
      page.sessionId,
      `(() => {
        const btn = document.querySelector('${cfg.trigger}');
        const readout = document.querySelector('#readout');
        const st = document.querySelector('#status')?.textContent || '';
        if (/failed/i.test(st)) return true;
        return btn && !btn.disabled && readout && readout.hidden === false;
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

    const counts = await readWorkerCounts(cdp, workerSid);
    const proof = await evalJSON(
      cdp,
      page.sessionId,
      `JSON.stringify({ tok: document.querySelector('${cfg.tok}')?.textContent ?? null })`,
    );
    const chunks = counts?.chunks ?? null;
    const realIds = counts?.doneTokens ?? null;
    const tok = Number(proof.tok);
    ok = check(
      `${label}: worker counted decoded chunks at the source`,
      Number.isInteger(chunks) && chunks >= 1,
      { chunks, counts },
    ) && ok;
    ok = check(
      `${label}: readout shows the resolved token-ID count`,
      Number.isFinite(tok) && tok >= 1 && tok === realIds,
      { readout: proof.tok, doneTokens: realIds },
    ) && ok;
    ok = check(
      `${label}: token-ID count differs from decoded-chunk count (0ly proof)`,
      Number.isInteger(chunks) && chunks >= 1 && chunks !== realIds,
      { chunks, tokens: realIds },
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
const chrome = await launchChrome({ webgpu: true });
const cdp = new CDP(chrome.ws);
try {
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "desktop", DESKTOP);
  for (const [name] of Object.entries(ROUTES)) await exercise(cdp, name, "mobile", MOBILE);
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
const allRoutesPassed = results.length === 8 && results.every((r) => r.pass);
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
