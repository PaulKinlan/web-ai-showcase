#!/usr/bin/env node
// Page-level degradation acceptance for qwen2-vl (web-ai-showcase-oow).
//
// The route is WebGPU-only and the ORT WebGPU backend bundled with the pinned transformers.js
// (3.7.5) crashes inside the model's own forward pass on some GPU/driver/browser builds
// ("[Concat] /model/layers.0/self_attn/Concat_7" was seen in an EARLIER run on different hardware).
// CORRECTION (oow review): on the box used for the d6adb38 run the adapter lacked shader-f16, init
// failed with "The device (webgpu) does not support fp16.", and the generation path was NEVER
// reached; the "10/10 product assertions" there were loader-label checks that pass with the fix
// reverted. This validator REQUIRES the model to fail and needs a real shader-f16 GPU to reach
// generation. For fix-vs-revert discrimination use scripts/validate-qwen2-vl-injected-errors.mjs
// (INJECTED worker errors, labelled as such). The shipped contract this validator drives FOR REAL
// in headless Chrome:
//   1. The visitor NEVER sees the raw kernel string as the user-facing message.
//   2. The page shows a LABELLED degradation: what happened + what the visitor can do.
//   3. The raw runtime text is demoted to a collapsed <details class="err-detail"> (still
//      inspectable, never the headline).
//   4. The answer region honestly says generation failed; no fake output.
// Every published rung (overview + basics + practical + wild + multi-model) is driven at desktop
// AND mobile. Intended: the 2.7 GB model downloads once into a per-run profile (reused across cells, never across runs) and later cells
// use the cache — NOT observed on the d6adb38 run (every cell logged download-required).
//
// This validator deliberately does NOT write a portfolio acceptance record: the family cannot
// generate successfully on this box, and a "passing" record would misrepresent that. A PASS here
// means "the degradation is honest and labelled", nothing more.
//
// Run: node scripts/validate-qwen2-vl-degradation.mjs [--artefacts <dir>]
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CDP,
  closePage,
  createIsolatedProfileDir,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const ARTEFACTS = process.argv.includes("--artefacts")
  ? process.argv[process.argv.indexOf("--artefacts") + 1]
  : join(tmpdir(), "qwen2-vl-degradation-artefacts");
mkdirSync(ARTEFACTS, { recursive: true });
const PROFILE_DIR = createIsolatedProfileDir("qwen2-vl-degradation");
const startCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  .trim();

const RUNGS = {
  overview: { route: "models/qwen2-vl/", prompt: null }, // default prompt is fine
  basics: { route: "models/qwen2-vl/basics/", prompt: null },
  practical: { route: "models/qwen2-vl/practical/", prompt: null },
  wild: { route: "models/qwen2-vl/wild/", prompt: "What is happening in this image?" },
  multimodel: { route: "models/qwen2-vl/multi-model/", prompt: null },
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let checks = 0;
let passed = 0;
let envPreconditionFails = 0;

function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${label}${
      detail ? ` — ${String(detail).slice(0, 300)}` : ""
    }`,
  );
  return condition;
}

function checkPrecondition(label, condition, detail = "") {
  checks++;
  if (condition) {
    passed++;
    console.log(`PASS  [PRECONDITION] ${label}${detail ? ` — ${String(detail).slice(0, 300)}` : ""}`);
  } else {
    envPreconditionFails++;
    console.log(`FAIL  [PRECONDITION] ${label}${detail ? ` — ${String(detail).slice(0, 300)}` : ""}`);
  }
  return condition;
}

// Wait until `expr` is truthy in the page (poll), returning the last value seen.
async function waitFor(cdp, sessionId, expr, timeoutMs, pollMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  let value = null;
  while (Date.now() < deadline) {
    value = await evalValue(cdp, sessionId, expr);
    if (value) return value;
    await sleep(pollMs);
  }
  return value;
}

const server = await startServer();
const url = (route) => `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`;
console.log(`profile: ${PROFILE_DIR} (persistent across cells so the model downloads once)`);
console.log(`artefacts: ${ARTEFACTS}`);
const chrome = await launchChrome({
  userDataDir: PROFILE_DIR,
  resetProfile: false,
  removeProfileOnKill: false,
  webgpu: true,
});
const cdp = new CDP(chrome.ws);

// Probe WebGPU adapter features for evidence (specifically shader-f16 requirement)
const probePage = await openPage(cdp, url("models/qwen2-vl/"));
const adapterInfo = await evalValue(
  cdp,
  probePage.sessionId,
  `(async () => {
    if (!("gpu" in navigator)) return { supported: false, reason: "navigator.gpu missing" };
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { supported: false, reason: "requestAdapter returned null" };
      const info = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
      return {
        supported: true,
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        isFallbackAdapter: adapter.isFallbackAdapter,
        features: [...adapter.features],
        hasShaderF16: adapter.features?.has?.("shader-f16") ?? false,
      };
    } catch (e) {
      return { supported: false, error: String(e?.message ?? e) };
    }
  })()`,
);
await closePage(cdp, probePage.targetId);
console.log(`WebGPU adapter probe: ${JSON.stringify(adapterInfo)}`);
console.log(`shader-f16 supported: ${adapterInfo?.hasShaderF16 ?? false}`);

const artefactIndex = { startCommit, adapter: adapterInfo, cells: [] };

let firstCell = true;
for (const [rung, cfg] of Object.entries(RUNGS)) {
  for (const [vpName, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
    const cell = `${rung}@${vpName}`;
    console.log(`\n=== cell ${cell} — ${cfg.route} ===`);
    const page = await openPage(cdp, url(cfg.route));
    await setViewport(cdp, page.sessionId, vp);
    try {
      // 1) Get the model ready: first cell downloads (explicit Download click — the page must
      //    never silently download 2.7 GB), later cells auto-init from the persistent cache.
      const state = await evalValue(
        cdp,
        page.sessionId,
        "document.querySelector('#model-loader .model-loader')?.dataset.state",
      );
      console.log(`  loader state on load: ${state}`);
      if (state === "download-required" || state === "partial") {
        await evalValue(
          cdp,
          page.sessionId,
          "document.querySelector('#model-loader .loader-actions button')?.click()",
        );
        console.log("  clicked Download — waiting for the model (one-time ~2.7 GB)…");
      } else {
        console.log("  model cached — auto-init in progress…");
      }
      const ready = await waitFor(
        cdp,
        page.sessionId,
        "document.querySelector('#model-loader .model-loader')?.dataset.state === 'ready' || " +
          "document.querySelector('#model-loader .model-loader')?.dataset.state === 'error' " +
          "? document.querySelector('#model-loader .model-loader').dataset.state : null",
        firstCell ? 30 * 60 * 1000 : 12 * 60 * 1000,
        5000,
      );
      if (!checkPrecondition(`${cell}: model initialises (WebGPU init succeeds on this box; requires shader-f16)`, ready === "ready", `state=${ready}`)) {
        // An init-time failure IS the other degradation path — still assert the labelled UI.
        const initStatus = await evalValue(
          cdp,
          page.sessionId,
          "document.querySelector('#model-loader .status')?.textContent",
        );
        check(
          `${cell}: init failure is labelled (no raw kernel text)`,
          !!initStatus && !/\[WebGPU\]|\/model\/layers|\bKernel\b/.test(initStatus),
          initStatus,
        );
        artefactIndex.cells.push({ cell, initState: ready, initStatus });
        continue;
      }

      // 2) Drive a real generation. On this box the ORT WebGPU Concat kernel crashes every time.
      if (cfg.prompt) {
        await evalValue(
          cdp,
          page.sessionId,
          `(() => { const p = document.querySelector('#prompt'); if (p) p.value = ${JSON.stringify(cfg.prompt)}; return true; })()`,
        );
      }
      await evalValue(cdp, page.sessionId, "document.querySelector('#run')?.click()");
      const settled = await waitFor(
        cdp,
        page.sessionId,
        "(() => { const s = document.querySelector('#status')?.textContent || ''; " +
          "return /failed|crashed|out of memory|took the GPU away/i.test(s) ? s : null; })()",
        8 * 60 * 1000,
        3000,
      );
      const status = settled ??
        (await evalValue(cdp, page.sessionId, "document.querySelector('#status')?.textContent || ''"));
      const shot = join(ARTEFACTS, `${cell}.png`);
      await screenshot(cdp, page.sessionId, shot);
      const detailText = await evalValue(
        cdp,
        page.sessionId,
        "document.querySelector('#status + details.err-detail pre')?.textContent || null",
      );
      const answerText = await evalValue(
        cdp,
        page.sessionId,
        `(
          document.querySelector('#answer') ??
          document.querySelector('#transcript') ??
          document.querySelector('#streamA') ??
          document.querySelector('#chat .turn.a')
        )?.textContent?.trim() || null`,
      );

      check(`${cell}: a failure state was reached (kernel crash reproduced)`, !!settled, status?.slice(0, 120));
      check(
        `${cell}: LABELLED degradation is the user-facing message`,
        /WebGPU backend crashed|took the GPU away|ran out of memory/i.test(status ?? ""),
        status?.slice(0, 160),
      );
      check(
        `${cell}: message says what the visitor can do`,
        /update your browser|different browser|Reload the page|Close other tabs/i.test(status ?? ""),
      );
      check(
        `${cell}: NO raw kernel text in the user-facing message`,
        !!status && !/\[WebGPU\]|\/model\/layers|\[Concat\]|Failed to generate kernel/.test(status),
        status?.slice(0, 160),
      );
      check(
        `${cell}: raw runtime text survives in collapsed <details.err-detail>`,
        typeof detailText === "string" && detailText.length > 0 &&
          /WebGPU|Kernel|device|memory/i.test(detailText),
        detailText?.slice(0, 120),
      );
      check(
        `${cell}: answer region honestly reports failure (no fake output)`,
        /failed/i.test(answerText ?? ""),
        answerText?.slice(0, 80),
      );
      artefactIndex.cells.push({ cell, status, detailText: detailText?.slice(0, 400), answerText, screenshot: shot });
    } finally {
      await closePage(cdp, page.targetId);
      firstCell = false;
    }
  }
}

writeFileSync(join(ARTEFACTS, "index.json"), JSON.stringify(artefactIndex, null, 2));
console.log(`\n${passed}/${checks} checks passed (${envPreconditionFails} environment precondition failures) — artefacts in ${ARTEFACTS}`);
if (envPreconditionFails > 0) {
  console.log(
    `NOTE: ${envPreconditionFails} precondition check(s) failed because this environment lacks WebGPU shader-f16.\n` +
    "The product degradation assertions passed, but the in-generation [Concat] path was not reached.\n" +
    "Exiting 1 to reflect that the run is honestly incomplete on this hardware.",
  );
}
await chrome.kill({ removeProfile: false });
server.server.close();
process.exit(passed === checks ? 0 : 1);
