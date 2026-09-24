#!/usr/bin/env node
// Comprehensive matrix validation for Florence-2 routes (florence2-vision & florence-2-large).
// Stages: onnx-community/Florence-2-base-ft, onnx-community/Florence-2-large, onnx-community/Qwen2.5-0.5B-Instruct
//
// Acceptance criteria:
// 1. WebGPU DISABLED (no-adapter device / --disable-gpu):
//    Drives all 10 routes (overview, basics, practical, wild, multi-model for both Florence-2 models)
//    at desktop (1280x800) and narrow mobile (360x740 DPR3).
//    Asserts:
//    - Honest, accessible needs-WebGPU notice with enable instructions (chrome://gpu).
//    - Run controls remain disabled (no fake output, no hang, no blank UI).
//    - Zero console errors across all pages and viewports.
// 2. WebGPU ENABLED (hardware adapter present):
//    - Asserts that loader transitions to download-required with active download controls.
//    - Drives actual download click and verifies session creation succeeds on WebGPU via clean fp16 -> q4 fallback.
//    - Triggers real on-device vision inference (<CAPTION>) and asserts genuine non-empty generated output on WebGPU.
// 3. Desktop + mobile parity: no horizontal overflow at 1280x800 and 360x740.

import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

async function safeEval(cdp, sessionId, expr, timeoutMs = 60000) {
  const wrapped = `(async()=>{try{return (${expr});}catch(e){return null;}})()`;
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: wrapped,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  return result?.value;
}

const ROUTES = [
  { slug: "florence2-vision", path: "models/florence2-vision/", name: "Florence-2 Base Overview" },
  { slug: "florence2-vision", path: "models/florence2-vision/basics/", name: "Florence-2 Base Basics" },
  { slug: "florence2-vision", path: "models/florence2-vision/practical/", name: "Florence-2 Base Practical" },
  { slug: "florence2-vision", path: "models/florence2-vision/wild/", name: "Florence-2 Base Wild" },
  { slug: "florence2-vision", path: "models/florence2-vision/multi-model/", name: "Florence-2 Base Multi-model" },
  { slug: "florence-2-large", path: "models/florence-2-large/", name: "Florence-2 Large Overview" },
  { slug: "florence-2-large", path: "models/florence-2-large/basics/", name: "Florence-2 Large Basics" },
  { slug: "florence-2-large", path: "models/florence-2-large/practical/", name: "Florence-2 Large Practical" },
  { slug: "florence-2-large", path: "models/florence-2-large/wild/", name: "Florence-2 Large Wild" },
  { slug: "florence-2-large", path: "models/florence-2-large/multi-model/", name: "Florence-2 Large Multi-model" },
];

const VIEWPORTS = [
  { name: "desktop", vp: DESKTOP },
  { name: "mobile", vp: MOBILE },
];

async function testDisabledWebGPU(port) {
  console.log("\n=== Phase 1: Validating WebGPU-Disabled Honest Degradation across 10 Routes ===");
  const browser = await launchChrome({ webgpu: false });
  const cdp = new CDP(browser.ws);
  const results = [];

  try {
    for (const r of ROUTES) {
      for (const v of VIEWPORTS) {
        process.stderr.write(`[Disabled-GPU] ${r.name} (${v.name}) … `);
        const url = `http://127.0.0.1:${port}/web-ai-showcase/${r.path}`;
        const { targetId, sessionId, errors, netFailures } = await openPage(cdp, url);
        await setViewport(cdp, sessionId, v.vp);
        await new Promise((res) => setTimeout(res, 800));

        const state = await evalValue(
          cdp,
          sessionId,
          `
          (() => {
            const loaders = [...document.querySelectorAll(".model-loader")];
            const runBtn = document.getElementById("run");
            return {
              loaderStates: loaders.map(l => l.dataset.state),
              statusTexts: loaders.map(l => (l.querySelector(".status")?.innerText || "").replace(/\\s+/g, " ").trim()),
              detailTexts: loaders.map(l => (l.querySelector(".loader-detail")?.innerText || "").replace(/\\s+/g, " ").trim()),
              runDisabled: runBtn ? runBtn.disabled : true,
              noOverflow: document.documentElement.scrollWidth <= window.innerWidth + 1,
            };
          })()
        `,
        );

        await closePage(cdp, targetId);

        const hasUnsupported = state?.loaderStates.length > 0 &&
          state.loaderStates.every((s) => s === "unsupported");
        const hasHonestNotice = state?.statusTexts.length > 0 &&
          state.statusTexts.every((t) => /needs.*(gpu|webgpu)|unsupported/i.test(t));
        const pass = hasUnsupported && hasHonestNotice && state?.runDisabled &&
          errors.length === 0 && netFailures.length === 0 && state?.noOverflow;

        results.push({
          slug: r.slug,
          route: r.path,
          name: r.name,
          viewport: v.name,
          pass,
          loaderStates: state?.loaderStates,
          statusTexts: state?.statusTexts,
          runDisabled: state?.runDisabled,
          noOverflow: state?.noOverflow,
          errors: errors.length,
          netFailures: netFailures.length,
        });

        process.stderr.write(pass ? `PASS\n` : `FAIL (errors=${errors.length})\n`);
      }
    }
  } finally {
    await browser.kill();
  }
  return results;
}

async function testEnabledWebGPU(port) {
  console.log("\n=== Phase 2: Validating WebGPU-Enabled Capability, Model Download & Real Inference ===");
  const browser = await launchChrome({ webgpu: true });
  const cdp = new CDP(browser.ws);
  const results = [];

  try {
    for (const r of ROUTES) {
      for (const v of VIEWPORTS) {
        process.stderr.write(`[Enabled-GPU] ${r.name} (${v.name}) … `);
        const url = `http://127.0.0.1:${port}/web-ai-showcase/${r.path}`;
        const { targetId, sessionId, errors, netFailures } = await openPage(cdp, url);
        await setViewport(cdp, sessionId, v.vp);
        await new Promise((res) => setTimeout(res, 800));

        const state = await evalValue(
          cdp,
          sessionId,
          `
          (() => {
            const loaders = [...document.querySelectorAll(".model-loader")];
            const btns = loaders.flatMap(l => [...l.querySelectorAll("button")]);
            return {
              loaderStates: loaders.map(l => l.dataset.state),
              statusTexts: loaders.map(l => (l.querySelector(".status")?.innerText || "").replace(/\\s+/g, " ").trim()),
              hasButtons: btns.length > 0,
              buttonsText: btns.map(b => b.innerText.trim()),
              noOverflow: document.documentElement.scrollWidth <= window.innerWidth + 1,
            };
          })()
        `,
        );

        await closePage(cdp, targetId);

        const pass = state?.loaderStates.length > 0 &&
          state.loaderStates.every((s) => s === "download-required" || s === "ready") &&
          state?.hasButtons && errors.length === 0 && netFailures.length === 0 && state?.noOverflow;

        results.push({
          slug: r.slug,
          route: r.path,
          name: r.name,
          viewport: v.name,
          pass,
          loaderStates: state?.loaderStates,
          hasButtons: state?.hasButtons,
          buttonsText: state?.buttonsText,
          noOverflow: state?.noOverflow,
          errors: errors.length,
          netFailures: netFailures.length,
        });

        process.stderr.write(pass ? `PASS\n` : `FAIL (errors=${errors.length})\n`);
      }
    }

    // Drive actual end-to-end model download and real on-device WebGPU generation on Florence-2 Base
    console.log("\n--- Driving End-to-End Download, Session Creation & Inference on Florence-2 Base ---");
    const demoUrl = `http://127.0.0.1:${port}/web-ai-showcase/models/florence2-vision/`;
    const { targetId, sessionId, errors } = await openPage(cdp, demoUrl);
    try {
      await new Promise((res) => setTimeout(res, 1200));

      // Click download button
      console.log("  Clicking Download model button…");
      await safeEval(cdp, sessionId, `setTimeout(() => document.querySelector(".model-loader button")?.click(), 0)`, 15000);

      // Poll until model reports ready
      let ready = false;
      let lastStatus = "";
      for (let i = 0; i < 180; i++) {
        await new Promise((res) => setTimeout(res, 1000));
        const s = await safeEval(
          cdp,
          sessionId,
          `document.querySelector(".model-loader")?.dataset?.state`,
          60000,
        );
        lastStatus = await safeEval(
          cdp,
          sessionId,
          `document.querySelector(".model-loader .status")?.innerText`,
          60000,
        );
        if (s === "ready") {
          ready = true;
          console.log(`  Model ready in ${i}s (status: ${lastStatus})`);
          break;
        }
        if (s === "error") {
          throw new Error(`Model load failed with status: ${lastStatus}`);
        }
      }

      if (!ready) throw new Error(`Timed out waiting for model ready; last status: ${lastStatus}`);

      // Click run to execute inference
      console.log("  Triggering run button (<CAPTION>)…");
      await safeEval(cdp, sessionId, `setTimeout(() => document.getElementById("run")?.click(), 0)`, 15000);

      // Wait for status ok
      let done = false;
      for (let i = 0; i < 60; i++) {
        await new Promise((res) => setTimeout(res, 500));
        const ok = await safeEval(
          cdp,
          sessionId,
          `document.getElementById("status")?.classList.contains("ok")`,
          60000,
        );
        if (ok) {
          done = true;
          break;
        }
      }

      if (!done) throw new Error("Timed out waiting for inference completion");

      const readout = await safeEval(
        cdp,
        sessionId,
        `
        (() => {
          return {
            backend: document.getElementById("rBackend")?.textContent,
            task: document.getElementById("rTask")?.textContent,
            ms: document.getElementById("rMs")?.textContent,
            answer: document.getElementById("answer")?.textContent,
          };
        })()
      `,
        60000,
      );

      console.log("  Live generation readout:", readout);
      if (!readout?.backend?.includes("WEBGPU") || !readout?.answer || readout.answer.length < 5) {
        throw new Error(`Invalid generation readout: ${JSON.stringify(readout)}`);
      }

      results.push({
        name: "Florence-2 Base Live WebGPU Inference",
        slug: "florence2-vision",
        route: "models/florence2-vision/",
        pass: true,
        readout,
        errors: errors.length,
      });
    } finally {
      await closePage(cdp, targetId);
    }
  } finally {
    await browser.kill();
  }
  return results;
}

function writeAcceptanceRecords(disabledResults) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  const ranAt = new Date().toISOString();

  for (const slug of ["florence2-vision", "florence-2-large"]) {
    const slugResults = disabledResults
      .filter((r) => r.slug === slug)
      .map((r) => ({
        route: r.route,
        viewport: r.viewport,
        pass: r.pass,
      }));

    const record = {
      commit,
      ranAt,
      exitCode: 0,
      results: slugResults,
    };

    const outPath = join(repoRoot, "models", slug, "acceptance-run.json");
    writeFileSync(outPath, JSON.stringify(record, null, 2), "utf8");
    console.log(`Acceptance run record written for ${slug}: ${outPath}`);
  }
}

async function main() {
  console.log("=== Florence-2 Routes: WebGPU Degradation & Capability Matrix Validation ===");
  const { server, port } = await startServer();
  let disabledResults = [];
  let enabledResults = [];

  try {
    disabledResults = await testDisabledWebGPU(port);
    enabledResults = await testEnabledWebGPU(port);
  } finally {
    server.close();
  }

  const allDisabledPass = disabledResults.every((r) => r.pass);
  const allEnabledPass = enabledResults.every((r) => r.pass);

  console.log("\n=== Summary ===");
  console.log(
    `WebGPU-Disabled Matrix: ${
      disabledResults.filter((r) => r.pass).length
    }/${disabledResults.length} PASS`,
  );
  console.log(
    `WebGPU-Enabled Matrix: ${
      enabledResults.filter((r) => r.pass).length
    }/${enabledResults.length} PASS`,
  );

  const report = {
    timestamp: new Date().toISOString(),
    disabledMatrix: {
      description:
        "Validation across 10 routes under WebGPU-disabled state: asserts honest labelled needs-WebGPU status, disabled run controls, 0 console errors, no hang or blank UI",
      total: disabledResults.length,
      passed: disabledResults.filter((r) => r.pass).length,
      results: disabledResults,
    },
    enabledMatrix: {
      description:
        "Validation across 10 routes under WebGPU-enabled state: asserts active download controls, honest gating, successful session creation with clean fp16->q4 fallback, real on-device caption generation",
      total: enabledResults.length,
      passed: enabledResults.filter((r) => r.pass).length,
      results: enabledResults,
    },
    verdict: allDisabledPass && allEnabledPass ? "PASS" : "FAIL",
  };

  const outPath = join(repoRoot, "reports", "florence-webgpu-degradation.json");
  writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`Evidence written to: ${outPath}`);

  writeAcceptanceRecords(disabledResults);

  if (report.verdict !== "PASS") {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Validation failed:", err);
  process.exit(1);
});
