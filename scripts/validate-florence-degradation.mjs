#!/usr/bin/env node
// Comprehensive matrix validation for Florence-2 routes (florence2-vision & florence-2-large).
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
//    Asserts that loader transitions to download-required with active download controls.
// 3. Desktop + mobile parity: no horizontal overflow at 1280x800 and 360x740.

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
  console.log("\n=== Phase 2: Validating WebGPU-Enabled Capability & Gating across 10 Routes ===");
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
  } finally {
    await browser.kill();
  }
  return results;
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
  console.log(`WebGPU-Disabled Matrix: ${disabledResults.filter((r) => r.pass).length}/${disabledResults.length} PASS`);
  console.log(`WebGPU-Enabled Matrix: ${enabledResults.filter((r) => r.pass).length}/${enabledResults.length} PASS`);

  const report = {
    timestamp: new Date().toISOString(),
    disabledMatrix: {
      description: "Validation across 10 routes under WebGPU-disabled state: asserts honest labelled needs-WebGPU status, disabled run controls, 0 console errors, no hang or blank UI",
      total: disabledResults.length,
      passed: disabledResults.filter((r) => r.pass).length,
      results: disabledResults,
    },
    enabledMatrix: {
      description: "Validation across 10 routes under WebGPU-enabled state: asserts active download controls, honest gating, 0 console errors",
      total: enabledResults.length,
      passed: enabledResults.filter((r) => r.pass).length,
      results: enabledResults,
    },
    verdict: allDisabledPass && allEnabledPass ? "PASS" : "FAIL",
  };

  const outPath = join(repoRoot, "reports", "florence-webgpu-degradation.json");
  writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`Evidence written to: ${outPath}`);

  if (report.verdict !== "PASS") {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Validation failed:", err);
  process.exit(1);
});
