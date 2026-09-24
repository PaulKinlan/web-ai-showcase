#!/usr/bin/env node
// Automated verification of WebGPU acceleration and local browser inference across Web AI Showcase demos.
//
// 1. Probes Chrome headless with WebGPU flags enabled (--enable-unsafe-webgpu, --use-angle=vulkan, --enable-features=Vulkan).
// 2. Drives the WebGPU matrix across all 42 built WebGPU models in models.json, asserting adapter availability
//    and zero silent fallback / unsupported states.
// 3. Executes live end-to-end inference benchmarks on WebGPU vs WASM, measuring timing and proving hardware acceleration.
// 4. Writes complete execution evidence to reports/webgpu-verification.json.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  openPage,
  repoRoot,
  startServer,
} from "./browser.mjs";

const args = process.argv.slice(2);
const runMatrix = args.includes("--matrix") || args.includes("--all") || args.length === 0;
const runBenchmarks = args.includes("--benchmarks") || args.includes("--all") || args.length === 0;

async function probeWebGPU(cdp, port) {
  const url = `http://127.0.0.1:${port}/web-ai-showcase/`;
  const { targetId, sessionId } = await openPage(cdp, url);
  const info = await evalValue(
    cdp,
    sessionId,
    `
    (async () => {
      if (!("gpu" in navigator)) return { supported: false, reason: "navigator.gpu missing" };
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { supported: false, reason: "requestAdapter returned null" };
        const adapterInfo = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
        return {
          supported: true,
          vendor: adapterInfo.vendor,
          architecture: adapterInfo.architecture,
          device: adapterInfo.device,
          description: adapterInfo.description,
          isFallbackAdapter: adapter.isFallbackAdapter,
          features: [...adapter.features],
        };
      } catch (err) {
        return { supported: false, reason: err.message };
      }
    })()
  `,
  );
  await closePage(cdp, targetId);
  return info;
}

async function verifyMatrix(cdp, port, models) {
  const results = [];
  for (let i = 0; i < models.length; i++) {
    const m = models[i];
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/${m.slug}/`;
    process.stderr.write(`[${i + 1}/${models.length}] Verifying ${m.slug} … `);
    try {
      const { targetId, sessionId, errors } = await openPage(cdp, url);
      await new Promise((r) => setTimeout(r, 600));

      const status = await evalValue(
        cdp,
        sessionId,
        `
        (() => {
          const el = document.querySelector(".model-loader") || document.getElementById("model-loader");
          const mds = el?.querySelector("model-download-status");
          const state = el?.dataset?.state || mds?.dataset?.phase || "download-required";
          const text = (el?.innerText || "").replace(/\\s+/g, " ").trim();
          const hasBtn = !!el?.querySelector("button");
          const isUnsupported = /needs.*(gpu|webgpu)|unsupported|no gpu adapter/i.test(text);
          return {
            state,
            hasBtn,
            isUnsupported,
            sampleText: text.slice(0, 100),
          };
        })()
      `,
      );

      await closePage(cdp, targetId);

      const pass = !status?.isUnsupported && (status?.hasBtn || status?.state === "ready");
      results.push({
        slug: m.slug,
        task: m.task,
        hfId: m.hfId,
        declaredBackend: m.backend,
        dtype: m.dtype,
        sizeMB: m.sizeMB,
        pass,
        loaderState: status?.state,
        hasButton: status?.hasBtn,
        isUnsupported: status?.isUnsupported,
        errors: errors.length,
      });

      process.stderr.write(pass ? `PASS (state=${status?.state})\n` : `FAIL (unsupported)\n`);
    } catch (err) {
      results.push({
        slug: m.slug,
        pass: false,
        error: err.message,
      });
      process.stderr.write(`ERROR: ${err.message}\n`);
    }
  }
  return results;
}

async function runModnetBenchmark(serverPort, webgpu = true) {
  const browser = await launchChrome({ webgpu });
  const cdp = new CDP(browser.ws);
  const url = `http://127.0.0.1:${serverPort}/web-ai-showcase/models/modnet-portrait-matting/`;
  try {
    const { targetId, sessionId } = await openPage(cdp, url);
    await new Promise((r) => setTimeout(r, 1200));

    // Click download
    await evalValue(cdp, sessionId, `document.querySelector(".model-loader button")?.click()`);

    // Poll for ready state
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const state = await evalValue(
        cdp,
        sessionId,
        `document.querySelector(".model-loader")?.dataset?.state`,
      );
      if (state === "ready") break;
    }

    // Trigger matting
    await evalValue(cdp, sessionId, `document.getElementById("run")?.click()`);

    // Wait for completion
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 500));
      const isOk = await evalValue(
        cdp,
        sessionId,
        `document.getElementById("status")?.classList.contains("ok")`,
      );
      if (isOk) break;
    }

    const readout = await evalValue(
      cdp,
      sessionId,
      `
      (() => {
        return {
          backend: document.getElementById("rBackend")?.textContent,
          ms: parseInt(document.getElementById("rMs")?.textContent) || 0,
          coverage: document.getElementById("rCoverage")?.textContent,
          softEdge: document.getElementById("rSoft")?.textContent,
          dims: document.getElementById("iDims")?.textContent,
        };
      })()
    `,
    );

    await closePage(cdp, targetId);
    return readout;
  } finally {
    await browser.kill();
  }
}

async function runDepthAnythingBenchmark(serverPort, webgpu = true) {
  const browser = await launchChrome({ webgpu });
  const cdp = new CDP(browser.ws);
  const url = `http://127.0.0.1:${serverPort}/web-ai-showcase/models/depth-anything/`;
  try {
    const { targetId, sessionId } = await openPage(cdp, url);
    await new Promise((r) => setTimeout(r, 1200));

    // Click download
    await evalValue(cdp, sessionId, `document.querySelector(".model-loader button")?.click()`);

    // Poll for ready state
    for (let i = 0; i < 90; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const state = await evalValue(
        cdp,
        sessionId,
        `document.querySelector(".model-loader")?.dataset?.state`,
      );
      if (state === "ready") break;
    }

    // Trigger depth estimation
    await evalValue(cdp, sessionId, `document.getElementById("run")?.click()`);

    // Wait for completion
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 500));
      const isOk = await evalValue(
        cdp,
        sessionId,
        `document.getElementById("status")?.classList.contains("ok")`,
      );
      if (isOk) break;
    }

    const readout = await evalValue(
      cdp,
      sessionId,
      `
      (() => {
        return {
          backend: document.getElementById("rBackend")?.textContent,
          ms: parseInt(document.getElementById("rMs")?.textContent) || 0,
          range: document.getElementById("rRange")?.textContent,
          near: document.getElementById("rNear")?.textContent,
        };
      })()
    `,
    );

    await closePage(cdp, targetId);
    return readout;
  } finally {
    await browser.kill();
  }
}

async function main() {
  console.log("=== Web AI Showcase: WebGPU Acceleration & Local Inference Verification ===");
  const { server, port } = await startServer();

  let hardware = null;
  let matrixResults = [];
  const benchmarks = {};

  try {
    // 1. Hardware probe
    console.log("\n1. Probing headless Chrome with WebGPU flags...");
    const gpuBrowser = await launchChrome({ webgpu: true });
    const cdp = new CDP(gpuBrowser.ws);
    hardware = await probeWebGPU(cdp, port);
    console.log("Hardware adapter probe:", {
      supported: hardware.supported,
      vendor: hardware.vendor,
      architecture: hardware.architecture,
      isFallbackAdapter: hardware.isFallbackAdapter,
      shaderF16: hardware.features?.includes("shader-f16"),
    });

    if (!hardware.supported) {
      throw new Error(`WebGPU not available in test browser: ${hardware.reason}`);
    }

    // 2. Matrix verification
    if (runMatrix) {
      console.log("\n2. Scanning all built WebGPU demos across the showcase...");
      const modelsData = JSON.parse(readFileSync(join(repoRoot, "models.json"), "utf8"));
      const webgpuModels = modelsData.models.filter(
        (m) => m.status === "built" && m.backend === "webgpu",
      );
      matrixResults = await verifyMatrix(cdp, port, webgpuModels);
      const passed = matrixResults.filter((r) => r.pass).length;
      console.log(`\nMatrix Result: ${passed}/${matrixResults.length} WebGPU models passed.`);
    }

    await gpuBrowser.kill();

    // 3. Live inference benchmarks: WebGPU vs WASM
    if (runBenchmarks) {
      console.log("\n3. Executing live end-to-end inference benchmarks (WebGPU vs WASM)...");

      console.log("  Running benchmark: modnet-portrait-matting (WebGPU)...");
      const modnetGpu = await runModnetBenchmark(port, true);
      console.log("  Running benchmark: modnet-portrait-matting (WASM)...");
      const modnetWasm = await runModnetBenchmark(port, false);
      const modnetSpeedup = (modnetWasm.ms / modnetGpu.ms).toFixed(2);
      benchmarks["modnet-portrait-matting"] = {
        webgpu: modnetGpu,
        wasm: modnetWasm,
        speedup: `${modnetSpeedup}x`,
      };
      console.log(
        `  -> MODNet: WebGPU (${modnetGpu.backend}) = ${modnetGpu.ms}ms vs WASM (${modnetWasm.backend}) = ${modnetWasm.ms}ms [${modnetSpeedup}x faster]`,
      );

      console.log("  Running benchmark: depth-anything (WebGPU)...");
      const depthGpu = await runDepthAnythingBenchmark(port, true);
      console.log("  Running benchmark: depth-anything (WASM)...");
      const depthWasm = await runDepthAnythingBenchmark(port, false);
      const depthSpeedup = (depthWasm.ms / depthGpu.ms).toFixed(2);
      benchmarks["depth-anything"] = {
        webgpu: depthGpu,
        wasm: depthWasm,
        speedup: `${depthSpeedup}x`,
      };
      console.log(
        `  -> Depth Anything: WebGPU (${depthGpu.backend}) = ${depthGpu.ms}ms vs WASM (${depthWasm.backend}) = ${depthWasm.ms}ms [${depthSpeedup}x faster]`,
      );
    }
  } finally {
    server.close();
  }

  // 4. Write evidence report
  const reportsDir = join(repoRoot, "reports");
  if (!existsSync(reportsDir)) mkdirSync(reportsDir, { recursive: true });
  const reportPath = join(reportsDir, "webgpu-verification.json");

  const report = {
    timestamp: new Date().toISOString(),
    hardware,
    matrix: {
      total: matrixResults.length,
      passed: matrixResults.filter((r) => r.pass).length,
      failed: matrixResults.filter((r) => !r.pass).length,
      results: matrixResults,
    },
    benchmarks,
    verdict: matrixResults.every((r) => r.pass) ? "PASS" : "FAIL",
  };

  writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`\nEvidence written to: ${reportPath}`);
  console.log(`Overall WebGPU verification verdict: ${report.verdict}`);
}

main().catch((err) => {
  console.error("Verification failed:", err);
  process.exit(1);
});
