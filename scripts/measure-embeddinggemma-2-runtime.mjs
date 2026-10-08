#!/usr/bin/env node
// Dedicated single-route runtime measurement for models/embeddinggemma-2/basics/ (bead web-ai-showcase-ij4).
//
// Records the exact cdn.jsdelivr.net URLs fetched by models/embeddinggemma-2/basics/ at runtime,
// along with HTTP status, redirect info, immutability, and decoded-byte SHA-256 hashes, so they
// can be pinned in runtime-integrity.json with zero guessed URLs.
//
// Fail-closed by construction:
//   1. Single module-level error latch checked after EVERY await and once more after drain before success.
//   2. Cold isolated profile, desktop viewport (1280x800), WebGPU requested from launch.
//   3. WebGPU capability verified early in-page; fails closed if adapter is unavailable.
//   4. Loader driven to "ready" then REAL inference executed (verifying cosine score output).
//   5. Worker and service_worker targets paused on start, Network.enable issued BEFORE resume.
//      If Network.enable fails, worker is NEVER resumed and whole Chrome is killed immediately.
//   6. Zero jsDelivr requests = FAILURE / INCONCLUSIVE, never a pass.
//   7. Response bodies decoded according to format (base64 for WASM, UTF-8 for JS) and real bytes hashed.
//   8. Every 2xx jsDelivr response MUST have a valid 64-character lowercase hex SHA-256.
//   9. Compact machine-readable summary printed for independent external audit, exposing PID and profile.
//
// ============================================================================
// ENVIRONMENT & CLEANUP HONESTY STATEMENT (binding design requirement 10):
//
// 1. ONE VIEWPORT PROVES ONLY THAT ENVIRONMENT:
//    This tool measures desktop viewport only (1280x800) under headless Chrome with
//    Vulkan/WebGPU flags enabled. It proves ONLY that this specific environment
//    executes this route and fetches these exact runtime assets. It says NOTHING
//    about mobile viewports, mobile memory limits, mobile GPU adapters, or physical
//    GPU hardware execution (e.g. discrete Nvidia/AMD/Apple GPUs).
//
// 2. SIGKILL CLEANUP RESIDUAL:
//    Under SIGKILL (kill -9, Linux OOM killer, or uncatchable container eviction),
//    no JavaScript execution occurs: no `finally` blocks, no `process.on('exit')`
//    handlers, and no signal traps can run. As a result, Chrome processes and the
//    temporary profile directory (`userDataDir`) can be orphaned on disk in `tmpdir()`.
//    We do NOT claim this is solved or preventable in user-space JS.
//    Mitigations:
//    a) External hard bounds: callers/harnesses must run with an external timeout
//       and process supervisor that tracks and reaps children.
//    b) Caller-side independent audit: we log the Chrome PID and profile directory
//       immediately on startup and include them in the machine-readable summary,
//       so external supervisors can verify process termination and profile removal.
//    c) Stale-profile prune on subsequent runs: `scripts/browser.mjs` provides
//       `pruneStaleAcceptanceProfiles()`, which is called automatically by
//       `scripts/acceptance-run.mjs`. It scans `tmpdir()` for `webai-chrome-profile-*`
//       directories older than 15 minutes that are not held by active Chrome processes,
//       and removes them.
// ============================================================================

import { createHash } from "node:crypto";
import { existsSync, rmSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import {
  CDP,
  DESKTOP,
  closePage,
  launchChrome,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

export const TARGET_ROUTE = "models/embeddinggemma-2/basics/";

// ============================================================================
// 1. SINGLE ERROR LATCH
// ============================================================================

let errorLatch = null;

export function tripLatch(err) {
  if (!errorLatch) {
    errorLatch = err instanceof Error ? err : new Error(String(err?.message || err));
  }
}

export function checkLatch() {
  if (errorLatch) {
    throw errorLatch;
  }
}

export function getLatchError() {
  return errorLatch;
}

export function resetLatch() {
  errorLatch = null;
}

// ============================================================================
// 7 & 8. DECODED BYTE HASHING & VALIDATION
// ============================================================================

export function isJsDelivrUrl(url) {
  return typeof url === "string" && (url.startsWith("https://cdn.jsdelivr.net/") || url.includes("cdn.jsdelivr.net"));
}

export function isValidSha256Hex(str) {
  return typeof str === "string" && /^[0-9a-f]{64}$/.test(str);
}

export function decodeResponseBody(body, base64Encoded) {
  if (base64Encoded) {
    return Buffer.from(body, "base64");
  }
  return Buffer.from(body, "utf8");
}

export function computeSha256Hex(bufferOrBytes) {
  const hash = createHash("sha256").update(bufferOrBytes).digest("hex");
  if (!isValidSha256Hex(hash)) {
    throw new Error(`Computed hash is not a valid 64-hex string: "${hash}"`);
  }
  return hash;
}

// ============================================================================
// NETWORK DRAIN HELPER
// ============================================================================

export async function drainPhase({
  pendingAttaches,
  pendingBodies,
  getLastActivityTime,
  quietMs = 1000,
  maxMs = 5000,
  pollMs = 100,
  sleepFn = (ms) => new Promise((r) => setTimeout(r, ms)),
}) {
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    checkLatch();
    if (pendingAttaches.size > 0) {
      await Promise.allSettled([...pendingAttaches]);
      checkLatch();
    }
    if (pendingBodies.size > 0) {
      await Promise.allSettled([...pendingBodies]);
      checkLatch();
    }
    const quietTime = Date.now() - getLastActivityTime();
    if (quietTime >= quietMs && pendingAttaches.size === 0 && pendingBodies.size === 0) {
      break;
    }
    await sleepFn(pollMs);
    checkLatch();
  }
  if (pendingAttaches.size > 0) {
    await Promise.allSettled([...pendingAttaches]);
    checkLatch();
  }
  if (pendingBodies.size > 0) {
    await Promise.allSettled([...pendingBodies]);
    checkLatch();
  }
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

export async function runMeasurement({
  launchChromeFn = launchChrome,
  startServerFn = startServer,
  cdpClass = CDP,
  openPageFn = openPage,
  closePageFn = closePage,
  setViewportFn = setViewport,
  drainPhaseFn = drainPhase,
  computeSha256HexFn = computeSha256Hex,
  decodeResponseBodyFn = decodeResponseBody,
  loaderTimeoutMs = 300_000,
  inferenceTimeoutMs = 120_000,
  drainQuietMs = 1000,
  drainMaxMs = 5000,
  pollIntervalMs = 500,
  log = console.log,
  warn = console.warn,
  error = console.error,
} = {}) {
  resetLatch();
  const startTime = Date.now();

  let server = null;
  let chromeInstance = null;
  let chromeKilled = false;

  const killChrome = async () => {
    if (chromeKilled) return;
    chromeKilled = true;
    try {
      if (chromeInstance && typeof chromeInstance.kill === "function") {
        await chromeInstance.kill({ removeProfile: true });
      }
    } catch { /* ignore kill error */ }
    if (chromeInstance?.userDataDir) {
      try {
        if (existsSync(chromeInstance.userDataDir)) {
          rmSync(chromeInstance.userDataDir, { recursive: true, force: true });
        }
      } catch { /* ignore rm error */ }
    }
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  try {
    // Step 1: Start HTTP server
    log("[egemma2-capture] Starting local HTTP server...");
    const serverResult = await startServerFn();
    checkLatch();
    server = serverResult.server;
    const port = serverResult.port;
    const base = `http://127.0.0.1:${port}/web-ai-showcase/`;
    const targetUrl = `${base}${TARGET_ROUTE}`;

    // Step 2: Launch Chrome with WebGPU enabled
    log("[egemma2-capture] Launching headless Chrome (desktop, WebGPU enabled)...");
    const isolatedPrefix = `egemma2-capture-${process.pid}-${Date.now()}`;
    chromeInstance = await launchChromeFn({
      webgpu: true,
      profilePrefix: isolatedPrefix,
      resetProfile: true,
      removeProfileOnKill: true,
    });
    checkLatch();

    const chromePid = chromeInstance.proc?.pid ?? "unknown";
    const userDataDir = chromeInstance.userDataDir ?? "unknown";
    log(`[egemma2-capture] Chrome launched (PID: ${chromePid}, Profile: ${userDataDir})`);

    // Step 3: Connect CDP
    const cdp = chromeInstance.ws ? new cdpClass(chromeInstance.ws) : chromeInstance.cdp;
    checkLatch();

    const requestTrackers = new Map();
    const recordedAssets = [];
    const pendingAttaches = new Set();
    const pendingBodies = new Set();
    const unresumedTargets = new Set();
    let lastNetworkActivityTime = Date.now();

    const evaluate = async (sessionId, expression, timeout = 60000) => {
      checkLatch();
      const wrapped = `(async()=>{try{return (${expression});}catch(err){return {__err:String(err?.stack||err)};}})()`;
      try {
        const res = await cdp.send(
          "Runtime.evaluate",
          {
            expression: wrapped,
            awaitPromise: true,
            returnByValue: true,
          },
          sessionId,
          timeout,
        );
        checkLatch();
        return res?.result?.value;
      } catch (err) {
        tripLatch(err);
        await killChrome();
        throw err;
      }
    };

    // Step 4: Register CDP event listener
    cdp.on((msg) => {
      try {
        const method = msg.method;

        if (method === "Target.attachedToTarget") {
          const { sessionId, targetInfo, waitingForDebugger } = msg.params || {};
          const type = targetInfo?.type || "unknown";
          lastNetworkActivityTime = Date.now();

          if (waitingForDebugger) {
            unresumedTargets.add(sessionId);
          }

          // Requirement 4: worker AND service_worker attach, Network.enable BEFORE Runtime.runIfWaitingForDebugger.
          // Worker must NEVER be resumed before Network.enable succeeds. If enable fails, do not resume.
          const attachTask = (async () => {
            try {
              await cdp.send("Network.enable", {}, sessionId);
              if (waitingForDebugger) {
                await cdp.send("Runtime.runIfWaitingForDebugger", {}, sessionId);
                unresumedTargets.delete(sessionId);
              }
            } catch (err) {
              const attachErr = new Error(
                `Target attach/enable failed for ${type} (session: ${sessionId}): ${err.message}`,
              );
              tripLatch(attachErr);
              await killChrome();
            }
          })();

          pendingAttaches.add(attachTask);
          attachTask.finally(() => pendingAttaches.delete(attachTask));
          return;
        }

        if (method === "Target.detachedFromTarget") {
          const { sessionId } = msg.params || {};
          if (sessionId) {
            unresumedTargets.delete(sessionId);
          }
          return;
        }

        if (method === "Network.requestWillBeSent") {
          const sid = msg.sessionId || "root";
          const { requestId, request, redirectResponse } = msg.params || {};
          const url = request?.url || "";
          const key = `${sid}:${requestId}`;
          lastNetworkActivityTime = Date.now();

          if (redirectResponse) {
            const redirUrl = redirectResponse.url || "";
            const redirStatus = redirectResponse.status || 0;
            const redirHeaders = redirectResponse.headers || {};
            const redirCc = redirHeaders["cache-control"] || redirHeaders["Cache-Control"] || "";
            if (isJsDelivrUrl(redirUrl)) {
              recordedAssets.push({
                url: redirUrl,
                status: redirStatus,
                redirected: true,
                immutable: /immutable/i.test(redirCc),
                sha256: null,
                decodedBytes: 0,
              });
            }
          }

          requestTrackers.set(key, {
            url,
            sessionId: sid,
            requestId,
            redirected: Boolean(redirectResponse),
            status: null,
            headers: null,
            immutable: false,
          });
          return;
        }

        if (method === "Network.responseReceived") {
          const sid = msg.sessionId || "root";
          const { requestId, response } = msg.params || {};
          const key = `${sid}:${requestId}`;
          lastNetworkActivityTime = Date.now();

          const tracker = requestTrackers.get(key);
          const status = response?.status || 0;
          const headers = response?.headers || {};
          const cc = headers["cache-control"] || headers["Cache-Control"] || "";
          const immutable = /immutable/i.test(cc);
          const url = response?.url || tracker?.url || "";

          if (tracker) {
            tracker.status = status;
            tracker.headers = headers;
            tracker.immutable = immutable;
            tracker.url = url;
          } else {
            requestTrackers.set(key, {
              url,
              sessionId: sid,
              requestId,
              redirected: false,
              status,
              headers,
              immutable,
            });
          }
          return;
        }

        if (method === "Network.loadingFinished") {
          const sid = msg.sessionId || "root";
          const { requestId } = msg.params || {};
          const key = `${sid}:${requestId}`;
          lastNetworkActivityTime = Date.now();

          const tracker = requestTrackers.get(key);
          if (!tracker) return;

          const url = tracker.url;
          const status = tracker.status;
          const immutable = tracker.immutable;
          const redirected = tracker.redirected;

          if (isJsDelivrUrl(url)) {
            const record = {
              url,
              status,
              redirected,
              immutable,
              sha256: null,
              decodedBytes: 0,
            };
            recordedAssets.push(record);

            if (status >= 200 && status < 300) {
              const bodyTask = (async () => {
                try {
                  const targetSid = sid === "root" ? undefined : sid;
                  const res = await cdp.send("Network.getResponseBody", { requestId }, targetSid);
                  const { body, base64Encoded } = res || {};
                  if (typeof body !== "string") {
                    throw new Error(`getResponseBody for ${url} returned invalid body: ${typeof body}`);
                  }
                  const decodedBytes = decodeResponseBodyFn(body, Boolean(base64Encoded));
                  const sha256 = computeSha256HexFn(decodedBytes);
                  if (!isValidSha256Hex(sha256)) {
                    throw new Error(`Invalid sha256 hex generated for ${url}: ${sha256}`);
                  }
                  record.sha256 = sha256;
                  record.decodedBytes = decodedBytes.length;
                } catch (err) {
                  tripLatch(new Error(`Failed to retrieve/hash response body for ${url}: ${err.message}`));
                  await killChrome();
                }
              })();

              pendingBodies.add(bodyTask);
              bodyTask.finally(() => pendingBodies.delete(bodyTask));
            }
          }
          return;
        }

        if (method === "Network.loadingFailed") {
          const sid = msg.sessionId || "root";
          const { requestId, errorText, canceled } = msg.params || {};
          const key = `${sid}:${requestId}`;
          lastNetworkActivityTime = Date.now();

          if (!canceled) {
            const tracker = requestTrackers.get(key);
            if (tracker && isJsDelivrUrl(tracker.url)) {
              tripLatch(new Error(`jsDelivr request failed for ${tracker.url}: ${errorText}`));
              void killChrome();
            }
          }
          return;
        }

        if (method === "Inspector.targetCrashed") {
          tripLatch(new Error("Target crashed during measurement"));
          void killChrome();
          return;
        }
      } catch (err) {
        tripLatch(err);
        void killChrome();
      }
    });

    // Step 5: Enable auto-attach with waitForDebuggerOnStart
    await cdp.send("Target.setAutoAttach", {
      autoAttach: true,
      waitForDebuggerOnStart: true,
      flatten: true,
    });
    checkLatch();

    await cdp.send("Target.setDiscoverTargets", { discover: true });
    checkLatch();

    // Step 6: Open the route
    log(`[egemma2-capture] Navigating to ${targetUrl}...`);
    const page = await openPageFn(cdp, targetUrl);
    checkLatch();

    await setViewportFn(cdp, page.sessionId, DESKTOP);
    checkLatch();

    // Step 7: WebGPU capability check in page
    log("[egemma2-capture] Checking WebGPU capability in browser context...");
    const gpuCheck = await evaluate(
      page.sessionId,
      `(async () => {
        if (!navigator.gpu) return { ok: false, reason: "navigator.gpu is undefined" };
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) return { ok: false, reason: "navigator.gpu.requestAdapter() returned null" };
          return { ok: true };
        } catch (err) {
          return { ok: false, reason: String(err?.message || err) };
        }
      })()`,
    );
    checkLatch();

    if (!gpuCheck || !gpuCheck.ok) {
      const reason = gpuCheck?.reason || "WebGPU requestAdapter failed";
      tripLatch(new Error(`WebGPU capability check failed: ${reason}. Cannot measure WebGPU-required route without WebGPU.`));
      await killChrome();
      checkLatch();
    }
    log("[egemma2-capture] WebGPU confirmed available.");

    // Step 8: Drive model loader to ready
    log("[egemma2-capture] Driving model loader to 'ready'...");
    let loaderReady = false;
    const loaderDeadline = Date.now() + loaderTimeoutMs;

    while (Date.now() < loaderDeadline) {
      checkLatch();
      const stateInfo = await evaluate(
        page.sessionId,
        `(() => {
          const loader = document.querySelector('.model-loader');
          if (!loader) return { mounted: false };
          const state = loader.dataset.state || "";
          const errText = loader.querySelector('.status.err')?.textContent || "";
          const btn = [...loader.querySelectorAll('button')].find(b => 
            /Download|Retry|Re-download|Continue|Retry local check/i.test(b.textContent) && !b.disabled
          );
          return { mounted: true, state, errText, canClick: Boolean(btn) };
        })()`,
      );
      checkLatch();

      if (!stateInfo || !stateInfo.mounted) {
        await sleep(pollIntervalMs);
        checkLatch();
        continue;
      }

      if (stateInfo.state === "ready") {
        loaderReady = true;
        log("[egemma2-capture] Model loader reached 'ready'.");
        break;
      }

      if (stateInfo.state === "unsupported" || stateInfo.state === "error" || stateInfo.state === "check-timeout") {
        tripLatch(new Error(
          `Model loader entered terminal failure state "${stateInfo.state}"` +
            (stateInfo.errText ? `: ${stateInfo.errText}` : ""),
        ));
        await killChrome();
        checkLatch();
      }

      if (stateInfo.canClick) {
        log(`[egemma2-capture] Clicking loader action button in state "${stateInfo.state}"...`);
        await evaluate(
          page.sessionId,
          `(() => {
            const btn = [...document.querySelectorAll('.model-loader button')].find(b => 
              /Download|Retry|Re-download|Continue|Retry local check/i.test(b.textContent) && !b.disabled
            );
            if (btn) btn.click();
          })()`,
        );
        checkLatch();
      }

      await sleep(pollIntervalMs);
      checkLatch();
    }

    if (!loaderReady) {
      tripLatch(new Error(`Timed out after ${loaderTimeoutMs}ms waiting for model loader to reach 'ready'.`));
      await killChrome();
      checkLatch();
    }

    // Step 9: Drive REAL inference
    log("[egemma2-capture] Driving real inference on route...");
    const runDeadline = Date.now() + 15_000;
    let runBtnReady = false;
    while (Date.now() < runDeadline) {
      checkLatch();
      const btnReady = await evaluate(
        page.sessionId,
        `(() => {
          const btn = document.getElementById('run');
          return Boolean(btn && !btn.disabled);
        })()`,
      );
      checkLatch();
      if (btnReady) {
        runBtnReady = true;
        break;
      }
      await sleep(200);
      checkLatch();
    }

    if (!runBtnReady) {
      tripLatch(new Error("Run button '#run' was not enabled after loader reached 'ready'."));
      await killChrome();
      checkLatch();
    }

    await evaluate(page.sessionId, "document.getElementById('run').click()");
    checkLatch();
    log("[egemma2-capture] Clicked '#run'. Waiting for inference completion...");

    const infDeadline = Date.now() + inferenceTimeoutMs;
    let inferenceCompleted = false;
    let inferenceScore = null;

    while (Date.now() < infDeadline) {
      checkLatch();
      const infStatus = await evaluate(
        page.sessionId,
        `(() => {
          const runBtn = document.getElementById('run');
          const status = document.getElementById('status');
          const score = document.getElementById('score');
          const prefixTable = document.getElementById('prefixTable');
          
          const isErr = Boolean(status && status.classList.contains('err') && !status.hidden);
          const errText = isErr ? status.textContent : "";
          const scoreText = score ? score.textContent.trim() : "";
          const hasValidScore = scoreText !== "" && scoreText !== "–" && scoreText !== "-";
          const hasTable = Boolean(prefixTable && prefixTable.querySelector('table'));
          const isRunning = Boolean(runBtn && runBtn.disabled);
          
          return {
            isErr,
            errText,
            scoreText,
            hasValidScore,
            hasTable,
            isRunning,
          };
        })()`,
      );
      checkLatch();

      if (infStatus?.isErr) {
        tripLatch(new Error(`Inference failed on page: ${infStatus.errText || "unknown error"}`));
        await killChrome();
        checkLatch();
      }

      if (infStatus?.hasValidScore && !infStatus.isRunning) {
        inferenceCompleted = true;
        inferenceScore = infStatus.scoreText;
        log(`[egemma2-capture] Inference succeeded! Cosine score: ${inferenceScore}`);
        break;
      }

      await sleep(pollIntervalMs);
      checkLatch();
    }

    if (!inferenceCompleted) {
      tripLatch(new Error(`Timed out after ${inferenceTimeoutMs}ms waiting for inference to complete.`));
      await killChrome();
      checkLatch();
    }

    // Step 10: Bounded network quiet / drain phase
    log("[egemma2-capture] Entering bounded network-quiet/drain phase...");
    await drainPhaseFn({
      pendingAttaches,
      pendingBodies,
      getLastActivityTime: () => lastNetworkActivityTime,
      quietMs: drainQuietMs,
      maxMs: drainMaxMs,
      pollMs: 100,
      sleepFn: sleep,
    });
    checkLatch();
    log("[egemma2-capture] Drain phase completed.");

    // Step 11: Fail-closed checks on captured assets
    const jsDelivrAssets = recordedAssets.filter((a) => isJsDelivrUrl(a.url));
    log(`[egemma2-capture] Total jsDelivr requests observed: ${jsDelivrAssets.length}`);

    // Requirement 6: NO GREEN ON ZERO REQUESTS
    if (jsDelivrAssets.length === 0) {
      tripLatch(new Error(
        `Failure/Inconclusive: zero jsDelivr requests were observed during measurement of ${TARGET_ROUTE}.`,
      ));
      await killChrome();
      checkLatch();
    }

    // Requirement 8: Every 2xx jsDelivr response MUST have a valid 64-hex SHA-256
    for (const asset of jsDelivrAssets) {
      if (asset.status >= 200 && asset.status < 300) {
        if (!asset.sha256) {
          tripLatch(new Error(
            `Successful jsDelivr request (${asset.status}) missing SHA-256: ${asset.url}`,
          ));
          await killChrome();
          checkLatch();
        }
        if (!isValidSha256Hex(asset.sha256)) {
          tripLatch(new Error(
            `SHA-256 for ${asset.url} is not a valid 64-character lowercase hex string: "${asset.sha256}"`,
          ));
          await killChrome();
          checkLatch();
        }
      }
    }

    // Ensure no target was left paused
    if (unresumedTargets.size > 0) {
      tripLatch(new Error(
        `Paused target(s) detected that were never resumed: ${[...unresumedTargets].join(", ")}`,
      ));
      await killChrome();
      checkLatch();
    }

    // Final latch check BEFORE reporting success (Requirement 1)
    checkLatch();

    return {
      status: "passed",
      route: TARGET_ROUTE,
      viewport: { width: DESKTOP.width, height: DESKTOP.height },
      chrome: {
        pid: chromePid,
        userDataDir,
      },
      webgpuAvailable: true,
      loaderState: "ready",
      inferenceResult: {
        completed: true,
        score: inferenceScore,
      },
      assets: recordedAssets,
      totalObservedAssets: recordedAssets.length,
      durationMs: Date.now() - startTime,
    };
  } finally {
    await killChrome();
    try {
      if (server?.close) {
        await new Promise((resolve) => server.close(resolve));
      }
    } catch { /* ignore server close error */ }
  }
}

// ============================================================================
// CLI ENTRY POINT
// ============================================================================

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const jsonOut = args.includes("--json");
  const outIdx = args.indexOf("--output");
  const outputPath = outIdx !== -1 && args[outIdx + 1] ? args[outIdx + 1] : null;

  try {
    const summary = await runMeasurement({
      log: jsonOut ? () => {} : console.log,
      warn: console.warn,
      error: console.error,
    });
    if (outputPath) {
      writeFileSync(outputPath, JSON.stringify(summary, null, 2) + "\n", "utf8");
    }
    if (jsonOut) {
      console.log(JSON.stringify(summary, null, 2));
    } else {
      console.log("\n=== MEASUREMENT SUMMARY ===");
      console.log(`Route: ${summary.route}`);
      console.log(`Status: ${summary.status}`);
      console.log(`Chrome PID: ${summary.chrome.pid}`);
      console.log(`Profile: ${summary.chrome.userDataDir}`);
      console.log(`Inference Score: ${summary.inferenceResult?.score}`);
      console.log(`Observed jsDelivr Assets: ${summary.assets.length}`);
      for (const a of summary.assets) {
        console.log(`  - [${a.status}] ${a.url} (${a.decodedBytes} bytes, SHA-256: ${a.sha256})`);
      }
      console.log(JSON.stringify(summary, null, 2));
    }
    process.exit(0);
  } catch (err) {
    const errMessage = err?.message || String(err);
    if (jsonOut) {
      console.log(JSON.stringify({
        status: "failed",
        route: TARGET_ROUTE,
        error: errMessage,
      }, null, 2));
    } else {
      console.error(`\n[egemma2-capture] FATAL: ${errMessage}`);
    }
    process.exit(1);
  }
}
