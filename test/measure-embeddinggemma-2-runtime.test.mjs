// Dedicated offline runner-path tests for scripts/measure-embeddinggemma-2-runtime.mjs.
//
// These tests exercise the REAL run path (runMeasurement) rather than reconstructed loops,
// without launching a real browser or network (offline, fast, deterministic).
//
// Coverage:
//   1. Zero-request result failing (NO GREEN ON ZERO REQUESTS).
//   2. Attach/enable failure failing non-zero rather than exiting 0.
//   3. Worker is NEVER resumed before Network.enable succeeds, and never resumed if enable fails.
//   4. Error latch checked after the final drain phase so a LATE failure cannot produce success.
//   5. Decoded hashing for base64 WASM and multi-byte UTF-8 JS.
//   6. Genuine 64-hex-character validation (not merely length 64).
//   7. WebGPU capability failure failing closed immediately.
//   8. Independent auditability: summary exposes Chrome PID and userDataDir.

import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  computeSha256Hex,
  decodeResponseBody,
  drainPhase,
  getLatchError,
  isValidSha256Hex,
  resetLatch,
  runMeasurement,
  tripLatch,
} from "../scripts/measure-embeddinggemma-2-runtime.mjs";

/**
 * Mock CDP client that simulates the Chrome DevTools Protocol in-memory.
 * Does NOT spawn or connect to any browser process.
 */
class MockCDPClient {
  constructor() {
    this.listeners = [];
    this.sentCalls = [];
    this.responseBodies = new Map(); // requestId -> { body, base64Encoded }
    this.queuedEvents = [];
    this.evalHandler = null;
    this.enableNetworkReject = false;
    this.onNetworkEnable = null;
    this.customSendHandler = null;
  }

  on(listener) {
    this.listeners.push(listener);
  }

  emit(msg) {
    for (const l of this.listeners) {
      l(msg);
    }
  }

  queueEvent(event) {
    this.queuedEvents.push(event);
  }

  async send(method, params = {}, sessionId, timeout) {
    this.sentCalls.push({ method, params, sessionId, timeout });

    if (this.customSendHandler) {
      const customRes = await this.customSendHandler(method, params, sessionId);
      if (customRes !== undefined) return customRes;
    }

    if (method === "Target.setAutoAttach") {
      // Emit any queued events after listeners are registered
      for (const ev of this.queuedEvents) {
        this.emit(ev);
      }
      return {};
    }

    if (method === "Network.enable") {
      if (this.enableNetworkReject) {
        throw new Error("Simulated Network.enable failure on target");
      }
      if (this.onNetworkEnable) {
        this.onNetworkEnable(sessionId);
      }
      return {};
    }

    if (method === "Runtime.runIfWaitingForDebugger") {
      return {};
    }

    if (method === "Target.setDiscoverTargets") {
      return {};
    }

    if (method === "Network.getResponseBody") {
      const resp = this.responseBodies.get(params.requestId);
      if (!resp) throw new Error(`Mock: No response body registered for requestId "${params.requestId}"`);
      return resp;
    }

    if (method === "Runtime.evaluate") {
      if (this.evalHandler) {
        const custom = await this.evalHandler(params.expression, sessionId);
        if (custom !== undefined) return { result: { value: custom } };
      }

      const expr = params.expression || "";
      // WebGPU capability gate
      if (expr.includes("navigator.gpu")) {
        return { result: { value: { ok: true } } };
      }
      // Loader state gate
      if (expr.includes(".model-loader") && expr.includes("dataset.state")) {
        return { result: { value: { mounted: true, state: "ready", errText: "", canClick: false } } };
      }
      // Run button readiness
      if (expr.includes("document.getElementById('run')") && expr.includes("!btn.disabled")) {
        return { result: { value: true } };
      }
      // Run button click
      if (expr.includes("document.getElementById('run').click()")) {
        return { result: { value: null } };
      }
      // Inference progress & cosine score
      if (expr.includes("hasValidScore") && expr.includes("prefixTable")) {
        return {
          result: {
            value: {
              isErr: false,
              errText: "",
              scoreText: "0.842",
              hasValidScore: true,
              hasTable: true,
              isRunning: false,
            },
          },
        };
      }
      return { result: { value: null } };
    }

    return {};
  }
}

/**
 * Build standard mock runner options.
 * Avoids any launch/connect calls to keep suite completely browser-free.
 */
function createMockHarness(customCdp = null) {
  const cdp = customCdp || new MockCDPClient();
  let chromeKilled = false;

  const mockLauncher = async () => ({
    proc: { pid: 12345 },
    userDataDir: "/tmp/webai-chrome-profile-test-12345",
    cdp,
    kill: async () => {
      chromeKilled = true;
    },
  });

  const mockServerStarter = async () => ({
    server: { close: (cb) => cb?.() },
    port: 8765,
  });

  const mockPageOpener = async () => ({
    targetId: "mock-page-target",
    sessionId: "mock-page-session",
    errors: [],
  });

  const mockViewportSetter = async () => {};

  return {
    cdp,
    getChromeKilled: () => chromeKilled,
    options: {
      launchChromeFn: mockLauncher,
      startServerFn: mockServerStarter,
      cdpClass: function () {
        return cdp;
      },
      openPageFn: mockPageOpener,
      setViewportFn: mockViewportSetter,
      loaderTimeoutMs: 500,
      inferenceTimeoutMs: 500,
      drainQuietMs: 50,
      drainMaxMs: 150,
      pollIntervalMs: 10,
      log: () => {},
      warn: () => {},
      error: () => {},
    },
  };
}

// ============================================================================
// TEST 1: ZERO-REQUEST RESULT FAILING (Requirement 6)
// ============================================================================
test("runner fails closed when zero jsDelivr requests are observed (NO GREEN ON ZERO REQUESTS)", async () => {
  // Falsifiability: If the `if (jsDelivrAssets.length === 0)` guard in runMeasurement is omitted,
  // this test will FAIL because runMeasurement would resolve with status: 'passed'.
  resetLatch();
  const harness = createMockHarness();

  // Emits NO jsDelivr requests — only a localhost app-shell asset when auto-attach happens
  harness.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "local-1",
      request: { url: "http://127.0.0.1:8765/web-ai-showcase/models/embeddinggemma-2/basics/index.html" },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "local-1",
      response: {
        url: "http://127.0.0.1:8765/web-ai-showcase/models/embeddinggemma-2/basics/index.html",
        status: 200,
        headers: {},
      },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "local-1" },
  });

  await assert.rejects(
    runMeasurement(harness.options),
    /Failure\/Inconclusive: zero jsDelivr requests were observed/,
    "Expected runner to reject when zero jsDelivr requests are observed",
  );
  assert.equal(harness.getChromeKilled(), true, "Expected Chrome to be killed when latch tripped");
});

// ============================================================================
// TEST 2: ATTACH / ENABLE FAILURE FAILING NON-ZERO (Requirements 1, 4, 5)
// ============================================================================
test("attach/enable failure trips latch, kills Chrome, and causes non-zero failure", async () => {
  // Falsifiability: If attach failure is caught and ignored (as the shared driver did),
  // this test will FAIL because runMeasurement would not reject and Chrome would not be killed.
  resetLatch();
  const harness = createMockHarness();
  harness.cdp.enableNetworkReject = true; // causes Network.enable to reject on worker

  // Queue worker attach event with waitingForDebugger: true
  harness.cdp.queueEvent({
    method: "Target.attachedToTarget",
    params: {
      sessionId: "worker-session-err",
      targetInfo: { type: "worker", url: "http://127.0.0.1:8765/web-ai-showcase/models/embeddinggemma-2/worker.js" },
      waitingForDebugger: true,
    },
  });

  await assert.rejects(
    runMeasurement(harness.options),
    /Target attach\/enable failed for worker/,
    "Expected runner to reject when Network.enable fails on worker attach",
  );
  assert.equal(harness.getChromeKilled(), true, "Expected Chrome to be killed on attach error");
  assert.ok(getLatchError(), "Expected module-level error latch to be set");
});

// ============================================================================
// TEST 3: WORKER IS NEVER RESUMED BEFORE ENABLE (Requirement 4)
// ============================================================================
test("worker is never resumed before Network.enable succeeds, and never resumed if enable fails", async () => {
  // Falsifiability: If runIfWaitingForDebugger is called before Network.enable or called
  // in a finally block when enable fails, this test will FAIL.

  // Sub-case 3a: Success path — Network.enable must precede Runtime.runIfWaitingForDebugger
  {
    resetLatch();
    const harness = createMockHarness();
    harness.cdp.responseBodies.set("js-req-1", {
      body: "console.log('test-js');",
      base64Encoded: false,
    });

    // Queue worker attach
    harness.cdp.queueEvent({
      method: "Target.attachedToTarget",
      params: {
        sessionId: "worker-subcase-3a",
        targetInfo: { type: "worker", url: "http://127.0.0.1:8765/web-ai-showcase/models/embeddinggemma-2/worker.js" },
        waitingForDebugger: true,
      },
    });

    // When worker Network.enable succeeds, emit worker requests
    harness.cdp.onNetworkEnable = (sid) => {
      if (sid === "worker-subcase-3a") {
        harness.cdp.emit({
          method: "Network.requestWillBeSent",
          sessionId: "worker-subcase-3a",
          params: {
            requestId: "js-req-1",
            request: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1" },
          },
        });
        harness.cdp.emit({
          method: "Network.responseReceived",
          sessionId: "worker-subcase-3a",
          params: {
            requestId: "js-req-1",
            response: {
              url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
              status: 200,
              headers: { "cache-control": "public, max-age=31536000, immutable" },
            },
          },
        });
        harness.cdp.emit({
          method: "Network.loadingFinished",
          sessionId: "worker-subcase-3a",
          params: { requestId: "js-req-1" },
        });
      }
    };

    const summary = await runMeasurement(harness.options);
    assert.equal(summary.status, "passed");

    // Verify call order on worker session
    const workerCalls = harness.cdp.sentCalls.filter((c) => c.sessionId === "worker-subcase-3a");
    const enableIdx = workerCalls.findIndex((c) => c.method === "Network.enable");
    const resumeIdx = workerCalls.findIndex((c) => c.method === "Runtime.runIfWaitingForDebugger");

    assert.ok(enableIdx !== -1, "Expected Network.enable on worker");
    assert.ok(resumeIdx !== -1, "Expected Runtime.runIfWaitingForDebugger on worker");
    assert.ok(enableIdx < resumeIdx, `Network.enable (${enableIdx}) must precede runIfWaitingForDebugger (${resumeIdx})`);
  }

  // Sub-case 3b: Failure path — worker must NEVER be resumed if enable fails
  {
    resetLatch();
    const harness = createMockHarness();
    harness.cdp.enableNetworkReject = true;

    harness.cdp.queueEvent({
      method: "Target.attachedToTarget",
      params: {
        sessionId: "worker-subcase-3b",
        targetInfo: { type: "worker", url: "http://127.0.0.1:8765/web-ai-showcase/models/embeddinggemma-2/worker.js" },
        waitingForDebugger: true,
      },
    });

    await assert.rejects(runMeasurement(harness.options));

    const workerResumed = harness.cdp.sentCalls.some(
      (c) => c.sessionId === "worker-subcase-3b" && c.method === "Runtime.runIfWaitingForDebugger",
    );
    assert.equal(workerResumed, false, "Worker must NEVER be resumed if Network.enable fails");
  }
});

// ============================================================================
// TEST 4: ERROR LATCH CHECKED AFTER FINAL DRAIN PHASE (Requirement 1)
// ============================================================================
test("error latch is checked after the final drain phase so a late failure cannot produce success", async () => {
  // Falsifiability: If checkLatch() after drainPhase() is removed, this test will FAIL
  // because runMeasurement would return status: 'passed' instead of rejecting.
  resetLatch();
  const harness = createMockHarness();

  harness.cdp.responseBodies.set("late-req", {
    body: "console.log('late-asset');",
    base64Encoded: false,
  });

  harness.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "late-req",
      request: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1" },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "late-req",
      response: {
        url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
        status: 200,
        headers: { "cache-control": "immutable" },
      },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "late-req" },
  });

  // Inject a late latch trip right when drain completes
  harness.options.drainPhaseFn = async (params) => {
    // drain finishes cleanly
    await drainPhase(params);
    // trip latch late — immediately after drain finishes
    tripLatch(new Error("LATE FAILURE: simulated asynchronous failure after drain phase"));
  };

  await assert.rejects(
    runMeasurement(harness.options),
    /LATE FAILURE: simulated asynchronous failure after drain phase/,
    "Expected runMeasurement to fail because the error latch is checked after the final drain phase",
  );
});

// ============================================================================
// TEST 5: DECODED HASHING FOR BASE64 WASM & MULTI-BYTE UTF-8 JS (Requirement 7)
// ============================================================================
test("decoded hashing decodes base64 WASM and multi-byte UTF-8 JS before computing SHA-256", async () => {
  // Falsifiability: If decodeResponseBody treated base64 as raw ASCII text, or failed
  // to decode UTF-8 into raw bytes, this test will FAIL due to hash mismatch.

  // 5a: WASM binary hashing (CDP returns base64 string)
  const wasmBinary = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]); // 8-byte WASM header
  const wasmBase64 = wasmBinary.toString("base64"); // "AGFzbQE="
  const decodedWasm = decodeResponseBody(wasmBase64, true);
  const wasmHash = computeSha256Hex(decodedWasm);

  const expectedWasmHash = createHash("sha256").update(wasmBinary).digest("hex");
  const badAsciiHash = createHash("sha256").update(Buffer.from(wasmBase64, "utf8")).digest("hex");

  assert.equal(decodedWasm.length, 8, "Expected 8 binary bytes for decoded WASM");
  assert.equal(wasmHash, expectedWasmHash, "Hash must match real decoded binary bytes");
  assert.notEqual(wasmHash, badAsciiHash, "Hash must NOT match base64 ASCII character bytes");

  // 5b: Multi-byte UTF-8 JS hashing (CDP returns UTF-8 text)
  const utf8JsText = "// Multi-byte test: 🚀 ✨ 漢字 é ñ\nconsole.log('hello');";
  const decodedJs = decodeResponseBody(utf8JsText, false);
  const jsHash = computeSha256Hex(decodedJs);

  const expectedJsBytes = Buffer.from(utf8JsText, "utf8");
  const expectedJsHash = createHash("sha256").update(expectedJsBytes).digest("hex");

  assert.equal(decodedJs.length, expectedJsBytes.length, "Decoded length must match UTF-8 byte count");
  assert.ok(decodedJs.length > utf8JsText.length, "UTF-8 byte count must exceed character length for multi-byte text");
  assert.equal(jsHash, expectedJsHash, "Hash must match UTF-8 byte representation");

  // 5c: Exercise real runner path with both WASM and JS assets
  resetLatch();
  const harness = createMockHarness();

  harness.cdp.responseBodies.set("wasm-req", {
    body: wasmBase64,
    base64Encoded: true,
  });
  harness.cdp.responseBodies.set("js-req", {
    body: utf8JsText,
    base64Encoded: false,
  });

  // Queue WASM asset
  harness.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "wasm-req",
      request: { url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.wasm" },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "wasm-req",
      response: {
        url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.wasm",
        status: 200,
        headers: { "cache-control": "immutable" },
      },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "wasm-req" },
  });

  // Queue JS asset
  harness.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "js-req",
      request: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1" },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "js-req",
      response: {
        url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
        status: 200,
        headers: { "cache-control": "immutable" },
      },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "js-req" },
  });

  const summary = await runMeasurement(harness.options);
  assert.equal(summary.status, "passed");

  const wasmAsset = summary.assets.find((a) => a.url.includes(".wasm"));
  const jsAsset = summary.assets.find((a) => a.url.includes("transformers"));

  assert.ok(wasmAsset, "Expected WASM asset in summary");
  assert.equal(wasmAsset.decodedBytes, 8, "WASM decodedBytes must be 8");
  assert.equal(wasmAsset.sha256, expectedWasmHash, "WASM SHA-256 must match binary decoded bytes");

  assert.ok(jsAsset, "Expected JS asset in summary");
  assert.equal(jsAsset.decodedBytes, expectedJsBytes.length, "JS decodedBytes must match UTF-8 byte count");
  assert.equal(jsAsset.sha256, expectedJsHash, "JS SHA-256 must match UTF-8 bytes");
});

// ============================================================================
// TEST 6: HEX-CHARACTER VALIDATION NOT JUST LENGTH (Requirement 8)
// ============================================================================
test("hex-character validation requires genuine 64 hex characters, not merely length 64", async () => {
  // Falsifiability: If validation only checked `str.length === 64`, "g".repeat(64) would pass.
  const validHex = "8d6716d9086f57c30a4bf367dba61b887593573c770c454465e8019b2703e743";
  const nonHexLetters = "g".repeat(64);
  const spaces = " ".repeat(64);
  const tooShort = "a".repeat(63);
  const tooLong = "a".repeat(65);
  const upperHex = validHex.toUpperCase();

  assert.equal(isValidSha256Hex(validHex), true, "Valid lowercase 64-hex must pass");
  assert.equal(isValidSha256Hex(nonHexLetters), false, "64 non-hex letters ('g') must FAIL");
  assert.equal(isValidSha256Hex(spaces), false, "64 spaces must FAIL");
  assert.equal(isValidSha256Hex(tooShort), false, "63 characters must FAIL");
  assert.equal(isValidSha256Hex(tooLong), false, "65 characters must FAIL");
  assert.equal(isValidSha256Hex(upperHex), false, "Uppercase hex must FAIL (must be lowercase)");

  // Runner path test: a 64-character non-hex hash (e.g. 'g'.repeat(64)) causes failure
  // Falsifiability: If validation only checked hash.length === 64, 'g'.repeat(64) would pass.
  resetLatch();
  const harnessHex = createMockHarness();

  harnessHex.cdp.responseBodies.set("non-hex-hash-req", {
    body: "console.log('test');",
    base64Encoded: false,
  });

  harnessHex.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "non-hex-hash-req",
      request: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1" },
    },
  });
  harnessHex.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "non-hex-hash-req",
      response: {
        url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
        status: 200,
        headers: { "cache-control": "immutable" },
      },
    },
  });
  harnessHex.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "non-hex-hash-req" },
  });

  // Inject a 64-char non-hex hash generator
  harnessHex.options.computeSha256HexFn = () => "g".repeat(64);

  await assert.rejects(
    runMeasurement(harnessHex.options),
    /Invalid sha256 hex generated|not a valid 64-character lowercase hex string/,
    "Expected runner to reject when SHA-256 is 64 characters but non-hex",
  );
});

// ============================================================================
// TEST 7: WEBGPU CAPABILITY CHECK FAILS CLOSED (Requirement 2)
// ============================================================================
test("WebGPU capability failure fails closed immediately with clear reason", async () => {
  // Falsifiability: If WebGPU check is removed or allowed to proceed without adapter,
  // this test will FAIL because runMeasurement would not reject with WebGPU failure.
  resetLatch();
  const harness = createMockHarness();

  // Mock navigator.gpu.requestAdapter returning null (no GPU adapter)
  harness.cdp.evalHandler = async (expr) => {
    if (expr.includes("navigator.gpu")) {
      return { ok: false, reason: "navigator.gpu.requestAdapter() returned null (no GPU adapter available)" };
    }
    return undefined;
  };

  await assert.rejects(
    runMeasurement(harness.options),
    /WebGPU capability check failed: navigator\.gpu\.requestAdapter\(\) returned null/,
    "Expected runner to reject immediately when WebGPU adapter is unavailable",
  );
  assert.equal(harness.getChromeKilled(), true, "Expected Chrome to be killed on WebGPU failure");
});

// ============================================================================
// TEST 8: INDEPENDENT AUDITABILITY & MACHINE-READABLE SUMMARY (Requirement 9)
// ============================================================================
test("machine-readable summary exposes Chrome PID, userDataDir, route, and asset hashes", async () => {
  // Falsifiability: If summary omits chrome PID or profile, or route, this test will FAIL.
  resetLatch();
  const harness = createMockHarness();

  harness.cdp.responseBodies.set("audit-req", {
    body: "console.log('auditable');",
    base64Encoded: false,
  });

  harness.cdp.queueEvent({
    method: "Network.requestWillBeSent",
    sessionId: "mock-page-session",
    params: {
      requestId: "audit-req",
      request: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1" },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.responseReceived",
    sessionId: "mock-page-session",
    params: {
      requestId: "audit-req",
      response: {
        url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
        status: 200,
        headers: { "cache-control": "public, max-age=31536000, immutable" },
      },
    },
  });
  harness.cdp.queueEvent({
    method: "Network.loadingFinished",
    sessionId: "mock-page-session",
    params: { requestId: "audit-req" },
  });

  const summary = await runMeasurement(harness.options);

  assert.equal(summary.status, "passed");
  assert.equal(summary.route, "models/embeddinggemma-2/basics/");
  assert.equal(summary.viewport.width, 1280);
  assert.equal(summary.viewport.height, 800);
  assert.equal(summary.chrome.pid, 12345, "PID must be present for external audit");
  assert.equal(summary.chrome.userDataDir, "/tmp/webai-chrome-profile-test-12345", "Profile path must be present");
  assert.equal(summary.webgpuAvailable, true);
  assert.equal(summary.loaderState, "ready");
  assert.equal(summary.inferenceResult.completed, true);
  assert.equal(summary.inferenceResult.score, "0.842");
  assert.equal(summary.assets.length, 1);
  assert.equal(summary.assets[0].url, "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1");
  assert.equal(summary.assets[0].status, 200);
  assert.equal(summary.assets[0].immutable, true);
  assert.equal(isValidSha256Hex(summary.assets[0].sha256), true);
});
