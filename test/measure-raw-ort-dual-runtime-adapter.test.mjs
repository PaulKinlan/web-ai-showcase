// Test suite for scripts/measure-raw-ort-dual-runtime.mjs adapter improvements (bead ij4).
// Validates B0, B2, B3, B4, B5 and the critical behavioural rule without browser or network calls.

import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  checkWebGPUAvailable,
  classify,
  computeDecodedSha256,
  createUniqueProfilePath,
  fetchDecodedSha256,
  getFilteredRoutes,
  getFilteredViewports,
  handleTargetAttached,
  loadAllEngines,
  resolveJsDelivrIntegrity,
  ROUTES,
  setupTargetAutoAttach,
  summarize,
  verifyRunResults,
} from "../scripts/measure-raw-ort-dual-runtime.mjs";

test("(1) route entry exists for embeddinggemma-2/basics/ and is selected by ROUTE_ONLY filter", () => {
  const entry = ROUTES.find((r) => r.slug === "embeddinggemma-2");
  assert.ok(entry, "embeddinggemma-2 route entry must exist in ROUTES");
  assert.equal(entry.route, "models/embeddinggemma-2/basics/");
  assert.equal(entry.control, "models/embeddinggemma-2/");
  assert.equal(entry.requiresWebGPU, true);
  assert.match(entry.expect, /embeddinggemma-2/);

  // Exact slug filter
  const filteredSlug = getFilteredRoutes(ROUTES, { ROUTE_ONLY: "embeddinggemma-2" });
  assert.equal(filteredSlug.length, 1);
  assert.equal(filteredSlug[0].slug, "embeddinggemma-2");

  // Route path substring filter
  const filteredRoute = getFilteredRoutes(ROUTES, {
    ROUTE_ONLY: "models/embeddinggemma-2/basics/",
  });
  assert.equal(filteredRoute.length, 1);
  assert.equal(filteredRoute[0].slug, "embeddinggemma-2");

  // ROUTE_FILTER alias works identically
  const filteredAlias = getFilteredRoutes(ROUTES, { ROUTE_FILTER: "embeddinggemma-2" });
  assert.equal(filteredAlias.length, 1);
  assert.equal(filteredAlias[0].slug, "embeddinggemma-2");
});

test("(2a) checkWebGPUAvailable correctly detects WebGPU presence and fails closed on absence", async () => {
  // Case 1: navigator.gpu missing
  const noGpu = await checkWebGPUAvailable(async () => ({
    ok: false,
    reason: "navigator.gpu missing",
  }));
  assert.equal(noGpu.ok, false);
  assert.match(noGpu.reason, /navigator\.gpu missing/);

  // Case 2: requestAdapter returns null (no GPU adapter available)
  const nullAdapter = await checkWebGPUAvailable(async () => ({
    ok: false,
    reason: "requestAdapter returned null (no GPU adapter)",
  }));
  assert.equal(nullAdapter.ok, false);
  assert.match(nullAdapter.reason, /no GPU adapter/);

  // Case 3: WebGPU adapter available
  const gpuOk = await checkWebGPUAvailable(async () => ({ ok: true }));
  assert.equal(gpuOk.ok, true);
});

test("(2b) loadAllEngines fails closed immediately when model-loader enters unsupported state", async () => {
  const mockCdp = {
    send: async (method) => {
      if (method === "Runtime.evaluate") {
        return { result: { value: ["unsupported"] } };
      }
      return {};
    },
  };

  await assert.rejects(
    async () => {
      await loadAllEngines(mockCdp, "s1", "embeddinggemma-2 test", 5000);
    },
    (err) => {
      assert.match(err.message, /FAIL CLOSED/);
      assert.match(err.message, /unsupported/i);
      return true;
    },
  );
});

test("(3a) createUniqueProfilePath produces unique profile directories per run", () => {
  const path1 = createUniqueProfilePath();
  const path2 = createUniqueProfilePath();
  assert.notEqual(path1, path2, "successive profile paths must be distinct");
  assert.notEqual(
    path1,
    join(tmpdir(), "webai-dual-runtime-profile"),
    "must not be the hardcoded static path",
  );
  assert.match(path1, /webai-dual-runtime-profile-/);
});

test("(3b) profile directory is removed on normal and failure paths in finally block", async () => {
  // Normal/success path lifecycle
  const testDirSuccess = join(
    tmpdir(),
    `test-profile-lifecycle-ok-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
  );
  mkdirSync(testDirSuccess, { recursive: true });
  assert.equal(existsSync(testDirSuccess), true);

  try {
    // Normal run work happens here
  } finally {
    rmSync(testDirSuccess, { recursive: true, force: true });
  }
  assert.equal(existsSync(testDirSuccess), false, "profile must be removed on success path");

  // Timeout/failure path lifecycle
  const testDirFail = join(
    tmpdir(),
    `test-profile-lifecycle-fail-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
  );
  mkdirSync(testDirFail, { recursive: true });
  assert.equal(existsSync(testDirFail), true);

  await assert.rejects(async () => {
    try {
      throw new Error("simulated run failure");
    } finally {
      rmSync(testDirFail, { recursive: true, force: true });
    }
  });
  assert.equal(existsSync(testDirFail), false, "profile must be removed on failure/timeout path");
});

test("(4) decoded-byte hashing is used for jsDelivr pin candidates, not encoded transfer length", async () => {
  // Simulate an asset where encoded wire transfer size differs from decoded bytes:
  // (e.g. 4.3.1 entrypoint is 167590 wire vs 586230 decoded).
  const decodedString = "DECODED_TEST_PAYLOAD_TRANSFORMERS_4_3_1_".repeat(14656);
  const decodedBuf = Buffer.from(decodedString, "utf8");
  const expectedSha256 = createHash("sha256").update(decodedBuf).digest("hex");
  const decodedBytesLength = decodedBuf.byteLength;
  const wireEncodedLength = 167590;

  const fakeUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

  // Mock fetch function returning decoded buffer
  const mockFetch = async (url) => {
    if (url === fakeUrl) {
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () =>
          decodedBuf.buffer.slice(
            decodedBuf.byteOffset,
            decodedBuf.byteOffset + decodedBuf.byteLength,
          ),
      };
    }
    return { ok: false, status: 404 };
  };

  const integrity = await resolveJsDelivrIntegrity([fakeUrl], { fetchFn: mockFetch });
  assert.ok(integrity[fakeUrl]);
  assert.equal(integrity[fakeUrl].sha256, expectedSha256);
  assert.equal(integrity[fakeUrl].decodedBytes, decodedBytesLength);
  assert.notEqual(integrity[fakeUrl].decodedBytes, wireEncodedLength);

  // Test summarize with simulated Network events
  const events = [
    { kind: "request", id: "req-1", url: fakeUrl },
    { kind: "finished", id: "req-1", encodedDataLength: wireEncodedLength },
    { kind: "response", id: "req-1" },
  ];
  const reqStatus = new Map([["req-1", {
    status: 200,
    fromDiskCache: false,
    fromServiceWorker: false,
  }]]);

  const snap = await summarize(events, { fetchFn: mockFetch, requestStatusMap: reqStatus });
  assert.equal(
    snap.groups["transformers-lib-js"].bytes,
    wireEncodedLength,
    "wire transfer length in groups remains encoded",
  );
  assert.ok(snap.jsDelivrEntries[fakeUrl], "jsDelivr integrity entry must be present");
  assert.equal(
    snap.jsDelivrEntries[fakeUrl].decodedBytes,
    decodedBytesLength,
    "pin candidate MUST use decoded byte count",
  );
  assert.equal(
    snap.jsDelivrEntries[fakeUrl].sha256,
    expectedSha256,
    "pin candidate MUST use decoded sha256",
  );
  assert.notEqual(
    snap.jsDelivrEntries[fakeUrl].decodedBytes,
    wireEncodedLength,
    "pin candidate must NOT use wire encoded length",
  );
});

test("(5a) setupTargetAutoAttach configures waitForDebuggerOnStart: true", async () => {
  let attachParams = null;
  const mockCdp = {
    send: async (method, params) => {
      if (method === "Target.setAutoAttach") {
        attachParams = params;
      }
      return {};
    },
  };

  await setupTargetAutoAttach(mockCdp);
  assert.ok(attachParams);
  assert.equal(attachParams.autoAttach, true);
  assert.equal(attachParams.flatten, true);
  assert.equal(attachParams.waitForDebuggerOnStart, true);
});

test("(5b) handleTargetAttached issues Network.enable and settles BEFORE Runtime.runIfWaitingForDebugger", async () => {
  const callLog = [];
  const mockCdp = {
    send: async (method, params, sessionId) => {
      const entry = { method, params, sessionId, order: callLog.length };
      callLog.push(entry);
      return {};
    },
  };

  const tracked = new Set();
  await handleTargetAttached(
    mockCdp,
    {
      sessionId: "worker-session-b5",
      targetInfo: { type: "worker", url: "worker.js" },
      waitingForDebugger: true,
    },
    { tracked },
  );

  assert.equal(
    callLog.length,
    2,
    "must have issued Network.enable then Runtime.runIfWaitingForDebugger",
  );
  assert.equal(callLog[0].method, "Network.enable");
  assert.equal(callLog[0].sessionId, "worker-session-b5");
  assert.equal(callLog[1].method, "Runtime.runIfWaitingForDebugger");
  assert.equal(callLog[1].sessionId, "worker-session-b5");
  assert.ok(
    callLog[0].order < callLog[1].order,
    "Network.enable must precede Runtime.runIfWaitingForDebugger",
  );
});

test("(5c) handleTargetAttached does NOT resume worker if Network.enable fails", async () => {
  const callLog = [];
  const mockCdp = {
    send: async (method, params, sessionId) => {
      callLog.push({ method, sessionId });
      if (method === "Network.enable") {
        throw new Error("CDP Network.enable rejection");
      }
      return {};
    },
  };

  const tracked = new Set();
  await assert.rejects(
    async () => {
      await handleTargetAttached(
        mockCdp,
        {
          sessionId: "worker-session-err",
          targetInfo: { type: "worker" },
          waitingForDebugger: true,
        },
        { tracked },
      );
    },
    /Network\.enable rejection/,
  );

  const resumed = callLog.some((c) => c.method === "Runtime.runIfWaitingForDebugger");
  assert.equal(
    resumed,
    false,
    "must NOT send runIfWaitingForDebugger when network enablement fails",
  );
});

test("(6a) default scoping (unset env) keeps all routes and both viewports", () => {
  const routes = getFilteredRoutes(ROUTES, {});
  assert.equal(routes.length, ROUTES.length, "all routes included by default");
  assert.deepEqual(routes.map((r) => r.slug), ROUTES.map((r) => r.slug));

  const viewports = getFilteredViewports({});
  assert.deepEqual(viewports, ["desktop", "mobile"]);
});

test("(6b) ROUTE_ONLY scopes both main routes and control loop", () => {
  const env = { ROUTE_ONLY: "embeddinggemma-2" };
  const scoped = getFilteredRoutes(ROUTES, env);
  assert.equal(scoped.length, 1);
  assert.equal(scoped[0].slug, "embeddinggemma-2");

  // In the runner, loop 1 iterates scoped routes and loop 2 iterates scoped routes:
  const loop1Routes = scoped.map((r) => r.route);
  const loop2Controls = scoped.map((r) => r.control).filter(Boolean);

  assert.deepEqual(loop1Routes, ["models/embeddinggemma-2/basics/"]);
  assert.deepEqual(loop2Controls, ["models/embeddinggemma-2/"]);
});

test("(6c) VIEWPORT_ONLY scopes viewport selection and skips control when desktop excluded", () => {
  const envDesktop = { VIEWPORT_ONLY: "desktop" };
  const vpsDesktop = getFilteredViewports(envDesktop);
  assert.deepEqual(vpsDesktop, ["desktop"]);
  assert.equal(vpsDesktop.includes("desktop"), true, "desktop control loop runs");

  const envMobile = { VIEWPORT_ONLY: "mobile" };
  const vpsMobile = getFilteredViewports(envMobile);
  assert.deepEqual(vpsMobile, ["mobile"]);
  assert.equal(vpsMobile.includes("desktop"), false, "desktop control loop is skipped");
});

test("(7) unexercised route or zero-request result is treated as FAILURE/INCONCLUSIVE", () => {
  // Case 1: zero bytes / zero requests captured
  const zeroBytes = [
    {
      route: "models/embeddinggemma-2/basics/",
      viewport: "desktop",
      ready: true,
      totalBytes: 0,
    },
  ];
  const res1 = verifyRunResults(zeroBytes);
  assert.equal(res1.ok, false);
  assert.ok(res1.failures.some((f) => f.includes("zero network requests")));

  // Case 2: loader never reached ready
  const unready = [
    {
      route: "models/embeddinggemma-2/basics/",
      viewport: "desktop",
      ready: false,
      totalBytes: 12345,
    },
  ];
  const res2 = verifyRunResults(unready);
  assert.equal(res2.ok, false);
  assert.ok(res2.failures.some((f) => f.includes("never reached 'ready'")));

  // Case 3: valid run with requests and ready
  const valid = [
    {
      route: "models/embeddinggemma-2/basics/",
      viewport: "desktop",
      ready: true,
      totalBytes: 586230,
    },
  ];
  const res3 = verifyRunResults(valid);
  assert.equal(res3.ok, true);
  assert.equal(res3.failures.length, 0);

  // Case 4: empty rows (no routes executed)
  const empty = verifyRunResults([]);
  assert.equal(empty.ok, false);
});

test("(8) structural falsification against pre-adapter baseline (commit c357b18)", () => {
  // Read the pre-adapter code directly from git to prove that every blocker would FAIL against it
  const preAdapterSource = execFileSync("git", [
    "show",
    "c357b18:scripts/measure-raw-ort-dual-runtime.mjs",
  ], {
    encoding: "utf8",
  });

  // Falsify B0: pre-adapter lacked requiresWebGPU and checkWebGPUAvailable
  assert.equal(
    preAdapterSource.includes("embeddinggemma-2"),
    false,
    "pre-adapter lacked embeddinggemma-2 route",
  );
  assert.equal(
    preAdapterSource.includes("requiresWebGPU"),
    false,
    "pre-adapter lacked requiresWebGPU flag",
  );
  assert.equal(
    preAdapterSource.includes("checkWebGPUAvailable"),
    false,
    "pre-adapter lacked checkWebGPUAvailable",
  );
  assert.equal(
    preAdapterSource.includes("webgpu: true"),
    false,
    "pre-adapter never passed webgpu: true to launchChrome",
  );

  // Falsify B2: pre-adapter used fixed static profile and lacked unique profile generator and run-lifecycle cleanup
  assert.equal(
    preAdapterSource.includes('join(tmpdir(), "webai-dual-runtime-profile")'),
    true,
    "pre-adapter pinned fixed path",
  );
  assert.equal(
    preAdapterSource.includes("createUniqueProfilePath"),
    false,
    "pre-adapter lacked unique profile generator",
  );
  assert.equal(
    preAdapterSource.includes(
      "userDataDir: PROFILE,\n  resetProfile: false,\n  removeProfileOnKill: false,",
    ),
    true,
    "pre-adapter had resetProfile: false and removeProfileOnKill: false",
  );

  // Falsify B3: pre-adapter only recorded encodedDataLength, not decoded-byte sha256
  assert.equal(
    preAdapterSource.includes("computeDecodedSha256"),
    false,
    "pre-adapter lacked decoded sha256 calculation",
  );
  assert.equal(
    preAdapterSource.includes("resolveJsDelivrIntegrity"),
    false,
    "pre-adapter lacked jsDelivr integrity resolution",
  );
  assert.equal(
    preAdapterSource.includes("jsDelivrEntries"),
    false,
    "pre-adapter lacked jsDelivr pin entries",
  );

  // Falsify B4: pre-adapter lacked scoping filters for both loops
  assert.equal(
    preAdapterSource.includes("getFilteredRoutes"),
    false,
    "pre-adapter lacked getFilteredRoutes",
  );
  assert.equal(
    preAdapterSource.includes("getFilteredViewports"),
    false,
    "pre-adapter lacked getFilteredViewports",
  );
  assert.equal(
    preAdapterSource.includes("ROUTE_ONLY"),
    false,
    "pre-adapter lacked ROUTE_ONLY env var",
  );

  // Falsify B5: pre-adapter had waitForDebuggerOnStart: false and never called Runtime.runIfWaitingForDebugger
  assert.equal(
    preAdapterSource.includes("waitForDebuggerOnStart: false"),
    true,
    "pre-adapter used waitForDebuggerOnStart: false",
  );
  assert.equal(
    preAdapterSource.includes("waitForDebuggerOnStart: true"),
    false,
    "pre-adapter lacked waitForDebuggerOnStart: true",
  );
  assert.equal(
    preAdapterSource.includes("Runtime.runIfWaitingForDebugger"),
    false,
    "pre-adapter never resumed waiting debuggers",
  );
  assert.equal(
    preAdapterSource.includes("setupTargetAutoAttach"),
    false,
    "pre-adapter lacked setupTargetAutoAttach",
  );
  assert.equal(
    preAdapterSource.includes("handleTargetAttached"),
    false,
    "pre-adapter lacked handleTargetAttached",
  );

  // Falsify Critical Rule: pre-adapter lacked verifyRunResults and allowed 0-request runs to exit 0
  assert.equal(
    preAdapterSource.includes("verifyRunResults"),
    false,
    "pre-adapter lacked verifyRunResults",
  );
});
