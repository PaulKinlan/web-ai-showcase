// Test suite for scripts/measure-raw-ort-dual-runtime.mjs adapter improvements (bead ij4).
// Validates F1, F2, F3, F4, F5 and the critical behavioural rule without browser or network calls.

import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  checkWebGPUAvailable,
  classify,
  computeDecodedSha256,
  computeDecodedSha256FromCdp,
  createUniqueProfilePath,
  decodeCdpResponseBody,
  fetchDecodedSha256,
  getDecodedResourceFromCdp,
  getFilteredRoutes,
  getFilteredViewports,
  getLaunchOptionsForRoute,
  handleTargetAttached,
  loadAllEngines,
  resolveJsDelivrIntegrity,
  ROUTES,
  routeRequiresWebGPU,
  run,
  setupTargetAutoAttach,
  summarize,
  verifyRunResults,
} from "../scripts/measure-raw-ort-dual-runtime.mjs";
import { isStaleProfileDirName } from "../scripts/browser.mjs";

function createMockHarness({
  onWorkerNetworkEnable = null,
  onGetResponseBody = null,
  loaderStates = ["ready"],
  requests = [],
  fetchFn = async () => ({ ok: true, status: 200, arrayBuffer: async () => Buffer.from("test") }),
} = {}) {
  const launchedInstances = [];
  const cdpCallLog = [];
  let activeProfile = null;

  const mockStartServer = async () => ({
    server: { close: () => {} },
    port: 9999,
  });

  const mockLaunchChrome = async (opts) => {
    activeProfile = opts.userDataDir;
    let killed = false;
    const instance = {
      ws: `ws://mock-chrome-${launchedInstances.length}`,
      kill: async () => { killed = true; },
      opts,
      get killed() { return killed; },
    };
    launchedInstances.push(instance);
    return instance;
  };

  const cdpInstances = [];
  const mockCdpFactory = (ws) => {
    const listeners = [];
    const cdp = {
      ws,
      on: (cb) => { listeners.push(cb); },
      send: async (method, params, sessionId) => {
        cdpCallLog.push({ method, params, sessionId });
        if (method === "Target.setAutoAttach") {
          return {};
        }
        if (method === "Target.setDiscoverTargets") {
          return {};
        }
        if (method === "Network.enable") {
          if (sessionId && sessionId.startsWith("worker-") && onWorkerNetworkEnable) {
            await onWorkerNetworkEnable(sessionId, params);
          }
          return {};
        }
        if (method === "Runtime.runIfWaitingForDebugger") {
          return {};
        }
        if (method === "Runtime.evaluate") {
          if (params?.expression?.includes(".model-loader")) {
            return { result: { value: loaderStates } };
          }
          if (params?.expression?.includes("navigator.gpu")) {
            return { result: { value: { ok: true } } };
          }
          return { result: { value: {} } };
        }
        if (method === "Network.getResponseBody") {
          if (onGetResponseBody) return await onGetResponseBody(params, sessionId);
          return { body: "console.log('test');", base64Encoded: false };
        }
        return {};
      },
      emit: (msg) => {
        for (const cb of listeners) cb(msg);
      },
    };
    cdpInstances.push(cdp);
    return cdp;
  };

  const mockOpenPage = async (cdp, _url) => {
    for (const req of requests) {
      cdp.emit({
        method: "Network.requestWillBeSent",
        params: { requestId: req.id, request: { url: req.url } },
        sessionId: req.sessionId || "page-s1",
      });
      cdp.emit({
        method: "Network.responseReceived",
        params: { requestId: req.id, response: { status: req.status || 200 } },
        sessionId: req.sessionId || "page-s1",
      });
      cdp.emit({
        method: "Network.loadingFinished",
        params: { requestId: req.id, encodedDataLength: req.bytes || 100 },
        sessionId: req.sessionId || "page-s1",
      });
    }
    return { targetId: "page-t1", sessionId: "page-s1", errors: [] };
  };

  const mockClosePage = async () => {};
  const mockSetViewport = async () => {};

  return {
    launchedInstances,
    cdpCallLog,
    cdpInstances,
    mockStartServer,
    mockLaunchChrome,
    mockCdpFactory,
    mockOpenPage,
    mockClosePage,
    mockSetViewport,
    fetchFn,
    getActiveProfile: () => activeProfile,
  };
}

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

test("(3a) createUniqueProfilePath produces unique profile directories matching stale-profile prune pattern", () => {
  const path1 = createUniqueProfilePath();
  const path2 = createUniqueProfilePath();
  assert.notEqual(path1, path2, "successive profile paths must be distinct");
  assert.notEqual(
    path1,
    join(tmpdir(), "webai-dual-runtime-profile"),
    "must not be the hardcoded static path",
  );
  // Matches the harness webai-chrome-profile pattern
  assert.match(path1, /webai-chrome-profile-dualruntime-/);

  // Verifies stale-profile prune compatibility (browser.mjs:939, :951)
  const dirName = path1.split("/").pop();
  assert.equal(isStaleProfileDirName(dirName), true, "directory name must match isStaleProfileDirName");
  const pidMatch = dirName.match(/webai-chrome-profile-[^-]+-(\d+)-/);
  assert.ok(pidMatch, "directory name must match PID extraction regex");
  assert.equal(Number(pidMatch[1]), process.pid, "extracted PID must match current process PID");
});

test("(3b) profile directory is created by run() and removed on success and failure paths (B2 / F4)", async () => {
  let createdProfileOk = null;
  const harnessOk = createMockHarness({
    requests: [{ id: "req-1", url: "http://127.0.0.1/app.js", status: 200, bytes: 50 }],
  });

  const testRoutes = [
    {
      slug: "bert-base-turkish-cased-ner",
      route: "models/bert-base-turkish-cased-ner/multi-model/",
      control: null,
      expect: "test",
    },
  ];

  await run({
    routes: testRoutes,
    env: { ROUTE_ONLY: "bert-base-turkish-cased-ner", VIEWPORT_ONLY: "desktop" },
    startServerFn: harnessOk.mockStartServer,
    launchChromeFn: (opts) => {
      createdProfileOk = opts.userDataDir;
      assert.equal(existsSync(createdProfileOk), true, "profile directory must exist during run");
      return harnessOk.mockLaunchChrome(opts);
    },
    cdpFactory: harnessOk.mockCdpFactory,
    openPageFn: harnessOk.mockOpenPage,
    closePageFn: harnessOk.mockClosePage,
    setViewportFn: harnessOk.mockSetViewport,
    sleepFn: async () => {},
  });

  assert.ok(createdProfileOk, "profile must have been created");
  assert.equal(existsSync(createdProfileOk), false, "profile directory must be unlinked on success path");

  // Failure path:
  let createdProfileFail = null;
  const harnessFail = createMockHarness();
  await assert.rejects(async () => {
    await run({
      routes: testRoutes,
      env: { ROUTE_ONLY: "bert-base-turkish-cased-ner", VIEWPORT_ONLY: "desktop" },
      startServerFn: harnessFail.mockStartServer,
      launchChromeFn: (opts) => {
        createdProfileFail = opts.userDataDir;
        assert.equal(existsSync(createdProfileFail), true, "profile directory must exist during run");
        return harnessFail.mockLaunchChrome(opts);
      },
      cdpFactory: harnessFail.mockCdpFactory,
      openPageFn: async () => {
        throw new Error("simulated page load failure");
      },
      closePageFn: harnessFail.mockClosePage,
      setViewportFn: harnessFail.mockSetViewport,
      sleepFn: async () => {},
    });
  });

  assert.ok(createdProfileFail, "profile must have been created");
  assert.equal(existsSync(createdProfileFail), false, "profile directory must be unlinked on failure path");
});

test("(4) decoded-byte hashing is used for jsDelivr pin candidates, not encoded transfer length", async () => {
  const decodedString = "DECODED_TEST_PAYLOAD_TRANSFORMERS_4_3_1_".repeat(14656);
  const decodedBuf = Buffer.from(decodedString, "utf8");
  const expectedSha256 = createHash("sha256").update(decodedBuf).digest("hex");
  const decodedBytesLength = decodedBuf.byteLength;
  const wireEncodedLength = 167590;

  const fakeUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

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

test("(6b) ROUTE_ONLY scopes both main routes and control loop in real runner path", async () => {
  const measuredRoutes = [];
  const harness = createMockHarness({
    requests: [{ id: "req-1", url: "http://127.0.0.1/app.js", status: 200, bytes: 50 }],
  });

  await run({
    env: { ROUTE_ONLY: "embeddinggemma-2", VIEWPORT_ONLY: "desktop" },
    startServerFn: harness.mockStartServer,
    launchChromeFn: harness.mockLaunchChrome,
    cdpFactory: harness.mockCdpFactory,
    openPageFn: async (cdp, url) => {
      measuredRoutes.push(url);
      return harness.mockOpenPage(cdp, url);
    },
    closePageFn: harness.mockClosePage,
    setViewportFn: harness.mockSetViewport,
    sleepFn: async () => {},
  });

  assert.equal(measuredRoutes.length, 2);
  assert.ok(measuredRoutes[0].includes("models/embeddinggemma-2/basics/"));
  assert.ok(measuredRoutes[1].includes("models/embeddinggemma-2/"));
  assert.equal(measuredRoutes.some((u) => u.includes("bert")), false);
  assert.equal(measuredRoutes.some((u) => u.includes("model2vec")), false);
  assert.equal(measuredRoutes.some((u) => u.includes("yolo")), false);
});

test("(6c) VIEWPORT_ONLY scopes viewport selection and skips control when desktop excluded in real runner path", async () => {
  const measuredRoutes = [];
  const harness = createMockHarness({
    requests: [{ id: "req-1", url: "http://127.0.0.1/app.js", status: 200, bytes: 50 }],
  });

  await run({
    env: { ROUTE_ONLY: "embeddinggemma-2", VIEWPORT_ONLY: "mobile" },
    startServerFn: harness.mockStartServer,
    launchChromeFn: harness.mockLaunchChrome,
    cdpFactory: harness.mockCdpFactory,
    openPageFn: async (cdp, url) => {
      measuredRoutes.push(url);
      return harness.mockOpenPage(cdp, url);
    },
    closePageFn: harness.mockClosePage,
    setViewportFn: harness.mockSetViewport,
    sleepFn: async () => {},
  });

  assert.equal(measuredRoutes.length, 1);
  assert.ok(measuredRoutes[0].includes("models/embeddinggemma-2/basics/"));
});

test("(7) unexercised route, zero-request result, or missing hash is treated as FAILURE/INCONCLUSIVE", () => {
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

  // Case 3: valid run with requests and ready and correct jsDelivr hash
  const valid = [
    {
      route: "models/embeddinggemma-2/basics/",
      viewport: "desktop",
      ready: true,
      totalBytes: 586230,
      observedJsDelivrUrls: ["https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1"],
      jsDelivrEntries: {
        "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1": {
          url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
          sha256: "8d6716d9086f57c30a4bf367dba61b887593573c770c454465e8019b2703e743",
          decodedBytes: 586230,
        },
      },
    },
  ];
  const res3 = verifyRunResults(valid);
  assert.equal(res3.ok, true);
  assert.equal(res3.failures.length, 0);

  // Case 4: observed jsDelivr URL has missing or error hash entry (F2)
  const missingHash = [
    {
      route: "models/embeddinggemma-2/basics/",
      viewport: "desktop",
      ready: true,
      totalBytes: 586230,
      observedJsDelivrUrls: ["https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1"],
      jsDelivrEntries: {
        "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1": {
          url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1",
          error: "HTTP 404 Not Found",
        },
      },
    },
  ];
  const res4 = verifyRunResults(missingHash);
  assert.equal(res4.ok, false);
  assert.ok(res4.failures.some((f) => f.includes("failed to resolve decoded SHA-256 for observed jsDelivr URL")));

  // Case 5: empty rows (no routes executed)
  const empty = verifyRunResults([]);
  assert.equal(empty.ok, false);
});

test("(8a) structural falsification against pre-adapter baseline (commit c357b18)", () => {
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

test("(8b) structural falsification against pre-fix review baseline (commit f3e2b98)", () => {
  const preFixSource = execFileSync("git", [
    "show",
    "f3e2b98:scripts/measure-raw-ort-dual-runtime.mjs",
  ], {
    encoding: "utf8",
  });

  // Falsify F1: pre-fix code discarded page auto-attach and swallowed worker attach errors
  assert.equal(
    preFixSource.includes("setupTargetAutoAttach(cdp, page.sessionId).catch(() => {})"),
    true,
    "pre-fix code discarded page auto-attach errors via catch-empty",
  );
  assert.equal(
    preFixSource.includes("void handleTargetAttached(cdp, msg.params, { tracked }).catch"),
    true,
    "pre-fix code swallowed attached target setup errors in void-catch",
  );

  // Falsify F2: pre-fix verifyRunResults never checked jsDelivr integrity entries
  assert.equal(
    preFixSource.includes("observedJsDelivrUrls"),
    false,
    "pre-fix code lacked observedJsDelivrUrls tracking",
  );
  assert.equal(
    preFixSource.includes("failed to resolve decoded SHA-256 for observed jsDelivr URL"),
    false,
    "pre-fix verifyRunResults never checked for missing or failed jsDelivr decoded SHA-256",
  );

  // Falsify F3: pre-fix code applied webgpu: true unconditionally and lacked routeRequiresWebGPU
  assert.equal(
    preFixSource.includes("routeRequiresWebGPU"),
    false,
    "pre-fix code lacked routeRequiresWebGPU helper",
  );
  assert.equal(
    preFixSource.includes("getLaunchOptionsForRoute"),
    false,
    "pre-fix code lacked getLaunchOptionsForRoute",
  );
  assert.equal(
    preFixSource.includes("webgpu: true,\n    });\n    const cdp = new CDP(chrome.ws);"),
    true,
    "pre-fix code launched single Chrome instance with unconditional webgpu: true",
  );

  // Falsify F4: pre-fix profile directory was not compatible with harness stale-prune pattern
  assert.equal(
    preFixSource.includes('webai-dual-runtime-profile-'),
    true,
    "pre-fix code generated webai-dual-runtime-profile- which fails isStaleProfileDirName",
  );
  assert.equal(
    preFixSource.includes('createIsolatedProfileDir("dualruntime")'),
    false,
    "pre-fix code lacked harness stale-prune profile prefix",
  );

  // Falsify F5: pre-fix code lacked decodeCdpResponseBody and computeDecodedSha256FromCdp
  assert.equal(
    preFixSource.includes("decodeCdpResponseBody"),
    false,
    "pre-fix code lacked decodeCdpResponseBody",
  );
  assert.equal(
    preFixSource.includes("computeDecodedSha256FromCdp"),
    false,
    "pre-fix code lacked computeDecodedSha256FromCdp",
  );
  assert.equal(
    preFixSource.includes("getDecodedResourceFromCdp"),
    false,
    "pre-fix code lacked getDecodedResourceFromCdp",
  );
});

test("(9) runner aborts promptly on target setup/Network.enable failure without resuming paused worker (F1)", async () => {
  const harness = createMockHarness({
    onWorkerNetworkEnable: async (sid) => {
      throw new Error(`Simulated Network.enable rejection on ${sid}`);
    },
  });

  const customOpenPage = async (cdp, url) => {
    const page = await harness.mockOpenPage(cdp, url);
    cdp.emit({
      method: "Target.attachedToTarget",
      params: {
        sessionId: "worker-s1",
        targetInfo: { type: "worker", url: "worker.js" },
        waitingForDebugger: true,
      },
    });
    return page;
  };

  const testRoutes = [
    {
      slug: "bert-base-turkish-cased-ner",
      route: "models/bert-base-turkish-cased-ner/multi-model/",
      control: null,
      expect: "test",
    },
  ];

  let runError = null;
  try {
    await run({
      routes: testRoutes,
      env: { ROUTE_ONLY: "bert-base-turkish-cased-ner", VIEWPORT_ONLY: "desktop" },
      startServerFn: harness.mockStartServer,
      launchChromeFn: harness.mockLaunchChrome,
      cdpFactory: harness.mockCdpFactory,
      openPageFn: customOpenPage,
      closePageFn: harness.mockClosePage,
      setViewportFn: harness.mockSetViewport,
      sleepFn: async () => {},
    });
  } catch (err) {
    runError = err;
  }

  // 1. Run must have failed promptly with the attach/network error
  assert.ok(runError, "run() must reject when target setup / Network.enable fails");
  assert.match(runError.message, /Network\.enable rejection on worker-s1/);

  // 2. Worker target MUST NOT have been resumed
  const resumedWorker = harness.cdpCallLog.some(
    (c) => c.method === "Runtime.runIfWaitingForDebugger" && c.sessionId === "worker-s1",
  );
  assert.equal(resumedWorker, false, "paused worker must NEVER be resumed after failed Network.enable");

  // 3. Chrome must have been torn down
  const launched = harness.launchedInstances[0];
  assert.ok(launched, "Chrome instance should have been created");
  assert.equal(launched.killed, true, "Chrome instance must be killed to ensure no hung target remains");

  // 4. Profile must be cleaned up
  assert.equal(existsSync(launched.opts.userDataDir), false, "profile directory must be removed on error path");
});

test("(10) runner fails closed when an observed jsDelivr URL has missing or failed decoded SHA-256 (F2)", async () => {
  const fakeJsDelivrUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
  const harness = createMockHarness({
    requests: [
      { id: "req-1", url: fakeJsDelivrUrl, status: 200, bytes: 586230 },
    ],
    // Both CDP getResponseBody and fetchFn fail to provide a hash
    onGetResponseBody: async () => {
      throw new Error("CDP getResponseBody unavailable");
    },
    fetchFn: async () => ({ ok: false, status: 404, statusText: "Not Found" }),
  });

  const testRoutes = [
    {
      slug: "bert-base-turkish-cased-ner",
      route: "models/bert-base-turkish-cased-ner/multi-model/",
      control: null,
      expect: "test",
    },
  ];

  await assert.rejects(
    async () => {
      await run({
        routes: testRoutes,
        env: { ROUTE_ONLY: "bert-base-turkish-cased-ner", VIEWPORT_ONLY: "desktop" },
        startServerFn: harness.mockStartServer,
        launchChromeFn: harness.mockLaunchChrome,
        cdpFactory: harness.mockCdpFactory,
        openPageFn: harness.mockOpenPage,
        closePageFn: harness.mockClosePage,
        setViewportFn: harness.mockSetViewport,
        fetchFn: harness.fetchFn,
        sleepFn: async () => {},
      });
    },
    (err) => {
      assert.match(err.message, /Measurement run failed verification/);
      assert.match(err.message, /failed to resolve decoded SHA-256 for observed jsDelivr URL/);
      assert.ok(err.message.includes(fakeJsDelivrUrl));
      return true;
    },
  );
});

test("(11) existing routes preserve --disable-gpu launch option (webgpu: false) and only embeddinggemma-2/basics/ receives webgpu: true (F3)", async () => {
  // Unit assertions
  assert.equal(routeRequiresWebGPU("models/bert-base-turkish-cased-ner/multi-model/"), false);
  assert.equal(routeRequiresWebGPU("models/bert-base-turkish-cased-ner/"), false);
  assert.equal(routeRequiresWebGPU("models/model2vec-static-embeddings/multi-model/"), false);
  assert.equal(routeRequiresWebGPU("models/model2vec-static-embeddings/"), false);
  assert.equal(routeRequiresWebGPU("models/yolo-world/multi-model/"), false);
  assert.equal(routeRequiresWebGPU("models/yolo-world/"), false);
  assert.equal(routeRequiresWebGPU("models/embeddinggemma-2/"), false);
  assert.equal(routeRequiresWebGPU("models/embeddinggemma-2/basics/"), true);

  assert.equal(getLaunchOptionsForRoute("models/bert-base-turkish-cased-ner/multi-model/").webgpu, false);
  assert.equal(getLaunchOptionsForRoute("models/embeddinggemma-2/basics/").webgpu, true);
  assert.equal(getLaunchOptionsForRoute("models/embeddinggemma-2/").webgpu, false);

  // Runner-level assertion: run under default unset scoping
  const harness = createMockHarness({
    requests: [{ id: "req-1", url: "http://127.0.0.1/app.js", status: 200, bytes: 100 }],
  });

  await run({
    env: { VIEWPORT_ONLY: "desktop" },
    startServerFn: harness.mockStartServer,
    launchChromeFn: harness.mockLaunchChrome,
    cdpFactory: harness.mockCdpFactory,
    openPageFn: harness.mockOpenPage,
    closePageFn: harness.mockClosePage,
    setViewportFn: harness.mockSetViewport,
    sleepFn: async () => {},
  });

  // Verify launch configurations:
  // First launch for existing routes 1, 2, 3: webgpu: false
  assert.equal(harness.launchedInstances[0].opts.webgpu, false, "existing routes must launch with webgpu: false");
  // Second launch for embeddinggemma-2/basics/: webgpu: true
  assert.equal(harness.launchedInstances[1].opts.webgpu, true, "embeddinggemma-2/basics/ must launch with webgpu: true");
  // Third launch for overview controls: webgpu: false
  assert.equal(harness.launchedInstances[2].opts.webgpu, false, "overview controls must launch with webgpu: false");
});

test("(12) decoded-byte hashing handles base64 WASM and UTF-8 JS correctly, rejecting naive string decode (F5)", async () => {
  // 1. WASM binary payload:
  // Binary header: \0asm\1\0\0\0 + custom bytes (12 bytes)
  const wasmBinary = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0xef, 0xbe, 0xad, 0xde]);
  const base64Wasm = wasmBinary.toString("base64");
  assert.equal(wasmBinary.byteLength, 12);
  assert.equal(base64Wasm.length, 16);

  const decodedWasm = decodeCdpResponseBody(base64Wasm, true);
  assert.deepEqual(decodedWasm, wasmBinary);

  const wasmResult = computeDecodedSha256FromCdp(base64Wasm, true);
  const expectedWasmSha256 = createHash("sha256").update(wasmBinary).digest("hex");
  const wrongNaiveSha256 = createHash("sha256").update(Buffer.from(base64Wasm, "utf8")).digest("hex");

  assert.equal(wasmResult.decodedBytes, 12, "decodedBytes must match true binary byte length (12)");
  assert.notEqual(wasmResult.decodedBytes, base64Wasm.length, "decodedBytes must not be base64 string length");
  assert.equal(wasmResult.sha256, expectedWasmSha256, "sha256 must match true binary bytes hash");
  assert.notEqual(wasmResult.sha256, wrongNaiveSha256, "sha256 must NOT match naive string decode");

  // 2. UTF-8 JS payload (with multi-byte unicode characters where char length != byte length)
  const jsPayload = "/* runtime */ const msg = 'dual-runtime \u2714 \u00a9';";
  const expectedJsBytes = Buffer.byteLength(jsPayload, "utf8");
  assert.notEqual(expectedJsBytes, jsPayload.length, "multi-byte string has byteLength != length");

  const decodedJs = decodeCdpResponseBody(jsPayload, false);
  assert.equal(decodedJs.byteLength, expectedJsBytes);

  const jsResult = computeDecodedSha256FromCdp(jsPayload, false);
  const expectedJsSha256 = createHash("sha256").update(Buffer.from(jsPayload, "utf8")).digest("hex");

  assert.equal(jsResult.decodedBytes, expectedJsBytes);
  assert.equal(jsResult.sha256, expectedJsSha256);

  // 3. CDP getDecodedResourceFromCdp helper
  const mockCdp = {
    send: async (method, params) => {
      if (params.requestId === "wasm-req") {
        return { body: base64Wasm, base64Encoded: true };
      }
      if (params.requestId === "js-req") {
        return { body: jsPayload, base64Encoded: false };
      }
      throw new Error("unexpected request");
    },
  };

  const wasmCdp = await getDecodedResourceFromCdp(mockCdp, "wasm-req");
  assert.equal(wasmCdp.decodedBytes, 12);
  assert.equal(wasmCdp.sha256, expectedWasmSha256);
  assert.equal(wasmCdp.base64Encoded, true);

  const jsCdp = await getDecodedResourceFromCdp(mockCdp, "js-req");
  assert.equal(jsCdp.decodedBytes, expectedJsBytes);
  assert.equal(jsCdp.sha256, expectedJsSha256);
  assert.equal(jsCdp.base64Encoded, false);
});
