// Synthetic fixtures for the crashpad-cleanup phase-1 prototype (bead web-ai-showcase-arr).
//
// EVERYTHING here is synthetic: fake "crashpad handler" processes are double-forked `sleep`s with
// a marker argv[0], spawned in their own sessions/PGIDs. No real Chrome, no global cleanup, no
// signals to any process outside this fixture's own adopted children, no 63s/SIGKILL scope.
//
// The file has two stages:
//   - outer (node --test): re-execs THIS file as a plain script under the pidfd bridge's
//     subreaper wrapper with CRASHPAD_CLEANUP_PHASE1=1 (the only way adoption can exist);
//   - inner (plain script, ARR_FIXTURE_SUBREAPER=1): runs the fixture assertions imperatively
//     and exits nonzero on the first failure, always tearing down its own fake handlers.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const HERE = dirname(fileURLToPath(import.meta.url));
const BRIDGE = join(HERE, "../scripts/lib/pidfd_bridge.py");
const SELF = fileURLToPath(import.meta.url);

if (process.env.ARR_FIXTURE_SUBREAPER !== "1") {
  test(
    "crashpad-cleanup fixtures (inner stage re-exec under subreaper wrapper)",
    // Linux-only by design: /proc, pidfd syscalls, PR_SET_CHILD_SUBREAPER.
    { skip: process.platform !== "linux" && "requires Linux /proc + pidfd" },
    () => {
      const r = spawnSync("python3", [BRIDGE, "subreaper-exec", "--", "node", SELF], {
        stdio: "inherit",
        env: { ...process.env, ARR_FIXTURE_SUBREAPER: "1", CRASHPAD_CLEANUP_PHASE1: "1" },
      });
      assert.equal(r.status, 0, "fixture run under subreaper must pass (see output above)");
    },
  );
} else {
  const cleanup = await import("../scripts/lib/crashpad-cleanup.mjs");
  const results = [];
  const check = (name, fn) => {
    try { fn(); results.push(`ok - ${name}`); }
    catch (e) { results.push(`FAIL - ${name}: ${e.message}`); }
  };

  const tokenA = `A-${randomUUID().slice(0, 8)}`;
  const tokenB = `B-${randomUUID().slice(0, 8)}`;
  const tokenC = `C-${randomUUID().slice(0, 8)}`; // sentinel: different "launch", never cleaned

  // Teardown-on-signal (review P2-6): if the fixture is interrupted, still kill exactly the
  // fake handlers this run spawned (their 300s lifetime is the outer bound otherwise).
  const teardownAll = () => {
    for (const token of [tokenA, tokenB, tokenC]) {
      for (const pid of cleanup.adoptedChildren()) {
        if (!cleanup.cmdlineOf(pid)?.includes(token)) continue;
        try { cleanup.pidfdSignal(pid, { expectCmdline: token, sig: "SIGKILL" }); } catch { /* gone */ }
      }
    }
  };
  process.on("SIGTERM", () => { teardownAll(); process.exit(143); });

  const spawnFake = (token) => {
    execFileSync("python3", [BRIDGE, "spawn-fake-handler", `--launch-token=${token}`, "--lifetime=300"]);
  };
  const pidsWithMarker = (marker) =>
    cleanup.adoptedChildren().filter((pid) => cleanup.cmdlineOf(pid)?.includes(marker));
  const alive = (pid) => existsSync(`/proc/${pid}`) && !cleanup.isZombieOf(pid);
  const eq = (a, b, msg) => { if (a !== b) throw new Error(`${msg}: expected ${b}, got ${a}`); };
  const deepEq = (a, b, msg) => {
    if (JSON.stringify([...a].sort()) !== JSON.stringify([...b].sort())) {
      throw new Error(`${msg}: expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
    }
  };

  // Two detached handlers per fake launch (mirrors Chrome's two crashpad handlers).
  spawnFake(tokenA); spawnFake(tokenA);
  spawnFake(tokenB); spawnFake(tokenB);
  spawnFake(tokenC);
  let aPids = [], bPids = [], cPids = [];
  for (let i = 0; i < 50 && (aPids.length < 2 || bPids.length < 2 || cPids.length < 1); i++) {
    await new Promise((r) => setTimeout(r, 100));
    aPids = pidsWithMarker(tokenA);
    bPids = pidsWithMarker(tokenB);
    cPids = pidsWithMarker(tokenC);
  }

  // The zombie assertion above is async now — make the check wrapper awaitable.
  // (check() calls remain synchronous-looking because each fn is awaited through here.)
  const checkAsync = async (name, fn) => {
    try { await fn(); results.push(`ok - ${name}`); }
    catch (e) { results.push(`FAIL - ${name}: ${e.message}`); }
  };

  try {
    check("adoption: detached handlers adopted by the subreaper (not PID 1)", () => {
      eq(aPids.length, 2, "launch A adopted handlers");
      eq(bPids.length, 2, "launch B adopted handlers");
      eq(cPids.length, 1, "sentinel C adopted handlers");
      for (const pid of [...aPids, ...bPids, ...cPids]) {
        if (!alive(pid)) throw new Error(`fake handler ${pid} not alive before cleanup`);
      }
    });

    await checkAsync("cleanup of launch A signals ONLY A's handlers; B and sentinel C survive", async () => {
      const { signaled, skipped } = cleanup.cleanupAdoptedLaunch(tokenA);
      deepEq(signaled, aPids, "exactly A's handlers are signaled");
      if (skipped.length < 3) throw new Error(`expected >=3 refusals (B×2, C×1), got ${skipped.length}`);
      for (const pid of aPids) {
        // Zombie held by THIS process: proves (a) adoption, (b) Node/libuv did NOT auto-reap the
        // adopted grandchild — while the zombie is held its PID cannot be reused. Poll briefly:
        // the signaled sleep's exit latency under load is not synchronous (review P2-4).
        let held = false;
        for (let i = 0; i < 20 && !held; i++) {
          held = cleanup.isZombieOf(pid);
          if (!held) await new Promise((r) => setTimeout(r, 50));
        }
        if (!held) throw new Error(`signaled handler ${pid} must be a zombie of this process`);
      }
      for (const pid of [...bPids, ...cPids]) {
        if (!alive(pid)) throw new Error(`B/C handler ${pid} must survive A's cleanup`);
      }
    });

    check("stale/invalid numeric PID is refused (no signal, fail-closed)", () => {
      let refused = false;
      try { cleanup.pidfdSignal(4194303, { expectCmdline: tokenA }); } catch { refused = true; }
      if (!refused) throw new Error("a stale numeric PID must never be signaled");
    });

    check("cross-launch protection: B's handler with A's marker is refused, B stays alive", () => {
      let refused = false;
      try { cleanup.pidfdSignal(bPids[0], { expectCmdline: tokenA }); } catch { refused = true; }
      if (!refused) throw new Error("an adopted but identity-mismatched child must not be signaled");
      if (!alive(bPids[0])) throw new Error("B's handler must survive the refused signal");
    });

    check("foreign process (not our adopted child) is refused by the PPID check", () => {
      let refused = false;
      try { cleanup.pidfdSignal(process.ppid, { expectCmdline: "anything" }); } catch { refused = true; }
      if (!refused) throw new Error("a non-adopted process must be refused even before identity inspection");
    });

    check("ambiguity fails closed: adopted child WITHOUT the marker is skipped, never signaled", () => {
      const { signaled, skipped } = cleanup.cleanupAdoptedLaunch(tokenB);
      deepEq(signaled, bPids, "exactly B's handlers are signaled");
      if (!skipped.some((s) => s.pid === cPids[0])) throw new Error("sentinel C must be skipped, never signaled");
      if (!alive(cPids[0])) throw new Error("sentinel C must still be alive");
    });
  } finally {
    // Tear down every surviving fake handler we spawned — fixture-owned processes only.
    teardownAll();
  }

  for (const line of results) console.log(line);
  const failures = results.filter((r) => r.startsWith("FAIL"));
  if (failures.length) {
    console.error(`${failures.length} fixture assertion(s) failed`);
    process.exit(1);
  }
  console.log(`all ${results.length} crashpad-cleanup fixture assertions passed`);
  process.exit(0);
}
