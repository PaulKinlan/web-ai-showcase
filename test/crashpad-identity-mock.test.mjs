// Focused: node --test test/crashpad-identity-mock.test.mjs
// Pure fake data + spies ONLY: never reads host /proc, spawns a fixture, or signals a process.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseProcStat, hasExactEnvMarker, discoverCandidate, bindAndSignalMock,
} from "../scripts/lib/crashpad-identity-mock.mjs";

const PID = 48123;
const OWNER = 47321;
const MARKER = "mock-launch-a-1";
const ENV = `OTHER=before\0CRASHPAD_LAUNCH_TOKEN=${MARKER}\0OTHER2=after\0`;
function stat(pid = PID, ppid = OWNER, starttime = "18446744073709551615", state = "S") {
  const fields = Array(20).fill("0");
  fields[0] = state;
  fields[1] = String(ppid);
  fields[19] = starttime;
  return `${pid} (fake handler ) with spaces) ${fields.join(" ")}`;
}
function fixture({ before = stat(), after = stat(), environ = ENV, failAt, fd = 98 } = {}) {
  const calls = [];
  let reads = 0;
  const io = {
    readProc(pid, file) {
      calls.push(`read:${file}:${pid}`);
      if (failAt === file) throw Error("simulated /proc failure");
      if (file === "stat") return ++reads === 1 ? before : after;
      if (file === "environ") return environ;
      throw Error("unexpected fake read");
    },
    pidfdOpen(pid) {
      calls.push(`open:${pid}`);
      if (failAt === "open") throw Error("stale pid");
      return fd;
    },
    pidfdSignal(pinnedFd, sig) {
      calls.push(`signal:${pinnedFd}:${sig}`);
      if (failAt === "signal") throw Error("process exited after verification");
    },
    pidfdClose(pinnedFd) {
      calls.push(`close:${pinnedFd}`);
      if (failAt === "close") throw Error("close failed");
    },
  };
  const candidate = discoverCandidate(PID, OWNER, io.readProc);
  return { candidate, io, calls };
}
const opts = { markerKey: "CRASHPAD_LAUNCH_TOKEN", markerValue: MARKER };
const bound = (f) => bindAndSignalMock(f.candidate, f.io, opts);
const signals = (calls) => calls.filter((c) => c.startsWith("signal:"));
const closes = (calls) => calls.filter((c) => c.startsWith("close:"));

test("stat parser preserves 64-bit starttime and comm with spaces/parentheses", () => {
  assert.deepEqual(parseProcStat(stat()), {
    pid: PID, ppid: OWNER, state: "S", starttime: "18446744073709551615",
  });
  for (const invalid of ["", "48123 malformed", stat().replace("18446744073709551615", "bad")]) {
    assert.equal(parseProcStat(invalid), null);
  }
});

test("environment marker requires complete NUL-delimited key and value", () => {
  assert.equal(hasExactEnvMarker(Buffer.from(ENV), opts.markerKey, MARKER), true);
  for (const bad of [
    `X_CRASHPAD_LAUNCH_TOKEN=${MARKER}\0`,
    `CRASHPAD_LAUNCH_TOKEN=${MARKER}-other\0`,
    `CRASHPAD_LAUNCH_TOKEN=${MARKER}`,
    `OTHER=x\0X_CRASHPAD_LAUNCH_TOKEN=${MARKER}\0`,
  ]) assert.equal(hasExactEnvMarker(bad, opts.markerKey, MARKER), false);
  assert.equal(hasExactEnvMarker(`X_CRASHPAD_LAUNCH_TOKEN=${MARKER}-other\0${ENV}`,
    opts.markerKey, MARKER), true);
  assert.equal(hasExactEnvMarker("", opts.markerKey, MARKER), false);
  assert.equal(hasExactEnvMarker(ENV, "CRASHPAD_LAUNCH_TOKEN", "not-a"), false);
});

test("happy path pins first, verifies post-open identity, then signals fd exactly once", () => {
  const f = fixture();
  assert.equal(f.candidate.starttime, "18446744073709551615");
  assert.deepEqual(bound(f), { action: "signaled" });
  assert.deepEqual(f.calls, [
    `read:stat:${PID}`, `open:${PID}`, `read:stat:${PID}`, `read:environ:${PID}`,
    "signal:98:SIGTERM", "close:98",
  ]);
  assert.equal(signals(f.calls).length, 1);
});

for (const [name, options] of [
  ["reused PID before pin (different starttime)", { after: stat(PID, OWNER, "18446744073709551614") }],
  ["foreign adopted child (different post-pin PPID)", { after: stat(PID, OWNER + 1) }],
  ["post-pin stat has a different PID", { after: stat(PID + 1) }],
  ["exited zombie candidate after pin", { after: stat(PID, OWNER, "18446744073709551615", "Z") }],
  ["missing marker", { environ: "OTHER=x\0" }],
  ["different launch B marker", { environ: "CRASHPAD_LAUNCH_TOKEN=mock-launch-b-1\0" }],
  ["empty zombie environ", { environ: "" }],
  ["unreadable environ", { failAt: "environ" }],
  ["unreadable post-pin stat", { before: stat(), after: null }],
  ["invalid fd from open", { fd: -1 }],
  ["stale PID at pidfd_open", { failAt: "open" }],
]) {
  test(`${name}: refuse without signal`, () => {
    const f = fixture(options);
    const result = bound(f);
    assert.equal(result.action, "skipped");
    assert.equal(signals(f.calls).length, 0);
    assert.equal(closes(f.calls).length, options.fd === -1 || options.failAt === "open" ? 0 : 1);
    if (options.failAt === "open") assert.deepEqual(f.calls, [`read:stat:${PID}`, `open:${PID}`]);
  });
}

test("discovery rejects ambiguous PPID/zombie/malformed stat before opening a pidfd", () => {
  for (const before of [stat(PID, OWNER + 1), stat(PID, OWNER, "33", "Z"), "invalid"]) {
    const f = fixture({ before });
    assert.equal(f.candidate, null);
    assert.equal(bound(f).action, "skipped");
    assert.equal(f.calls.some((c) => c.startsWith("open:") || c.startsWith("signal:")), false);
  }
});

test("fd 0 is valid, and a close error is explicitly reported even after a signal", () => {
  const zero = fixture({ fd: 0 });
  assert.deepEqual(bound(zero), { action: "signaled" });
  assert.deepEqual(closes(zero.calls), ["close:0"]);
  const closeError = fixture({ failAt: "close" });
  assert.deepEqual(bound(closeError), { action: "signaled", closeFailed: true });
  assert.deepEqual(signals(closeError.calls), ["signal:98:SIGTERM"]);
});

test("simulated fd-send failure never reports a successful signal and still closes fd", () => {
  const f = fixture({ failAt: "signal" });
  assert.equal(bound(f).action, "skipped");
  assert.deepEqual(signals(f.calls), ["signal:98:SIGTERM"]); // attempted; fake throws (no real signal)
  assert.deepEqual(closes(f.calls), ["close:98"]);
});
