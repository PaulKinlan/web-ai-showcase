// Crashpad-cleanup phase 1 prototype (bead web-ai-showcase-arr, child of 04n).
//
// DISABLED BY DEFAULT: every public entry point throws unless
// CRASHPAD_CLEANUP_PHASE1=1 is set. Nothing in scripts/browser.mjs imports this
// module; wiring into the live Chrome launch/kill path is phase 2 and requires
// the separately authorized real-Chrome A/B gate.
//
// Design (credits proposal on 04n, parser review 1c805947): the launch wrapper
// runs under PR_SET_CHILD_SUBREAPER so Chrome's double-forked crashpad handlers
// are ADOPTED by the wrapper (instead of PID 1) at latest at browser exit.
// Cleanup then signals only adopted, identity-verified candidates, exclusively
// via pidfd_send_signal after pidfd_open pinning — never a numeric kill.
//
// This module owns enumeration + identity policy; the syscalls live in the
// audited python bridge scripts/lib/pidfd_bridge.py (Node cannot issue them).

import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BRIDGE = join(dirname(fileURLToPath(import.meta.url)), "pidfd_bridge.py");
const FLAG = "CRASHPAD_CLEANUP_PHASE1";

function assertEnabled() {
  if (process.env[FLAG] !== "1") {
    throw new Error(
      `crashpad-cleanup phase 1 is a disabled prototype — set ${FLAG}=1 (fixture use only; ` +
      `no live Chrome integration is authorized yet, bead web-ai-showcase-arr)`,
    );
  }
}

/** Launch CMD under a subreaper wrapper (the ONLY way adoption can happen). */
export function subreaperExecArgs(cmd, args) {
  assertEnabled();
  return { file: "python3", args: [BRIDGE, "subreaper-exec", "--", cmd, ...args] };
}

/** List kernel-adopted children of `ppid` (default: this process) from /proc. */
export function adoptedChildren(ppid = process.pid) {
  assertEnabled();
  const out = [];
  for (const entry of readdirSync("/proc")) {
    if (!/^[0-9]+$/.test(entry)) continue;
    let stat;
    try {
      stat = readFileSync(`/proc/${entry}/stat`, "utf8");
    } catch { continue; } // raced exit: not ours to judge
    const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    const pidPpid = Number(rest[1]); // field 4 overall
    if (pidPpid === ppid) out.push(Number(entry));
  }
  return out;
}

/** Read a process's cmdline (empty for zombies — fail-closed upstream). */
export function cmdlineOf(pid) {
  assertEnabled();
  try {
    return readFileSync(`/proc/${pid}/cmdline`, "utf8").replaceAll("\0", " ").trim();
  } catch { return null; }
}

/** Zombie state evidence: proves adoption AND that Node/libuv did not auto-reap. */
export function isZombieOf(pid, ppid = process.pid) {
  assertEnabled();
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    return rest[0] === "Z" && Number(rest[1]) === ppid;
  } catch { return false; }
}

/**
 * Signal ONE adopted candidate via the pidfd bridge. Throws (fails closed) on
 * any refusal; returns the bridge's confirmation line on success.
 */
export function pidfdSignal(pid, { expectCmdline, sig = "SIGTERM" }) {
  assertEnabled();
  if (!expectCmdline) throw new Error("pidfdSignal requires an identity marker (fail-closed)");
  return execFileSync("python3", [
    BRIDGE, "pidfd-signal",
    `--pid=${pid}`, `--expected-ppid=${process.pid}`,
    `--require-cmdline-substr=${expectCmdline}`, `--signal=${sig}`,
  ], { encoding: "utf8" }).trim();
}

/**
 * The phase-1 cleanup pass: among MY adopted children, signal exactly those
 * whose cmdline carries `launchMarker`. Ambiguity (unreadable, wrong parent,
 * missing marker) is skipped by the bridge with a refusal; anything unexpected
 * is collected into `skipped` for the caller to report — never signaled.
 */
export function cleanupAdoptedLaunch(launchMarker, { sig = "SIGTERM" } = {}) {
  assertEnabled();
  const signaled = [];
  const skipped = [];
  for (const pid of adoptedChildren()) {
    try {
      pidfdSignal(pid, { expectCmdline: launchMarker, sig });
      signaled.push(pid);
    } catch (e) {
      skipped.push({ pid, reason: String(e.stderr || e.message).trim() });
    }
  }
  return { signaled, skipped };
}
