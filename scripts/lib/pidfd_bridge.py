#!/usr/bin/env python3
"""pidfd bridge for crashpad-cleanup phase 1 (bead web-ai-showcase-arr).

Node cannot issue the prctl/pidfd syscalls this design needs, so the phase-1
prototype keeps EVERY unsafe operation inside this small auditable helper
(stdlib ctypes only, no third-party deps). Subcommands:

  subreaper-exec -- CMD [ARGS...]
      Set PR_SET_CHILD_SUBREAPER on THIS process, then exec CMD. Orphaned
      descendants of CMD (e.g. double-forked crashpad-like handlers that escape
      their process group) are adopted by this process instead of PID 1.

  spawn-fake-handler --launch-token TOKEN --lifetime SECONDS
      Double-fork + setsid a fake "crashpad handler" whose /proc cmdline reads
      `crashpad-like --launch=TOKEN` (via bash exec -a), detached into its own
      process group. Synthetic fixture material ONLY.

  pidfd-signal --pid PID --expected-ppid PPID --require-cmdline-substr S --signal SIG
      The ONLY signaling path. Order is safety-critical:
        1. pidfd_open(PID) FIRST — the pidfd pins the process, so a later PID
           reuse cannot redirect the signal.
        2. verify /proc/PID identity (ppid + cmdline substring) AFTER opening;
           a reused or exited PID fails closed here.
        3. pidfd_send_signal(pidfd, SIG) — never a numeric kill.
      Any ambiguity (missing pid, wrong ppid, missing cmdline marker, zombie)
      means NO signal and exit 2.
"""

import ctypes
import os
import signal
import sys

libc = ctypes.CDLL(None, use_errno=True)

PR_SET_CHILD_SUBREAPER = 36
SYS_pidfd_open = 434        # x86_64
SYS_pidfd_send_signal = 424 # x86_64


def fail(msg):
    print(f"pidfd-bridge: {msg}", file=sys.stderr)
    sys.exit(2)


def cmd_subreaper_exec(argv):
    if not argv or argv[0] != "--":
        fail("subreaper-exec requires -- CMD [ARGS...]")
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        fail(f"prctl(PR_SET_CHILD_SUBREAPER) failed: errno {ctypes.get_errno()}")
    os.execvp(argv[1], argv[1:])


def cmd_spawn_fake_handler(argv):
    opts = dict(a.split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    token = opts.get("--launch-token")
    lifetime = int(opts.get("--lifetime", "300"))
    if not token:
        fail("spawn-fake-handler requires --launch-token")
    # Double-fork so the intermediate parent exits immediately and the handler
    # is orphaned at once (adopted by the subreaper), mirroring how Chrome's
    # crashpad handler escapes before the browser exits.
    if os.fork() > 0:
        os.wait()  # reap the intermediate child, then return to the caller
        return
    os.setsid()  # own session + process group: NOT signalable via the parent's PGID
    if os.fork() > 0:
        os._exit(0)
    os.closerange(0, 3)
    os.execlp("bash", "bash", "-c",
              f'exec -a "crashpad-like --launch={token}" sleep {lifetime}')


def cmd_pidfd_signal(argv):
    opts = dict(a.split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    try:
        pid = int(opts["--pid"])
        expected_ppid = int(opts["--expected-ppid"])
        needle = opts["--require-cmdline-substr"]
        sig = getattr(signal, opts.get("--signal", "SIGTERM"))
    except (KeyError, ValueError, AttributeError) as e:
        fail(f"bad arguments: {e}")

    # Step 1: pin the process BEFORE any identity inspection.
    fd = libc.syscall(SYS_pidfd_open, pid, 0)
    if fd < 0:
        fail(f"pidfd_open({pid}) failed: errno {ctypes.get_errno()} (stale/invalid pid — refused)")
    try:
        # Step 2: identity verification AFTER the pin. A zombie exposes an empty
        # cmdline and fails closed; a reused PID shows the wrong identity.
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read().decode()
            actual_ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            fail(f"cannot read /proc/{pid}/stat — ambiguous, refused")
        if actual_ppid != expected_ppid:
            fail(f"pid {pid} ppid {actual_ppid} != expected {expected_ppid} — not our adopted child, refused")
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            fail(f"cannot read /proc/{pid}/cmdline — ambiguous, refused")
        if needle not in cmdline:
            fail(f"pid {pid} cmdline lacks marker {needle!r}: {cmdline[:120]!r} — refused")
        # Step 3: signal via the pinned pidfd only.
        if libc.syscall(SYS_pidfd_send_signal, fd, sig, None, 0) != 0:
            fail(f"pidfd_send_signal({pid}, {sig}) failed: errno {ctypes.get_errno()}")
        print(f"pidfd-bridge: signaled pid {pid} ({cmdline.strip()[:80]}) via pidfd")
    finally:
        os.close(fd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        fail("usage: pidfd_bridge.py <subreaper-exec|spawn-fake-handler|pidfd-signal> ...")
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"subreaper-exec": cmd_subreaper_exec,
     "spawn-fake-handler": cmd_spawn_fake_handler,
     "pidfd-signal": cmd_pidfd_signal}.get(cmd, lambda _: fail(f"unknown subcommand {cmd!r}"))(rest)
