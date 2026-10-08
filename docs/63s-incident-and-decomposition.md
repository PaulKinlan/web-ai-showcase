# 63s — INCIDENT RECORD and read-only decomposition plan

Status: **SAFETY HOLD.** No worker that spawns processes will be dispatched. No fixture parent, no
SIGKILL, no setsid, no Chrome, no same-brief retry. This document is the READ-ONLY deliverable: an exact
incident record and a bounded A/B/C decomposition. Nothing here spawns an OS child.

## 1. Incident record (exact facts, nothing inferred)

### Failed runs

| Attempt | Run id | Outcome | Log dir |
|---|---|---|---|
| 1 | `deeb6030-0fd9-483f-b87b-7813074086ce` | timeout `1800000ms`, **0 files produced** | `/tmp/pi-subagents-uid-1000/async-subagent-runs/deeb6030-0fd9-483f-b87b-7813074086ce` |
| 2 | `540c3230-ef0f-421f-883a-a9045df0de8a` | timeout `1800000ms`, **0 files produced** | `/tmp/pi-subagents-uid-1000/async-subagent-runs/540c3230-ef0f-421f-883a-a9045df0de8a` |

Attempt 1 was the full brief; attempt 2 was the deliberately smaller piece-1 brief (guardian module plus
one focused test). **Both produced zero lines and no commit.**

### Orphan (attempt 1 only)

| Field | Value |
|---|---|
| pid | `2117824` |
| cmdline | `node -e setInterval(()=>{}, 1000);` |
| cwd | `/home/exedev/worktrees/web-ai-showcase-63s` (this bead's worktree) |
| `FLEET_LANE` | `embedding2` (this lane) |
| `FLEET_BEAD` | `web-ai-showcase-ik9` (inherited from the lane env; not the owning bead) |
| ppid | **`1`** — already reparented to init; **ORPHANED** |
| pgid | `2117824` — **its own** process-group leader |
| sid | `2117824` — **its own** session leader |
| state | `S` |
| started | `Thu Oct 8 08:59:52 2026` (during attempt 1) |
| siblings in its group | none |

The orphan was **its own session and process-group leader**. A guardian that killed only its own process
group would **not** have caught this process. That is a live instance of the group-escape residual the
approved design already names. Whether it was a deliberately-isolated sentinel or a fake Chrome child is
**not asserted** — the failed worker left no code and no log to determine it.

### Cleanup performed

Ownership was verified by `/proc` cwd + `FLEET_LANE` **before** any signal was sent; nothing was killed by
command name or by scanning.

1. `kill -TERM 2117824` — accepted.
2. Bounded wait of 10 × 1s: process exited **within 10s**. **SIGKILL was not required.**
3. Verification: process absent from `/proc`.

### Zero-residual scan (after cleanup)

- Scan A: every `/proc/<pid>/environ` readable by this user, filtered on `FLEET_LANE=embedding2`
  → **0 processes**.
- Scan B: every `/proc/<pid>/cwd` inside this worktree → **0 processes**.
- Worktree: `0` modified/untracked files.

Attempt 2 left **no** orphan (same two scans → 0). The leak was attempt 1's fixture, **not** a systematic
effect of dispatching.

## 2. Lesson

The external cleanup boundary is **inadequate** for fixtures that deliberately spawn processes. The
fixture parent is a *test artifact*, and when the runner is killed by an outer timeout the fixture's own
children are not contained: a 30-minute runner timeout produced a PID-1 orphan with its own session. An
outer timeout killing the fixture parent is **neither containment nor safe cleanup**. Therefore fixtures
that spawn processes must not be run until an **external supervisor** — one that outlives the test runner
— is designed and independently reviewed for exactly the escaped-session case.

## 3. Read-only decomposition

### A — pure module / handshake contract, **NO OS child**

A pure, injectable module implementing the approved contract with **no process spawning at all**: session
and process-group intent as *data*, handshake as a pure state machine, writer-EOF lifetime as an
injectable event, and group-teardown expressed as a recorded *decision* rather than a `kill`. Tests use
**mocks only**; they assert decisions, ordering and fail-closed behaviour, never real signals.

This is safe to build now: it spawns nothing, signals nothing, and cannot orphan anything.

Must be provable in A: fail-closed-before-spawn on handshake/ownership failure · **no unconditional
SIGTERM group kill** (an unauthenticated SIGTERM must produce *no* teardown decision) · EOF-only trigger
after a verified handshake · no stored-PGID and no numeric-fallback path anywhere · guardian-loss and
group-escape surfaced as observable outcomes.

### B — external fixture-supervisor design (**read-only design first**)

A supervisor that **outlives the test runner** and can still clean up after the runner is SIGKILLed.
It must use **recorded child identities** (pid **plus start time**), never a scan and never a bare numeric
kill. It must handle the escaped-session case observed above (a child with its own SID/PGID).

Requirements B must satisfy to be approved:
- independent of the test runner's lifetime — a runner SIGKILL must not stop it;
- cleanup runs **even when the runner is killed by an outer timeout**;
- covers children that escaped into their **own session**, using recorded identities;
- never signals a process it did not record;
- a foreign sentinel survives every path;
- **hard-bounded**, and self-terminating.

**If B cannot guarantee cleanup under an outer timeout, 63s stays BLOCKED** rather than testing unsafely.
That is a real possible outcome and will be reported as such.

### C — fake-process fixture (**only after coord approves B**)

Not before. No fixture parent SIGKILL, no setsid, no Chrome until B is approved.

## 4. Ask

Review A and B. A is safe to implement immediately. B must be independently reviewed with a supervisor
safety proof (including the escaped-session case) **before any new process is spawned**.
