# 63s — Guardian process containment (design, PHASE 1)

Status: **AWAITING COORDINATOR APPROVAL.** No code written. No browser launched. No process signalled.

Read-only oracle verdict: pipe-only guardian **can** meet the scoped best-effort target, but
implementation of the originally-proposed design is **BLOCKED** until the lifecycle contract below is
approved, because the proposal as written contains two behaviours the acceptance bar prohibits.

## Problem (measured, not inferred)

`scripts/browser.mjs` `spawnChromeOnce` launches Chrome with `detached: true`, so Chrome is its own
process-group leader. Validators run under `timeout -k 30`; that escalation sends an **untrappable
SIGKILL**, which runs no JavaScript, bypassing the harness exit/signal hooks and any `finally`. Chrome is
reparented to PID 1 and the browser group survives. Observed: `eg2conformance`, group alive under PID1.

`9z1` suppresses crashpad but does **not** prevent whole-browser orphan. `04n` is a different mechanism
(crashpad survives ordinary teardown) and is explicitly out of scope here. The rejected `c3h` approach
(`/proc` cmdline+env revalidation, then `process.kill(pid)`) must not return — PID-reuse TOCTOU.

## Scoped acceptance target

Best-effort containment of the validator-parent SIGKILL path **while a correctly handshaken guardian
remains alive and Chrome descendants remain in its process group**. This is **not** an unconditional
no-orphans guarantee and must never be described as one.

## Two defects in the proposal as written (both prohibited, not residual)

1. **Unauthenticated SIGTERM → group kill.** An unconditional "on SIGTERM, kill the group" handler could
   terminate a *healthy* validator's browser if any other actor signals the guardian. The acceptance bar
   makes premature termination of a healthy validator's browser a **blocker**. Group termination must be
   restricted to verified parent-lifetime EOF or an authenticated intentional shutdown.
2. **Incidental writer closure → premature EOF.** If the designated writer can be closed by anything
   other than parent death, EOF fires while the parent is healthy. The writer must be **exclusively
   parent-owned for precisely the browser's intended lifetime**.

## Revised lifecycle contract (the approval ask)

Implementation requires all five, each provable:

1. Guardian runs in a **new session**; the parent is outside it. Chrome and fake descendants are admitted
   **only after** a successful handshake.
2. An **exclusively parent-owned writer** is held throughout the browser's intended lifetime.
3. **No unauthenticated SIGTERM-to-group-kill path.**
4. **Only the live guardian** signals its **own verified group**, with **no stale-PGID fallback** and no
   numeric `/proc`-checked kills.
5. Guardian loss and group escape are **reported**, not masked.

Ownership is established **before Chrome can launch**; if the handshake fails, **fail the launch**.
The healthy-parent / early-EOF fixture is a **hard acceptance gate**.

PID-reuse soundness holds **only** while the guardian is alive, verified as its own session/PGID leader,
and signalling its current group itself. Neither the validator nor any later cleanup job may signal a
saved guardian PGID.

## Explicit residuals (named, never masked)

- **Guardian death is fail-open.** If SIGKILLed before or after handshake, nobody remains to act on parent
  EOF. A live parent can detect guardian exit; a simultaneously dead parent cannot. The guardian does not
  solve its own death.
- **PGID escape.** Same-group containment does **not** cover descendants that call `setsid()`/`setpgid()`.
  Chrome zygote/renderer/GPU/helper membership requires observation in later exclusive browser validation
  and must **not** be inferred from `detached:false`. crashpad-outside-group is a stated containment limit
  here even though its own bead is 04n.
- **Inherited writer / CLOEXEC limits.** `CLOEXEC` is not a universal no-inheritance guarantee: an
  explicitly passed or duplicated writer, or a fork retaining it before exec, prevents EOF. Conversely,
  losing the sole writer early produces a **false** parent-death indication.

## Fixture contract (fake processes only)

Known fixture parent; guardian; owned fake Chrome **A** and grandchild **B**; unrelated same-UID **foreign
sentinel** in another session. **No real Chrome, no shared profile.** An **independent supervisor** holds
handles and verifies an identity/start-time plus SID/PGID handshake **before any signal**. Terminate
**only the known fixture-parent `ChildProcess` handle** — never discovered PIDs.

**Discriminating positive cases** (must genuinely prove teardown, each owned process asserted, not merely
guardian exit):
- parent SIGKILL after handshake → **A and B** disappear, sentinel survives
- normal parent exit → same teardown
- early handshake failure → A/B must never launch or survive
- late A exit while B remains → guardian must still contain **B**, not equate A's exit with completed teardown

**Hard safety case:** a healthy parent functions while unrelated descriptors close → the guardian must
**not** kill A/B. Any attainable premature closure of the designated writer during a valid lifecycle is a
**blocker**, not a residual.

**Expected-negative, isolated:** deliberately retain the parent pipe writer in a holder, then SIGKILL the
parent. Assert and prominently report **NOT-CONTAINED** — EOF does not fire and A/B remain pending
independent sandbox cleanup. **Never** included in any containment pass count. Guardian-death and
PGID-escape injections likewise demonstrate residuals only, in disposable isolation.

Positive containment counts and expected-negative residuals are reported **separately**. Tests that only
prove a guardian exited, or that clean up their own leaked children before checking, do not discriminate
the failure being addressed.

**Negative cases can orphan processes themselves** and therefore require a disposable isolation boundary
and an independent test supervisor. An external timeout killing the fixture parent is neither containment
nor safe cleanup.

## Options for stronger containment (NOT authorized)

- **pidfd.** Could independently report parent death despite a leaked writer, without PID-reuse ambiguity.
  The parent must open a pidfd for **itself while alive** and transfer the kernel handle to the guardian;
  opening by a remembered numeric PID after parent death reintroduces the `c3h` race. Acquisition/transfer
  failure must prevent Chrome launch if mandated. Scope increase, **not a prerequisite** for the
  residual-bearing pipe-only target.
- **Delegated cgroup v2 with `cgroup.kill`.** Would include descendants that change PGID. This VM's
  process runs in `/system.slice/fleet-sdk-host.service`; `cgroup.procs`, `cgroup.subtree_control` and
  `cgroup.kill` are root-owned and `cgroup.kill` is root-writable only. Requires Paul + fleet-ops approval;
  **not available for implementation in this bead.**

## Next step

Coordinator approval of the revised pipe ownership, shutdown and handshake contract, **including explicit
acceptance of the best-effort residuals**, before any implementation.
