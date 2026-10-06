#!/usr/bin/env node
// Gated landing — ONE tracked implementation of the atomic ASSERT -> BRANCH -> ACT push block
// (bead web-ai-showcase-053).
//
// WHY THIS EXISTS: "dry-run then push" written literally is NOT a preflight — it is a rehearsal
// followed by the same action, and it pushes EVEN WHEN the dry run printed '[rejected]'. The
// refusal line is only a decision point if the refusal branch can never reach the push. So this
// script contains NO unasserted action: the push is reachable only from the branch that asserted
// all three parts below. Several lanes built this block independently tonight; four copies of one
// implementation is the condition where a single script helps and four copies diverge.
//
// REFERENCE: ~/merger-push-gated.sh (merger lane, heavily rehearsed 2026-10-05; rehearsal
// provenance recorded in bead web-ai-showcase-053). This is an EXTRACTION into this repo's script
// conventions (Node .mjs, deno fmt/check, node --test), not a byte copy. Deliberate deviations
// are listed under DEVIATIONS below; everything else follows the reference's measured behaviour.
//
// THE FOUR-BRANCH FORM, in this exact order (the order IS the safety property):
//   1. output contains '[rejected]'                 -> REFUSED  rc=1 (do NOT push; recovery is
//      fetch + re-merge + RE-GATE the merged tree)
//   2. ELIF output contains 'Everything up-to-date'  -> NO-OP    rc=2 (throwaway-ref probe only;
//      do NOT push)
//   3. ELIF output matches the ANCHORED regex
//      '^ *[0-9a-f]{4,}\.\.[0-9a-f]{4,} +HEAD -> '  -> OK       rc=0 (push)
//   4. ELSE                                          -> UNKNOWN  rc=4 (fail closed; do NOT push)
//
// NON-OBVIOUS REQUIREMENTS enforced here (each measured rather than assumed, per the bead):
//   * REJECTION IS TESTED FIRST. A refusal can still contain sha lines, so a block that looks for
//     the update row first can classify a REFUSAL AS A PASS and push. '[rejected]' is the one
//     string whose PRESENCE is decisive.
//   * THE UPDATE ROW IS AN ANCHORED REGEX, not a substring: prose that merely MENTIONS the row
//     (git hints, echoed logs) must not satisfy it.
//   * MATCH '[rejected]', NEVER A WORDING: BOTH parentheticals occur against real remotes, and
//     which one prints depends on whether the local remote-tracking ref is stale. Measured
//     2026-10-06 (git 2.43.0) against the real origin, dry-run: a diverged push AND an ancestor
//     push with up-to-date tracking knowledge were each refused with '(non-fast-forward)' (as was
//     the captured bak-k3-p12 refusal from the 053 rehearsal), while a push whose remote ref had
//     moved PAST the local tracking ref was refused with '(fetch first)'. Both captures are
//     pasted on bead web-ai-showcase-2rj (the /tmp rehearsal paths are reclaimed). Only the
//     token '[rejected]' is decisive, so wording drift can never misclassify.
//   * THE ROW'S NEW SHA IS THE SECOND VALUE ('<target-old>..<my-new>'). Measured on this repo
//     (2026-10-05): pushing onto a strictly-behind ancestor printed '1a7c567..3e07c4c HEAD ->
//     <ref>' with HEAD = 3e07c4c. Asserting the FIRST value compares HEAD against the remote's
//     OLD sha and REFUSES EVERY LEGITIMATE LANDING.
//   * THREE-PART ASSERTION: the row exists, no '[rejected]' anywhere in the capture, and the
//     row's new sha is a prefix of `git rev-parse HEAD` — a statement about the TREE IN FRONT OF
//     YOU, not about a ref that merely differs from the target.
//   * PRECONDITIONS ASSERTED FIRST: worktree CLEAN (an uncommitted merge leaves HEAD at the
//     target and reproduces the 'Everything up-to-date' false green) and HEAD != target.
//   * STATUS CAPTURED UNPIPED INTO A FILE; assertions run against the file. git's stdout is
//     redirected straight into a log file (never a pipe), the file is the evidence artifact, and
//     every assertion reads the file back.
//   * READBACK AFTER PUSH: `git ls-remote origin refs/heads/<branch>` must equal HEAD, so
//     authorship is exact by construction — no other writer can produce my sha.
//
// DEVIATIONS from ~/merger-push-gated.sh (each deliberate, each tested):
//   d1. --dry-run-output FORCES --stub-push. The reference allowed a canned dry-run row to gate
//       a real push; nothing legitimate needs that, and a synthetic capture must never authorize
//       a real mutation.
//   d2. The NO-OP classification branch (2, via dry-run output) also runs the throwaway probe,
//       matching the bead's parenthetical "(throwaway-ref probe only)". The reference probed
//       only on the HEAD==target precondition path. The probe is always a --dry-run.
//   d3. Branch 2 vs branch 3 precedence follows the bead's literal ELIF order ('Everything
//       up-to-date' is checked before the row). The reference gave the row precedence. For a
//       single-ref push the two strings never co-occur in real git output, so this is
//       unobservable on real captures; on a synthetic double-match it is the conservative
//       choice (NO-OP, never push).
//   d4. Node instead of bash: same decision procedure; every git call carries a hard timeout.
//       A dry-run that times out or crashes is UNKNOWN rc=4 even if the capture holds a complete
//       row — a killed capture is never a pass (bounds discipline: 124/137/143-class outcomes
//       are fail-or-unknown).
//
// NOT DONE / NOT CLAIMED (honest limits):
//   * Cannot CREATE a remote branch: the target ref must already resolve on origin (rc=9). The
//     first push of a new branch stays a manual `git push -u`.
//   * Gates AUTHORSHIP, not content: it proves the tree you gated is the tree that landed. It
//     does not run this repo's review/gate stages — those run BEFORE this block, on the merged
//     tree, and their verdicts are unaffected by a green landing here.
//   * The dry-run -> push window is not a lock: a concurrent writer can still move the target.
//     The READBACK (rc=3) is the detector. A readback pass is exact (my sha, no other writer).
//     The rc=3 path cannot be exercised end-to-end without a concurrent remote writer; the
//     equality it asserts is unit/integration covered on the passing side.
//   * Wording-dependent by necessity: it matches git's stable push-status strings ('[rejected]',
//     'Everything up-to-date', the 'old..new HEAD -> ref' row). A git wording/locale change falls
//     through to UNKNOWN rc=4 (fail closed). Verified against this VM's git (real captures in
//     test/gated-landing.test.mjs) and the merger lane's real-remote captures. No locale is set.
//   * The probe ref 'gated-landing-probe-<epoch>' is only ever passed to `git push --dry-run`:
//     it proves the write path negotiates and publishes NOTHING. Note origin already carries
//     unrelated 'probe/…' branches from other lanes, so the rehearsal control is "no NEW probe
//     refs" (snapshot `git ls-remote origin 'refs/heads/*probe*'` before/after), not "none".
//
// USAGE:
//   node scripts/gated-landing.mjs --branch <branch> [--workdir <dir>] [--stub-push]
//                                  [--dry-run-output <file>] [--log-dir <dir>]
//     --stub-push             rehearse: decide and report rc, never mutate (push skipped)
//     --dry-run-output <file> parser rehearsal: classify a captured/synthetic output instead of
//                             contacting origin. FORCES --stub-push (d1). The HEAD-vs-target
//                             precondition is reported but not enforced (a real ahead-merge
//                             cannot be faked); cleanliness IS still enforced.
//   Env fallbacks preserved from the reference's rehearsed interface: STUB_PUSH=1, DRYRUN_OUT=<f>.
//   Logs default to a fresh mkdtemp dir OUTSIDE the repo (in-repo logs would dirty the very
//   clean-worktree precondition this script asserts); the log dir path is printed.
//
// EXIT CODES (distinct per branch so a caller cannot mistake one for another):
//    0  OK — three-part assertion held (pushed and read back equal, or WOULD push under stub)
//    1  REFUSED — '[rejected]' in the capture, OR the row's new sha is not my HEAD (the row is
//        about a different tree). Do NOT push; fetch + re-merge + RE-GATE.
//    2  NO-OP — nothing to land ('Everything up-to-date', or HEAD == target). Throwaway-ref
//        dry-run probe only; do NOT push.
//    3  READBACK FAIL — push ran but origin's ref != HEAD. STOP; do not retry blind.
//    4  UNKNOWN — unrecognised/truncated capture, a row whose new sha cannot be parsed, or a
//        dry-run that did not complete. Fail closed; do NOT push.
//    5  PRECONDITION — worktree not clean. Commit or stash; an uncommitted merge makes
//        'Everything up-to-date' a false green.
//    9  INFRA/USAGE — cannot resolve refs/heads/<branch> on origin (or HEAD, or the workdir);
//        refusing to guess. Also bad arguments.
import { spawnSync } from "node:child_process";
import {
  appendFileSync,
  closeSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  realpathSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

/** Anchored update-row regex — the classification pattern (branch 3). Exported for tests. */
export const UPDATE_ROW_RE = /^ *[0-9a-f]{4,}\.\.[0-9a-f]{4,} +HEAD -> /;
// Sha extraction from the MATCHED row only. git abbreviates to >= 7 hex chars. The SECOND capture
// is the row's NEW sha — the local/mine side (measured; see header).
const ROW_SHA_RE = /\b([0-9a-f]{7,})\.\.([0-9a-f]{7,})[ \t]+HEAD ->/;

const GIT_TIMEOUT_MS = 180_000; // hard bound on every git call (push/fetch/dry-run)
const LS_REMOTE_TIMEOUT_MS = 60_000;
const PROBE_TIMEOUT_MS = 120_000;
const SHA40 = /^[0-9a-f]{40}$/;

const say = (msg) => console.log(`[gated-landing] ${msg}`);
const readLines = (p) => readFileSync(p, "utf8").split(/\r?\n/).filter((l) => l.trim() !== "");

/**
 * Parse a captured dry-run into classification FACTS. The update row is matched LINE-ANCHORED
 * (git's leading indent allowed), so prose that merely mentions the row cannot satisfy it.
 */
export function classifyDryRunOutput(text) {
  const lines = text.split(/\r?\n/);
  const rejected = lines.some((l) => l.includes("[rejected]"));
  const upToDate = lines.some((l) => l.includes("Everything up-to-date"));
  let matchedLine = null;
  for (const line of lines) {
    if (UPDATE_ROW_RE.test(line)) {
      matchedLine = line;
      break;
    }
  }
  let rowOld = "";
  let rowNew = "";
  if (matchedLine !== null) {
    const m = matchedLine.match(ROW_SHA_RE);
    if (m) {
      rowOld = m[1];
      rowNew = m[2];
    }
  }
  return { rejected, upToDate, updateRow: matchedLine !== null, rowOld, rowNew, matchedLine };
}

/**
 * THE FOUR-BRANCH FORM, in this exact order. Pure: facts + my HEAD -> verdict. The empty-read
 * guard lives AFTER classification (inside branch 3), not before it: a refusal or an unrelated
 * output has no row and must be reported as REFUSED / UNRECOGNISED, not as an unreadable row —
 * getting that wrong keeps the exit code but mislabels the reason, which is how a log misleads
 * its next reader.
 */
export function decide(facts, headFull) {
  if (facts.rejected) return { code: 1, verdict: "REFUSED", reason: "rejected" };
  if (facts.upToDate) return { code: 2, verdict: "NO-OP", reason: "uptodate" };
  if (facts.updateRow) {
    if (!facts.rowNew) {
      return { code: 4, verdict: "UNKNOWN", reason: "row-new-sha-unreadable" };
    }
    if (!headFull.startsWith(facts.rowNew)) {
      return { code: 1, verdict: "REFUSED", reason: "row-about-a-different-tree" };
    }
    return { code: 0, verdict: "OK", reason: "asserted" };
  }
  return { code: 4, verdict: "UNKNOWN", reason: "unrecognised" };
}

/**
 * Run git. When `logPath` is given, stdout+stderr are REDIRECTED straight into that file (never a
 * pipe — no SIGPIPE/pipefail trap can eat the status or truncate the capture). `incomplete` marks
 * a timeout/crash (status null, a signal, or a spawn error): a bounds-discipline fail-or-unknown,
 * never a pass.
 */
function git(args, { cwd, logPath, timeoutMs = GIT_TIMEOUT_MS } = {}) {
  let fd = null;
  let stdio;
  if (logPath) {
    fd = openSync(logPath, "w");
    stdio = ["ignore", fd, fd];
  } else {
    stdio = ["ignore", "pipe", "pipe"];
  }
  try {
    const r = spawnSync("git", args, { cwd, stdio, timeout: timeoutMs, encoding: "utf8" });
    const incomplete = Boolean(r.error) || r.status === null || Boolean(r.signal);
    return { status: r.status, signal: r.signal, error: r.error, stdout: r.stdout, incomplete };
  } finally {
    if (fd !== null) closeSync(fd);
  }
}

/** Append a visible marker to a capture whose writer did not finish. */
function noteIncomplete(res, args, logPath) {
  if (!res.incomplete) return;
  appendFileSync(
    logPath,
    `\n[gated-landing] internal: git ${args[0]} did not complete (status=${res.status} ` +
      `signal=${res.signal} error=${res.error && res.error.message}) — capture may be truncated; ` +
      `never a pass\n`,
  );
}

/** Resolve refs/heads/<branch> on origin (stdout captured in-process; no shell pipe). */
function remoteSha(cwd, branch) {
  const r = git(["ls-remote", "origin", `refs/heads/${branch}`], {
    cwd,
    timeoutMs: LS_REMOTE_TIMEOUT_MS,
  });
  if (r.incomplete) return null;
  const sha = (r.stdout ?? "").trim().split("\t")[0] ?? "";
  return SHA40.test(sha) ? sha : null;
}

/**
 * Throwaway-ref probe: a --dry-run push to 'gated-landing-probe-<epoch>'. Proves the write path
 * negotiates; PUBLISHES NOTHING. Returns whether negotiation was observed.
 */
function probeWritePath(cwd, probelog) {
  const probeRef = `gated-landing-probe-${Date.now()}`;
  const p = git(["push", "--dry-run", "origin", `HEAD:refs/heads/${probeRef}`], {
    cwd,
    logPath: probelog,
    timeoutMs: PROBE_TIMEOUT_MS,
  });
  noteIncomplete(p, ["push"], probelog);
  say(`probe rc=${p.status} (unpiped) — throwaway ref, --dry-run only, publishes nothing`);
  for (const l of readLines(probelog)) say(`    | ${l}`);
  const negotiated = !p.incomplete && readFileSync(probelog, "utf8").includes("* [new branch]");
  say(
    negotiated
      ? "probe OK — write path negotiated"
      : "probe did NOT negotiate — treat the write path as UNPROVEN",
  );
  return negotiated;
}

/** Parse CLI args (flags first, then the reference's rehearsed env fallbacks). */
export function parseArgs(argv) {
  const o = {
    branch: null,
    workdir: process.cwd(),
    stubPush: false,
    dryRunOutput: null,
    logDir: null,
    help: false,
    stubForced: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--branch") o.branch = argv[++i];
    else if (a === "--workdir") o.workdir = argv[++i];
    else if (a === "--stub-push") o.stubPush = true;
    else if (a === "--dry-run-output") o.dryRunOutput = argv[++i];
    else if (a === "--log-dir") o.logDir = argv[++i];
    else if (a === "--help" || a === "-h") o.help = true;
    else return { ...o, error: `unknown argument: ${a}` };
  }
  if (o.help) return o;
  if (!o.stubPush && process.env.STUB_PUSH === "1") o.stubPush = true;
  if (!o.dryRunOutput && process.env.DRYRUN_OUT) o.dryRunOutput = process.env.DRYRUN_OUT;
  if (o.dryRunOutput) {
    // d1: a synthetic capture never authorizes a real mutation.
    if (!o.stubPush) {
      o.stubPush = true;
      o.stubForced = true;
    }
  }
  if (!o.branch) return { ...o, error: "--branch <branch> is required" };
  return o;
}

/**
 * The gated landing itself. Returns the process exit code (see table in the header).
 */
export function run(options) {
  const { branch, workdir: work, stubPush: stub } = options;
  const canned = options.dryRunOutput || null;
  const logDir = options.logDir ?? mkdtempSync(join(tmpdir(), "gated-landing-"));
  mkdirSync(logDir, { recursive: true });
  const drylog = join(logDir, "dry-run.log");
  const probelog = join(logDir, "probe.log");
  const pushlog = join(logDir, "push.log");
  say(
    `log dir ${logDir} (outside the repo: in-repo logs would dirty the clean-worktree precondition)`,
  );

  // ---------- 0. PRECONDITIONS — explicit, not assumed ----------
  const targetSha = remoteSha(work, branch);
  if (targetSha === null) {
    say(`cannot resolve refs/heads/${branch} on origin — refusing to guess`);
    say(
      "(this script cannot CREATE a remote branch; push the first copy manually, then land with it)",
    );
    return 9;
  }
  const head = git(["rev-parse", "HEAD"], { cwd: work });
  const headFull = (head.stdout ?? "").trim();
  if (head.incomplete || !SHA40.test(headFull)) {
    say(`cannot resolve HEAD in ${work} — refusing to guess`);
    return 9;
  }
  const dirty = git(["status", "--porcelain"], { cwd: work });
  const dirtyText = dirty.stdout ?? "";
  if (dirtyText.trim() !== "") {
    say("PRECONDITION FAILED: worktree is NOT clean:");
    for (const l of dirtyText.split(/\r?\n/).slice(0, 10)) if (l.trim()) say(`    | ${l}`);
    say("An UNCOMMITTED MERGE leaves HEAD at the target, so 'Everything up-to-date' would be read");
    say("as a false green. Commit or stash the merge first, then re-run.");
    return 5;
  }
  say(`precondition: worktree clean, HEAD=${headFull}, target(${branch})=${targetSha}`);

  if (headFull === targetSha && !canned) {
    say(
      "VERDICT: NOTHING TO LAND — HEAD already equals the target: my HEAD has nothing the target",
    );
    say(
      "lacks (you have not merged yet, or you are about to land nothing). Probing the write path",
    );
    say("with a throwaway ref; NOT pushing.");
    probeWritePath(work, probelog);
    return 2;
  }
  if (canned) {
    say(`REHEARSAL MODE: canned dry-run output from ${canned}`);
    say("HEAD-vs-target precondition reported, not enforced (a real ahead-merge cannot be faked);");
    say(`cleanliness WAS enforced. head=${headFull} target(${branch})=${targetSha}`);
    if (headFull === targetSha) {
      say("(note: HEAD currently EQUALS the target — the real path would NO-OP here)");
    }
    if (options.stubForced) {
      say("--dry-run-output forces --stub-push: a synthetic capture never authorizes a mutation");
    }
    if (!existsSync(canned)) {
      say(`canned dry-run file not found: ${canned}`);
      return 9;
    }
  }

  // ---------- 1. DRY RUN (real target), output to FILE, status UNPIPED ----------
  let dryRc = 0;
  let dryIncomplete = false;
  if (canned) {
    copyFileSync(canned, drylog);
  } else {
    const f = git(["fetch", "-q", "origin"], { cwd: work }); // best-effort; the dry-run below
    if (f.status !== 0) {
      say(`fetch rc=${f.status} (best-effort; the dry-run contacts origin itself)`);
    }
    const d = git(["push", "--dry-run", "origin", `HEAD:refs/heads/${branch}`], {
      cwd: work,
      logPath: drylog,
      timeoutMs: GIT_TIMEOUT_MS,
    });
    noteIncomplete(d, ["push"], drylog);
    dryRc = d.status;
    dryIncomplete = d.incomplete;
  }
  say(
    canned
      ? `dry-run: CANNED capture from ${canned} (nothing ran; there is no real rc)`
      : `dry-run rc=${dryRc} (unpiped; artifact below is the evidence, not the status)`,
  );
  for (const l of readLines(drylog)) say(`    | ${l}`);

  // Assert on the FILE (read back — the same bytes a reviewer would inspect).
  const facts = classifyDryRunOutput(readFileSync(drylog, "utf8"));
  let decision = decide(facts, headFull);
  if (!canned && dryIncomplete && decision.code === 0) {
    // d4: a killed capture is never a pass, even if it holds a complete-looking row.
    decision = { code: 4, verdict: "UNKNOWN", reason: "dry-run-did-not-complete" };
  }

  // ---------- 2. BRANCH: the refusal branch NEVER reaches the push ----------
  switch (decision.reason) {
    case "rejected":
      say("VERDICT: REJECTED — DO NOT PUSH.");
      say("recovery: FETCH + RE-MERGE + RE-GATE the merged tree. Never a blind retry.");
      return 1;
    case "uptodate":
      say("VERDICT: NOTHING TO LAND — MY HEAD HAS NOTHING THE TARGET LACKS.");
      say("Either you have not merged yet, or you are about to land nothing. Do not push.");
      probeWritePath(work, probelog); // d2: throwaway-ref probe only; do NOT push
      return 2;
    case "unrecognised":
      say(
        "VERDICT: UNRECOGNISED OUTPUT — DO NOT PUSH (fail-closed). No update row, no up-to-date,",
      );
      say("no rejection. A git wording change, a locale, or a truncated capture falls through to");
      say("here; the safe answer is always 'do not push'.");
      return 4;
    case "row-new-sha-unreadable":
      say("VERDICT: ROW MATCHED BUT ITS NEW SHA IS UNREADABLE — DO NOT PUSH (fail-closed).");
      say("An update row matched the anchored regex but no sha pair could be parsed from it.");
      return 4;
    case "row-about-a-different-tree":
      say("VERDICT: THE ROW IS ABOUT A DIFFERENT TREE — DO NOT PUSH.");
      say(`  row's new (mine) sha : ${facts.rowNew}`);
      say(`  row's old (target)  : ${facts.rowOld}`);
      say(`  my HEAD             : ${headFull}`);
      say("A worktree still merged from a previous landing looks exactly like this: the row is");
      say("genuine and contains no '[rejected]', but the sha it names is not what you gated.");
      return 1;
    case "dry-run-did-not-complete":
      say(
        "VERDICT: UNKNOWN — the dry-run did not complete (timeout/crash); a complete-looking row",
      );
      say("in a killed capture is still never a pass. DO NOT PUSH.");
      return 4;
    default:
      break; // asserted — fall through to the ACT
  }
  say(
    `assertion: row exists, no rejection, and row's new sha ${facts.rowNew} is my HEAD ` +
      `(old ${facts.rowOld})`,
  );

  // ---------- 3. ACT (reachable only from the asserted row) ----------
  if (stub) {
    say("VERDICT: OK — WOULD PUSH NOW (--stub-push: mutation skipped, nothing moved)");
    return 0;
  }
  say("VERDICT: OK — pushing now");
  const push = git(["push", "origin", `HEAD:refs/heads/${branch}`], {
    cwd: work,
    logPath: pushlog,
    timeoutMs: GIT_TIMEOUT_MS,
  });
  noteIncomplete(push, ["push"], pushlog);
  for (const l of readLines(pushlog)) say(`    | ${l}`);
  say(`push rc=${push.status}`);
  if (push.incomplete) {
    say("push DID NOT COMPLETE (timeout/crash) — the readback below decides truthfully");
  }

  // ---------- 4. READBACK: authorship is exact by construction ----------
  const remoteNow = remoteSha(work, branch);
  say(`readback: remote=${remoteNow}  local=${headFull}`);
  if (remoteNow === headFull) {
    say("READBACK PASS — landing verified by READ (no other writer can produce my sha)");
    return 0;
  }
  say("READBACK FAIL — the record and the ref disagree. STOP; do not retry blind.");
  return 3;
}

function main(argv) {
  const parsed = parseArgs(argv);
  if (parsed.error) {
    say(parsed.error);
    say(
      "usage: node scripts/gated-landing.mjs --branch <branch> [--workdir <dir>] " +
        "[--stub-push] [--dry-run-output <file>] [--log-dir <dir>] (see the file header)",
    );
    return 9;
  }
  if (parsed.help) {
    say(
      "usage: node scripts/gated-landing.mjs --branch <branch> [--workdir <dir>] " +
        "[--stub-push] [--dry-run-output <file>] [--log-dir <dir>] (see the file header)",
    );
    return 0;
  }
  return run(parsed);
}

const invokedAsScript = process.argv[1] &&
  import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
if (invokedAsScript) process.exitCode = main(process.argv.slice(2));
