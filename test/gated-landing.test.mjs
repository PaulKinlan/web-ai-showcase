// Tests for scripts/gated-landing.mjs (bead web-ai-showcase-053) — the atomic
// ASSERT -> BRANCH -> ACT landing block. These cover the TRAPS, not the happy path:
//   * a refusal that ALSO contains sha lines must be REFUSED (row-first logic would push);
//   * prose merely MENTIONING the update row must not satisfy the anchored regex;
//   * a truncated capture must be UNKNOWN, never OK;
//   * a row whose SECOND sha is not HEAD must fail (the first value is the remote's OLD sha);
//   * the refusal token is '[rejected]' under BOTH measured wordings ('(fetch first)' and
//     '(non-fast-forward)').
// Integration tests drive the real script against LOCAL-PATH bare remotes (no network), which is
// also how the '(non-fast-forward)' wording was measured — a local-path remote prints it even
// pre-fetch, while the real remote's divergence wording is '(fetch first)'. Both are REFUSED.
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  classifyDryRunOutput,
  decide,
  parseArgs,
  UPDATE_ROW_RE,
} from "../scripts/gated-landing.mjs";

const SCRIPT = new URL("../scripts/gated-landing.mjs", import.meta.url).pathname;
const HEAD = "3e07c4c0d5221607ec47ac8179ddd747c0786737"; // fake-but-well-formed stand-in HEAD

// ---------------------------------------------------------------------------
// Unit: the classifier and the four-branch decision, in the spec's exact order.
// ---------------------------------------------------------------------------

test("TRAP: a refusal that also contains sha lines is REFUSED rc=1, never a pass", () => {
  // Row-first logic sees the update row and classifies this REFUSAL AS A PASS, then pushes.
  const capture = [
    "To https://example.invalid/repo.git",
    "   1a7c567..3e07c4c  HEAD -> main", // a sha row IS present
    " ! [rejected]        HEAD -> main (fetch first)",
    "error: failed to push some refs",
  ].join("\n");
  const facts = classifyDryRunOutput(capture);
  assert.equal(facts.rejected, true);
  assert.equal(facts.updateRow, true); // the trap precondition: both matched
  const d = decide(facts, HEAD);
  assert.equal(d.code, 1);
  assert.equal(d.reason, "rejected");
});

test("refusal matches the TOKEN '[rejected]' under both measured wordings", () => {
  for (const wording of ["(fetch first)", "(non-fast-forward)"]) {
    const d = decide(
      classifyDryRunOutput(` ! [rejected]        HEAD -> main ${wording}\n`),
      HEAD,
    );
    assert.equal(d.code, 1, `wording ${wording} must be REFUSED`);
    assert.equal(d.reason, "rejected");
  }
});

test("rejection wins over EVERY other marker, including an up-to-date line", () => {
  const d = decide(
    classifyDryRunOutput(
      "Everything up-to-date\n   1a7c567..3e07c4c  HEAD -> main\n ! [rejected]  HEAD -> main\n",
    ),
    HEAD,
  );
  assert.equal(d.code, 1);
});

test("TRAP: prose merely MENTIONING the row does not satisfy the anchored regex", () => {
  const capture = [
    "hint: the row would have been 1a7c567..3e07c4c  HEAD -> main had it applied",
    "  see 1a7c567..3e07c4c  HEAD -> main in the log above",
    "note: '   1a7c567..3e07c4c  HEAD -> main' quoted with text before it",
  ].join("\n");
  const facts = classifyDryRunOutput(capture);
  assert.equal(facts.updateRow, false, "prose mention must not match the anchored row");
  const d = decide(facts, HEAD);
  assert.equal(d.code, 4, "prose-only capture is UNKNOWN, never OK");
  assert.equal(d.reason, "unrecognised");
});

test("TRAP: a truncated capture is UNKNOWN rc=4, not OK", () => {
  for (
    const capture of [
      "", // killed before any output
      "To https://example.invalid/repo.git\n", // banner only, no row
      "   1a7c567..3e0", // cut mid-row: no 'HEAD ->' terminator
      "   1a7c567..3e07c4c  HEAD", // cut before '-> ref'
    ]
  ) {
    const facts = classifyDryRunOutput(capture);
    assert.equal(facts.updateRow, false, JSON.stringify(capture));
    assert.equal(decide(facts, HEAD).code, 4, JSON.stringify(capture));
  }
});

test("TRAP: a row whose SECOND sha is not my HEAD is REFUSED (row about a different tree)", () => {
  const capture = "   1a7c567..fffffff  HEAD -> main\n";
  const facts = classifyDryRunOutput(capture);
  assert.equal(facts.updateRow, true);
  assert.equal(facts.rowNew, "fffffff");
  const d = decide(facts, HEAD);
  assert.equal(d.code, 1);
  assert.equal(d.reason, "row-about-a-different-tree");
});

test("TRAP: the row's NEW sha is the SECOND value — asserting the first refuses every landing", () => {
  // Measured 2026-10-05 on this repo: pushing onto a strictly-behind ancestor printed
  // '1a7c567..3e07c4c HEAD -> <ref>' with HEAD = 3e07c4c. The FIRST value is the remote's OLD
  // sha; the SECOND is mine. A correct gate compares HEAD against the second and PASSES here;
  // a naive first-value gate compares HEAD against 1a7c567 and refuses this legitimate landing.
  const capture = "   1a7c567..3e07c4c  HEAD -> main\n";
  const facts = classifyDryRunOutput(capture);
  assert.equal(facts.rowOld, "1a7c567");
  assert.equal(facts.rowNew, "3e07c4c");
  assert.equal(decide(facts, HEAD).code, 0, "second-value comparison must PASS the real landing");
  // And the mirror image — a row whose first value happens to equal my HEAD prefix — must FAIL,
  // proving the first value is never consulted as 'mine'.
  const swapped = "   3e07c4c..1a7c567  HEAD -> main\n";
  const d2 = decide(classifyDryRunOutput(swapped), HEAD);
  assert.equal(d2.code, 1, "first-value match must NOT pass");
  assert.equal(d2.reason, "row-about-a-different-tree");
});

test("happy path: real-shaped row, second sha a prefix of HEAD -> OK rc=0", () => {
  const capture = "To /remote\n   1a7c567..3e07c4c  HEAD -> main\n";
  const d = decide(classifyDryRunOutput(capture), HEAD);
  assert.deepEqual(d, { code: 0, verdict: "OK", reason: "asserted" });
});

test("'Everything up-to-date' is NO-OP rc=2", () => {
  const d = decide(classifyDryRunOutput("Everything up-to-date\n"), HEAD);
  assert.deepEqual(d, { code: 2, verdict: "NO-OP", reason: "uptodate" });
});

test("d3: a row AND up-to-date double-match is the conservative NO-OP, never a push", () => {
  // Unobservable in real single-ref git output; on a synthetic double-match the spec's literal
  // ELIF order (up-to-date before the row) is the branch that can never reach the push.
  const d = decide(
    classifyDryRunOutput("Everything up-to-date\n   1a7c567..3e07c4c  HEAD -> main\n"),
    HEAD,
  );
  assert.equal(d.code, 2);
});

test("anchored row with an unparseable sha pair is UNKNOWN rc=4, fail-closed", () => {
  // The classification regex allows {4,} hex (the spec's literal form); sha extraction requires
  // git's real abbreviation floor of 7. A short-sha row matches the row but yields no new sha.
  assert.ok(UPDATE_ROW_RE.test("   abcd..ef01  HEAD -> main"));
  const facts = classifyDryRunOutput("   abcd..ef01  HEAD -> main\n");
  assert.equal(facts.updateRow, true);
  assert.equal(facts.rowNew, "");
  const d = decide(facts, HEAD);
  assert.equal(d.code, 4);
  assert.equal(d.reason, "row-new-sha-unreadable");
});

test("the update-row regex is anchored: leading git indent allowed, leading text is not", () => {
  assert.ok(UPDATE_ROW_RE.test("   1a7c567..3e07c4c  HEAD -> main"));
  assert.ok(UPDATE_ROW_RE.test("1a7c567..3e07c4c HEAD -> main"));
  assert.ok(!UPDATE_ROW_RE.test("x 1a7c567..3e07c4c  HEAD -> main"));
  assert.ok(!UPDATE_ROW_RE.test("   1a7c567..3e07c4c  HEAD -> main extra".replace("HEAD", "HD"))); // sanity
});

test("parseArgs: --dry-run-output FORCES --stub-push (d1: a synthetic capture never mutates)", () => {
  const o = parseArgs(["--branch", "main", "--dry-run-output", "/tmp/x"]);
  assert.equal(o.stubPush, true);
  assert.equal(o.stubForced, true);
  const explicit = parseArgs(["--branch", "main", "--dry-run-output", "/tmp/x", "--stub-push"]);
  assert.equal(explicit.stubPush, true);
  assert.equal(explicit.stubForced, false);
});

test("parseArgs: --branch is required; unknown arguments are rejected", () => {
  assert.ok(parseArgs([]).error);
  assert.ok(parseArgs(["--branch", "main", "--frobnicate"]).error);
  assert.equal(parseArgs(["--branch", "main"]).error, undefined);
});

// ---------------------------------------------------------------------------
// Integration: the real script, as a subprocess, against LOCAL-PATH remotes.
// No network; every capture below is REAL git output unless labelled synthetic.
// ---------------------------------------------------------------------------

const hasGit = spawnSync("git", ["--version"], { encoding: "utf8" }).status === 0;

function sh(args, cwd) {
  const r = spawnSync("git", args, { cwd, encoding: "utf8" });
  assert.equal(r.status, 0, `git ${args.join(" ")}: ${r.stderr}`);
  return (r.stdout ?? "").trim();
}

/** A work repo with a local-path bare 'origin' and branch main published. */
function makeRepo(t) {
  const base = mkdtempSync(join(tmpdir(), "gated-landing-test-"));
  const work = join(base, "work");
  const remote = join(base, "origin.git");
  mkdirSync(work);
  sh(["init", "-b", "main"], work);
  sh(["config", "user.email", "test@example.invalid"], work);
  sh(["config", "user.name", "gated-landing-test"], work);
  sh(["config", "commit.gpgsign", "false"], work);
  writeFileSync(join(work, "f.txt"), "one\n");
  sh(["add", "f.txt"], work);
  sh(["commit", "-m", "one"], work);
  sh(["init", "--bare", remote], base);
  sh(["remote", "add", "origin", remote], work);
  sh(["push", "-u", "origin", "main"], work);
  const logDir = join(base, "logs");
  mkdirSync(logDir);
  t.after(() => spawnSync("rm", ["-rf", base]));
  return { base, work, remote, logDir };
}

function commitMore(work, name) {
  writeFileSync(join(work, name), `${name}\n`);
  sh(["add", name], work);
  sh(["commit", "-m", name], work);
  return sh(["rev-parse", "HEAD"], work);
}

function runScript(args) {
  return spawnSync("node", [SCRIPT, ...args], { encoding: "utf8", timeout: 120_000 });
}

test("integration REFUSED (REAL capture): diverged local-path remote, sha-ish refusal, no push", {
  skip: !hasGit,
}, (t) => {
  const { work, remote, logDir } = makeRepo(t);
  // A rival clone advances origin/main so our push is genuinely rejected.
  const rival = join(mkdtempSync(join(tmpdir(), "gated-landing-rival-")), "rival");
  sh(["clone", remote, rival], tmpdir());
  sh(["config", "user.email", "rival@example.invalid"], rival);
  sh(["config", "user.name", "rival"], rival);
  sh(["config", "commit.gpgsign", "false"], rival);
  commitMore(rival, "rival.txt");
  sh(["push", "origin", "main"], rival);
  const originSha = sh(["rev-parse", "HEAD"], rival);
  commitMore(work, "mine.txt"); // diverge
  const before = sh(["rev-parse", "refs/heads/main"], remote);

  const r = runScript(["--branch", "main", "--workdir", work, "--log-dir", logDir]);
  assert.equal(r.status, 1, r.stdout);
  const drylog = readFileSync(join(logDir, "dry-run.log"), "utf8");
  assert.ok(drylog.includes("[rejected]"), drylog);
  // Measured finding 1: the LOCAL-PATH remote's wording is '(non-fast-forward)' even pre-fetch;
  // the real remote's is '(fetch first)'. The token, never the wording, is what matched.
  assert.ok(drylog.includes("(non-fast-forward)"), drylog);
  assert.equal(sh(["rev-parse", "refs/heads/main"], remote), before, "refusal must not push");
  assert.equal(originSha, before);
});

test(
  "integration OK (REAL full path): dry-run row -> push -> readback equal",
  { skip: !hasGit },
  (t) => {
    const { work, remote, logDir } = makeRepo(t);
    const head = commitMore(work, "two.txt");
    const r = runScript(["--branch", "main", "--workdir", work, "--log-dir", logDir]);
    assert.equal(r.status, 0, r.stdout);
    const drylog = readFileSync(join(logDir, "dry-run.log"), "utf8");
    assert.ok(UPDATE_ROW_RE.test(drylog.split(/\r?\n/).find((l) => UPDATE_ROW_RE.test(l)) ?? ""));
    assert.equal(sh(["rev-parse", "refs/heads/main"], remote), head, "readback: origin == HEAD");
    assert.ok(r.stdout.includes("READBACK PASS"), r.stdout);
  },
);

test("integration NO-OP (REAL): HEAD == target -> rc=2, probe negotiates and publishes nothing", {
  skip: !hasGit,
}, (t) => {
  const { work, remote, logDir } = makeRepo(t);
  const refsBefore = sh(["for-each-ref", "--format=%(refname)"], remote);
  const r = runScript(["--branch", "main", "--workdir", work, "--log-dir", logDir]);
  assert.equal(r.status, 2, r.stdout);
  const probelog = readFileSync(join(logDir, "probe.log"), "utf8");
  assert.ok(probelog.includes("[new branch]"), probelog); // dry-run negotiated
  const refsAfter = sh(["for-each-ref", "--format=%(refname)"], remote);
  assert.equal(refsAfter, refsBefore, "the throwaway probe is --dry-run only: no ref published");
});

test(
  "integration PRECONDITION: a dirty worktree is rc=5 before any dry run",
  { skip: !hasGit },
  (t) => {
    const { work, logDir } = makeRepo(t);
    writeFileSync(join(work, "uncommitted.txt"), "dirty\n");
    const r = runScript(["--branch", "main", "--workdir", work, "--log-dir", logDir]);
    assert.equal(r.status, 5, r.stdout);
    assert.ok(r.stdout.includes("PRECONDITION FAILED"), r.stdout);
  },
);

test("integration UNKNOWN (synthetic): a truncated canned capture is rc=4, never OK", {
  skip: !hasGit,
}, (t) => {
  const { base, work, logDir } = makeRepo(t);
  const canned = join(base, "truncated.log");
  writeFileSync(canned, "To /remote\n   1a7c567..3e0"); // cut mid-row
  const r = runScript([
    "--branch",
    "main",
    "--workdir",
    work,
    "--log-dir",
    logDir,
    "--dry-run-output",
    canned,
  ]);
  assert.equal(r.status, 4, r.stdout);
  assert.ok(r.stdout.includes("forces --stub-push"), "d1 must be announced");
});

test("integration REFUSED (synthetic trap): refusal WITH a sha row is rc=1 and nothing moves", {
  skip: !hasGit,
}, (t) => {
  const { base, work, remote, logDir } = makeRepo(t);
  const head = sh(["rev-parse", "HEAD"], work);
  const canned = join(base, "refusal-with-row.log");
  writeFileSync(
    canned,
    `To /remote\n   1a7c567..${
      head.slice(0, 7)
    }  HEAD -> main\n ! [rejected]        HEAD -> main (non-fast-forward)\n`,
  );
  const before = sh(["rev-parse", "refs/heads/main"], remote);
  const r = runScript([
    "--branch",
    "main",
    "--workdir",
    work,
    "--log-dir",
    logDir,
    "--dry-run-output",
    canned,
  ]);
  assert.equal(r.status, 1, r.stdout);
  assert.equal(sh(["rev-parse", "refs/heads/main"], remote), before);
});

test("integration OK-stub (synthetic row): canned row gating is rc=0 and CANNOT mutate (d1)", {
  skip: !hasGit,
}, (t) => {
  const { base, work, remote, logDir } = makeRepo(t);
  const head = sh(["rev-parse", "HEAD"], work);
  const canned = join(base, "ok-row.log");
  writeFileSync(canned, `To /remote\n   1a7c567..${head.slice(0, 7)}  HEAD -> main\n`);
  const before = sh(["rev-parse", "refs/heads/main"], remote);
  const r = runScript([
    "--branch",
    "main",
    "--workdir",
    work,
    "--log-dir",
    logDir,
    "--dry-run-output",
    canned,
  ]);
  assert.equal(r.status, 0, r.stdout);
  assert.ok(r.stdout.includes("WOULD PUSH NOW"), r.stdout);
  assert.equal(sh(["rev-parse", "refs/heads/main"], remote), before, "forced stub: no mutation");
  // No push log may exist — the ACT never ran.
  assert.ok(!readdirSync(logDir).includes("push.log"));
});

test("integration INFRA: an unknown branch on origin is rc=9 (refusing to guess)", {
  skip: !hasGit,
}, (t) => {
  const { work, logDir } = makeRepo(t);
  const r = runScript(["--branch", "no-such-branch", "--workdir", work, "--log-dir", logDir]);
  assert.equal(r.status, 9, r.stdout);
});
