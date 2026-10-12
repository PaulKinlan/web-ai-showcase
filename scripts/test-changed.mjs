#!/usr/bin/env node
// Changed-test router for the tiered fleet gate (bead web-ai-showcase-360, ADR web-ai-showcase-cxw).
//
// WHY THIS EXISTS
// The full Node suite (`node --test "test/**/*.test.mjs"`) costs ~412s (architect measurement, 287).
// Running it for every implementer commit starves the fleet's single heavy-job slot, so implementers
// used to run only the ~8s static checks — which exercise NONE of the 47 unit test files. A lane
// could then ship a logic regression that only surfaced at the merger gate, minutes later, on a
// queue everyone shares.
//
// WHAT IT DOES
// 1. Resolves a base commit: `origin/main`, refreshed by default (`--no-fetch` to skip), and takes
//    `git merge-base <base> HEAD`. The diff is `origin/main...HEAD`, i.e. what this branch adds.
// 2. Parses `git diff --name-status -z -M`, not `--name-only`: with -M a rename is ONE record
//    carrying BOTH sides, so the old path — which every test that imported it still names — is
//    judged too. `--name-only` prints only the new path; that is the trap this keeps out.
// 3. Builds a dependency index over the module graph. For every file under test/, test-browser/,
//    scripts/ and lib/ it records static/dynamic import edges plus literal in-repo path tokens,
//    directory-scan roots (`readdirSync("models")`, `join(ROOT,"models",slug)`) and repo-wide scans
//    (`git ls-files`, `git grep`, `git archive HEAD`). Those are resolved TRANSITIVELY, so a test
//    that audits the repo through scripts/audit-model-currency.mjs is routed by what that script
//    reads, not by what the test file happens to mention.
// 4. Routes each changed path to the set of tests whose dependency closure contains it.
//
// FAIL-CLOSED, BY CONSTRUCTION (this is the whole point)
//   * a changed path with NO routed test triggers the fallback (default: the entire unit suite) —
//     never a green "nothing to run";
//   * a DELETED test file triggers the fallback (deleting a test removes coverage; the fast gate
//     must not go green on that alone);
//   * an unresolvable base, a failed `git diff`, or an unreadable config triggers the fallback;
//   * a changed scripts/test-changed.config.json forces the full fallback: a gate must not be
//     narrowed by a diff nobody re-verified;
//   * every mapped test is either RUN or reported as a declared-heavy deferral / browser-gated
//     exclusion — nothing is dropped silently;
//   * `--allow-unmapped` is the only way to run a partial set; it prints a loud WARN and is not for
//     CHECK_FAST_CMD.
// A path is "inert" (no test needed) only AFTER the dependency index has declined it, so a `.md`
// that a test reads is still routed: test/media-manifest.test.mjs sweeps every tracked text file
// with `git ls-files` for conflict markers, and test/runtime-pins.test.mjs asserts on AGENTS.md and
// CLAUDE.md — in this repo even documentation changes are genuinely test-bearing.
//
// TIERS (ADR cxw / bead ehf)
// Tests named in the config's `heavy` list are ROUTED but not RUN by the fast tier; each deferral is
// printed as `fast-gate: deferred-heavy:` so it is visible in the log and never silent. `--tier
// heavy` runs exactly the routed heavy tests (the separately scheduled tier with a durable verdict);
// `--tier all` (alias `--strict`) runs everything routed.
//
// TEST-BROWSER is excluded from the fleet gate until cgroup containment (bead jdz) is verified; a
// changed browser test is REPORTED (`fast-gate: browser-gated:`) and never absorbed silently.
//
// NO CHROME, NO NETWORK beyond `git fetch`, NO `npm ci` (fleet-deps hard-links node_modules).
//
// EVIDENCE MARKERS (why they exist — bead web-ai-showcase-75ym)
// The coverage step of the fleet gate (remote/check-coverage.py) reads only this log: it cannot see
// the dependency graph below, so it attributes coverage from filename stems and `import` statements.
// For a changed DATA file — a fixture a test reads rather than imports — that guess fails, and the
// report warned "1 unmapped changed file" on a run where the suite covering it HAD in fact executed
// (94s: test/fixtures/safe-url-worker.mjs, 7 suites / 97 tests). The router knows the mapping, so it
// states it machine-readably, one line per changed path:
//   fast-gate: mapped: <changed path> -> <suite> [<suite> ...]
// `mapped:` is evidence, never a decision: it is printed only for the paths this same code already
// routes to a SPECIFIC test (the `specific.size > 0` branch below), never for a sweep-only or
// unmapped path, so it cannot turn an unrouted change green — those still print `unmapped:` and
// still force the fallback. A consumer that requires the named suite to have EXECUTED (`completed:`)
// keeps fail-closed semantics; the marker alone asserts mapping, not execution.
//
// Usage (the CHECK_FAST_CMD shape ADR cxw describes):
//   fleet-deps && node scripts/test-changed.mjs
// Inspection: --list, --json, --explain. See --help.
//
// Config (optional, scripts/test-changed.config.json; every key optional, `_note` keys are ignored):
//   {
//     "fallback": { "mode": "all", "list": [] },   // "all" (default) | "list" (bounded core set)
//     "heavy": ["test/slow.test.mjs"],             // routed but deferred by the fast tier
//     "browser": ["public/**", "**/*.css"],        // browser-facing: reported BROWSER_UNVERIFIED
//     "inert": ["docs/**"],                        // extra no-test-needed globs (last resort only)
//     "map": { "models/**": ["test/x.test.mjs"] }, // extra routing overrides
//     "external": [{ "globs": ["server.ts"], "cmd": ["deno", "task", "test:server"] }]
//   }
// An `external` rule's command is EXECUTED by this gate, so it is verification rather than a
// promise: it exists for paths another runner owns (the Deno server proxy tests) that no Node test
// covers. A declared-but-unrun claim would be the green skip this router exists to prevent — there
// is deliberately no flag that skips a matched external rule.

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const UNIT_TEST_DIR = "test";
export const BROWSER_TEST_DIR = "test-browser";
export const TEST_SUFFIX = ".test.mjs";
export const CONFIG_REL = "scripts/test-changed.config.json";

// Directories scanned to build the module graph. models/ is deliberately absent: it holds thousands
// of worker/demo files that are DATA to these tests (read as text, or scanned as directories), never
// modules the suite imports. Reaching their contents is the dir-ref mechanism's job.
const GRAPH_DIRS = ["test", "test-browser", "scripts", "lib"];
const GRAPH_FILE_RE = /\.(?:mjs|js|cjs|ts)$/;
const SKIP_DIRS = new Set(["node_modules", ".git", ".beads", "vendor", "dist", "build"]);
const MAX_SOURCE_BYTES = 2 * 1024 * 1024;

const FETCH_TIMEOUT_MS = 180_000;
const DEPS_TIMEOUT_MS = 600_000;

// Documentation-ish paths that count as "inert" ONLY after the dependency index declines them.
const INERT_EXT = new Set([".md", ".txt", ".rst"]);
const INERT_NAMES = new Set(["LICENSE", "LICENCE", "NOTICE", ".gitignore", ".gitattributes"]);
const INERT_PREFIXES = [".github/", ".beads/", "docs/"];

// ── git ────────────────────────────────────────────────────────────────────────────────────────

/** Run git in a repo. Never throws; returns { ok, out, err, status }. */
export function gitOut(repoDir, args, opts = {}) {
  try {
    const out = execFileSync("git", ["-C", repoDir, ...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      maxBuffer: 64 * 1024 * 1024,
      timeout: opts.timeout ?? 120_000,
      env: { ...process.env, GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0" },
    });
    return { ok: true, out, err: "", status: 0 };
  } catch (e) {
    return {
      ok: false,
      out: String(e.stdout ?? ""),
      err: String(e.stderr ?? e.message ?? "").trim(),
      status: e.status ?? null,
    };
  }
}

/** Repo root for a directory, or null when it is not inside a git work tree. */
export function resolveRepoRoot(dir) {
  const r = gitOut(dir, ["rev-parse", "--show-toplevel"]);
  return r.ok && r.out.trim() ? r.out.trim() : null;
}

export function toPosix(p) {
  return String(p).split(sep).join("/").replace(/^\.\//, "");
}

/** Resolve a token against the directory of `fromRel`, POSIX-normalised, or null if it escapes. */
export function resolveRel(fromRel, token) {
  const joined = toPosix(normalize(join(dirname(fromRel), token)));
  if (joined.startsWith("../") || joined === ".." || joined.startsWith("/")) return null;
  return joined;
}

/**
 * Fresh base commit for the branch diff.
 * Order: explicit --base, freshly fetched origin/main, local origin/main, main, origin/HEAD.
 * Returns { ref, sha, refreshed, warnings[], error }; `error` means "unknown base" (fail closed).
 */
export function resolveBase(
  repoDir,
  { base, fetch = true, fetchTimeoutMs = FETCH_TIMEOUT_MS } = {},
) {
  const warnings = [];
  const revParse = (ref) => {
    const r = gitOut(repoDir, ["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]);
    return r.ok && r.out.trim() ? r.out.trim() : null;
  };
  if (base) {
    const sha = revParse(base);
    return sha ? { ref: base, sha, refreshed: false, warnings, error: null } : {
      ref: base,
      sha: null,
      refreshed: false,
      warnings,
      error: `base ref ${base} does not resolve to a commit`,
    };
  }
  let refreshed = false;
  if (fetch) {
    // `git fetch origin main` (rather than a bare fetch) keeps this bounded on repos with many refs.
    let r = gitOut(repoDir, ["fetch", "--quiet", "--no-tags", "origin", "main"], {
      timeout: fetchTimeoutMs,
    });
    if (!r.ok) {
      r = gitOut(repoDir, ["fetch", "--quiet", "--no-tags", "origin"], { timeout: fetchTimeoutMs });
    }
    if (r.ok) refreshed = true;
    else {warnings.push(
        `origin/main could not be refreshed (${r.err || "fetch failed"}); using the local ref`,
      );}
  }
  for (const ref of ["origin/main", "main", "origin/HEAD"]) {
    const sha = revParse(ref);
    if (sha) return { ref, sha, refreshed, warnings, error: null };
  }
  return {
    ref: null,
    sha: null,
    refreshed,
    warnings,
    error: "no base ref (origin/main, main or origin/HEAD) resolves; bring the repo up to date",
  };
}

/** `git merge-base <base> HEAD`; `error` means "cannot tell what changed" (fail closed). */
export function mergeBase(repoDir, baseRef, head = "HEAD") {
  const r = gitOut(repoDir, ["merge-base", baseRef, head]);
  if (r.ok && r.out.trim()) return { sha: r.out.trim(), error: null };
  // HEAD already equal to base has no merge-base output on some histories; rev-parse covers that.
  const headSha = gitOut(repoDir, ["rev-parse", head]);
  const baseSha = gitOut(repoDir, ["rev-parse", baseRef]);
  if (headSha.ok && baseSha.ok && headSha.out.trim() && headSha.out.trim() === baseSha.out.trim()) {
    return { sha: headSha.out.trim(), error: null };
  }
  return {
    sha: null,
    error: `git merge-base ${baseRef} ${head} failed: ${r.err || "no common ancestor"}`,
  };
}

/**
 * Parse `git diff --name-status -z -M` output.
 * Records are `STATUS\0path\0` and, for R/C, `STATUS\0old\0new\0`.
 * Returns [{ status, path, oldPath }] where `path` is the NEW path (== oldPath for A/M/D).
 */
export function parseNameStatusZ(raw) {
  const parts = String(raw).split("\0");
  const out = [];
  for (let i = 0; i < parts.length;) {
    const status = parts[i++];
    if (!status) continue;
    const code = status[0];
    if (code === "R" || code === "C") {
      const oldPath = parts[i++];
      const newPath = parts[i++];
      if (oldPath && newPath) out.push({ status: code, path: newPath, oldPath });
      continue;
    }
    const p = parts[i++];
    if (p) out.push({ status: code, path: p, oldPath: p });
  }
  return out;
}

/**
 * Changed paths against the merge base, with renames carrying both sides.
 *
 * The WORKING TREE is included by default (`--committed-only` opts out): a lane that edits files and
 * runs the gate before committing must not be told "nothing changed" — an empty diff there is a
 * clean green run for code nobody tested. Untracked files count as additions.
 */
export function changedPaths(repoDir, baseSha, head = "HEAD", { includeWorktree = true } = {}) {
  const merged = new Map();
  const add = (entries) => {
    for (const e of entries) merged.set(`${e.status}\0${e.oldPath}\0${e.path}`, e);
  };
  const committed = gitOut(repoDir, ["diff", "--name-status", "-z", "-M", baseSha, head]);
  if (!committed.ok) {
    return { entries: [], error: `git diff ${baseSha}..${head} failed: ${committed.err}` };
  }
  add(parseNameStatusZ(committed.out));
  if (includeWorktree) {
    const dirty = gitOut(repoDir, ["diff", "--name-status", "-z", "-M", head]);
    if (dirty.ok) add(parseNameStatusZ(dirty.out));
    const untracked = gitOut(repoDir, ["ls-files", "--others", "--exclude-standard", "-z"]);
    if (untracked.ok) {
      add(
        untracked.out
          .split("\0")
          .filter(Boolean)
          .map((path) => ({ status: "A", path, oldPath: path })),
      );
    }
  }
  return { entries: [...merged.values()], error: null };
}

// ── source scanning ───────────────────────────────────────────────────────────────────────────

/** Remove comments so call-context detection is not fooled by prose. String literals survive. */
export function stripComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === '"' || c === "'" || c === "`") {
      out += c;
      i++;
      while (i < n) {
        const s = src[i];
        if (s === "\\") {
          out += s + (src[i + 1] ?? "");
          i += 2;
          continue;
        }
        out += s;
        i++;
        if (s === c) break;
        if (s === "\n" && c !== "`") break; // unterminated single-line string
      }
      continue;
    }
    if (c === "/" && d === "/") {
      while (i < n && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && d === "*") {
      i += 2;
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

const QUOTED_RE = /(["'`])((?:\\.|(?!\1)[^\\\n])*)\1/g;
const WALK_CALL_RE =
  /\b(?:readdirSync|readdir|opendirSync|globSync|glob|walkSync|walk|readdirp)\s*\(([\s\S]{0,240}?)\)/g;
const EXEC_CALL_RE = /\b(?:execSync|execFileSync|spawnSync|exec|spawn)\s*\(([\s\S]{0,400}?)\)/g;
const PATH_CALL_BEFORE_RE =
  /\b(?:join|resolve|readFileSync|readFile|readdirSync|readdir|existsSync|statSync|lstatSync|mkdirSync|writeFileSync|access|accessSync|openSync|cpSync|copyFileSync|globSync|realpathSync)\s*\(\s*[^()]{0,120}$/;
const REL_IMPORT_RE =
  /\bfrom\s*(["'])([^"'\n]+)\1|\bimport\s*\(\s*(["'])([^"'\n]+)\3\s*\)|\brequire\s*\(\s*(["'])([^"'\n]+)\5\s*\)/g;

function isUsableToken(tok) {
  if (!tok || tok.length > 256) return false;
  if (/\s/.test(tok)) return false;
  if (tok.startsWith("/") || tok.startsWith("~") || tok.startsWith("#")) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(tok)) return false; // http:, data:, node:, file:
  return true;
}

/** Trim a quoted token to its path part (drop ?query / #fragment and trailing punctuation). */
function cleanToken(raw) {
  let tok = raw.replace(/\\(.)/g, "$1").trim();
  const cut = tok.search(/[?#]/);
  if (cut >= 0) tok = tok.slice(0, cut);
  return tok.replace(/[),;]+$/, "");
}

const isDirAt = (repoDir, rel) => {
  try {
    return statSync(join(repoDir, rel)).isDirectory();
  } catch {
    return false;
  }
};
const isFileAt = (repoDir, rel) => {
  try {
    return statSync(join(repoDir, rel)).isFile();
  } catch {
    return false;
  }
};

/**
 * Scan one source file for the repo paths it depends on.
 * Returns { imports, fileRefs, dirRefs, repoWide } — all repo-relative POSIX paths.
 * Literal references are taken from the RAW text (comments included) so prose cannot hide a
 * dependency; call-context detection uses comment-stripped text so prose cannot invent one.
 */
export function scanSourceRefs(repoDir, fileRel, topLevel) {
  const refs = { imports: [], fileRefs: [], dirRefs: [], repoWide: false };
  const abs = join(repoDir, fileRel);
  let raw;
  try {
    if (statSync(abs).size > MAX_SOURCE_BYTES) return refs;
    raw = readFileSync(abs, "utf8");
  } catch {
    return refs;
  }
  const stripped = stripComments(raw);
  const files = new Set();
  const dirs = new Set();
  const imports = new Set();

  const resolveSpecifier = (spec) => {
    if (!spec.startsWith("./") && !spec.startsWith("../")) return null;
    const rel = resolveRel(fileRel, spec);
    if (!rel) return null;
    const candidates = extname(rel)
      ? [rel]
      : [rel, `${rel}.mjs`, `${rel}.js`, `${rel}/index.mjs`, `${rel}/index.js`];
    for (const cand of candidates) if (isFileAt(repoDir, cand)) return cand;
    return rel; // deleted/moved module: still a real edge
  };

  const addResolved = (rawTok, { fromScan = false, pathContext = false } = {}) => {
    const tok = cleanToken(rawTok);
    if (!isUsableToken(tok)) return;
    const relative = tok.startsWith("./") || tok.startsWith("../");
    const target = relative ? resolveRel(fileRel, tok) : toPosix(normalize(tok));
    if (!target || target.startsWith("../")) return;
    if (!relative) {
      if (!topLevel.has(target.split("/")[0]) && !topLevel.has(target)) return;
      // A BARE top-level name with no extension is only a path when something is using it as one.
      // Without this, a stray "lib" archive-extension string in scripts/audio-provenance-lib.mjs
      // becomes a scan root and routes every lib/** change to a test that never reads it.
      const bare = !target.includes("/") && !extname(target);
      if (bare && !fromScan && !pathContext) return;
    }
    const bareTopLevel = target.split("/").length === 1 && topLevel.has(target) && !extname(target);
    if (tok.endsWith("/") || bareTopLevel || isDirAt(repoDir, target)) {
      dirs.add(target.replace(/\/$/, ""));
      return;
    }
    files.add(target);
    if (!extname(target)) {
      const resolved = resolveSpecifier(`./${target}`);
      if (resolved) files.add(resolved);
    }
  };

  // Pass 1 — every quoted token.
  for (const m of raw.matchAll(QUOTED_RE)) {
    const before = raw.slice(Math.max(0, m.index - 140), m.index);
    addResolved(m[2], { pathContext: PATH_CALL_BEFORE_RE.test(before) });
  }

  // Pass 2 — scan roots and repo-wide scans (comment-stripped, so prose cannot fake them).
  for (const m of stripped.matchAll(WALK_CALL_RE)) {
    for (const q of m[1].matchAll(QUOTED_RE)) addResolved(q[2], { fromScan: true });
  }
  for (const m of stripped.matchAll(EXEC_CALL_RE)) {
    const args = m[1];
    // Both spellings count: execSync("git ls-files") and execFileSync("git", ["ls-files"]). A
    // repo-wide reader is a real dependency of every tracked file, so missing one would drop a test
    // from a change it does read.
    if (/\bgit\b[^)]{0,80}?\b(?:ls-files|ls-tree|archive|grep)\b/.test(args)) refs.repoWide = true;
    for (const q of args.matchAll(QUOTED_RE)) addResolved(q[2], { fromScan: true });
  }

  // Pass 3 — module edges (relative specifiers only; bare specifiers are node_modules).
  for (const m of stripped.matchAll(REL_IMPORT_RE)) {
    const spec = m[2] ?? m[4] ?? m[6];
    if (!spec) continue;
    const resolved = resolveSpecifier(spec);
    if (resolved) {
      imports.add(resolved);
      files.add(resolved);
    }
  }

  refs.fileRefs = [...files].sort();
  refs.dirRefs = [...dirs].sort();
  refs.imports = [...imports].sort();
  return refs;
}

/** Recursively list graph source files (repo-relative POSIX paths). */
export function listGraphFiles(repoDir) {
  const out = [];
  const walk = (relDir) => {
    let entries;
    try {
      entries = readdirSync(join(repoDir, relDir), { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (e.name.startsWith(".") || SKIP_DIRS.has(e.name)) continue;
      const rel = relDir ? `${relDir}/${e.name}` : e.name;
      if (e.isDirectory()) walk(rel);
      else if (GRAPH_FILE_RE.test(e.name)) out.push(rel);
    }
  };
  for (const d of GRAPH_DIRS) walk(d);
  for (const e of readdirSync(repoDir, { withFileTypes: true })) {
    if (e.isFile() && GRAPH_FILE_RE.test(e.name)) out.push(e.name);
  }
  return out.sort();
}

/** All unit test files on disk (test/**\/*.test.mjs, excluding test/lib and test/fixtures). */
export function listUnitTests(repoDir) {
  const out = [];
  const walk = (relDir) => {
    let entries;
    try {
      entries = readdirSync(join(repoDir, relDir), { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (e.name.startsWith(".")) continue;
      const rel = `${relDir}/${e.name}`;
      if (e.isDirectory()) {
        if (rel === "test/lib" || rel === "test/fixtures") continue;
        walk(rel);
      } else if (e.name.endsWith(TEST_SUFFIX)) out.push(rel);
    }
  };
  walk(UNIT_TEST_DIR);
  return out.sort();
}

/** Shape test for a unit test path, including files deleted by this commit. */
export function isUnitTestPath(p) {
  return p.startsWith(`${UNIT_TEST_DIR}/`) && p.endsWith(TEST_SUFFIX) && !p.startsWith("test/lib/");
}

export function isBrowserTestPath(p) {
  return p === BROWSER_TEST_DIR || p.startsWith(`${BROWSER_TEST_DIR}/`);
}

function isInertPath(p, config) {
  if (INERT_NAMES.has(p)) return true;
  if (INERT_PREFIXES.some((pre) => p.startsWith(pre))) return true;
  if (INERT_EXT.has(extname(p))) return true;
  return (config?.inert ?? []).some((glob) => globMatch(p, glob));
}

// ── dependency index ──────────────────────────────────────────────────────────────────────────

/** Tiny glob matcher: `*` within a segment, `**` across segments; a plain dir also covers its subtree. */
export function globMatch(path, pattern) {
  const p = toPosix(path);
  const pat = toPosix(pattern);
  const re = new RegExp(
    `^${
      pat
        .split("/")
        .map((seg) =>
          seg === "**"
            ? ".*"
            : seg.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]*").replace(
              /\?/g,
              "[^/]",
            )
        )
        .join("/")
    }$`,
  );
  if (re.test(p)) return true;
  if (!pat.includes("*") && (p === pat || p.startsWith(`${pat}/`))) return true;
  return false;
}

/**
 * Build the module graph and its inverted dependency index.
 * Returns { nodes, exact, dirEntries, repoWideTests, allUnitTests, topLevel }.
 */
export function buildIndex(repoDir) {
  const topLevel = new Set(
    readdirSync(repoDir, { withFileTypes: true })
      .filter((e) => !SKIP_DIRS.has(e.name))
      .map((e) => e.name),
  );
  const nodes = new Map();
  for (const rel of listGraphFiles(repoDir)) {
    nodes.set(rel, scanSourceRefs(repoDir, rel, topLevel));
  }
  const allUnitTests = listUnitTests(repoDir);

  const closureMemo = new Map();
  const closure = (rel) => {
    if (closureMemo.has(rel)) return closureMemo.get(rel);
    const acc = { files: new Set(), dirs: new Set(), repoWide: false };
    closureMemo.set(rel, acc); // pre-set so import cycles terminate
    const node = nodes.get(rel);
    if (!node) return acc;
    if (node.repoWide) acc.repoWide = true;
    for (const f of node.fileRefs) acc.files.add(f);
    for (const d of node.dirRefs) acc.dirs.add(d);
    for (const imp of node.imports) {
      const sub = closure(imp);
      sub.files.forEach((x) => acc.files.add(x));
      sub.dirs.forEach((x) => acc.dirs.add(x));
      if (sub.repoWide) acc.repoWide = true;
    }
    return acc;
  };

  const exact = new Map(); // repo path -> Set<test>
  const dirEntries = []; // { dir, test }
  const repoWideTests = new Set();
  for (const t of allUnitTests) {
    const deps = closure(t);
    for (const f of deps.files) {
      if (!exact.has(f)) exact.set(f, new Set());
      exact.get(f).add(t);
    }
    for (const d of deps.dirs) dirEntries.push({ dir: d, test: t });
    if (deps.repoWide) repoWideTests.add(t);
  }
  return { nodes, exact, dirEntries, repoWideTests, allUnitTests, topLevel };
}

/**
 * Tests that depend on a changed path, split by strength (bead web-ai-showcase-360).
 *
 * `specific` is a real dependency: a test imports this module, names this file, or enumerates the
 * directory that contains it. `sweep` is a repo-wide sweep (`git ls-files`, `git archive HEAD`): it
 * DOES read the path, so it always runs, but on its own it never proves a new module works. Ignoring
 * that distinction would let any brand-new file be "verified" by a conflict-marker sweep and never
 * reach the fallback.
 */
export function testsForPath(index, p, config = {}) {
  const specific = new Set();
  const sweep = new Set();
  const ex = index.exact.get(p);
  if (ex) ex.forEach((t) => specific.add(t));
  for (const { dir, test } of index.dirEntries) {
    if (p === dir || p.startsWith(`${dir}/`)) specific.add(test);
  }
  for (const [pattern, tests] of Object.entries(config.map ?? {})) {
    if (globMatch(p, pattern)) { for (const t of tests) specific.add(t); }
  }
  index.repoWideTests.forEach((t) => sweep.add(t));
  return { specific, sweep };
}

// ── routing ───────────────────────────────────────────────────────────────────────────────────

export const DEFAULT_FALLBACK = { mode: "all", list: [] };

/**
 * Route changed paths to tests. Pure: no I/O, so the fail-closed rules are directly unit-testable.
 */
export function routeChanges({
  entries,
  index,
  config = {},
  tier = "fast",
  unknownBase = null,
  forceAll = false,
  allowUnmapped = false,
}) {
  const fallbackCfg = { ...DEFAULT_FALLBACK, ...(config.fallback ?? {}) };
  const heavy = new Set(config.heavy ?? []);
  const browserGlobs = Array.isArray(config.browser) ? config.browser : [];
  const run = new Set();
  const unmapped = [];
  const deletedTests = [];
  const renamedTests = [];
  const browserChanged = [];
  const browserUnverified = [];
  const inert = [];
  const reasons = [];
  // Changed path -> the SPECIFIC suites that cover it (75ym). Evidence for the coverage reporter;
  // never consulted by any routing decision below.
  const mapped = [];
  const externalRules = Array.isArray(config.external) ? config.external : [];
  const external = new Map(); // label -> { label, cmd, timeoutMs, paths:Set }

  for (const entry of entries) {
    const renamed = entry.status === "R" || entry.status === "C";
    const targets = renamed ? [entry.oldPath, entry.path] : [entry.path];
    for (const target of targets) {
      // The OLD side of a rename no longer exists: running it would hand `node --test` a path it
      // silently ignores, and claiming it as `completed:` would be a false record. The new side runs.
      const isOldSide = renamed && target === entry.oldPath;
      const unitTest = isUnitTestPath(target);
      const { specific, sweep } = testsForPath(index, target, config);
      if (unitTest && !isOldSide && entry.status !== "D") specific.add(target);
      sweep.forEach((t) => run.add(t));
      specific.forEach((t) => run.add(t));
      // A path another runner verifies is covered by a command this invocation RUNS, never by a
      // declaration alone — an unrunnable claim would be exactly the green skip this gate forbids.
      let externallyCovered = false;
      for (const rule of externalRules) {
        if (!(rule.globs ?? []).some((g) => globMatch(target, g))) continue;
        externallyCovered = true;
        if (!external.has(rule.label)) {
          external.set(rule.label, {
            label: rule.label,
            cmd: rule.cmd,
            timeoutMs: rule.timeoutMs,
            paths: new Set(),
          });
        }
        external.get(rule.label).paths.add(target);
      }
      if (externallyCovered) {
        reasons.push({
          path: target,
          reason: `covered by external runner(s): ${[...external.keys()].join(", ")}`,
        });
      }
      // Browser-facing paths have no gate while cgroup containment is blocked (bead jdz): say so on
      // every run instead of letting a fast-tier PASS read as browser verification.
      if (browserGlobs.some((g) => globMatch(target, g))) {
        browserUnverified.push(target);
        reasons.push({
          path: target,
          reason: "browser-facing: BROWSER_UNVERIFIED (Chrome gate blocked by web-ai-showcase-jdz)",
        });
      }
      if (unitTest && entry.status === "D") {
        deletedTests.push(target);
        reasons.push({ path: target, reason: "deleted unit test file (forces the fallback)" });
        continue;
      }
      if (unitTest && isOldSide) {
        renamedTests.push({ from: entry.oldPath, to: entry.path });
        reasons.push({
          path: target,
          reason: `renamed to ${entry.path}: the new path runs, the old path no longer exists`,
        });
        continue;
      }
      if (unitTest) {
        reasons.push({ path: target, reason: "changed unit test file" });
        continue;
      }
      if (isBrowserTestPath(target)) {
        browserChanged.push(target);
        reasons.push({
          path: target,
          reason: "browser test: Chrome-gated, excluded from the fleet gate",
        });
        continue;
      }
      if (specific.size > 0) {
        mapped.push({ path: target, suites: [...specific].sort() });
        reasons.push({ path: target, reason: `mapped to ${[...specific].sort().join(", ")}` });
        continue;
      }
      if (isInertPath(target, config)) {
        inert.push(target);
        reasons.push({
          path: target,
          reason: "inert (documentation/asset with no test dependency)",
        });
        continue;
      }
      if (externallyCovered) continue;
      unmapped.push(target);
      reasons.push({
        path: target,
        reason: sweep.size > 0
          ? `NO specific test (only repo-wide sweep: ${[...sweep].sort().join(", ")})`
          : "NO mapped test",
      });
    }
  }

  // Anything here triggers the fallback: a change the fast gate cannot route is never skipped.
  // `code` is the machine-readable half (coord: markers must be greppable), `prose` the human half.
  let code = null;
  let paths = [];
  if (entries.some((e) => e.path === CONFIG_REL)) {
    // A gate-config diff is the strongest signal: nobody re-verified the new policy.
    code = "config-change";
    paths = [CONFIG_REL];
  } else if (deletedTests.length > 0) {
    code = "deleted-test";
    paths = deletedTests;
  } else if (unmapped.length > 0) {
    code = "unmapped";
    paths = unmapped;
  } else if (unknownBase) {
    code = "unknown-base";
  } else if (forceAll) {
    code = "all-flag";
  }

  let fallback = null;
  if (code && !allowUnmapped) {
    let mode = fallbackCfg.mode === "list" ? "list" : "all";
    let list = [];
    if (mode === "list") {
      list = (Array.isArray(fallbackCfg.list) ? fallbackCfg.list : []).filter((t) =>
        typeof t === "string"
      );
      const missing = list.filter((t) => !index.allUnitTests.includes(t));
      if (list.length === 0 || missing.length > 0) {
        reasons.push({
          path: CONFIG_REL,
          reason: `fallback list is empty or names missing test(s) [${
            missing.join(", ")
          }]; widening to the full unit suite`,
        });
        mode = "all";
      }
    }
    fallback = {
      mode,
      reason: code,
      paths,
      detail: unknownBase ?? null,
      list: mode === "list" ? list : [],
    };
    if (mode === "all") index.allUnitTests.forEach((t) => run.add(t));
    else list.forEach((t) => run.add(t));
  }

  // Tiering (ADR cxw / bead ehf): heavy tests are routed and reported, but the fast tier defers them.
  const routed = [...run].sort();
  let toRun = routed;
  let deferred = [];
  if (tier === "fast") {
    deferred = routed.filter((t) => heavy.has(t));
    toRun = routed.filter((t) => !heavy.has(t));
  } else if (tier === "heavy") {
    toRun = routed.filter((t) => heavy.has(t));
    deferred = routed.filter((t) => !heavy.has(t));
  }

  // A path is only reported as unmapped when this decision did not cover the whole unit suite.
  const decisionCoversAll = Boolean(fallback && fallback.mode === "all" && tier !== "heavy");
  return {
    run: toRun,
    deferred,
    routed,
    unmapped: decisionCoversAll ? [] : unmapped,
    unmappedAll: unmapped,
    deletedTests,
    renamedTests,
    browserChanged,
    browserUnverified,
    inert,
    mapped: [...mapped].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0)),
    external: [...external.values()].map((e) => ({ ...e, paths: [...e.paths].sort() })),
    fallback,
    decisionCoversAll,
    reasons,
    tier,
    heavy: [...heavy].sort(),
  };
}

// ── config ───────────────────────────────────────────────────────────────────────────────────

export function loadConfig(repoDir, explicitPath) {
  const rel = explicitPath ?? CONFIG_REL;
  const abs = resolve(repoDir, rel);
  const warnings = [];
  if (!existsSync(abs)) return { config: {}, path: null, warnings };
  try {
    const parsed = JSON.parse(readFileSync(abs, "utf8"));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("config must be an object");
    }
    const known = new Set(["fallback", "heavy", "inert", "map", "external", "browser"]);
    for (const k of Object.keys(parsed)) {
      if (k.startsWith("_")) continue; // `_note`-style documentation keys
      if (!known.has(k)) warnings.push(`unknown config key ${JSON.stringify(k)} ignored`);
    }
    return { config: validateConfig(parsed, warnings), path: rel, warnings };
  } catch (e) {
    // Fail closed: an unreadable config must not narrow the gate.
    warnings.push(
      `config ${rel} is not valid JSON (${e.message}); using strict defaults (fallback = all)`,
    );
    return { config: {}, path: null, warnings };
  }
}

/** Reject malformed config sections instead of letting a typo silently narrow the gate. */
export function validateConfig(parsed, warnings = []) {
  const config = { ...parsed };
  const bad = (msg, key) => {
    warnings.push(`${msg}; ignoring ${key}`);
    delete config[key];
  };
  if (config.fallback !== undefined) {
    if (!config.fallback || typeof config.fallback !== "object" || Array.isArray(config.fallback)) {
      bad("fallback must be an object", "fallback");
    } else {
      const mode = config.fallback.mode ?? "all";
      const list = Array.isArray(config.fallback.list) ? config.fallback.list.map(String) : [];
      if (mode !== "all" && mode !== "list") {
        warnings.push(
          `fallback.mode ${JSON.stringify(mode)} is neither "all" nor "list"; using "all"`,
        );
      }
      config.fallback = { mode: mode === "list" ? "list" : "all", list };
    }
  }
  for (const key of ["heavy", "inert", "browser"]) {
    if (config[key] === undefined) continue;
    if (!Array.isArray(config[key])) bad(`${key} must be an array of globs/paths`, key);
    else config[key] = config[key].map(String);
  }
  if (config.map !== undefined) {
    if (!config.map || typeof config.map !== "object" || Array.isArray(config.map)) {
      bad("map must be an object of glob -> [test]", "map");
    } else {
      for (const [k, v] of Object.entries(config.map)) {
        if (!Array.isArray(v)) {
          warnings.push(`map[${JSON.stringify(k)}] must be an array of test paths; ignored`);
          delete config.map[k];
        } else config.map[k] = v.map(String);
      }
    }
  }
  if (config.external !== undefined) {
    if (!Array.isArray(config.external)) {
      bad("external must be an array of {globs, cmd} rules", "external");
    } else {
      const kept = [];
      config.external.forEach((r, i) => {
        const ok = r && typeof r === "object" && Array.isArray(r.globs) && r.globs.length > 0 &&
          Array.isArray(r.cmd) && r.cmd.length > 0 && typeof r.cmd[0] === "string";
        if (!ok) {
          warnings.push(`external[${i}] needs {glob: [...], cmd: [...]}; ignored`);
          return;
        }
        kept.push({
          label: typeof r.label === "string" && r.label ? r.label : r.cmd.join(" "),
          globs: r.globs.map(String),
          cmd: r.cmd.map(String),
          timeoutMs: Number.isFinite(r.timeoutMs) && r.timeoutMs > 0 ? r.timeoutMs : undefined,
        });
      });
      config.external = kept;
    }
  }
  return config;
}

// ── runner ───────────────────────────────────────────────────────────────────────────────────

export const PREFIX = "fast-gate:";
const out = (line) => process.stdout.write(`${line}\n`);
const warn = (line) => process.stderr.write(`${line}\n`);

function findOnPath(cmd) {
  const r = spawnSync("bash", ["-lc", `command -v ${cmd}`], { encoding: "utf8", timeout: 30_000 });
  return r.status === 0 && r.stdout.trim() ? r.stdout.trim() : null;
}

/**
 * Environment for child processes.
 * NODE_TEST_CONTEXT is DELETED deliberately: when this router is invoked from inside a
 * `node --test` run, that variable makes the nested runner skip every file and exit 0 — a silently
 * green gate. The fast gate must run the files it routed whatever invoked it.
 */
function childEnv() {
  const env = { ...process.env, GIT_TERMINAL_PROMPT: "0" };
  delete env.NODE_TEST_CONTEXT;
  return env;
}

/** fleet-deps hard-links node_modules from the per-VM store; never `npm ci`. Error string|null. */
function ensureDeps(repoDir, { enabled = true } = {}) {
  if (!enabled) return null;
  if (existsSync(join(repoDir, "node_modules"))) return null;
  const deps = findOnPath("fleet-deps");
  if (!deps) {
    return "node_modules is missing and fleet-deps is not on PATH; run fleet-deps first. Refusing to run tests that cannot load their pinned parser deps.";
  }
  out(`${PREFIX} bootstrapping node_modules with fleet-deps (never npm ci)`);
  const r = spawnSync(deps, [], {
    cwd: repoDir,
    stdio: "inherit",
    timeout: DEPS_TIMEOUT_MS,
    env: childEnv(),
  });
  if (r.status !== 0) {
    return `fleet-deps failed (status ${r.status ?? "signal"}); node_modules may be incomplete`;
  }
  return null;
}

export const HELP = `Usage: node scripts/test-changed.mjs [options]

Route the tests this branch needs, then run them (default) or report them.

  --repo <dir>        repo to inspect (default: cwd)
  --base <ref>        base ref (default: freshly fetched origin/main)
  --head <ref>        head ref (default: HEAD)
  --no-fetch          do not refresh origin/main first
  --committed-only    judge committed changes only, ignoring the working tree
  --tier <fast|heavy|all>  which tier to RUN (default fast; routed heavy tests are deferred)
  --strict            alias for --tier all
  --all               ignore change detection and run the whole tier
  --config <path>     gate config (default ${CONFIG_REL})
  --list              print the routed tests and exit without running
  --json              print the routing decision as JSON and exit
  --explain           print the per-path routing reasons
  --allow-unmapped    run the mapped tests even when paths are unmapped (LOUD warning; NOT for the gate)
  --no-deps           skip the fleet-deps bootstrap
  --quiet             fewer progress lines (warnings, deferrals and run output still shown)
  --help
`;

export function parseArgs(argv) {
  const opts = {
    repo: null,
    base: null,
    head: "HEAD",
    fetch: true,
    committedOnly: false,
    tier: "fast",
    all: false,
    config: null,
    list: false,
    json: false,
    explain: false,
    allowUnmapped: false,
    deps: true,
    quiet: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    switch (a) {
      case "--repo":
        opts.repo = next();
        break;
      case "--base":
        opts.base = next();
        break;
      case "--head":
        opts.head = next();
        break;
      case "--no-fetch":
        opts.fetch = false;
        break;
      case "--committed-only":
        opts.committedOnly = true;
        break;
      case "--tier": {
        const t = next();
        if (!["fast", "heavy", "all"].includes(t)) {
          throw new Error(`--tier must be fast|heavy|all (got ${t})`);
        }
        opts.tier = t;
        break;
      }
      case "--strict":
        opts.tier = "all";
        break;
      case "--all":
        opts.all = true;
        break;
      case "--config":
        opts.config = next();
        break;
      case "--list":
        opts.list = true;
        break;
      case "--json":
        opts.json = true;
        break;
      case "--explain":
        opts.explain = true;
        break;
      case "--allow-unmapped":
        opts.allowUnmapped = true;
        break;
      case "--no-deps":
        opts.deps = false;
        break;
      case "--quiet":
        opts.quiet = true;
        break;
      case "--help":
      case "-h":
        opts.help = true;
        break;
      default:
        throw new Error(`unknown argument ${a}`);
    }
  }
  return opts;
}

/** Assemble the routing decision for a repo. Shared by main() and the unit tests. */
export function plan(
  {
    repo,
    base,
    head = "HEAD",
    fetch = true,
    config: configOpt,
    all = false,
    tier = "fast",
    allowUnmapped = false,
    includeWorktree = true,
  },
) {
  const repoRoot = resolveRepoRoot(repo) ?? repo;
  const { config, path: configPath, warnings } = loadConfig(repoRoot, configOpt);
  let baseSha = null;
  let unknownBase = null;
  const baseWarnings = [];
  if (all) {
    unknownBase = "--all: change detection skipped";
  } else {
    const b = resolveBase(repoRoot, { base, fetch });
    baseWarnings.push(...b.warnings);
    if (b.error) unknownBase = b.error;
    else {
      const mb = mergeBase(repoRoot, b.ref, head);
      if (mb.error) unknownBase = mb.error;
      else baseSha = mb.sha;
    }
  }
  const index = buildIndex(repoRoot);
  let entries = [];
  if (baseSha) {
    const d = changedPaths(repoRoot, baseSha, head, { includeWorktree });
    if (d.error) unknownBase = d.error;
    else entries = d.entries;
  }
  const route = routeChanges({
    entries,
    index,
    config,
    tier,
    unknownBase,
    forceAll: all,
    allowUnmapped,
  });
  return {
    repoRoot,
    configPath,
    configWarnings: warnings,
    baseWarnings,
    baseSha,
    entries,
    index,
    route,
  };
}

export function main(argv = process.argv.slice(2)) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    warn(`${PREFIX} ${e.message}`);
    return 2;
  }
  if (opts.help) {
    process.stdout.write(HELP);
    return 0;
  }

  const p = plan({
    repo: opts.repo ? resolve(opts.repo) : process.cwd(),
    base: opts.base,
    head: opts.head,
    fetch: opts.fetch,
    config: opts.config,
    all: opts.all,
    tier: opts.tier,
    allowUnmapped: opts.allowUnmapped,
    includeWorktree: !opts.committedOnly,
  });
  const { route } = p;
  p.configWarnings.forEach((w) => warn(`${PREFIX} WARN: ${w}`));
  p.baseWarnings.forEach((w) => warn(`${PREFIX} WARN: ${w}`));
  if (!p.baseSha) {
    warn(
      `${PREFIX} FAIL-CLOSED: unknown base (${
        route.fallback?.detail ?? "unresolvable"
      }); running the fallback`,
    );
  }

  const missingRouted = route.run.filter((t) => !existsSync(join(p.repoRoot, t)));
  if (opts.json) {
    process.stdout.write(
      `${
        JSON.stringify(
          {
            base: p.baseSha,
            config: p.configPath,
            changed: p.entries.map((e) => ({ status: e.status, path: e.path, oldPath: e.oldPath })),
            missingRouted,
            ...route,
          },
          null,
          2,
        )
      }\n`,
    );
    return missingRouted.length > 0 ? 2 : 0;
  }
  if (opts.list) {
    route.run.forEach((t) => process.stdout.write(`${t}\n`));
    return 0;
  }

  if (!opts.quiet) out(`${PREFIX} changed ${p.entries.length} path(s)`);
  for (const n of route.browserChanged) out(`${PREFIX} browser-gated: ${n}`);
  for (const n of route.browserUnverified) {
    out(`BROWSER_UNVERIFIED: ${n} (Chrome gate blocked by web-ai-showcase-jdz)`);
  }
  for (const n of route.deletedTests) out(`${PREFIX} deleted-test: ${n}`);
  for (const r of route.renamedTests) out(`${PREFIX} renamed-test: ${r.from} -> ${r.to}`);
  for (const m of route.mapped) out(`${PREFIX} mapped: ${m.path} -> ${m.suites.join(" ")}`);
  for (const n of route.unmapped) out(`unmapped: ${n}`);
  if (route.fallback) {
    out(
      `${PREFIX} fallback: mode=${route.fallback.mode} reason=${route.fallback.reason} paths=${
        route.fallback.paths.slice(0, 3).join(" ")
      }`,
    );
    if (route.fallback.detail) out(`${PREFIX} fallback-detail: ${route.fallback.detail}`);
  }
  for (const t of route.deferred) out(`${PREFIX} deferred-heavy: ${t}`);
  for (const e of route.external) out(`${PREFIX} external: ${e.paths.join(", ")} -> ${e.label}`);
  if (opts.explain) {
    for (const r of route.reasons) out(`${PREFIX} route: ${r.path} -> ${r.reason}`);
  }

  // A routed test that does not exist would be silently ignored by `node --test` (it errors only
  // when EVERY path is missing) and then reported as completed. Refuse instead of pretending.
  if (missingRouted.length > 0) {
    for (const t of missingRouted) out(`${PREFIX} MISSING-ROUTED-TEST: ${t}`);
    warn(
      `${PREFIX} FAIL-CLOSED: ${missingRouted.length} routed test file(s) do not exist on disk; refusing to run a partial set`,
    );
    return 2;
  }

  const needExternal = route.external.length > 0;
  if (route.run.length === 0 && !needExternal) {
    if (route.deferred.length > 0) {
      // Every routed test is a declared-heavy deferral: say DEFERRED, never "none test-bearing".
      for (const t of route.deferred) out(`${PREFIX} deferred-only: ${t}`);
      out(
        `${PREFIX} result: DEFERRED (0 test file(s) run, ${route.deferred.length} deferred, 0 external, ${route.unmapped.length} unmapped)`,
      );
      return 0;
    }
    if (route.unmappedAll.length > 0) {
      warn(
        `${PREFIX} FAIL-CLOSED: ${route.unmappedAll.length} changed path(s) have no routed test${
          opts.allowUnmapped ? " (--allow-unmapped)" : ""
        }; NOT verified`,
      );
      return 2;
    }
    out(`${PREFIX} no-tests: ${p.entries.length} changed path(s), none test-bearing`);
    return 0;
  }

  let status = 0;
  if (route.run.length > 0) {
    const depErr = ensureDeps(p.repoRoot, { enabled: opts.deps });
    if (depErr) {
      warn(`${PREFIX} ${depErr}`);
      return 3;
    }
    if (!opts.quiet) out(`${PREFIX} running ${route.run.length} test file(s) [tier=${route.tier}]`);
    for (const t of route.run) out(`${PREFIX} routed: ${t}`);
    const res = spawnSync(process.execPath, ["--test", ...route.run], {
      cwd: p.repoRoot,
      stdio: "inherit",
      env: childEnv(),
    });
    status = res.status ?? 2;
    if (status === 0 || status === 1) {
      // Node's runner executes every file before reporting, so a completed invocation is per-file proof.
      out(`${PREFIX} completed: ${route.run.join(" ")}`);
    }
  }

  // External runners (e.g. `deno task test:server`) are EXECUTED here, so a config declaration is
  // never an unverified claim. A failure fails the gate.
  let externalFailed = false;
  for (const e of needExternal ? route.external : []) {
    out(`${PREFIX} external-run: ${e.label} (${e.cmd.join(" ")})`);
    const r = spawnSync(e.cmd[0], e.cmd.slice(1), {
      cwd: p.repoRoot,
      stdio: "inherit",
      timeout: e.timeoutMs ?? 300_000,
      env: childEnv(),
    });
    if (r.status === 0) out(`${PREFIX} external-ok: ${e.label}`);
    else {
      externalFailed = true;
      out(`${PREFIX} external-fail: ${e.label} (status ${r.status ?? "signal"})`);
    }
  }

  out(
    `${PREFIX} result: ${
      status !== 0 || externalFailed ? "FAIL" : "PASS"
    } (${route.run.length} test file(s) run, ${route.deferred.length} deferred, ${
      needExternal ? route.external.length : 0
    } external, ${route.unmapped.length} unmapped)`,
  );
  if (status !== 0) return status;
  return externalFailed ? 1 : 0;
}

const invokedDirectly = (() => {
  try {
    return Boolean(process.argv[1]) &&
      resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
})();
if (invokedDirectly) process.exitCode = main();
