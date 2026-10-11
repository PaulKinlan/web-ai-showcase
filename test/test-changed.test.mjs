// Unit tests for scripts/test-changed.mjs — the fail-closed changed-test router (bead
// web-ai-showcase-360, ADR web-ai-showcase-cxw).
//
// Why these exist: this router decides what the fast gate verifies, and an empty run set is
// indistinguishable from a green suite. So every fail-closed rule below asserts the FALLBACK (or a
// loud marker), never a quiet skip. Fixtures are synthetic temp git repos — no network, no browser
// (test/suite-stays-browser-free.test.mjs still applies), and nothing writes to this checkout.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildIndex,
  changedPaths,
  CONFIG_REL,
  globMatch,
  isBrowserTestPath,
  isUnitTestPath,
  parseNameStatusZ,
  resolveRel,
  routeChanges,
  scanSourceRefs,
  stripComments,
  testsForPath,
  validateConfig,
} from "../scripts/test-changed.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROUTER = join(ROOT, "scripts", "test-changed.mjs");
const ROUTER_SRC = readFileSync(ROUTER, "utf8");
// Repository-shaped paths live in a JSON fixture (test/fixtures/router-paths.json) because a literal
// path in THIS file is a dependency claim to the router: naming docs/, public/ or server.ts here
// would route this test suite (seconds) on unrelated changes to those files.
const PATHS = JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "router-paths.json"), "utf8"));
const DOC = PATHS.inertDoc; // docs/readme.md — built-in inert prefix
const DOC2 = PATHS.inertDocOther; // docs/x.md — same, for the reason-code test
const CSS = PATHS.browserCss; // public/styles.css — matches the shipped `browser` globs
const SERVER = PATHS.serverTs; // server.ts — covered by the config's external rule

// ── helpers ───────────────────────────────────────────────────────────────────────────────────

/** A synthetic index, so routing rules are tested without touching a filesystem. */
function fakeIndex({ tests, exact = {}, dirs = [], repoWide = [] }) {
  return {
    nodes: new Map(),
    exact: new Map(Object.entries(exact).map(([k, v]) => [k, new Set(v)])),
    dirEntries: dirs,
    repoWideTests: new Set(repoWide),
    allUnitTests: tests,
    topLevel: new Set(),
  };
}

const ALL = ["test/a.test.mjs", "test/b.test.mjs"];
const change = (path, status = "M") => ({ status, path, oldPath: path });

function git(cwd, ...args) {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" });
}
/** A throwaway repo with a bare `origin` whose main is the base commit. */
function makeRepo({ files, extra = {} } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "test-changed-"));
  const bare = `${dir}-origin.git`;
  execFileSync("git", ["init", "-q", "--bare", bare]);
  const run = (cwd, ...args) => execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" });
  run(dir, "init", "-q", "-b", "main");
  run(dir, "config", "user.email", "t@example.com");
  run(dir, "config", "user.name", "Test");
  run(dir, "config", "commit.gpgsign", "false");
  for (const [rel, body] of Object.entries({ ...files, ...extra })) write(dir, rel, body);
  run(dir, "add", "-A");
  run(dir, "commit", "-q", "-m", "base");
  run(dir, "remote", "add", "origin", bare);
  run(dir, "push", "-q", "origin", "main");
  return {
    dir,
    bare,
    run,
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
    cleanupBare: () => rmSync(bare, { recursive: true, force: true }),
  };
}

function write(dir, rel, body) {
  const abs = join(dir, rel);
  mkdirSync(dirname(abs), { recursive: true });
  writeFileSync(abs, body);
}

function runRouter(dir, args = [], opts = {}) {
  return execFileSync("node", [ROUTER, "--repo", dir, ...args], {
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
    ...opts,
  });
}

function routeJson(dir, args = []) {
  const res = runRouter(dir, [...args, "--json"]);
  // The JSON payload follows any routing notice lines.
  const start = res.indexOf("{");
  return JSON.parse(res.slice(start));
}

// ── pure parsing ──────────────────────────────────────────────────────────────────────────────

test("parseNameStatusZ keeps BOTH sides of a rename (--name-only loses the old path)", () => {
  const raw = [
    "M",
    "lib/a.mjs",
    "A",
    "lib/new.mjs",
    "D",
    "lib/gone.mjs",
    "R100",
    "lib/old.mjs",
    "lib/renamed.mjs",
    "",
  ].join("\0");
  const entries = parseNameStatusZ(raw);
  assert.deepEqual(
    entries.filter((e) => e.status !== "R").map((e) => [e.status, e.path]),
    [["M", "lib/a.mjs"], ["A", "lib/new.mjs"], ["D", "lib/gone.mjs"]],
  );
  const ren = entries.find((e) => e.status === "R");
  assert.deepEqual([ren.oldPath, ren.path], ["lib/old.mjs", "lib/renamed.mjs"]);
});

test("stripComments drops prose but keeps string literals, including URLs and code", () => {
  const src = `// readFileSync("lib/a.js") in a comment
const url = "https://example.com/x?y=1"; /* "lib/ignored.js" */
const p = "lib/real.js";
`;
  const out = stripComments(src);
  assert.match(out, /lib\/real\.js/);
  assert.match(out, /https:\/\/example\.com/);
  assert.doesNotMatch(out, /lib\/a\.js/);
  assert.doesNotMatch(out, /lib\/ignored\.js/);
});

test("globMatch handles **, *, a plain directory prefix and non-matches", () => {
  assert.ok(globMatch("models/x/worker.js", "models/**"));
  assert.ok(globMatch(SERVER, SERVER));
  assert.ok(globMatch("models/x/worker.js", "models/x"));
  assert.ok(globMatch("test/a.test.mjs", "test/*.test.mjs"));
  assert.ok(!globMatch("test/nested/a.test.mjs", "test/*.test.mjs"));
  assert.ok(!globMatch("lib/a.js", "models/**"));
});

test("isUnitTestPath / isBrowserTestPath classify shapes, including deleted files", () => {
  assert.ok(isUnitTestPath("test/x.test.mjs"));
  assert.ok(!isUnitTestPath("test/lib/helper.mjs"));
  assert.ok(!isUnitTestPath("test-browser/x.test.mjs"));
  assert.ok(isBrowserTestPath("test-browser/x.test.mjs"));
  assert.ok(isBrowserTestPath("test-browser"));
});

test("resolveRel resolves relative tokens and refuses to escape the repo", () => {
  assert.equal(resolveRel("test/a.test.mjs", "../lib/a.js"), "lib/a.js");
  assert.equal(resolveRel("scripts/x.mjs", "../../etc/passwd"), null);
});

// ── fail-closed routing rules (pure) ──────────────────────────────────────────────────────────

test("an unmapped changed path triggers the FULL fallback, never an empty run", () => {
  const index = fakeIndex({
    tests: ALL,
    exact: { "lib/a.mjs": ["test/a.test.mjs"] },
    repoWide: ["test/b.test.mjs"],
  });
  const r = routeChanges({ entries: [change("brand-new/thing.mjs", "A")], index });
  assert.deepEqual(r.unmappedAll, ["brand-new/thing.mjs"]);
  assert.deepEqual(r.run, ALL, "the fallback runs the whole unit suite");
  assert.equal(r.fallback.mode, "all");
  assert.notEqual(r.run.length, 0);
});

test("a repo-wide sweep test alone does NOT count as a mapping", () => {
  // test/b.test.mjs sweeps the repo (git ls-files), so it runs for any change — but a new module is
  // still unverified, and that must reach the fallback rather than a quiet single-test pass.
  const index = fakeIndex({ tests: ALL, repoWide: ["test/b.test.mjs"] });
  const path = "newarea/thing.mjs";
  const { specific, sweep } = testsForPath(index, path);
  assert.equal(specific.size, 0);
  assert.deepEqual([...sweep], ["test/b.test.mjs"]);
  const r = routeChanges({ entries: [change(path, "A")], index });
  assert.equal(r.fallback?.mode, "all");
  assert.deepEqual(r.unmappedAll, [path]);
  assert.ok(r.run.includes("test/b.test.mjs"), "the sweep test still runs");
});

test("a changed unit test runs itself; a DELETED unit test forces the fallback", () => {
  const index = fakeIndex({ tests: ALL });
  const changed = routeChanges({ entries: [change("test/a.test.mjs")], index });
  assert.deepEqual(changed.run, ["test/a.test.mjs"]);
  assert.equal(changed.fallback, null);

  const deleted = routeChanges({
    entries: [{ status: "D", path: "test/a.test.mjs", oldPath: "test/a.test.mjs" }],
    index,
  });
  assert.deepEqual(deleted.deletedTests, ["test/a.test.mjs"]);
  assert.equal(deleted.fallback?.mode, "all");
  assert.deepEqual(deleted.run, ALL);
});

test("a rename judges BOTH sides: importers of the old path run, the unmapped new path falls back", () => {
  const index = fakeIndex({ tests: ALL, exact: { "lib/a.mjs": ["test/a.test.mjs"] } });
  const r = routeChanges({
    entries: [{ status: "R", path: "lib/renamed.mjs", oldPath: "lib/a.mjs" }],
    index,
  });
  assert.ok(r.run.includes("test/a.test.mjs"), "the test that still imports lib/a.mjs must run");
  assert.deepEqual(r.unmappedAll, ["lib/renamed.mjs"]);
  assert.equal(r.fallback?.mode, "all");
});

test("a shared helper change routes every test that imports it", () => {
  const index = fakeIndex({
    tests: ALL,
    exact: { "test/lib/isolated-repo.mjs": ["test/a.test.mjs", "test/b.test.mjs"] },
  });
  const r = routeChanges({ entries: [change("test/lib/isolated-repo.mjs")], index });
  assert.deepEqual(r.run, ALL);
  assert.equal(r.fallback, null);
});

test("docs-only changes are inert: no suite runs, and no fallback is invented", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [change("docs/notes.md", "A")], index });
  assert.deepEqual(r.run, []);
  assert.deepEqual(r.inert, ["docs/notes.md"]);
  assert.deepEqual(r.unmappedAll, []);
  assert.equal(r.fallback, null);
});

test("a markdown file a test actually reads is ROUTED, not treated as inert", () => {
  const index = fakeIndex({ tests: ALL, exact: { "AGENTS.md": ["test/b.test.mjs"] } });
  const r = routeChanges({ entries: [change("AGENTS.md")], index });
  assert.deepEqual(r.run, ["test/b.test.mjs"]);
  assert.deepEqual(r.inert, []);
  assert.equal(r.fallback, null);
});

test("a changed gate config forces the full fallback", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [change(CONFIG_REL)], index });
  assert.equal(r.fallback?.mode, "all");
  assert.deepEqual(r.run, ALL);
});

test("an unknown base runs the whole suite and says so", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [], index, unknownBase: "origin/main does not resolve" });
  assert.equal(r.fallback?.mode, "all");
  assert.equal(r.fallback.reason, "unknown-base", "the reason code is machine-readable");
  assert.match(r.fallback.detail, /origin\/main does not resolve/);
  assert.deepEqual(r.run, ALL);
});

test("every fallback carries a machine-readable reason code", () => {
  const index = fakeIndex({ tests: ALL });
  const cases = [
    [[change("newarea/thing.mjs", "A")], {}, "unmapped"],
    [[{ status: "D", path: "test/a.test.mjs", oldPath: "test/a.test.mjs" }], {}, "deleted-test"],
    [[change(CONFIG_REL)], {}, "config-change"],
    [[change(DOC2)], { forceAll: true }, "all-flag"],
  ];
  for (const [entries, extra, expected] of cases) {
    const r = routeChanges({ entries, index, ...extra });
    assert.equal(r.fallback?.reason, expected, `entries ${JSON.stringify(entries)}`);
  }
  const paths = routeChanges({ entries: [change("newarea/thing.mjs", "A")], index });
  assert.deepEqual(
    paths.fallback.paths,
    ["newarea/thing.mjs"],
    "the trigger paths stay on the marker",
  );
});

test("docs-only changes do not widen to the suite even when a repo-wide sweep test exists", () => {
  // coord, 2026-10-11: only proven non-runtime paths are inert, and no needless all-unit fallback.
  const index = fakeIndex({ tests: ALL, repoWide: ["test/b.test.mjs"] });
  const r = routeChanges({ entries: [change("docs/notes.md", "A")], index });
  assert.equal(r.fallback, null);
  assert.deepEqual(r.run, ["test/b.test.mjs"], "the cheap repo-wide sweep still runs");
  assert.deepEqual(r.unmappedAll, []);
});

test("browser-facing paths are reported as unverified without suppressing their routing", () => {
  // coord, 2026-10-11: a browser change must say BROWSER_UNVERIFIED/jdz, never imply coverage.
  const index = fakeIndex({ tests: ALL });
  const config = { browser: ["public/**", "**/*.css"] };
  const r = routeChanges({ entries: [change(CSS)], index, config });
  assert.deepEqual(r.browserUnverified, [CSS]);
  assert.deepEqual(r.unmappedAll, [CSS], "the marker is not coverage");
  assert.equal(
    r.fallback?.reason,
    "unmapped",
    "an unmapped browser path still triggers the fallback",
  );

  const mapped = routeChanges({
    entries: [change(CSS)],
    index: fakeIndex({ tests: ALL, exact: { [CSS]: ["test/a.test.mjs"] } }),
    config,
  });
  assert.deepEqual(mapped.browserUnverified, [CSS]);
  assert.deepEqual(mapped.run, ["test/a.test.mjs"], "routing is untouched by the report");
});

test("a changed browser test is reported, never silently absorbed", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [change("test-browser/mobile-viewport.test.mjs")], index });
  assert.deepEqual(r.browserChanged, ["test-browser/mobile-viewport.test.mjs"]);
  assert.equal(r.fallback, null, "the browser gate's exclusion is a stated policy, not a fallback");
  assert.deepEqual(r.unmappedAll, []);
});

// ── tiering ───────────────────────────────────────────────────────────────────────────────────

test("heavy tests are ROUTED but deferred by the fast tier, and remain visible", () => {
  const index = fakeIndex({
    tests: ALL,
    exact: { "lib/a.mjs": ["test/a.test.mjs", "test/b.test.mjs"] },
  });
  const config = { heavy: ["test/b.test.mjs"] };
  const fast = routeChanges({ entries: [change("lib/a.mjs")], index, config, tier: "fast" });
  assert.deepEqual(fast.run, ["test/a.test.mjs"]);
  assert.deepEqual(fast.deferred, ["test/b.test.mjs"]);
  assert.deepEqual(fast.routed, ALL, "the deferral is a visible decision, not a dropped test");

  const heavy = routeChanges({ entries: [change("lib/a.mjs")], index, config, tier: "heavy" });
  assert.deepEqual(heavy.run, ["test/b.test.mjs"]);

  const all = routeChanges({ entries: [change("lib/a.mjs")], index, config, tier: "all" });
  assert.deepEqual(all.run, ALL);
});

test("fallback mode=list runs the bounded core set, and WIDENS when the list is unusable", () => {
  const index = fakeIndex({ tests: ALL, repoWide: ["test/b.test.mjs"] });
  const bounded = routeChanges({
    entries: [change("newarea/thing.mjs", "A")],
    index,
    config: { fallback: { mode: "list", list: ["test/b.test.mjs"] } },
  });
  assert.deepEqual(bounded.run, ["test/b.test.mjs"]);
  assert.equal(bounded.fallback.mode, "list");

  for (const list of [[], ["test/gone.test.mjs"]]) {
    const widened = routeChanges({
      entries: [change("newarea/thing.mjs", "A")],
      index,
      config: { fallback: { mode: "list", list } },
    });
    assert.equal(
      widened.fallback.mode,
      "all",
      `list ${JSON.stringify(list)} must widen, not run nothing`,
    );
    assert.deepEqual(widened.run, ALL);
  }
});

test("an external runner covers a path — and is collected so the gate really runs it", () => {
  const index = fakeIndex({ tests: ALL });
  const config = {
    external: [{
      globs: [SERVER],
      cmd: ["deno", "task", "test:server"],
      label: "deno server",
    }],
  };
  const r = routeChanges({ entries: [change(SERVER)], index, config });
  assert.deepEqual(r.unmappedAll, []);
  assert.equal(r.fallback, null);
  assert.deepEqual(r.external.map((e) => e.label), ["deno server"]);
  assert.deepEqual(r.external[0].paths, [SERVER]);
  assert.deepEqual(r.external[0].cmd, ["deno", "task", "test:server"]);
});

test("without an external rule the same path still falls back", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [change(SERVER)], index });
  assert.deepEqual(r.unmappedAll, [SERVER]);
  assert.equal(r.fallback?.mode, "all");
});

test("--allow-unmapped records the gap without widening the run set", () => {
  const index = fakeIndex({ tests: ALL, exact: { "lib/a.mjs": ["test/a.test.mjs"] } });
  const r = routeChanges({
    entries: [change("lib/a.mjs"), change("brand-new/thing.mjs", "A")],
    index,
    allowUnmapped: true,
  });
  assert.equal(r.fallback, null);
  assert.deepEqual(r.run, ["test/a.test.mjs"]);
  assert.deepEqual(r.unmappedAll, ["brand-new/thing.mjs"]);
  assert.deepEqual(r.unmapped, ["brand-new/thing.mjs"]);
});

test("when the decision covers the whole suite, no path is reported unmapped", () => {
  const index = fakeIndex({ tests: ALL });
  const r = routeChanges({ entries: [change("brand-new/thing.mjs", "A")], index });
  assert.equal(r.decisionCoversAll, true);
  assert.deepEqual(
    r.unmapped,
    [],
    "reporting 'unmapped' after running everything would be a false alarm",
  );
  assert.deepEqual(r.unmappedAll, ["brand-new/thing.mjs"], "the underlying gap is still recorded");
});

test("a repo-wide git scan written in argv form is detected too (P2 review finding)", () => {
  // `execFileSync("git", ["ls-files"])` separates git from its subcommand with comma/bracket text,
  // so a whitespace-only regex misses it and the sweep test silently drops out of the fallback set.
  // The cases live in test/fixtures/scan-cases.json (see its __note) so this test file's own source
  // does not contain the call pattern — otherwise IT would be detected as a repo-wide sweep test and
  // would run on every change in the repository.
  const scanCases = JSON.parse(readFileSync(join(ROOT, "test/fixtures/scan-cases.json"), "utf8"));
  const dir = mkdtempSync(join(tmpdir(), "test-changed-scan-"));
  try {
    for (const { src, repoWide } of scanCases) {
      write(
        dir,
        "scan.mjs",
        `import { execSync, execFileSync, spawnSync } from "node:child_process";\n${src}\n`,
      );
      const refs = scanSourceRefs(dir, "scan.mjs", new Set(["scan.mjs"]));
      assert.equal(refs.repoWide, repoWide, `repoWide for: ${src}`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the router's own test file is not a repo-wide sweep test", () => {
  // A sweep test is routed for EVERY change; this one asserting its own absence keeps the gate fast
  // (the fixtures that contain the scan patterns are JSON, not scanned modules).
  const index = buildIndex(ROOT);
  assert.ok(
    !index.repoWideTests.has("test/test-changed.test.mjs"),
    "test/test-changed.test.mjs must not sweep the repo: it would run on every unrelated change",
  );
});

// ── config validation ─────────────────────────────────────────────────────────────────────────

test("validateConfig refuses malformed sections instead of silently narrowing the gate", () => {
  const warnings = [];
  const config = validateConfig(
    {
      fallback: { mode: "sometimes", list: "nope" },
      heavy: "test/a.test.mjs",
      browser: "public/**",
      map: { "models/**": "test/a.test.mjs" },
      external: [
        { globs: [SERVER] },
        { cmd: ["x"] },
        { globs: ["a.ts"], cmd: ["node", "x"] },
        7,
      ],
    },
    warnings,
  );
  assert.equal(config.fallback.mode, "all");
  assert.deepEqual(config.fallback.list, []);
  assert.equal(config.heavy, undefined);
  assert.equal(config.browser, undefined);
  assert.deepEqual(config.map, {});
  assert.deepEqual(config.external.map((e) => e.label), ["node x"]);
  assert.ok(warnings.length >= 4, `expected warnings, got ${JSON.stringify(warnings)}`);
});

test("validateConfig accepts the browser glob list that drives BROWSER_UNVERIFIED", () => {
  const warnings = [];
  const config = validateConfig({ browser: ["public/**", "**/*.css"] }, warnings);
  assert.deepEqual(config.browser, ["public/**", "**/*.css"]);
  assert.deepEqual(warnings, []);
});

test("the router never shells out to npm (fleet-deps owns node_modules)", () => {
  // Prose may mention npm; an INVOCATION may not exist. fleet-check runs in worktrees where
  // `npm ci` would rebuild the world on the VM's scarcest resource, so the router never calls it.
  assert.doesNotMatch(ROUTER_SRC, /(?:spawnSync|execFileSync|execSync)\(\s*["'`]npm["'`]/);
  assert.doesNotMatch(ROUTER_SRC, /["'`]npm["'`]\s*,\s*\[\s*["'`](?:ci|install)/);
  assert.match(ROUTER_SRC, /fleet-deps/);
});

// ── CLI + real git ────────────────────────────────────────────────────────────────────────────

const PASSING_TEST = `import { test } from "node:test";
import assert from "node:assert/strict";
test("ok", () => { assert.equal(1, 1); });
`;
const FAILING_TEST = `import { test } from "node:test";
import assert from "node:assert/strict";
test("no", () => { assert.equal(1, 2); });
`;

function fixture() {
  return makeRepo({
    files: {
      "lib/a.mjs": "export const a = 1;\n",
      "test/a.test.mjs": `import { a } from "../lib/a.mjs";\n${PASSING_TEST}`,
      [DOC]: "# docs\n",
      [CSS]: ":root { --warm: #fff; }\n",
      "package.json": '{ "name": "fixture", "private": true }\n',
    },
  });
}

test("CLI: a changed library routes its importer and runs it green", () => {
  const repo = fixture();
  try {
    write(repo.dir, "lib/a.mjs", "export const a = 2;\n");
    const planned = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.deepEqual(planned.run, ["test/a.test.mjs"]);
    assert.equal(planned.fallback, null);

    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
    ]);
    assert.match(out, /^fast-gate: completed: test\/a\.test\.mjs$/m);
    assert.match(out, /^fast-gate: result: PASS/m);
    // The file really EXECUTED: a nested `node --test` that inherited NODE_TEST_CONTEXT would skip
    // it and still exit 0, so the runner's own summary is the honest proof.
    assert.match(out, /pass 1/, "the routed test file must actually run");
    assert.doesNotMatch(out, /skipping running files/);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: the default base is a freshly fetched origin/main", () => {
  const repo = fixture();
  try {
    write(repo.dir, "lib/a.mjs", "export const a = 3;\n");
    const out = runRouter(repo.dir, ["--json"]);
    const planned = JSON.parse(out.slice(out.indexOf("{")));
    assert.deepEqual(planned.run, ["test/a.test.mjs"]);
    assert.ok(planned.base, "origin/main resolved to a merge base");
    assert.match(out, /fetched|base/, "the run reports which base it used");
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: an unknown new module falls back to the whole suite and names the unverified path", () => {
  const repo = fixture();
  try {
    write(repo.dir, "brand-new/thing.mjs", "export const t = 1;\n");
    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
    ]);
    assert.match(
      out,
      /^fast-gate: fallback: mode=all reason=unmapped paths=brand-new\/thing\.mjs$/m,
    );
    assert.match(out, /test\/a\.test\.mjs/, "the fallback includes the fixture's unit test");
    // When the decision covered the whole suite, an `unmapped:` line would be a false blindness
    // alarm — the fallback line above is the honest report.
    assert.doesNotMatch(out, /^unmapped: /m);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a failing routed test exits non-zero, and completed: is only claimed after a real run", () => {
  const repo = fixture();
  try {
    write(repo.dir, "test/a.test.mjs", FAILING_TEST);
    assert.throws(
      () => runRouter(repo.dir, ["--base", "origin/main", "--no-fetch", "--no-deps", "--quiet"]),
      (e) => {
        assert.equal(e.status, 1, "a failing test must fail the gate");
        assert.match(String(e.stdout), /^fast-gate: result: FAIL/m);
        assert.match(
          String(e.stdout),
          /^fast-gate: completed: test\/a\.test\.mjs$/m,
          "the file did execute",
        );
        return true;
      },
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: --list prints exactly the routed tests, and --json carries the reasons", () => {
  const repo = fixture();
  try {
    write(repo.dir, "lib/a.mjs", "export const a = 4;\n");
    const listed = runRouter(repo.dir, ["--base", "origin/main", "--no-fetch", "--list"]);
    assert.equal(listed.trim(), "test/a.test.mjs");
    const planned = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.ok(
      planned.reasons.some((r) =>
        r.path === "lib/a.mjs" && /mapped to test\/a\.test\.mjs/.test(r.reason)
      ),
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: an external rule's command is really executed, and its failure fails the gate", () => {
  const repo = fixture();
  try {
    const marker = join(repo.dir, "external-ran.txt");
    const config = join(repo.dir, "external.config.json");
    writeFileSync(
      config,
      JSON.stringify({
        external: [{
          globs: [SERVER],
          cmd: ["node", "-e", `require("fs").writeFileSync(${JSON.stringify(marker)},"yes")`],
        }],
      }),
    );
    write(repo.dir, SERVER, "export const s = 1;\n");
    const ok = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
      "--config",
      config,
    ]);
    assert.ok(existsSync(marker), "the declared external command must actually run");
    assert.match(ok, /^fast-gate: external-ok:/m);
    assert.match(ok, /^fast-gate: result: PASS/m);

    const failing = join(repo.dir, "external-fail.config.json");
    writeFileSync(
      failing,
      JSON.stringify({
        external: [{ globs: [SERVER], cmd: ["node", "-e", "process.exit(3)"] }],
      }),
    );
    assert.throws(
      () =>
        runRouter(repo.dir, [
          "--base",
          "origin/main",
          "--no-fetch",
          "--no-deps",
          "--quiet",
          "--config",
          failing,
        ]),
      (e) => e.status === 1 && /external-fail/.test(String(e.stdout)),
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a deleted test file falls back, and a rename reports both sides", () => {
  const repo = fixture();
  try {
    repo.run(repo.dir, "rm", "-q", "test/a.test.mjs");
    const deleted = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.deepEqual(deleted.deletedTests, ["test/a.test.mjs"]);
    assert.equal(deleted.fallback.mode, "all");
    repo.run(repo.dir, "reset", "-q", "--hard", "origin/main");

    repo.run(repo.dir, "mv", "lib/a.mjs", "lib/renamed.mjs");
    const renamed = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.equal(renamed.changed.length, 1);
    assert.equal(renamed.changed[0].status, "R");
    assert.ok(
      renamed.run.includes("test/a.test.mjs") || renamed.fallback,
      "the importer of the old path must be routed (or the whole suite must run)",
    );
    assert.ok(renamed.unmappedAll.includes("lib/renamed.mjs"), "the new path has no test");
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: docs-only changes run nothing and say so (no fallback, no silent pass)", () => {
  const repo = fixture();
  try {
    write(repo.dir, DOC, "# docs changed\n");
    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
    ]);
    assert.match(out, /fast-gate: no-tests:/);
    assert.doesNotMatch(out, /FALLBACK/);
    assert.deepEqual(routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]).run, []);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: an unknown flag is refused rather than absorbed", () => {
  const repo = fixture();
  try {
    assert.throws(
      () => runRouter(repo.dir, ["--not-a-flag"]),
      (e) => e.status === 2 && /unknown argument/.test(String(e.stderr)),
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a matched external rule cannot be skipped by a flag (P2 review finding)", () => {
  // The only way to a partial set is --allow-unmapped, which is loud and not for CHECK_FAST_CMD.
  const repo = fixture();
  try {
    assert.throws(
      () => runRouter(repo.dir, ["--no-external"]),
      (e) => e.status === 2 && /unknown argument/.test(String(e.stderr)),
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a change routed only to heavy tests reports DEFERRED, not 'none test-bearing'", () => {
  // P3 review finding: an empty fast-tier run with deferred tests is not a test-free change.
  const repo = fixture();
  const cfgDir = mkdtempSync(join(tmpdir(), "test-changed-cfg-"));
  try {
    const config = join(cfgDir, "heavy.config.json");
    writeFileSync(config, JSON.stringify({ heavy: ["test/a.test.mjs"] }));
    write(repo.dir, "lib/a.mjs", "export const a = 7;\n");
    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
      "--config",
      config,
    ]);
    assert.match(out, /^fast-gate: deferred-only: test\/a\.test\.mjs$/m);
    assert.match(out, /^fast-gate: result: DEFERRED/m);
    assert.doesNotMatch(out, /none test-bearing/);
    assert.match(out, /^fast-gate: deferred-heavy: test\/a\.test\.mjs$/m);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
    rmSync(cfgDir, { recursive: true, force: true });
  }
});

test("CLI: a renamed test file runs its NEW path and never the vanished old one", () => {
  const repo = fixture();
  try {
    repo.run(repo.dir, "mv", "test/a.test.mjs", "test/a-moved.test.mjs");
    const planned = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.ok(planned.run.includes("test/a-moved.test.mjs"), "the new path runs");
    assert.ok(
      !planned.run.includes("test/a.test.mjs"),
      "the vanished old path must never be routed (node --test would ignore it silently)",
    );
    assert.deepEqual(planned.renamedTests, [
      { from: "test/a.test.mjs", to: "test/a-moved.test.mjs" },
    ]);

    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
    ]);
    assert.match(
      out,
      /^fast-gate: renamed-test: test\/a\.test\.mjs -> test\/a-moved\.test\.mjs$/m,
    );
    assert.match(out, /^fast-gate: result: PASS/m);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a routed test that does not exist fails closed instead of silently not running", () => {
  // node --test ignores a missing path when other files are present and exits 0, so a partial set
  // would be reported as `completed:`. Refuse it.
  const repo = fixture();
  try {
    const config = join(repo.dir, "stale.config.json");
    writeFileSync(config, JSON.stringify({ map: { "lib/a.mjs": ["test/gone.test.mjs"] } }));
    write(repo.dir, "lib/a.mjs", "export const a = 8;\n");
    assert.throws(
      () =>
        runRouter(repo.dir, [
          "--base",
          "origin/main",
          "--no-fetch",
          "--no-deps",
          "--quiet",
          "--config",
          config,
        ]),
      (e) => {
        assert.equal(e.status, 2);
        assert.match(String(e.stdout), /MISSING-ROUTED-TEST: test\/gone\.test\.mjs/);
        assert.doesNotMatch(String(e.stdout), /fast-gate: completed:/);
        return true;
      },
    );
    // --json carries it too, and exits 2 rather than reporting a routable set.
    assert.throws(
      () =>
        runRouter(repo.dir, [
          "--base",
          "origin/main",
          "--no-fetch",
          "--no-deps",
          "--quiet",
          "--json",
          "--config",
          config,
        ]),
      (e) => {
        assert.equal(e.status, 2);
        assert.deepEqual(JSON.parse(String(e.stdout)).missingRouted, ["test/gone.test.mjs"]);
        return true;
      },
    );
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: a browser-facing change prints BROWSER_UNVERIFIED with the jdz blocker", () => {
  const repo = fixture();
  try {
    const config = join(repo.dir, "browser.config.json");
    writeFileSync(config, JSON.stringify({ browser: ["public/**", "**/*.css"] }));
    write(repo.dir, CSS, ":root { --warm: #fefefe; }\n");
    const out = runRouter(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--no-deps",
      "--quiet",
      "--config",
      config,
    ]);
    assert.match(
      out,
      /^BROWSER_UNVERIFIED: public\/styles\.css \(Chrome gate blocked by web-ai-showcase-jdz\)$/m,
    );
    assert.match(out, /^fast-gate: fallback: mode=all reason=unmapped/m);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("CLI: an uncommitted working-tree edit is judged, not reported as 'nothing changed'", () => {
  // A lane that runs the gate before committing must not get a silent green "no-tests" pass.
  const repo = fixture();
  try {
    write(repo.dir, "lib/a.mjs", "export const a = 9;\n");
    const planned = routeJson(repo.dir, ["--base", "origin/main", "--no-fetch"]);
    assert.equal(planned.changed.length, 1);
    assert.deepEqual(planned.run, ["test/a.test.mjs"]);

    const committedOnly = routeJson(repo.dir, [
      "--base",
      "origin/main",
      "--no-fetch",
      "--committed-only",
    ]);
    assert.deepEqual(committedOnly.changed, [], "--committed-only is the explicit opt-out");
    assert.deepEqual(committedOnly.run, [], "and it must not invent a fallback for an empty diff");
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});

test("changedPaths reads the working tree against the merge base, with both rename sides", () => {
  const repo = fixture();
  try {
    repo.run(repo.dir, "mv", "lib/a.mjs", "lib/moved.mjs");
    repo.run(repo.dir, "add", "-A");
    const { entries, error } = changedPaths(repo.dir, "origin/main");
    assert.equal(error, null);
    assert.deepEqual(entries, [{ status: "R", path: "lib/moved.mjs", oldPath: "lib/a.mjs" }]);
  } finally {
    repo.cleanup();
    repo.cleanupBare();
  }
});
