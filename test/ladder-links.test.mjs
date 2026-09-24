// Deterministic tests for the ladder-link gate (web-ai-showcase-500).
//
// Acceptance criteria 3 and 4 of that bead live here as durable proof:
//   * a fake advertised href on a built route makes the gate exit non-zero;
//   * removing it makes the gate exit zero again;
//   * a real directory with no index.html also fails (the class the bengali 404 belonged to);
//   * an advertised rung absent from the family's acceptance.json is REPORTED as an advisory
//     while the gate still exits zero — and nothing is written into the manifest.
//
// The gate is run against a throw-away --root fixture tree, so these tests never touch the real
// catalogue and run in the same download-free node suite as the rest of the gates.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const GATE = fileURLToPath(new URL("../scripts/check-ladder-links.mjs", import.meta.url));
const SLUG = "fixture-family";

// A fixture family that advertises every rung it has on disk.
const OVERVIEW = `<!doctype html><html><body>
  <a href="basics/">Basics</a>
  <a href="practical/">Practical</a>
  <a href="#top">Top</a>
  <a href="https://example.com/elsewhere">External</a>
  <a href="/web-ai-showcase/">Site root</a>
  <a href="../other-family/">Sibling family</a>
  <a href="mailto:someone@example.com">Mail</a>
</body></html>`;

const makeRoot = () => {
  const root = mkdtempSync(join(tmpdir(), "ladder-links-"));
  mkdirSync(join(root, "models", SLUG, "basics"), { recursive: true });
  mkdirSync(join(root, "models", SLUG, "practical"), { recursive: true });
  writeFileSync(join(root, "models", SLUG, "basics", "index.html"), "<html>basics</html>");
  writeFileSync(join(root, "models", SLUG, "practical", "index.html"), "<html>practical</html>");
  writeFileSync(join(root, "models", SLUG, "index.html"), OVERVIEW);
  writeFileSync(
    join(root, "models", SLUG, "acceptance.json"),
    JSON.stringify({
      id: SLUG,
      rungs: [
        { rung: "overview", route: `models/${SLUG}/`, inside: true },
        { rung: "basics", route: `models/${SLUG}/basics/` },
        { rung: "practical", route: `models/${SLUG}/practical/` },
      ],
    }),
  );
  writeFileSync(
    join(root, "models.json"),
    JSON.stringify({
      models: [{ slug: SLUG, status: "built" }, { slug: "unbuilt", status: "pending" }],
    }),
  );
  return root;
};

/** Run the gate against the fixture root; returns {code, out}. */
const runGate = (root) => {
  try {
    const out = execFileSync("node", [GATE, "--root", root], { encoding: "utf8" });
    return { code: 0, out };
  } catch (e) {
    return { code: e.status ?? 1, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
};

const withRoot = (fn) => {
  const root = makeRoot();
  try {
    return fn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
};

test("green: a family whose advertised rungs all exist passes", () => {
  withRoot((root) => {
    const { code, out } = runGate(root);
    assert.equal(code, 0, out);
    assert.match(out, /PASS {2}every advertised in-family link on a built route resolves/);
    assert.match(out, /0 failure\(s\)/);
    // Out-of-scope link shapes must not be reported as failures.
    assert.doesNotMatch(out, /^FAIL/m);
    assert.match(out, /external\/site-absolute/);
  });
});

test("MUTANT red: an advertised href with no target fails the gate", () => {
  withRoot((root) => {
    const page = join(root, "models", SLUG, "index.html");
    writeFileSync(
      page,
      OVERVIEW.replace('<a href="basics/">', '<a href="multi-model/">Basics</a><a href="basics/">'),
    );
    const { code, out } = runGate(root);
    assert.equal(code, 1, out);
    assert.match(
      out,
      /FAIL {2}fixture-family: models\/fixture-family\/index\.html advertises href="multi-model\/" — missing/,
    );
  });
});

test("MUTANT red: a directory without index.html fails (the real 404 class)", () => {
  withRoot((root) => {
    mkdirSync(join(root, "models", SLUG, "wild"));
    writeFileSync(
      join(root, "models", SLUG, "index.html"),
      OVERVIEW.replace('<a href="basics/">', '<a href="wild/">Wild</a><a href="basics/">'),
    );
    const { code, out } = runGate(root);
    assert.equal(code, 1, out);
    assert.match(out, /advertises href="wild\/" — directory without index\.html/);
  });
});

test("mutant removal restores green (both directions proven)", () => {
  withRoot((root) => {
    const page = join(root, "models", SLUG, "index.html");
    const mutated = OVERVIEW.replace(
      '<a href="basics/">',
      '<a href="wild/">Wild</a><a href="basics/">',
    );
    writeFileSync(page, mutated);
    assert.equal(runGate(root).code, 1, "mutant must be red first");
    writeFileSync(page, OVERVIEW);
    const { code, out } = runGate(root);
    assert.equal(code, 0, out);
  });
});

test("advisory: an advertised rung missing from acceptance.json is reported, never written", () => {
  withRoot((root) => {
    mkdirSync(join(root, "models", SLUG, "wild"), { recursive: true });
    writeFileSync(join(root, "models", SLUG, "wild", "index.html"), "<html>wild</html>");
    writeFileSync(
      join(root, "models", SLUG, "index.html"),
      OVERVIEW.replace('<a href="basics/">', '<a href="wild/">Wild</a><a href="basics/">'),
    );
    const manifestPath = join(root, "models", SLUG, "acceptance.json");
    const before = readFileSync(manifestPath, "utf8");

    const { code, out } = runGate(root);
    assert.equal(code, 0, `advisory must not fail the gate:\n${out}`);
    assert.match(out, /ADVISORY {2}1 advertised-but-unenumerated rung\(s\)/);
    assert.match(out, /fixture-family: wild/);
    // Criterion 4: reporting must not manufacture an acceptance record.
    assert.equal(
      readFileSync(manifestPath, "utf8"),
      before,
      "gate must not modify acceptance.json",
    );
  });
});

test("a dangling link in a ladder page is caught, not only on the overview", () => {
  withRoot((root) => {
    writeFileSync(
      join(root, "models", SLUG, "basics", "index.html"),
      '<html><a href="missing-runner.js">runner</a></html>',
    );
    const { code, out } = runGate(root);
    assert.equal(code, 1, out);
    assert.match(
      out,
      /models\/fixture-family\/basics\/index\.html advertises href="missing-runner\.js" — missing/,
    );
  });
});

test("a pending (unpublished) family is out of scope", () => {
  withRoot((root) => {
    mkdirSync(join(root, "models", "unbuilt"), { recursive: true });
    writeFileSync(
      join(root, "models", "unbuilt", "index.html"),
      '<html><a href="ghost/">ghost</a></html>',
    );
    const { code, out } = runGate(root);
    assert.equal(code, 0, out);
    assert.doesNotMatch(out, /unbuilt/);
  });
});
