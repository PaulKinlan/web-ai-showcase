// Focused lifecycle test for the wav2vec2/XLSR CTC-ASR release family.
//
// These 12 demos adopt the shared loader's release/dispose contract (see lib/model-loader.js and
// commit 324cb48). Release is only GENUINE when: the engine is lazily (re)created inside load so a
// fresh worker exists per load, dispose actually terminates that worker (freeing the WASM heap +
// ONNX CTC session — never a silent no-op), and onDispose returns the visible controls to their
// pre-load disabled state and drops the engine reference. This test asserts that wiring statically
// so a future edit can't silently regress a page into a fake "released" state.
//
// Run: node --test test/asr-release-lifecycle.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const FAMILY = [
  "chinese-xlsr-asr",
  "french-xlsr-asr",
  "italian-xlsr-asr",
  "japanese-xlsr-asr",
  "korean-xlsr-asr",
  "portuguese-xlsr-asr",
  "spanish-xlsr-asr",
  "thai-xlsr-asr",
  "finnish-voxpopuli-asr",
  "polish-voxpopuli-asr",
  "xlsr-multilingual-asr",
  "wav2vec2-asr",
];

const read = (slug) => readFileSync(join(ROOT, "models", slug, "index.html"), "utf8");

for (const slug of FAMILY) {
  test(`${slug}: genuine, non-no-op release wiring`, () => {
    const src = read(slug);

    // Engine is lazy — no module-scope eager `const engine = new X()` left behind (that would
    // pin the worker for the page's life with no way to reclaim it).
    assert.match(src, /^ {6}let engine = null;$/m, "engine must be declared lazily as null");
    assert.doesNotMatch(
      src,
      /const engine = new \w+\(\)/,
      "no eager engine construction may remain",
    );

    // A fresh engine (fresh worker) is created at the top of load, so release-then-reload works.
    const loadCreate = src.match(/load: async \(onProgress\) => \{\n\s*engine = new (\w+)\(\);/);
    assert.ok(loadCreate, "load must construct a fresh engine before engine.load()");
    const Cls = loadCreate[1];

    // The init-time ready read is guarded (engine is null before the first load and after release).
    assert.match(
      src,
      /\$\("run"\)\.disabled = !\(engine && engine\.ready\);/,
      "the pre-load ready read must be null-guarded",
    );

    // dispose is a REAL teardown of the worker created above — terminating it frees the WASM heap +
    // ONNX session. This is the assertion that fails if someone swaps in a no-op.
    assert.match(
      src,
      /dispose: \(\) => engine\.worker\.terminate\(\)/,
      "dispose must terminate the engine worker (genuine memory release, not a no-op)",
    );

    // onDispose returns the controls onReady enabled (rec/upload/run) to disabled and drops the
    // engine so the visible control state matches the released state (invariant 12).
    const onDispose = src.match(/onDispose: \(\) => \{([\s\S]*?)\},/);
    assert.ok(onDispose, "onDispose handler must exist");
    const body = onDispose[1];
    for (const id of ["rec", "upload", "run"]) {
      assert.match(
        body,
        new RegExp(`\\$\\("${id}"\\)\\.disabled = true;`),
        `onDispose must disable ${id}`,
      );
    }
    assert.match(body, /engine = null;/, "onDispose must drop the engine reference");

    // Guard against a silent no-op dispose slipping in.
    assert.doesNotMatch(src, /dispose: \(\) => \{\s*\}/, "dispose must not be an empty no-op");

    // The engine module genuinely owns the worker dispose targets (this.worker = new Worker).
    const engineJs = src.match(/from "\.\/([a-z0-9]+\.js)"/);
    assert.ok(engineJs, "engine module import must be resolvable");
    const mod = readFileSync(join(ROOT, "models", slug, engineJs[1]), "utf8");
    assert.match(
      mod,
      /this\.worker = new Worker\(/,
      `${engineJs[1]} must own a real Worker for ${Cls}`,
    );
  });
}

test("family size is the expected 12-page release increment", () => {
  assert.equal(FAMILY.length, 12);
});
