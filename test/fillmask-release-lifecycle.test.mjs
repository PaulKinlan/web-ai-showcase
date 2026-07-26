// Focused lifecycle test for the BERT-family fill-mask release family.
//
// These 10 demos adopt the shared loader's release/dispose contract (see lib/model-loader.js and
// commits 324cb48 / ea35391). Release is only GENUINE when: the engine is lazily (re)created inside
// load so a fresh worker exists per load, dispose actually terminates that worker (freeing the WASM
// heap + ONNX masked-LM session — never a silent no-op), and onDispose returns the visible controls
// to their pre-load disabled state and drops the engine reference. Four pages (camembert,
// chinese-roberta-wwm, german, portuguese) carry a SECOND loader for an on-demand comparison model;
// it must satisfy the same contract. This test asserts that wiring statically so a future edit
// can't silently regress a page into a fake "released" state.
//
// Run: node --test test/fillmask-release-lifecycle.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// slug -> { cls: main engine class, ctrl: text/code control id, langs: codebert's extra lang
// buttons, second: { var, cls, flag } for the on-demand comparison loader }
const FAMILY = {
  "albert-fill-mask": { cls: "FillMaskEngine", ctrl: "text" },
  "bert-fill-mask": { cls: "FillMaskEngine", ctrl: "text" },
  "camembert-fill-mask": {
    cls: "CamembertFillMaskEngine",
    ctrl: "text",
    second: { var: "xlm", cls: "XlmrFillMaskEngine", flag: "xlmReady" },
  },
  "chinese-bert-fill-mask": { cls: "ChineseFillMaskEngine", ctrl: "text" },
  "chinese-roberta-wwm-fill-mask": {
    cls: "ChineseRobertaFillMaskEngine",
    ctrl: "text",
    second: { var: "bert", cls: "ChineseBertFillMaskEngine", flag: "bertReady" },
  },
  "codebert-fill-mask": { cls: "CodeFillEngine", ctrl: "code", langs: true },
  "german-bert-fill-mask": {
    cls: "GermanFillMaskEngine",
    ctrl: "text",
    second: { var: "mbert", cls: "MbertFillMaskEngine", flag: "mlReady" },
  },
  "modernbert-fill-mask": { cls: "ModernBertEngine", ctrl: "text" },
  "portuguese-bert-fill-mask": {
    cls: "PortugueseFillMaskEngine",
    ctrl: "text",
    second: { var: "mbert", cls: "MbertFillMaskEngine", flag: "mlReady" },
  },
  "xlm-roberta-fill-mask": { cls: "XlmrFillMaskEngine", ctrl: "text" },
};

const read = (slug) => readFileSync(join(ROOT, "models", slug, "index.html"), "utf8");

// Locate the route-local JS module that defines `export class <Cls>` (second engines may live in a
// sibling route's module, e.g. camembert imports XlmrFillMaskEngine from xlm-roberta-fill-mask).
function classOwnsWorker(cls) {
  for (const slug of Object.keys(FAMILY)) {
    const dir = join(ROOT, "models", slug);
    for (const f of readdirSync(dir)) {
      if (!f.endsWith(".js")) continue;
      const mod = readFileSync(join(dir, f), "utf8");
      if (new RegExp(`export class ${cls}\\b`).test(mod)) {
        assert.match(
          mod,
          /this\.worker = new Worker\(/,
          `${slug}/${f} must own a real Worker for ${cls}`,
        );
        return;
      }
    }
  }
  assert.fail(`no route-local module defines export class ${cls}`);
}

function assertGenuineRelease(src, label, { varName = "engine", flag = "loaded", cls, controls }) {
  // Engine is lazy — no module-scope eager construction left behind (that would pin the worker for
  // the page's life with no way to reclaim it).
  assert.match(
    src,
    new RegExp(`^ {6}let ${varName} = null;$`, "m"),
    `${label}: ${varName} must be declared lazily as null`,
  );
  assert.doesNotMatch(
    src,
    new RegExp(`const ${varName} = new \\w+\\(\\)`),
    `${label}: no eager ${varName} construction may remain`,
  );

  // A fresh engine (fresh worker) is created at the top of load, so release-then-reload works.
  assert.match(
    src,
    new RegExp(`load: async \\(onProgress\\) => \\{\\n\\s*${varName} = new ${cls}\\(\\);`),
    `${label}: load must construct a fresh ${cls} before ${varName}.load()`,
  );

  // dispose is a REAL teardown of the worker created above — terminating it frees the WASM heap +
  // ONNX session. This is the assertion that fails if someone swaps in a no-op.
  assert.match(
    src,
    /dispose: \(e\) => e\.worker\.terminate\(\)/,
    `${label}: dispose must terminate the engine worker (genuine memory release, not a no-op)`,
  );
  assert.doesNotMatch(
    src,
    /dispose: [^,]*=>\s*\{\s*\}/,
    `${label}: dispose must not be an empty no-op`,
  );

  // onDispose returns the controls onReady enabled to disabled and drops the engine, so the
  // visible control state matches the released state (invariant 12).
  const onDispose = src.match(
    new RegExp(`onDispose: \\(\\) => \\{([\\s\\S]*?${varName} = null;\\n        \\},)`),
  );
  assert.ok(onDispose, `${label}: onDispose handler must exist and null the engine`);
  const body = onDispose[1];
  for (const id of controls) {
    assert.match(
      body,
      new RegExp(`\\$\\("${id}"\\)\\.disabled = true;`),
      `${label}: onDispose must disable ${id}`,
    );
  }
  assert.match(body, new RegExp(`${flag} = false;`), `${label}: onDispose must clear ${flag}`);
}

for (const [slug, cfg] of Object.entries(FAMILY)) {
  test(`${slug}: genuine, non-no-op release wiring`, () => {
    const src = read(slug);

    const controls = [cfg.ctrl, "insertMask", "topk"];
    assertGenuineRelease(src, `${slug} main loader`, {
      cls: cfg.cls,
      controls,
    });
    if (cfg.langs) {
      const onDispose = src.match(/onDispose: \(\) => \{([\s\S]*?engine = null;\n        \},)/);
      assert.match(
        onDispose[1],
        /for \(const b of \$\("langs"\)\.children\) b\.disabled = true;/,
        "onDispose must disable the language starter buttons onReady enabled",
      );
    }

    if (cfg.second) {
      const s = cfg.second;
      assertGenuineRelease(src, `${slug} comparison loader`, {
        varName: s.var,
        flag: s.flag,
        cls: s.cls,
        controls: [],
      });
      const onDispose = src.match(
        new RegExp(`onDispose: \\(\\) => \\{([\\s\\S]*?${s.var} = null;\\n        \\},)`),
      );
      assert.match(
        onDispose[1],
        /\$\("vsGrid"\)\.hidden = true;/,
        "comparison onDispose must hide the stale comparison grid",
      );
      classOwnsWorker(s.cls);
    }

    classOwnsWorker(cfg.cls);
  });
}

test("family size is the expected 10-page release increment", () => {
  assert.equal(Object.keys(FAMILY).length, 10);
});
