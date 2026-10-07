// web-ai-showcase-9z1: verify crashpad / breakpad suppression flags in test harness browser launch args.
// Pure static and argument generation tests — browser-free (verified by test/suite-stays-browser-free.test.mjs).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS,
  getChromeLaunchArgs,
} from "../scripts/browser.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

test("HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS exports recognized Chrome 154 flags", () => {
  assert.ok(Array.isArray(HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS), "must be an array");
  assert.ok(
    HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS.includes("--disable-breakpad"),
    "must include --disable-breakpad",
  );
  assert.ok(
    HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS.includes("--disable-crashpad-for-testing"),
    "must include --disable-crashpad-for-testing",
  );
  assert.ok(
    !HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS.includes("--disable-crash-reporter"),
    "must NOT include --disable-crash-reporter (absent/no-op on Chrome 154)",
  );
});

test("getChromeLaunchArgs incorporates suppression flags into default headless arguments", () => {
  const args = getChromeLaunchArgs();
  assert.ok(args.includes("--headless=new"), "must include --headless=new");
  assert.ok(args.includes("--no-sandbox"), "must include --no-sandbox");
  assert.ok(args.includes("--disable-dev-shm-usage"), "must include --disable-dev-shm-usage");
  assert.ok(args.includes("--hide-scrollbars"), "must include --hide-scrollbars");
  assert.ok(args.includes("--remote-debugging-port=0"), "must include --remote-debugging-port=0");
  assert.ok(args.includes("--remote-allow-origins=*"), "must include --remote-allow-origins=*");
  assert.ok(
    args.includes("--autoplay-policy=no-user-gesture-required"),
    "must include --autoplay-policy=no-user-gesture-required",
  );
  assert.ok(args.includes("--disable-breakpad"), "launch args must include --disable-breakpad");
  assert.ok(
    args.includes("--disable-crashpad-for-testing"),
    "launch args must include --disable-crashpad-for-testing",
  );
  assert.ok(
    !args.includes("--disable-crash-reporter"),
    "launch args must NOT include --disable-crash-reporter",
  );
  assert.equal(args.at(-1), "about:blank", "last argument must be about:blank");
});

test("getChromeLaunchArgs handles userDataDir correctly", () => {
  const targetDir = "/tmp/test-profile-12345";
  const args = getChromeLaunchArgs({ userDataDir: targetDir });
  assert.ok(args.includes(`--user-data-dir=${targetDir}`), "must include user-data-dir flag");
});

test("getChromeLaunchArgs respects webgpu option while preserving suppression flags", () => {
  const gpuDisabled = getChromeLaunchArgs({ webgpu: false });
  assert.ok(gpuDisabled.includes("--disable-gpu"));
  assert.ok(!gpuDisabled.includes("--enable-unsafe-webgpu"));
  assert.ok(gpuDisabled.includes("--disable-breakpad"));
  assert.ok(gpuDisabled.includes("--disable-crashpad-for-testing"));

  const gpuEnabled = getChromeLaunchArgs({ webgpu: true });
  assert.ok(!gpuEnabled.includes("--disable-gpu"));
  assert.ok(gpuEnabled.includes("--enable-unsafe-webgpu"));
  assert.ok(gpuEnabled.includes("--use-angle=vulkan"));
  assert.ok(gpuEnabled.includes("--enable-features=Vulkan"));
  assert.ok(gpuEnabled.includes("--disable-breakpad"));
  assert.ok(gpuEnabled.includes("--disable-crashpad-for-testing"));
});

test("getChromeLaunchArgs passes extraArgs through without duplicate GPU args", () => {
  const extra = ["--js-flags=--max-old-space-size=4096", "--custom-flag"];
  const args = getChromeLaunchArgs({ extraArgs: extra });
  assert.ok(args.includes("--js-flags=--max-old-space-size=4096"));
  assert.ok(args.includes("--custom-flag"));
  assert.ok(args.includes("--disable-breakpad"));
  assert.ok(args.includes("--disable-crashpad-for-testing"));
});

test("static check: scripts/browser.mjs source wires launch args into spawnChromeOnce", () => {
  const src = readFileSync(join(ROOT, "scripts/browser.mjs"), "utf8");

  // Verify the suppression constants and helpers are defined and exported
  assert.match(
    src,
    /export\s+const\s+HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS\s*=\s*\[/,
    "browser.mjs must export HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS",
  );
  assert.match(
    src,
    /export\s+function\s+getChromeLaunchArgs\s*\(/,
    "browser.mjs must export getChromeLaunchArgs helper",
  );

  // Verify flags are present in source
  assert.match(src, /"--disable-breakpad"/, "browser.mjs must contain --disable-breakpad");
  assert.match(
    src,
    /"--disable-crashpad-for-testing"/,
    "browser.mjs must contain --disable-crashpad-for-testing",
  );

  // Verify absent flag is NOT passed to Chrome
  assert.ok(
    !src.includes('"--disable-crash-reporter"'),
    "browser.mjs must NOT pass --disable-crash-reporter (absent/no-op in Chrome 154)",
  );

  // Verify spawnChromeOnce calls getChromeLaunchArgs and passes the result to spawn
  assert.match(
    src,
    /const\s+args\s*=\s*getChromeLaunchArgs\(\s*\{\s*userDataDir,\s*extraArgs,\s*webgpu\s*\}\s*\);/,
    "spawnChromeOnce must obtain args via getChromeLaunchArgs",
  );
  assert.match(
    src,
    /spawn\s*\(\s*findChrome\(\s*\)\s*,\s*args\s*,/,
    "spawnChromeOnce must pass getChromeLaunchArgs result to spawn",
  );
});
