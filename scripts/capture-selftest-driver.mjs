// One-off verification driver for the capture-ux + media-pipeline self-test harness.
// Reuses scripts/browser.mjs (CDP + static server), loads the harness with NO camera/mic present,
// reads window.__selftestResults, and captures light+dark × desktop+mobile screenshots into
// /home/paulkinlan/tmp. Not part of the repo gates — an evidence generator for this build.

import {
  BASE,
  CDP,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const OUT = process.env.OUTDIR || "/home/paulkinlan/tmp";
const URL_PATH = BASE + "lib/__capture-selftest__/index.html";

async function setTheme(cdp, sessionId, scheme) {
  await cdp.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: scheme }],
  }, sessionId);
  await new Promise((r) => setTimeout(r, 250));
}

const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const url = `http://127.0.0.1:${port}${URL_PATH}`;
const { targetId, sessionId, errors, netFailures } = await openPage(cdp, url);

// Wait for the self-test to complete.
let done = false;
for (let i = 0; i < 60; i++) {
  done = await evalValue(cdp, sessionId, "window.__selftestDone === true");
  if (done) break;
  await new Promise((r) => setTimeout(r, 500));
}

const results = await evalValue(cdp, sessionId, "JSON.stringify(window.__selftestResults)");
const parsed = results ? JSON.parse(results) : null;

// Prove the AudioWorklet → bounded-chunk → feature-worker path delivers real frames, running the
// AudioContext under a synthesized USER GESTURE (userGesture:true) so it can start headless.
let liveAudio = null;
try {
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: "window.__test.runAudioLive()",
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  }, sessionId);
  liveAudio = result?.value;
} catch (e) {
  liveAudio = { error: e.message };
}

// Screenshot: initial states (light desktop). Reset first so panels show rationale/dropzone.
await evalValue(cdp, sessionId, "(window.__test.resetAll(), 'ok')");
await new Promise((r) => setTimeout(r, 300));
await setViewport(cdp, sessionId, DESKTOP);
await setTheme(cdp, sessionId, "light");
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-desktop-light.png`);
await setTheme(cdp, sessionId, "dark");
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-desktop-dark.png`);

// Trigger the camera request (→ denied/unavailable) and load the audio fallback, then screenshot the
// actual capture components (scrolled into view) so the rationale / failure / review states are seen.
const photoState = await evalValue(cdp, sessionId, "window.__test.triggerPhotoRequest()");
await evalValue(cdp, sessionId, "window.__test.loadAudioFallback()");
await new Promise((r) => setTimeout(r, 400));
const scrollToCaps =
  "(document.getElementById('cap-file-h').scrollIntoView({block:'start'}), 'ok')";
await setTheme(cdp, sessionId, "light");
await evalValue(cdp, sessionId, scrollToCaps);
await new Promise((r) => setTimeout(r, 200));
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-desktop-states.png`);
await setTheme(cdp, sessionId, "dark");
await evalValue(cdp, sessionId, scrollToCaps);
await new Promise((r) => setTimeout(r, 200));
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-desktop-states-dark.png`);

// Mobile viewport — capture the components region.
await setViewport(cdp, sessionId, MOBILE);
await setTheme(cdp, sessionId, "light");
await evalValue(cdp, sessionId, scrollToCaps);
await new Promise((r) => setTimeout(r, 300));
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-mobile-light.png`);
await setTheme(cdp, sessionId, "dark");
await evalValue(cdp, sessionId, scrollToCaps);
await new Promise((r) => setTimeout(r, 300));
await screenshot(cdp, sessionId, `${OUT}/capture-selftest-mobile-dark.png`);

const gumAtLoad = await evalValue(cdp, sessionId, "window.__selftestResults.gumCallsAtLoad");

await closePage(cdp, targetId);
chrome.kill();
server.close();

console.log("\n=== SELF-TEST RESULTS ===");
if (parsed) {
  for (const t of parsed.tests) {
    console.log(`${t.pass ? "PASS" : "FAIL"}  ${t.name}  — ${t.detail}`);
  }
  console.log(`\nSummary: ${parsed.summary}`);
  console.log(`imagePipeline: ${JSON.stringify(parsed.imagePipeline)}`);
  console.log(`audioMode: ${parsed.audioMode}`);
  console.log(`live audio (user-gesture): ${JSON.stringify(liveAudio)}`);
} else {
  console.log("No results captured (selftestDone=" + done + ")");
}
console.log(`\ngumCallsAtLoad (auto-permission check): ${gumAtLoad}`);
console.log(`photoState after user-initiated request: ${photoState}`);
console.log(`console errors: ${errors.length ? JSON.stringify(errors) : "none"}`);
console.log(`network failures: ${netFailures.length ? JSON.stringify(netFailures) : "none"}`);
console.log(`screenshots written to ${OUT}`);
const allPass = parsed && parsed.tests.every((t) => t.pass) && gumAtLoad === 0;
process.exit(allPass ? 0 : 1);
