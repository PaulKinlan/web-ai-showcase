// Focused tests for real Chrome navigation and fail-closed errorText handling (web-ai-showcase-c3h).
// Bounded execution, single sequential browser, full process and server cleanup.
import test from "node:test";
import assert from "node:assert/strict";
import {
  CDP,
  chromeAvailable,
  closePage,
  evalValue,
  launchChrome,
  openPage,
  startServer,
} from "../scripts/browser.mjs";

const browserSkip = chromeAvailable()
  ? false
  : "no Chrome/Chromium resolvable — browser-driven test; install one or set CHROME_BIN";

test(
  "Chrome sequential navigation lifecycle",
  { skip: browserSkip, timeout: 30_000 },
  async (t) => {
    const { server, port } = await startServer();
    const chrome = await launchChrome();
    const cdp = new CDP(chrome.ws);

    try {
      await t.test("real successful navigation loads document and updates location", async () => {
        const url = `http://127.0.0.1:${port}/web-ai-showcase/models/`;
        const page = await openPage(cdp, url);
        try {
          assert.ok(page.sessionId, "openPage must return an active sessionId");
          assert.ok(page.targetId, "openPage must return an active targetId");
          assert.deepEqual(page.errors, [], "real page load must not produce console errors");

          const href = await evalValue(cdp, page.sessionId, "window.location.href");
          assert.equal(
            href,
            url,
            "window.location.href must match target URL and not remain about:blank",
          );
        } finally {
          await closePage(cdp, page.targetId);
        }
      });

      await t.test("failing navigation fails closed on Page.navigate errorText and closes target", async () => {
        // Unsafe port 1 (tcpmux) is blocked by Chrome with net::ERR_UNSAFE_PORT, returning immediate errorText.
        const badUrl = "http://127.0.0.1:1/nonexistent";
        await assert.rejects(
          async () => {
            await openPage(cdp, badUrl);
          },
          (err) => {
            assert.match(
              err.message,
              /Page\.navigate failed:\s+net::ERR_/,
              "must fail closed with Page.navigate errorText",
            );
            return true;
          },
          "openPage must reject when Page.navigate returns errorText",
        );
      });
    } finally {
      await chrome.kill();
      server.close();
    }
  },
);
