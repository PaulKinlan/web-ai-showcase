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
    let chrome = null;

    try {
      chrome = await launchChrome();
      const cdp = new CDP(chrome.ws);

      await t.test("real successful navigation loads document and updates location", async () => {
        const url = `http://127.0.0.1:${port}/web-ai-showcase/`;
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

          const status = await evalValue(
            cdp,
            page.sessionId,
            "performance.getEntriesByType('navigation')[0]?.responseStatus",
          );
          assert.equal(status, 200, "navigation responseStatus must be HTTP 200");

          const title = await evalValue(cdp, page.sessionId, "document.title");
          assert.match(
            title,
            /Web AI Showcase/,
            "document.title must contain expected showcase title",
          );

          const heading = await evalValue(
            cdp,
            page.sessionId,
            "document.querySelector('h1')?.textContent",
          );
          assert.match(
            heading,
            /Every model, running in your browser/,
            "h1 heading must match expected showcase index content",
          );
        } finally {
          await closePage(cdp, page.targetId);
        }
      });

      await t.test("failing navigation fails closed on Page.navigate errorText and closes target", async () => {
        const { targetInfos: beforeTargets } = await cdp.send("Target.getTargets");
        const beforeIds = beforeTargets.map((t) => t.targetId).sort();
        const beforePageCount = beforeTargets.filter((t) => t.type === "page").length;

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

        // Target closure in Chrome is asynchronous; poll briefly until target count stabilizes.
        const deadline = Date.now() + 5000;
        let afterTargets = [];
        while (Date.now() < deadline) {
          const res = await cdp.send("Target.getTargets");
          afterTargets = res.targetInfos;
          if (afterTargets.length === beforeTargets.length) {
            break;
          }
          await new Promise((r) => setTimeout(r, 50));
        }

        assert.equal(
          afterTargets.length,
          beforeTargets.length,
          "isolated target count must remain unchanged after failed navigation",
        );

        const afterPageCount = afterTargets.filter((t) => t.type === "page").length;
        assert.equal(
          afterPageCount,
          beforePageCount,
          "isolated page target count must remain unchanged after failed navigation",
        );

        const afterIds = afterTargets.map((t) => t.targetId).sort();
        assert.deepEqual(
          afterIds,
          beforeIds,
          "Target.getTargets list must match before and after failed openPage (target must be closed)",
        );
      });
    } finally {
      try {
        if (chrome) {
          await chrome.kill();
        }
      } finally {
        server.close();
      }
    }
  },
);
