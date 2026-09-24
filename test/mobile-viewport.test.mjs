import test from "node:test";
import assert from "node:assert/strict";
import {
  CDP,
  closePage,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "../scripts/browser.mjs";

const SUBCLASS_B_ROUTES = [
  "models/bge-sentence-similarity/",
  "models/labse-embeddings/",
  "models/nomic-embeddings/",
  "models/nomic-embeddings/basics/",
  "models/e5-embeddings/basics/",
  "models/m2m100-translation/",
  "models/m2m100-translation/basics/",
  "models/marianmt-translation/",
  "models/marianmt-translation/basics/",
  "models/gemma-2-2b-webllm/practical/",
  "models/gemma-3-webllm/practical/",
  "models/mistral-webllm/practical/",
  "models/smollm2-chat/practical/",
  "models/stablelm-webllm/practical/",
];

const SUBCLASS_A_SAMPLE_ROUTES = [
  "models/clipseg-text-segmentation/",
  "models/codegen-350m/practical/",
];

const CHECK_EXPR = `(() => {
  const requested = 360;
  const vw = window.innerWidth;
  const sw = document.documentElement.scrollWidth;
  const panelContentEdges = (el) => {
    const panel = el.closest(".panel");
    if (!panel || panel === el) return null;
    const pr = panel.getBoundingClientRect();
    const cs = getComputedStyle(panel);
    return {
      left: pr.left + parseFloat(cs.paddingLeft || 0) + parseFloat(cs.borderLeftWidth || 0),
      right: pr.right - parseFloat(cs.paddingRight || 0) - parseFloat(cs.borderRightWidth || 0),
    };
  };
  const controls = [...document.querySelectorAll("button, input, select, textarea, label, output")]
    .filter((el) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      return r.width > 0 && r.height > 0 && cs.display !== "none" && cs.visibility !== "hidden";
    });
  const escaping = [];
  for (const el of controls) {
    const r = el.getBoundingClientRect();
    const edges = panelContentEdges(el);
    const overViewportRight = +(r.right - vw).toFixed(2);
    const overViewportLeft = +(0 - r.left).toFixed(2);
    const overPanelRight = edges === null ? null : +(r.right - edges.right).toFixed(2);
    const overPanelLeft = edges === null ? null : +(edges.left - r.left).toFixed(2);
    if (overViewportRight > 1 || overViewportLeft > 1 ||
      (overPanelRight !== null && overPanelRight > 1) || (overPanelLeft !== null && overPanelLeft > 1)) {
      escaping.push({
        tag: el.tagName.toLowerCase(),
        id: el.id || null,
        overViewportRight,
        overViewportLeft,
        overPanelRight,
        overPanelLeft
      });
    }
  }
  return {
    requested,
    innerWidth: vw,
    expanded: vw !== requested,
    scrollWidth: sw,
    overflow: sw - vw,
    controlsChecked: controls.length,
    escaping,
    pass: vw === requested && escaping.length === 0 && (sw - vw) <= 1
  };
})()`;

test("360px mobile viewport: all 14 sub-class B routes fit within viewport and panel content box", async (t) => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    for (const r of SUBCLASS_B_ROUTES) {
      await t.test(`route fits 360px: ${r}`, async () => {
        const url = `http://127.0.0.1:${port}/web-ai-showcase/${r}`;
        const { targetId, sessionId, errors } = await openPage(cdp, url);
        await setViewport(cdp, sessionId, MOBILE);
        await new Promise((res) => setTimeout(res, 500));

        const result = await evalValue(cdp, sessionId, CHECK_EXPR);
        await closePage(cdp, targetId);

        assert.equal(errors.length, 0, `Route ${r} produced console errors: ${errors.join(" | ")}`);
        assert.equal(result.expanded, false, `Route ${r} widened viewport to ${result.innerWidth}px`);
        assert.equal(result.innerWidth, 360, `Route ${r} innerWidth is ${result.innerWidth}px, expected 360`);
        assert.ok(result.overflow <= 1, `Route ${r} has horizontal overflow: ${result.overflow}px`);
        assert.equal(result.escaping.length, 0, `Route ${r} has escaping controls: ${JSON.stringify(result.escaping)}`);
        assert.equal(result.pass, true, `Route ${r} failed 360px containment check`);
      });
    }
  } finally {
    await browser.kill();
    server.close();
  }
});

test("360px mobile viewport: sub-class A sample routes fit within 360px viewport", async (t) => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    for (const r of SUBCLASS_A_SAMPLE_ROUTES) {
      await t.test(`sample route fits 360px: ${r}`, async () => {
        const url = `http://127.0.0.1:${port}/web-ai-showcase/${r}`;
        const { targetId, sessionId, errors } = await openPage(cdp, url);
        await setViewport(cdp, sessionId, MOBILE);
        await new Promise((res) => setTimeout(res, 500));

        const result = await evalValue(cdp, sessionId, CHECK_EXPR);
        await closePage(cdp, targetId);

        assert.equal(errors.length, 0, `Route ${r} produced console errors: ${errors.join(" | ")}`);
        assert.equal(result.expanded, false, `Route ${r} widened viewport to ${result.innerWidth}px`);
        assert.equal(result.innerWidth, 360, `Route ${r} innerWidth is ${result.innerWidth}px, expected 360`);
        assert.equal(result.escaping.length, 0, `Route ${r} has escaping controls: ${JSON.stringify(result.escaping)}`);
      });
    }
  } finally {
    await browser.kill();
    server.close();
  }
});

test("MUTANT PROOF: guard detects viewport widening (innerWidth !== 360)", async () => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/bge-sentence-similarity/`;
    const { targetId, sessionId } = await openPage(cdp, url);
    await setViewport(cdp, sessionId, MOBILE);

    // Verify clean baseline
    const clean = await evalValue(cdp, sessionId, CHECK_EXPR);
    assert.equal(clean.pass, true);

    // Inject viewport-widening mutant
    await evalValue(
      cdp,
      sessionId,
      `(() => {
        const d = document.createElement("div");
        d.id = "mutant-widener";
        d.style.width = "480px";
        d.style.height = "10px";
        d.style.minWidth = "480px";
        document.body.prepend(d);
      })()`,
    );

    const mutated = await evalValue(cdp, sessionId, CHECK_EXPR);
    await closePage(cdp, targetId);

    // Assert that the mutant was caught (either expanded === true or overflow > 1)
    const caught = mutated.expanded === true || mutated.overflow > 1 || mutated.pass === false;
    assert.ok(caught, "Guard failed to detect viewport widening mutant");
  } finally {
    await browser.kill();
    server.close();
  }
});

test("MUTANT PROOF: guard detects right-edge panel content box escape", async () => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/bge-sentence-similarity/`;
    const { targetId, sessionId } = await openPage(cdp, url);
    await setViewport(cdp, sessionId, MOBILE);

    // Inject right-edge escape mutant
    await evalValue(
      cdp,
      sessionId,
      `(() => {
        const btn = document.querySelector(".panel button") || document.querySelector("button");
        btn.style.position = "relative";
        btn.style.left = "80px";
      })()`,
    );

    const mutated = await evalValue(cdp, sessionId, CHECK_EXPR);
    await closePage(cdp, targetId);

    assert.equal(mutated.pass, false, "Guard failed to detect right-edge panel escape mutant");
    assert.ok(mutated.escaping.length > 0, "No escaping controls reported for right-edge mutant");
    assert.ok(mutated.escaping.some((c) => c.overPanelRight > 1), "overPanelRight was not flagged");
  } finally {
    await browser.kill();
    server.close();
  }
});

test("MUTANT PROOF: guard detects left-edge panel content box escape", async () => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/bge-sentence-similarity/`;
    const { targetId, sessionId } = await openPage(cdp, url);
    await setViewport(cdp, sessionId, MOBILE);

    // Inject left-edge escape mutant
    await evalValue(
      cdp,
      sessionId,
      `(() => {
        const btn = document.querySelector(".panel button") || document.querySelector("button");
        btn.style.position = "relative";
        btn.style.left = "-60px";
      })()`,
    );

    const mutated = await evalValue(cdp, sessionId, CHECK_EXPR);
    await closePage(cdp, targetId);

    assert.equal(mutated.pass, false, "Guard failed to detect left-edge panel escape mutant");
    assert.ok(mutated.escaping.length > 0, "No escaping controls reported for left-edge mutant");
    assert.ok(mutated.escaping.some((c) => c.overPanelLeft > 1), "overPanelLeft was not flagged");
  } finally {
    await browser.kill();
    server.close();
  }
});
