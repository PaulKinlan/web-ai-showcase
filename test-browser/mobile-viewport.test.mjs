import test from "node:test";
import assert from "node:assert/strict";
import {
  CDP,
  chromeAvailable,
  closePage,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "../scripts/browser.mjs";

// This file is BROWSER-DRIVEN. It lives outside test/ so the CI step that runs the bare node suite
// stays browser-free and fast by construction (web-ai-showcase-cj5). When no browser is resolvable the
// tests SKIP with a stated reason instead of failing five times on spawn ENOENT after a retry storm.
const SKIP_NO_BROWSER =
  "no Chrome/Chromium resolvable — this is a browser-driven test; install one or set CHROME_BIN, " +
  "or run it via: deno task test:viewport";
const browserSkip = chromeAvailable() ? false : SKIP_NO_BROWSER;

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

// Right-edge panel-escape mutant for the proof below. The target control is chosen BY GEOMETRY —
// the visible in-flow control whose right edge sits nearest the panel's right edge — and the
// injection distance is DERIVED (panel right - control right + 40px), so the escape is a guarantee
// rather than a bet on a fixed 80px clearing that control's gap. `position: relative` is only used
// where `left` is a genuine offset (static/relative), so the applied shift is the measured one.
// Returns `{ found:false, reason }` instead of falling through to a different element when there is
// no candidate, so the harness fails loudly.
const RIGHT_EDGE_ESCAPE_MUTANT = `(() => {
  const panelFor = (el) => {
    const panel = el.closest(".panel");
    return !panel || panel === el ? null : panel;
  };
  const panelContentRight = (panel) => {
    const pr = panel.getBoundingClientRect();
    const cs = getComputedStyle(panel);
    return pr.right - parseFloat(cs.paddingRight || 0) - parseFloat(cs.borderRightWidth || 0);
  };
  const controls = [...document.querySelectorAll("button, input, select, textarea, label, output")]
    .filter((el) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      const offsetSettable = cs.position === "static" || cs.position === "relative";
      return r.width > 0 && r.height > 0 && cs.display !== "none" && cs.visibility !== "hidden" &&
        offsetSettable && panelFor(el) !== null;
    });
  if (controls.length === 0) {
    return { found: false, reason: "no visible in-flow control inside a .panel" };
  }
  let best = null;
  for (const el of controls) {
    const gap = panelContentRight(panelFor(el)) - el.getBoundingClientRect().right;
    if (best === null || Math.abs(gap) < Math.abs(best.gap)) best = { el, gap };
  }
  const panel = panelFor(best.el);
  const panelRect = panel.getBoundingClientRect();
  const btnRect = best.el.getBoundingClientRect();
  const offset = Math.max(panelRect.right - btnRect.right + 40, 40);
  best.el.style.position = "relative";
  best.el.style.left = offset + "px";
  const after = best.el.getBoundingClientRect();
  return {
    found: true,
    tag: best.el.tagName.toLowerCase(),
    id: best.el.id || null,
    offset: +offset.toFixed(2),
    contentRight: +panelContentRight(panel).toFixed(2),
    beforeRight: +btnRect.right.toFixed(2),
    afterRight: +after.right.toFixed(2),
  };
})()`;

test("360px mobile viewport: all 14 sub-class B routes fit within viewport and panel content box", { skip: browserSkip }, async (t) => {
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

test("360px mobile viewport: sub-class A sample routes fit within 360px viewport", { skip: browserSkip }, async (t) => {
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

test("MUTANT PROOF: guard detects viewport widening (innerWidth !== 360)", { skip: browserSkip }, async () => {
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

test("MUTANT PROOF: guard detects right-edge panel content box escape", { skip: browserSkip }, async () => {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);

  try {
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/bge-sentence-similarity/`;
    const { targetId, sessionId } = await openPage(cdp, url);
    await setViewport(cdp, sessionId, MOBILE);

    // Inject a right-edge escape mutant whose distance is DERIVED from the rendered geometry, so a
    // correct guard MUST report it. The selected control (and its existence) is reported back so a
    // failed premise can be attributed to the harness rather than blamed on the guard.
    const injected = await evalValue(cdp, sessionId, RIGHT_EDGE_ESCAPE_MUTANT);
    const mutated = injected && injected.found ? await evalValue(cdp, sessionId, CHECK_EXPR) : null;
    await closePage(cdp, targetId);

    assert.ok(
      injected && injected.found,
      `HARNESS: could not select a right-edge panel-escape target — ${
        injected?.reason ?? "injection expression produced no result"
      }`,
    );
    assert.ok(
      mutated && mutated.escaping.length > 0 && mutated.escaping.some((c) => c.overPanelRight > 1),
      `HARNESS: the mutant did not escape — proof premise not met. Injected ${injected.tag}${
        injected.id ? "#" + injected.id : ""
      } by ${injected.offset}px (panel content right ${injected.contentRight}; control right ${
        injected.beforeRight
      } -> ${injected.afterRight}); guard escaping: ${JSON.stringify(mutated?.escaping ?? null)}`,
    );
    assert.equal(
      mutated.pass,
      false,
      "GUARD: the containment guard failed to report pass:false for a control driven past the panel's right content edge",
    );
  } finally {
    await browser.kill();
    server.close();
  }
});

test("MUTANT PROOF: guard detects left-edge panel content box escape", { skip: browserSkip }, async () => {
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
