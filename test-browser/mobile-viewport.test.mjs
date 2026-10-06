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
        mutantTarget: el.getAttribute("data-mutant-target") || null,
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

// Clean panel baseline check for the mutant proof below. Asserts that before any injection, no
// visible in-flow control inside a panel already escapes the panel's content edges (within a 1px
// subpixel rendering tolerance matching the guard).
const PANEL_BASELINE_EXPR = `(() => {
  const panelFor = (el) => {
    const panel = el.closest(".panel");
    return !panel || panel === el ? null : panel;
  };
  const panelContentEdges = (panel) => {
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
      const inFlow = cs.position === "static" || cs.position === "relative";
      return r.width > 0 && r.height > 0 && cs.display !== "none" && cs.visibility !== "hidden" &&
        inFlow && panelFor(el) !== null;
    });
  const escaping = [];
  for (const el of controls) {
    const edges = panelContentEdges(panelFor(el));
    const r = el.getBoundingClientRect();
    const overRight = +(r.right - edges.right).toFixed(2);
    const overLeft = +(edges.left - r.left).toFixed(2);
    if (overRight > 1 || overLeft > 1) {
      escaping.push({
        tag: el.tagName.toLowerCase(),
        id: el.id || null,
        right: +r.right.toFixed(2),
        left: +r.left.toFixed(2),
        contentRight: +edges.right.toFixed(2),
        contentLeft: +edges.left.toFixed(2),
        overRight,
        overLeft,
      });
    }
  }
  return {
    controlsCount: controls.length,
    escaping,
    clean: escaping.length === 0,
  };
})()`;

// Right-edge panel-escape mutant for the proof below. Selects a visible in-flow control inside a
// panel that is currently CONTAINED (right edge within panel content right + 1px tolerance),
// preferring the control closest to the panel's right content edge. Attaches a unique
// `data-mutant-target` attribute token so measurements, the premise, and the final guard verdict
// bind to the same control identity. Derives the injection offset from measured geometry
// (panel content right - control right + 40px, min 40px) to guarantee escape past the panel's
// right content edge. Returns independent before/after containment measurements and target token.
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
      const panel = panelFor(el);
      if (!offsetSettable || !panel || r.width <= 0 || r.height <= 0 || cs.display === "none" || cs.visibility === "hidden") {
        return false;
      }
      // Require the candidate control to be contained beforehand (within 1px tolerance)
      const pr = panelContentRight(panel);
      return r.right <= pr + 1;
    });
  if (controls.length === 0) {
    return { found: false, reason: "no visible in-flow contained control inside a .panel" };
  }
  let best = null;
  for (const el of controls) {
    const gap = panelContentRight(panelFor(el)) - el.getBoundingClientRect().right;
    if (best === null || Math.abs(gap) < Math.abs(best.gap)) best = { el, gap };
  }
  const targetToken = "mutant-target-" + Math.random().toString(36).slice(2, 10);
  best.el.setAttribute("data-mutant-target", targetToken);
  const panel = panelFor(best.el);
  const panelRect = panel.getBoundingClientRect();
  const btnRect = best.el.getBoundingClientRect();
  const contentRight = panelContentRight(panel);
  const offset = Math.max(panelRect.right - btnRect.right + 40, 40);
  best.el.style.position = "relative";
  best.el.style.left = offset + "px";
  const after = best.el.getBoundingClientRect();
  return {
    found: true,
    tag: best.el.tagName.toLowerCase(),
    id: best.el.id || null,
    targetToken,
    offset: +offset.toFixed(2),
    contentRight: +contentRight.toFixed(2),
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

    // 1. Clean panel baseline assertion: verify independently of CHECK_EXPR that no visible
    // in-flow control inside a panel already escapes its content box before injection.
    const baseline = await evalValue(cdp, sessionId, PANEL_BASELINE_EXPR);

    // 2. Inject a right-edge escape mutant on a contained control. The mutant binds a unique
    // data-mutant-target attribute and derives the offset from measured geometry.
    const injected = await evalValue(cdp, sessionId, RIGHT_EDGE_ESCAPE_MUTANT);
    const mutated = injected && injected.found ? await evalValue(cdp, sessionId, CHECK_EXPR) : null;
    await closePage(cdp, targetId);

    // Verify baseline was clean before injection
    assert.ok(
      baseline && baseline.clean,
      `HARNESS: panel baseline is not clean — pre-existing escaping controls detected before injection: ${JSON.stringify(
        baseline?.escaping ?? null,
      )}`,
    );

    // 3. Harness premise assertions (existence + independent before/after containment measurements):
    assert.ok(
      injected && injected.found,
      `HARNESS: could not select a right-edge panel-escape target — ${
        injected?.reason ?? "injection expression produced no result"
      }`,
    );

    const TOLERANCE = 1.0; // 1px subpixel rendering tolerance matching guard
    assert.ok(
      injected.beforeRight <= injected.contentRight + TOLERANCE,
      `HARNESS: selected control was not contained before injection — proof premise not met. Selected ${
        injected.tag
      }${injected.id ? "#" + injected.id : ""} (beforeRight: ${
        injected.beforeRight
      }, contentRight: ${injected.contentRight}, tolerance: ${TOLERANCE}px)`,
    );
    assert.ok(
      injected.afterRight > injected.contentRight + TOLERANCE,
      `HARNESS: the mutant did not escape — proof premise not met. Injected ${injected.tag}${
        injected.id ? "#" + injected.id : ""
      } by ${injected.offset}px (control right: ${injected.beforeRight} -> ${
        injected.afterRight
      }, panel content right: ${injected.contentRight}, tolerance: ${TOLERANCE}px)`,
    );

    // 4. Guard verdict assertions: verify CHECK_EXPR reports pass:false AND flags the
    // specific injected control (bound by targetToken) as escaping the right panel edge.
    assert.equal(
      mutated.pass,
      false,
      "GUARD: the containment guard failed to report pass:false for a control driven past the panel's right content edge",
    );
    assert.ok(
      mutated.escaping.some(
        (c) => c.mutantTarget === injected.targetToken && c.overPanelRight > 1,
      ),
      `GUARD: the containment guard did not flag the injected control as escaping the panel's right content edge (target: ${
        injected.targetToken
      }, guard escaping: ${JSON.stringify(mutated.escaping)})`,
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
