// Hostile ledger sourceUrl values must never become a live href on image-credits/ or via image-credit.js.
import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  CDP,
  chromeAvailable,
  closePage,
  launchChrome,
  openPage,
  startServer,
} from "../scripts/browser.mjs";

const skip = chromeAvailable() ? false : "no Chrome/Chromium resolvable — browser-driven test";
const HOSTILE = [
  "javascript:window.__x=1",
  "JaVaScRiPt:window.__x=1",
  "java\tscript:window.__x=1",
  "data:text/html,<b>1</b>",
  "vbscript:x",
  "//evil.test/p",
  "\\\\evil.test/p",
  "https:evil.test/p",
  "",
];
const STUB = `(() => { const real = window.fetch; const H = ${JSON.stringify(HOSTILE)}; let n = 0;
  window.fetch = async (u, ...r) => { const res = await real.call(window, u, ...r);
    if (!String(u).endsWith('image-provenance/ledger.json')) return res;
    const l = await res.json();
    for (const e of l.entries) if (e.provenance) e.provenance.sourceUrl = H[n++ % H.length];
    window.__stubbed = n;
    return new Response(JSON.stringify(l), { status: 200, headers: { 'content-type': 'application/json' } }); }; })()`;

test("hostile ledger sourceUrl yields no live script/cross-origin href", { skip }, async () => {
  let server, chrome, cdp, page;
  try {
    let port;
    ({ server, port } = await startServer());
    chrome = await launchChrome();
    cdp = new CDP(chrome.ws);
    const blank = await openPage(cdp, "about:blank");
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: STUB }, blank.sessionId);
    await cdp.send("Page.navigate", {
      url: `http://127.0.0.1:${port}/web-ai-showcase/image-credits/`,
    }, blank.sessionId);
    page = blank;
    let v;
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 400));
      const { result } = await cdp.send("Runtime.evaluate", {
        returnByValue: true,
        expression: `JSON.stringify({
        cards: document.querySelectorAll('figure.credit').length,
        bad: [...document.querySelectorAll('#sections a[href]')].map(a=>a.href).filter(h=>!/^https?:\\/\\/(127\\.0\\.0\\.1|commons\\.)/.test(h)),
        js: [...document.querySelectorAll('a[href]')].filter(a=>/^(javascript|data|vbscript):/i.test(a.getAttribute('href'))).length,
        executed: !!window.__x, stubbed: window.__stubbed|0 })`,
      }, blank.sessionId);
      v = JSON.parse(result.value);
      if (v.cards > 0) break;
    }
    assert.ok(v.cards > 0, "credit cards rendered");
    assert.ok(v.stubbed > 0, "hostile stub rewrote ledger sourceUrls");
    assert.deepEqual(v.bad, []);
    assert.equal(v.js, 0);
    assert.equal(v.executed, false);
  } finally {
    if (page) await closePage(cdp, page.targetId);
    chrome?.kill();
    server?.close();
  }
});
