// Browser proof for gallery attribution: exercise real DOM click/selection on two viewports
// with hostile manifest values, without changing the checked-in manifest or downloading models.
import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  CDP,
  chromeAvailable,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  screenshot,
  setViewport,
  startServer,
} from "../scripts/browser.mjs";

const browserSkip = chromeAvailable()
  ? false
  : "no Chrome/Chromium resolvable — browser-driven test; install one or set CHROME_BIN";
let server, port, chrome, cdp;
try {
  if (!browserSkip) {
    ({ server, port } = await startServer());
    chrome = await launchChrome();
    cdp = new CDP(chrome.ws);
  }
  for (const [name, viewport] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
    await test(`gallery attribution rejects hostile markup and links in ${name}`, {
      skip: browserSkip,
    }, async () => {
      const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/`);
      try {
        await setViewport(cdp, page.sessionId, viewport);
        const { result, exceptionDetails } = await cdp.send(
          "Runtime.evaluate",
          {
            expression: `(async () => {
            const { createExampleGallery } = await import('/web-ai-showcase/lib/example-gallery.js');
            const manifest = await (await fetch('/web-ai-showcase/media/manifest.json')).json();
            const original = manifest.assets[0];
            const hostile = {
              ...original, id: 'hostile-attribution',
              creator: '<img src=x onerror="window.__galleryXss=true">',
              licenseName: '<svg onload="window.__galleryXss=true">',
              source: '<script>window.__galleryXss=true</script>',
              sourceUrl: 'javascript:window.__galleryXss=true',
            };
            const safe = { ...original, id: 'safe-relative-attribution',
              creator: 'Creator & Co', licenseName: 'CC-BY',
              source: 'Local credit', sourceUrl: '/web-ai-showcase/media/manifest.json' };
            const empty = { ...original, id: 'no-link-attribution',
              source: 'No link', sourceUrl: '' };
            const realFetch = window.fetch;
            window.fetch = async (url, ...opts) => String(url).endsWith('/web-ai-showcase/media/gallery-security-fixture.json')
              ? new Response(JSON.stringify({ assets: [hostile, safe, empty] }),
                { status: 200, headers: { 'content-type': 'application/json' } })
              : realFetch.call(window, url, ...opts);
            const mount = document.createElement('div');
            mount.id = 'gallery-security-proof';
            document.body.prepend(mount);
            try {
              await createExampleGallery({
                mount, manifestUrl: '/web-ai-showcase/media/gallery-security-fixture.json',
              });
              const buttons = [...mount.querySelectorAll('.exg-item')];
              buttons[0].click();
              const cap = () => mount.querySelector('.exg-caption small');
              let caption = cap();
              const hostileResult = {
                text: caption.textContent, anchors: caption.querySelectorAll('a').length,
                injectedNodes: caption.querySelectorAll('img, svg, script').length,
                executed: !!window.__galleryXss,
              };
              const overflow = document.documentElement.scrollWidth > document.documentElement.clientWidth;
              buttons[1].click();
              caption = cap();
              const safeResult = {
                href: caption.querySelector('a')?.href,
                text: caption.textContent,
                rel: caption.querySelector('a')?.rel,
              };
              buttons[2].click();
              caption = cap();
              const emptyResult = { text: caption.textContent, anchors: caption.querySelectorAll('a').length };
              return { hostileResult, safeResult, emptyResult, overflow };
            } finally {
              window.fetch = realFetch;
            }
          })()`,
            awaitPromise: true,
            returnByValue: true,
          },
          page.sessionId,
          30000,
        );
        assert.equal(exceptionDetails, undefined, JSON.stringify(exceptionDetails));
        const value = result.value;
        assert.equal(value.hostileResult.executed, false);
        assert.equal(value.hostileResult.injectedNodes, 0);
        assert.equal(value.hostileResult.anchors, 0);
        assert.match(value.hostileResult.text, /<img src=x onerror=/);
        assert.match(value.hostileResult.text, /<script>window/);
        assert.equal(
          value.safeResult.href,
          `http://127.0.0.1:${port}/web-ai-showcase/media/manifest.json`,
        );
        assert.match(value.safeResult.text, /Creator & Co.*Local credit/);
        assert.equal(value.safeResult.rel, "noopener noreferrer");
        assert.equal(value.emptyResult.anchors, 0);
        assert.match(value.emptyResult.text, /No link/);
        assert.equal(value.overflow, false);
        assert.deepEqual(page.errors, []);
        await screenshot(cdp, page.sessionId, `/tmp/yz5-gallery-attribution-${name}.png`);
      } finally {
        await closePage(cdp, page.targetId);
      }
    });
  }
} finally {
  chrome?.kill();
  server?.close();
}
