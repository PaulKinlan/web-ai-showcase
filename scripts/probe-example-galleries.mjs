// Criterion 4: drive every gallery route in headless Chrome, desktop + mobile, and assert the
// example gallery actually RENDERS images rather than the honest-failure message.
//
// Viewport is applied BEFORE the page is navigated (a post-load override measures a layout the
// page never actually rendered into), and the gallery is POLLED rather than slept on, so a slow
// fetch is not read as "no images".
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

const ROUTES = [
  "animegan-cartoonization",
  "ddcolor-image-colorization",
  "fast-neural-style-transfer",
  "gfpgan-face-restoration",
  "iat-low-light-enhancement",
  "informative-drawings-lineart",
  "lama-image-inpainting",
  "microdehaze-image-dehazing",
  "nafnet-image-deblurring",
  "real-esrgan-super-resolution",
  "scunet-image-denoising",
  "uvdoc-document-dewarping",
];

const MEASURE = `(() => {
  const t = document.body.innerText;
  const m = t.match(/Example gallery unavailable[^\\n]*/);
  const imgs = [...document.querySelectorAll(".exg img")];
  let broken = 0;
  for (const i of imgs) if (i.complete && i.naturalWidth === 0) broken++;
  return JSON.stringify({
    unavailable: m ? m[0] : null,
    galleryImages: imgs.length,
    brokenImages: broken,
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  });
})()`;

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const results = [];

for (const slug of ROUTES) {
  for (const [vpName, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
    const url = `http://127.0.0.1:${port}/web-ai-showcase/models/${slug}/`;
    let targetId = null;
    try {
      const page = await openPage(cdp, "about:blank");
      targetId = page.targetId;
      const sessionId = page.sessionId;
      await setViewport(cdp, sessionId, vp); // before navigation
      await cdp.send("Page.navigate", { url }, sessionId, 30000);
      // Poll for the gallery instead of a fixed sleep.
      let v = {};
      for (let i = 0; i < 25; i++) {
        await new Promise((r) => setTimeout(r, 400));
        const { result } = await cdp.send(
          "Runtime.evaluate",
          { expression: MEASURE, returnByValue: true },
          sessionId,
          20000,
        );
        v = JSON.parse(result?.value ?? "{}");
        if (v.galleryImages > 0 || v.unavailable) break;
      }
      const errors = page.errors || [];
      const pass = !v.unavailable && v.galleryImages > 0 && v.brokenImages === 0 &&
        errors.length === 0;
      results.push({ slug, viewport: vpName, pass, ...v, consoleErrors: errors.length });
      process.stderr.write(
        `${
          pass ? "PASS" : "FAIL"
        } ${slug} ${vpName} imgs=${v.galleryImages} broken=${v.brokenImages} overflow=${v.overflow} err=${errors.length}${
          v.unavailable ? " :: " + v.unavailable : ""
        }\n`,
      );
    } catch (err) {
      results.push({ slug, viewport: vpName, pass: false, error: String(err.message || err) });
      process.stderr.write(`ERROR ${slug} ${vpName}: ${err.message}\n`);
    } finally {
      if (targetId) {
        try {
          await closePage(cdp, targetId);
        } catch { /* ignore */ }
      }
    }
  }
}
chrome.kill();
server.close();
const failed = results.filter((r) => !r.pass);
console.log(`\n=== ${results.length - failed.length}/${results.length} cells PASS ===`);
for (const f of failed) console.log("  FAIL", JSON.stringify(f));
console.log("\nOVERFLOW (reported, not gated here — layout is not this bead's scope):");
for (const r of results) if (r.overflow) console.log(`  ${r.slug} ${r.viewport}: ${r.overflow}px`);
process.exit(failed.length ? 1 : 0);
