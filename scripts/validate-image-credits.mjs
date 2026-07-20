#!/usr/bin/env node
// Browser validation (real headless Chrome) for the image-provenance surface:
//  - /image-credits/ renders from the ledger (cards, counts, no console errors, no horizontal overflow)
//  - the shared annotator (public/image-credit.js) adds a visible credit line + per-image title on demos
//    that display a licensed image (people demo + a non-people licensed demo), at desktop AND mobile.
import {
  closePage,
  evalValue,
  launchChrome,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

const BASE = "/web-ai-showcase/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}${BASE}`;
const { CDP } = await import("./browser.mjs");
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);

const results = [];
const rec = (name, pass, detail) => {
  results.push(!!pass);
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
};

try {
  // 1) /image-credits/ page
  let pg = await openPage(cdp, base + "image-credits/");
  await sleep(1500); // ledger fetch + render
  const credits = await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
    const sections = document.querySelectorAll('#sections section').length;
    const cards = document.querySelectorAll('.credit').length;
    const counts = (document.getElementById('counts')||{}).textContent || '';
    const overflow = document.documentElement.scrollWidth - window.innerWidth;
    const licBadges = document.querySelectorAll('.lic').length;
    const sourceLinks = [...document.querySelectorAll('.credit a')].filter(a=>/commons\\.wikimedia/.test(a.href)).length;
    return { sections, cards, counts, overflow, licBadges, sourceLinks };
  })()`,
  );
  rec(
    "/image-credits/ renders sections + credit cards",
    credits.sections >= 3 && credits.cards >= 10,
    JSON.stringify({ sections: credits.sections, cards: credits.cards }),
  );
  rec(
    "/image-credits/ shows counts + license badges + Commons source links",
    /image files/.test(credits.counts) && credits.licBadges >= 10 && credits.sourceLinks >= 5,
    `badges=${credits.licBadges} commonsLinks=${credits.sourceLinks}`,
  );
  rec(
    "/image-credits/ no console errors",
    pg.errors.length === 0,
    pg.errors.slice(0, 2).join(" | "),
  );
  rec(
    "/image-credits/ no horizontal overflow (desktop)",
    credits.overflow <= 2,
    `overflow=${credits.overflow}`,
  );
  // mobile
  await setViewport(cdp, pg.sessionId, {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await sleep(400);
  const ov = await evalValue(
    cdp,
    pg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
  );
  rec("/image-credits/ no horizontal overflow (mobile 390px)", ov <= 2, `overflow=${ov}`);
  await closePage(cdp, pg.targetId);

  // 2) Annotator (public/image-credit.js) — validated deterministically via a DOM fixture on a light
  //    page (real Chrome, real fetch, real DOM), isolated from demos whose heavy dual-worker model init
  //    pegs the main thread. This exercises the exact code path every demo page uses.
  pg = await openPage(cdp, base + "image-credits/");
  await sleep(300);
  await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
    const strip = '<div class="sample-strip">' +
      '<img class="sample-thumb" src="${BASE}models/face-embedding/person-b.jpg">' +      // identifiable person (CC-BY-SA)
      '<img class="sample-thumb" src="${BASE}models/food-classification/sample-pizza.jpg">' + // non-people licensed (CC-BY-SA)
      '<img class="sample-thumb" src="${BASE}models/mgp-str-ocr/sample-stop.png">' +        // first-party procedural (no credit)
      '</div>';
    document.querySelector('main').insertAdjacentHTML('beforeend', '<div id="fixture">' + strip + '</div>');
    return true;
  })()`,
  );
  // import the annotator fresh (cache-bust) so it runs against the fixture
  await evalValue(
    cdp,
    pg.sessionId,
    `import('${BASE}public/image-credit.js?fixture=' + (performance.now()|0)).then(()=>{})`,
  );
  await sleep(1200);
  const ann = await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
    const fx = document.getElementById('fixture');
    const credit = fx.querySelector('.img-credit');
    const imgs = [...fx.querySelectorAll('img.sample-thumb')];
    const licensedTitled = imgs.filter(i => /person-b|sample-pizza/.test(i.src) && i.title && i.title.length > 6).length;
    const proceduralTitled = imgs.filter(i => /sample-stop/.test(i.src) && i.title && i.title.length > 0).length;
    return {
      hasCredit: !!credit,
      creditText: credit ? credit.textContent.slice(0, 200) : '',
      hasAllCreditsLink: !!(credit && credit.querySelector('a[href*="image-credits"]')),
      hasCommonsLink: !!(credit && credit.querySelector('a[href*="commons.wikimedia"]')),
      licensedTitled,
      proceduralTitled,
    };
  })()`,
  );
  rec(
    "annotator renders a contextual credit line for licensed sample images",
    ann.hasCredit && /—|CC|Public|Commons/i.test(ann.creditText),
    ann.creditText,
  );
  rec(
    "annotator credit links to /image-credits/ and to the Commons source",
    ann.hasAllCreditsLink && ann.hasCommonsLink,
    "",
  );
  rec(
    "annotator sets per-image attribution (title) on both licensed images",
    ann.licensedTitled === 2,
    `${ann.licensedTitled}/2 titled`,
  );
  rec(
    "annotator leaves first-party procedural images uncredited (restraint)",
    ann.proceduralTitled === 0,
    `${ann.proceduralTitled} procedural titled (want 0)`,
  );
  rec(
    "annotator page has no console errors",
    pg.errors.length === 0,
    pg.errors.slice(0, 2).join(" | "),
  );
  await closePage(cdp, pg.targetId);
  // (The "no synthetic/StyleGAN claim" honesty regression is asserted deterministically by grep in
  //  test/image-provenance.test.mjs — more reliable than eval-ing a demo whose model thread is busy.)
} finally {
  chrome.kill();
  server.close();
}

const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} image-credits checks passed.`);
process.exit(passed === results.length ? 0 : 1);
