#!/usr/bin/env node
// Real-browser validation for the homepage discovery/search work (headless Chrome via the repo's CDP
// harness — the same harness the conformance + responsive checks use). Not a unit test: it drives the
// actual page. Chrome DevTools MCP was not connected in this session; this is the honest equivalent.
//
// Checks (desktop + mobile): render + no horizontal overflow, unique stable section ids, every jump
// link resolves, direct fragment deep-link scrolls, search (built-only, ranking, no-results), clear,
// ?q= URL round-trip + back/forward, keyboard submit, console/network clean, and that /explore/ still
// loads. Writes reports/home-validation.json + screenshots.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  BASE,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const OUT = join(repoRoot, "reports");
mkdirSync(OUT, { recursive: true });
const results = [];
const rec = (name, pass, detail) => {
  results.push({ name, pass: !!pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function pollCount(cdp, sessionId, timeoutMs = 20000, want = null) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const txt = await evalValue(
      cdp,
      sessionId,
      `document.getElementById('search-count')?.textContent || ''`,
    );
    const ready = txt && /\bdemos?\b/.test(txt) && !/Loading/.test(txt);
    if (ready && (!want || want.test(txt))) return txt;
    await sleep(300);
  }
  return await evalValue(
    cdp,
    sessionId,
    `document.getElementById('search-count')?.textContent || ''`,
  );
}

async function typeQuery(cdp, sessionId, q) {
  await evalValue(
    cdp,
    sessionId,
    `(() => { const el = document.getElementById('q'); el.focus(); el.value=${
      JSON.stringify(q)
    }; el.dispatchEvent(new Event('input',{bubbles:true})); return true; })()`,
  );
}

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}${BASE}`;
let chrome;
try {
  chrome = await launchChrome();
} catch (e) {
  console.error("Chrome unavailable:", e.message);
  server.close();
  process.exit(2);
}
const { CDP } = await import("./browser.mjs");
const cdp = new CDP(chrome.ws);

try {
  // ── DESKTOP ───────────────────────────────────────────────────────────────────────────────────────
  let pg = await openPage(cdp, base);
  await sleep(1200); // catalogue render (models.json)

  rec("desktop: console clean on load", pg.errors.length === 0, pg.errors.slice(0, 3).join(" | "));
  rec(
    "desktop: network clean on load",
    pg.netFailures.length === 0,
    pg.netFailures.slice(0, 3).join(" | "),
  );

  const overflow = await evalValue(
    cdp,
    pg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
  );
  rec("desktop: no horizontal overflow", overflow <= 1, `scrollWidth-innerWidth=${overflow}`);

  const sectionInfo = await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
      const heads = [...document.querySelectorAll('.cat-heading')].map(h => h.id);
      const jumps = [...document.querySelectorAll('.jump-list a')].map(a => a.getAttribute('href'));
      const unresolved = jumps.filter(h => !document.getElementById(h.slice(1)));
      const dupes = heads.filter((id,i) => heads.indexOf(id)!==i);
      const allCatPrefixed = heads.every(id => id.startsWith('cat-'));
      return { heads, jumps, unresolved, dupes, allCatPrefixed, jumpCount: jumps.length };
    })()`,
  );
  rec(
    "desktop: section ids all cat-* prefixed",
    sectionInfo.allCatPrefixed,
    sectionInfo.heads.join(","),
  );
  rec(
    "desktop: section ids unique",
    sectionInfo.dupes.length === 0,
    "dupes=" + sectionInfo.dupes.join(","),
  );
  rec(
    "desktop: every jump link resolves",
    sectionInfo.unresolved.length === 0,
    "unresolved=" + sectionInfo.unresolved.join(","),
  );
  rec(
    "desktop: jump nav matches section count",
    sectionInfo.jumpCount === sectionInfo.heads.length,
    `${sectionInfo.jumpCount} links / ${sectionInfo.heads.length} sections`,
  );

  const chrome1 = await evalValue(
    cdp,
    pg.sessionId,
    `!!document.querySelector('#chips .chip') && !!document.querySelector('#start-here .curated-item') && !!document.getElementById('coverage').textContent.match(/built/)`,
  );
  rec("desktop: chips + curated + coverage present", chrome1);

  await screenshot(cdp, pg.sessionId, join(OUT, "home-desktop.png"));

  // Permalink heading is a real anchor to its own id
  const permalinkOk = await evalValue(
    cdp,
    pg.sessionId,
    `(() => { const a = document.querySelector('.cat-heading .permalink'); const h = a?.closest('.cat-heading'); return a && h && a.getAttribute('href') === '#' + h.id; })()`,
  );
  rec("desktop: heading permalink targets its own id", permalinkOk);

  // In-page jump-link CLICK must land a LATE section at its offset (the H1 regression case).
  const lateId = sectionInfo.heads[sectionInfo.heads.length - 2] || sectionInfo.heads[0];
  await evalValue(
    cdp,
    pg.sessionId,
    `(() => { const a = [...document.querySelectorAll('.jump-list a')].find(x => x.getAttribute('href') === '#' + ${
      JSON.stringify(lateId)
    }); a?.click(); return !!a; })()`,
  );
  await sleep(2000);
  const clickLand = await evalValue(
    cdp,
    pg.sessionId,
    `(() => { const el = document.getElementById(${
      JSON.stringify(lateId)
    }); const r = el.getBoundingClientRect(); const want = parseFloat(getComputedStyle(el).scrollMarginTop)||0; const atBottom = Math.ceil(window.scrollY+window.innerHeight) >= document.documentElement.scrollHeight-1; return { top: Math.round(r.top), want: Math.round(want), atBottom, hash: location.hash }; })()`,
  );
  rec(
    `jump-link click lands #${lateId} at offset + updates hash`,
    (Math.abs(clickLand.top - clickLand.want) <= 4 || clickLand.atBottom) &&
      clickLand.hash === "#" + lateId,
    JSON.stringify(clickLand),
  );
  await closePage(cdp, pg.targetId);

  // ── Direct fragment deep-link (fresh navigation to #cat-audio) ─────────────────────────────────────
  const secKeys = sectionInfo.heads;
  for (const id of secKeys) {
    const fpg = await openPage(cdp, base + "#" + id);
    await sleep(2200); // allow the converging scroll to settle past content-visibility estimation
    const info = await evalValue(
      cdp,
      fpg.sessionId,
      `(() => { const el = document.getElementById(${
        JSON.stringify(id)
      }); if(!el) return {exists:false};
        const r = el.getBoundingClientRect();
        const wantTop = parseFloat(getComputedStyle(el).scrollMarginTop) || 0;
        const atBottom = Math.ceil(window.scrollY + window.innerHeight) >= document.documentElement.scrollHeight - 1;
        return { exists:true, top: Math.round(r.top), wantTop: Math.round(wantTop), scrollY: Math.round(window.scrollY), atBottom }; })()`,
    );
    // The heading must land AT its scroll-margin offset (clearing the sticky bar) — not just "scrolled".
    // The final section may be pinned by page bottom; accept atBottom in that case.
    const landed = info.exists && (Math.abs(info.top - info.wantTop) <= 4 || info.atBottom);
    rec(`deep-link #${id} lands heading at sticky offset`, landed, JSON.stringify(info));
    await closePage(cdp, fpg.targetId);
  }

  // ── Search (built-only, ranking, no-results, clear, URL, back/forward) ─────────────────────────────
  pg = await openPage(cdp, base);
  await sleep(1000);

  await typeQuery(cdp, pg.sessionId, "transcribe audio to text");
  let count = await pollCount(cdp, pg.sessionId);
  const searchState = await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
    const results = document.getElementById('results');
    const cat = document.getElementById('catalogue-view');
    const cards = [...document.querySelectorAll('#result-list .model-card')];
    const firstHref = cards[0]?.getAttribute('href') || '';
    const firstText = (cards[0]?.textContent || '').toLowerCase();
    const allModelsHref = cards.every(c => c.getAttribute('href').includes('models/'));
    return { resultsVisible: !results.hidden, catHidden: cat.hidden, nCards: cards.length, firstHref, firstText, allModelsHref, url: location.search };
  })()`,
  );
  rec(
    "search: results surface shown, catalogue hidden",
    searchState.resultsVisible && searchState.catHidden,
  );
  rec(
    "search: has results",
    searchState.nCards > 0,
    `${searchState.nCards} cards · count="${count}"`,
  );
  rec(
    "search: results link to demo routes (built)",
    searchState.allModelsHref && searchState.firstHref.includes("models/"),
    searchState.firstHref,
  );
  const asrTop = /whisper|moonshine|wav2vec|asr|speech/.test(searchState.firstText);
  rec("search: ASR intent ranks a speech demo first", asrTop, searchState.firstText.slice(0, 60));
  rec("search: ?q= reflects query", /q=transcribe/.test(searchState.url), searchState.url);
  const countNum = parseInt(count, 10);
  rec(
    "search: count is exact (all matches shown, not capped)",
    countNum === searchState.nCards,
    `count=${countNum} cards=${searchState.nCards}`,
  );

  // No-results (settle: debounce + worker round-trip, then assert on the new query's render)
  await typeQuery(cdp, pg.sessionId, "zzzxxqqwwv");
  await sleep(4000);
  const noRes = await evalValue(
    cdp,
    pg.sessionId,
    `(() => { const nr = document.querySelector('#result-list .no-results'); const cnt = document.getElementById('search-count').textContent; return { hasNoResults: !!nr, cnt, hasChips: !!nr?.querySelector('.chip'), hasExplore: !!nr?.querySelector('a[href*="explore"]') }; })()`,
  );
  rec(
    "search: honest no-results state",
    noRes.hasNoResults && /0 demos/.test(noRes.cnt),
    JSON.stringify(noRes),
  );
  rec("search: no-results offers suggestions + explore link", noRes.hasChips && noRes.hasExplore);

  // Clear restores catalogue
  await evalValue(cdp, pg.sessionId, `document.getElementById('q-clear').click()`);
  await sleep(400);
  const cleared = await evalValue(
    cdp,
    pg.sessionId,
    `(() => { const results = document.getElementById('results'); const cat = document.getElementById('catalogue-view'); return { resultsHidden: results.hidden, catVisible: !cat.hidden, url: location.search, val: document.getElementById('q').value }; })()`,
  );
  rec(
    "clear: catalogue restored, results hidden, ?q cleared",
    cleared.resultsHidden && cleared.catVisible && cleared.url === "" && cleared.val === "",
  );
  await closePage(cdp, pg.targetId);

  // Direct ?q= load (bookmark) + back/forward
  const qpg = await openPage(cdp, base + "?q=whisper");
  await sleep(500);
  const qcount = await pollCount(cdp, qpg.sessionId);
  const bookmark = await evalValue(
    cdp,
    qpg.sessionId,
    `(() => ({ val: document.getElementById('q').value, resultsVisible: !document.getElementById('results').hidden, n: document.querySelectorAll('#result-list .model-card').length }))()`,
  );
  rec(
    "bookmark ?q=whisper restores search on load",
    bookmark.val === "whisper" && bookmark.resultsVisible && bookmark.n > 0,
    `${bookmark.n} cards · "${qcount}"`,
  );

  // Keyboard: Enter submits (form submit path)
  const kb = await evalValue(
    cdp,
    qpg.sessionId,
    `(() => { const el = document.getElementById('q'); el.focus(); el.value='depth'; el.dispatchEvent(new Event('input',{bubbles:true})); document.getElementById('search-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})); return document.activeElement === el; })()`,
  );
  await pollCount(cdp, qpg.sessionId, 8000);
  const kbState = await evalValue(cdp, qpg.sessionId, `location.search`);
  rec("keyboard: Enter/submit runs search + updates URL", /q=depth/.test(kbState), kbState);
  await closePage(cdp, qpg.targetId);

  // ── MOBILE ─────────────────────────────────────────────────────────────────────────────────────────
  const mpg = await openPage(cdp, base);
  await setViewport(cdp, mpg.sessionId, MOBILE);
  await sleep(1200);
  const mOverflow = await evalValue(
    cdp,
    mpg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
  );
  rec("mobile: no horizontal overflow", mOverflow <= 1, `scrollWidth-innerWidth=${mOverflow}`);
  const tapSizes = await evalValue(
    cdp,
    mpg.sessionId,
    `(() => {
    const els = [document.getElementById('q'), document.getElementById('q-clear'), ...document.querySelectorAll('#chips .chip'), ...document.querySelectorAll('.jump-list a')].filter(Boolean);
    const small = els.filter(e => { const r = e.getBoundingClientRect(); return r.height > 0 && r.height < 24; });
    return { checked: els.length, tooSmall: small.length };
  })()`,
  );
  rec(
    "mobile: interactive controls meet ≥24px min target",
    tapSizes.tooSmall === 0,
    JSON.stringify(tapSizes),
  );
  await typeQuery(cdp, mpg.sessionId, "remove background");
  await pollCount(cdp, mpg.sessionId);
  const mSearch = await evalValue(
    cdp,
    mpg.sessionId,
    `document.querySelectorAll('#result-list .model-card').length`,
  );
  rec("mobile: search returns results", mSearch > 0, `${mSearch} cards`);
  const mOverflow2 = await evalValue(
    cdp,
    mpg.sessionId,
    `document.documentElement.scrollWidth - window.innerWidth`,
  );
  rec("mobile: no overflow with results shown", mOverflow2 <= 1, `${mOverflow2}`);
  await screenshot(cdp, mpg.sessionId, join(OUT, "home-mobile.png"));
  await closePage(cdp, mpg.targetId);

  // ── /explore/ still works ────────────────────────────────────────────────────────────────────────
  const epg = await openPage(cdp, base + "explore/");
  await sleep(1500);
  const exploreOk = await evalValue(
    cdp,
    epg.sessionId,
    `!!document.getElementById('q') && document.querySelectorAll('#filters, #results').length >= 1`,
  );
  rec("/explore/ still loads (route unchanged)", exploreOk, "console errors=" + epg.errors.length);
  await closePage(cdp, epg.targetId);
} finally {
  chrome.kill();
  server.close();
}

const passed = results.filter((r) => r.pass).length;
writeFileSync(
  join(OUT, "home-validation.json"),
  JSON.stringify({ passed, total: results.length, results }, null, 2),
);
console.log(`\n${passed}/${results.length} checks passed. Report: reports/home-validation.json`);
process.exit(passed === results.length ? 0 : 1);
