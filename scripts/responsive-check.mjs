#!/usr/bin/env node
// Mobile + desktop parity matrix harness (parity invariant).
//
// For each built model page, at BOTH classes — mobile (≈360×740, DPR3, touch) and desktop
// (≈1280×800) — it loads the page, screenshots it, and asserts PROGRAMMATICALLY:
//   • no horizontal overflow  (documentElement.scrollWidth <= innerWidth + 1)
//   • no interactive control clipped off the right edge of the viewport
//   • console clean (no errors/exceptions during load)
//   • network clean (no failed requests)
// It NEVER triggers a large model download (fresh profile ⇒ cache-absent ⇒ honest Download state).
//
// AUTOMATED-ONLY signal — this seeds the `support` record as "needs-review" (never "ok"): only a real
// human/agent matrix pass, reading the screenshots for legibility/tap-targets/focus/dialogs, may flip
// a class to "ok" (or "unsupported" WITH evidence). Coverage stays honest.
//
// Usage:
//   node scripts/responsive-check.mjs                 # scan every built page
//   node scripts/responsive-check.mjs --slug bert-ner # one page
//   node scripts/responsive-check.mjs --limit 10      # first N
//   node scripts/responsive-check.mjs --seed-support  # write "needs-review"/lastChecked into models.json
// Emits reports/responsive/results.json + reports/responsive/index.html.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  BASE,
  CDP,
  closePage,
  DESKTOP,
  escapeHtml,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";
import { builtModels, loadCatalogue } from "./conformance-lib.mjs";

const args = process.argv.slice(2);
const opt = (n) => args.includes(n);
const val = (n, d) => {
  const i = args.indexOf(n);
  return i !== -1 ? args[i + 1] : d;
};
const NOW = process.env.CONFORMANCE_GENERATED_AT || val("--now", "2026-07-19T00:00:00Z");
const outDir = join(repoRoot, "reports", "responsive");

// Per-class programmatic checks. Returns { pass, overflow, offscreen, console, network, notes }.
async function checkClass(cdp, sessionId, vp, errors, netFailures) {
  await setViewport(cdp, sessionId, vp);
  const metrics = await evalValue(
    cdp,
    sessionId,
    `(()=>{const de=document.documentElement;const overflow=de.scrollWidth-window.innerWidth;
      const controls=[...document.querySelectorAll('button,a,input,select,textarea,[role=button],[tabindex]')];
      let clipped=0, small=0;
      for(const c of controls){const r=c.getBoundingClientRect();
        if(r.width===0&&r.height===0)continue;
        if(r.right>window.innerWidth+1||r.left<-1)clipped++;
        if((r.width>0&&r.width<24)||(r.height>0&&r.height<24))small++;}
      return {overflow, clipped, small, controls:controls.length};})()`,
  );
  const overflowOk = (metrics?.overflow ?? 0) <= 1;
  const offscreenOk = (metrics?.clipped ?? 0) === 0;
  const consoleOk = errors.length === 0;
  const networkOk = netFailures.length === 0;
  const notes = [];
  if (!overflowOk) notes.push(`horizontal overflow ${metrics.overflow}px`);
  if (!offscreenOk) notes.push(`${metrics.clipped} control(s) clipped off-viewport`);
  if (!consoleOk) notes.push(`console: ${errors.slice(0, 2).join(" | ")}`);
  if (!networkOk) notes.push(`network fail: ${netFailures.slice(0, 2).join(" | ")}`);
  if (metrics?.small) notes.push(`${metrics.small} sub-24px target(s) — agent to verify tap size`);
  return {
    pass: overflowOk && offscreenOk && consoleOk && networkOk,
    overflow: metrics?.overflow ?? null,
    clipped: metrics?.clipped ?? null,
    console: consoleOk,
    network: networkOk,
    notes,
  };
}

async function scanPage(cdp, port, slug) {
  const url = `http://127.0.0.1:${port}${BASE}models/${slug}/`;
  const screenDir = join(outDir, "screens", slug);
  mkdirSync(screenDir, { recursive: true });
  const ctx = await openPage(cdp, url);
  try {
    // Desktop first (also the openPage default), then mobile.
    const desktop = await checkClass(cdp, ctx.sessionId, DESKTOP, ctx.errors, ctx.netFailures);
    try {
      await screenshot(cdp, ctx.sessionId, join(screenDir, "desktop.png"));
    } catch { /* best-effort */ }
    const mobile = await checkClass(cdp, ctx.sessionId, MOBILE, ctx.errors, ctx.netFailures);
    try {
      await screenshot(cdp, ctx.sessionId, join(screenDir, "mobile.png"));
    } catch { /* best-effort */ }
    return { slug, desktop, mobile };
  } finally {
    await closePage(cdp, ctx.targetId);
  }
}

function renderRollup(runs) {
  const rows = runs.map((r) => {
    const cell = (c) =>
      `<td class="${c.pass ? "ok" : "warn"}">${c.pass ? "clean" : "review"}${
        c.notes.length ? "<br><small>" + escapeHtml(c.notes.join("; ")) + "</small>" : ""
      }</td>`;
    return `<tr><td><b>${r.slug}</b></td>${cell(r.desktop)}${cell(r.mobile)}
      <td><a href="screens/${r.slug}/desktop.png">D</a> · <a href="screens/${r.slug}/mobile.png">M</a></td></tr>`;
  }).join("\n");
  const dOk = runs.filter((r) => r.desktop.pass).length;
  const mOk = runs.filter((r) => r.mobile.pass).length;
  return `<!doctype html><meta charset="utf-8"><title>Responsive matrix · web-ai-showcase</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-family:Georgia,serif}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}
.ok{color:#0a7d33}.warn{color:#8a6d00}.summary{background:#f5f3ee;padding:1rem;border-radius:8px}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}td,th{border-color:#333}.summary{background:#1c1c1c}}</style>
<h1>Responsive matrix — web-ai-showcase</h1>
<p class="summary">${runs.length} pages scanned · desktop automated-clean <b>${dOk}</b> · mobile automated-clean <b>${mOk}</b>.
Automated signal only ⇒ <b>needs-review</b>, never <b>ok</b>. Generated ${NOW}.</p>
<table><thead><tr><th>demo</th><th>desktop (1280×800)</th><th>mobile (360×740 DPR3)</th><th>shots</th></tr></thead>
<tbody>${rows}</tbody></table>`;
}

function seedSupport(runs) {
  const catPath = join(repoRoot, "models.json");
  const raw = JSON.parse(readFileSync(catPath, "utf8"));
  const arr = Array.isArray(raw) ? raw : raw.models;
  const byId = new Map(arr.map((m) => [m.slug, m]));
  let touched = 0;
  for (const r of runs) {
    const m = byId.get(r.slug);
    if (!m) continue;
    m.support = m.support || { desktop: "untested", mobile: "untested" };
    // Automated signal ⇒ needs-review, and only if the class is not already an agent-set verdict.
    for (const cls of ["desktop", "mobile"]) {
      const cur = m.support[cls];
      if (cur === "untested" || cur === "needs-review") {
        m.support[cls] = "needs-review";
      }
    }
    const notes = [
      ...r.desktop.notes.map((n) => "desktop: " + n),
      ...r.mobile.notes.map((n) => "mobile: " + n),
    ];
    m.support.lastChecked = NOW;
    if (notes.length) m.support.automatedNotes = notes;
    else delete m.support.automatedNotes;
    touched++;
  }
  writeFileSync(catPath, JSON.stringify(raw, null, 2) + "\n");
  console.log(
    `support seeded to "needs-review" on ${touched} built entries (automated-only signal).`,
  );
}

async function main() {
  mkdirSync(outDir, { recursive: true });
  let slugs = builtModels(loadCatalogue()).map((m) => m.slug);
  const s = val("--slug");
  if (s) slugs = [s];
  const lim = val("--limit");
  if (lim) slugs = slugs.slice(0, Number(lim));

  const { server, port } = await startServer();
  const chrome = await launchChrome();
  const cdp = new CDP(chrome.ws);
  const runs = [];
  try {
    for (const slug of slugs) {
      process.stderr.write(`  scanning ${slug} … `);
      let r;
      try {
        r = await scanPage(cdp, port, slug);
      } catch (e) {
        const note = ["scan error: " + (e.message || e)];
        r = {
          slug,
          desktop: {
            pass: false,
            notes: note,
            overflow: null,
            clipped: null,
            console: false,
            network: false,
          },
          mobile: {
            pass: false,
            notes: note,
            overflow: null,
            clipped: null,
            console: false,
            network: false,
          },
        };
      }
      runs.push(r);
      process.stderr.write(
        `desktop ${r.desktop.pass ? "clean" : "review"} · mobile ${
          r.mobile.pass ? "clean" : "review"
        }\n`,
      );
    }
  } finally {
    chrome.kill();
    server.close();
  }

  writeFileSync(
    join(outDir, "results.json"),
    JSON.stringify({ generatedAt: NOW, pages: runs.length, runs }, null, 2) + "\n",
  );
  writeFileSync(join(outDir, "index.html"), renderRollup(runs));
  if (opt("--seed-support")) seedSupport(runs);

  const dOk = runs.filter((r) => r.desktop.pass).length;
  const mOk = runs.filter((r) => r.mobile.pass).length;
  console.log("\n=== RESPONSIVE MATRIX (automated pre-scan) ===");
  console.log(`pages: ${runs.length}`);
  console.log(`desktop automated-clean: ${dOk}/${runs.length}`);
  console.log(`mobile  automated-clean: ${mOk}/${runs.length}`);
  console.log(`(automated signal ⇒ needs-review; only an agent matrix pass flips a class to ok)`);
  console.log(`wrote reports/responsive/results.json + reports/responsive/index.html`);
}

main().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
