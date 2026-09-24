#!/usr/bin/env node
// Class-level 360px overflow sweep (web-ai-showcase-a2u).
//
// vtk fixed ONE page and hardened ONE family's validator. The sweep that found the
// class found 20 more failing pages (6 expanding the viewport exactly like
// CodeGen, 14 with a control poking out of its panel), and a per-family assertion
// cannot see pages whose family nobody is touching. This runs the same assertion
// across every built route so the class cannot regress silently.
//
// The assertion is lifted verbatim from scripts/validate-codegen-350m.mjs (the
// vtk fix), including its symmetric bounds: a page can satisfy
// "scrollWidth - innerWidth == 0" simply by WIDENING the viewport, so the
// requested width is asserted FIRST, then the scroll delta, then that no control
// escapes the viewport or the content box of its enclosing .panel.
//
// Model hosts are blocked and downloads never start: this is a STRUCTURAL sweep,
// which is what makes ~170 pages take minutes rather than hours. It is a layout
// gate, not an inference test.
//
//   node scripts/sweep-360-overflow.mjs                # every built route
//   node scripts/sweep-360-overflow.mjs --slugs a,b,c   # specific routes
//   node scripts/sweep-360-overflow.mjs --limit 20      # first N
//   node scripts/sweep-360-overflow.mjs --json out.json # write the evidence ledger
//   node scripts/sweep-360-overflow.mjs --report-only   # never exit 1 (advisory run)
//
// Exit 1 when any checked page expands the viewport or lets a control escape.
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  BASE,
  CDP,
  closePage,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
  evalValue,
} from "./browser.mjs";
import { builtModels, loadCatalogue } from "./conformance-lib.mjs";

const args = process.argv.slice(2);
const val = (name, dflt) => {
  const i = args.indexOf(name);
  return i !== -1 ? args[i + 1] : dflt;
};
const opt = (name) => args.includes(name);

/** Hosts that would start a model download; blocked so the sweep stays structural. */
const MODEL_HOSTS = [
  "*huggingface.co/*",
  "*cdn-lfs.huggingface.co/*",
  "*cdn-lfs-us-1.hf.co/*",
  "*hf.co/*",
  "*cdn.jsdelivr.net/*",
  "*cdnjs.cloudflare.com/*",
  "*storage.googleapis.com/*",
  "*mediapipe*",
  "*mlc-ai*",
];

const slugs = val("--slugs", "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
const limit = Number(val("--limit", "0")) || 0;
const jsonPath = val("--json", "");
const reportOnly = opt("--report-only");
const overviewOnly = opt("--overview-only");

const catalogue = loadCatalogue();
const built = builtModels(catalogue).map((m) => m.slug).sort();
const targets = slugs.length ? slugs.filter((s) => built.includes(s)) : built;
const selected = limit ? targets.slice(0, limit) : targets;

/** Every page a family publishes: the overview plus each ladder rung. */
function familyPages(slug) {
  const dir = join(repoRoot, "models", slug);
  const pages = [];
  const walk = (d) => {
    for (const ent of readdirSync(d, { withFileTypes: true })) {
      const full = join(d, ent.name);
      if (ent.isDirectory()) walk(full);
      else if (ent.name === "index.html") pages.push(full);
    }
  };
  if (existsSync(dir)) walk(dir);
  return pages
    .map((f) => f.slice(join(repoRoot, "models", slug).length + 1).replace(/index\.html$/, ""))
    .sort((a, b) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)));
}
const missing = slugs.filter((s) => !built.includes(s));
if (missing.length) {
  console.error(`unknown built slug(s): ${missing.join(", ")}`);
  process.exit(2);
}

/** The lifted assertion. Kept as one string so it stays identical to the validator's. */
const assertionFor = (requested) => `(() => {
  const panelContentEdges = (el) => {
    const panel = el.closest('.panel');
    if (!panel || panel === el) return null;
    const pr = panel.getBoundingClientRect();
    const cs = getComputedStyle(panel);
    return {
      left: pr.left + parseFloat(cs.paddingLeft || 0) + parseFloat(cs.borderLeftWidth || 0),
      right: pr.right - parseFloat(cs.paddingRight || 0) - parseFloat(cs.borderRightWidth || 0),
    };
  };
  const vw = window.innerWidth;
  const controls = [...document.querySelectorAll('button, input, select, textarea, label, output')]
    .filter((el) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      return r.width > 0 && r.height > 0 && cs.display !== 'none' && cs.visibility !== 'hidden';
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
        tag: el.tagName.toLowerCase(), id: el.id || null,
        cls: typeof el.className === 'string' ? el.className.split(/\\s+/).slice(0, 2).join('.') : null,
        overViewportRight, overViewportLeft, overPanelRight, overPanelLeft,
      });
    }
  }
  return {
    requested: ${requested}, innerWidth: vw, expanded: vw !== ${requested},
    overflow: document.documentElement.scrollWidth - vw,
    controlsChecked: controls.length,
    escaping: escaping.slice(0, 6),
    escapingCount: escaping.length,
  };
})()`;

const diagnose = opt("--diagnose");

/** Names the elements whose intrinsic width exceeds the requested viewport. */
const DIAGNOSE_JS = `(() => {
  const vw = window.innerWidth;
  const offenders = [];
  for (const el of document.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const parent = el.parentElement;
    let outsideParent = false;
    if (parent) {
      const pr = parent.getBoundingClientRect();
      if (pr.width > 0 && r.right - pr.right > 1) outsideParent = true;
    }
    // Content that overflows its own box also forces min-content width, and it is
    // invisible to a bounds-only check (grapheme-to-phoneme expands to 398px with
    // no element wider than the viewport).
    const contentOverflow = el.scrollWidth - el.clientWidth;
    const text = (el.childElementCount === 0 ? (el.textContent || "").trim() : "");
    if (r.width > vw || r.right - vw > 1 || outsideParent || contentOverflow > 1) {
      offenders.push({
        tag: el.tagName.toLowerCase(), id: el.id || null,
        cls: typeof el.className === 'string' ? el.className.split(/\\s+/).slice(0, 3).join('.') : null,
        width: +r.width.toFixed(1), right: +r.right.toFixed(1),
        minWidth: cs.minWidth, maxWidth: cs.maxWidth,
        display: cs.display, flex: cs.display.includes('flex') ? cs.flex : null,
        outsideParent, parentTag: parent ? parent.tagName.toLowerCase() : null,
        contentOverflow: contentOverflow > 1 ? contentOverflow : null,
        whiteSpace: contentOverflow > 1 ? cs.whiteSpace : null,
        text: contentOverflow > 1 ? text.slice(0, 60) : null,
      });
    }
  }
  return { vw, count: offenders.length, offenders: offenders.slice(0, 8) };
})()`;

const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const results = [];

try {
  for (const slug of selected) {
    const rungs = overviewOnly ? [""] : familyPages(slug);
    for (const rung of rungs) {
      const url = `http://127.0.0.1:${port}${BASE}models/${slug}/${rung}`;
    // Blocked before navigation, so no download starts at all.
    const ctx = await openPage(cdp, url, { blockUrls: MODEL_HOSTS });
    const label = rung ? `${slug}/${rung.replace(/\/$/, "")}` : slug;
    try {
      await setViewport(cdp, ctx.sessionId, MOBILE);
      if (diagnose) {
        const d = await evalValue(cdp, ctx.sessionId, DIAGNOSE_JS);
        console.log(`\n=== ${label} (innerWidth ${d.vw}) — ${d.count} offending element(s) ===`);
        for (const o of d.offenders) {
          console.log(
            `  ${o.tag}${o.id ? "#" + o.id : ""}${o.cls ? "." + o.cls : ""} ` +
              `w=${o.width} right=${o.right} minW=${o.minWidth} maxW=${o.maxWidth} ` +
              `display=${o.display}${o.flex ? ` flex=${o.flex}` : ""}` +
              `${o.outsideParent ? ` ESCAPES ${o.parentTag}` : ""}` +
              `${o.contentOverflow ? ` CONTENT+${o.contentOverflow} ws=${o.whiteSpace} "${o.text}"` : ""}`,
          );
        }
        results.push({ slug, rung: rung || null, status: "diagnose", failures: [], metrics: d });
        continue;
      }
      const metrics = await evalValue(cdp, ctx.sessionId, assertionFor(MOBILE.width));
      const failures = [];
      if (metrics?.expanded) failures.push(`viewport expanded to ${metrics.innerWidth}px at a requested ${MOBILE.width}px`);
      if ((metrics?.overflow ?? 0) > 1) failures.push(`horizontal overflow ${metrics.overflow}px`);
      if ((metrics?.escapingCount ?? 0) > 0) {
        const worst = (metrics.escaping ?? [])[0];
        const detail = worst
          ? ` (worst: ${worst.tag}${worst.id ? "#" + worst.id : ""} ` +
            `${[worst.overViewportRight && `right+${worst.overViewportRight}`, worst.overPanelRight && `panel+${worst.overPanelRight}`].filter(Boolean).join(", ")})`
          : "";
        failures.push(`${metrics.escapingCount} control(s) escape${detail}`);
      }
      const consoleErrors = ctx.errors ?? [];
      if (consoleErrors.length) failures.push(`console: ${consoleErrors.slice(0, 1).join(" | ")}`);

      results.push({
        slug,
        rung: rung || null,
        status: failures.length ? "fail" : "clean",
        failures,
        metrics,
      });
      const mark = failures.length ? "FAIL" : "clean";
      console.log(`  ${mark.padEnd(5)} ${label}${failures.length ? ` — ${failures.join("; ")}` : ""}`);
    } catch (error) {
      results.push({ slug, rung: rung || null, status: "error", failures: [String(error)] });
      console.log(`  ERROR ${label} — ${String(error).slice(0, 140)}`);
    } finally {
      await closePage(cdp, ctx.targetId).catch(() => {});
    }
    }
  }
} finally {
  chrome.kill();
  server.close();
}

const failing = results.filter((r) => r.status === "fail");
const errored = results.filter((r) => r.status === "error");
const expanded = results.filter((r) => r.metrics?.expanded);
const escaping = results.filter((r) => (r.metrics?.escapingCount ?? 0) > 0);

console.log(
  `\nsweep-360: ${results.length} route(s) · ${results.length - failing.length - errored.length} clean · ` +
    `${expanded.length} viewport-expansion · ${escaping.length} escaping-control · ${errored.length} error(s)`,
);
if (failing.length) {
  console.log("\nSub-class A (viewport expands; the old scroll-delta assertion reads 0):");
  for (const r of expanded) console.log(`  ${r.rung ? r.slug + "/" + r.rung.replace(/\/$/, "") : r.slug}: innerWidth ${r.metrics.innerWidth} (requested ${MOBILE.width})`);
  console.log("\nSub-class B (control escapes its panel content box):");
  for (const r of escaping) {
    console.log(`  ${r.rung ? r.slug + "/" + r.rung.replace(/\/$/, "") : r.slug}: ${r.metrics.escapingCount} control(s) — ${JSON.stringify(r.metrics.escaping[0])}`);
  }
}

if (jsonPath) {
  const out = join(repoRoot, jsonPath);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(
    out,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        requestedWidth: MOBILE.width,
        modelHostsBlocked: MODEL_HOSTS,
        totals: {
          checked: results.length,
          clean: results.length - failing.length - errored.length,
          viewportExpansion: expanded.length,
          escapingControls: escaping.length,
          errors: errored.length,
        },
        results,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`\nwrote ${jsonPath}`);
}

process.exit(reportOnly || (failing.length === 0 && errored.length === 0) ? 0 : 1);
