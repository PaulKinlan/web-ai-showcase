#!/usr/bin/env node
// Deterministic, headless-Chrome-backed conformance runner for the web-ai-showcase lifecycle.
//
// Runs ONE suite or run-ALL. For each built model's immutable models/<slug>/conformance.json it drives
// the page in headless Chrome (no GPU, fresh profile) and checks each assertion, emitting exact counts:
//   tested / total / pass / fail / blocked   (+ manual = manual-evidenced, screenshot captured, needs
//                                              an agent verdict; NOT an auto pass)
// blocked = the device/feature is GENUINELY unavailable (e.g. a WebGPU-only model in a no-GPU runner
// that shows its honest needs-WebGPU fallback) — explicit, never counted as a pass.
//
// Download-free + deterministic: a FRESH browser profile means every model is cache-absent, so the
// shared auto-init loader shows a Download button and NEVER auto-downloads a large model (per the
// mandate). Same tree ⇒ same result. Timestamps come from --now / CONFORMANCE_GENERATED_AT.
//
// Usage:
//   node scripts/conformance.mjs --slug bert-ner        # run one suite
//   node scripts/conformance.mjs --all                  # run every suite
//   node scripts/conformance.mjs --all --limit 10       # first N (audit sampling)
//   node scripts/conformance.mjs --all --no-screenshots # skip manual-evidence screenshots (faster)
// Emits reports/conformance/results.json + reports/conformance/index.html (rollup) + a CLI table.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  BASE,
  CDP,
  closePage,
  DESKTOP,
  escapeHtml,
  evalBool,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";
import { builtModels, modelSource } from "./conformance-lib.mjs";

const args = process.argv.slice(2);
const opt = (name) => args.includes(name);
const val = (name, d) => {
  const i = args.indexOf(name);
  return i !== -1 ? args[i + 1] : d;
};
const NOW = process.env.CONFORMANCE_GENERATED_AT || val("--now", "2026-07-19T00:00:00Z");
const wantScreens = !opt("--no-screenshots");
const outDir = join(repoRoot, "reports", "conformance");

function sourceCheck(slug, test) {
  const { all } = modelSource(slug);
  const re = new RegExp(test.pattern, "i");
  const hit = re.test(all);
  return test.mode === "absent" ? !hit : hit;
}

// Run one assertion → { state: pass|fail|blocked|manual, evidence }
async function runAssertion(cdp, ctx, slug, a) {
  const { sessionId, errors } = ctx;
  switch (a.kind) {
    case "console-clean":
      return errors.length === 0
        ? { state: "pass", evidence: "no console errors / exceptions during load" }
        : { state: "fail", evidence: "console errors: " + errors.slice(0, 3).join(" | ") };
    case "source": {
      const ok = sourceCheck(slug, a.test);
      return ok
        ? { state: "pass", evidence: `source ${a.test.mode} /${a.test.pattern}/` }
        : { state: "fail", evidence: `source did not satisfy ${a.test.mode} /${a.test.pattern}/` };
    }
    case "page-text": {
      const ok = await evalBool(
        cdp,
        sessionId,
        `(document.documentElement.innerText+' '+document.documentElement.innerHTML).includes(${
          JSON.stringify(a.test)
        })`,
      );
      return ok
        ? { state: "pass", evidence: `page mentions ${JSON.stringify(a.test)}` }
        : { state: "fail", evidence: `page does not mention ${JSON.stringify(a.test)}` };
    }
    case "dom": {
      const sel = JSON.stringify(a.test.selector);
      const ok = await evalBool(
        cdp,
        sessionId,
        `document.querySelectorAll(${sel}).length >= ${a.test.min ?? 1}`,
      );
      return ok
        ? { state: "pass", evidence: `>=${a.test.min ?? 1} match for ${a.test.selector}` }
        : { state: "fail", evidence: `no match for ${a.test.selector}` };
    }
    case "script": {
      const ok = await evalBool(cdp, sessionId, a.test);
      return ok
        ? { state: "pass", evidence: "expression true in page" }
        : { state: "fail", evidence: "expression false in page" };
    }
    case "capability": {
      if (a.test.probe === "webgpu") {
        const has = await evalBool(
          cdp,
          sessionId,
          "('gpu' in navigator) && ((await navigator.gpu.requestAdapter())!=null)",
        );
        if (has) return { state: "pass", evidence: "WebGPU adapter available; model can run" };
        // No adapter. Read the loader's honest state text.
        const status = await evalBool(
          cdp,
          sessionId,
          "((document.querySelector('.model-loader')?.innerText)||'').trim().length > 0",
        );
        const gpuOnly = await evalBool(
          cdp,
          sessionId,
          "/unsupported|can't run|cannot run|needs.*(gpu|webgpu)|requires.*(gpu|webgpu)|no gpu adapter/i.test((document.querySelector('.model-loader')?.innerText)||document.documentElement.innerText)",
        );
        if (gpuOnly) {
          // Genuinely GPU-only on this no-GPU runner — honest, explicit device-unavailable.
          return {
            state: "blocked",
            evidence: "no GPU adapter; page shows an honest needs-WebGPU/unsupported state",
          };
        }
        if (status) {
          // Honest WASM-fallback/download state on a no-GPU device — labelled, not blank, not faked.
          return {
            state: "pass",
            evidence:
              "no GPU adapter; loader shows an honest labelled WASM-fallback/download state (not blank, not faked)",
          };
        }
        return {
          state: "fail",
          evidence: "no GPU adapter and loader shows a blank/unlabelled state",
        };
      }
      return { state: "manual", evidence: "unknown capability probe" };
    }
    case "responsive": {
      const vp = a.deviceClass === "mobile" ? MOBILE : DESKTOP;
      await setViewport(cdp, sessionId, vp);
      const ok = await evalBool(
        cdp,
        sessionId,
        "document.documentElement.scrollWidth <= window.innerWidth + 1",
      );
      await setViewport(cdp, sessionId, DESKTOP); // restore
      return ok
        ? { state: "pass", evidence: `${vp.width}x${vp.height}: no horizontal overflow` }
        : {
          state: "fail",
          evidence: `${vp.width}x${vp.height}: horizontal overflow (scrollWidth>innerWidth)`,
        };
    }
    case "manual-evidenced": {
      const ev = ctx.shots?.length
        ? `screenshot evidence: ${
          ctx.shots.map((s) => s.replace(repoRoot, "")).join(", ")
        } — agent Reads + records verdict`
        : "needs agent verdict";
      return { state: "manual", evidence: ev };
    }
    default:
      return { state: "manual", evidence: "unhandled kind " + a.kind };
  }
}

async function runSuite(cdp, port, slug) {
  const suite = JSON.parse(
    readFileSync(join(repoRoot, "models", slug, "conformance.json"), "utf8"),
  );
  const url = `http://127.0.0.1:${port}${BASE}models/${slug}/`;
  const screenDir = join(outDir, "screens", slug);
  const ctx = await openPage(cdp, url);
  ctx.shots = [];
  if (wantScreens) {
    mkdirSync(screenDir, { recursive: true });
    try {
      const d = join(screenDir, "desktop.png");
      await screenshot(cdp, ctx.sessionId, d);
      ctx.shots.push(d);
      await setViewport(cdp, ctx.sessionId, MOBILE);
      const m = join(screenDir, "mobile.png");
      await screenshot(cdp, ctx.sessionId, m);
      ctx.shots.push(m);
      await setViewport(cdp, ctx.sessionId, DESKTOP);
    } catch { /* screenshots best-effort */ }
  }
  const results = [];
  try {
    for (const a of suite.assertions) {
      let r;
      try {
        r = await runAssertion(cdp, ctx, slug, a);
      } catch (e) {
        r = { state: "fail", evidence: "runner error: " + (e.message || e) };
      }
      results.push({
        id: a.id,
        category: a.category,
        deviceClass: a.deviceClass,
        kind: a.kind,
        ...r,
      });
    }
  } finally {
    await closePage(cdp, ctx.targetId);
  }
  return { slug, total: suite.assertions.length, ...tallyResults(results), results };
}

function tallyResults(results) {
  const t = { pass: 0, fail: 0, blocked: 0, manual: 0 };
  for (const r of results) t[r.state]++;
  return {
    tested: t.pass + t.fail + t.blocked,
    pass: t.pass,
    fail: t.fail,
    blocked: t.blocked,
    manual: t.manual,
  };
}

function aggregate(runs) {
  return runs.reduce((a, r) => {
    a.total += r.total;
    a.tested += r.tested;
    a.pass += r.pass;
    a.fail += r.fail;
    a.blocked += r.blocked;
    a.manual += r.manual;
    return a;
  }, { total: 0, tested: 0, pass: 0, fail: 0, blocked: 0, manual: 0 });
}

function renderRollup(runs) {
  const agg = aggregate(runs);
  const rows = runs.map((r) => {
    const detail = r.results.map((x) =>
      `<tr class="a ${x.state}"><td>${x.id}</td><td>${x.category}</td><td>${x.deviceClass}</td><td>${x.state}</td><td>${
        escapeHtml(x.evidence || "")
      }</td></tr>`
    ).join("");
    return `<details><summary><b>${r.slug}</b> — ${r.pass}✓ ${r.fail}✗ ${r.blocked}▨ ${r.manual}◍ / ${r.total}</summary>
      <table class="det"><thead><tr><th>assertion</th><th>category</th><th>class</th><th>state</th><th>evidence</th></tr></thead><tbody>${detail}</tbody></table></details>`;
  }).join("\n");
  return `<!doctype html><meta charset="utf-8"><title>Conformance rollup · web-ai-showcase</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-family:Georgia,serif}table{border-collapse:collapse;width:100%}.det td,.det th{border:1px solid #ddd;padding:.25rem .5rem;font-size:13px;text-align:left}
.summary{background:#f5f3ee;padding:1rem;border-radius:8px}.pass td:nth-child(4){color:#0a7d33;font-weight:600}.fail td:nth-child(4){color:#c0392b;font-weight:700}
.blocked td:nth-child(4){color:#8a6d00}.manual td:nth-child(4){color:#555}details{margin:.4rem 0}summary{cursor:pointer;padding:.3rem 0}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}.summary{background:#1c1c1c}.det td,.det th{border-color:#333}}</style>
<h1>Conformance rollup — web-ai-showcase</h1>
<p class="summary">Suites: <b>${runs.length}</b> · assertions <b>${agg.total}</b> · tested <b>${agg.tested}</b>
(pass <b>${agg.pass}</b> · fail <b>${agg.fail}</b> · blocked <b>${agg.blocked}</b>) · manual-evidenced awaiting verdict <b>${agg.manual}</b>.
<br>Generated ${NOW} · headless Chrome (no GPU, fresh profile, download-free). blocked = genuinely device/feature-unavailable, honest, never a pass.</p>
${rows}`;
}

async function main() {
  mkdirSync(outDir, { recursive: true });
  let slugs;
  if (opt("--all")) {
    slugs = builtModels().map((m) => m.slug);
    const lim = val("--limit");
    if (lim) slugs = slugs.slice(0, Number(lim));
  } else {
    const s = val("--slug");
    if (!s) {
      console.error("Usage: node scripts/conformance.mjs --slug <slug> | --all [--limit N]");
      process.exit(2);
    }
    slugs = [s];
  }

  const { server, port } = await startServer();
  const chrome = await launchChrome();
  const cdp = new CDP(chrome.ws);
  const runs = [];
  try {
    for (const slug of slugs) {
      process.stderr.write(`  running ${slug} … `);
      let r;
      try {
        r = await runSuite(cdp, port, slug);
      } catch (e) {
        // A hung/failed page must not stall the whole matrix — record it and move on.
        const suite = JSON.parse(
          readFileSync(join(repoRoot, "models", slug, "conformance.json"), "utf8"),
        );
        const results = suite.assertions.map((a) => ({
          id: a.id,
          category: a.category,
          deviceClass: a.deviceClass,
          kind: a.kind,
          state: "fail",
          evidence: "page-level runner error: " + (e.message || e),
        }));
        r = { slug, total: suite.assertions.length, ...tallyResults(results), results };
      }
      runs.push(r);
      process.stderr.write(`${r.pass}✓ ${r.fail}✗ ${r.blocked}▨ ${r.manual}◍/${r.total}\n`);
    }
  } finally {
    chrome.kill();
    server.close();
  }

  const agg = aggregate(runs);
  const resultsPath = join(outDir, "results.json");
  writeFileSync(
    resultsPath,
    JSON.stringify({ generatedAt: NOW, suites: runs.length, aggregate: agg, runs }, null, 2) + "\n",
  );
  writeFileSync(join(outDir, "index.html"), renderRollup(runs));

  console.log("\n=== CONFORMANCE RESULTS ===");
  console.log(`suites: ${runs.length}`);
  for (const r of runs) {
    console.log(
      `  ${r.slug.padEnd(28)} ${String(r.pass).padStart(2)}✓ ${String(r.fail).padStart(2)}✗ ` +
        `${String(r.blocked).padStart(2)}▨ ${String(r.manual).padStart(2)}◍ / ${r.total}`,
    );
  }
  console.log(
    `\nassertions ${agg.total} · tested ${agg.tested} (pass ${agg.pass} · fail ${agg.fail} · ` +
      `blocked ${agg.blocked}) · manual-evidenced (needs agent verdict) ${agg.manual}`,
  );
  console.log(`wrote ${resultsPath.replace(repoRoot, "")} + reports/conformance/index.html`);
  if (agg.fail > 0) {
    console.log(`\nNOTE: ${agg.fail} failing assertion(s) — fix the DEMO, never weaken the suite.`);
  }
}

main().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
