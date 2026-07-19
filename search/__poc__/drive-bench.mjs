#!/usr/bin/env node
// Headless-Chrome driver for the search POC. Loads search/__poc__/bench.html at the repo's standard
// desktop (1280×800) and mobile (360×740 DPR3) viewports, waits for window.__RESULTS__, and prints a
// consolidated JSON report. Reuses the repo's own harness (scripts/browser.mjs) so the base path,
// fresh profile, and CDP plumbing match the conformance runner exactly.
//
// Usage: node search/__poc__/drive-bench.mjs   (writes search/__poc__/results.json)

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  BASE,
  CDP,
  DESKTOP,
  launchChrome,
  MOBILE,
  setViewport,
  startServer,
} from "../../scripts/browser.mjs";

const URL_PATH = `${BASE}search/__poc__/bench.html`;

async function runViewport(cdp, port, vp, label) {
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  await cdp.send("Page.enable", {}, sessionId);
  await cdp.send("Runtime.enable", {}, sessionId);
  await setViewport(cdp, sessionId, vp);
  await cdp.send("Page.navigate", { url: `http://localhost:${port}${URL_PATH}` }, sessionId);

  // Poll for window.__RESULTS__ (build + benches + probes can take a few s incl. sqlite network).
  const deadline = Date.now() + 60000;
  let results = null;
  while (Date.now() < deadline) {
    const { result } = await cdp.send(
      "Runtime.evaluate",
      {
        expression: "window.__RESULTS__ ? JSON.stringify(window.__RESULTS__) : ''",
        returnByValue: true,
      },
      sessionId,
    );
    if (result.value) {
      results = JSON.parse(result.value);
      break;
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  await cdp.send("Target.closeTarget", { targetId }).catch(() => {});
  if (!results) throw new Error(`${label}: timed out waiting for __RESULTS__`);
  return results;
}

async function main() {
  const { server, port } = await startServer();
  const chrome = await launchChrome();
  const cdp = new CDP(chrome.ws);

  const report = { generatedAt: new Date().toISOString(), viewports: {} };
  for (const [label, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
    process.stderr.write(`running ${label}…\n`);
    report.viewports[label] = await runViewport(cdp, port, vp, label);
  }

  chrome.kill();
  server.close();

  writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(report, null, 2));
  // Compact summary to stdout.
  for (const [label, r] of Object.entries(report.viewports)) {
    if (r.error) {
      console.log(`\n=== ${label} === ERROR: ${r.error}`);
      continue;
    }
    console.log(`\n=== ${label} (${r.vw}×${r.vh} dpr${r.dpr}) ===`);
    console.log(`  docs=${r.build.docCount} terms=${r.build.terms} avgDocLen=${r.build.avgDocLen}`);
    console.log(
      `  build: fetch=${r.build.fetchedMs}ms index=${r.build.buildIndexMs}ms vec=${r.build.buildVecMs}ms total=${r.build.totalBuildMs}ms (roundtrip ${r.buildRoundTripMs}ms)`,
    );
    console.log(
      `  index=${(r.build.indexBytes / 1048576).toFixed(2)}MB vecF32=${
        (r.build.vecF32Bytes / 1048576).toFixed(2)
      }MB vecI8=${(r.build.vecI8Bytes / 1048576).toFixed(2)}MB`,
    );
    console.log(
      `  lexical p50/p95=${r.bench.lexical.p50}/${r.bench.lexical.p95}ms · semF32=${r.bench.semanticF32.p50}/${r.bench.semanticF32.p95}ms · semI8=${r.bench.semanticI8.p50}/${r.bench.semanticI8.p95}ms`,
    );
    console.log(
      `  roundtrip: lexical=${r.lexicalRT}ms semF32=${r.semanticF32RT}ms semI8=${r.semanticI8RT}ms`,
    );
    console.log(
      `  MAIN-THREAD long tasks=${r.mainThreadLongTasks.count} ${
        JSON.stringify(r.mainThreadLongTasks.durationsMs)
      }`,
    );
    console.log(`  OPFS: ${JSON.stringify(r.opfs)}`);
    console.log(`  sqlite: ${JSON.stringify(r.sqlite)}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
