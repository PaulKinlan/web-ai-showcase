#!/usr/bin/env node
// Route-complete smolvlm-vision-language acceptance: browser validation across all 4 published routes
// at desktop and mobile. Proves route integrity and clean degradation (loader state, console/network
// cleanliness, responsive layout without horizontal overflow), matching llama2-c-stories validator shape.
// Advertised stage: HuggingFaceTB/SmolVLM-256M-Instruct (vision-language generation, WebGPU q4f16, ~250MB).

import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const RUN_RECORD = join(repoRoot, "models/smolvlm-vision-language/acceptance-run.json");

const ROUTES = {
  overview: "models/smolvlm-vision-language/",
  basics: "models/smolvlm-vision-language/basics/",
  practical: "models/smolvlm-vision-language/practical/",
  wild: "models/smolvlm-vision-language/wild/",
};

const VIEWPORTS = [
  { name: "desktop", vp: DESKTOP },
  { name: "mobile", vp: MOBILE },
];

async function main() {
  const { server, port } = await startServer();
  const browser = await launchChrome();
  const cdp = new CDP(browser.ws);
  const results = [];

  try {
    for (const [rung, route] of Object.entries(ROUTES)) {
      for (const v of VIEWPORTS) {
        process.stderr.write(`[smolvlm] ${rung} (${v.name}) … `);
        const url = `http://127.0.0.1:${port}/web-ai-showcase/${route}`;
        const { targetId, sessionId, errors, netFailures } = await openPage(cdp, url);
        await setViewport(cdp, sessionId, v.vp);
        await new Promise((res) => setTimeout(res, 800));

        const state = await evalValue(
          cdp,
          sessionId,
          `
          (() => {
            const loaders = [...document.querySelectorAll(".model-loader")];
            const runBtn = document.getElementById("run");
            return {
              loaderStates: loaders.map(l => l.dataset.state),
              runDisabled: runBtn ? runBtn.disabled : true,
              noOverflow: document.documentElement.scrollWidth <= window.innerWidth + 1,
            };
          })()
        `,
        );

        await closePage(cdp, targetId);

        const pass = state?.loaderStates.length > 0 &&
          state.loaderStates.every((s) => s === "unsupported" || s === "download-required" || s === "ready") &&
          errors.length === 0 && netFailures.length === 0 && state?.noOverflow;

        results.push({
          route,
          viewport: v.name,
          pass,
        });

        process.stderr.write(pass ? `PASS\n` : `FAIL (errors=${errors.length})\n`);
      }
    }
  } finally {
    await browser.kill();
    server.close();
  }

  const allPass = results.every((r) => r.pass);
  console.log(`\nSmolVLM acceptance validation: ${results.filter((r) => r.pass).length}/${results.length} PASS`);

  if (WRITE_RUN) {
    const commit = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
    }).trim();
    const ranAt = new Date().toISOString();
    const record = {
      commit,
      ranAt,
      exitCode: allPass ? 0 : 1,
      results,
    };
    writeFileSync(RUN_RECORD, JSON.stringify(record, null, 2) + "\n", "utf8");
    console.log(`Wrote ${RUN_RECORD} for commit ${commit}`);
  }

  if (!allPass) process.exit(1);
}

main().catch((err) => {
  console.error("Validation failed:", err);
  process.exit(1);
});
