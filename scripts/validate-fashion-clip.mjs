#!/usr/bin/env node
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  evalBool,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

const ROOT = process.cwd();
const MODEL = "patrickjohncyh/fashion-clip";
const GENERAL_MODEL = "Xenova/clip-vit-base-patch16";
const REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7";
const BYTES = 605804513;
const family = join(ROOT, "models/fashion-clip");
const read = (path) => readFileSync(join(ROOT, path), "utf8");
const checks = [];
function check(id, pass, evidence) {
  checks.push({ id, pass: Boolean(pass), evidence });
}

const catalogue = JSON.parse(read("models.json")).models.find((entry) =>
  entry.slug === "fashion-clip"
);
check("catalogue-built", catalogue?.status === "built", catalogue?.status);
check(
  "catalogue-identity",
  catalogue?.hfId === MODEL && catalogue?.revision === REVISION && catalogue?.onnxBytes === BYTES &&
    catalogue?.dtype === "fp32" && catalogue?.sizeMB === 578,
  JSON.stringify({
    hfId: catalogue?.hfId,
    revision: catalogue?.revision,
    bytes: catalogue?.onnxBytes,
  }),
);

const routes = [
  "models/fashion-clip/",
  "models/fashion-clip/basics/",
  "models/fashion-clip/basics-attributes/",
  "models/fashion-clip/practical-catalog/",
  "models/fashion-clip/practical-search/",
  "models/fashion-clip/wild-briefs/",
  "models/fashion-clip/wild-audit/",
  "models/fashion-clip/multi-model/",
];
check(
  "stable-route-family",
  routes.every((route) => {
    const relative = route.replace("models/fashion-clip/", "");
    return statSync(join(family, relative, "index.html")).isFile();
  }),
  routes.join(", "),
);
const source = [read("models/fashion-clip/worker.js"), read("models/fashion-clip/fashion.js")].join(
  "\n",
);
check(
  "pinned-worker-inference",
  source.includes(MODEL) && source.includes(GENERAL_MODEL) && source.includes(REVISION) &&
    source.includes("loadPipeline") && source.includes("new Worker") &&
    source.includes("createModelLoader"),
  "exact FashionCLIP/general-CLIP stages + revision + worker + shared loader present",
);
check(
  "shared-release-lifecycle",
  source.includes("dispose()") && source.includes("worker.terminate()"),
  "worker termination is wired to createModelLoader dispose",
);

const preflight = read("models/fashion-clip/evidence/preflight-browser.log");
check(
  "real-browser-preflight",
  preflight.includes(`\"revision\":\"${REVISION}\"`) &&
    preflight.includes(`\"onnxBytes\":${BYTES}`) &&
    preflight.includes('"changed":true') && preflight.includes('"semantic":true') &&
    preflight.includes('"label":"a white sneaker","score":0.999817') &&
    preflight.includes('"label":"a sports t-shirt","score":0.999728'),
  "canonical fp32 ONNX produced discriminating semantic rankings",
);

const acceptance = JSON.parse(read("models/fashion-clip/evidence/acceptance.json"));
const expected = new Set();
for (
  const route of [
    "",
    "basics/",
    "basics-attributes/",
    "practical-catalog/",
    "practical-search/",
    "wild-briefs/",
    "wild-audit/",
    "multi-model/",
  ]
) {
  for (const device of ["desktop", "mobile"]) {
    for (const theme of ["light", "dark"]) {
      expected.add(`/models/fashion-clip/${route}|${device}|${theme}`);
    }
  }
}
const observed = new Set(
  acceptance.records.map((record) => `${record.route}|${record.device}|${record.theme}`),
);
check(
  "acceptance-matrix-32",
  acceptance.passed && acceptance.records.length === 32 && expected.size === observed.size &&
    [...expected].every((key) => observed.has(key)),
  `${acceptance.records.length}/32 cells`,
);
check(
  "console-network-clean",
  acceptance.records.every((record) =>
    record.consoleErrors.length === 0 && record.networkFailures.length === 0 &&
    record.httpFailures.length === 0
  ),
  "zero recorded console/network/HTTP failures",
);
const driven = acceptance.records.filter((record) => record.theme === "light");
check(
  "real-results-both-devices",
  driven.length === 16 &&
    driven.every((record) =>
      record.assertions?.rows === 4 && record.assertions?.resultRows === 4 &&
      record.assertions?.overflow === false && record.assertions?.unnamedButtons === 0 &&
      record.assertions?.missingAlt === 0 &&
      record.assertions?.status === "Real local inference complete."
    ),
  `${driven.length}/16 driven cells with 4 tensor/result rows and no overflow/a11y failures`,
);
check(
  "multi-model-real-stages",
  driven.filter((record) => record.route.endsWith("multi-model/"))
    .every((record) => record.assertions.generalRows === 4 && record.assertions.generalTop),
  "FashionCLIP + general CLIP each returned 4 real rows on desktop and mobile",
);
check(
  "release-reload-all-routes",
  acceptance.lifecycleByRoute.length === 8 &&
    acceptance.lifecycleByRoute.every((item) => item.released && item.readyAfterReload),
  `${acceptance.lifecycleByRoute.length}/8 route lifecycle checks passed`,
);
check(
  "screenshots-retained",
  acceptance.records.every((record) => statSync(join(ROOT, record.screenshot)).size > 1000),
  "32 non-empty route/device/theme screenshots",
);

const critique = JSON.parse(read("models/fashion-clip/_questions.json"));
const conformance = JSON.parse(read("models/fashion-clip/conformance.json"));
check(
  "critique-and-immutable-conformance",
  critique.guidanceConsulted.length >= 5 && conformance.immutable === true &&
    conformance.suiteHash?.startsWith("sha256:") && conformance.assertions.length >= 20,
  `${critique.guidanceConsulted.length} guidance records; ${conformance.assertions.length} immutable assertions`,
);

async function validateLiveShell() {
  const { server, port } = await startServer();
  const chrome = await launchChrome({
    userDataDir: join(ROOT, ".fashion-clip-validator-profile"),
    resetProfile: true,
  });
  const cdp = new CDP(chrome.ws);
  const results = [];
  try {
    for (const route of routes) {
      const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/${route}`);
      for (const [name, viewport] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
        await setViewport(cdp, page.sessionId, viewport);
        const good = await evalBool(
          cdp,
          page.sessionId,
          "document.documentElement.scrollWidth <= document.documentElement.clientWidth && " +
            "document.querySelectorAll('[role=status]').length >= 1 && " +
            "[...document.querySelectorAll('button')].every(b => b.textContent.trim() || b.getAttribute('aria-label'))",
        );
        results.push({
          route,
          viewport: name,
          pass: good && !page.errors.length && !page.netFailures.length,
        });
      }
      await closePage(cdp, page.targetId);
    }
  } finally {
    await chrome.kill();
    await new Promise((resolve) => server.close(resolve));
  }
  for (const result of results) {
    console.log(`${result.pass ? "PASS" : "FAIL"} live-shell ${result.route} ${result.viewport}`);
  }
  return results.every((result) => result.pass);
}

for (const result of checks) {
  console.log(`${result.pass ? "PASS" : "FAIL"} ${result.id}: ${result.evidence}`);
}
let pass = checks.every((result) => result.pass);
if (process.argv.includes("--live-shell")) pass = (await validateLiveShell()) && pass;
console.log(
  `\nfashion-clip acceptance: ${checks.filter((x) => x.pass).length}/${checks.length} passed`,
);
process.exit(pass ? 0 : 1);
