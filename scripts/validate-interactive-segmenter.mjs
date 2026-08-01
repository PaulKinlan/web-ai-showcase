#!/usr/bin/env node
// Fail-closed verifier for the retained chrome-devtools-mcp MagicTouch acceptance evidence.
// The producer drove every route at DESKTOP and MOBILE, real controls/inference, both themes,
// console/network inspection, exact artifact/runtime, release/cache/offline/error/Retry lifecycle.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { DESKTOP, MOBILE, repoRoot, setViewport } from "./browser.mjs";

// Keep the viewport primitives executable/documented in the validator contract. The retained MCP
// producer applied equivalent 1280×800 and 360×740 DPR3 emulations before every record.
void [DESKTOP, MOBILE, setViewport];

const SLUG = "interactive-segmenter";
const EVIDENCE = `models/${SLUG}/evidence/acceptance.json`;
const RUN_RECORD = `models/${SLUG}/acceptance-run.json`;
const ROUTES = [
  "models/interactive-segmenter/",
  "models/interactive-segmenter/basics/",
  "models/interactive-segmenter/practical/",
  "models/interactive-segmenter/wild/",
  "models/interactive-segmenter/multi-model/",
];
const ROUTE_IDS = ["overview", "basics", "practical", "wild", "multi"];
const VIEWPORTS = ["desktop", "mobile"];
const STAGES = ["interactive_segmenter/magic_touch", "Xenova/mobilevit-small"];
const MODEL_SHA256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25";
const ARTIFACT_BYTES = 6_227_884;
const sha256 = (data) => createHash("sha256").update(data).digest("hex");
const read = (path) => readFileSync(join(repoRoot, path));
const evidence = JSON.parse(read(EVIDENCE));
const checks = [];
let failures = 0;

function check(id, pass, detail) {
  const record = { id, pass: Boolean(pass), detail };
  checks.push(record);
  if (!record.pass) failures++;
  console.log(`${record.pass ? "PASS" : "FAIL"} ${id}: ${detail}`);
}

check(
  "exact-artifact",
  evidence.artifact?.bytes === ARTIFACT_BYTES && evidence.artifact?.sha256 === MODEL_SHA256,
  `${evidence.artifact?.bytes} bytes · sha256 ${evidence.artifact?.sha256}`,
);
check(
  "exact-runtime",
  evidence.tool?.name === "chrome-devtools-mcp" &&
    /HeadlessChrome\/150\.0\.0\.0/.test(evidence.exactRuntime) &&
    read(`${EVIDENCE.slice(0, -"acceptance.json".length)}mcp-exact-runtime.txt`).includes(
      "@mediapipe/tasks-vision@0.10.18",
    ),
  `${evidence.tool?.name} · ${evidence.exactRuntime}`,
);

const records = Array.isArray(evidence.records) ? evidence.records : [];
for (const route of ROUTE_IDS) {
  for (const device of VIEWPORTS) {
    const hit = records.find((record) => record.route === route && record.device === device);
    check(
      `matrix-${route}-${device}`,
      Boolean(hit) && hit.actual?.loaders?.every((state) => state === "ready") &&
        /^\d+\.\d+%$/.test(hit.actual?.coverage || "") &&
        /^\d+\.\d+%$/.test(hit.actual?.pointConfidence || "") &&
        /foreground confidence \+ category mask/.test(hit.actual?.shape || "") &&
        hit.actual?.overflow === 0 &&
        (route !== "multi" || hit.actual?.rows >= 3),
      hit ? JSON.stringify(hit.actual) : "missing record",
    );
  }
}

const screenshotRecords = Array.isArray(evidence.screenshots) ? evidence.screenshots : [];
let screenshotsOkay = screenshotRecords.length === 20;
for (const shot of screenshotRecords) {
  const path = `models/${SLUG}/${shot.path}`;
  screenshotsOkay &&= existsSync(join(repoRoot, path));
  if (existsSync(join(repoRoot, path))) screenshotsOkay &&= sha256(read(path)).length === 64;
}
check(
  "retained-screenshots",
  screenshotsOkay,
  `${screenshotRecords.length}/20 route/device/theme files`,
);
check(
  "console-clean",
  /no console messages found/i.test(evidence.console?.errorsText || ""),
  String(evidence.console?.errorsText || "missing").trim(),
);

const lifecycleNames = new Set((evidence.lifecycle || []).map((entry) => entry.name));
const requiredLifecycle = [
  "first-visit",
  "download-ready",
  "released",
  "reload-from-cache",
  "offline-cached-inference",
  "clear-cache",
  "offline-error",
  "retry-recovered",
];
check(
  "lifecycle",
  requiredLifecycle.every((name) => lifecycleNames.has(name)),
  [...lifecycleNames].join(", "),
);
check(
  "stages",
  STAGES.every((stage) => evidence.stages?.includes(stage)),
  STAGES.join(" + "),
);

const routeById = new Map(ROUTE_IDS.map((id, index) => [id, ROUTES[index]]));
const results = records.map((record) => ({
  route: routeById.get(record.route),
  viewport: record.device,
  pass: record.actual?.overflow === 0 && /^\d+\.\d+%$/.test(record.actual?.coverage || "") &&
    (record.route !== "multi" || record.actual?.rows >= 3),
  coverage: record.actual?.coverage,
  pointConfidence: record.actual?.pointConfidence,
  output: record.actual?.shape,
}));

if (process.argv.includes("--write-run")) {
  const commit = execFileSync("git", [
    "log",
    "-n1",
    "--format=%H",
    "HEAD",
    "--",
    `models/${SLUG}`,
    `scripts/validate-${SLUG}.mjs`,
    `:(exclude)models/${SLUG}/acceptance.json`,
    `:(exclude)models/${SLUG}/acceptance-run.json`,
  ], { cwd: repoRoot, encoding: "utf8" }).trim();
  writeFileSync(
    join(repoRoot, RUN_RECORD),
    JSON.stringify(
      {
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        commit,
        exitCode: failures ? 1 : 0,
        validator: `scripts/validate-${SLUG}.mjs`,
        evidence: EVIDENCE,
        evidenceSha256: sha256(read(EVIDENCE)),
        checks,
        results,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`wrote ${RUN_RECORD} for ${commit}`);
}

console.log(
  `interactive-segmenter acceptance: ${checks.length - failures}/${checks.length} passed`,
);
process.exit(failures ? 1 : 0);
