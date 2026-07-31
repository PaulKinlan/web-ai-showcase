#!/usr/bin/env node
// Fail-closed verifier for the structured output of produce-fashion-clip-evidence.mjs.
// Use --reproduce to run the real browser producer first. The default path is download-free but does
// not trust prose: it verifies the producer hash/source commit, frozen assertion denominator, every
// control/state/network record, exact artifact identities, and every screenshot content hash.

import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const PRODUCER = "scripts/produce-fashion-clip-evidence.mjs";
const EVIDENCE = "models/fashion-clip/evidence/acceptance.json";
const RUN = "models/fashion-clip/acceptance-run.json";
const MODELS = {
  fashion: {
    id: "patrickjohncyh/fashion-clip",
    revision: "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
    artifact: "onnx/model.onnx",
    bytes: 605804513,
  },
  general: {
    id: "Xenova/clip-vit-base-patch16",
    revision: "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
    artifact: "onnx/model_quantized.onnx",
    bytes: 152040303,
    sha256: "cf5b03d7c03cd78498b0d59a905552b549ae91af4e99ffb985103aa9424d2272",
  },
};
const routes = [
  "/models/fashion-clip/",
  "/models/fashion-clip/basics/",
  "/models/fashion-clip/basics-attributes/",
  "/models/fashion-clip/practical-catalog/",
  "/models/fashion-clip/practical-search/",
  "/models/fashion-clip/wild-briefs/",
  "/models/fashion-clip/wild-audit/",
  "/models/fashion-clip/multi-model/",
];
const requiredStates = [
  "first-visit-absent",
  "download-required",
  "downloading-progress",
  "initialising-ready",
  "current-cached-auto-init",
  "partial",
  "evicted",
  "error",
  "retry",
  "offline-cached",
  "stale-update",
  "corrupt-response-body",
  "unsupported-webgpu-capability",
  "release",
  "clear-cache",
];
const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");
const read = (path) => readFileSync(join(ROOT, path));

if (process.argv.includes("--reproduce")) {
  const result = spawnSync(process.execPath, [PRODUCER], {
    cwd: ROOT,
    stdio: "inherit",
    timeout: 3_600_000,
    killSignal: "SIGKILL",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const evidence = JSON.parse(read(EVIDENCE));
const run = JSON.parse(read(RUN));
const checks = [];
function check(id, pass, evidenceText) {
  checks.push({ id, pass: Boolean(pass), evidence: evidenceText });
  console.log(`${pass ? "PASS" : "FAIL"} ${id}: ${evidenceText}`);
}

const producerHash = sha256(read(PRODUCER));
const boundSourceCommit = execFileSync("git", [
  "log",
  "-n1",
  "--format=%H",
  "HEAD",
  "--",
  PRODUCER,
  "models/fashion-clip/worker.js",
  "models/fashion-clip/fashion.js",
  "models/fashion-clip/demo.js",
  "models/fashion-clip/index.html",
  "models/fashion-clip/multi-model/index.html",
  "models/fashion-clip/basics/index.html",
  "models/fashion-clip/basics-attributes/index.html",
  "models/fashion-clip/practical-catalog/index.html",
  "models/fashion-clip/practical-search/index.html",
  "models/fashion-clip/wild-briefs/index.html",
  "models/fashion-clip/wild-audit/index.html",
], { cwd: ROOT, encoding: "utf8" }).trim();
check(
  "producer-bound",
  evidence.schemaVersion === 2 && evidence.producer?.path === PRODUCER &&
    evidence.producer?.sha256 === producerHash &&
    evidence.producer?.sourceCommit === boundSourceCommit &&
    run.producerSha256 === producerHash && run.sourceCommit === boundSourceCommit,
  `producer sha256=${producerHash}; source commit=${boundSourceCommit}`,
);
check(
  "exact-model-identities",
  Object.entries(MODELS).every(([stage, model]) => {
    const actual = evidence.models?.[stage];
    return actual?.id === model.id && actual?.revision === model.revision &&
      actual?.artifact === model.artifact && actual?.bytes === model.bytes &&
      (!model.sha256 || actual?.sha256 === model.sha256);
  }),
  `${MODELS.fashion.id}@${MODELS.fashion.revision}; ${MODELS.general.id}@${MODELS.general.revision} (${MODELS.general.bytes} bytes, sha256 ${MODELS.general.sha256})`,
);

const cells = new Set();
let recordsOkay = evidence.records?.length === 16;
for (const record of evidence.records ?? []) {
  cells.add(`${record.route}|${record.device}`);
  const labels = record.realOutput?.labels ?? [];
  recordsOkay &&= routes.includes(record.route) && ["desktop", "mobile"].includes(record.device) &&
    record.controls?.firstEdit?.changed === true && record.controls?.secondEdit?.changed === true &&
    record.controls?.reset?.changed === true && record.controls?.upload?.keyboardFocused === true &&
    record.controls?.upload?.after?.name === "fashion-shirt.jpg" &&
    record.controls?.shirt?.pressed === "true" && record.controls?.sneaker?.pressed === "true" &&
    record.controls?.memoryControls?.length === (record.route.endsWith("multi-model/") ? 2 : 1) &&
    record.controls.memoryControls.every((item) =>
      item.summary?.focused && item.summary?.opened && item.summary?.closed &&
      ["pass", "not-applicable"].includes(item.measure?.status)
    ) &&
    record.realOutput?.status === "Real local inference complete." &&
    record.realOutput?.rows === 3 && record.realOutput?.resultRows === 3 &&
    labels.length === 3 && labels.every((label) =>
      label.startsWith(`edited ${record.routeId} ${record.device}`)
    ) &&
    (record.route.endsWith("multi-model/")
      ? record.realOutput?.generalRows === 3
      : record.realOutput?.generalRows === 0) &&
    record.lifecycle?.length === (record.route.endsWith("multi-model/") ? 2 : 1) &&
    record.lifecycle.every((item) =>
      item.release?.focused && item.reload?.focused && item.finalState === "ready"
    ) &&
    record.themes?.length === 2 && record.themes.every((theme) =>
      ["light", "dark"].includes(theme.theme) && theme.overflow === false &&
      theme.unnamedButtons === 0 && theme.missingAlt === 0 &&
      theme.status === "Real local inference complete." && theme.rows === 3 &&
      theme.resultRows === 3 &&
      (record.route.endsWith("multi-model/") ? theme.generalRows === 3 : theme.generalRows === 0)
    );
}
const expectedCells = routes.flatMap((route) =>
  ["desktop", "mobile"].map((device) => `${route}|${device}`)
);
recordsOkay &&= cells.size === 16 && expectedCells.every((cell) => cells.has(cell));
check(
  "route-control-inference-matrix",
  recordsOkay,
  `${cells.size}/16 route/device cells contain changed controls, key events, real output, both themes and release/re-init`,
);

const stateRows = evidence.stateMatrix ?? [];
const stateOkay = Object.keys(MODELS).every((stage) =>
  requiredStates.every((state) => {
    const matches = stateRows.filter((row) => row.stage === stage && row.state === state);
    if (matches.length === 0) return false;
    return matches.every((row) => {
      if (
        ["stale-update", "corrupt-response-body", "unsupported-webgpu-capability"].includes(state)
      ) {
        return row.status === "not-applicable" && row.evidence.length > 40;
      }
      return row.status === "pass";
    });
  })
) && !stateRows.some((row) => row.status === "fail");
check(
  "loader-state-matrix",
  stateOkay,
  `${stateRows.filter((row) => row.status === "pass").length} pass; ${
    stateRows.filter((row) => row.status === "not-applicable").length
  } honest N/A; 0 fail`,
);

const requests = evidence.network?.requests ?? [];
const networkReview = evidence.network?.review;
const requestRowsOkay = requests.length > 0 &&
  requests.every((request) =>
    /^https?:/.test(request.requestedUrl) &&
    (request.terminal?.startsWith("finished") ||
      ["redirect", "failed"].includes(request.terminal)) &&
    (!request.terminal?.startsWith("finished") || Number.isFinite(request.encodedDataLength)) &&
    (request.status == null || (request.status >= 200 && request.status < 400))
  );
const artifactProofOkay = Object.entries(MODELS).every(([stage, model]) => {
  const proof = evidence.network?.artifactProofs?.[stage];
  return proof?.initial?.requestedUrl &&
    decodeURIComponent(proof.initial.requestedUrl).includes(
      `/${model.id}/resolve/${model.revision}/${model.artifact}`,
    ) &&
    proof.initial.headers?.repoCommit === model.revision &&
    proof.initial.declaredBytes === model.bytes &&
    proof.final?.status === 200 && proof.final?.declaredBytes === model.bytes &&
    proof.initial.requestKey === proof.final.requestKey;
});
check(
  "complete-network-ledger",
  requestRowsOkay && networkReview?.complete === true &&
    networkReview.unexpectedOrigins?.length === 0 && networkReview.unpinned?.length === 0 &&
    networkReview.unfinished?.length === 0 && networkReview.badStatus?.length === 0 &&
    networkReview.unexpectedFailures?.length === 0 &&
    networkReview.expectedFailures?.length >= 2 &&
    new Set(networkReview.expectedFailures.map((item) => item.phase)).size === 2 &&
    artifactProofOkay,
  `${requests.length} requests; exact pinned artifacts=${artifactProofOkay}; expected blocked failures=${networkReview?.expectedFailures?.length}; unexpected=${networkReview?.unexpectedFailures?.length}`,
);
check(
  "general-content-digest",
  evidence.contentVerification?.general?.bytes === MODELS.general.bytes &&
    evidence.contentVerification?.general?.sha256 === MODELS.general.sha256 &&
    decodeURIComponent(evidence.contentVerification?.general?.url ?? "").includes(
      `/${MODELS.general.id}/resolve/${MODELS.general.revision}/${MODELS.general.artifact}`,
    ),
  `${evidence.contentVerification?.general?.bytes} bytes; sha256=${evidence.contentVerification?.general?.sha256}`,
);

const shots = evidence.screenshots ?? [];
const screenshotCells = new Set();
let screenshotsOkay = shots.length === 32;
for (const shot of shots) {
  screenshotCells.add(`${shot.route}|${shot.device}|${shot.theme}`);
  const absolute = join(ROOT, shot.path);
  const bytes = statSync(absolute).size;
  screenshotsOkay &&= bytes === shot.bytes && bytes > 1000 &&
    sha256(read(shot.path)) === shot.sha256 &&
    shot.resultState === "ready with changed-input real inference";
}
check(
  "bound-screenshots",
  screenshotsOkay && screenshotCells.size === 32,
  `${screenshotCells.size}/32 screenshot cells match retained byte lengths and SHA-256 digests`,
);

const frozen = evidence.denominator?.assertionIds ?? [];
const actual = (evidence.assertions ?? []).map((item) => item.id);
const assertionsOkay = evidence.passed === true && evidence.denominator?.exactOrder === true &&
  evidence.denominator?.assertions === frozen.length &&
  JSON.stringify(frozen) === JSON.stringify(actual) &&
  evidence.assertions.every((item) => item.status === "pass") &&
  run.frozenDenominator?.exactOrder === true &&
  JSON.stringify(run.frozenDenominator?.assertionIds) === JSON.stringify(frozen) &&
  run.exitCode === 0;
check(
  "frozen-denominator",
  assertionsOkay,
  `${actual.length}/${frozen.length} exact ordered assertions pass; producer passed=${evidence.passed}; run exit=${run.exitCode}`,
);

const pass = checks.every((item) => item.pass);
console.log(
  `\nFashionCLIP acceptance verification: ${
    checks.filter((item) => item.pass).length
  }/${checks.length} passed`,
);
process.exit(pass ? 0 : 1);
