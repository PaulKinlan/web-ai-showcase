// EmbeddingGemma 2 acceptance: validator-owned split-run half artifacts + mechanical merge.
//
// WHY: the full both-viewport acceptance matrix takes ~2750s, but the box reaps a browser at 45 min
// (2700s). So the matrix is split into two viewport-scoped runs (desktop-only, mobile-only), each
// comfortably under the limit, and the two HALF records are merged into the single 10-cell record the
// project expects (models/embeddinggemma-2/acceptance-run.json).
//
// HALF PATH (documented contract — raw per-half evidence lives here, never at the final path):
//   models/embeddinggemma-2/acceptance-runs/desktop-half.json
//   models/embeddinggemma-2/acceptance-runs/mobile-half.json
// A single-viewport run with --write-run writes ONLY its half; it can never write the final record.
// The final record is produced exclusively by `node scripts/validate-embeddinggemma-2.mjs --merge-halves`
// after both halves exist and pass every fail-closed rule below.
//
// FAIL-CLOSED RULES (each prints a clear reason; a refusal writes nothing):
//   • a missing half                        → no record
//   • exitCode !== 0 or pass !== true       → no record (failed / timed out)
//   • commit or validator blob sha differs between halves, or from the HEAD/blob being merged → no record (stale)
//   • the same viewport supplied twice      → no record (duplicate)
//   • a half without exactly 5 results      → no record
//   • duplicate/overlapping routes within a viewport, or a route set that is not the expected 5 → no record
//   • any half cell whose viewport does not match its half, or that is not passing → no record
// Publish happens only at 10/10 cells, atomically (temp file + rename), and is idempotent: re-merging
// the same good halves produces a byte-identical record (every field is derived from the halves).
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export const VALIDATOR_PATH = fileURLToPath(new URL("./validate-embeddinggemma-2.mjs", import.meta.url));
export const FINAL_RECORD_PATH = join(repoRoot, "models/embeddinggemma-2/acceptance-run.json");
export const HALF_DIR = join(repoRoot, "models/embeddinggemma-2/acceptance-runs");
export const HALF_PATHS = {
  desktop: join(HALF_DIR, "desktop-half.json"),
  mobile: join(HALF_DIR, "mobile-half.json"),
};
// The exact 5 routes the family advertises (models/embeddinggemma-2/acceptance.json). The merge requires
// each viewport half to cover exactly this set, once each — the same 5 unique routes per viewport.
export const EXPECTED_ROUTES = [
  "models/embeddinggemma-2/",
  "models/embeddinggemma-2/basics/",
  "models/embeddinggemma-2/practical/",
  "models/embeddinggemma-2/wild/",
  "models/embeddinggemma-2/multi-model/",
];
export const HALF_KIND = "embeddinggemma-2-half";

export function sha256(text) {
  return createHash("sha256").update(String(text)).digest("hex");
}

/** Hash of the validator source AS RUN — a half is stale if this differs from what is being merged. */
export function computeValidatorBlobSha() {
  return sha256(readFileSync(VALIDATOR_PATH, "utf8"));
}

/**
 * Atomically write JSON (temp file in the same directory, then rename). A crash mid-write leaves only a
 * uniquely-named temp file, never a truncated record at the target path.
 */
function atomicWriteJson(path, obj) {
  const dir = dirname(path);
  mkdirSync(dir, { recursive: true });
  const tmp = join(
    dir,
    `.${basename(path)}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  );
  try {
    writeFileSync(tmp, JSON.stringify(obj, null, 2) + "\n", "utf8");
    renameSync(tmp, path);
  } catch (err) {
    try {
      rmSync(tmp, { force: true });
    } catch {
      /* best-effort cleanup */
    }
    throw err;
  }
}

/**
 * Write one viewport-scoped half artifact. `viewport` is the viewport the validator ACTUALLY ran (it is
 * derived from the run loop, never from a flag), so a half cannot claim a viewport it did not run.
 */
export function writeHalfRecord({ viewport, results, commit, ranAt, exitCode, pass, validatorBlobSha }) {
  if (viewport !== "desktop" && viewport !== "mobile") {
    throw new Error(`writeHalfRecord: viewport must be desktop|mobile, got ${JSON.stringify(viewport)}`);
  }
  const path = HALF_PATHS[viewport];
  atomicWriteJson(path, {
    kind: HALF_KIND,
    viewport,
    commit,
    validatorBlobSha,
    ranAt,
    exitCode,
    pass,
    results,
  });
  console.log(`WROTE half ${path} for commit ${commit} (${results.length} cells, exit ${exitCode})`);
  return path;
}

function fail(reason) {
  return { ok: false, reason, record: null };
}

function routesMatchExpected(routes) {
  if (routes.length !== EXPECTED_ROUTES.length) return false;
  const sorted = [...routes].sort();
  const expected = [...EXPECTED_ROUTES].sort();
  return sorted.every((r, i) => r === expected[i]);
}

/**
 * Pure validation of the two parsed halves. Returns { ok, reason, record } — on success `record` is the
 * exact merged object (before any disk write), so the write step can stay atomic and separate.
 */
export function validateHalfPair({ desktop, mobile, currentCommit, currentValidatorBlobSha }) {
  const REQUIRED = ["viewport", "results", "commit", "validatorBlobSha", "ranAt", "exitCode", "pass"];
  for (const [label, half] of [["desktop", desktop], ["mobile", mobile]]) {
    if (!half || typeof half !== "object" || Array.isArray(half)) {
      return fail(`missing half: the ${label} half is absent or not an object`);
    }
    for (const field of REQUIRED) {
      if (half[field] === undefined) {
        return fail(`malformed ${label} half: missing required field "${field}"`);
      }
    }
    if (!Array.isArray(half.results)) return fail(`malformed ${label} half: "results" is not an array`);
    if (typeof half.commit !== "string" || !half.commit) return fail(`malformed ${label} half: "commit" is not a non-empty string`);
    if (typeof half.validatorBlobSha !== "string" || !half.validatorBlobSha) return fail(`malformed ${label} half: "validatorBlobSha" is not a non-empty string`);
    if (typeof half.exitCode !== "number" || !Number.isInteger(half.exitCode)) return fail(`malformed ${label} half: "exitCode" is not an integer`);
    if (typeof half.pass !== "boolean") return fail(`malformed ${label} half: "pass" is not a boolean`);
  }

  // Duplicate / slot integrity: a half is read from its viewport-named path and must claim that viewport.
  if (desktop.viewport === mobile.viewport) {
    return fail(`duplicate half: both halves claim viewport "${desktop.viewport}"`);
  }
  if (desktop.viewport !== "desktop") {
    return fail(`viewport mismatch: the desktop half slot claims viewport "${desktop.viewport}"`);
  }
  if (mobile.viewport !== "mobile") {
    return fail(`viewport mismatch: the mobile half slot claims viewport "${mobile.viewport}"`);
  }

  // Failed / timed out half.
  for (const [label, half] of [["desktop", desktop], ["mobile", mobile]]) {
    if (half.exitCode !== 0) {
      return fail(`${label} half is failed/timed out: exitCode is ${half.exitCode}, not 0`);
    }
    if (half.pass !== true) {
      return fail(`${label} half is marked failed: pass is false`);
    }
  }

  // Stale: source commit and validator blob must agree across halves AND with the HEAD/blob being merged.
  if (desktop.commit !== mobile.commit) {
    return fail(`stale halves: source commit differs (${desktop.commit.slice(0, 7)} vs ${mobile.commit.slice(0, 7)})`);
  }
  if (currentCommit && desktop.commit !== currentCommit) {
    return fail(`stale half: halves cite commit ${desktop.commit.slice(0, 7)} but HEAD is ${currentCommit.slice(0, 7)}`);
  }
  if (desktop.validatorBlobSha !== mobile.validatorBlobSha) {
    return fail(`stale halves: validator blob sha differs (${desktop.validatorBlobSha.slice(0, 12)}… vs ${mobile.validatorBlobSha.slice(0, 12)}…)`);
  }
  if (currentValidatorBlobSha && desktop.validatorBlobSha !== currentValidatorBlobSha) {
    return fail(`stale half: halves cite validator blob ${desktop.validatorBlobSha.slice(0, 12)}… but the current validator is ${currentValidatorBlobSha.slice(0, 12)}…`);
  }

  // Exactly 5 results per half, the expected 5 unique routes, no overlap within a viewport, and every
  // cell must claim the half's own viewport and be passing.
  for (const [label, half] of [["desktop", desktop], ["mobile", mobile]]) {
    if (half.results.length !== EXPECTED_ROUTES.length) {
      return fail(`${label} half has ${half.results.length} results (expected exactly ${EXPECTED_ROUTES.length})`);
    }
    const routes = half.results.map((r) => r?.route);
    if (routes.some((r) => typeof r !== "string" || !r)) {
      return fail(`${label} half has a result with a missing/non-string route`);
    }
    if (new Set(routes).size !== routes.length) {
      const dup = routes.find((r, i) => routes.indexOf(r) !== i);
      return fail(`${label} half has a duplicate route within the viewport: "${dup}"`);
    }
    if (!routesMatchExpected(routes)) {
      return fail(`${label} half routes do not match the expected 5-route set (got ${JSON.stringify(routes)})`);
    }
    for (const cell of half.results) {
      if (cell.viewport !== half.viewport) {
        return fail(`${label} half cell for route "${cell.route}" claims viewport "${cell.viewport}" but the half ran "${half.viewport}"`);
      }
      if (cell.pass !== true) {
        return fail(`${label} half cell for route "${cell.route}" is not passing`);
      }
    }
  }

  // Canonical ordering: desktop block first (routes in expected order), then the mobile block. Cells are
  // rebuilt here so the final record carries EXACTLY {route, viewport, pass} — no half bookkeeping leaks in.
  const results = [
    ...EXPECTED_ROUTES.map((route) => ({ route, viewport: "desktop", pass: true })),
    ...EXPECTED_ROUTES.map((route) => ({ route, viewport: "mobile", pass: true })),
  ];

  const halves = [
    {
      viewport: "desktop",
      path: "models/embeddinggemma-2/acceptance-runs/desktop-half.json",
      commit: desktop.commit,
      validatorBlobSha: desktop.validatorBlobSha,
      ranAt: desktop.ranAt,
      exitCode: desktop.exitCode,
      pass: desktop.pass,
    },
    {
      viewport: "mobile",
      path: "models/embeddinggemma-2/acceptance-runs/mobile-half.json",
      commit: mobile.commit,
      validatorBlobSha: mobile.validatorBlobSha,
      ranAt: mobile.ranAt,
      exitCode: mobile.exitCode,
      pass: mobile.pass,
    },
  ];
  // Deterministic ranAt (latest half run) so re-merging the same halves yields byte-identical output.
  const ranAt = [desktop.ranAt, mobile.ranAt].filter(Boolean).sort().at(-1) ?? "";

  const record = {
    commit: desktop.commit,
    ranAt,
    exitCode: 0,
    results,
    splitMethod: {
      kind: "viewport-split-merge",
      assembledFrom: "two viewport-scoped half runs merged into one 10-cell record",
      validator: "scripts/validate-embeddinggemma-2.mjs",
      validatorBlobSha: desktop.validatorBlobSha,
      halves,
    },
  };

  return { ok: true, reason: null, record };
}

/**
 * Read both halves, validate, and atomically publish the merged 10-cell record. Returns
 * { ok, reason, record } — a refusal returns ok:false and writes nothing, so a pre-existing good record
 * is never clobbered by a bad merge.
 */
export function mergeHalves({
  desktopHalfPath = HALF_PATHS.desktop,
  mobileHalfPath = HALF_PATHS.mobile,
  finalRecordPath = FINAL_RECORD_PATH,
  currentCommit,
  currentValidatorBlobSha,
} = {}) {
  const read = (path, label) => {
    if (!existsSync(path)) return { error: `missing half: ${path} does not exist (${label} run not present)` };
    try {
      return { half: JSON.parse(readFileSync(path, "utf8")) };
    } catch (err) {
      return { error: `half ${path} does not parse: ${err.message}` };
    }
  };

  const desktopRead = read(desktopHalfPath, "desktop");
  if (desktopRead.error) return fail(desktopRead.error);
  const mobileRead = read(mobileHalfPath, "mobile");
  if (mobileRead.error) return fail(mobileRead.error);

  const verdict = validateHalfPair({
    desktop: desktopRead.half,
    mobile: mobileRead.half,
    currentCommit,
    currentValidatorBlobSha,
  });
  if (!verdict.ok) return verdict;

  atomicWriteJson(finalRecordPath, verdict.record);
  console.log(`MERGED halves -> ${finalRecordPath} (${verdict.record.results.length}/10 cells)`);
  return verdict;
}
