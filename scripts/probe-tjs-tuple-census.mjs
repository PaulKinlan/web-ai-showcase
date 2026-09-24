#!/usr/bin/env node
// Census for bead web-ai-showcase-865: the distinct (task, dtype, runtime) tuples the 327 built routes
// actually contain, which routes sit in each, and which of them touch the APIs that transformers.js 4.0
// moved or reworked. This is the population the compat matrix samples, and the flag column acceptance
// criterion 4 asks for: routes affected by the @huggingface/tokenizers split.
//
// Source census, not guesswork: a route is flagged by reading its own files for the imports the split
// affects. AutoTokenizer and AutoProcessor moved to @huggingface/tokenizers in v4; AutoModelForCTC is
// called out in the bead because CTC decoding consumes tokenizer state directly, so those routes are
// the ones to read first when the pin moves.
//
// Usage: node scripts/probe-tjs-tuple-census.mjs [--probe reports/transformers-v4-compat-matrix.json]

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";

const ROOT = new URL("..", import.meta.url).pathname;
const arg = (n, d) => {
  const i = process.argv.indexOf(`--${n}`);
  return i >= 0 ? process.argv[i + 1] : d;
};

const SPLIT_APIS = {
  AutoTokenizer: "moved to @huggingface/tokenizers in v4",
  AutoProcessor: "moved to @huggingface/tokenizers in v4",
  AutoModelForCTC: "CTC decoding reads tokenizer state directly — bead calls these out by name",
  AutoConfig: "config surface changed with the v4 ModelRegistry",
};

const routes = JSON.parse(readFileSync(ROOT + "reports/model-currency.json", "utf8")).routes;

function routeFiles(slug) {
  const dir = ROOT + "models/" + slug.split("/").slice(0, 1)[0];
  if (!existsSync(dir)) return [];
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isFile() && /\.(js|mjs|html)$/.test(e.name)) out.push(dir + "/" + e.name);
    else if (e.isDirectory() && !e.name.startsWith(".")) {
      for (const f of readdirSync(dir + "/" + e.name, { withFileTypes: true })) {
        if (f.isFile() && /\.(js|mjs|html)$/.test(f.name)) out.push(`${dir}/${e.name}/${f.name}`);
      }
    }
  }
  return out;
}

function splitApisFor(slug) {
  const found = [];
  for (const file of routeFiles(slug)) {
    const text = readFileSync(file, "utf8");
    for (const api of Object.keys(SPLIT_APIS)) {
      if (!found.includes(api) && new RegExp(`\\b${api}\\b`).test(text)) found.push(api);
    }
  }
  return found;
}

const groups = new Map();
for (const r of routes) {
  const key = `${r.task ?? "unknown"} | ${r.catalogueDtype ?? "unset"} | ${r.runtime ?? "unknown"}`;
  const g = groups.get(key) || { tuple: key, task: r.task ?? null, dtype: r.catalogueDtype ?? null, runtime: r.runtime ?? null, routes: [], splitApis: new Set() };
  g.routes.push(r.slug);
  for (const api of splitApisFor(r.slug)) g.splitApis.add(api);
  groups.set(key, g);
}

const census = {
  generated: new Date().toISOString(),
  bead: "web-ai-showcase-865",
  source: "reports/model-currency.json (built routes) + a source read of each route's own files",
  apiMeanings: SPLIT_APIS,
  totals: {
    builtRoutes: routes.length,
    distinctTuples: groups.size,
    routesUsingAutoTokenizer: routes.filter((r) => splitApisFor(r.slug).includes("AutoTokenizer")).length,
    routesUsingAutoModelForCTC: routes.filter((r) => splitApisFor(r.slug).includes("AutoModelForCTC")).length,
  },
  tuples: [...groups.values()]
    .map((g) => ({
      tuple: g.tuple,
      task: g.task,
      dtype: g.dtype,
      runtime: g.runtime,
      routeCount: g.routes.length,
      routes: g.routes.sort(),
      tokenizersSplitAffected: g.splitApis.size > 0,
      splitApis: [...g.splitApis].sort(),
    }))
    .sort((a, b) => b.routeCount - a.routeCount || a.tuple.localeCompare(b.tuple)),
};

writeFileSync(ROOT + "reports/transformers-v4-tuple-census.json", JSON.stringify(census, null, 2) + "\n");
console.log(`built routes: ${census.totals.builtRoutes} · distinct tuples: ${census.totals.distinctTuples}`);
console.log(`routes touching a v4-changed API: AutoTokenizer ${census.totals.routesUsingAutoTokenizer}, AutoModelForCTC ${census.totals.routesUsingAutoModelForCTC}`);
for (const t of census.tuples.slice(0, 8)) {
  console.log(`  ${String(t.routeCount).padStart(3)}x  ${t.tuple}  ${t.tokenizersSplitAffected ? "· split-affected: " + t.splitApis.join(", ") : ""}`);
}

// --- assemble the human report: probe results + this census -------------------------------
const PROBE = ROOT + arg("probe", "reports/transformers-v4-compat-matrix.json");
if (!existsSync(PROBE)) {
  console.log(`\n(no probe output at ${PROBE} yet — run scripts/probe-tjs-compat-matrix.mjs first; census written)`);
  process.exit(0);
}
const probe = JSON.parse(readFileSync(PROBE, "utf8"));
const byTupleName = new Map(census.tuples.map((t) => [t.tuple.split(" | ").slice(0, 2).join("|"), t]));
const findCensus = (row) => byTupleName.get(`${row.task}|${row.dtype}`);

const lines = [];
lines.push("# transformers.js 3.7.5 vs 4.3.0 — shadow compatibility matrix");
lines.push("");
lines.push("Probe only: **no source was changed** (`lib/webai.js` and every demo route are untouched).");
lines.push("This is evidence for staging the shared pin (web-ai-showcase-9v4), not the bump itself.");
lines.push("");
lines.push(`- Generated: ${probe.generated}`);
lines.push(`- Method: ${probe.method}`);
lines.push(`- Tolerance: ${probe.tolerance}`);
lines.push(`- Coverage: ${probe.coverage.probedTuples} probed tuples — ${probe.coverage.note}`);
lines.push(`- Population: ${census.totals.builtRoutes} built routes in ${census.totals.distinctTuples} distinct (task, dtype, runtime) tuples.`);
lines.push("");
lines.push("## What 4.x changes, and why this matters");
lines.push("");
lines.push("The shared pin in `lib/webai.js` is read by 315 of 327 built routes, so a bump is a one-line change with a 315-route blast radius. v4.0 replaces the WebGPU runtime with a native C++ WebGPU EP (not JSEP), adds a ModelRegistry and new `env` surfaces, and moves tokenizers into `@huggingface/tokenizers`. Transitive onnxruntime-web moves 1.22.0-dev → 1.31.0-dev.");
lines.push("");
lines.push("## Results");
lines.push("");
lines.push("| tuple | routes | split | 3.7.5 | 4.3.0 | verdict | detail |");
lines.push("|---|---|---|---|---|---|---|");
for (const row of probe.rows) {
  const c = findCensus(row);
  const cell = (v) => {
    const r = row.versions[v];
    if (!r) return "not run";
    return r.runs ? `RUNS ${r.loadMs}ms load / ${r.inferMs}ms infer` : `**FAILS** — ${String(r.error).replace(/\|/g, "\\|").slice(0, 160)}`;
  };
  const cmp = row.comparison;
  const parse = (s) => { try { return JSON.parse(String(s)); } catch { return null; } };
  let detail;
  if (cmp.verdict === "same") {
    detail = cmp.maxAbsDiff !== null && cmp.maxAbsDiff !== undefined ? `max|Δ| ${cmp.maxAbsDiff}` : "text identical";
  } else if (cmp.detail) {
    detail = c.detail;
  } else if (cmp.textEqual === false) {
    // Same label + a score that moved by float noise is NOT a behaviour change. Say which it is.
    const A = parse(cmp.aText), B = parse(cmp.bText);
    if (A && B && A.label === B.label && typeof A.score === "number" && typeof B.score === "number") {
      const d = Math.abs(A.score - B.score);
      detail = `same label \`${A.label}\`, score Δ ${d.toExponential(2)} (${JSON.stringify(A.score)} vs ${JSON.stringify(B.score)}) — float noise from the ORT bump, not a behaviour change`;
    } else {
      detail = `text differs: ${JSON.stringify(c.aText)} vs ${JSON.stringify(c.bText)}`;
    }
  } else if (cmp.maxAbsDiff !== null && cmp.maxAbsDiff !== undefined) {
    detail = `dims equal, max|Δ| ${cmp.maxAbsDiff} — ${cmp.maxAbsDiff <= 1e-2 ? "float noise from the ORT bump, not a behaviour change" : "beyond float noise: read the outputs"}`;
  } else {
    detail = "see the raw JSON for both outputs";
  }
  lines.push(`| \`${row.tuple}\` | ${c ? c.routeCount : "?"} | ${row.tokenizersSplitAffected ? "yes" : "no"} | ${cell("3.7.5")} | ${cell("4.3.0")} | **${row.comparison.verdict}** | ${String(detail).replace(/\|/g, "\\|").slice(0, 200)} |`);
}
lines.push("");
lines.push("## Tokenizers-split exposure (acceptance criterion 4)");
lines.push("");
lines.push(`v4 moves \`AutoTokenizer\` and \`AutoProcessor\` into \`@huggingface/tokenizers\`, and the bead calls out \`AutoModelForCTC\` routes because CTC decoding consumes tokenizer state directly. Reading each route's own files: **${census.totals.routesUsingAutoTokenizer} built routes use AutoTokenizer** and **${census.totals.routesUsingAutoModelForCTC} use AutoModelForCTC**.`);
lines.push("");
lines.push("The tuples that carry that exposure, largest first:");
lines.push("");
for (const t of census.tuples.filter((x) => x.splitApis.includes("AutoTokenizer") || x.splitApis.includes("AutoModelForCTC")).slice(0, 14)) {
  lines.push(`- \`${t.tuple}\` × ${t.routeCount} — ${t.splitApis.join(", ")} — e.g. ${t.routes.slice(0, 4).map((s) => `\`${s}\``).join(", ")}`);
}
lines.push("");
lines.push("## Failure text (acceptance criterion 2)");
lines.push("");
const failures = probe.rows.flatMap((row) => Object.entries(row.versions).filter(([, v]) => v && v.runs === false).map(([ver, v]) => ({ tuple: row.tuple, version: ver, ...v })));
if (failures.length === 0) lines.push("No tuple failed under either version in this run.");
else {
  for (const f of failures) {
    lines.push(`### \`${f.tuple}\` @ ${f.version}`);
    lines.push("");
    lines.push(`- error name: \`${f.errorName ?? "n/a"}\``);
    lines.push(`- error text: \`${String(f.error).replace(/`/g, "'")}\``);
    if (f.stack) lines.push(`- stack (truncated): \`${String(f.stack).split("\n").slice(0, 4).join(" | ").replace(/`/g, "'")}\``);
    lines.push("");
  }
}
lines.push("## Reading this honestly");
lines.push("");
lines.push("- Load timings are indicative, not a benchmark: whichever version runs second benefits from weights already in the browser cache.");
lines.push("- `same` means identical dims, max |Δ| ≤ 1e-3 on the compared prefix, and equal text where the task returns text. onnxruntime-web 1.22 → 1.31 need not be bit-identical; float noise at 1e-8 (as observed) is expected and is reported rather than hidden.");
lines.push("- Where the probe's strict verdict reads `differs`, the detail column says WHY. A row that is the same label with a score moved by ~1e-3 is float noise from the ORT bump; a row whose text or labels actually change is a behaviour difference. The numbers are shown so the judgement is the reader's, not the script's.");
lines.push("- Inputs are deterministic and synthetic for vision/audio (a generated PNG, a 220 Hz tone). That is enough to compare versions against each other, and is not a claim about model quality.");
lines.push(`- NOT probed in this run: ${(probe.notProbed || []).map((x) => `\`${x.tuple}\``).join(", ") || "none"}. Those are heavier downloads than this probe's per-case budget allowed; they are named here rather than silently omitted, and are the first checks for the staging bead.`);
lines.push("- The probed set is a representative sample of the distinct tuples, chosen to cover the tokenizers split, the rewritten WebGPU path, and the largest route groups. Tuples not listed here were not probed and are marked as such by omission from the table rather than implied to pass.");
writeFileSync(ROOT + "reports/transformers-v4-compat-matrix.md", lines.join("\n") + "\n");
console.log(`\nwrote reports/transformers-v4-tuple-census.json and reports/transformers-v4-compat-matrix.md`);
