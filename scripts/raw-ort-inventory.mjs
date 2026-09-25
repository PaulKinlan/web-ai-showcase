#!/usr/bin/env node
// Raw-ORT runtime-pin inventory (bead web-ai-showcase-62m).
//
// The raw-ORT routes (self-managed InferenceSession, no transformers.js) pin onnxruntime-web in
// four different ways — literal CDN URLs, an ORT_VER / ORT_VERSION constant, four bundle variants
// (ort.min, ort.wasm.min, ort.webgpu.min, ort.all.min) — so a literal grep undercounts. This script
// resolves every form from the source of truth and reports:
//   * slug, resolved version, how it is pinned, bundle variant, requested execution providers;
//   * bundle bloat: routes that ship a WebGPU/all bundle but only ever request the wasm EP;
//   * dual-runtime routes: a route that also loads @huggingface/transformers (two ORT builds in one
//     page — the double-WASM cost check in the bead).
//
// Usage:
//   node scripts/raw-ort-inventory.mjs            # markdown table + summary
//   node scripts/raw-ort-inventory.mjs --json     # machine-readable
//   node scripts/raw-ort-inventory.mjs --md       # markdown only (report regeneration)
//
// Read-only and network-free; the same tree always produces the same rows.

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const PIN_RE = /onnxruntime-web@([0-9]+\.[0-9]+\.[0-9]+)/;
const CONST_RE = /(?:ORT_VER|ORT_VERSION)\s*=\s*["']([0-9]+\.[0-9]+\.[0-9]+)["']/;
const BUNDLE_RE = /ort(?:\.(wasm|webgpu|all))?\.min\.mjs/;
const TJS_RE = /@huggingface\/transformers@([0-9]+\.[0-9]+\.[0-9]+)/;

function filesIn(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...filesIn(p));
    else if (/\.(js|mjs|html|json)$/.test(e.name) && !/_questions\.json$/.test(e.name)) out.push(p);
  }
  return out;
}

function rowFor(slug) {
  const dir = join(ROOT, "models", slug);
  if (!existsSync(dir)) return null;
  const files = filesIn(dir);
  // worker.js owns inference; fall back to the first file that mentions the runtime.
  const owners = files.filter((f) => PIN_RE.test(readFileSync(f, "utf8")) ||
    CONST_RE.test(readFileSync(f, "utf8")));
  if (owners.length === 0) return null;
  const file = owners.find((f) => f.endsWith("worker.js")) ?? owners[0];
  const src = readFileSync(file, "utf8");
  const literal = PIN_RE.exec(src);
  const constant = CONST_RE.exec(src);
  const bundle = BUNDLE_RE.exec(src);
  const eps = [
    ...new Set(
      [...src.matchAll(/executionProviders:\s*\[([^\]]*)\]/g)]
        .map((m) => m[1].replace(/["'\s]/g, ""))
        .filter(Boolean),
    ),
  ];
  const tjs = TJS_RE.exec(files.map((f) => readFileSync(f, "utf8")).join("\n"));
  return {
    slug,
    version: literal?.[1] ?? constant?.[1] ?? "unknown",
    pinned: literal ? "literal" : constant ? "constant" : "unknown",
    bundle: bundle ? (bundle[1] ? `ort.${bundle[1]}.min.mjs` : "ort.min.mjs") : "none",
    eps,
    dualRuntime: Boolean(tjs),
    tjsVersion: tjs?.[1] ?? null,
    file: file.slice(ROOT.length),
  };
}

export function inventory() {
  const slugs = readdirSync(join(ROOT, "models"), { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort();
  return slugs.map(rowFor).filter(Boolean);
}

const count = (rows, key) =>
  rows.reduce((acc, r) => ((acc[r[key]] = (acc[r[key]] ?? 0) + 1), acc), {});

export function renderMarkdown(rows) {
  const bloat = rows.filter((r) => r.bundle !== "ort.wasm.min.mjs" && r.eps.every((e) => e === "wasm"));
  const dual = rows.filter((r) => r.dualRuntime);
  const lines = [
    "# Raw-ORT runtime-pin inventory (web-ai-showcase-62m)",
    "",
    `Routes with a self-managed onnxruntime-web pin: **${rows.length}**. ` +
      `Versions: ${Object.entries(count(rows, "version")).map(([v, n]) => `${v}×${n}`).join(", ")}. ` +
      `Bundles: ${Object.entries(count(rows, "bundle")).map(([b, n]) => `${b}×${n}`).join(", ")}.`,
    "",
    "| slug | version | pinned via | bundle | execution providers | dual-runtime |",
    "| --- | --- | --- | --- | --- | --- |",
    ...rows.map((r) =>
      `| ${r.slug} | ${r.version} | ${r.pinned} | ${r.bundle} | ${r.eps.join(" ") || "(none)"} | ${
        r.dualRuntime ? `transformers.js ${r.tjsVersion}` : "no"
      } |`
    ),
    "",
    `## Bundle bloat — non-wasm bundle, wasm-only EP (${bloat.length})`,
    "",
    ...(bloat.length ? bloat.map((r) => `- **${r.slug}** ships \`${r.bundle}\` but requests \`wasm\` only (${r.version})`) : ["none"]),
    "",
    `## Dual-runtime routes — transformers.js + raw ORT in one page (${dual.length})`,
    "",
    ...(dual.length ? dual.map((r) => `- **${r.slug}** — raw ORT ${r.version} + transformers.js ${r.tjsVersion}`) : ["none"]),
    "",
  ];
  return lines.join("\n");
}

const isMain = process.argv[1] === fileURLToPath(import.meta.url);
if (isMain) {
  const rows = inventory();
  if (process.argv.includes("--json")) console.log(JSON.stringify(rows, null, 2));
  else console.log(renderMarkdown(rows));
}
