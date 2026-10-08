#!/usr/bin/env -S deno run --allow-net --allow-read --allow-write
// Generates the runtime integrity manifest used by sw.js to verify third-party runtime code before it
// is served from, or stored in, the shell cache.
//
// WHY THIS EXISTS: lib/webai.js loads @huggingface/transformers with a dynamic import(), and import()
// has no integrity option in browsers today, so a service worker is the only place the bytes can be
// checked. Modern Web Guidance's security guide says SRI is "strictly for immutable, versioned assets"
// and these jsDelivr version paths are exactly that (measured cache-control: immutable, max-age=1yr),
// but SRI's normal mechanism is unavailable to a dynamic import - hence verification in the SW.
//
// Two things that WILL silently break the manifest if forgotten:
//  1. HASH THE DECODED BYTES. jsDelivr reports the COMPRESSED size in content-length while serving
//     brotli/gzip, so the wasm reads as ~3.9 MiB over the wire and is 20.6 MiB decoded.
//  2. THE VERSION LIST IS NOT DERIVED FROM THE ROOT PIN. Three runtime versions are in use across the
//     repo (the shared pin plus allowlisted local overrides). A version used by a route but absent
//     here simply will not be trusted-or-cached by the SW; nothing breaks, that route just loses
//     offline reuse. Add it here when a new pin is introduced.
//
// Run: deno run --allow-net --allow-read --allow-write scripts/runtime-integrity.mjs
// This is a maintenance script and is NOT part of any gate: it needs network and the CDN contents only
// change when a pinned version changes.
// No external import: adding jsr:@std/path would mean a lockfile change for two path joins, and
// import.meta.url gives a URL that Deno's read/write helpers accept directly.
const VERSIONS = ["3.7.5", "4.2.0", "4.3.0"];
// Assets the bundle needs but does NOT name through its public path, so the emitted-name scan misses
// them. ort-wasm-simd-threaded.jsep.mjs is the Emscripten glue that loads the wasm; without it in the
// manifest the worker treats it as unknown and never persists it, which breaks 3.7.5 routes offline.
const EXTRA_ASSETS = {
  "3.7.5": ["ort-wasm-simd-threaded.jsep.mjs"],
};
// NOT COVERED BY THIS MANIFEST, and deliberately not silently.
//
// CORRECTION, because an earlier version of this comment was WRONG and it was used to ask for a policy
// decision: it claimed 42 references used an UNVERSIONED onnxruntime-web/dist/ path that could not be
// hash-pinned. That was an artifact of my own grep, whose sed stripped the @version before counting.
// A literal search finds ZERO unversioned references: 69 pin @1.21.0 and 16 pin @1.20.1. Every ort
// reference is versioned and therefore pinnable, and the assets below are pinned explicitly.
const ORT_ASSETS = [
  // Loaded directly by model workers (34 + 1 + 8 references).
  "onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs",
  "onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs",
  "onnxruntime-web@1.21.0/dist/ort.min.mjs",
  // Companion loader + binary assets, from the assets recorded as actually fetched during a real run in
  // reports/dual-runtime-62m3.json. Taken from that evidence rather than guessed, so an asset the
  // report does not show being fetched is deliberately not vouched for here.
  "onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.jsep.mjs",
  "onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.jsep.wasm",
  "onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs",
  "onnxruntime-web@1.21.0/dist/ort-wasm-simd-threaded.mjs",
  "onnxruntime-web@1.21.0/dist/ort-wasm-simd-threaded.wasm",
  // 1.22.0 is not referenced by any worker: it appears in the run report, so something in the stack
  // reaches for it. Pinned from observation; if it stops being fetched this entry is harmless.
  "onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.mjs",
  "onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.wasm",
  "onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs",
  // Found by the browser audit: the transformers 4.3.0 route (all-distilroberta-v1) pulls a DATE-STAMPED
  // PRERELEASE ort build, chosen inside 4.3.0 rather than by any worker here. It is still an exact
  // immutable artifact - measured no redirect, cache-control immutable, and byte-identical across two
  // fetches - so it is pinnable. Worth knowing when reviewing the 4.3.0 rollout that a built route
  // depends on a -dev. build; if 4.3.0 ever changes which ort it selects, that version falls out of this
  // manifest and silently drops to pass-through (served unverified, never persisted).
  "onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.asyncify.mjs",
  "onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.asyncify.wasm",
];
// Exact versioned URLs used by BUILT routes that do not come from either bundle above. Every entry here
// was measured before being added: status 200, no redirect, no query string, and
// `cache-control: ... immutable`. A URL that failed any of those is NOT listed, because a mutable or
// redirecting URL cannot be pinned - the hash would change or apply to the wrong response - and per the
// decision on the bead such a URL must not enter the trusted shell cache.
//
// mediapipe: 7 built routes; the wasm directory contains EXACTLY these four files, so the set is
// complete rather than a guess. FilesetResolver picks the simd or nosimd pair at runtime, so both are
// needed or one backend loses offline.
// outetts: 1 built route. Its ESM bundle statically pulls one further pinned transformers version.
// It also references /npm/fs/+esm, which is a 404 and unversioned - a phantom, not a served asset - so
// it is deliberately absent and is served pass-through without ever being persisted.
const PINNED_EXACT = [
  "@mediapipe/tasks-vision@0.10.18",
  "@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_internal.js",
  "@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_internal.wasm",
  "@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_nosimd_internal.js",
  "@mediapipe/tasks-vision@0.10.18/wasm/vision_wasm_nosimd_internal.wasm",
  "outetts@0.2.0/+esm",
  "@huggingface/transformers@3.1.2/+esm",
  // Found by the browser completeness audit, not by reading source: the outetts route's 3.1.2 bundle
  // fetches its own ort binary, which the emitted-name scan cannot see because 3.1.2 constructs the URL
  // at runtime. Its jsep .mjs glue returns 404 for this version (the glue is inlined), so only the .wasm
  // is pinned - pinning a 404 would be pinning nothing.
  "@huggingface/transformers@3.1.2/dist/ort-wasm-simd-threaded.jsep.wasm",
];
// Every listed URL is verified on serve and then cached. Membership IS the policy: a URL absent from
// this manifest is pass-through, served from the network unverified and never persisted. That is NOT
// integrity protection: the only thing pass-through buys is that unverified bytes cannot be REUSED from
// our cache. The field is recorded per URL so the decision is reviewable per asset
// rather than implied by the code path, and a test pins its value.
const POLICY = "verify-then-cache";
const NOT_COVERED = [
  "onnxruntime-web@*/dist/** not listed above (other variants of the pinned versions)",
  "onnxruntime-web/dist/* (unversioned - none referenced today, but unpinnable if ever introduced)",
  "@mediapipe/tasks-vision@*/** other than the four wasm files listed",
  "cdn.jsdelivr.net/npm/fs/+esm (referenced by outetts, returns 404, unversioned)",
];
const PACKAGE = "@huggingface/transformers";
const manifestPath = new URL("../runtime-integrity.json", import.meta.url);
const swPath = new URL("../sw.js", import.meta.url);

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const manifest = {};
const skipped = [];

for (const version of VERSIONS) {
  const entry = `https://cdn.jsdelivr.net/npm/${PACKAGE}@${version}`;
  const res = await fetch(entry, { signal: AbortSignal.timeout(30000) });
  if (!res.ok) {
    console.error(`SKIP ${version}: entry fetch status ${res.status}`);
    continue;
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  manifest[entry] = { sha256: await sha256(bytes), bytes: bytes.length, policy: POLICY };

  // The bundle resolves its own runtime assets through its public path, so those literals enumerate
  // the sibling files it can request. Anything else it might fetch cannot be enumerated statically,
  // which is a limitation recorded in the manifest itself.
  const text = new TextDecoder().decode(bytes);
  const names = new Set([...text.matchAll(/n\.p\s*\+\s*"([^"]+)"/g)].map((m) => m[1]));
  for (const extra of EXTRA_ASSETS[version] ?? []) names.add(extra);
  for (const name of [...names].sort()) {
    const url = `${entry}/dist/${name}`;
    try {
      const assetRes = await fetch(url, { signal: AbortSignal.timeout(60000) });
      if (!assetRes.ok) {
        skipped.push(`${url} (status ${assetRes.status})`);
        continue;
      }
      const assetBytes = new Uint8Array(await assetRes.arrayBuffer());
      manifest[url] = { sha256: await sha256(assetBytes), bytes: assetBytes.length, policy: POLICY };
    } catch (error) {
      skipped.push(`${url} (${error.message})`);
    }
  }
}

// Orthogonal third-party runtime: model workers import it directly rather than through
// @huggingface/transformers, so the bundle scan above cannot see it. Listed explicitly, and a version
// that grows new companion assets would fall to the unpinned path until it is added here.
for (const asset of ORT_ASSETS) {
  const url = `https://cdn.jsdelivr.net/npm/${asset}`;
  try {
    const assetRes = await fetch(url, { signal: AbortSignal.timeout(60000) });
    if (!assetRes.ok) {
      skipped.push(`${url} (status ${assetRes.status})`);
      continue;
    }
    const assetBytes = new Uint8Array(await assetRes.arrayBuffer());
    manifest[url] = { sha256: await sha256(assetBytes), bytes: assetBytes.length, policy: POLICY };
  } catch (error) {
    // A pinned asset that disappeared is a real signal, not noise: it means a worker's pinned URL no
    // longer resolves, so it is reported rather than swallowed.
    skipped.push(`${url} (${error.message})`);
  }
}

// Exact versioned URLs used by built routes outside both bundles above, each already measured.
for (const asset of PINNED_EXACT) {
  const url = `https://cdn.jsdelivr.net/npm/${asset}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(60000) });
    if (!res.ok) {
      skipped.push(`${url} (status ${res.status})`);
      continue;
    }
    const cacheControl = res.headers.get("cache-control") ?? "";
    const bytes = new Uint8Array(await res.arrayBuffer());
    manifest[url] = { sha256: await sha256(bytes), bytes: bytes.length, policy: POLICY };
    // Pinning a non-immutable URL would be worse than not pinning it: the hash would silently stop
    // matching and the asset would fail closed. Report it loudly instead.
    if (!/immutable/.test(cacheControl)) {
      console.error(`WARNING not immutable, revisit before trusting: ${url} (${cacheControl || "no cache-control"})`);
    }
    if (res.url !== url) {
      console.error(`WARNING redirected, pin the destination instead: ${url} -> ${res.url}`);
    }
  } catch (error) {
    skipped.push(`${url} (${error.message})`);
  }
}

const ordered = Object.fromEntries(Object.entries(manifest).sort());
const doc = {
  "//": "Generated by scripts/runtime-integrity.mjs. Do not hand-edit. sha256 is over DECODED bytes.",
  generatedFor: `${PACKAGE} versions ${VERSIONS.join(", ")}, pinned onnxruntime-web assets, and exact URLs used by built mediapipe/outetts routes`,
  limitation:
    "Covers the @huggingface/transformers entry bundles, the asset filenames they emit via their public path, and an explicit list of pinned onnxruntime-web, mediapipe and outetts assets taken from worker references and a recorded runtime report. A URL that is NOT in this manifest receives no integrity check at all: sw.js fetches it and hands the bytes to the page unverified. Pass-through is not integrity protection - it only means unverified bytes are never persisted and so cannot be reused from our cache. Not covered either: the first uncontrolled dynamic import before the SW is controlling, and any browser HTTP-cache poisoning, which happens below the SW.",
  notCovered: NOT_COVERED,
  urls: ordered,
};
if (skipped.length) doc.skipped = skipped;

await Deno.writeTextFile(manifestPath, `${JSON.stringify(doc, null, 2)}\n`);

// Embed the same data in sw.js between markers so the worker needs NO network to obtain it: an
// importScripts() of a separate manifest file would fail offline, which is precisely when the cached
// runtime is being used. A test asserts the embedded block matches this JSON, catching drift.
const sw = await Deno.readTextFile(swPath);
const START = "// >>> runtime-integrity (generated by scripts/runtime-integrity.mjs)";
const END = "// <<< runtime-integrity";
const block = `${START}\nconst RUNTIME_INTEGRITY = ${JSON.stringify(ordered, null, 2)};\n${END}`;
const startIndex = sw.indexOf(START);
const endIndex = sw.indexOf(END);
if (startIndex === -1 || endIndex === -1) {
  console.error("sw.js has no runtime-integrity marker block - add it before generating.");
  Deno.exit(1);
}
const updated = sw.slice(0, startIndex) + block + sw.slice(endIndex + END.length);
await Deno.writeTextFile(swPath, updated);

console.log(`wrote ${manifestPath} with ${Object.keys(ordered).length} urls`);
for (const [url, v] of Object.entries(ordered)) {
  console.log(`  ${v.sha256}  ${(v.bytes / 1024 / 1024).toFixed(2)} MiB  ${url}`);
}
if (skipped.length) console.log(`skipped (referenced but not served): ${skipped.join("; ")}`);
