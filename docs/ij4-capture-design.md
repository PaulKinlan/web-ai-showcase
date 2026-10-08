# ij4 capture design (design/dry-run stage, no browser run) — v2, corrected

Goal: attribute the ACTUAL network requests `models/embeddinggemma-2/` makes at runtime so
`runtime-integrity.json` can pin exactly the URLs this route uses — no guessed pins.

## v2 CORRECTION NOTICE (read this first)

v1 of this document asserted "blocker B1": that the chosen tool's click matcher would not match this
route's cold-case button. **That claim was WRONG and is retracted.** A cold, absent model renders
`Download model (~175 MB)` (`lib/model-loader.js:427-434`, state `download-required`), which the
existing regex `/Download|Retry|.../i` matches (`scripts/measure-raw-ort-dual-runtime.mjs:161`).
`Load model into memory` is rendered only in the `released` state (`lib/model-loader.js:307`).
No matcher change is needed. How I got it wrong: I read the list of button labels without tracing
the STATE CONDITIONS that select them, so I treated a post-release label as the cold-case one.
v1 also linked that wrong cause to the five earlier INCONCLUSIVE runs. That link is withdrawn — the
real cause is B0 below, and it is evidenced rather than inferred.

## B0 — PRIMARY, BLOCKING, and the proven cause of all five failed runs

**This route is WebGPU-only, and every tool used so far launched Chrome with `--disable-gpu`.**

- route: `models/embeddinggemma-2/basics/index.html:216` sets `requiresWebGPU: true`, and `:220`
  loads with `device: "webgpu"`. Verified independently during ik9: `wasm+q4` fails with
  `GatherBlockQuantized`; only `webgpu+q4` works.
- harness default: `scripts/browser.mjs:369-375` — `const webgpu = options.webgpu ?? false`, and
  `getChromeLaunchArgs` yields `["--disable-gpu"]` when false vs
  `["--enable-unsafe-webgpu","--use-angle=vulkan","--enable-features=Vulkan"]` when true
  (`scripts/browser.mjs:254-258`).
- callers: `scripts/verify-runtime-integrity-coverage.mjs:51` calls `launchChrome({})`, and the
  chosen `scripts/measure-raw-ort-dual-runtime.mjs:71-75` calls `launchChrome({userDataDir, ...})`.
  Neither passes `webgpu: true`.
- consequence: `lib/model-loader.js:360-375` gates on `model.requiresWebGPU` by awaiting
  `adapterAvailable()` BEFORE importing the runtime. With `--disable-gpu` the capability check
  cannot pass, so the runtime is never imported and **zero** jsDelivr requests are made.

That is exactly the observed signature of all five failed runs: navigation succeeded, zero controls
became clickable, zero jsDelivr requests, INCONCLUSIVE every time. The adapter MUST pass
`webgpu: true` and MUST fail closed (abort with a clear reason) if WebGPU is unavailable, so a
capability failure can never again be mistaken for "this route fetches nothing".

## Instrument choice (unchanged, and still correct)

Chosen: `scripts/measure-raw-ort-dual-runtime.mjs`, which polls `.model-loader` `dataset.state`
until every loader reads `"ready"` (`:139-161`), auto-attaches to `worker`/`service_worker` targets
and calls `Network.enable` per session (`:82-95, :121-126, :207-211`), and whose `classify()` already
buckets `@huggingface/transformers@` and `onnxruntime-web@` into js vs wasm (`:56-64`).

## B1 — RETRACTED (see correction notice)

## B2 — must-fix: fixed, non-removed profile

`PROFILE = join(tmpdir(), "webai-dual-runtime-profile")` (`:30`) with `resetProfile: false` and
`removeProfileOnKill: false` (`:72-74`); `--fresh` wipes at START only (`:66`). Two consequences:
a disk leak of weights+runtime, and — worse for correctness — a warm profile may satisfy runtime
requests from HTTP cache, so the captured set could UNDERSTATE a cold visit. Cache hits need not
erase every CDP request event, but the risk is real and the fix is cheap. Run cold, remove in
`finally`, and prefer a unique per-run directory. Wrap server and Chrome startup in the same cleanup.

## B3 — must-fix: the instrument does not hash decoded bytes

It records URL and status and the ENCODED transfer length (`:97-114, :167-201`); it does not read or
hash response bodies. The manifest policy hashes DECODED bytes, and the difference is material: the
4.3.1 entrypoint is 167590 on the wire vs 586230 decoded. The adapter must fetch and SHA-256 the
decoded bytes for each exact observed successful URL.

## B4 — must-fix: scope BOTH loops

Adding one entry currently schedules desktop, mobile and an overview control (`:232-243`). The filter
must apply to both loops, and the design must name which cold route/viewport produces the
authoritative URL set. Do not silently run the full 30-minute route set.

## B5 — must-fix: early-worker-request race

Network enablement on attach is asynchronous and `waitForDebuggerOnStart: false` (`:124`), so a
worker that starts and fetches very early can be missed. Auto-attach alone is NOT proof of complete
capture. The adapter must settle this (e.g. attach-and-enable before navigation, or re-verify by
comparing against a known-must-appear URL) rather than assume it.

## Minimal additive adapter

1. one route entry (existing shape) for `models/embeddinggemma-2/basics/`;
2. `webgpu: true` plus fail-closed capability handling (B0);
3. cold, unique profile with `finally` removal (B2);
4. decoded-byte SHA-256 per observed successful URL (B3);
5. scoping applied to both loops, with the authoritative route/viewport named (B4);
6. a race-settling or re-verification step (B5).

## Pre-conditions to prove from code BEFORE the single permitted run

extended matcher reaches `ready` cold AND with WebGPU enabled; scoping cannot silently run the whole
route set; cold-profile lifecycle and leak check; the decoded-hash path; per-session request
correlation; fail-closed handling of incomplete or failed requests.

## What this does NOT claim

4.3.1 remains UNVERIFIED until a pin lands and a landed browser check proves it. The static hashes
below are candidates to CONFIRM, not pre-approved pins. A capture alone also cannot prove SW
protection on a first, uncontrolled visit.

## Static evidence banked (read-only, no browser)

- entrypoint `.../@huggingface/transformers@4.3.1`: no redirect, immutable, decoded 586230 bytes,
  sha256 `8d6716d9086f57c30a4bf367dba61b887593573c770c454465e8019b2703e743`
- `.../dist/ort-wasm-simd-threaded.jsep.mjs`: immutable, 46851, sha256 `c2f80e915e9df63289788a99d434d8c4e00e64e9c1f030f4b80b022458dd2c99`
- `...+esm`: immutable, 457197, sha256 `2b5fd7625907ddc4269a9a2420ef78159aaebbe87122e34f64c6327e5add6df1`
- 4.3.1 absent from all 30 landed manifest entries.
- the bundle builds its ORT url as a TEMPLATE (`onnxruntime-web@${pt.versions.web}/dist/`); two of
  four static path probes 404'd, so the transitive set cannot be enumerated statically.
- ORT `1.31.0-dev.20260914-8d85527a0` asyncify `.mjs`/`.wasm` are ALREADY pinned and immutable.
