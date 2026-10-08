# ij4 capture design (design/dry-run stage, no browser run)

Goal: attribute the ACTUAL network requests that `models/embeddinggemma-2/` makes at runtime,
so `runtime-integrity.json` can pin exactly the URLs this route uses - no guessed pins.

## Instrument choice

Rejected: `scripts/verify-runtime-integrity-coverage.mjs`. Five runs, three of them mine,
all INCONCLUSIVE. Its trigger probes four fixed selectors ONCE each and bails on `disabled`;
this route's loader is disabled while its async capability check runs, so the probe sees zero
controls. That tool is tuned for other routes.

Chosen: `scripts/measure-raw-ort-dual-runtime.mjs`. It already does what ij4 needs:
- drives every loader to `ready` by polling `.model-loader` `dataset.state` until all are
  `"ready"` (`loadAllEngines`, budget 30 min) - status-based, not one-click;
- attaches and enables `Network.enable` **per worker/service_worker session** via auto-attach,
  so Worker-originated runtime fetches are visible at all;
- `classify(url)` already buckets `@huggingface/transformers@` and `onnxruntime-web@` into
  js vs wasm - exactly the attribution ij4 needs.

## Compatibility: proven from code, not assumed

- ready signal: `lib/model-loader.js` sets `root.dataset.state = state` (line 138) and starts at
  `data-state="checking"` (line 75). The driver polls exactly that property. **Compatible.**
- worker sessions: driver auto-attaches and enables per worker session. **Compatible.**

## Two blockers found by dry-run (both must be fixed in the adapter)

### B1 (fatal if unfixed): the click matcher does not match this route's primary button
Driver regex: `/Download|Retry|Re-download|Continue|Retry local check/i`
Actual `lib/model-loader.js` button labels: "Load model into memory", "Re-download missing assets",
"Retry capability check", "Retry clear", "Retry local check", "Retry update", "Retry".
`"Load model into memory"` - the button shown when the model is ABSENT, i.e. the cold case we need -
matches NONE of those alternatives. With a cold profile the driver would click nothing, never reach
ready, and report the route unexercised. Any adapter MUST add a matcher for it.
This is very likely the same class of cause behind the failed coverage-tool runs.

### B2: fixed profile + no removal = leak, and it corrupts the measurement
`PROFILE = join(tmpdir(), "webai-dual-runtime-profile")` is a FIXED path, launched with
`resetProfile: false` and `removeProfileOnKill: false`, so the profile persists after the run.
`--fresh` wipes it at START only (line 66), never at END. Two consequences:
1. disk leak (this profile will hold model weights + runtime, hundreds of MB and growing), which is
   the exact pattern recorded in the fleet lessons as a repeated GiB-leak cause;
2. worse for correctness: a warm profile can satisfy runtime requests from the HTTP cache, so the
   capture may observe FEWER urls than a cold visit - and would understate the real fetch set.
The adapter MUST run cold (fresh profile) and remove the profile in a `finally`.

## Minimal additive adapter

1. Add ONE route entry, same shape as the existing ones:
   `{ slug: "embeddinggemma-2", route: "models/embeddinggemma-2/basics/",
      control: "models/embeddinggemma-2/",
      expect: "transformers.js 4.3.1 embedding runtime" }`
2. Extend the click matcher to include `Load model into memory` (additive alternative).
3. Scope the run to the single route (env filter, default unchanged) so it does not also pay the
   30-minute budget on unrelated routes.
4. Cold profile + guaranteed removal in `finally`; verify 0 leaked profiles after.
5. Capture, for every observed cdn.jsdelivr.net url: kind from `classify`, status, and
   DECODED-byte SHA-256 (the manifest policy hashes decoded bytes; wire content-length differs,
   e.g. the 4.3.1 entrypoint is 167590 wire vs 586230 decoded).

## What this does NOT claim

4.3.1 remains UNVERIFIED until a pin lands and a landed browser check proves it. This document is a
design; no protection claim attaches to it.

## Static evidence already banked (read-only, no browser)

- entrypoint `https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1`: no redirect,
  `immutable`, decoded 586230 bytes, sha256 `8d6716d9086f57c30a4bf367dba61b887593573c770c454465e8019b2703e743`
- `.../dist/ort-wasm-simd-threaded.jsep.mjs`: immutable, 46851 bytes, sha256 `c2f80e915e9df63289788a99d434d8c4e00e64e9c1f030f4b80b022458dd2c99`
- `...+esm`: immutable, 457197 bytes, sha256 `2b5fd7625907ddc4269a9a2420ef78159aaebbe87122e34f64c6327e5add6df1`
- 4.3.1 is absent from all 30 landed manifest entries.
- the bundle builds its ORT url as a TEMPLATE (`onnxruntime-web@${pt.versions.web}/dist/`), so the
  transitive set cannot be enumerated statically - two of four static path probes 404'd.
- ORT `1.31.0-dev.20260914-8d85527a0` asyncify `.mjs`/`.wasm` are ALREADY pinned and immutable,
  so if that version is selected the multi-MB wasm is already verify-then-cached.
