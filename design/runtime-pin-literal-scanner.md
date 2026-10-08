# Literal runtime-pin scanner — design and migration

> **Amendment (web-ai-showcase-j9z implementation, 2026-10-08):** the "local,
> zero-dependency lexical state machine" below was REJECTED by the nested-template red fixture
> (`ea72200`) and superseded by the pinned-parser architecture of
> `design/runtime-pin-ast-parser.md` (bead `web-ai-showcase-j6i`, landed `889f796`). Everywhere
> this document says "lexer"/"lexical state machine", read: acorn AST spans +
> `classifyJsOffset` via the single entry point `scripts/runtime-pin-parser.mjs`. The ledger
> `context` vocabulary was mechanically renamed in the same migration (template/tool-template →
> `template-dynamic`, tool-regex → `regex`/`regex-escaped`, json-prose/json-generated-note →
> `json`, wildcard-coverage-note → `string`); the ledger amendment records the mapping. All
> other semantics — whole raw specifier, no truncation/decoding, strict ledger equality,
> discovery never auto-approves, fail-closed unsupported contexts, additive counters,
> preservation invariants — are unchanged and binding.

Design record: `web-ai-showcase-kvt` (independent DeepSeek review `9760bfa4`). Implementation: `web-ai-showcase-j9z`; dynamic/bare/split URL coverage remains `vs2`.

## Phase 1: additive, bounded coverage

Discover complete, unescaped `http(s)` CDN URL literals for `@huggingface/transformers` and `onnxruntime-web` in executable JS/MJS and HTML contexts. Use local, zero-dependency JS/HTML lexical state machines (code, comments, quoted strings, template strings, HTML tags/display-only regions); no generic version-token regular expression. A non-interpolated backtick is a literal. Escapes, interpolated backticks, ambiguous syntax and unquoted runtime URL attributes are unsupported: fail unless an exact reviewed exception applies. JS comments and HTML comments/`<pre>`/`<code>` are inert for this new pass; the legacy numeric scanner retains its deliberate prose false-red tests. Do not claim execution/dataflow proof for a literal in source code.

Keep the existing numeric grep/JS scan, strict-raw suffix tests, Transformers route-scoped override, ONNX Runtime global version set, derived `sw.js` region plus exact measured-version exemptions, binary-NUL checks and legacy per-runtime scan floors (`>=350` Transformers and `>=120` ORT). Phase 2 replacement or comment-semantic changes require a separate bead and explicit approval.

## Whole raw package specifier

Retain the entire raw quoted URL before invoking WHATWG `new URL(raw)`. Validate `http(s)`, the approved CDN origin and its package path, including the two-segment scoped package name. Slice the ORIGINAL URL after the full `package@` marker to the next real `/` path separator or end of the literal. The nonempty resulting raw candidate must equal an authorized version exactly, with no decoding, normalization, prefix acceptance or stripping. A `?`, `#`, `%`, `^`, `~`, `|`, `+`, `.`, `_`, `!`, `@`, `;`, `:`, `,` or `&` before that real slash is part of the candidate and therefore rejects; `%2F` is not a real slash. Query/hash after `/dist/...` are outside the version candidate, consistent with the legacy path boundary. An empty candidate such as `".../transformers@" + version` fails unsupported, not clean. Every matching literal on a line must be judged independently; a later bad URL cannot hide behind an earlier authorized one.

## Coverage and reviewed golden ledger

`inventory/runtime-pin-marker-ledger.json` holds the reviewed baseline of **21** nonnumeric/escaped contiguous marker occurrences: dynamic templates D=7, tooling patterns T=9, inert comments C=2, inert wildcard W=1 and non-executable JSON J=2. It also records the `scripts/runtime-integrity.mjs` split pair at lines 113/126 separately: the pair has **no contiguous marker** and is a `vs2` gap, not a passing literal URL. The approximately 559 numeric hits remain covered by the legacy scan; this ledger does not allowlist runtime versions.

The source-discovery pass proposes candidates but NEVER writes/auto-approves the golden ledger. Reconcile the discovered sorted `(path, source-line SHA-256, raw/escaped marker, context, ordinalOnLine, expectedCountOnLine)` entries against the reviewed file with strict equality. `ordinalOnLine` is 1-based marker position among occurrences on that source line; `expectedCountOnLine` is the total marker multiplicity on that line. A reorder, second marker, deletion, changed line/span, changed context or new marker (including JSON) fails until a human-reviewed ledger change is committed. The baseline hashes cover the **full raw source line**, LF excluded, including package marker AND version/token. If a later migration chooses expression-span hashes, the span MUST still include the complete package name/marker and raw token, not merely `${ORT_VERSION}`. The split pair is pinned by exact path plus both source-line fingerprints; a new split/percent-encoded assembly is not discovered by the contiguous-marker pass and remains an explicit `vs2` residual.

The D entries lock template source TEXT only; they do **not** validate resolved `ORT_VERSION`/`ORT_VER` constants. They are explicit, independently reviewed `vs2` coverage gaps, not approval of dynamic runtime URLs. A new dynamic expression is unsupported by default. Tooling/wildcard/JSON exceptions are per exact path, source fingerprint, context and occurrence count—never blanket exemptions for `scripts/`, `models/`, or JSON. Regex-literal escaped-slash `@huggingface\\/transformers@` must be discovered as a marker variant for census/unsupported classification, not decoded into an allowed URL candidate. Any NUL-bearing file with either raw package marker fails closed, even if its candidate is floating. No shell interpolation for new discovery; use Node fs/argv APIs.

## Acceptance and stop conditions

First prove RED on the existing isolated `test/runtime-pins.test.mjs` fixture: floating/partial/range CDN URLs (`latest`, `next`, `3`, `3.7`, `^3.7.5`, `v3.7.5`, `3.x`, ORT `latest` and `1`) currently return gate status 0 incorrectly. GREEN after the fix requires all nine to fail naming the entire candidate, while exact `3.7.5` and `1.21.0` URL controls remain green. Add boundary mutants for punctuation, encoded slash, real slash and bare end; two URLs on one line; JS/HTML executable versus comment/display contexts; escapes/templates/concat; marker ledger additions/edits/deletions, JSON and same-line duplicates; binary NUL; derived measured/rogue; scoped override. Require a real positive complete literal URL for **each** runtime (>0 discovery counters) alongside unchanged legacy numeric floors. Use focused tests, independent different-family exact-head review, then `fleet-check` once per committed final tree. No browser or live catalogue scan. Stop rather than add a heuristic if the bounded lexer cannot classify an executable literal or a new systemic context appears.
