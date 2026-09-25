# @mediapipe/tasks-vision version probe: 0.10.x vs 1.0.x

- **Generated:** 2026-09-25T17:39:43.835Z
- **Tool:** `scripts/probe-mpver.mjs`
- **Task tested:** `GestureRecognizer` init via `FilesetResolver` in Web Workers (headless Chrome via CDP)
- **Current pin:** `TASKS_VISION_VERSION = "0.10.18"` in `lib/mediapipe.js`

## Summary Results Table

| Version | Module Worker (Standard) | Module Worker (`useModule=true`) | Classic Worker (Standard) | Verdict |
|---|---|---|---|---|
| `0.10.18` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ✅ PASS (2154ms) | **Current Pin (Stable Classic)** |
| `0.10.20` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ✅ PASS (1004ms) | Unsupported |
| `0.10.21` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ❌ FAIL: `Failed to execute 'importScripts' on ...` | ✅ PASS (1131ms) | Unsupported |
| `0.10.22` | ❌ FAIL: `worker.onerror ` | ❌ FAIL: `worker.onerror ` | ❌ FAIL: `Failed to fetch dynamically imported ...` | Broken CDN bundle |
| `0.10.32` | ❌ FAIL: `self.import is not a function` | ❌ FAIL: `self.import is not a function` | ✅ PASS (1298ms) | Unsupported |
| `0.10.35` | ❌ FAIL: `ModuleFactory not set.` | ✅ PASS (1062ms) | ✅ PASS (903ms) | Unsupported |
| `1.0.0` | ❌ FAIL: `ModuleFactory not set.` | ✅ PASS (843ms) | ✅ PASS (697ms) | Unsupported |
| `1.0.1` | ❌ FAIL: `ModuleFactory not set.` | ✅ PASS (591ms) | ✅ PASS (617ms) | **Blocked: ModuleFactory not set** |

## Detailed Findings & Root Cause Analysis

### 1. Standard Module Worker Init Fails on 1.0.x (`ModuleFactory not set.`)
When calling `FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm")` inside a module worker (`{ type: "module" }`):
1. `importScripts()` is not supported and throws a `TypeError`.
2. MediaPipe's fallback catches the error and performs `await import(".../vision_wasm_internal.js")`.
3. Because `useModule` defaults to `false`, MediaPipe fetches `vision_wasm_internal.js`. That script wraps its output in `var ModuleFactory = (() => ...)` which is local to the module scope and does not set `globalThis.ModuleFactory`.
4. MediaPipe checks `if (!self.ModuleFactory) throw Error("ModuleFactory not set.");`, which aborts initialization immediately.

### 2. The `useModule=true` Asymmetry Across Worker Types
MediaPipe 1.0.x introduced an undocumented second parameter: `FilesetResolver.forVisionTasks(wasmPath, useModule = false)`.
- When `useModule = true` is passed, it loads `vision_wasm_module_internal.js`, which includes `globalThis.ModuleFactory = ModuleFactory; export default ModuleFactory;`. This allows module workers to succeed.
- However, if `useModule = true` is passed in a **classic worker**, `importScripts(".../vision_wasm_module_internal.js")` fails with a syntax error because `export default` is illegal in classic scripts.
- The shared helper in `lib/mediapipe.js` cannot blindly set `useModule = true` without breaking classic worker callers.

### 3. Current Pin Verdict: `0.10.18` Stays
- Acceptance criterion 2: *"If 1.0.1 fails module-worker init, the pin STAYS and the negative result is recorded (a blocked-style note) so it is not re-attempted."*
- Standard module-worker init fails on 1.0.1.
- In classic workers, `0.10.18` is completely stable and battle-tested across all 7 built routes:
  1. `gesture-recognizer`
  2. `hand-landmarker`
  3. `pose-landmarker`
  4. `face-detector`
  5. `face-landmarker`
  6. `image-segmenter`
  7. `interactive-segmenter`
- Bumping classic workers to 1.0.1 yields zero module-worker compatibility benefit while incurring major-version upgrade risk across all 7 production vision routes.
- Therefore, the pin in `lib/mediapipe.js` remains `0.10.18`.
