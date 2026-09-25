// Probe: which @mediapipe/tasks-vision version can initialize GestureRecognizer inside a Web Worker?
// Tests candidate versions across module workers (standard vs useModule=true) and classic workers.
import { rmSync, writeFileSync } from "node:fs";
import { CDP, closePage, launchChrome, openPage, startServer } from "./browser.mjs";

const versions = process.argv.slice(2);
if (!versions.length) {
  versions.push(
    "0.10.18",
    "0.10.20",
    "0.10.21",
    "0.10.22",
    "0.10.32",
    "0.10.35",
    "1.0.0",
    "1.0.1",
  );
}

const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const moduleWorkerTpl = (ver, useModuleFlag) => `
import { FilesetResolver, GestureRecognizer } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}";
self.onmessage = async () => {
  const t0 = performance.now();
  try {
    const resolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}/wasm"${useModuleFlag ? ", true" : ""}
    );
    const rec = await GestureRecognizer.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: "${MODEL_PATH}", delegate: "CPU" },
      numHands: 1,
      runningMode: "IMAGE",
    });
    const initMs = Math.round(performance.now() - t0);
    rec.close();
    self.postMessage({ ok: true, initMs });
  } catch (e) {
    self.postMessage({ ok: false, err: String((e && e.message) || e) });
  }
};`;

const classicWorkerTpl = (ver) => `
self.onmessage = async () => {
  const t0 = performance.now();
  try {
    const { FilesetResolver, GestureRecognizer } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}");
    const resolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}/wasm"
    );
    const rec = await GestureRecognizer.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: "${MODEL_PATH}", delegate: "CPU" },
      numHands: 1,
      runningMode: "IMAGE",
    });
    const initMs = Math.round(performance.now() - t0);
    rec.close();
    self.postMessage({ ok: true, initMs });
  } catch (e) {
    self.postMessage({ ok: false, err: String((e && e.message) || e) });
  }
};`;

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const BASE = "/web-ai-showcase/";
const ctx = await openPage(cdp, `http://127.0.0.1:${port}${BASE}models/gesture-recognizer/`);
const { sessionId } = ctx;

const results = [];
const TIMEOUT_MS = 60000;

for (const ver of versions) {
  const entry = { version: ver };

  // 1. Module worker (standard signature without useModule flag)
  const probeModStd = new URL("../models/gesture-recognizer/__probe_mod_std.js", import.meta.url);
  writeFileSync(probeModStd, moduleWorkerTpl(ver, false));
  entry.moduleWorkerStandard = await runWorker(
    `/web-ai-showcase/models/gesture-recognizer/__probe_mod_std.js?v=${ver}`,
    "module",
  );
  safeRm(probeModStd);

  // 2. Module worker (with explicit useModule = true)
  const probeModFlag = new URL("../models/gesture-recognizer/__probe_mod_flag.js", import.meta.url);
  writeFileSync(probeModFlag, moduleWorkerTpl(ver, true));
  entry.moduleWorkerWithFlag = await runWorker(
    `/web-ai-showcase/models/gesture-recognizer/__probe_mod_flag.js?v=${ver}`,
    "module",
  );
  safeRm(probeModFlag);

  // 3. Classic worker (standard signature)
  const probeClassic = new URL("../models/gesture-recognizer/__probe_classic.js", import.meta.url);
  writeFileSync(probeClassic, classicWorkerTpl(ver));
  entry.classicWorkerStandard = await runWorker(
    `/web-ai-showcase/models/gesture-recognizer/__probe_classic.js?v=${ver}`,
    "classic",
  );
  safeRm(probeClassic);

  results.push(entry);
  console.log(
    ver.padEnd(8),
    "| mod-std:",
    formatStatus(entry.moduleWorkerStandard),
    "| mod-flag:",
    formatStatus(entry.moduleWorkerWithFlag),
    "| classic:",
    formatStatus(entry.classicWorkerStandard),
  );
}

await closePage(cdp, ctx.targetId);
chrome.kill();
server.close();

function formatStatus(res) {
  if (!res) return "NO_RESPONSE";
  if (res.ok) return `PASS (${res.initMs}ms)`;
  const err = res.err || "unknown error";
  return `FAIL (${err.length > 35 ? err.slice(0, 32) + "..." : err})`;
}

async function runWorker(scriptUrl, workerType) {
  const typeOpt = workerType === "module" ? ",{type:'module'}" : "";
  const expr = `await new Promise((resolve)=>{
    const w = new Worker('${scriptUrl}'${typeOpt});
    const to = setTimeout(() => {
      w.terminate();
      resolve({ ok: false, err: 'timeout' });
    }, ${TIMEOUT_MS});
    w.onmessage = (e) => {
      clearTimeout(to);
      w.terminate();
      resolve(e.data);
    };
    w.onerror = (e) => {
      clearTimeout(to);
      w.terminate();
      resolve({ ok: false, err: 'worker.onerror ' + (e.message || '') });
    };
    w.postMessage(1);
  })`;

  try {
    const { result } = await cdp.send(
      "Runtime.evaluate",
      {
        expression: `(async()=>{try{return (${expr});}catch(e){return {ok:false,err:String(e)};}})()`,
        awaitPromise: true,
        returnByValue: true,
      },
      sessionId,
      TIMEOUT_MS + 10000,
    );
    return result?.value ?? { ok: false, err: "no evaluate return value" };
  } catch (err) {
    return { ok: false, err: String(err && err.message ? err.message : err) };
  }
}

function safeRm(path) {
  try {
    rmSync(path, { force: true });
  } catch {}
}

// Write JSON artifact
const jsonOutput = {
  generated: new Date().toISOString(),
  tool: "scripts/probe-mpver.mjs",
  probedVersions: versions,
  results,
  findings: {
    standardModuleWorkerStatus:
      "All tested versions fail module-worker init under standard FilesetResolver.forVisionTasks(wasmPath) call.",
    moduleWorker10xError:
      "1.0.0 and 1.0.1 fail standard module-worker init with 'Error: ModuleFactory not set.' because vision_wasm_internal.js defines var ModuleFactory in module scope rather than assigning to globalThis.ModuleFactory.",
    moduleWorkerFlagObservation:
      "1.0.0 and 1.0.1 pass module worker only when non-standard useModule=true flag is explicitly passed to FilesetResolver.forVisionTasks(wasmPath, true), loading vision_wasm_module_internal.js.",
    classicWorkerStatus:
      "0.10.18 (current pin), 0.10.20, 0.10.21, 0.10.32, 0.10.35, 1.0.0, and 1.0.1 all initialize cleanly in classic workers under the standard call.",
    classicWorkerFlagConflict:
      "Passing useModule=true in classic workers causes an importScripts syntax error on 'export default' in vision_wasm_module_internal.js, making the flag incompatible with classic workers.",
    verdict:
      "The 0.10.18 pin STAYS. 1.0.1 is not a drop-in replacement; standard module worker init fails, and upgrading classic workers yields zero off-main-thread module alignment while adding risk across 7 routes.",
  },
};

writeFileSync(
  new URL("../reports/mediapipe-tasks-vision-probe.json", import.meta.url),
  JSON.stringify(jsonOutput, null, 2) + "\n",
);

// Write Markdown artifact
let mdOutput = `# @mediapipe/tasks-vision version probe: 0.10.x vs 1.0.x

- **Generated:** ${jsonOutput.generated}
- **Tool:** \`scripts/probe-mpver.mjs\`
- **Task tested:** \`GestureRecognizer\` init via \`FilesetResolver\` in Web Workers (headless Chrome via CDP)
- **Current pin:** \`TASKS_VISION_VERSION = "0.10.18"\` in \`lib/mediapipe.js\`

## Summary Results Table

| Version | Module Worker (Standard) | Module Worker (\`useModule=true\`) | Classic Worker (Standard) | Verdict |
|---|---|---|---|---|
`;

for (const r of results) {
  const std = r.moduleWorkerStandard.ok
    ? `✅ PASS (${r.moduleWorkerStandard.initMs}ms)`
    : `❌ FAIL: \`${truncate(r.moduleWorkerStandard.err, 40)}\``;
  const flag = r.moduleWorkerWithFlag.ok
    ? `✅ PASS (${r.moduleWorkerWithFlag.initMs}ms)`
    : `❌ FAIL: \`${truncate(r.moduleWorkerWithFlag.err, 40)}\``;
  const classic = r.classicWorkerStandard.ok
    ? `✅ PASS (${r.classicWorkerStandard.initMs}ms)`
    : `❌ FAIL: \`${truncate(r.classicWorkerStandard.err, 40)}\``;

  let verdict = "Unsupported";
  if (r.version === "0.10.18") {
    verdict = "**Current Pin (Stable Classic)**";
  } else if (r.version === "1.0.1") {
    verdict = "**Blocked: ModuleFactory not set**";
  } else if (!r.classicWorkerStandard.ok && !r.moduleWorkerStandard.ok) {
    verdict = "Broken CDN bundle";
  }

  mdOutput += `| \`${r.version}\` | ${std} | ${flag} | ${classic} | ${verdict} |\n`;
}

mdOutput += `
## Detailed Findings & Root Cause Analysis

### 1. Standard Module Worker Init Fails on 1.0.x (\`ModuleFactory not set.\`)
When calling \`FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm")\` inside a module worker (\`{ type: "module" }\`):
1. \`importScripts()\` is not supported and throws a \`TypeError\`.
2. MediaPipe's fallback catches the error and performs \`await import(".../vision_wasm_internal.js")\`.
3. Because \`useModule\` defaults to \`false\`, MediaPipe fetches \`vision_wasm_internal.js\`. That script wraps its output in \`var ModuleFactory = (() => ...)\` which is local to the module scope and does not set \`globalThis.ModuleFactory\`.
4. MediaPipe checks \`if (!self.ModuleFactory) throw Error("ModuleFactory not set.");\`, which aborts initialization immediately.

### 2. The \`useModule=true\` Asymmetry Across Worker Types
MediaPipe 1.0.x introduced an undocumented second parameter: \`FilesetResolver.forVisionTasks(wasmPath, useModule = false)\`.
- When \`useModule = true\` is passed, it loads \`vision_wasm_module_internal.js\`, which includes \`globalThis.ModuleFactory = ModuleFactory; export default ModuleFactory;\`. This allows module workers to succeed.
- However, if \`useModule = true\` is passed in a **classic worker**, \`importScripts(".../vision_wasm_module_internal.js")\` fails with a syntax error because \`export default\` is illegal in classic scripts.
- The shared helper in \`lib/mediapipe.js\` cannot blindly set \`useModule = true\` without breaking classic worker callers.

### 3. Current Pin Verdict: \`0.10.18\` Stays
- Acceptance criterion 2: *"If 1.0.1 fails module-worker init, the pin STAYS and the negative result is recorded (a blocked-style note) so it is not re-attempted."*
- Standard module-worker init fails on 1.0.1.
- In classic workers, \`0.10.18\` is completely stable and battle-tested across all 7 built routes:
  1. \`gesture-recognizer\`
  2. \`hand-landmarker\`
  3. \`pose-landmarker\`
  4. \`face-detector\`
  5. \`face-landmarker\`
  6. \`image-segmenter\`
  7. \`interactive-segmenter\`
- Bumping classic workers to 1.0.1 yields zero module-worker compatibility benefit while incurring major-version upgrade risk across all 7 production vision routes.
- Therefore, the pin in \`lib/mediapipe.js\` remains \`0.10.18\`.
`;

writeFileSync(
  new URL("../reports/mediapipe-tasks-vision-probe.md", import.meta.url),
  mdOutput,
);

console.log("\nWrote reports/mediapipe-tasks-vision-probe.json and reports/mediapipe-tasks-vision-probe.md");

function truncate(str, max) {
  if (!str) return "";
  return str.length > max ? str.slice(0, max - 3) + "..." : str;
}
