#!/usr/bin/env node
// Shared headless-Chrome harness for the conformance runner + responsive matrix check.
//
// Zero external deps: a tiny static file server (serves the repo under the GitHub-Pages base path),
// Chrome launched headless with a FRESH profile (so every model is cache-absent ⇒ the shared auto-init
// loader shows a Download button and NEVER auto-downloads a large model — deterministic + download-
// free), and a minimal CDP client over Node's built-in WebSocket.

import { createServer } from "node:http";
import { execFileSync, spawn } from "node:child_process";
import { accessSync, constants, existsSync, readdirSync, readFileSync, realpathSync, rmSync, statfsSync, statSync, writeFileSync } from "node:fs";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { constants as osConstants, loadavg, tmpdir } from "node:os";

export const repoRoot = fileURLToPath(new URL("..", import.meta.url));
export const BASE = "/web-ai-showcase/";
export const DESKTOP = { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false };
export const MOBILE = { width: 360, height: 740, deviceScaleFactor: 3, mobile: true };

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".wasm": "application/wasm",
};

export function startServer() {
  const root = realpathSync(repoRoot);
  const insideRoot = (path) => path === root || path.startsWith(root + sep);
  const server = createServer((req, res) => {
    try {
      let p = decodeURIComponent(req.url.split("?")[0]);
      if (p.includes("\0")) throw new Error("Invalid static path");
      if (p.startsWith(BASE)) p = p.slice(BASE.length - 1);
      let fsPath = resolve(root, p.replace(/^\/+/, ""));
      if (!insideRoot(fsPath)) throw new Error("Static path escapes repo");
      try {
        if (statSync(fsPath).isDirectory()) fsPath = join(fsPath, "index.html");
      } catch { /* 404 below */ }
      // A symlink within the repo must not make the final file escape it either.
      fsPath = realpathSync(fsPath);
      if (!insideRoot(fsPath)) throw new Error("Static path escapes repo");
      const body = readFileSync(fsPath);
      res.writeHead(200, { "content-type": MIME[extname(fsPath)] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

let warnedAboutRejectedChromeBin = null;

function isRunnableFile(path) {
  try {
    return existsSync(path) && statSync(path).isFile() && (accessSync(path, constants.X_OK), true);
  } catch {
    return false;
  }
}

function findChromeOnPath() {
  for (const b of ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"]) {
    try {
      return execFileSync("which", [b]).toString().trim();
    } catch { /* next */ }
  }
  return null;
}

/**
 * Resolve a browser binary. Returns { binary, rejected }: `binary` is the path to use (null when there
 * is none) and `rejected` names a CHROME_BIN that was ignored, so callers can say which value was bad.
 *
 * An explicit CHROME_BIN wins, so a CI step can declare the dependency rather than hope — but only when
 * it points at a runnable file. Trusting it on non-emptiness alone (the cj5 shape) meant a stale value
 * made chromeAvailable() report true, so the honest skip never fired and the suite died in the 240s
 * spawn-ENOENT retry storm that cj5 removed (web-ai-showcase-fdy). A rejected value falls through to
 * the PATH search rather than failing outright, so a machine that does have Chrome keeps working.
 */
export function resolveChromeBinary() {
  const declared = (process.env.CHROME_BIN || "").trim();
  if (declared) {
    if (isRunnableFile(declared)) return { binary: declared, rejected: null };
    if (warnedAboutRejectedChromeBin !== declared) {
      warnedAboutRejectedChromeBin = declared;
      console.warn(
        `CHROME_BIN=${declared} is not an executable file; ignoring it and searching PATH.`,
      );
    }
    return { binary: findChromeOnPath(), rejected: declared };
  }
  return { binary: findChromeOnPath(), rejected: null };
}

function findChrome() {
  // Previously this returned the literal "google-chrome-stable", so a machine without Chrome produced
  // spawn ENOENT inside a retry loop (4 attempts per call, 5 tests → a multi-minute red that reads like
  // a test regression). Return null and let callers report a missing dependency.
  return resolveChromeBinary().binary;
}

/** Whether a Chrome/Chromium binary is resolvable — lets browser-driven tests skip honestly. */
export function chromeAvailable() {
  return findChrome() !== null;
}

export const EVALUATE_RETRY_ENV = "CDP_EVALUATE_RETRIES";

/**
 * How many times a timed-out `Runtime.evaluate` is retried (web-ai-showcase-50s).
 *
 * Deep acceptance suites drive multi-hundred-MB WASM stages while other lanes run their own browsers;
 * under that shared-box load the renderer can stop answering for minutes and one `CDP timeout` used to
 * abort a 20-minute run that had already collected all its route evidence. The retry is opt-in through
 * the environment so nothing changes for callers that do not ask for it, and it is bounded so a
 * genuinely hung page still fails.
 */
export function evaluateRetryCount(env = process.env) {
  const raw = env?.[EVALUATE_RETRY_ENV];
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(Math.floor(n), 10);
}

/** True only for the shape a loaded box produces: an evaluate that outlived its timeout. */
export function isTransientEvaluateTimeout(method, error) {
  return method === "Runtime.evaluate" && /CDP timeout/.test(String(error?.message ?? ""));
}

/**
 * Current box load. A deep suite's heavy re-download/reload steps are the observed stall points, so
 * callers can warn (or refuse) before burning an hour on a doomed run (web-ai-showcase-50s).
 */
export function loadAverage() {
  const [one, five, fifteen] = loadavg();
  return { one, five, fifteen };
}

/** A one-line warning when the box is too busy, or null when it is quiet enough. */
export function loadWarning(threshold = 30, load = loadAverage()) {
  if (!(load.one > threshold)) return null;
  return `box load average is ${load.one.toFixed(1)} (threshold ${threshold}) — deep acceptance suites stall under shared-box load; prefer a quieter window or expect retries`;
}

export class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
      } else if (msg.method) {
        for (const l of this.listeners) l(msg);
      }
    });
  }
  send(method, params = {}, sessionId, timeoutMs = 15000) {
    const attempts = evaluateRetryCount() + 1;
    const attempt = (n) =>
      this.#sendOnce(method, params, sessionId, timeoutMs).catch((error) => {
        if (n + 1 >= attempts || !isTransientEvaluateTimeout(method, error)) throw error;
        console.log(
          `  [cdp evaluate retry ${n + 1}/${attempts - 1}] ${String(error.message).slice(0, 90)}`,
        );
        return attempt(n + 1);
      });
    return attempt(0);
  }
  #sendOnce(method, params = {}, sessionId, timeoutMs = 15000) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout after ${timeoutMs}ms: ${method}`));
        }
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  on(fn) {
    this.listeners.push(fn);
  }
}

function connectOnce(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.addEventListener("open", () => resolve(ws));
    ws.addEventListener("error", (e) => reject(new Error("ws error: " + (e.message || url))));
  });
}

// The DevTools endpoint can briefly refuse the WS upgrade right after the port file appears (a
// startup race, worse under IO/memory pressure). Retry a few times with a short settle delay.
async function connect(url, attempts = 6) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await connectOnce(url);
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 400));
    }
  }
  throw lastErr;
}

const detachedProcessGroup = process.platform !== "win32";

function signalProcessTree(proc) {
  try {
    if (detachedProcessGroup) process.kill(-proc.pid, "SIGKILL");
    else proc.kill("SIGKILL");
  } catch { /* already stopped */ }
}

async function stopProcessTree(proc) {
  signalProcessTree(proc);
  if (proc.exitCode !== null || proc.signalCode !== null) return;
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 5_000);
    proc.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

export const HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS = [
  "--disable-breakpad",
];

export function getChromeLaunchArgs({ userDataDir, extraArgs = [], webgpu = false } = {}) {
  const wantWebGPU = webgpu || extraArgs.some((a) => typeof a === "string" && (a.includes("webgpu") || a.includes("vulkan")));
  const gpuArgs = wantWebGPU
    ? ["--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan"]
    : ["--disable-gpu"];
  return [
    "--headless=new",
    "--no-sandbox",
    ...gpuArgs,
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    "--remote-debugging-port=0",
    // Suppress Chrome crashpad handler daemon processes (web-ai-showcase-9z1).
    // In headless test runs, chrome_crashpad_handler detaches into its own process group
    // reparented to PID 1, escaping signalProcessTree group kills and leaking background daemons.
    // Chrome 154 recognizes --disable-breakpad; --disable-crashpad-for-testing causes
    // net::ERR_ABORTED on renderer navigations in Chrome 154 (web-ai-showcase-c3h).
    ...HEADLESS_CHROME_CRASH_SUPPRESSION_FLAGS,
    // Chrome 111+ rejects DevTools WebSocket upgrades unless the connecting origin is allow-listed.
    // Without this the CDP client's WS handshake is closed immediately ("ws error"). Harmless on older
    // Chrome. Required for the harness to run on modern Chrome.
    "--remote-allow-origins=*",
    // CDP-synthesised clicks are not user gestures, so AudioContext.resume() would hang forever when a
    // validator exercises a real play button. Allow autoplay in the TEST browser only (shipped pages
    // still resume on genuine user clicks). Standard practice (Puppeteer/Playwright default).
    "--autoplay-policy=no-user-gesture-required",
    ...extraArgs.filter((a) => !gpuArgs.includes(a)),
    ...(userDataDir ? [`--user-data-dir=${userDataDir}`] : []),
    "about:blank",
  ];
}

async function spawnChromeOnce(userDataDir, resetProfile, extraArgs = [], webgpu = false) {
  if (resetProfile) {
    try {
      rmSync(userDataDir, { recursive: true, force: true });
    } catch { /* ignore */ }
  } else {
    // These are process-lifetime artifacts, not cache data. A SIGKILLed previous cell cannot clean
    // them itself, and a stale DevTools port must never be mistaken for the fresh process's port.
    for (
      const name of ["DevToolsActivePort", "SingletonCookie", "SingletonLock", "SingletonSocket"]
    ) {
      try {
        rmSync(join(userDataDir, name), { force: true });
      } catch { /* ignore */ }
    }
  }
  const args = getChromeLaunchArgs({ userDataDir, extraArgs, webgpu });
  const proc = spawn(findChrome(), args, { detached: detachedProcessGroup, stdio: ["ignore", "ignore", "ignore"] });
  const portFile = join(userDataDir, "DevToolsActivePort");
  let wsUrl = null;
  for (let i = 0; i < 150 && !wsUrl; i++) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      const [port, path] = readFileSync(portFile, "utf8").trim().split("\n");
      if (port && path) wsUrl = `ws://127.0.0.1:${port}${path}`;
    } catch { /* not ready */ }
  }
  if (!wsUrl) {
    await stopProcessTree(proc);
    return null;
  }
  await new Promise((r) => setTimeout(r, 300)); // let the endpoint finish coming up before upgrading
  try {
    const ws = await connect(wsUrl);
    return { proc, ws };
  } catch {
    await stopProcessTree(proc);
    return null;
  }
}

export const activeChromeInstances = new Set();

export function cleanupAllChromeInstances() {
  for (const instance of activeChromeInstances) {
    try {
      instance.kill({ removeProfile: true });
    } catch { /* ignore */ }
  }
  activeChromeInstances.clear();
}

let globalExitHooksRegistered = false;
function registerGlobalExitHooks() {
  if (globalExitHooksRegistered) return;
  globalExitHooksRegistered = true;
  process.on("exit", () => {
    cleanupAllChromeInstances();
  });
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.once(signal, () => {
      cleanupAllChromeInstances();
      const code = 128 + (osConstants.signals[signal] ?? 0);
      process.exit(code);
    });
  }
}

export function createIsolatedProfileDir(prefix = "conformance") {
  return join(
    tmpdir(),
    `webai-chrome-profile-${prefix}-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  );
}

let runLogSeq = 0;
export function getRunLogPath(slug) {
  return join(
    tmpdir(),
    `acceptance-${slug}-${Date.now()}-${process.pid}-${++runLogSeq}.log`,
  );
}

export async function launchChrome(options = {}) {
  registerGlobalExitHooks();
  const userDataDir = options.userDataDir || createIsolatedProfileDir(options.profilePrefix || "conformance");
  const resetProfile = options.resetProfile ?? true;
  const removeProfileOnKill = options.removeProfileOnKill ?? true;
  const webgpu = options.webgpu ?? false;
  const extraArgs = options.extraArgs || [];
  // A missing browser is a dependency error, not a flake: say so immediately instead of retrying a
  // spawn that cannot succeed (web-ai-showcase-cj5).
  const resolution = resolveChromeBinary();
  if (!resolution.binary) {
    // Name a rejected CHROME_BIN: "your CHROME_BIN points at nothing" is a different fix from
    // "install a browser", and the operator is the only one who can tell which they meant.
    const rejected = resolution.rejected
      ? ` CHROME_BIN=${resolution.rejected} was ignored because it is not an executable file.`
      : "";
    throw new Error(
      "No Chrome/Chromium found on PATH (set CHROME_BIN to point at one). Browser-driven tests and " +
        `validators need a browser; this is a missing dependency, not a test regression.${rejected}`,
    );
  }
  // Chrome can intermittently fail to expose its endpoint under IO/memory pressure — retry the whole
  // spawn a few times before giving up so the harness is reliable in constrained sandboxes.
  let started = null;
  for (let attempt = 0; attempt < 4 && !started; attempt++) {
    started = await spawnChromeOnce(userDataDir, resetProfile, extraArgs, webgpu);
    if (!started) await new Promise((r) => setTimeout(r, 500));
  }
  if (!started) throw new Error("Chrome did not expose a DevTools endpoint (after retries)");
  let killPromise = null;
  let instance = null;
  const killStarted = ({ removeProfile = removeProfileOnKill } = {}) => {
    if (instance) activeChromeInstances.delete(instance);
    if (!killPromise) {
      try {
        started.ws.close();
      } catch { /* ignore */ }
      signalProcessTree(started.proc);
      // Keep legacy fire-and-forget callers safe: their profile is removed synchronously even if
      // they call process.exit() without awaiting the returned process-tree completion promise.
      if (removeProfile) {
        try {
          rmSync(userDataDir, { recursive: true, force: true });
        } catch { /* ignore */ }
      }
      killPromise = stopProcessTree(started.proc);
    }
    return killPromise;
  };
  instance = {
    proc: started.proc,
    ws: started.ws,
    userDataDir,
    kill: killStarted,
  };
  activeChromeInstances.add(instance);
  return instance;
}

// Open a fresh page/session; collect console errors + failed network requests during load; navigate;
// settle. Returns { targetId, sessionId, errors, rawErrors, classifiedErrors, netFailures }.
// Transition skips masked by the classifier are self-reported on stderr (classifiedConsoleNotice)
// so a page can never read as console-clean merely because errors were classified away (2zh).
export async function openPage(cdp, url) {
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const errors = [];
  const rawErrors = [];
  const classifiedErrors = [];
  const netFailures = [];

  const recordError = (desc) => {
    rawErrors.push(desc);
    if (isTransitionSkipAbortError(desc)) {
      classifiedErrors.push({ type: "transition-skip", error: desc });
    } else {
      errors.push(desc);
    }
  };

  cdp.on((msg) => {
    if (msg.sessionId !== sessionId) return;
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      const desc = msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
      recordError(desc);
    }
    if (msg.method === "Runtime.exceptionThrown") {
      const desc = msg.params.exceptionDetails?.exception?.description || "exception";
      recordError(desc);
    }
    if (msg.method === "Network.loadingFailed" && !msg.params.canceled) {
      netFailures.push(msg.params.errorText + " " + (msg.params.type || ""));
    }
  });
  await cdp.send("Page.enable", {}, sessionId);
  await cdp.send("Runtime.enable", {}, sessionId);
  await cdp.send("Network.enable", {}, sessionId);
  await cdp.send("Emulation.setDeviceMetricsOverride", DESKTOP, sessionId);
  const loaded = new Promise((resolve) => {
    cdp.on((msg) => {
      if (msg.sessionId === sessionId && msg.method === "Page.loadEventFired") resolve();
    });
  });
  const nav = await cdp.send("Page.navigate", { url }, sessionId);
  if (nav?.errorText) {
    await closePage(cdp, targetId);
    throw new Error(`Page.navigate failed: ${nav.errorText} (${url})`);
  }
  await Promise.race([loaded, new Promise((r) => setTimeout(r, 8000))]);
  await new Promise((r) => setTimeout(r, 1500)); // settle: loader auto-init resolves to absent state
  const notice = classifiedConsoleNotice({ errors, classifiedErrors, netFailures });
  if (notice) console.warn(notice); // masked skips are never silent (web-ai-showcase-2zh)
  return { targetId, sessionId, errors, rawErrors, classifiedErrors, netFailures };
}

export async function closePage(cdp, targetId) {
  try {
    await cdp.send("Target.closeTarget", { targetId });
  } catch { /* ignore */ }
}

export async function evalBool(cdp, sessionId, expr) {
  const wrapped = `(async()=>{try{return !!(${expr});}catch(e){return false;}})()`;
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: wrapped,
    awaitPromise: true,
    returnByValue: true,
  }, sessionId);
  return result?.value === true;
}

export async function evalValue(cdp, sessionId, expr) {
  const wrapped = `(async()=>{try{return (${expr});}catch(e){return null;}})()`;
  const { result } = await cdp.send("Runtime.evaluate", {
    expression: wrapped,
    awaitPromise: true,
    returnByValue: true,
  }, sessionId);
  return result?.value;
}

export async function setViewport(cdp, sessionId, vp) {
  await cdp.send("Emulation.setDeviceMetricsOverride", vp, sessionId);
  await cdp.send("Emulation.setTouchEmulationEnabled", { enabled: vp.mobile }, sessionId);
  await new Promise((r) => setTimeout(r, 250)); // reflow
}

export async function screenshot(cdp, sessionId, file) {
  const { data } = await cdp.send("Page.captureScreenshot", { format: "png" }, sessionId);
  writeFileSync(file, Buffer.from(data, "base64"));
}

export function escapeHtml(s) {
  return String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
  );
}

/**
 * Classify Chrome's uncatchable cross-document view-transition deadline skip (web-ai-showcase-67b, web-ai-showcase-43l).
 *
 * Matches ONLY the exact AbortError / Transition was skipped error produced by Chrome's C++ navigation engine
 * when an incoming document misses its deadline during cross-document navigation.
 * Never matches arbitrary AbortErrors (e.g. fetch abort, user abort) or general page exceptions.
 */
export function isTransitionSkipAbortError(err) {
  if (err == null) return false;
  const name = typeof err === "object" ? String(err.name ?? "") : "";
  const message = typeof err === "object" ? String(err.message ?? "") : "";
  const desc = typeof err === "object" ? String(err.description ?? "") : "";
  const fullText = typeof err === "string" ? err : `${name} ${message} ${desc}`;
  const hasAbortType = /\b(AbortError|DOMException)\b/.test(fullText);
  const hasSkipMessage = /\bTransition was skipped\b/.test(fullText);
  return hasAbortType && hasSkipMessage;
}

/**
 * Filter a list of console errors or exceptions, classifying known uncatchable browser-engine events
 * while preserving all real page/runtime errors (web-ai-showcase-43l).
 */
export function filterConsoleErrors(errors) {
  if (!Array.isArray(errors)) return [];
  return errors.filter((e) => !isTransitionSkipAbortError(e));
}

/**
 * Masked console errors must never be silent (web-ai-showcase-2zh).
 *
 * The classifier above intentionally drops a real Chrome error shape. A run that reports only
 * {errors, network} cannot distinguish "console was clean" from "console was clean because N errors
 * were classified away", and a storm looks identical to a clean run. Every page therefore
 * self-reports its masked count, and more than this many masked errors on one page load escalates
 * from an informational line to an explicit WARNING. Ten sits comfortably above the one or two
 * skips a normal cross-document navigation produces, and well below a real storm's dozens.
 */
export const CLASSIFIED_WARN_THRESHOLD = 10;

/**
 * Runtime-neutral console/network snapshot for PASS lines: a validator can print this shape
 * verbatim (`JSON.stringify(consoleSummary(page))`) so `classified` is visible next to
 * `errors`/`network` instead of being dropped.
 */
export function consoleSummary(page) {
  return {
    errors: page?.errors ?? [],
    classified: page?.classifiedErrors?.length ?? 0,
    network: page?.netFailures ?? [],
  };
}

/**
 * One stderr line for a page whose console was only clean because errors were masked, or null when
 * nothing was masked. Below the threshold it is an informational summary; above it, a WARNING that
 * names the first masked error so a storm is actionable (web-ai-showcase-2zh).
 */
export function classifiedConsoleNotice(page) {
  const summary = consoleSummary(page);
  if (summary.classified === 0) return null;
  const detail = JSON.stringify(summary);
  if (summary.classified > CLASSIFIED_WARN_THRESHOLD) {
    const first = String(page?.classifiedErrors?.[0]?.error ?? "").slice(0, 160);
    return `WARNING: ${detail} — ${summary.classified} console errors were masked as ` +
      `cross-document transition skips (threshold ${CLASSIFIED_WARN_THRESHOLD}); ` +
      `first masked: ${first}`;
  }
  return `[console] ${detail}`;
}

/**
 * Format a truthful acceptance summary string.
 *
 * Never prints "N/N checks passed" unless total === expectedChecks and results match expectedCells.
 * If checks or cells are incomplete or truncated, prints "REACHED N of EXPECTED M checks — INCOMPLETE"
 * so a partial or stalled run cannot masquerade as complete (web-ai-showcase-5m1).
 */
export function formatAcceptanceSummary({
  passed,
  total,
  expectedChecks,
  results = [],
  expectedCells,
}) {
  const checksPassed = passed === total;
  const checksComplete = expectedChecks == null || total === expectedChecks;
  const cellsComplete = expectedCells == null || (results && results.length === expectedCells);
  const allCellsPassed = !results || results.every((r) => r.pass);

  const isComplete = checksPassed && checksComplete && cellsComplete && allCellsPassed;

  if (isComplete) {
    const checksPart = `${passed}/${expectedChecks ?? total} checks passed`;
    const cellsPart = results && expectedCells != null
      ? ` across ${results.length}/${expectedCells} route cells`
      : "";
    return {
      ok: true,
      message: `${checksPart}${cellsPart}`,
    };
  }

  let message;
  if (expectedChecks != null && total !== expectedChecks) {
    const cellsMsg = expectedCells != null && results && results.length !== expectedCells
      ? ` across ${results.length} of EXPECTED ${expectedCells} route cells`
      : "";
    message = `REACHED ${passed} of EXPECTED ${expectedChecks} checks (${total} attempted)${cellsMsg} — INCOMPLETE`;
  } else if (!checksPassed) {
    message = `${passed}/${total} checks passed (${total - passed} failed) — FAILED`;
  } else if (expectedCells != null && results && results.length !== expectedCells) {
    message = `${passed}/${total} checks passed across only ${results.length} of EXPECTED ${expectedCells} route cells — INCOMPLETE`;
  } else {
    message = `${passed}/${total} checks passed (${results.filter((r) => !r.pass).length} route cells failed) — FAILED`;
  }

  return {
    ok: false,
    message,
  };
}

export function printAcceptanceSummary(opts) {
  const summary = formatAcceptanceSummary(opts);
  console.log(`\n${summary.message}`);
  return summary.ok;
}

/**
 * Capture current git HEAD commit hash.
 */
export function captureHeadCommit(cwd = repoRoot) {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

/**
 * Verify that git HEAD has not moved during an acceptance run.
 * Prevents writing an acceptance-run record that cites a commit whose tree was never tested (web-ai-showcase-5m1).
 */
export function assertHeadUnchanged(startCommit, cwd = repoRoot) {
  const endCommit = captureHeadCommit(cwd);
  if (!startCommit || !endCommit) return true;
  if (startCommit !== endCommit) {
    throw new Error(
      `HEAD moved during acceptance run: started at ${startCommit.slice(0, 7)}, ended at ${endCommit.slice(0, 7)} — refusing to write stale run record`,
    );
  }
  return true;
}

/**
 * Safely write an acceptance run record to disk, verifying git HEAD has not moved.
 * Catches HEAD-moved errors and prints a clean explanatory refusal without a raw Node stack trace (web-ai-showcase-4l8).
 * Returns true if written, false if refused.
 */
export function writeAcceptanceRunRecord({
  runRecordPath,
  startCommit,
  results,
  exitCode = 0,
  cwd = repoRoot,
}) {
  try {
    assertHeadUnchanged(startCommit, cwd);
  } catch (err) {
    console.error(`\nREFUSAL: ${err.message}`);
    return false;
  }
  const runRecord = {
    commit: startCommit,
    ranAt: new Date().toISOString(),
    exitCode,
    results,
  };
  writeFileSync(runRecordPath, JSON.stringify(runRecord, null, 2) + "\n", "utf8");
  console.log(`WROTE ${runRecordPath} for commit ${startCommit}`);
  return true;
}

/**
 * Write a FAILING diagnostic run record when a suite aborts before writing its own (web-ai-showcase-50s).
 *
 * A 20-minute run that collected every route check and then hit a renderer stall used to leave NO
 * artifact and dirty tracked screenshots. This keeps the failure point and the checks that did pass:
 * the record carries `exitCode: 1` and `aborted: true`, so check-portfolio-acceptance still fails on
 * the family until a clean run writes its own record — the evidence simply survives.
 *
 * Refuses (returns false) when HEAD moved during the run, exactly like writeAcceptanceRunRecord.
 */
export function writeAbortedAcceptanceRun({
  runRecordPath,
  startCommit,
  reason,
  assertions = [],
  results = [],
  stages = null,
  matrix = null,
  notes = [],
  cwd = repoRoot,
}) {
  try {
    assertHeadUnchanged(startCommit, cwd);
  } catch (err) {
    console.error(`\nREFUSAL: ${err.message}`);
    return false;
  }
  const failed = assertions.filter((a) => a.state !== "pass").length;
  const record = {
    schemaVersion: 1,
    aborted: true,
    commit: startCommit,
    ranAt: new Date().toISOString(),
    exitCode: 1,
    abortReason: String(reason ?? "run aborted").slice(0, 1000),
    summary: {
      checks: assertions.length,
      passed: assertions.length - failed,
      failed,
      cells: results.length,
      cellsPassed: results.filter((r) => r.pass === true).length,
      aborted: true,
    },
    assertions,
    results,
    ...(stages ? { stages } : {}),
    ...(matrix ? { matrix } : {}),
    notes: [
      "DIAGNOSTIC RECORD: the acceptance run aborted before it could write its own record. It exists so",
      "the failure point and the checks that did pass survive (web-ai-showcase-50s). The gate fails on",
      "exitCode 1; re-run the validator for a clean record.",
      ...notes,
    ],
  };
  writeFileSync(runRecordPath, JSON.stringify(record, null, 2) + "\n", "utf8");
  console.log(`WROTE aborted diagnostic record ${runRecordPath} for commit ${startCommit}`);
  return true;
}

/** Format byte count into human-readable representation. */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let val = bytes / 1024;
  let unitIndex = 0;
  while (val >= 1024 && unitIndex < units.length - 1) {
    val /= 1024;
    unitIndex++;
  }
  return `${val.toFixed(1)} ${units[unitIndex]}`;
}

/** Check if a process ID is currently alive. */
export function isPidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === "EPERM";
  }
}

/**
 * Identify whether a directory name matches the pattern of a temporary acceptance
 * browser profile directory.
 */
export function isStaleProfileDirName(name) {
  if (typeof name !== "string") return false;
  if (name.endsWith(".lock") || name.endsWith(".log") || name.endsWith(".json")) return false;
  return name.includes("-acceptance-") || name.startsWith("webai-chrome-profile-");
}

/**
 * Recursively compute total directory size in bytes.
 */
export function getDirectorySize(dirPath) {
  let total = 0;
  try {
    const entries = readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      const full = join(dirPath, entry.name);
      try {
        if (entry.isDirectory()) {
          total += getDirectorySize(full);
        } else if (entry.isFile()) {
          total += statSync(full).size;
        }
      } catch { /* unreadable or unlinked */ }
    }
  } catch { /* directory unreadable */ }
  return total;
}

/**
 * Discover user-data-dir paths currently held by active Chrome processes.
 * Inspects /proc on Linux, falling back to ps or pgrep cross-platform.
 */
export function getActiveChromeUserDataDirs({
  procDir = "/proc",
  exec = execFileSync,
  currentPid = process.pid,
} = {}) {
  const dirs = new Set();
  if (process.platform === "linux" && existsSync(procDir)) {
    try {
      const entries = readdirSync(procDir);
      for (const entry of entries) {
        if (!/^\d+$/.test(entry) || Number(entry) === currentPid) continue;
        try {
          const cmdline = readFileSync(join(procDir, entry, "cmdline"), "utf8");
          for (const arg of cmdline.split("\0")) {
            if (arg.startsWith("--user-data-dir=")) {
              const val = arg.slice("--user-data-dir=".length).trim();
              if (val) dirs.add(resolve(val));
            }
          }
        } catch { /* process exited */ }
      }
      if (dirs.size > 0) return dirs;
    } catch { /* fall back to ps */ }
  }

  // Cross-platform fallback via ps
  try {
    const output = exec("ps", ["-eo", "args"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    for (const line of output.split("\n")) {
      const match = line.match(/--user-data-dir=([^\s]+)/);
      if (match) {
        const val = match[1].trim();
        if (val) dirs.add(resolve(val));
      }
    }
  } catch { /* ps not available */ }

  if (dirs.size === 0) {
    try {
      const output = exec("pgrep", ["-a", "chrome"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      });
      for (const line of output.split("\n")) {
        const match = line.match(/--user-data-dir=([^\s]+)/);
        if (match) {
          const val = match[1].trim();
          if (val) dirs.add(resolve(val));
        }
      }
    } catch { /* pgrep not available */ }
  }

  return dirs;
}

/**
 * Query filesystem capacity for a directory (defaults to tmpdir()).
 */
export function getTmpfsCapacity(targetDir = tmpdir()) {
  try {
    const stats = statfsSync(targetDir);
    const bsize = stats.bsize || 4096;
    const totalBytes = Number(BigInt(bsize) * BigInt(stats.blocks));
    const freeBytes = Number(BigInt(bsize) * BigInt(stats.bavail));
    const usedBytes = totalBytes - freeBytes;
    const percentUsed = totalBytes > 0 ? (usedBytes / totalBytes) * 100 : 0;
    return { totalBytes, freeBytes, usedBytes, percentUsed };
  } catch {
    return null;
  }
}

/**
 * Prune stale acceptance profile directories in tmpdir (web-ai-showcase-msz).
 *
 * Scans for directories matching `-acceptance-` or `webai-chrome-profile-`,
 * verifies they are not held by any live Chrome process (via activeDirs),
 * verifies any encoded PID is not running, and only removes if mtime > 15 minutes old.
 */
export function pruneStaleAcceptanceProfiles({
  dir = tmpdir(),
  maxAgeMs = 15 * 60 * 1000,
  activeDirs = null,
  now = Date.now(),
  unlink = rmSync,
  dryRun = false,
  log = console.log,
  warn = console.warn,
} = {}) {
  const active = activeDirs ?? getActiveChromeUserDataDirs();
  const activeNormalized = new Set([...active].map((d) => resolve(d)));

  const candidates = [];
  try {
    const entries = readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const name = entry.name;
      if (!isStaleProfileDirName(name)) continue;

      const fullPath = join(dir, name);
      const normalizedPath = resolve(fullPath);

      if (activeNormalized.has(normalizedPath)) continue;
      let inUse = false;
      for (const act of activeNormalized) {
        if (act === normalizedPath || act.startsWith(normalizedPath + "/")) {
          inUse = true;
          break;
        }
      }
      if (inUse) continue;

      const pidMatch = name.match(/webai-chrome-profile-[^-]+-(\d+)-/);
      if (pidMatch) {
        const pid = Number(pidMatch[1]);
        if (isPidAlive(pid)) continue;
      }

      try {
        const st = statSync(fullPath);
        const ageMs = now - st.mtimeMs;
        if (ageMs <= maxAgeMs) continue;

        const sizeBytes = getDirectorySize(fullPath);
        candidates.push({ path: fullPath, sizeBytes });
      } catch { /* unlinked or unreadable */ }
    }
  } catch (err) {
    if (typeof warn === "function") {
      warn(`[acceptance-run] failed to scan ${dir} for stale profiles: ${err.message}`);
    }
  }

  let freedBytes = 0;
  let prunedCount = 0;
  for (const { path, sizeBytes } of candidates) {
    try {
      if (!dryRun) {
        unlink(path, { recursive: true, force: true });
      }
      freedBytes += sizeBytes;
      prunedCount++;
    } catch (err) {
      if (typeof warn === "function") {
        warn(`[acceptance-run] failed to remove stale profile ${path}: ${err.message}`);
      }
    }
  }

  const capacity = getTmpfsCapacity(dir);
  const warnedHighUsage = capacity !== null && capacity.percentUsed > 80;

  if (typeof log === "function") {
    const capInfo = capacity
      ? ` · ${dir} ${formatBytes(capacity.freeBytes)} free of ${formatBytes(capacity.totalBytes)} (${capacity.percentUsed.toFixed(1)}% used)`
      : "";
    if (prunedCount > 0) {
      log(`[acceptance-run] pruned ${prunedCount} stale profile(s) freeing ${formatBytes(freedBytes)}${capInfo}`);
    } else {
      log(`[acceptance-run] no stale profile directories to prune${capInfo}`);
    }
  }

  if (warnedHighUsage && typeof warn === "function") {
    warn(
      `[acceptance-run] WARNING: ${dir} is ${capacity.percentUsed.toFixed(1)}% full (>80%). Running low on tmpfs space may cause Chrome/WASM to crash silently.`,
    );
  }

  return { prunedCount, freedBytes, capacity, warnedHighUsage };
}
