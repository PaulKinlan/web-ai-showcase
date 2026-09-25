#!/usr/bin/env node
// Shared headless-Chrome harness for the conformance runner + responsive matrix check.
//
// Zero external deps: a tiny static file server (serves the repo under the GitHub-Pages base path),
// Chrome launched headless with a FRESH profile (so every model is cache-absent ⇒ the shared auto-init
// loader shows a Download button and NEVER auto-downloads a large model — deterministic + download-
// free), and a minimal CDP client over Node's built-in WebSocket.

import { createServer } from "node:http";
import { execFileSync, spawn } from "node:child_process";
import { accessSync, constants, existsSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

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
  const server = createServer((req, res) => {
    try {
      let p = decodeURIComponent(req.url.split("?")[0]);
      if (p.startsWith(BASE)) p = p.slice(BASE.length - 1);
      let fsPath = join(repoRoot, p.replace(/^\/+/, ""));
      try {
        if (statSync(fsPath).isDirectory()) fsPath = join(fsPath, "index.html");
      } catch { /* 404 below */ }
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
  const wantWebGPU = webgpu || extraArgs.some((a) => typeof a === "string" && (a.includes("webgpu") || a.includes("vulkan")));
  const gpuArgs = wantWebGPU
    ? ["--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan"]
    : ["--disable-gpu"];
  const proc = spawn(findChrome(), [
    "--headless=new",
    "--no-sandbox",
    ...gpuArgs,
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    "--remote-debugging-port=0",
    // Chrome 111+ rejects DevTools WebSocket upgrades unless the connecting origin is allow-listed.
    // Without this the CDP client's WS handshake is closed immediately ("ws error"). Harmless on older
    // Chrome. Required for the harness to run on modern Chrome.
    "--remote-allow-origins=*",
    // CDP-synthesised clicks are not user gestures, so AudioContext.resume() would hang forever when a
    // validator exercises a real play button. Allow autoplay in the TEST browser only (shipped pages
    // still resume on genuine user clicks). Standard practice (Puppeteer/Playwright default).
    "--autoplay-policy=no-user-gesture-required",
    ...extraArgs.filter((a) => !gpuArgs.includes(a)),
    `--user-data-dir=${userDataDir}`,
    "about:blank",
  ], { detached: detachedProcessGroup, stdio: ["ignore", "ignore", "ignore"] });
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
      process.exit(130);
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
// settle. Returns { targetId, sessionId, errors, netFailures }.
export async function openPage(cdp, url) {
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const errors = [];
  const netFailures = [];
  cdp.on((msg) => {
    if (msg.sessionId !== sessionId) return;
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      errors.push(msg.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
    }
    if (msg.method === "Runtime.exceptionThrown") {
      errors.push(msg.params.exceptionDetails?.exception?.description || "exception");
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
  await cdp.send("Page.navigate", { url }, sessionId);
  await Promise.race([loaded, new Promise((r) => setTimeout(r, 8000))]);
  await new Promise((r) => setTimeout(r, 1500)); // settle: loader auto-init resolves to absent state
  return { targetId, sessionId, errors, netFailures };
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
