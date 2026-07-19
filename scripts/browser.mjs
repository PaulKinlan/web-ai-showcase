#!/usr/bin/env node
// Shared headless-Chrome harness for the conformance runner + responsive matrix check.
//
// Zero external deps: a tiny static file server (serves the repo under the GitHub-Pages base path),
// Chrome launched headless with a FRESH profile (so every model is cache-absent ⇒ the shared auto-init
// loader shows a Download button and NEVER auto-downloads a large model — deterministic + download-
// free), and a minimal CDP client over Node's built-in WebSocket.

import { createServer } from "node:http";
import { execFileSync, spawn } from "node:child_process";
import { readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

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

function findChrome() {
  for (const b of ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"]) {
    try {
      return execFileSync("which", [b]).toString().trim();
    } catch { /* next */ }
  }
  return "google-chrome-stable";
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

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.addEventListener("open", () => resolve(ws));
    ws.addEventListener("error", (e) => reject(new Error("ws error: " + (e.message || url))));
  });
}

export async function launchChrome() {
  const userDataDir = join(repoRoot, ".conformance-chrome-profile");
  try {
    rmSync(userDataDir, { recursive: true, force: true });
  } catch { /* ignore */ }
  const proc = spawn(findChrome(), [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    "--remote-debugging-port=0",
    `--user-data-dir=${userDataDir}`,
    "about:blank",
  ], { stdio: ["ignore", "ignore", "ignore"] });
  const portFile = join(userDataDir, "DevToolsActivePort");
  let wsUrl = null;
  for (let i = 0; i < 100 && !wsUrl; i++) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      const [port, path] = readFileSync(portFile, "utf8").trim().split("\n");
      if (port && path) wsUrl = `ws://127.0.0.1:${port}${path}`;
    } catch { /* not ready */ }
  }
  if (!wsUrl) throw new Error("Chrome did not expose a DevTools endpoint");
  const ws = await connect(wsUrl);
  return {
    proc,
    ws,
    userDataDir,
    kill() {
      try {
        proc.kill("SIGKILL");
      } catch { /* ignore */ }
      try {
        rmSync(userDataDir, { recursive: true, force: true });
      } catch { /* ignore */ }
    },
  };
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
