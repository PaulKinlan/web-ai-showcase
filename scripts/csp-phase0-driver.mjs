// Phase 0 CSP measurement driver (web-ai-showcase-ega) — LOCAL ONLY, Report-Only, no enforcement.
//
// Starts `deno run --allow-net --allow-env server.ts` with CSP_REPORT_ONLY=1 (the dev-flagged
// Report-Only header + ephemeral sanitized stdout sink added in server.ts), then drives ONE
// headless browser through representative routes, really running inference where the page offers
// it, and collects: (a) server-side CSP-REPORT sink lines, (b) browser Log/console violations,
// (c) every external origin actually contacted, (d) screenshots, (e) whether inference completed
// (headless Chrome here has no WebGPU — SwiftShader/WASM path).
//
// Usage: OUTDIR=/tmp/klj-evidence/ega-phase0 node scripts/csp-phase0-driver.mjs
// Run through fleet-gate (bounded); it kills the browser and the Deno server in `finally`.
import { spawn } from "node:child_process";
import { mkdirSync, openSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  BASE,
  CDP,
  chromeAvailable,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  screenshot,
  setViewport,
} from "./browser.mjs";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const OUT = process.env.OUTDIR || "/tmp/klj-evidence/ega-phase0";
const PORT = Number(process.env.PORT || 8000);
const ORIGIN = `http://127.0.0.1:${PORT}`;
mkdirSync(OUT, { recursive: true });

if (!chromeAvailable()) {
  console.error("no Chrome/Chromium resolvable — cannot measure (set CHROME_BIN)");
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 1. Start the dev-flagged Deno proxy; stdout (with CSP-REPORT lines) goes to a file ──────────
const serverLogPath = join(OUT, "server-stdout.log");
const serverLogFd = openSync(serverLogPath, "w");
const deno = spawn("deno", ["run", "--allow-net", "--allow-env", "server.ts"], {
  cwd: repoRoot,
  env: { ...process.env, CSP_REPORT_ONLY: "1" },
  stdout: serverLogFd,
  stderr: serverLogFd,
});
let serverUp = false;
for (let i = 0; i < 60 && !serverUp; i++) {
  try {
    const r = await fetch(`${ORIGIN}/`);
    serverUp = r.status === 200 && r.headers.get("content-security-policy-report-only") != null;
    if (!serverUp && r.status === 200) console.warn("server up but NO Report-Only header — flag not honored");
  } catch { /* not yet */ }
  if (!serverUp) await sleep(500);
}
if (!serverUp) {
  console.error("Deno server did not come up with the Report-Only header (see " + serverLogPath + ")");
  deno.kill("SIGTERM");
  process.exit(2);
}
console.log(`server up on ${ORIGIN} with CSP Report-Only (dev flag)`);

// ── 2. One browser, all routes ─────────────────────────────────────────────────────────────────
const ROUTES = [
  { path: "/", name: "home", runInference: false, timeoutMs: 30_000, mobileShot: true },
  { path: "/models/yolos-detection/", name: "yolos-detection-transformers", runInference: true, timeoutMs: 420_000, mobileShot: false },
  { path: "/models/face-detector/", name: "face-detector-mediapipe", runInference: true, timeoutMs: 240_000, mobileShot: true },
  { path: "/models/silero-vad/", name: "silero-vad-ort", runInference: true, timeoutMs: 300_000, mobileShot: false },
];

const chrome = await launchChrome({ profilePrefix: "ega-phase0" });
const cdp = new CDP(chrome.ws);
let currentRoute = "(startup)";
const summary = { startedAt: new Date().toISOString(), policyHeader: null, routes: [], externalOrigins: {}, violationsBrowser: [], webgpu: null, sinkPositiveControl: null };

try {
  // Positive control: prove the report path itself works, so a zero-violation run can never be
  // misread as a dead sink. Marked synthetic and excluded from the violation tally.
  try {
    const ctl = await fetch(`${ORIGIN}/__csp-report`, {
      method: "POST",
      headers: { "content-type": "application/csp-report" },
      body: JSON.stringify({ "csp-report": { "blocked-uri": "https://synthetic-control.invalid/x?Signature=must-be-stripped", "violated-directive": "default-src", "synthetic-control": true } }),
    });
    summary.sinkPositiveControl = { status: ctl.status };
    console.log(`sink positive control: HTTP ${ctl.status} (expect 204)`);
  } catch (e) {
    summary.sinkPositiveControl = { error: String(e.message || e) };
    console.log(`sink positive control FAILED: ${summary.sinkPositiveControl.error}`);
  }

  // Record the exact served policy header (evidence).
  const probe = await fetch(`${ORIGIN}/`);
  summary.policyHeader = probe.headers.get("content-security-policy-report-only");

  const externalHits = new Map(); // origin -> count
  cdp.on((msg) => {
    if (msg.method === "Network.responseReceived") {
      try {
        const u = new URL(msg.params.response.url);
        if (u.hostname !== "127.0.0.1") {
          const key = u.origin;
          externalHits.set(key, (externalHits.get(key) || 0) + 1);
          summary.externalOrigins[key] ??= { count: 0, routes: [] };
          summary.externalOrigins[key].count++;
          if (!summary.externalOrigins[key].routes.includes(currentRoute)) summary.externalOrigins[key].routes.push(currentRoute);
        }
      } catch { /* ignore */ }
    }
    if (msg.method === "Log.entryAdded") {
      const e = msg.params.entry;
      if ((e.source === "security" || e.source === "violation" || /Content Security Policy/i.test(e.text || ""))) {
        summary.violationsBrowser.push({ route: currentRoute, text: (e.text || "").slice(0, 500), url: e.url || null });
      }
    }
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      const t = msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
      if (/Refused to|Content Security Policy/i.test(t)) {
        summary.violationsBrowser.push({ route: currentRoute, text: t.slice(0, 500), url: null });
      }
    }
  });

  for (const route of ROUTES) {
    currentRoute = route.name;
    console.log(`\n── route: ${route.path} (${route.name}) ──`);
    const { targetId, sessionId, errors } = await openPage(cdp, `${ORIGIN}${BASE}${route.path.replace(/^\//, "")}`);
    const ev = async (expression, awaitPromise = false) => {
      const r = await cdp.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise }, sessionId);
      if (r.exceptionDetails) return { __err: r.exceptionDetails.exception?.description || "eval error" };
      return r.result.value;
    };
    await cdp.send("Log.enable", {}, sessionId).catch(() => {});
    setViewport(cdp, sessionId, DESKTOP).catch?.(() => {});
    await cdp.send("Emulation.setDeviceMetricsOverride", DESKTOP, sessionId).catch(() => {});

    if (route.name === "home" && summary.webgpu === null) {
      summary.webgpu = await ev(`!!(navigator.gpu)`);
      console.log(`WebGPU available (SwiftShader headless): ${summary.webgpu}`);
    }

    const result = { route: route.name, path: route.path, consoleErrorsOnLoad: errors.length, inference: null, status: null, screenshots: [] };

    if (route.runInference) {
      // 1) the central model loader gates the download behind a user click — press it.
      const loaderClicked = await ev(`(() => {
        const host = document.getElementById("model-loader");
        if (!host) return "no-loader";
        const btns = [...host.querySelectorAll("button")];
        const b = btns.find(x => /download|continue/i.test(x.textContent));
        if (b) { b.click(); return "clicked:" + b.textContent.trim().slice(0, 40); }
        return "no-button:" + btns.length;
      })()`);
      console.log(`loader: ${JSON.stringify(loaderClicked)}`);
      // 2) wait for #run to become enabled (model + input ready), bounded.
      const deadline = Date.now() + route.timeoutMs;
      let runEnabled = false;
      while (Date.now() < deadline) {
        runEnabled = await ev(`(() => { const b = document.getElementById("run"); return !!b && !b.disabled; })()`);
        if (runEnabled === true) break;
        if (runEnabled && runEnabled.__err) break;
        await sleep(1000);
      }
      // 3) click a sample chip first if nothing is preselected.
      await ev(`(() => { const s = document.getElementById("samples"); if (s) { const c = [...s.children].find(x => x.getAttribute("aria-pressed") === "true") || s.children[0]; c?.click?.(); } return true; })()`);
      await sleep(500);
      runEnabled = await ev(`(() => { const b = document.getElementById("run"); return !!b && !b.disabled; })()`);
      if (runEnabled === true) {
        await ev(`document.getElementById("run").click(); true`);
        console.log("run clicked — waiting for inference…");
        // 4) wait for a terminal-ish state: status text settles and is not a spinner/progress message.
        let last = "";
        while (Date.now() < deadline) {
          await sleep(2000);
          const st = await ev(`(() => { const s = document.getElementById("status"); const r = document.getElementById("readout");
            return JSON.stringify({ status: s && !s.hidden ? s.textContent.trim().slice(0, 160) : null,
              readout: r && !r.hidden ? r.textContent.trim().slice(0, 160) : null,
              stageKids: (document.getElementById("stage")||{children:[]}).children.length }); })()`);
          const parsed = JSON.parse(st || "{}");
          const sig = JSON.stringify(parsed);
          if (sig === last && (parsed.readout || parsed.stageKids > 0 || /done|complete|ms\b|found|detected|speech|silence/i.test(parsed.status || ""))) break;
          last = sig;
        }
        result.inference = JSON.parse(last || "{}");
      } else {
        result.inference = { enabled: false, note: "run never enabled within timeout" };
      }
      result.status = await ev(`(() => { const s = document.getElementById("status"); return s ? { hidden: s.hidden, cls: s.className, text: s.textContent.trim().slice(0, 200) } : null; })()`);
    }

    const shot = join(OUT, `${route.name}-desktop.png`);
    await screenshot(cdp, sessionId, shot);
    result.screenshots.push(shot);
    if (route.mobileShot) {
      await cdp.send("Emulation.setDeviceMetricsOverride", MOBILE, sessionId).catch(() => {});
      await sleep(600);
      const mshot = join(OUT, `${route.name}-mobile.png`);
      await screenshot(cdp, sessionId, mshot);
      result.screenshots.push(mshot);
      await cdp.send("Emulation.setDeviceMetricsOverride", DESKTOP, sessionId).catch(() => {});
    }
    console.log(`route ${route.name}: inference=${JSON.stringify(result.inference)?.slice(0, 200)} status=${JSON.stringify(result.status)?.slice(0, 160)}`);
    summary.routes.push(result);
    await closePage(cdp, targetId).catch(() => {});
  }
} finally {
  currentRoute = "(teardown)";
  await chrome.kill?.().catch(() => {});
  deno.kill("SIGTERM");
  await sleep(500);
  try { deno.kill("SIGKILL"); } catch { /* already gone */ }
}

// ── 3. Collect the server-side sink lines ───────────────────────────────────────────────────────
const serverLogText = readFileSync(serverLogPath, "utf8");
const sinkLines = serverLogText.split("\n").filter((l) => l.startsWith("CSP-REPORT ")).map((l) => {
  try { return JSON.parse(l.slice("CSP-REPORT ".length)); } catch { return { raw: l.slice(0, 300) }; }
});
const synthetic = sinkLines.filter((r) => r["synthetic-control"] === true || String(r["blocked-uri"] || r.blockedURI || "").includes("synthetic-control.invalid"));
const real = sinkLines.filter((r) => !synthetic.includes(r));
summary.violationsSink = real;
summary.sinkSyntheticControlReceived = synthetic.length;
summary.externalOriginList = [...externalHits.entries()].sort((a, b) => b[1] - a[1]);
summary.finishedAt = new Date().toISOString();

const outPath = join(OUT, "summary.json");
const { writeFileSync } = await import("node:fs");
writeFileSync(outPath, JSON.stringify(summary, null, 1));

console.log("\n=== PHASE 0 SUMMARY ===");
console.log(`policy served: ${summary.policyHeader ? "yes" : "NO"}`);
console.log(`WebGPU in headless (SwiftShader): ${summary.webgpu}`);
console.log(`browser-side violations: ${summary.violationsBrowser.length}`);
console.log(`sink reports: ${real.length} real + ${synthetic.length} synthetic control (control sent: ${JSON.stringify(summary.sinkPositiveControl)})`);
for (const v of summary.violationsBrowser.slice(0, 20)) console.log(`  [browser][${v.route}] ${v.text.slice(0, 200)}`);
for (const v of real.slice(0, 20)) console.log(`  [sink] ${JSON.stringify(v).slice(0, 250)}`);
console.log("external origins contacted:");
for (const [o, n] of summary.externalOriginList) console.log(`  ${n}x ${o}`);
for (const r of summary.routes) console.log(`route ${r.route}: inference=${JSON.stringify(r.inference)?.slice(0, 180)}`);
console.log(`\nfull summary: ${outPath}`);
process.exit(0);
