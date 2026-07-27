// ai-text-detection end-to-end acceptance validator (real inference in headless Chrome).
// Family routes (all five, desktop + mobile):
//   models/ai-text-detection/              overview (see-inside softmax surface)
//   models/ai-text-detection/basics/       human vs GPT-2 pairs
//   models/ai-text-detection/practical/    batch triage + threshold slider
//   models/ai-text-detection/wild/         adversarial gauntlet (honest failures REPORTED)
//   models/ai-text-detection/multi-model/  GPT-2 generates → detector catches
// Advertised stages (both named + driven for real):
//   stage 1 generator: Xenova/gpt2 (int8, ~280 MB, transformers.js text-generation)
//   stage 2 detector:  onnx-community/roberta-base-openai-detector-ONNX (q8, ~126 MB)
//
// Flow: fresh profile → first-visit Download gate → REAL download with progress → real
// per-route inference at DESKTOP and MOBILE viewports → cached-reload auto-init →
// zero-console-error / zero-failed-network / zero-overflow checks. Exit 0 only when every
// route x viewport pair passes.
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

// Clean up stale harness Chromes/profiles from interrupted runs (they lock the profile dir).
import { execFileSync } from "node:child_process";
import { rmSync } from "node:fs";
try {
  execFileSync("pkill", ["-f", "conformance-chrome-profile"]);
} catch { /* none */ }
try {
  rmSync(new URL("../.conformance-chrome-profile", import.meta.url), {
    recursive: true,
    force: true,
  });
} catch { /* ignore */ }

const DETECTOR_ID = "onnx-community/roberta-base-openai-detector-ONNX";
const GENERATOR_ID = "Xenova/gpt2";

// The exact route matrix this validator drives (literal paths; the acceptance gate greps these).
const ROUTES = {
  overview: "models/ai-text-detection/",
  basics: "models/ai-text-detection/basics/",
  practical: "models/ai-text-detection/practical/",
  wild: "models/ai-text-detection/wild/",
  multimodel: "models/ai-text-detection/multi-model/",
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const url = (route) => `http://127.0.0.1:${port}/web-ai-showcase/${route}`;

let pass = 0, total = 0;
const routeResults = {}; // `${route}@${viewport}` -> { route, viewport, pass }
const rr = (route, viewport) => {
  const key = `${route}@${viewport}`;
  if (!routeResults[key]) routeResults[key] = { route, viewport, pass: true };
  return routeResults[key];
};
const chk = (name, cond, detail, scope) => {
  total++;
  if (cond) pass++;
  if (scope && !cond) scope.pass = false;
  console.log(
    `${cond ? "PASS" : "FAIL"}  ${name}${detail ? " — " + String(detail).slice(0, 220) : ""}`,
  );
};

async function evalIn(sid, expr, timeoutMs = 45000) {
  const wrapped =
    `(async()=>{try{return (${expr});}catch(e){return "ERR:" + (e && e.message || e);}})()`;
  const { result } = await cdp.send(
    "Runtime.evaluate",
    { expression: wrapped, awaitPromise: true, returnByValue: true },
    sid,
    timeoutMs,
  );
  return result?.value;
}

// A single evaluate can transiently stall while the renderer main thread is saturated by a
// multi-hundred-MB download + Cache Storage writes; polling must ride out those stalls and only
// fail at its own deadline.
async function waitFor(sid, expr, timeoutMs, pollMs = 2500, label = "", onPoll) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      if (await evalIn(sid, expr)) return true;
      if (onPoll) await onPoll();
    } catch (e) {
      console.log(
        `  [${label}] poll eval stall (kept polling): ${String(e?.message || e).slice(0, 120)}`,
      );
    }
    await sleep(pollMs);
  }
  throw new Error("TIMEOUT waiting for: " + label);
}

// Download driver with full evidence logging: polls the shared loader state every few seconds,
// prints each sample (state + real progress text), and retries the honest error state up to 3
// attempts (a CDN stall surfaces as loader error + Retry — the validate-manga-ocr pattern).
// Returns { ok, samples } — samples are the evidence that a REAL download with progress happened.
const dlState =
  `(() => { const ml = document.querySelector(".model-loader"); const dl = document.querySelector("model-download-status");
    return JSON.stringify({
      state: ml?.dataset.state || "",
      prog: (dl ? dl.textContent : (ml?.querySelector(".status")?.textContent || "")).replace(/\\s+/g, " ").trim().slice(0, 160) }); })()`;
const clickRetryOrDownload =
  `(() => { const b = [...document.querySelectorAll(".model-loader button")]
      .find((x) => /Retry|Download/i.test(x.textContent));
    if (!b) return false; b.click(); return true; })()`;

async function downloadUntilReady(
  sid,
  readyExpr,
  label,
  attempts = 3,
  perAttemptMs = 8 * 60 * 1000,
) {
  const samples = [];
  for (let attempt = 1; attempt <= attempts; attempt++) {
    if (attempt > 1) {
      console.log(`  [${label}] retry attempt ${attempt}`);
      await evalIn(sid, clickRetryOrDownload);
    }
    const t0 = Date.now();
    while (Date.now() - t0 < perAttemptMs) {
      let st = null;
      try {
        st = JSON.parse(await evalIn(sid, dlState, 45000));
        if (st) {
          samples.push(st);
          console.log(
            `  [${label}] ${Math.round((Date.now() - t0) / 1000)}s ${JSON.stringify(st)}`,
          );
        }
        if (await evalIn(sid, readyExpr, 45000)) return { ok: true, samples };
      } catch (e) {
        console.log(
          `  [${label}] poll eval stall (kept polling): ${String(e?.message || e).slice(0, 120)}`,
        );
      }
      if (st?.state === "error") break; // honest error -> Retry on the next attempt
      await sleep(4000);
    }
  }
  return { ok: false, samples };
}

const loaderState =
  `(() => { const ml = document.querySelector(".model-loader"); if (!ml) return null;
    return JSON.stringify({
      state: ml.dataset.state || "",
      buttons: [...ml.querySelectorAll("button")].map((b) => b.textContent.trim()),
      status: (ml.querySelector(".status")?.textContent || "").trim().slice(0, 160) }); })()`;

const clickDownload = `(() => { const b = [...document.querySelectorAll(".model-loader button")]
      .find((x) => /Download/i.test(x.textContent));
    if (!b) return false; b.click(); return true; })()`;

const verdict = `JSON.stringify({ label: document.querySelector("#vLabel")?.textContent || "",
    conf: document.querySelector("#vConf")?.textContent || "",
    ms: document.querySelector("#rMs")?.textContent || "",
    backend: document.querySelector("#rBackend")?.textContent || "",
    probs: [...document.querySelectorAll(".prob-score")].map((e) => e.textContent) })`;

const overflowOk = `document.documentElement.scrollWidth <= window.innerWidth + 1`;
const pct = (s) => Number.parseFloat(String(s).replace(/[^0-9.]/g, ""));

function pageHygiene(pg, route, viewport, scope) {
  chk(
    `${viewport} ${route}: zero console errors`,
    pg.errors.length === 0,
    pg.errors.slice(0, 3).join(" | "),
    scope,
  );
  chk(
    `${viewport} ${route}: zero failed network requests`,
    pg.netFailures.length === 0,
    pg.netFailures.slice(0, 3).join(" | "),
    scope,
  );
}

// Wait for the shared loader to reach ready on a page where the model is already cached
// (auto-init, NO Download button), then assert the gate is gone.
async function waitAutoInit(sid, readyExpr, route, viewport, scope) {
  await waitFor(sid, readyExpr, 5 * 60 * 1000, 3000, `${route} cached auto-init`);
  const st = JSON.parse(await evalIn(sid, loaderState));
  chk(
    `${viewport} ${route}: cached auto-init (no Download gate)`,
    st && !st.buttons.some((b) => /Download/i.test(b)),
    JSON.stringify(st),
    scope,
  );
}

try {
  // =================================================================================
  // DESKTOP pass — fresh profile: first visit is a real Download gate + real download.
  // =================================================================================

  // ---- 1. OVERVIEW (desktop, first visit) -----------------------------------------
  const ovScope = rr(ROUTES.overview, "desktop");
  const ov = await openPage(cdp, url(ROUTES.overview));
  await setViewport(cdp, ov.sessionId, DESKTOP);
  await sleep(2500);
  {
    const st = JSON.parse(await evalIn(ov.sessionId, loaderState));
    const gated = st && st.buttons.some((b) => /Download/i.test(b));
    const inputDisabled = await evalIn(ov.sessionId, `document.querySelector("#text")?.disabled`);
    chk(
      `desktop overview: first-visit Download gate for ${DETECTOR_ID}`,
      gated && inputDisabled === true,
      JSON.stringify(st),
      ovScope,
    );
  }
  chk(
    "desktop overview: Download click accepted",
    await evalIn(ov.sessionId, clickDownload) === true,
    "",
    ovScope,
  );
  const dl = await downloadUntilReady(
    ov.sessionId,
    `document.querySelector("#text") && !document.querySelector("#text").disabled`,
    "overview download",
  );
  const sawProgress = dl.samples.some((s) =>
    /\d+(\.\d+)?\s*MB/.test(s.prog) && /%|files/i.test(s.prog)
  );
  chk(
    "desktop overview: real download with visible progress (~126 MB q8 detector)",
    dl.ok && sawProgress,
    dl.samples.find((s) => /MB/.test(s.prog))?.prog ||
      JSON.stringify(dl.samples.at(-1) || "(no samples)"),
    ovScope,
  );
  // default human text auto-scored
  await waitFor(
    ov.sessionId,
    `/human-written|machine-written/.test(document.querySelector("#vLabel")?.textContent || "")`,
    120000,
    1000,
    "overview auto-score",
  );
  {
    const v = JSON.parse(await evalIn(ov.sessionId, verdict));
    chk(
      "desktop overview: default human text reads human-written",
      /human-written/.test(v.label) && pct(v.conf) >= 90,
      JSON.stringify(v),
      ovScope,
    );
    chk(
      "desktop overview: backend + latency surfaced (WASM path)",
      v.backend.length > 0 && /ms/.test(v.ms),
      JSON.stringify(v),
      ovScope,
    );
    // see-inside softmax surface: two competing probabilities that sum to ~100%
    const probs = v.probs.map(pct).filter((n) => Number.isFinite(n));
    const sum = probs.reduce((a, b) => a + b, 0);
    chk(
      "desktop overview: see-inside softmax surface (2 probs summing ~100%)",
      probs.length === 2 && sum > 99 && sum < 101,
      JSON.stringify(v.probs),
      ovScope,
    );
  }
  // GPT-2 chip flips the verdict to machine-written
  await evalIn(ov.sessionId, `[...document.querySelectorAll(".chip")][0].click()`);
  await waitFor(
    ov.sessionId,
    `/machine-written/.test(document.querySelector("#vLabel")?.textContent || "")`,
    60000,
    1000,
    "overview GPT-2 chip verdict",
  );
  {
    const v = JSON.parse(await evalIn(ov.sessionId, verdict));
    chk(
      "desktop overview: real GPT-2 output chip reads machine-written",
      pct(v.conf) >= 90,
      JSON.stringify(v),
      ovScope,
    );
  }
  // short-text chip: the known waver/false-positive — assert it is REPORTED, not that it passes
  await evalIn(
    ov.sessionId,
    `[...document.querySelectorAll(".chip")].find((c) => /short text/.test(c.textContent)).click()`,
  );
  await waitFor(
    ov.sessionId,
    `/[0-9]/.test(document.querySelector("#vConf")?.textContent || "")`,
    60000,
    1000,
    "overview short-text chip verdict",
  );
  await sleep(1500);
  {
    const v = JSON.parse(await evalIn(ov.sessionId, verdict));
    chk(
      "desktop overview: short-text waver honestly REPORTED (verdict + conf shown)",
      /written/.test(v.label) && Number.isFinite(pct(v.conf)),
      JSON.stringify(v),
      ovScope,
    );
  }
  chk(
    "desktop overview: no horizontal overflow",
    await evalIn(ov.sessionId, overflowOk) === true,
    "",
    ovScope,
  );

  // ---- cached-reload auto-init (navigate reload; model now in Cache Storage) --------
  await cdp.send("Page.navigate", { url: url(ROUTES.overview) }, ov.sessionId);
  await sleep(4000);
  await waitFor(
    ov.sessionId,
    `document.querySelector("#text") && !document.querySelector("#text").disabled`,
    5 * 60 * 1000,
    3000,
    "overview reload auto-init",
  );
  {
    const st = JSON.parse(await evalIn(ov.sessionId, loaderState));
    chk(
      "desktop overview: cached reload AUTO-INIT (ready, no Download, Clear offered)",
      st && !st.buttons.some((b) => /Download/i.test(b)) &&
        st.buttons.some((b) => /Clear/i.test(b)),
      JSON.stringify(st),
      ovScope,
    );
    await waitFor(
      ov.sessionId,
      `/human-written|machine-written/.test(document.querySelector("#vLabel")?.textContent || "")`,
      120000,
      1000,
      "reload auto-score",
    );
    const v = JSON.parse(await evalIn(ov.sessionId, verdict));
    chk(
      "desktop overview: reload auto-scores default text",
      /written/.test(v.label),
      JSON.stringify(v),
      ovScope,
    );
  }
  pageHygiene(ov, ROUTES.overview, "desktop", ovScope);
  await closePage(cdp, ov.targetId);

  // ---- 2. BASICS (desktop) ---------------------------------------------------------
  const baScope = rr(ROUTES.basics, "desktop");
  const ba = await openPage(cdp, url(ROUTES.basics));
  await setViewport(cdp, ba.sessionId, DESKTOP);
  await waitAutoInit(
    ba.sessionId,
    `document.querySelector("#pairRun") && !document.querySelector("#pairRun").disabled`,
    ROUTES.basics,
    "desktop",
    baScope,
  );
  await evalIn(ba.sessionId, `document.querySelector("#pairRun").click()`);
  await waitFor(
    ba.sessionId,
    `document.querySelectorAll("#pairs .pair").length === 6`,
    180000,
    1000,
    "basics pairs scored",
  );
  {
    const pairs = JSON.parse(
      await evalIn(
        ba.sessionId,
        `JSON.stringify([...document.querySelectorAll("#pairs .pair")].map((p) => ({
          src: p.querySelector(".src").textContent, v: p.querySelector(".v").textContent })))`,
      ),
    );
    const gpt2 = pairs.filter((p) => /GPT-2/.test(p.src));
    const human = pairs.filter((p) => /human/.test(p.src));
    const separated = pairs.length === 6 && gpt2.length === 3 && human.length === 3 &&
      gpt2.every((p) => /machine-written/.test(p.v) && pct(p.v.split("machine").pop()) >= 90) &&
      human.every((p) => /human-written/.test(p.v) && pct(p.v.split("machine").pop()) <= 10);
    chk(
      "desktop basics: 6/6 human vs Xenova/gpt2 pairs separated by the real detector",
      separated,
      JSON.stringify(pairs),
      baScope,
    );
  }
  chk(
    "desktop basics: no horizontal overflow",
    await evalIn(ba.sessionId, overflowOk) === true,
    "",
    baScope,
  );
  pageHygiene(ba, ROUTES.basics, "desktop", baScope);
  await closePage(cdp, ba.targetId);

  // ---- 3. PRACTICAL (desktop) ------------------------------------------------------
  const prScope = rr(ROUTES.practical, "desktop");
  const pr = await openPage(cdp, url(ROUTES.practical));
  await setViewport(cdp, pr.sessionId, DESKTOP);
  await waitAutoInit(
    pr.sessionId,
    `document.querySelector("#run") && !document.querySelector("#run").disabled`,
    ROUTES.practical,
    "desktop",
    prScope,
  );
  await evalIn(pr.sessionId, `document.querySelector("#run").click()`);
  await waitFor(
    pr.sessionId,
    `document.querySelectorAll("#queue .ticket").length >= 6`,
    180000,
    1000,
    "practical triage",
  );
  {
    const t = JSON.parse(
      await evalIn(
        pr.sessionId,
        `JSON.stringify({ flag: document.querySelector("#rFlag").textContent,
          clear: document.querySelector("#rClear").textContent,
          ms: document.querySelector("#rMs").textContent,
          tickets: [...document.querySelectorAll("#queue .ticket")].map((x) =>
            x.querySelector(".route").textContent.replace(/\\s+/g, " ")) })`,
      ),
    );
    chk(
      "desktop practical: real batch triage (>=6 tickets, >=1 flagged, batch latency shown)",
      t.tickets.length >= 6 && Number.parseInt(t.flag, 10) >= 1 && /ms/.test(t.ms),
      JSON.stringify(t),
      prScope,
    );
    // threshold slider -> 80% re-triages live without re-running the model
    await evalIn(
      pr.sessionId,
      `(() => { const s = document.querySelector("#th"); s.value = 80;
        s.dispatchEvent(new Event("input", { bubbles: true })); return s.value; })()`,
    );
    await sleep(800);
    const t2 = JSON.parse(
      await evalIn(
        pr.sessionId,
        `JSON.stringify({ thVal: document.querySelector("#thVal").textContent,
          flag: document.querySelector("#rFlag").textContent })`,
      ),
    );
    chk(
      "desktop practical: threshold slider live re-triage (80%)",
      t2.thVal === "80%" && Number.parseInt(t2.flag, 10) >= 1 &&
        Number.parseInt(t2.flag, 10) <= Number.parseInt(t.flag, 10),
      JSON.stringify(t2),
      prScope,
    );
  }
  chk(
    "desktop practical: no horizontal overflow",
    await evalIn(pr.sessionId, overflowOk) === true,
    "",
    prScope,
  );
  pageHygiene(pr, ROUTES.practical, "desktop", prScope);
  await closePage(cdp, pr.targetId);

  // ---- 4. WILD (desktop) ------------------------------------------------------------
  const wiScope = rr(ROUTES.wild, "desktop");
  const wi = await openPage(cdp, url(ROUTES.wild));
  await setViewport(cdp, wi.sessionId, DESKTOP);
  await waitAutoInit(
    wi.sessionId,
    `document.querySelector("#run") && !document.querySelector("#run").disabled`,
    ROUTES.wild,
    "desktop",
    wiScope,
  );
  await evalIn(wi.sessionId, `document.querySelector("#run").click()`);
  await waitFor(
    wi.sessionId,
    `document.querySelectorAll("#probes .probe").length === 8`,
    240000,
    1500,
    "wild gauntlet",
  );
  {
    const g = JSON.parse(
      await evalIn(
        wi.sessionId,
        `JSON.stringify({ n: document.querySelector("#rN").textContent,
          waver: document.querySelector("#rWaver").textContent,
          ms: document.querySelector("#rMs").textContent,
          probes: [...document.querySelectorAll("#probes .probe")].map((p) => ({
            name: p.querySelector(".name").textContent,
            r: p.querySelector(".route").textContent.replace(/\\s+/g, " "),
            wavers: p.classList.contains("waver") })) })`,
      ),
    );
    const probe = (re) => g.probes.find((p) => re.test(p.name));
    const machinePct = (p) => pct(p.r.split("machine").pop());
    chk(
      "desktop wild: 8-probe gauntlet ran (real batch, latency shown)",
      g.n === "8" && g.probes.length === 8 && /ms/.test(g.ms),
      JSON.stringify({ n: g.n, waver: g.waver, ms: g.ms }),
      wiScope,
    );
    const control = probe(/Control/);
    chk(
      "desktop wild: GPT-2 control caught (reads machine >= 90%)",
      control && /reads machine/.test(control.r) && machinePct(control) >= 90,
      JSON.stringify(control),
      wiScope,
    );
    // EXPECTED honest failures — assert they are REPORTED, not that they pass as detections.
    const aside = probe(/Tiny human aside/);
    chk(
      "desktop wild: short-aside false positive honestly REPORTED (not a confident clean pass)",
      aside && Number.isFinite(machinePct(aside)) && (machinePct(aside) >= 25 || aside.wavers),
      JSON.stringify(aside),
      wiScope,
    );
    const splice = probe(/splice/);
    chk(
      "desktop wild: human+AI splice miss honestly REPORTED (reads human)",
      splice && /reads human/.test(splice.r) && machinePct(splice) <= 60,
      JSON.stringify(splice),
      wiScope,
    );
    const modern = probe(/modern-LLM/);
    chk(
      "desktop wild: modern-LLM-style miss honestly REPORTED (reads human)",
      modern && /reads human/.test(modern.r) && machinePct(modern) <= 60,
      JSON.stringify(modern),
      wiScope,
    );
  }
  chk(
    "desktop wild: no horizontal overflow",
    await evalIn(wi.sessionId, overflowOk) === true,
    "",
    wiScope,
  );
  pageHygiene(wi, ROUTES.wild, "desktop", wiScope);
  await closePage(cdp, wi.targetId);

  // ---- 5. MULTI-MODEL (desktop): Xenova/gpt2 writes -> detector catches --------------
  const mmScope = rr(ROUTES.multimodel, "desktop");
  const mm = await openPage(cdp, url(ROUTES.multimodel));
  await setViewport(cdp, mm.sessionId, DESKTOP);
  await sleep(4000);
  const twoLoaders = `JSON.stringify([...document.querySelectorAll(".model-loader")].map((ml) => ({
      buttons: [...ml.querySelectorAll("button")].map((b) => b.textContent.trim()),
      status: (ml.querySelector(".status")?.textContent || "").trim().slice(0, 120) })))`;
  {
    const loaders = JSON.parse(await evalIn(mm.sessionId, twoLoaders));
    const genGate = loaders[0] && loaders[0].buttons.some((b) => /Download/i.test(b));
    chk(
      `desktop multi-model: two stages gated independently — ${GENERATOR_ID} Download gate on fresh profile`,
      loaders.length === 2 && genGate,
      JSON.stringify(loaders),
      mmScope,
    );
  }
  // detector (cached from the overview download) auto-inits while GPT-2 stays gated
  await waitFor(
    mm.sessionId,
    `(() => { const mls = [...document.querySelectorAll(".model-loader")];
      return mls.some((ml) => /ready|running/i.test(ml.textContent)); })()`,
    5 * 60 * 1000,
    3000,
    "multi-model detector auto-init",
  );
  // real GPT-2 download (~280 MB int8) with visible progress
  await evalIn(
    mm.sessionId,
    `(() => { const gen = document.querySelectorAll(".model-loader")[0];
      const b = [...gen.querySelectorAll("button")].find((x) => /Download/i.test(x.textContent));
      if (b) b.click(); return !!b; })()`,
  );
  const dl2 = await downloadUntilReady(
    mm.sessionId,
    `document.querySelector("#go") && !document.querySelector("#go").disabled`,
    "multi-model gpt2 download",
    3,
    10 * 60 * 1000,
  );
  chk(
    `desktop multi-model: real download with visible progress (~280 MB int8 ${GENERATOR_ID})`,
    dl2.ok && dl2.samples.some((s) => /\d+(\.\d+)?\s*MB/.test(s.prog)),
    dl2.samples.find((s) => /MB/.test(s.prog))?.prog ||
      JSON.stringify(dl2.samples.at(-1) || "(no samples)"),
    mmScope,
  );
  // generate -> detect -> CAUGHT
  await evalIn(mm.sessionId, `document.querySelector("#go").click()`);
  await waitFor(
    mm.sessionId,
    `!/appears here/.test(document.querySelector("#genOut")?.textContent || "") &&
     document.querySelector("#dMs")?.textContent !== "–"`,
    5 * 60 * 1000,
    3000,
    "multi-model generate + detect",
  );
  {
    const run = JSON.parse(
      await evalIn(
        mm.sessionId,
        `JSON.stringify({ gen: document.querySelector("#genOut").textContent.slice(0, 160),
          gTokens: document.querySelector("#gTokens").textContent,
          gMs: document.querySelector("#gMs").textContent,
          gBackend: document.querySelector("#gBackend").textContent,
          label: document.querySelector("#vLabel").textContent,
          conf: document.querySelector("#vConf").textContent,
          dMs: document.querySelector("#dMs").textContent,
          dBackend: document.querySelector("#dBackend").textContent,
          call: document.querySelector("#callOut").textContent })`,
      ),
    );
    chk(
      "desktop multi-model: Xenova/gpt2 really generated tokens (WASM, count + ms shown)",
      run.gen.length > 40 && Number.parseInt(run.gTokens, 10) > 0 && /ms/.test(run.gMs) &&
        run.gBackend.length > 0,
      JSON.stringify({
        gTokens: run.gTokens,
        gMs: run.gMs,
        gBackend: run.gBackend,
        gen: run.gen.slice(0, 80),
      }),
      mmScope,
    );
    chk(
      `desktop multi-model: ${DETECTOR_ID} CAUGHT the generator's own output`,
      /machine-written/.test(run.label) && pct(run.conf) >= 90 && /CAUGHT/.test(run.call) &&
        run.dBackend.length > 0 && /ms/.test(run.dMs),
      JSON.stringify({ label: run.label, conf: run.conf, call: run.call, dMs: run.dMs }),
      mmScope,
    );
  }
  // stealth duel: greedy vs sampled — both arms caught
  await evalIn(mm.sessionId, `document.querySelector("#duel").click()`);
  await waitFor(
    mm.sessionId,
    `document.querySelectorAll("#duelOut .pair").length === 2`,
    8 * 60 * 1000,
    3000,
    "multi-model duel",
  );
  {
    const duel = JSON.parse(
      await evalIn(
        mm.sessionId,
        `JSON.stringify([...document.querySelectorAll("#duelOut .pair")].map((p) => ({
          arm: p.querySelector(".src").textContent, v: p.querySelector(".v").textContent })))`,
      ),
    );
    chk(
      "desktop multi-model: stealth duel — greedy AND sampled arms both caught",
      duel.length === 2 && duel.every((d) => /caught/.test(d.v) && pct(d.v) >= 90),
      JSON.stringify(duel),
      mmScope,
    );
  }
  chk(
    "desktop multi-model: no horizontal overflow",
    await evalIn(mm.sessionId, overflowOk) === true,
    "",
    mmScope,
  );
  pageHygiene(mm, ROUTES.multimodel, "desktop", mmScope);
  await closePage(cdp, mm.targetId);

  // =================================================================================
  // MOBILE pass (360x740 DPR3) — both stages now cached; each route auto-inits and
  // drives REAL inference again at the mobile viewport.
  // =================================================================================

  // ---- overview (mobile) -----------------------------------------------------------
  const movScope = rr(ROUTES.overview, "mobile");
  const mov = await openPage(cdp, url(ROUTES.overview));
  await setViewport(cdp, mov.sessionId, MOBILE);
  await waitAutoInit(
    mov.sessionId,
    `document.querySelector("#text") && !document.querySelector("#text").disabled`,
    ROUTES.overview,
    "mobile",
    movScope,
  );
  await evalIn(mov.sessionId, `[...document.querySelectorAll(".chip")][0].click()`);
  await waitFor(
    mov.sessionId,
    `/machine-written/.test(document.querySelector("#vLabel")?.textContent || "")`,
    90000,
    1000,
    "mobile overview GPT-2 chip verdict",
  );
  {
    const v = JSON.parse(await evalIn(mov.sessionId, verdict));
    chk(
      "mobile overview: real inference at 360px (GPT-2 chip reads machine-written)",
      pct(v.conf) >= 90,
      JSON.stringify(v),
      movScope,
    );
    const probs = v.probs.map(pct).filter((n) => Number.isFinite(n));
    chk(
      "mobile overview: see-inside softmax surface rendered at 360px",
      probs.length === 2,
      JSON.stringify(v.probs),
      movScope,
    );
  }
  chk(
    "mobile overview: no horizontal overflow at 360px",
    await evalIn(mov.sessionId, overflowOk) === true,
    "",
    movScope,
  );
  pageHygiene(mov, ROUTES.overview, "mobile", movScope);
  await closePage(cdp, mov.targetId);

  // ---- basics (mobile) ---------------------------------------------------------------
  const mbaScope = rr(ROUTES.basics, "mobile");
  const mba = await openPage(cdp, url(ROUTES.basics));
  await setViewport(cdp, mba.sessionId, MOBILE);
  await waitAutoInit(
    mba.sessionId,
    `document.querySelector("#pairRun") && !document.querySelector("#pairRun").disabled`,
    ROUTES.basics,
    "mobile",
    mbaScope,
  );
  await evalIn(mba.sessionId, `document.querySelector("#pairRun").click()`);
  await waitFor(
    mba.sessionId,
    `document.querySelectorAll("#pairs .pair").length === 6`,
    180000,
    1000,
    "mobile basics pairs",
  );
  {
    const pairs = JSON.parse(
      await evalIn(
        mba.sessionId,
        `JSON.stringify([...document.querySelectorAll("#pairs .pair")].map((p) => ({
          src: p.querySelector(".src").textContent, v: p.querySelector(".v").textContent })))`,
      ),
    );
    const separated = pairs.length === 6 &&
      pairs.filter((p) => /GPT-2/.test(p.src)).every((p) => /machine-written/.test(p.v)) &&
      pairs.filter((p) => /human/.test(p.src)).every((p) => /human-written/.test(p.v));
    chk("mobile basics: 6/6 pairs separated at 360px", separated, JSON.stringify(pairs), mbaScope);
  }
  chk(
    "mobile basics: no horizontal overflow at 360px",
    await evalIn(mba.sessionId, overflowOk) === true,
    "",
    mbaScope,
  );
  pageHygiene(mba, ROUTES.basics, "mobile", mbaScope);
  await closePage(cdp, mba.targetId);

  // ---- practical (mobile) -------------------------------------------------------------
  const mprScope = rr(ROUTES.practical, "mobile");
  const mpr = await openPage(cdp, url(ROUTES.practical));
  await setViewport(cdp, mpr.sessionId, MOBILE);
  await waitAutoInit(
    mpr.sessionId,
    `document.querySelector("#run") && !document.querySelector("#run").disabled`,
    ROUTES.practical,
    "mobile",
    mprScope,
  );
  await evalIn(mpr.sessionId, `document.querySelector("#run").click()`);
  await waitFor(
    mpr.sessionId,
    `document.querySelectorAll("#queue .ticket").length >= 6`,
    180000,
    1000,
    "mobile practical triage",
  );
  {
    const t = JSON.parse(
      await evalIn(
        mpr.sessionId,
        `JSON.stringify({ flag: document.querySelector("#rFlag").textContent,
          tickets: document.querySelectorAll("#queue .ticket").length })`,
      ),
    );
    chk(
      "mobile practical: real batch triage at 360px (>=6 tickets, >=1 flagged)",
      t.tickets >= 6 && Number.parseInt(t.flag, 10) >= 1,
      JSON.stringify(t),
      mprScope,
    );
  }
  chk(
    "mobile practical: no horizontal overflow at 360px",
    await evalIn(mpr.sessionId, overflowOk) === true,
    "",
    mprScope,
  );
  pageHygiene(mpr, ROUTES.practical, "mobile", mprScope);
  await closePage(cdp, mpr.targetId);

  // ---- wild (mobile) --------------------------------------------------------------------
  const mwiScope = rr(ROUTES.wild, "mobile");
  const mwi = await openPage(cdp, url(ROUTES.wild));
  await setViewport(cdp, mwi.sessionId, MOBILE);
  await waitAutoInit(
    mwi.sessionId,
    `document.querySelector("#run") && !document.querySelector("#run").disabled`,
    ROUTES.wild,
    "mobile",
    mwiScope,
  );
  await evalIn(mwi.sessionId, `document.querySelector("#run").click()`);
  await waitFor(
    mwi.sessionId,
    `document.querySelectorAll("#probes .probe").length === 8`,
    240000,
    1500,
    "mobile wild gauntlet",
  );
  {
    const g = JSON.parse(
      await evalIn(
        mwi.sessionId,
        `JSON.stringify({ n: document.querySelector("#rN").textContent,
          ms: document.querySelector("#rMs").textContent,
          control: [...document.querySelectorAll("#probes .probe")].map((p) =>
            p.querySelector(".name").textContent + " :: " +
            p.querySelector(".route").textContent.replace(/\\s+/g, " ")).find((s) => /Control/.test(s)) })`,
      ),
    );
    chk(
      "mobile wild: 8-probe gauntlet at 360px, GPT-2 control caught",
      g.n === "8" && /reads machine/.test(g.control || "") &&
        pct((g.control || "").split("machine").pop()) >= 90,
      JSON.stringify(g),
      mwiScope,
    );
  }
  chk(
    "mobile wild: no horizontal overflow at 360px",
    await evalIn(mwi.sessionId, overflowOk) === true,
    "",
    mwiScope,
  );
  pageHygiene(mwi, ROUTES.wild, "mobile", mwiScope);
  await closePage(cdp, mwi.targetId);

  // ---- multi-model (mobile): cached Xenova/gpt2 generates, detector catches -------------
  const mmmScope = rr(ROUTES.multimodel, "mobile");
  const mmm = await openPage(cdp, url(ROUTES.multimodel));
  await setViewport(cdp, mmm.sessionId, MOBILE);
  await waitAutoInit(
    mmm.sessionId,
    `document.querySelector("#go") && !document.querySelector("#go").disabled`,
    ROUTES.multimodel,
    "mobile",
    mmmScope,
  );
  await evalIn(mmm.sessionId, `document.querySelector("#go").click()`);
  await waitFor(
    mmm.sessionId,
    `!/appears here/.test(document.querySelector("#genOut")?.textContent || "") &&
     document.querySelector("#dMs")?.textContent !== "–"`,
    5 * 60 * 1000,
    3000,
    "mobile multi-model generate + detect",
  );
  {
    const run = JSON.parse(
      await evalIn(
        mmm.sessionId,
        `JSON.stringify({ gTokens: document.querySelector("#gTokens").textContent,
          label: document.querySelector("#vLabel").textContent,
          conf: document.querySelector("#vConf").textContent,
          call: document.querySelector("#callOut").textContent })`,
      ),
    );
    chk(
      "mobile multi-model: Xenova/gpt2 generates at 360px and the detector CAUGHT it",
      Number.parseInt(run.gTokens, 10) > 0 && /machine-written/.test(run.label) &&
        pct(run.conf) >= 90 && /CAUGHT/.test(run.call),
      JSON.stringify(run),
      mmmScope,
    );
  }
  chk(
    "mobile multi-model: no horizontal overflow at 360px",
    await evalIn(mmm.sessionId, overflowOk) === true,
    "",
    mmmScope,
  );
  pageHygiene(mmm, ROUTES.multimodel, "mobile", mmmScope);
  await closePage(cdp, mmm.targetId);
} catch (e) {
  console.log("ABORT", String(e?.stack || e).slice(0, 500));
} finally {
  console.log(`\n${pass}/${total} checks passed`);
  console.log("ROUTE-RESULTS-JSON: " + JSON.stringify(Object.values(routeResults)));
  const allRoutesPass = Object.keys(ROUTES).every((k) =>
    ["desktop", "mobile"].every((vp) => routeResults[`${ROUTES[k]}@${vp}`]?.pass === true)
  );
  chrome.kill();
  try {
    server.close();
  } catch { /* ignore */ }
  process.exit(pass === total && allRoutesPass && total > 0 ? 0 : 1);
}
