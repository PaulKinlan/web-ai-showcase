#!/usr/bin/env node
// Route-complete real-browser acceptance for canonical Turkish BERT NER.
// Drives overview/Basics/Practical/Wild/Multi-model at desktop+mobile and light+dark.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { mkdtempSync } from "node:fs";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  screenshot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const SLUG = "bert-base-turkish-cased-ner";
const RUN_FILE = join(repoRoot, `models/${SLUG}/acceptance-run.json`);
const SCREEN_DIR = join(repoRoot, `reports/acceptance/${SLUG}`);
const PROFILE = mkdtempSync(join(tmpdir(), "turkish-ner-acceptance-"));
// Keep routes and stages literal: the portfolio gate statically verifies every advertised page and
// real model stage before it trusts the dynamic browser evidence below.
const ROUTES = {
  overview: "models/bert-base-turkish-cased-ner/",
  basics: "models/bert-base-turkish-cased-ner/basics/",
  practical: "models/bert-base-turkish-cased-ner/practical/",
  wild: "models/bert-base-turkish-cased-ner/wild/",
  multimodel: "models/bert-base-turkish-cased-ner/multi-model/",
};
const STAGES = [
  "akdeniz27/bert-base-turkish-cased-ner",
  "onnx-community/xlm-roberta-base-language-detection-ONNX",
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let server, chrome;
const results = [];
let checks = 0, passed = 0;
function check(label, ok, detail = "") {
  checks++;
  if (ok) passed++;
  console.log(`${ok ? "PASS" : "FAIL"} ${label}${detail ? ` — ${detail}` : ""}`);
  return ok;
}
async function evaluate(cdp, sid, expression, timeout = 45000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression})}catch(error){return {__error:String(error?.stack||error)}}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sid,
    timeout,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(cdp, sid, expression, timeout, label) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await evaluate(cdp, sid, expression).catch(() => false)) return;
    if ((Date.now() - start) % 10000 < 1800) {
      console.log(`  [${label}] ${Math.round((Date.now() - start) / 1000)}s`);
    }
    await sleep(1500);
  }
  throw new Error(`hard timeout ${timeout}ms: ${label}`);
}
async function ensureReady(cdp, sid, count, label) {
  const start = Date.now(), lifecycle = [];
  let last = "";
  while (Date.now() - start < 12 * 60_000) {
    const snapshot = await evaluate(
      cdp,
      sid,
      `([...document.querySelectorAll('.model-loader')].map(x=>({state:x.dataset.state||'',status:(x.querySelector('.status')?.textContent||'').trim(),buttons:[...x.querySelectorAll('button')].map(b=>b.textContent.trim())})))`,
    );
    const encoded = JSON.stringify(snapshot);
    if (encoded !== last) {
      lifecycle.push({ atMs: Date.now() - start, loaders: snapshot });
      console.log(`  [${label}] ${encoded}`);
      last = encoded;
    }
    if (snapshot.length === count && snapshot.every((x) => x.state === "ready")) return lifecycle;
    await evaluate(
      cdp,
      sid,
      `(()=>{const buttons=[...document.querySelectorAll('.model-loader button')].filter(b=>/Download|Retry|Re-download/i.test(b.textContent)&&!b.disabled);setTimeout(()=>buttons.forEach(b=>b.click()),0);return buttons.length})()`,
    );
    await sleep(2500);
  }
  throw new Error(`hard timeout: ${label} loader`);
}
async function drive(cdp, page, route) {
  const sid = page.sessionId;
  await evaluate(
    cdp,
    sid,
    `(()=>{window.__longTasks=[];new PerformanceObserver(list=>window.__longTasks.push(...list.getEntries().map(x=>({name:x.name,startTime:x.startTime,duration:x.duration})))).observe({type:'longtask',buffered:true});return true})()`,
  );
  if (route === "overview") {
    await waitFor(
      cdp,
      sid,
      `!document.querySelector('#readout')?.hidden`,
      120000,
      "overview result",
    );
    const v = await evaluate(
      cdp,
      sid,
      `({out:document.querySelector('#out')?.textContent,entities:document.querySelector('#entities')?.textContent,tokens:[...document.querySelectorAll('#tokens .tok-chip')].map(x=>x.textContent),count:Number(document.querySelector('#rEnts')?.textContent)})`,
    );
    check(
      "overview real aligned PER/LOC spans",
      /Mustafa Kemal Atatürk/.test(v.entities) && /Ankara/.test(v.entities) && v.count >= 2,
      JSON.stringify(v),
    );
    check(
      "overview inspectable BIO tokens",
      v.tokens.some((x) => /B-PER/.test(x)) && v.tokens.some((x) => /I-PER/.test(x)),
      JSON.stringify(v.tokens),
    );
  }
  if (route === "basics") {
    await evaluate(
      cdp,
      sid,
      `(()=>{const t=document.querySelector('#text');t.value="Elif Şafak, Boğaziçi Üniversitesi'nde Ahmet Hamdi Tanpınar üzerine konuştu.";t.dispatchEvent(new Event('input',{bubbles:true}));return true})()`,
    );
    await waitFor(
      cdp,
      sid,
      `/Tanpınar/.test(document.querySelector('#entities')?.textContent||'')`,
      120000,
      "basics wordpiece result",
    );
    const v = await evaluate(
      cdp,
      sid,
      `({entities:document.querySelector('#entities')?.textContent,out:document.querySelector('#out')?.textContent})`,
    );
    check(
      "basics Turkish casing + WordPiece aggregation",
      /Elif Şafak/.test(v.entities) && /Boğaziçi Üniversitesi/.test(v.entities) &&
        /Ahmet Hamdi Tanpınar/.test(v.entities),
      JSON.stringify(v),
    );
  }
  if (route === "practical") {
    await waitFor(
      cdp,
      sid,
      `!document.querySelector('#results')?.hidden`,
      120000,
      "practical index",
    );
    await evaluate(
      cdp,
      sid,
      `(()=>{document.querySelector('#typeBoxes input[value=LOC]').click();document.querySelector('#redact').click();return true})()`,
    );
    const v = await evaluate(
      cdp,
      sid,
      `({index:document.querySelector('#index')?.textContent,json:document.querySelector('#json')?.textContent,redacted:document.querySelector('#redactOut')?.textContent,unique:Number(document.querySelector('#rUnique')?.textContent)})`,
    );
    check(
      "practical index JSON",
      v.unique >= 3 && /\"type\"/.test(v.json) &&
        /Ahmet Bolat|Türk Hava Yolları|Baykar/.test(v.index),
      JSON.stringify(v),
    );
    check(
      "practical review-first redaction",
      /\[KİŞİ\]|\[KURUM\]|\[YER\]/.test(v.redacted),
      v.redacted,
    );
  }
  if (route === "wild") {
    await waitFor(cdp, sid, `!document.querySelector('#fillArea')?.hidden`, 120000, "wild cast");
    await evaluate(
      cdp,
      sid,
      `(()=>{const i=document.querySelector('.remix-input');if(i)i.value='Zeynep';document.querySelector('#rebuild').click();return true})()`,
    );
    const v = await evaluate(
      cdp,
      sid,
      `({cast:document.querySelector('#cast')?.textContent,result:document.querySelector('#result')?.textContent,count:Number(document.querySelector('#rEnts')?.textContent)})`,
    );
    check(
      "wild recast derives from live spans",
      v.count >= 2 && /Zeynep/.test(v.result) && !/Louis XIV/.test(v.cast),
      JSON.stringify(v),
    );
  }
  if (route === "multimodel") {
    await evaluate(cdp, sid, `(()=>{document.querySelector('#run').click();return true})()`);
    await waitFor(
      cdp,
      sid,
      `!document.querySelector('#steps')?.hidden`,
      180000,
      "multi-model result",
    );
    const v = await evaluate(
      cdp,
      sid,
      `({language:document.querySelector('#rLang')?.textContent,confidence:document.querySelector('#rConfidence')?.textContent,entities:document.querySelector('#entities')?.textContent,tokens:document.querySelectorAll('#tokens .tok-chip').length})`,
    );
    check(
      "multi-model language detector stage",
      /tr/i.test(v.language) && parseFloat(v.confidence) > 50,
      JSON.stringify(v),
    );
    check(
      "multi-model canonical NER stage",
      /Elif Şafak/.test(v.entities) && /Boğaziçi Üniversitesi/.test(v.entities) && v.tokens > 5,
      JSON.stringify(v),
    );
  }
}
async function lifecycleRelease(cdp, sid) {
  const before = await evaluate(cdp, sid, `document.querySelector('.model-loader')?.dataset.state`);
  await evaluate(
    cdp,
    sid,
    `(()=>{[...document.querySelectorAll('.model-loader button')].find(b=>/Release from memory/i.test(b.textContent))?.click();return true})()`,
  );
  await waitFor(
    cdp,
    sid,
    `document.querySelector('.model-loader')?.dataset.state==='released'`,
    60000,
    "release",
  );
  const released = await evaluate(
    cdp,
    sid,
    `({state:document.querySelector('.model-loader')?.dataset.state,disabled:document.querySelector('#text')?.disabled,buttons:[...document.querySelectorAll('.model-loader button')].map(b=>b.textContent.trim())})`,
  );
  await evaluate(
    cdp,
    sid,
    `(()=>{[...document.querySelectorAll('.model-loader button')].find(b=>/Load model into memory/i.test(b.textContent))?.click();return true})()`,
  );
  const reload = await ensureReady(cdp, sid, 1, "release reload");
  check(
    "release disables controls and cached reload returns ready",
    before === "ready" && released.state === "released" && released.disabled === true,
    JSON.stringify(released),
  );
  return { before, released, reload };
}
try {
  mkdirSync(SCREEN_DIR, { recursive: true });
  ({ server } = await startServer());
  chrome = await launchChrome({
    userDataDir: PROFILE,
    resetProfile: true,
    removeProfileOnKill: true,
  });
  const cdp = new CDP(chrome.ws);
  for (const [route, path] of Object.entries(ROUTES)) {
    for (const device of ["desktop", "mobile"]) {
      for (const theme of ["light", "dark"]) {
        console.log(`\n=== ${route} ${device} ${theme} ===`);
        const page = await openPage(
          cdp,
          `http://127.0.0.1:${server.address().port}/web-ai-showcase/${path}`,
        );
        await setViewport(cdp, page.sessionId, device === "mobile" ? MOBILE : DESKTOP);
        await cdp.send("Emulation.setEmulatedMedia", {
          features: [{ name: "prefers-color-scheme", value: theme }],
        }, page.sessionId);
        const lifecycle = await ensureReady(
          cdp,
          page.sessionId,
          route === "multimodel" ? 2 : 1,
          `${route} ${device} ${theme}`,
        );
        await drive(cdp, page, route);
        let release = null;
        if (route === "overview" && device === "desktop" && theme === "light") {
          release = await lifecycleRelease(cdp, page.sessionId);
        }
        const hygiene = await evaluate(
          cdp,
          page.sessionId,
          `({overflow:document.documentElement.scrollWidth-innerWidth,unnamedButtons:[...document.querySelectorAll('button')].filter(b=>!((b.textContent||'').trim()||b.getAttribute('aria-label')||b.getAttribute('aria-labelledby'))).length,smallControls:[...document.querySelectorAll('button,input:not([type=checkbox]),select,summary')].filter(x=>{const r=x.getBoundingClientRect();return r.width>0&&r.height>0&&(r.width<44||r.height<44)}).map(x=>({tag:x.tagName,text:(x.textContent||x.value||'').trim(),width:x.getBoundingClientRect().width,height:x.getBoundingClientRect().height})),longTasks:window.__longTasks||[],controls:[...document.querySelectorAll('button,input,textarea,select,summary')].map(x=>({tag:x.tagName,type:x.type||'',name:(x.textContent||x.value||x.getAttribute('aria-label')||'').trim(),disabled:!!x.disabled}))})`,
        );
        check(
          `${route} ${device} ${theme} responsive/a11y hygiene`,
          hygiene.overflow <= 1 && hygiene.unnamedButtons === 0 &&
            hygiene.smallControls.length === 0,
          JSON.stringify(hygiene),
        );
        check(
          `${route} ${device} ${theme} console/network clean`,
          page.errors.length === 0 && page.netFailures.length === 0,
          JSON.stringify({ console: page.errors, network: page.netFailures }),
        );
        const shot = join(SCREEN_DIR, `${route}-${device}-${theme}.png`);
        await screenshot(cdp, page.sessionId, shot);
        const network = await evaluate(
          cdp,
          page.sessionId,
          `performance.getEntriesByType('resource').map(x=>({name:x.name,initiatorType:x.initiatorType,duration:x.duration,transferSize:x.transferSize,encodedBodySize:x.encodedBodySize,decodedBodySize:x.decodedBodySize}))`,
        );
        results.push({
          route,
          device,
          theme,
          url: path,
          screenshot: shot.replace(repoRoot + "/", ""),
          lifecycle,
          release,
          hygiene,
          consoleErrors: page.errors,
          networkFailures: page.netFailures,
          network,
        });
        await closePage(cdp, page.targetId);
        await sleep(500);
      }
    }
  }
  const record = {
    schemaVersion: 1,
    slug: SLUG,
    implementationCommit: null,
    runAt: new Date().toISOString(),
    runner: {
      chrome: "HeadlessChrome/150",
      profile: "fresh then shared across route cells",
      modelRevision: "99995f7d2be4b3a28c74f0d36ee97f8c04ee0571",
      modelBytes: 440394743,
      stages: STAGES,
    },
    summary: {
      checks,
      passed,
      failed: checks - passed,
      cells: results.length,
      screenshots: results.length,
    },
    results,
  };
  if (WRITE_RUN) writeFileSync(RUN_FILE, JSON.stringify(record, null, 2) + "\n");
  console.log(
    `\nRESULT ${passed}/${checks}; cells=${results.length}; screenshots=${results.length}`,
  );
  if (passed !== checks) process.exitCode = 1;
} finally {
  server?.close();
  if (chrome) await chrome.kill();
  rmSync(PROFILE, { recursive: true, force: true });
}
