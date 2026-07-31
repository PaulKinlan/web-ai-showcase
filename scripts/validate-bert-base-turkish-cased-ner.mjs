#!/usr/bin/env node
// Fail-closed, route-complete real-browser acceptance for canonical Turkish BERT NER.
// Drives every page sample/control at desktop + mobile, captures populated full-page light/dark
// evidence, and exercises failure/Retry/current/offline/partial/corrupt/clear/disposal lifecycles.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const SLUG = "bert-base-turkish-cased-ner";
const PRIMARY = "akdeniz27/bert-base-turkish-cased-ner";
const PRIMARY_REVISION = "99995f7d2be4b3a28c74f0d36ee97f8c04ee0571";
const PRIMARY_URL = `https://huggingface.co/${PRIMARY}/resolve/${PRIMARY_REVISION}/model.onnx`;
const PRIMARY_BYTES = 440394743;
const PRIMARY_SHA256 = "a8f8a685d1a3dbf4a22a0c3ec9810f12a7035062fd61d79cadb759c24ace4482";
const LANGUAGE = "onnx-community/xlm-roberta-base-language-detection-ONNX";
const LANGUAGE_REVISION = "919c87aa2749131ae1ab709931a16bf1cc9774ea";
const LANGUAGE_BYTES = 278836241;
const LANGUAGE_SHA256 = "e3a2f1b44ea6a76683e4655127531b05c6b568fe643bd39bddf7f7c62ab182c9";
const ROUTES = {
  overview: "models/bert-base-turkish-cased-ner/",
  basics: "models/bert-base-turkish-cased-ner/basics/",
  practical: "models/bert-base-turkish-cased-ner/practical/",
  wild: "models/bert-base-turkish-cased-ner/wild/",
  multimodel: "models/bert-base-turkish-cased-ner/multi-model/",
};
const STAGES = [PRIMARY, LANGUAGE];
const RUN_FILE = join(repoRoot, `models/${SLUG}/acceptance-run.json`);
const REAL_SCREEN_DIR = join(repoRoot, `reports/acceptance/${SLUG}`);
const TEMP_ROOT = mkdtempSync(join(tmpdir(), "turkish-ner-acceptance-"));
const SCREEN_DIR = WRITE_RUN ? REAL_SCREEN_DIR : join(TEMP_ROOT, "screens");
const PROFILE = join(TEMP_ROOT, "profile");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
const assertions = [];
const lifecycle = [];
const screenshots = [];
const network = [];
const trackedSessions = new Set();
const requestChains = new Map();
let blocks = [];
let server;
let chrome;
let failures = 0;

function check(id, ok, evidence, details) {
  assertions.push({
    id,
    state: ok ? "pass" : "fail",
    evidence,
    ...(details == null ? {} : { details }),
  });
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} ${id} — ${evidence}`);
  return ok;
}
async function evaluate(cdp, sessionId, expression, timeout = 60000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression})}catch(error){return {__error:String(error?.stack||error)}}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeout,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}
async function waitFor(cdp, sessionId, expression, label, timeout = 15 * 60_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await evaluate(cdp, sessionId, expression).catch(() => false)) return;
    if ((Date.now() - started) % 15000 < 800) {
      console.log(`  [${label}] ${Math.round((Date.now() - started) / 1000)}s`);
    }
    await sleep(700);
  }
  throw new Error(`hard timeout ${timeout}ms: ${label}`);
}
async function click(cdp, sid, selector) {
  return evaluate(
    cdp,
    sid,
    `(()=>{const node=document.querySelector(${
      JSON.stringify(selector)
    });if(!node||node.disabled)return false;node.click();return true})()`,
  );
}
async function setValue(cdp, sid, selector, value, event = "input") {
  return evaluate(
    cdp,
    sid,
    `(()=>{const node=document.querySelector(${
      JSON.stringify(selector)
    });if(!node)return false;node.value=${JSON.stringify(value)};node.dispatchEvent(new Event(${
      JSON.stringify(event)
    },{bubbles:true}));return true})()`,
  );
}
async function loaderSnapshot(cdp, sid) {
  return evaluate(
    cdp,
    sid,
    `([...document.querySelectorAll('.model-loader')].map(node=>({state:node.dataset.state,status:(node.querySelector('.status')?.textContent||'').trim(),buttons:[...node.querySelectorAll('button')].map(button=>button.textContent.trim())})))`,
  );
}
async function clickLoaderActions(cdp, sid, pattern) {
  return evaluate(
    cdp,
    sid,
    `(()=>{let n=0;for(const button of document.querySelectorAll('.model-loader button'))if(${pattern}.test(button.textContent)&&!button.disabled){button.click();n++}return n})()`,
  );
}
async function ensureReady(cdp, sid, count, label) {
  const observed = [];
  const started = Date.now();
  let last = "";
  while (Date.now() - started < 15 * 60_000) {
    const snapshot = await loaderSnapshot(cdp, sid);
    const encoded = JSON.stringify(snapshot);
    if (encoded !== last) {
      observed.push({ atMs: Date.now() - started, loaders: snapshot });
      console.log(`  [${label}] ${encoded}`);
      last = encoded;
    }
    if (snapshot.length === count && snapshot.every((item) => item.state === "ready")) {
      return observed;
    }
    await clickLoaderActions(cdp, sid, "/Download|Retry|Re-download|Continue/i");
    await sleep(1200);
  }
  throw new Error(`hard timeout: ${label}`);
}
async function enableNetwork(cdp, sessionId) {
  if (!sessionId || trackedSessions.has(sessionId)) return;
  trackedSessions.add(sessionId);
  await cdp.send("Network.enable", {}, sessionId).catch(() => trackedSessions.delete(sessionId));
  if (trackedSessions.has(sessionId)) {
    await cdp.send(
      "Network.setBlockedURLs",
      { urls: blocks.length ? blocks : ["__never_match__"] },
      sessionId,
    ).catch(() => {});
  }
}
async function setBlocks(cdp, patterns) {
  blocks = patterns;
  await Promise.all([...trackedSessions].map((sid) =>
    cdp.send("Network.setBlockedURLs", {
      urls: patterns.length ? patterns : ["__never_match__"],
    }, sid).catch(() => {})
  ));
}
async function attachPage(cdp, page) {
  await enableNetwork(cdp, page.sessionId);
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: false,
  }, page.sessionId);
  await sleep(200);
}
async function fullScreenshot(cdp, sid, file) {
  await evaluate(
    cdp,
    sid,
    `(async()=>{const style=document.createElement('style');style.textContent='*{content-visibility:visible!important}';document.head.append(style);for(const node of document.querySelectorAll('header,section,footer')){node.scrollIntoView({block:'center'});await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))}scrollTo(0,0);return true})()`,
  );
  const metrics = await cdp.send("Page.getLayoutMetrics", {}, sid);
  const size = metrics.cssContentSize;
  const { data } = await cdp.send(
    "Page.captureScreenshot",
    {
      format: "png",
      captureBeyondViewport: true,
      fromSurface: true,
      clip: {
        x: 0,
        y: 0,
        width: Math.min(size.width, 1600),
        height: Math.min(size.height, 12000),
        scale: 1,
      },
    },
    sid,
    60000,
  );
  writeFileSync(file, Buffer.from(data, "base64"));
  return { cssWidth: size.width, cssHeight: size.height };
}
async function toggleEveryDetails(cdp, sid) {
  return evaluate(
    cdp,
    sid,
    `(()=>{const out=[];for(const details of document.querySelectorAll('details')){const summary=details.querySelector('summary');if(!summary)continue;summary.click();const opened=details.open;summary.click();out.push({text:summary.textContent.trim(),opened,closed:!details.open})}return out})()`,
  );
}
async function hygiene(cdp, sid, requestedViewport) {
  return evaluate(
    cdp,
    sid,
    `({
    viewport:{
      requestedWidth:${requestedViewport.width},
      requestedHeight:${requestedViewport.height},
      innerWidth,
      innerHeight,
      visualViewportWidth:window.visualViewport?.width??null,
      visualViewportHeight:window.visualViewport?.height??null,
      devicePixelRatio
    },
    overflow:document.documentElement.scrollWidth-innerWidth,
    unnamedButtons:[...document.querySelectorAll('button')].filter(b=>!((b.textContent||'').trim()||b.getAttribute('aria-label')||b.getAttribute('aria-labelledby'))).length,
    smallControls:[...document.querySelectorAll('button,input:not([type=checkbox]),select,summary')].filter(x=>{const r=x.getBoundingClientRect();return r.width>0&&r.height>0&&(r.width<44||r.height<44)}).map(x=>{const r=x.getBoundingClientRect();return {tag:x.tagName,text:(x.textContent||x.value||'').trim(),width:r.width,height:r.height}}),
    controls:[...document.querySelectorAll('button,input,textarea,select,summary')].map(x=>({tag:x.tagName,type:x.type||'',name:(x.textContent||x.value||x.getAttribute('aria-label')||'').trim(),disabled:!!x.disabled}))
  })`,
  );
}
async function waitStableResult(cdp, sid, selector, label, timeout = 180000) {
  await waitFor(
    cdp,
    sid,
    `document.querySelector(${JSON.stringify(selector)}) && !document.querySelector(${
      JSON.stringify(selector)
    }).hidden && (document.querySelector(${
      JSON.stringify(selector)
    }).textContent||'').trim().length>0`,
    label,
    timeout,
  );
}

async function driveOverview(cdp, sid) {
  const samples = [];
  for (
    const [index, button] of (await evaluate(
      cdp,
      sid,
      `[...document.querySelectorAll('#chips [data-set]')].map((b,i)=>({i,text:b.textContent.trim(),value:b.dataset.set}))`,
    )).entries()
  ) {
    await click(cdp, sid, `#chips [data-set]:nth-child(${index + 1})`);
    await waitStableResult(cdp, sid, "#entities", `overview sample ${index}`);
    await sleep(1200);
    const observed = await evaluate(
      cdp,
      sid,
      `({input:document.querySelector('#text').value,entities:document.querySelector('#entities').textContent,tokens:document.querySelectorAll('#tokens .tok-chip').length})`,
    );
    samples.push({ label: button.text, expected: button.value, ...observed });
  }
  const latestText = "Elif Şafak İstanbul'da konuştu.";
  await setValue(cdp, sid, "#text", "Mustafa Kemal Atatürk Ankara'da konuştu.");
  await setValue(cdp, sid, "#text", latestText);
  await waitFor(
    cdp,
    sid,
    `/Elif Şafak/.test(document.querySelector('#entities')?.textContent||'')`,
    "overview latest-wins",
  );
  const latest = await evaluate(
    cdp,
    sid,
    `({input:document.querySelector('#text').value,entities:document.querySelector('#entities').textContent,error:document.querySelector('#runStatus').hidden?'':document.querySelector('#runStatus').textContent})`,
  );
  check(
    "overview-all-samples",
    samples.length === 4 &&
      samples.every((row) => row.input === row.expected && row.tokens > 0 && row.entities.trim()),
    "All four overview examples produced inspectable real output.",
    samples,
  );
  check(
    "overview-stale-suppression",
    latest.input === latestText && /Elif Şafak/.test(latest.entities) &&
      !/Atatürk/.test(latest.entities) && !latest.error,
    "Rapid edits retained only the newest result without an abort error.",
    latest,
  );
  return { samples, latest };
}
async function driveBasics(cdp, sid) {
  const samples = [];
  const buttons = await evaluate(
    cdp,
    sid,
    `[...document.querySelectorAll('#chips [data-set]')].map(b=>({label:b.textContent.trim(),value:b.dataset.set}))`,
  );
  for (let index = 0; index < buttons.length; index++) {
    await click(cdp, sid, `#chips [data-set]:nth-child(${index + 1})`);
    await waitStableResult(cdp, sid, "#entities", `basics sample ${index}`);
    await sleep(1200);
    samples.push(
      await evaluate(
        cdp,
        sid,
        `({label:${JSON.stringify(buttons[index].label)},expected:${
          JSON.stringify(buttons[index].value)
        },input:document.querySelector('#text').value,entities:document.querySelector('#entities').textContent})`,
      ),
    );
  }
  check(
    "basics-all-samples",
    samples.length === 4 &&
      samples.every((row) => row.input === row.expected && row.entities.trim()),
    "All four Basics examples were activated and produced spans.",
    samples,
  );
  return { samples };
}
async function drivePractical(cdp, sid) {
  await waitFor(cdp, sid, `!document.querySelector('#results')?.hidden`, "practical index");
  const copyBefore = await evaluate(cdp, sid, `navigator.clipboard.readText().catch(()=>"")`);
  await click(cdp, sid, "#copy");
  await waitFor(
    cdp,
    sid,
    `/Copied/.test(document.querySelector('#copy').textContent)`,
    "JSON copy",
  );
  const copiedJson = await evaluate(cdp, sid, `navigator.clipboard.readText().catch(()=>"")`);
  const toggles = [];
  for (const type of ["PER", "ORG", "LOC"]) {
    await click(cdp, sid, `#typeBoxes input[value=${type}]`);
    await click(cdp, sid, "#redact");
    toggles.push(
      await evaluate(
        cdp,
        sid,
        `({type:${
          JSON.stringify(type)
        },checked:document.querySelector('#typeBoxes input[value=${type}]').checked,text:document.querySelector('#redactOut').textContent,marks:document.querySelectorAll('#redactOut .redact-mark').length})`,
      ),
    );
    await click(cdp, sid, `#typeBoxes input[value=${type}]`);
  }
  await click(cdp, sid, "#redact");
  await click(cdp, sid, "#copyRedact");
  await waitFor(
    cdp,
    sid,
    `/Copied/.test(document.querySelector('#copyRedact').textContent)`,
    "redacted copy",
  );
  const copiedRedaction = await evaluate(cdp, sid, `navigator.clipboard.readText().catch(()=>"")`);
  const summary = await evaluate(
    cdp,
    sid,
    `({json:document.querySelector('#json').textContent,index:document.querySelector('#index').textContent,redacted:document.querySelector('#redactOut').textContent,marks:document.querySelectorAll('#redactOut .redact-mark').length})`,
  );
  check(
    "practical-copy-controls",
    copiedJson.includes('"entities"') && copiedJson !== copyBefore &&
      copiedRedaction === summary.redacted,
    "Both copy buttons wrote their visible payloads.",
    { copiedJsonBytes: copiedJson.length, copiedRedaction },
  );
  check(
    "practical-all-type-toggles",
    toggles.length === 3 && toggles.every((row) => row.marks > 0 || /Nothing/.test(row.text)),
    "PER/ORG/LOC were each toggled and redaction rerun.",
    toggles,
  );
  check(
    "practical-turkish-redaction-marks",
    summary.marks > 0 && /\[KİŞİ\]|\[KURUM\]|\[YER\]/.test(summary.redacted),
    "Turkish placeholders render with typed highlight markup.",
    summary,
  );
  return { toggles, summary, copiedJsonBytes: copiedJson.length };
}
async function driveWild(cdp, sid) {
  const samples = [];
  const buttons = await evaluate(
    cdp,
    sid,
    `[...document.querySelectorAll('#chips [data-set]')].map(b=>({label:b.textContent.trim(),value:b.dataset.set}))`,
  );
  for (let index = 0; index < buttons.length; index++) {
    await click(cdp, sid, `#chips [data-set]:nth-child(${index + 1})`);
    await click(cdp, sid, "#run");
    await waitFor(
      cdp,
      sid,
      `!document.querySelector('#fillArea')?.hidden && document.querySelectorAll('.remix-input').length>0`,
      `wild sample ${index}`,
    );
    samples.push(
      await evaluate(
        cdp,
        sid,
        `({label:${JSON.stringify(buttons[index].label)},expected:${
          JSON.stringify(buttons[index].value)
        },input:document.querySelector('#text').value,fields:document.querySelectorAll('.remix-input').length})`,
      ),
    );
  }
  await click(cdp, sid, "#chips [data-set]:first-child");
  await click(cdp, sid, "#run");
  await waitFor(
    cdp,
    sid,
    `document.querySelectorAll('.remix-input').length>=3 && !document.querySelector('#fillArea').hidden`,
    "wild default cast",
  );
  const replacements = await evaluate(
    cdp,
    sid,
    `(()=>{const values=['Zeynep','Paris','Buckingham'];const fields=[...document.querySelectorAll('.remix-input')];fields.forEach((field,index)=>field.value=values[index]||('Replacement '+index));document.querySelector('#rebuild').click();return {fieldCount:fields.length,values:fields.map(f=>f.value),result:document.querySelector('#result').textContent}})()`,
  );
  check(
    "wild-all-samples",
    samples.length === 3 && samples.every((row) => row.input === row.expected && row.fields > 0),
    "All three Wild stories were run through real NER.",
    samples,
  );
  check(
    "wild-all-fields-end-to-start",
    replacements.fieldCount >= 3 &&
      replacements.values.every((value) => replacements.result.includes(value)) &&
      replacements.result === "Zeynep Paris'u fethetti ve Buckingham'nı yaptırdı.",
    "Every replacement field was populated and the rebuilt story preserved offsets.",
    replacements,
  );
  return { samples, replacements };
}
async function runMulti(cdp, sid, buttonIndex) {
  await click(cdp, sid, `[data-set]:nth-child(${buttonIndex + 1})`);
  const expected = await evaluate(
    cdp,
    sid,
    `document.querySelectorAll('[data-set]')[${buttonIndex}].dataset.set`,
  );
  await click(cdp, sid, "#run");
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#run')?.disabled===true`,
    `multi sample ${buttonIndex} started`,
  );
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#run')?.disabled===false && !document.querySelector('#steps')?.hidden && !document.querySelector('#readout')?.hidden`,
    `multi sample ${buttonIndex}`,
  );
  return evaluate(
    cdp,
    sid,
    `({expected:${
      JSON.stringify(expected)
    },input:document.querySelector('#text').value,language:document.querySelector('#rLang').textContent,confidence:document.querySelector('#rConfidence').textContent,entities:document.querySelector('#entities').textContent,tokens:document.querySelectorAll('#tokens .tok-chip').length})`,
  );
}
async function driveMultimodel(cdp, sid) {
  const cases = [];
  for (let index = 0; index < 3; index++) cases.push(await runMulti(cdp, sid, index));
  check(
    "multimodel-all-language-cases",
    cases.length === 3 &&
      cases.every((row) => row.input === row.expected && row.language && row.tokens > 0) &&
      /^tr$/i.test(cases[0].language) && /^en$/i.test(cases[1].language),
    "Turkish, English contrast, and code-switched samples ran through both real stages.",
    cases,
  );
  return { cases };
}
const DRIVERS = {
  overview: driveOverview,
  basics: driveBasics,
  practical: drivePractical,
  wild: driveWild,
  multimodel: driveMultimodel,
};

async function clearLoader(cdp, sid, mountSelector) {
  return evaluate(
    cdp,
    sid,
    `(()=>{const mount=document.querySelector(${
      JSON.stringify(mountSelector)
    });const button=[...mount.querySelectorAll('button')].find(b=>/Clear cached model/.test(b.textContent));if(!button)return false;button.click();return true})()`,
  );
}

try {
  mkdirSync(SCREEN_DIR, { recursive: true });
  if (WRITE_RUN) rmSync(REAL_SCREEN_DIR, { recursive: true, force: true });
  mkdirSync(SCREEN_DIR, { recursive: true });
  ({ server } = await startServer());
  chrome = await launchChrome({
    userDataDir: PROFILE,
    resetProfile: true,
    removeProfileOnKill: true,
  });
  const cdp = new CDP(chrome.ws);
  cdp.on((message) => {
    if (message.method === "Target.attachedToTarget") {
      const type = message.params.targetInfo?.type;
      if (type === "worker" || type === "service_worker") {
        void enableNetwork(cdp, message.params.sessionId);
      }
      return;
    }
    if (!trackedSessions.has(message.sessionId)) return;
    if (message.method === "Network.requestWillBeSent") {
      const key = `${message.sessionId}:${message.params.requestId}`;
      if (!requestChains.has(key)) {
        requestChains.set(key, { requestedUrl: message.params.request.url });
      }
    } else if (message.method === "Network.responseReceived") {
      const key = `${message.sessionId}:${message.params.requestId}`;
      const chain = requestChains.get(key) || { requestedUrl: message.params.response.url };
      const headers = message.params.response.headers || {};
      const length = headers["content-length"] ?? headers["Content-Length"] ??
        headers["x-linked-size"] ?? headers["X-Linked-Size"];
      network.push({
        requestedUrl: chain.requestedUrl,
        responseUrl: message.params.response.url,
        status: message.params.response.status,
        contentLength: /^\d+$/.test(String(length ?? "")) ? Number(length) : null,
        type: message.params.type,
      });
    }
  });
  await cdp.send("Target.setDiscoverTargets", { discover: true });
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    flatten: true,
    waitForDebuggerOnStart: false,
  });

  const base = `http://127.0.0.1:${server.address().port}/web-ai-showcase/`;
  const first = await openPage(cdp, base + ROUTES.overview);
  await attachPage(cdp, first);
  await cdp.send("Browser.grantPermissions", {
    origin: new URL(base).origin,
    permissions: ["clipboardReadWrite", "clipboardSanitizedWrite"],
  }).catch(() => {});
  await waitFor(
    cdp,
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='download-required'`,
    "first visit absent",
  );
  lifecycle.push({
    event: "first-visit",
    state: "download-required",
    snapshot: await loaderSnapshot(cdp, first.sessionId),
  });
  await setBlocks(cdp, [`*${PRIMARY}*`]);
  await clickLoaderActions(cdp, first.sessionId, "/Download/i");
  await waitFor(
    cdp,
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='error'`,
    "visible blocked download",
    120000,
  );
  lifecycle.push({
    event: "network-failure",
    state: "error",
    snapshot: await loaderSnapshot(cdp, first.sessionId),
  });
  check(
    "lifecycle-error-retry",
    lifecycle.at(-1).snapshot[0].buttons.some((button) => /Retry/.test(button)),
    "Blocked primary download surfaced a visible Retry.",
    lifecycle.at(-1),
  );
  await setBlocks(cdp, []);
  await clickLoaderActions(cdp, first.sessionId, "/Retry/i");
  lifecycle.push({
    event: "retry",
    transitions: await ensureReady(cdp, first.sessionId, 1, "retry recovery"),
  });
  first.errors.length = 0;
  first.netFailures.length = 0;

  // Real in-flight disposal: start a debounced request, release immediately, then prove the loader
  // settles and cached reload works. Unit tests assert the exact promise rejection/stale suppression.
  await setValue(
    cdp,
    first.sessionId,
    "#text",
    "Elif Şafak İstanbul Ankara Türkiye Boğaziçi Üniversitesi Ahmet Hamdi Tanpınar hakkında konuştu. "
      .repeat(12),
  );
  await sleep(380);
  await clickLoaderActions(cdp, first.sessionId, "/Release from memory/i");
  await waitFor(
    cdp,
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='released'`,
    "dispose race release",
    120000,
  );
  const race = await evaluate(
    cdp,
    first.sessionId,
    `({state:document.querySelector('.model-loader').dataset.state,textDisabled:document.querySelector('#text').disabled,error:document.querySelector('#runStatus').hidden?'':document.querySelector('#runStatus').textContent})`,
  );
  lifecycle.push({ event: "disposal-race", ...race });
  await clickLoaderActions(cdp, first.sessionId, "/Load model into memory/i");
  lifecycle.push({
    event: "cached-reload",
    transitions: await ensureReady(cdp, first.sessionId, 1, "cached reload"),
  });
  check(
    "lifecycle-disposal-race",
    race.state === "released" && race.textDisabled && !race.error,
    "Release during inference settled without stale output/error, then cached reload reached ready.",
    race,
  );

  await clickLoaderActions(cdp, first.sessionId, "/Release from memory/i");
  await waitFor(
    cdp,
    first.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='released'`,
    "offline release",
  );
  await setBlocks(cdp, [
    "https://huggingface.co/*",
    "https://*.hf.co/*",
    "https://*.xethub.hf.co/*",
  ]);
  await clickLoaderActions(cdp, first.sessionId, "/Load model into memory/i");
  await ensureReady(cdp, first.sessionId, 1, "offline cached load");
  await setValue(cdp, first.sessionId, "#text", "Elif Şafak İstanbul'da konuştu.");
  await waitFor(
    cdp,
    first.sessionId,
    `/Elif Şafak/.test(document.querySelector('#entities')?.textContent||'')`,
    "offline inference",
  );
  lifecycle.push({
    event: "offline-cached",
    state: "ready",
    entities: await evaluate(
      cdp,
      first.sessionId,
      `document.querySelector('#entities').textContent`,
    ),
  });
  await setBlocks(cdp, []);
  check(
    "lifecycle-offline-cached",
    /Elif Şafak/.test(lifecycle.at(-1).entities),
    "Pinned cached bytes revalidated and inferred while Hub URLs were blocked.",
    lifecycle.at(-1),
  );
  await closePage(cdp, first.targetId);

  // Every route × device runs real inference and every visible route sample/control. Both theme
  // screenshots are captured after output and lower controls have been scrolled/materialized.
  for (const [route, path] of Object.entries(ROUTES)) {
    for (const [device, viewport] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      console.log(`\n=== ${route} ${device} ===`);
      const page = await openPage(cdp, base + path);
      await attachPage(cdp, page);
      await setViewport(cdp, page.sessionId, viewport);
      const transitions = await ensureReady(
        cdp,
        page.sessionId,
        route === "multimodel" ? 2 : 1,
        `${route} ${device}`,
      );
      const controlEvidence = await DRIVERS[route](cdp, page.sessionId);
      const details = await toggleEveryDetails(cdp, page.sessionId);
      const cellHygiene = await hygiene(cdp, page.sessionId, viewport);
      const viewportMatches = device !== "mobile" ||
        (cellHygiene.viewport.requestedWidth === MOBILE.width &&
          cellHygiene.viewport.innerWidth === MOBILE.width &&
          cellHygiene.viewport.visualViewportWidth === MOBILE.width);
      if (device === "mobile") {
        check(
          `${route}-mobile-effective-viewport`,
          viewportMatches,
          `Requested, layout, and visual viewport widths must remain exactly ${MOBILE.width} CSS px.`,
          cellHygiene.viewport,
        );
      }
      const captures = [];
      for (const theme of ["light", "dark"]) {
        await cdp.send("Emulation.setEmulatedMedia", {
          features: [{ name: "prefers-color-scheme", value: theme }],
        }, page.sessionId);
        const file = join(SCREEN_DIR, `${route}-${device}-${theme}.png`);
        const dimensions = await fullScreenshot(cdp, page.sessionId, file);
        const record = {
          route,
          device,
          theme,
          path: relative(repoRoot, file),
          kind: "full-page-after-scroll",
          ...dimensions,
        };
        screenshots.push(record);
        captures.push(record);
      }
      const ok = viewportMatches && cellHygiene.overflow <= 1 && cellHygiene.unnamedButtons === 0 &&
        cellHygiene.smallControls.length === 0 && page.errors.length === 0 &&
        page.netFailures.length === 0 && details.every((item) => item.opened && item.closed);
      check(
        `${route}-${device}-hygiene`,
        ok,
        "Real inference/control matrix completed; all details toggled; no overflow, unnamed/small controls, console errors, or failed requests.",
        {
          hygiene: cellHygiene,
          details,
          consoleErrors: page.errors,
          networkFailures: page.netFailures,
        },
      );
      results.push({
        name: route,
        route: path,
        viewport: device,
        themes: ["light", "dark"],
        pass: ok,
        transitions,
        controlEvidence,
        details,
        hygiene: cellHygiene,
        consoleErrors: page.errors,
        networkFailures: page.netFailures,
        screenshots: captures,
      });
      await closePage(cdp, page.targetId);
    }
  }

  // Eviction/partial is a missing recorded file. Restore via the route's honest Re-download action.
  const evictPage = await openPage(cdp, base + ROUTES.overview);
  await attachPage(cdp, evictPage);
  await ensureReady(cdp, evictPage.sessionId, 1, "pre-eviction ready");
  await waitFor(
    cdp,
    evictPage.sessionId,
    `(async()=>{const m=await import('/web-ai-showcase/lib/model-cache.js');return (await m.listValidationRecords()).some(r=>r.modelId===${
      JSON.stringify(PRIMARY)
    }&&r.files?.length>1)})()`,
    "validation record",
    120000,
  );
  await sleep(3000);
  const eviction = await evaluate(
    cdp,
    evictPage.sessionId,
    `(async()=>{const m=await import('/web-ai-showcase/lib/model-cache.js');const record=(await m.listValidationRecords()).find(r=>r.modelId===${
      JSON.stringify(PRIMARY)
    });const url=record.files.find(url=>url!==${
      JSON.stringify(PRIMARY_URL)
    })||record.files[0];let removed=false;for(const name of await caches.keys()){const cache=await caches.open(name);if(await cache.delete(url)){removed=true;break}}return {removed,url,files:record.files.length,key:record.key}})()`,
  );
  await closePage(cdp, evictPage.targetId);
  const partial = await openPage(cdp, base + ROUTES.overview);
  await attachPage(cdp, partial);
  await waitFor(
    cdp,
    partial.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='partial'`,
    "partial state",
  );
  const partialSnapshot = await loaderSnapshot(cdp, partial.sessionId);
  lifecycle.push({ event: "partial-eviction", eviction, snapshot: partialSnapshot });
  await clickLoaderActions(cdp, partial.sessionId, "/Re-download/i");
  await ensureReady(cdp, partial.sessionId, 1, "partial recovery");
  check(
    "lifecycle-partial-recovery",
    eviction.removed && partialSnapshot[0].buttons.some((button) => /Re-download/.test(button)),
    "A missing recorded asset produced partial/Re-download and recovered.",
    lifecycle.at(-1),
  );

  // Poison the exact primary cache key. Auto-init must validate, delete, fail visibly, and only cache
  // the subsequent network bytes after exact length+SHA verification succeeds.
  await clickLoaderActions(cdp, partial.sessionId, "/Release from memory/i");
  await waitFor(
    cdp,
    partial.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='released'`,
    "pre-corrupt release",
  );
  await evaluate(
    cdp,
    partial.sessionId,
    `(async()=>{const cache=await caches.open('transformers-cache');await cache.put(${
      JSON.stringify(PRIMARY_URL)
    },new Response(new Uint8Array([1,2,3]),{status:200}));return true})()`,
  );
  await clickLoaderActions(cdp, partial.sessionId, "/Load model into memory/i");
  await waitFor(
    cdp,
    partial.sessionId,
    `document.querySelector('.model-loader')?.dataset.state==='error'`,
    "corrupt cache rejection",
    120000,
  );
  const corrupt = await evaluate(
    cdp,
    partial.sessionId,
    `(async()=>({snapshot:[...document.querySelectorAll('.model-loader')].map(n=>({state:n.dataset.state,status:n.querySelector('.status').textContent,buttons:[...n.querySelectorAll('button')].map(b=>b.textContent)})),poisonStillCached:!!(await caches.match(${
      JSON.stringify(PRIMARY_URL)
    }))}))()`,
  );
  lifecycle.push({ event: "corrupt-cache", ...corrupt });
  check(
    "lifecycle-corrupt-fail-closed",
    !corrupt.poisonStillCached && /integrity check failed/.test(corrupt.snapshot[0].status) &&
      corrupt.snapshot[0].buttons.some((button) => /Retry/.test(button)),
    "Wrong-length cached primary bytes were deleted and rejected with visible Retry.",
    corrupt,
  );
  await clickLoaderActions(cdp, partial.sessionId, "/Retry/i");
  await ensureReady(cdp, partial.sessionId, 1, "corrupt retry recovery");
  lifecycle.push({ event: "corrupt-retry-recovered", state: "ready" });
  await closePage(cdp, partial.targetId);

  // Multi-model disposal race covers the second engine; then drive both Release and both Clear controls.
  const finalPage = await openPage(cdp, base + ROUTES.multimodel);
  await attachPage(cdp, finalPage);
  await ensureReady(cdp, finalPage.sessionId, 2, "final multi ready");
  await click(cdp, finalPage.sessionId, "#run");
  await sleep(10);
  await evaluate(
    cdp,
    finalPage.sessionId,
    `(()=>{const button=[...document.querySelectorAll('#lang-loader button')].find(b=>/Release from memory/.test(b.textContent));button?.click();return !!button})()`,
  );
  await waitFor(
    cdp,
    finalPage.sessionId,
    `document.querySelector('#lang-loader .model-loader')?.dataset.state==='released'`,
    "language dispose race",
    120000,
  );
  const languageRace = await evaluate(
    cdp,
    finalPage.sessionId,
    `({state:document.querySelector('#lang-loader .model-loader').dataset.state,runDisabled:document.querySelector('#run').disabled,error:document.querySelector('#runStatus').hidden?'':document.querySelector('#runStatus').textContent})`,
  );
  await clickLoaderActions(cdp, finalPage.sessionId, "/Load model into memory/i");
  await ensureReady(cdp, finalPage.sessionId, 2, "language reload");
  lifecycle.push({ event: "language-disposal-race", ...languageRace });
  check(
    "language-disposal-race",
    languageRace.state === "released" && languageRace.runDisabled && !languageRace.error,
    "Language worker release during a two-stage run rejected/suppressed stale work and reloaded.",
    languageRace,
  );
  // Release both once so every Release control is explicitly operated, reload, then clear each cache.
  await clickLoaderActions(cdp, finalPage.sessionId, "/Release from memory/i");
  await waitFor(
    cdp,
    finalPage.sessionId,
    `document.querySelectorAll('.model-loader[data-state="released"]').length===2`,
    "both released",
  );
  await clickLoaderActions(cdp, finalPage.sessionId, "/Load model into memory/i");
  await ensureReady(cdp, finalPage.sessionId, 2, "both reloaded");
  await clearLoader(cdp, finalPage.sessionId, "#lang-loader");
  await waitFor(
    cdp,
    finalPage.sessionId,
    `document.querySelector('#lang-loader .model-loader')?.dataset.state==='download-required'`,
    "language clear",
  );
  await clearLoader(cdp, finalPage.sessionId, "#ner-loader");
  await waitFor(
    cdp,
    finalPage.sessionId,
    `document.querySelector('#ner-loader .model-loader')?.dataset.state==='download-required'`,
    "primary clear",
  );
  const cleared = await evaluate(
    cdp,
    finalPage.sessionId,
    `(async()=>{const m=await import('/web-ai-showcase/lib/model-cache.js');const records=(await m.listValidationRecords()).filter(r=>[${
      JSON.stringify(PRIMARY)
    },${
      JSON.stringify(LANGUAGE)
    }].includes(r.modelId));return {records:records.length,primaryFiles:(await m.scanCachedFiles(${
      JSON.stringify(PRIMARY)
    })).length,languageFiles:(await m.scanCachedFiles(${
      JSON.stringify(LANGUAGE)
    })).length,states:[...document.querySelectorAll('.model-loader')].map(n=>n.dataset.state)}})()`,
  );
  lifecycle.push({ event: "clear-both", ...cleared });
  check(
    "lifecycle-clear-both",
    cleared.records === 0 && cleared.primaryFiles === 0 && cleared.languageFiles === 0 &&
      cleared.states.every((state) => state === "download-required"),
    "Both visible Clear controls removed files/metadata and returned to Download.",
    cleared,
  );
  await closePage(cdp, finalPage.targetId);
  lifecycle.push({
    event: "stale-update",
    state: "not-applicable",
    reason:
      `Both stages pin immutable revisions (${PRIMARY_REVISION}, ${LANGUAGE_REVISION}); mutable Update is intentionally disabled.`,
  });
  lifecycle.push({
    event: "unsupported-device",
    state: "not-applicable",
    reason:
      "Both stages use baseline WASM with no optional WebGPU feature gate; desktop/mobile behavior is exercised rather than fabricating unsupported UI.",
  });

  const primaryNetwork = network.find((row) =>
    row.requestedUrl.includes(PRIMARY) && row.requestedUrl.includes(PRIMARY_REVISION) &&
    row.requestedUrl.includes("model.onnx") && row.status === 200 &&
    row.contentLength === PRIMARY_BYTES
  );
  const languageNetwork = network.find((row) =>
    row.requestedUrl.includes(LANGUAGE) && row.requestedUrl.includes(LANGUAGE_REVISION) &&
    row.requestedUrl.includes("model_quantized.onnx") && row.status === 200 &&
    row.contentLength === LANGUAGE_BYTES
  );
  check(
    "primary-pinned-network-bytes",
    Boolean(primaryNetwork),
    `Observed exact pinned primary request and ${PRIMARY_BYTES}-byte response.`,
    primaryNetwork || network.filter((row) => row.requestedUrl.includes(PRIMARY)),
  );
  check(
    "language-pinned-network-bytes",
    Boolean(languageNetwork),
    `Observed exact pinned language q8 request and ${LANGUAGE_BYTES}-byte response.`,
    languageNetwork || network.filter((row) => row.requestedUrl.includes(LANGUAGE)),
  );

  if (WRITE_RUN) {
    const desktop = screenshots.filter((row) => row.device === "desktop").map((row) =>
      join(repoRoot, row.path)
    );
    const mobile = screenshots.filter((row) => row.device === "mobile").map((row) =>
      join(repoRoot, row.path)
    );
    for (const [device, files] of [["desktop", desktop], ["mobile", mobile]]) {
      const output = join(SCREEN_DIR, `mosaic-${device}.png`);
      execFileSync("montage", [
        ...files,
        "-thumbnail",
        device === "desktop" ? "320x200" : "180x370",
        "-tile",
        "2x",
        "-geometry",
        "+8+8",
        output,
      ]);
      screenshots.push({
        route: "all",
        device,
        theme: "light+dark",
        path: relative(repoRoot, output),
        kind: "mosaic-of-populated-full-page-captures",
      });
    }
  }

  const implementationCommit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  const record = {
    schemaVersion: 2,
    slug: SLUG,
    commit: implementationCommit,
    implementationCommit,
    runAt: new Date().toISOString(),
    exitCode: failures ? 1 : 0,
    runner: {
      chrome: "Headless Chrome via raw CDP",
      profile: "fresh, then shared across lifecycle and route cells",
      stages: [
        {
          model: PRIMARY,
          revision: PRIMARY_REVISION,
          file: "model.onnx",
          dtype: "fp32",
          bytes: PRIMARY_BYTES,
          sha256: PRIMARY_SHA256,
        },
        {
          model: LANGUAGE,
          revision: LANGUAGE_REVISION,
          file: "onnx/model_quantized.onnx",
          dtype: "q8",
          bytes: LANGUAGE_BYTES,
          sha256: LANGUAGE_SHA256,
        },
      ],
    },
    frozenDenominator: {
      routes: Object.values(ROUTES),
      devices: ["desktop", "mobile"],
      themes: ["light", "dark"],
      routeDeviceCells: 10,
      screenshots: WRITE_RUN ? 22 : 20,
      controls:
        "every page sample, textarea/action, Practical copy+PER/ORG/LOC toggles, every Wild replacement field, all three language cases, every details, both Release and both Clear controls",
      lifecycle: [
        "first visit",
        "network error",
        "Retry",
        "current cached",
        "disposal race",
        "offline cached",
        "eviction/partial",
        "Re-download",
        "corrupt cache",
        "corrupt Retry",
        "clear to absent",
        "immutable Update N/A",
        "feature unsupported N/A",
      ],
    },
    summary: {
      assertions: assertions.length,
      passed: assertions.filter((row) => row.state === "pass").length,
      failed: failures,
      routeDeviceCells: results.length,
      screenshots: WRITE_RUN ? 22 : screenshots.length,
    },
    assertions,
    lifecycle,
    results,
    screenshots,
    network: {
      primary: primaryNetwork || null,
      language: languageNetwork || null,
      expected: {
        [PRIMARY]: { revision: PRIMARY_REVISION, bytes: PRIMARY_BYTES, sha256: PRIMARY_SHA256 },
        [LANGUAGE]: { revision: LANGUAGE_REVISION, bytes: LANGUAGE_BYTES, sha256: LANGUAGE_SHA256 },
      },
      failedClosed: true,
    },
    residualRisks: [
      "Mobile is Chrome device emulation on desktop hardware, not a physical low-memory phone.",
      "WebCrypto SHA-256 validates the primary before use/cache and on cache hits; Transformers.js owns language-stage cache validation, while its immutable revision and exact upstream q8 identity are pinned and network-attested here.",
      "Screenshots are automated populated evidence for independent visual review, not self-acceptance.",
      "Immutable revisions make mutable stale-Update honestly not applicable; baseline WASM makes feature-gated unsupported honestly not applicable.",
    ],
  };
  if (WRITE_RUN) writeFileSync(RUN_FILE, JSON.stringify(record, null, 2) + "\n");
  console.log(
    `\nRESULT ${record.summary.passed}/${record.summary.assertions}; cells=${results.length}; screenshots=${record.summary.screenshots}`,
  );
  if (failures) process.exitCode = 1;
} finally {
  server?.close();
  if (chrome) await chrome.kill();
  if (!WRITE_RUN) rmSync(TEMP_ROOT, { recursive: true, force: true });
}
