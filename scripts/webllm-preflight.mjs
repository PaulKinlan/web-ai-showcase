// webllm-preflight.mjs — does a family's model LOAD at all? (coord's preflight-first rule, Phase 1)
//
// The sweep has now produced four families blocked by four different runtime defects, and every one of
// them cost a full cycle of family code before it surfaced. This probe answers the cheapest question
// first: open the page in the real WebGPU-harness browser, watch the loader, and report whether the
// engine accepts the model, refuses it by name, or never finishes — before any family work happens.
//
//   node scripts/webllm-preflight.mjs models/<slug>/ [seconds]
import { CDP, launchChrome, openPage, startServer } from "./browser.mjs";

const route = process.argv[2];
const budgetS = Number(process.argv[3] ?? 900);
if (!route) {
  console.error("usage: node scripts/webllm-preflight.mjs models/<slug>/ [seconds]");
  process.exit(2);
}

const { server, port } = await startServer();
const chrome = await launchChrome({ webgpu: true });
const cdp = new CDP(chrome.ws);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  const page = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/${route}`);
  const started = Date.now();
  let last = { loader: "", status: "" };
  let outcome = "UNRESOLVED";
  while (Date.now() - started < budgetS * 1000) {
    const { result } = await cdp.send(
      "Runtime.evaluate",
      {
        expression: `JSON.stringify({
          loader: (document.querySelector('#model-loader')?.textContent ?? '').replace(/\\s+/g, ' ').trim().slice(0, 260),
          status: (document.querySelector('#status')?.textContent ?? '').replace(/\\s+/g, ' ').trim().slice(0, 260),
          inputDisabled: document.querySelector('#input')?.disabled ?? null,
        })`,
        returnByValue: true,
      },
      page.sessionId,
      20_000,
    );
    const state = JSON.parse(result.value);
    last = state;
    // Some family pages do NOT auto-download: they show their loader's own "Download model (~N MB)"
    // control and wait for a click (this probe stalled 600s on gemma-2-2b because of it). Drive the
    // page's own affordance, the same way the validators' ensureReady does.
    await cdp.send(
      "Runtime.evaluate",
      {
        expression: `(() => {
          let clicks = 0;
          for (const b of document.querySelectorAll('#model-loader button, [id$="-loader"] button')) {
            if (/download|retry|re-download/i.test(b.textContent || '') && !b.disabled) { b.click(); clicks++; }
          }
          return clicks;
        })()`,
        returnByValue: true,
      },
      page.sessionId,
      10_000,
    );
    const text = `${state.loader} ${state.status}`;
    const refused = /Model initialisation failed|window_size|ERROR_CODE|no GPU adapter|needs WebGPU|refuse/i.test(text);
    if (refused) {
      outcome = "REFUSED";
      console.log(`PREFLIGHT ${route}: REFUSED after ${Math.round((Date.now() - started) / 1000)}s — ${text.slice(0, 220)}`);
      break;
    }
    if (state.inputDisabled === false) {
      outcome = "LOADS";
      console.log(`PREFLIGHT ${route}: LOADS after ${Math.round((Date.now() - started) / 1000)}s — ${text.slice(0, 200)}`);
      break;
    }
    await wait(5_000);
  }
  if (outcome === "UNRESOLVED") {
    console.log(`PREFLIGHT ${route}: UNRESOLVED within ${budgetS}s — loader: ${last.loader.slice(0, 200)}`);
  }
} finally {
  try {
    server.close();
  } catch { /* ignore */ }
  try {
    chrome.kill();
  } catch { /* ignore */ }
  process.exit(0);
}
