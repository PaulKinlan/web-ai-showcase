#!/usr/bin/env node
// Real-browser integration: an HF-shaped large-file request follows a local 302 to a foreign
// origin; the downloader must reject before writing response bytes to IndexedDB. No model download.
import { createServer } from "node:http";
import { CDP, closePage, evalValue, launchChrome, openPage, startServer } from "./browser.mjs";

const foreign = createServer((req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Range, If-Range",
    });
    return res.end();
  }
  res.writeHead(200, {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/octet-stream",
    "Content-Length": "4",
  });
  res.end("evil");
});
const source = createServer((req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Range, If-Range",
    });
    return res.end();
  }
  res.writeHead(302, {
    "Access-Control-Allow-Origin": "*",
    Location: `http://127.0.0.1:${foreign.address().port}/foreign`,
  });
  res.end();
});
const listen = async (server) => {
  server.listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
};
const close = (server) => new Promise((resolve) => server.close(resolve));
let app;
let chrome;
let cdp;
let page;
try {
  await listen(foreign);
  await listen(source);
  app = await startServer();
  chrome = await launchChrome();
  cdp = new CDP(chrome.ws);
  page = await openPage(cdp, `http://127.0.0.1:${app.port}/web-ai-showcase/`);
  const requested = "https://huggingface.co/example/model/resolve/main/onnx/model.onnx";
  const result = await evalValue(
    cdp,
    page.sessionId,
    `(async () => {
    const {downloadModelFile, resumeState, clearPartial} = await import('/web-ai-showcase/lib/model-download.js');
    const url = ${JSON.stringify(requested)};
    const actualFetch = globalThis.fetch.bind(globalThis);
    await clearPartial(url);
    globalThis.fetch = (input, init) => {
      if (String(input) === url) return actualFetch(${
      JSON.stringify(`http://127.0.0.1:${source.address().port}/redirect`)
    }, init);
      if (String(input).includes('/paths-info/')) {
        return Promise.resolve({ok:false, url:String(input), type:'basic'});
      }
      return actualFetch(input, init);
    };
    try {
      let error = null;
      try { await downloadModelFile({url}); } catch (e) { error = String(e); }
      const state = await resumeState(url);
      return {error, persistedBytes: state?.receivedBytes || 0};
    } finally { globalThis.fetch = actualFetch; await clearPartial(url); }
  })()`,
    20000,
  );
  if (
    !result?.error?.includes("Untrusted model download redirect") || result.persistedBytes !== 0
  ) {
    throw new Error(`Foreign final origin was accepted/persisted: ${JSON.stringify(result)}`);
  }
  console.log("PASS browser: foreign 302 rejected before IndexedDB body persistence", result);
} finally {
  if (page && cdp) await closePage(cdp, page.targetId).catch(() => {});
  chrome?.kill();
  app?.server.close();
  if (source.listening) await close(source);
  if (foreign.listening) await close(foreign);
}
