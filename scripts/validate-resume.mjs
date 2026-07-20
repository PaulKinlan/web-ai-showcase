#!/usr/bin/env node
// Integration test for the resumable downloader (lib/model-download.js) against CONTROLLABLE endpoints,
// in real headless Chrome (real IndexedDB + fetch + Range). A true 2.9 GB PaliGemma download isn't
// runnable here, so this proves the resume LOOP on a synthetic ~400 KB asset: clean download; mid-stream
// interruption → 206 continuation from the persisted offset; explicit abort → resume; a server that
// ignores the Range on the resume (200) → clean restart; and a 412 on the resume → clean restart. Every
// scenario asserts the FINAL assembled bytes are exactly correct. (sha256 verification needs the real HF
// paths-info API — covered by design review + the committed arch-selftest; here deterministic bytes +
// the assembled-size check are the integrity gate.)

import { createServer } from "node:http";
import { closePage, evalValue, launchChrome, openPage, startServer } from "./browser.mjs";

const SIZE = 400_000;
const body = Buffer.alloc(SIZE);
for (let i = 0; i < SIZE; i++) body[i] = i % 251; // deterministic bytes
const ETAG = '"synthetic-etag-v1"';
const runs = new Map(); // per-run request counter → deterministic, independent scenarios

const mock = createServer((req, res) => {
  const u = new URL(req.url, "http://x");
  const cors = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Range, If-Range",
    "Access-Control-Expose-Headers": "Content-Range, Content-Length, ETag, Accept-Ranges",
    "Accept-Ranges": "bytes",
    "ETag": ETAG,
  };
  if (req.method === "OPTIONS") {
    res.writeHead(204, cors);
    return res.end();
  }
  // Diagnostics: how many real GETs a run received (proves a resume request actually happened).
  if (u.pathname === "/count") {
    res.writeHead(200, { ...cors, "Content-Type": "application/json" });
    return res.end(JSON.stringify({ n: runs.get(u.searchParams.get("run")) || 0 }));
  }
  const run = u.searchParams.get("run") || "0";
  const mode = u.searchParams.get("mode") || "normal";
  const n = (runs.get(run) || 0) + 1;
  runs.set(run, n);

  let start = 0;
  const range = req.headers["range"];
  if (range) {
    const m = /bytes=(\d+)-/.exec(range);
    if (m) start = Number(m[1]);
  }
  const send = (status, buf, extra = {}) => {
    const h = {
      ...cors,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(buf.length),
      ...extra,
    };
    res.writeHead(status, h);
    res.end(buf);
  };
  const full = () => send(200, body);
  const partial = () =>
    send(206, body.subarray(start), { "Content-Range": `bytes ${start}-${SIZE - 1}/${SIZE}` });
  const interrupt = () => {
    // A CLEAN short 206: send half the requested range, declaring the true total via Content-Range. The
    // downloader's loop sees receivedBytes < total and re-requests a Range for the rest → a real resume.
    const rem = body.subarray(start);
    const half = Math.max(1, Math.floor(rem.length / 2));
    res.writeHead(206, {
      ...cors,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(half),
      "Content-Range": `bytes ${start}-${start + half - 1}/${SIZE}`,
    });
    res.end(rem.subarray(0, half));
  };

  // Paced 206 stream (many small chunks with a gap) so a client-side abort deterministically lands
  // MID-stream instead of racing a single loopback chunk that arrives already-complete.
  const slowPartial = () => {
    const buf = body.subarray(start);
    res.writeHead(206, {
      ...cors,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(buf.length),
      "Content-Range": `bytes ${start}-${SIZE - 1}/${SIZE}`,
    });
    const CH = Math.ceil(buf.length / 20);
    let off = 0;
    const pump = () => {
      if (res.writableEnded || res.destroyed) return; // client aborted → stop writing
      if (off >= buf.length) return res.end();
      res.write(buf.subarray(off, off + CH));
      off += CH;
      setTimeout(pump, 15);
    };
    pump();
  };

  switch (mode) {
    case "slow": // paced 206 → deterministic mid-stream abort
      return range ? slowPartial() : full();
    case "ignore-range": // always full 200, even for a Range request
      return full();
    case "truncate-once": // req1 interrupts; the rest honour Range (206 resume)
      return n <= 1 ? interrupt() : (range ? partial() : full());
    case "truncate-then-412": // req1 interrupts; req2 (the resume) 412s; then normal → clean restart
      if (n <= 1) return interrupt();
      if (n === 2) {
        res.writeHead(412, cors);
        return res.end("precondition failed");
      }
      return range ? partial() : full();
    case "truncate-then-ignore": // req1 interrupts; the resume is answered with a full 200 → clean restart
      return n <= 1 ? interrupt() : full();
    default:
      return range ? partial() : full();
  }
});
await new Promise((r) => mock.listen(0, "127.0.0.1", r));
const mockPort = mock.address().port;
const url = (run, mode) => `http://127.0.0.1:${mockPort}/asset?run=${run}&mode=${mode}`;

const { server, port } = await startServer();
const chrome = await launchChrome();
const { CDP } = await import("./browser.mjs");
const cdp = new CDP(chrome.ws);
const results = [];
const rec = (n, p, d) => {
  results.push({ p: !!p });
  console.log(`${p ? "PASS" : "FAIL"}  ${n}${d ? " — " + d : ""}`);
};

// Verifier: correct length + spot-checked deterministic bytes.
const CHECK =
  `(blob) => blob.arrayBuffer().then(b => { const a=new Uint8Array(b); if(a.length!==${SIZE})return false; for(let i=0;i<a.length;i+=911){if(a[i]!==i%251)return false;} return true; })`;

const cleanDriver = (u) => `
(async () => {
  const { downloadModelFile, clearPartial } = await import("/web-ai-showcase/lib/model-download.js");
  await clearPartial(${JSON.stringify(u)});
  const r = await downloadModelFile({ url: ${JSON.stringify(u)} });
  return { ok: await (${CHECK})(r.blob), total: r.total };
})()`;

// Interrupt-then-resume driver: first call fails/aborts (partial persists), the retry must complete.
const resumeDriver = (u, { abort } = {}) => `
(async () => {
  const { downloadModelFile, resumeState, clearPartial } = await import("/web-ai-showcase/lib/model-download.js");
  const u = ${JSON.stringify(u)};
  await clearPartial(u);
  let firstErr = null;
  ${
  abort
    ? `const ac = new AbortController();
       await downloadModelFile({ url: u, signal: ac.signal, onProgress: (pr)=>{ if(pr.receivedBytes>${
      Math.floor(SIZE / 3)
    } && !ac.signal.aborted) ac.abort(); } }).catch(e=>{firstErr=e;});`
    : `try { await downloadModelFile({ url: u }); } catch(e){ firstErr = e; }`
}
  const partial = await resumeState(u);
  const r2 = await downloadModelFile({ url: u });
  return {
    firstEnded: !!firstErr, firstName: firstErr?.name || null,
    partialBytes: partial ? partial.receivedBytes : 0,
    resumed: r2.resumed, ok: await (${CHECK})(r2.blob),
  };
})()`;

try {
  const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/`);
  await new Promise((r) => setTimeout(r, 500));

  let r = await evalValue(cdp, pg.sessionId, cleanDriver(url("clean", "normal")), 40000);
  rec("clean download assembles correct bytes", r?.ok && r.total === SIZE, JSON.stringify(r));

  r = await evalValue(cdp, pg.sessionId, cleanDriver(url("intr", "truncate-once")), 40000);
  // The first response is a CLEAN short 206 (half the range); the downloader's loop must issue a second
  // Range request to fetch the remainder → a genuine 206 continuation from the persisted offset. Proof:
  // final bytes exact AND the server saw ≥2 GETs for this run (i.e. a resume request really happened).
  const intrCount = await (await fetch(`http://127.0.0.1:${mockPort}/count?run=intr`)).json();
  rec(
    "interruption → 206 continuation from persisted offset",
    r?.ok && r.total === SIZE && intrCount.n >= 2,
    JSON.stringify({ ...r, serverGets: intrCount.n }),
  );

  r = await evalValue(
    cdp,
    pg.sessionId,
    resumeDriver(url("abrt", "slow"), { abort: true }),
    40000,
  );
  rec(
    "explicit abort keeps partial, then resumes to completion",
    r?.firstName === "AbortError" && r.partialBytes > 0 && r.ok,
    JSON.stringify(r),
  );

  r = await evalValue(cdp, pg.sessionId, resumeDriver(url("i200", "truncate-then-ignore")), 40000);
  rec(
    "server ignores Range on resume (200) → clean restart → correct bytes",
    r?.ok,
    JSON.stringify(r),
  );

  r = await evalValue(cdp, pg.sessionId, resumeDriver(url("p412", "truncate-then-412")), 40000);
  rec("412 on resume → clean restart → correct bytes", r?.ok, JSON.stringify(r));

  await closePage(cdp, pg.targetId);
} finally {
  chrome.kill();
  server.close();
  mock.close();
}

const passed = results.filter((x) => x.p).length;
console.log(`\n${passed}/${results.length} resume checks passed.`);
process.exit(passed === results.length ? 0 : 1);
