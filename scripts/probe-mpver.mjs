// Probe: which @mediapipe/tasks-vision version can init GestureRecognizer inside a MODULE worker?
import { writeFileSync } from "node:fs";
import { CDP, closePage, launchChrome, openPage, startServer } from "./browser.mjs";

const versions = process.argv.slice(2);
if (!versions.length) versions.push("0.10.18", "0.10.22", "0.10.21", "0.10.20");

const workerTpl = (ver) =>
  `import { FilesetResolver, GestureRecognizer } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}";
self.onmessage = async () => {
  try {
    const resolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ver}/wasm");
    const rec = await GestureRecognizer.createFromOptions(resolver, { baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task", delegate: "CPU" }, numHands: 1, runningMode: "IMAGE" });
    self.postMessage({ ok: true });
  } catch (e) { self.postMessage({ ok: false, err: String(e && e.message || e) }); }
};`;

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const BASE = "/web-ai-showcase/";
const ctx = await openPage(cdp, `http://127.0.0.1:${port}${BASE}models/gesture-recognizer/`);
const { sessionId } = ctx;

for (const ver of versions) {
  writeFileSync(
    new URL("../models/gesture-recognizer/__probe_worker.js", import.meta.url),
    workerTpl(ver),
  );
  const res = await (async () => {
    const expr =
      `await new Promise((resolve)=>{const w=new Worker('/web-ai-showcase/models/gesture-recognizer/__probe_worker.js?v=${ver}',{type:'module'});const to=setTimeout(()=>resolve({ok:false,err:'timeout'}),60000);w.onmessage=(e)=>{clearTimeout(to);w.terminate();resolve(e.data);};w.onerror=(e)=>{clearTimeout(to);resolve({ok:false,err:'worker.onerror '+(e.message||'')});};w.postMessage(1);})`;
    const { result } = await cdp.send(
      "Runtime.evaluate",
      {
        expression:
          `(async()=>{try{return (${expr});}catch(e){return {ok:false,err:String(e)};}})()`,
        awaitPromise: true,
        returnByValue: true,
      },
      sessionId,
      70000,
    );
    return result?.value;
  })();
  console.log(ver, "=>", JSON.stringify(res));
}
try {
  rmSyncSafe(new URL("../models/gesture-recognizer/__probe_worker.js", import.meta.url));
} catch {}
function rmSyncSafe(u) {
  import("node:fs").then((fs) => fs.rmSync(u, { force: true }));
}
await closePage(cdp, ctx.targetId);
chrome.kill();
server.close();
