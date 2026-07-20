// model-prefetch.mjs — RESUMABLE, cache-compatible prefetch of a Transformers.js model's files.
//
// WHY: Transformers.js 3.7.5 cannot issue Range requests, cannot be aborted, and never caches partial
// (206) responses — so it can't resume a 2.9 GB PaliGemma download on its own (verified against the
// 3.7.5 source: it only reads `transformers-cache` Cache Storage by the exact remote-URL key, and on a
// cache MISS does a plain un-resumable `fetch`). This module makes resume REAL without patching the
// library and without double-downloading or ~2× storage:
//
//   1. Resolve the required files' exact remote URLs + byte sizes + git-LFS sha256 (HF paths-info).
//   2. For each file, if it is already in `transformers-cache` under the exact key → it's cached; skip.
//   3. Otherwise download it RESUMABLY (lib/model-download.js: Range/206/If-Range→sha256-verified blob;
//      partials persist in IndexedDB and survive reload/abort), then `cache.put(exactURL, Response)`
//      into `transformers-cache` with a Content-Length header (required so the library's own progress
//      reports a correct total), and free the partial → net ~1× storage (transient peak ≈ largest file).
//   4. from_pretrained() then hits `cache.match(exactURL)` for every file → NO network, no re-download,
//      and lib/model-cache.js's returning-visit / "clear cached model" logic keeps working unchanged.
//
// This runs in the WORKER (fetch + WebCrypto sha256 = integrity work off the main thread). It emits
// events shaped for lib/download-tracker.mjs. Deps are injectable so the orchestration is unit-testable
// without a network or a browser.

// The exact Cache Storage key Transformers.js 3.7.5 uses: env.remoteHost + '{model}/resolve/{rev}/{file}'.
export function assetUrl(modelId, revision, path) {
  // encodeURIComponent(revision) mirrors Transformers.js 3.7.5 exactly (hub.js builds the key that way);
  // identical for "main", only diverges for exotic revisions.
  return `https://huggingface.co/${modelId}/resolve/${encodeURIComponent(revision)}/${path}`;
}

const isLarge = (bytes) => typeof bytes === "number" && bytes > 1_000_000; // >1 MB ⇒ resumable path

/**
 * @param {object} o
 * @param {string} o.modelId                    e.g. "onnx-community/paligemma2-3b-pt-224"
 * @param {string[]} o.files                    repo-relative paths (e.g. ["config.json","onnx/…q4f16.onnx"])
 * @param {string} [o.revision="main"]
 * @param {(evt:object)=>void} o.onEvent        tracker events: {status:"initiate"|"progress"|"done"|"error"|"file-verifying", file, loaded?, total?, message?}
 * @param {AbortSignal} [o.signal]              aborting keeps partials → a later call resumes
 * @param {object} [o.deps]                     injectable for tests: {resolveInfo, cacheOpen, download, existsInCache}
 * @returns {Promise<{cached:string[], downloaded:string[], totalBytes:number}>}
 */
export async function prefetchModel(
  { modelId, files, revision = "main", onEvent = () => {}, signal, deps = {} },
) {
  const resolveInfo = deps.resolveInfo || resolveInfoHF;
  const cacheOpen = deps.cacheOpen || (() => caches.open("transformers-cache"));
  const download = deps.download || defaultDownload;
  const existsInCache = deps.existsInCache ||
    (async (cache, url) => !!(await cache.match(url)));

  const cache = await cacheOpen();

  // 1) Resolve exact URLs + sizes (+ sha256) so the tracker has a COMPLETE denominator up front — no
  //    discovery flicker, and late per-file callbacks can't cause a bogus 100→0.
  const infos = await resolveInfo({ modelId, revision, files, signal });
  for (const info of infos) {
    onEvent({ status: "initiate", file: info.file, total: info.size ?? null });
  }
  const totalBytes = infos.reduce((n, i) => n + (i.size || 0), 0);

  const cached = [];
  const downloaded = [];

  // 2) Process each file. (Sequential keeps peak transient storage to ~one large file; the library will
  //    still load the components concurrently afterwards, from cache.)
  for (const info of infos) {
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    const url = info.url;

    if (await existsInCache(cache, url)) {
      onEvent({ status: "done", file: info.file, total: info.size ?? null, cached: true });
      cached.push(info.file);
      continue;
    }

    if (!isLarge(info.size)) {
      // Small file (JSON/tokenizer config): a plain fetch is fine; store it under the exact key so the
      // library hits cache too. No resume needed for tiny files.
      onEvent({ status: "download", file: info.file, total: info.size ?? null });
      const res = deps.simpleFetch
        ? await deps.simpleFetch(url, signal)
        : await fetch(url, { signal });
      if (!res.ok) {
        onEvent({ status: "error", file: info.file, message: `HTTP ${res.status}` });
        throw new Error(`Failed ${info.file}: HTTP ${res.status}`);
      }
      const blob = await res.blob();
      await cache.put(
        url,
        new Response(blob, {
          status: 200,
          headers: {
            "Content-Length": String(blob.size),
            "Content-Type": res.headers.get("Content-Type") || "application/octet-stream",
          },
        }),
      );
      onEvent({ status: "progress", file: info.file, loaded: blob.size, total: blob.size });
      onEvent({ status: "done", file: info.file, total: blob.size });
      downloaded.push(info.file);
      continue;
    }

    // 3) Large file: resumable download + integrity, then cache-populate under the exact key.
    const { blob, total } = await download({
      url,
      signal,
      onProgress: (p) =>
        onEvent({
          status: "progress",
          file: info.file,
          loaded: p.receivedBytes,
          total: p.total ?? info.size ?? null,
        }),
      onEvent: (e) => {
        if (e.type === "verifying") {
          onEvent({ status: "file-verifying", file: info.file, total: info.size ?? null });
        }
      },
    });
    await cache.put(
      url,
      new Response(blob, {
        status: 200,
        headers: {
          "Content-Length": String(total ?? blob.size), // REQUIRED: else the library's progress total = 0
          "Content-Type": "application/octet-stream",
          "Accept-Ranges": "bytes",
          ...(info.oid ? { ETag: `"${info.oid}"` } : {}),
        },
      }),
    );
    onEvent({ status: "done", file: info.file, total: total ?? blob.size });
    downloaded.push(info.file);
  }

  return { cached, downloaded, totalBytes };
}

// ── default dep: resolve sizes + sha256 via the HF paths-info API (one batched call) ──────────────────
async function resolveInfoHF({ modelId, revision, files, signal }) {
  let sizes = new Map();
  let oids = new Map();
  try {
    const res = await fetch(
      `https://huggingface.co/api/models/${modelId}/paths-info/${encodeURIComponent(revision)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: files, expand: false }),
        signal,
      },
    );
    if (res.ok) {
      const arr = await res.json();
      if (Array.isArray(arr)) {
        for (const e of arr) {
          if (!e?.path) continue;
          const size = e.size ?? e.lfs?.size ?? null;
          if (typeof size === "number") sizes.set(e.path, size);
          const oid = e.lfs?.oid;
          if (typeof oid === "string" && /^[0-9a-f]{64}$/i.test(oid)) {
            oids.set(e.path, oid.toLowerCase());
          }
        }
      }
    }
  } catch {
    /* offline / API down → sizes unknown; the tracker degrades to "known so far" honestly */
  }
  return files.map((path) => ({
    file: path,
    url: assetUrl(modelId, revision, path),
    size: sizes.get(path) ?? null,
    oid: oids.get(path) ?? null,
  }));
}

// ── default dep: the repo's resumable downloader ──────────────────────────────────────────────────────
async function defaultDownload(args) {
  const { downloadModelFile } = await import("./model-download.js");
  return downloadModelFile(args);
}
