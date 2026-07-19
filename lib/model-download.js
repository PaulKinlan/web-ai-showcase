// Resumable, integrity-verified Hugging Face model-file download — a NEW, INDEPENDENT capability.
//
// This does NOT replace lib/model-cache.js (which validates transformers.js's own Cache Storage) or the
// auto-init loader. It is a standalone downloader for cases where a demo wants a *raw model file* with a
// real progress bar and true interrupt-resume (valuable for the hundreds-of-MB WebGPU models where a
// restart-from-zero is painful). It returns a verified Blob; the caller decides what to do with it.
//
// Grounded in the proven feasibility probe (scratchpad/hf-download-resume-findings.md, "GO"):
//   • Always fetch the `…/resolve/<rev>/<path>` URL (NOT a signed CDN URL) — the browser follows the 302
//     to the Xet CDN transparently and RE-SENDS the Range header; the final response is CORS-readable.
//   • The CDN returns 206 Partial Content with a correct Content-Range and a STRONG ETag (the Xet content
//     hash) — usable as the `If-Range` validator for a correct conditional resume.
//   • Response byte offsets map 1:1 to file bytes (no Content-Encoding on the binary), so offset math is safe.
//   • The whole-file git-LFS sha256 is exposed as `X-Linked-ETag` on the resolve response and, robustly,
//     via the HF `paths-info` API (`lfs.oid`) — enabling a REAL integrity check on completion.
//   • Storage is best-effort on *.github.io: partials can be evicted. We persist {url,total,etag,sha256,
//     receivedBytes} + the growing blob in IndexedDB, call navigator.storage.persist() opportunistically,
//     and treat an evicted/mismatched partial as a CLEAN RESTART — never a fake resume.
//
// Status handling in the resume loop:
//   • 206 → honoured range: APPEND the streamed bytes to the partial.
//   • 200 → server ignored the range / file changed (or first full download): if we had a partial, DISCARD
//           it and restart from 0, then consume the full body.
//   • 412 / 416 → precondition failed / range unsatisfiable (stale validator or shrunk file): DISCARD the
//           partial and cleanly restart the loop from 0 with no validator.
// On completion we assert receivedBytes === total AND verify sha256 (WebCrypto) before returning "ready".
// A hash mismatch discards the partial and throws — corruption is NEVER presented as success.
//
// modern-web-guidance retained: `performance` (stream progress off the parse/paint path; the digest is a
// one-shot at the end), `deprioritize-background-fetches` (a background pre-fetch may pass priority:'low').
// Primary specs (MDN): fetch Range/If-Range, ReadableStream reader, IndexedDB, crypto.subtle.digest,
// AbortController/AbortSignal, StorageManager.persist/estimate.

const DB_NAME = "web-ai-showcase-downloads";
const DB_VERSION = 1;
const META_STORE = "meta"; // keyPath "url" → {url,total,etag,sha256,receivedBytes,updatedAt}
const CHUNK_STORE = "chunks"; // keyPath "seq" (autoIncrement); index "url"; each {seq,url,blob}
const FLUSH_BYTES = 4 * 1024 * 1024; // buffer ~4MB in memory before persisting a chunk (fewer IDB writes)
const MAX_RESTARTS = 3; // guard against a pathological 200/412/416 loop

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: "url" });
      }
      if (!db.objectStoreNames.contains(CHUNK_STORE)) {
        const s = db.createObjectStore(CHUNK_STORE, { keyPath: "seq", autoIncrement: true });
        s.createIndex("url", "url", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, stores, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode);
    let out;
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new DOMException("Transaction aborted", "AbortError"));
    out = fn(t);
  });
}

function reqP(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getMeta(db, url) {
  return tx(db, META_STORE, "readonly", (t) => reqP(t.objectStore(META_STORE).get(url)))
    .then((r) => r ?? null);
}

async function putMeta(db, meta) {
  meta.updatedAt = new Date().toISOString();
  return tx(db, META_STORE, "readwrite", (t) => t.objectStore(META_STORE).put(meta));
}

/** Append a super-chunk blob for a url (insertion order == byte order via autoIncrement seq). */
async function addChunk(db, url, blob) {
  return tx(db, CHUNK_STORE, "readwrite", (t) => t.objectStore(CHUNK_STORE).add({ url, blob }));
}

/** All chunk blobs for a url, in byte order (ascending seq). */
async function getChunks(db, url) {
  return tx(db, CHUNK_STORE, "readonly", (t) => {
    const idx = t.objectStore(CHUNK_STORE).index("url");
    const blobs = [];
    return new Promise((resolve, reject) => {
      const cursor = idx.openCursor(IDBKeyRange.only(url));
      cursor.onsuccess = () => {
        const c = cursor.result;
        if (c) {
          blobs.push(c.value.blob);
          c.continue();
        } else resolve(blobs);
      };
      cursor.onerror = () => reject(cursor.error);
    });
  });
}

/** Delete every chunk for a url and reset its meta.receivedBytes to 0 (the CLEAN-RESTART primitive). */
async function resetPartial(db, url, meta) {
  await tx(db, CHUNK_STORE, "readwrite", (t) => {
    const idx = t.objectStore(CHUNK_STORE).index("url");
    const cursor = idx.openCursor(IDBKeyRange.only(url));
    cursor.onsuccess = () => {
      const c = cursor.result;
      if (c) {
        c.delete();
        c.continue();
      }
    };
  });
  if (meta) {
    meta.receivedBytes = 0;
    await putMeta(db, meta);
  }
}

/** Remove ALL persisted state for a url (chunks + meta). Public helper. */
export async function clearPartial(url) {
  let db;
  try {
    db = await openDB();
  } catch {
    return;
  }
  const meta = await getMeta(db, url).catch(() => null);
  await resetPartial(db, url, meta).catch(() => {});
  await tx(db, META_STORE, "readwrite", (t) => t.objectStore(META_STORE).delete(url)).catch(
    () => {},
  );
  db.close();
}

/**
 * Current persisted resume state for a url, or null. Does not touch the network.
 * @returns {Promise<null|{url,total:number|null,etag:string|null,sha256:string|null,receivedBytes:number,
 *   ratio:number|null,updatedAt?:string}>}
 */
export async function resumeState(url) {
  let db;
  try {
    db = await openDB();
  } catch {
    return null;
  }
  const meta = await getMeta(db, url).catch(() => null);
  db.close();
  if (!meta) return null;
  return {
    url: meta.url,
    total: meta.total ?? null,
    etag: meta.etag ?? null,
    sha256: meta.sha256 ?? null,
    receivedBytes: meta.receivedBytes || 0,
    ratio: meta.total ? (meta.receivedBytes || 0) / meta.total : null,
    updatedAt: meta.updatedAt,
  };
}

function parseResolveUrl(url) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (u.host !== "huggingface.co") return null;
  // /{repo…}/resolve/{revision}/{path…}
  const m = u.pathname.match(/^\/(.+?)\/resolve\/([^/]+)\/(.+)$/);
  if (!m) return null;
  let repo = m[1];
  let repoType = "models";
  if (repo.startsWith("datasets/")) {
    repoType = "datasets";
    repo = repo.slice("datasets/".length);
  } else if (repo.startsWith("spaces/")) {
    repoType = "spaces";
    repo = repo.slice("spaces/".length);
  }
  return { repoType, repo, revision: decodeURIComponent(m[2]), path: decodeURIComponent(m[3]) };
}

/** The whole-file git-LFS sha256 for an HF resolve URL, via the paths-info API. Null if unavailable
 *  (non-HF url, non-LFS small file, or offline) → the caller falls back to size-only verification. */
async function fetchLinkedSha256(url, signal) {
  const p = parseResolveUrl(url);
  if (!p) return null;
  try {
    const res = await fetch(
      `https://huggingface.co/api/${p.repoType}/${p.repo}/paths-info/${
        encodeURIComponent(p.revision)
      }`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: [p.path], expand: false }),
        signal,
      },
    );
    if (!res.ok) return null;
    const arr = await res.json();
    if (!Array.isArray(arr)) return null;
    const entry = arr.find((e) => e.path === p.path) ?? arr[0];
    const oid = entry?.lfs?.oid;
    return typeof oid === "string" && /^[0-9a-f]{64}$/i.test(oid) ? oid.toLowerCase() : null;
  } catch {
    return null;
  }
}

function totalFromContentRange(res) {
  const cr = res.headers.get("Content-Range"); // "bytes start-end/total"
  if (cr) {
    const m = cr.match(/\/(\d+)\s*$/);
    if (m) return Number(m[1]);
  }
  return null;
}

function strongEtag(res) {
  const et = res.headers.get("ETag");
  if (!et) return null;
  return et.startsWith("W/") ? null : et; // weak validators are useless for If-Range resume
}

async function sha256Hex(blob) {
  const buf = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Consume a response body stream, persisting ~FLUSH_BYTES super-chunks and reporting progress. On any
 *  error (including AbortError) the buffered bytes are flushed first so the partial stays resumable. */
async function streamAppend(db, meta, body, emit) {
  const reader = body.getReader();
  let buffered = [];
  let bufferedBytes = 0;

  const flush = async () => {
    if (bufferedBytes === 0) return;
    const blob = new Blob(buffered);
    await addChunk(db, meta.url, blob);
    meta.receivedBytes += bufferedBytes;
    await putMeta(db, meta);
    buffered = [];
    bufferedBytes = 0;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffered.push(value);
      bufferedBytes += value.byteLength;
      emit({ receivedBytes: meta.receivedBytes + bufferedBytes, total: meta.total });
      if (bufferedBytes >= FLUSH_BYTES) await flush();
    }
    await flush();
  } catch (err) {
    try {
      await flush();
    } catch { /* best effort — keep whatever we already persisted */ }
    throw err;
  }
}

/**
 * Download an HF model file with resume + integrity verification. Returns a VERIFIED Blob.
 *
 * @param {Object} opts
 * @param {string} opts.url                       An HF `…/resolve/<rev>/<path>` URL.
 * @param {(p:{receivedBytes:number,total:number|null,ratio:number|null})=>void} [opts.onProgress]
 * @param {AbortSignal} [opts.signal]             Aborting keeps the partial for a later resume.
 * @param {(e:object)=>void} [opts.onEvent]       Lifecycle hook: {type:"resume-state"|"response"|"restart"|
 *                                                "verifying"|"complete", ...} — useful for diagnostics/tests.
 * @param {"auto"|"high"|"low"} [opts.priority]   fetch() priority hint (default "auto").
 * @returns {Promise<{blob:Blob, total:number, sha256:string|null, sha256Verified:boolean, resumed:boolean}>}
 */
export async function downloadModelFile({ url, onProgress, signal, onEvent, priority = "auto" }) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  // Best-effort durable storage so a large partial is less likely to be evicted mid-download.
  try {
    if (navigator.storage?.persist && !(await navigator.storage.persisted?.())) {
      await navigator.storage.persist();
    }
  } catch { /* best effort */ }

  const db = await openDB();
  try {
    let meta = await getMeta(db, url);
    if (!meta) {
      meta = { url, total: null, etag: null, sha256: null, receivedBytes: 0 };
    } else {
      // Trust the recorded byte length only as far as the persisted chunks actually cover it. If the
      // browser evicted the chunk store but kept meta (or vice-versa), reconcile to the real bytes so we
      // never claim a resume we can't back with data — an HONEST fallback, not a fake resume.
      const chunks = await getChunks(db, url).catch(() => []);
      const realBytes = chunks.reduce((n, b) => n + b.size, 0);
      if (realBytes !== (meta.receivedBytes || 0)) {
        meta.receivedBytes = realBytes;
        if (realBytes === 0) {
          meta.etag = null; // nothing to validate against → fresh start
        }
        await putMeta(db, meta);
      }
    }
    onEvent?.({ type: "resume-state", receivedBytes: meta.receivedBytes, total: meta.total });
    const startedFrom = meta.receivedBytes;

    // Obtain the expected whole-file sha256 up front (if it's an HF LFS file). Independent of the bytes.
    if (!meta.sha256) {
      meta.sha256 = await fetchLinkedSha256(url, signal);
      if (meta.sha256) await putMeta(db, meta);
    }

    const emit = ({ receivedBytes, total }) => {
      onProgress?.({ receivedBytes, total, ratio: total ? receivedBytes / total : null });
    };

    let restarts = 0;
    while (true) {
      if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");

      const headers = {};
      if (meta.receivedBytes > 0) {
        headers["Range"] = `bytes=${meta.receivedBytes}-`;
        if (meta.etag) headers["If-Range"] = meta.etag;
      } else {
        headers["Range"] = "bytes=0-"; // uniform: a 206 gives us Content-Range (→ total) + strong ETag
      }

      const res = await fetch(url, { headers, signal, priority, redirect: "follow" });
      onEvent?.({
        type: "response",
        status: res.status,
        resumed: meta.receivedBytes > 0,
        contentRange: res.headers.get("Content-Range"),
      });

      if (res.status === 206) {
        const t = totalFromContentRange(res);
        if (t != null) meta.total = t;
        const et = strongEtag(res);
        if (et) meta.etag = et;
        await putMeta(db, meta);
        await streamAppend(db, meta, res.body, emit);
      } else if (res.status === 200) {
        // Full response: server ignored the range or the file changed. Discard any partial, restart at 0.
        if (meta.receivedBytes > 0) {
          onEvent?.({ type: "restart", reason: "200-full-response" });
          await resetPartial(db, url, meta);
        }
        const len = Number(res.headers.get("Content-Length"));
        if (Number.isFinite(len) && len > 0) meta.total = len;
        const et = strongEtag(res);
        if (et) meta.etag = et;
        await putMeta(db, meta);
        await streamAppend(db, meta, res.body, emit);
      } else if (res.status === 412 || res.status === 416) {
        // Stale validator / unsatisfiable range → clean restart from scratch.
        onEvent?.({ type: "restart", reason: `status-${res.status}` });
        try {
          await res.body?.cancel();
        } catch { /* ignore */ }
        meta.etag = null;
        await resetPartial(db, url, meta);
        if (++restarts > MAX_RESTARTS) {
          throw new Error(
            `Download restarted too many times (last status ${res.status}) for ${url}`,
          );
        }
        continue;
      } else {
        throw new Error(`Unexpected HTTP ${res.status} downloading ${url}`);
      }

      // Stream drained. If we have the whole file, finish; otherwise the connection ended early and the
      // outer loop resumes with a Range request from the new receivedBytes.
      if (meta.total != null && meta.receivedBytes >= meta.total) break;
      if (meta.total == null) break; // no known total (non-range server) → single 200 consumed fully
    }

    // Assemble + verify BEFORE declaring success.
    const blob = new Blob(await getChunks(db, url));
    if (meta.total != null && blob.size !== meta.total) {
      throw new Error(`Assembled size ${blob.size} ≠ expected ${meta.total} for ${url}`);
    }
    onEvent?.({ type: "verifying", sha256: meta.sha256 });
    let sha256Verified = false;
    if (meta.sha256) {
      const actual = await sha256Hex(blob);
      if (actual !== meta.sha256) {
        await resetPartial(db, url, meta); // corrupt → discard so the next attempt is a clean restart
        throw new Error(`sha256 mismatch for ${url}: got ${actual}, expected ${meta.sha256}`);
      }
      sha256Verified = true;
    }
    onEvent?.({ type: "complete", sha256: meta.sha256, sha256Verified, total: blob.size });

    // Success: free the on-disk partial (avoid double-storing) and return the verified blob.
    await resetPartial(db, url, meta);
    await tx(db, META_STORE, "readwrite", (t) => t.objectStore(META_STORE).delete(url)).catch(
      () => {},
    );

    return {
      blob,
      total: blob.size,
      sha256: meta.sha256,
      sha256Verified,
      resumed: startedFrom > 0,
    };
  } finally {
    db.close();
  }
}
