// Shared model availability / cache-integrity / version layer.
//
// The rule (Paul's architecture mandate): if a valid CURRENT version of a model is already on-device
// — either exposed by a browser-native runtime or already downloaded + validated in the local cache —
// the demo initialises it AUTOMATICALLY. We only ask the user to Download (absent) or Update (a newer
// version exists than the validated cached one). We never silently re-download a large model, and we
// distinguish current / stale / partial / evicted assets, verifying integrity before "ready".
//
// How it works: Transformers.js and WebLLM store model files in Cache Storage. We scan caches for a
// model's files, and keep a per-model VALIDATION RECORD in IndexedDB (the files that were present when
// a load last succeeded, plus the HF repo revision). On revisit we compare recorded files vs what's
// still cached (eviction => missing => partial) and the recorded revision vs the live HF revision
// (differs => update available). Cache Storage entries are atomic (the browser never stores a partial
// response), so "corrupt" reduces to "some entries evicted" = partial.

const DB_NAME = "web-ai-showcase";
const STORE = "model-validations";
const revisionCache = new Map();

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "key" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idb(mode, fn) {
  try {
    const db = await openDB();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const store = tx.objectStore(STORE);
      const out = fn(store);
      // getRecord() returns a box {_result}; unwrap it even when the value is legitimately null
      // (a missing key). Using `?? out` here wrongly returned the truthy box for null results, which
      // made inspectModel() treat a never-seen model as "current" and auto-download on first visit.
      tx.oncomplete = () =>
        resolve(out && typeof out === "object" && "_result" in out ? out._result : out);
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    return null; // IDB unavailable (private mode etc.) — degrade to "unverified" behaviour.
  }
}

export async function getRecord(key) {
  return idb("readonly", (s) => {
    const r = s.get(key);
    const box = {};
    r.onsuccess = () => (box._result = r.result ?? null);
    return box;
  });
}

async function putRecord(rec) {
  return idb("readwrite", (s) => s.put(rec));
}

/** URLs of a model's files currently present in ANY Cache Storage cache. */
export async function scanCachedFiles(modelId) {
  if (!("caches" in self)) return [];
  const needle = `/${modelId}/`;
  const found = [];
  for (const name of await caches.keys()) {
    const cache = await caches.open(name);
    for (const req of await cache.keys()) {
      if (req.url.includes(needle) || req.url.includes(encodeURIComponent(modelId))) {
        found.push(req.url);
      }
    }
  }
  return found;
}

/** Live HF repo revision (main commit sha) — the "version" signal. Null when offline/unreachable. */
export async function remoteRevision(modelId) {
  if (revisionCache.has(modelId)) return revisionCache.get(modelId);
  let sha = null;
  try {
    const res = await fetch(`https://huggingface.co/api/models/${modelId}`, { cache: "no-store" });
    if (res.ok) sha = (await res.json()).sha ?? null;
  } catch {
    sha = null; // offline: treat cached as current so offline use keeps working.
  }
  revisionCache.set(modelId, sha);
  return sha;
}

/**
 * Inspect on-device availability of a model.
 * @returns {Promise<{state:'current'|'update'|'partial'|'unverified'|'absent', record?:object,
 *   missing?:string[], cachedRevision?:string, remoteRevision?:string, cachedFiles:number}>}
 */
export async function inspectModel({ key, modelId }) {
  const [record, cachedNow] = await Promise.all([getRecord(key), scanCachedFiles(modelId)]);
  if (!record && cachedNow.length === 0) return { state: "absent", cachedFiles: 0 };
  if (!record && cachedNow.length > 0) {
    // Files exist but we never validated them — attempt an auto-init that verifies (see loader).
    return { state: "unverified", cachedFiles: cachedNow.length };
  }
  const present = new Set(cachedNow);
  const missing = (record.files ?? []).filter((u) => !present.has(u));
  if (missing.length > 0) {
    return { state: "partial", record, missing, cachedFiles: cachedNow.length };
  }
  const remote = await remoteRevision(modelId);
  if (remote && record.revision && remote !== record.revision) {
    return {
      state: "update",
      record,
      cachedRevision: record.revision,
      remoteRevision: remote,
      cachedFiles: cachedNow.length,
    };
  }
  return { state: "current", record, cachedFiles: cachedNow.length };
}

/** Record that a model loaded successfully — capture its cached files + the current revision. */
export async function recordValidated({ key, modelId, runtime, dtype }) {
  const [files, revision] = await Promise.all([scanCachedFiles(modelId), remoteRevision(modelId)]);
  await putRecord({
    key,
    modelId,
    runtime,
    dtype,
    revision,
    files,
    validatedAt: new Date().toISOString(),
    fileCount: files.length,
  });
}

/** Delete a model's cached files + its validation record (the per-model "clear cache" control). */
export async function clearModelCache(modelId, key) {
  let removed = 0;
  if ("caches" in self) {
    const needle = `/${modelId}/`;
    for (const name of await caches.keys()) {
      const cache = await caches.open(name);
      for (const req of await cache.keys()) {
        if (req.url.includes(needle) || req.url.includes(encodeURIComponent(modelId))) {
          if (await cache.delete(req)) removed++;
        }
      }
    }
  }
  if (key) await idb("readwrite", (s) => s.delete(key));
  revisionCache.delete(modelId);
  return removed;
}

/** Approximate origin storage usage/quota, for the storage/cache UI. */
export async function storageEstimate() {
  try {
    if (navigator.storage?.estimate) return await navigator.storage.estimate();
  } catch { /* ignore */ }
  return null;
}
