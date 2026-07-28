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

export class ModelCacheTimeoutError extends Error {
  constructor(operation, timeoutMs) {
    super(`${operation} did not finish within ${timeoutMs} ms`);
    this.name = "ModelCacheTimeoutError";
    this.operation = operation;
    this.timeoutMs = timeoutMs;
  }
}

/** Bound browser storage APIs that can otherwise remain pending indefinitely. */
export function settleWithin(promise, timeoutMs, operation = "Model cache check") {
  let timer;
  return Promise.race([
    Promise.resolve(promise),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new ModelCacheTimeoutError(operation, timeoutMs)), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timer));
}

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

/** Every successfully validated model variant recorded for this origin. */
export async function listValidationRecords() {
  const records = await idb("readonly", (s) => {
    const r = s.getAll();
    const box = {};
    r.onsuccess = () => (box._result = r.result ?? []);
    return box;
  });
  return Array.isArray(records) ? records : [];
}

async function putRecord(rec) {
  return idb("readwrite", (s) => s.put(rec));
}

/** URLs of a model's files currently present in ANY Cache Storage cache. Slow-path/background only. */
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

/**
 * Fast path for a validated model: probe only its recorded URLs, in parallel. Never enumerate every
 * request in every origin cache during the user-facing "local copy" check.
 */
async function checkRecordedFiles(urls, timeoutMs) {
  const expected = [...new Set((urls ?? []).filter(Boolean))];
  if (!("caches" in self) || expected.length === 0) return { present: [], missing: expected };
  const matches = await settleWithin(
    Promise.all(expected.map((url) => caches.match(url))),
    timeoutMs,
    "Recorded model file check",
  );
  const present = [];
  const missing = [];
  matches.forEach((response, index) => (response ? present : missing).push(expected[index]));
  return { present, missing };
}

/** Live HF repo revision (main commit sha) — the "version" signal. Null when offline/unreachable. */
export async function remoteRevision(modelId, timeoutMs = 4000) {
  if (revisionCache.has(modelId)) return revisionCache.get(modelId);
  let sha = null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`https://huggingface.co/api/models/${modelId}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (res.ok) sha = (await res.json()).sha ?? null;
  } catch {
    sha = null; // offline/slow: treat cached as current so offline use keeps working.
  } finally {
    clearTimeout(timer);
  }
  revisionCache.set(modelId, sha);
  return sha;
}

/**
 * Inspect on-device availability of a model using local metadata and exact cache keys only.
 * @returns {Promise<{state:'current'|'partial'|'absent', record?:object,
 *   missing?:string[], cachedFiles:number}>}
 */
export async function inspectModel({ key, timeoutMs = 750 }) {
  // LOCAL ONLY. Remote revision discovery is intentionally a separate background operation so a slow
  // network can never hold the demo on "Checking for a local copy…".
  const record = await settleWithin(getRecord(key), timeoutMs, "Local validation record check");
  if (!record) return { state: "absent", cachedFiles: 0 };

  // Older/non-cache-backed runtimes may have a successful validation record but no Cache Storage URLs.
  // Preserve their existing auto-init behaviour; the runtime itself remains the final validation.
  const expected = record.files ?? [];
  if (expected.length === 0) return { state: "current", record, cachedFiles: 0 };

  const { present, missing } = await checkRecordedFiles(expected, timeoutMs);
  if (missing.length > 0) {
    return { state: "partial", record, missing, cachedFiles: present.length };
  }
  return { state: "current", record, cachedFiles: present.length };
}

/** Check for a newer remote revision after local availability has settled; never blocks local use. */
export async function checkModelUpdate({ modelId, record, timeoutMs = 4000 }) {
  if (!record?.revision) return null;
  const remote = await remoteRevision(modelId, timeoutMs);
  if (!remote || remote === record.revision) return null;
  return {
    state: "update",
    record,
    cachedRevision: record.revision,
    remoteRevision: remote,
    cachedFiles: record.files?.length ?? 0,
  };
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

/** Cached request metadata without reading large response bodies into memory. */
export async function scanCacheInventory() {
  if (!("caches" in self)) return [];
  const entries = [];
  for (const cacheName of await caches.keys()) {
    const cache = await caches.open(cacheName);
    for (const request of await cache.keys()) {
      const response = await cache.match(request);
      const length = response?.headers.get("content-length") ??
        response?.headers.get("x-linked-size");
      entries.push({
        cacheName,
        url: request.url,
        bytes: /^\d+$/.test(length ?? "") ? Number(length) : null,
      });
    }
  }
  return entries;
}

/** Delete exact cached requests, leaving app-shell and unrelated library entries untouched. */
export async function clearCachedUrls(urls) {
  if (!("caches" in self)) return 0;
  const wanted = new Set(urls);
  let removed = 0;
  for (const cacheName of await caches.keys()) {
    const cache = await caches.open(cacheName);
    for (const request of await cache.keys()) {
      if (wanted.has(request.url) && await cache.delete(request)) removed++;
    }
  }
  return removed;
}

/** Delete all validation records for one model id. */
export async function clearModelRecords(modelId) {
  const records = await listValidationRecords();
  let removed = 0;
  for (const record of records) {
    if (record.modelId === modelId) {
      await idb("readwrite", (s) => s.delete(record.key));
      removed++;
    }
  }
  return removed;
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
export async function storageEstimate(timeoutMs = 1500) {
  try {
    if (navigator.storage?.estimate) {
      return await settleWithin(
        navigator.storage.estimate(),
        timeoutMs,
        "Storage estimate",
      );
    }
  } catch { /* unavailable or slow: storage detail is optional */ }
  return null;
}
