// Service worker — offline app shell + resilient serving of large model blobs.
// Adapted from image-embedding-lab's sw.js. Bump SW_VERSION on every deploy.
//
// CACHE OWNERSHIP (learned the hard way in image-embedding-lab):
// - Transformers.js keeps its OWN Cache Storage cache ('transformers-cache') and writes every model
//   file there — including loads the SW never sees (first visit before control, hard reloads). NEVER
//   delete or duplicate it. Serve model files via caches.match() across ALL caches; do not re-store
//   another copy. Double-storing ~GB models is what pushes an origin into quota eviction.
// - We only cache the small app shell (HTML/CSS/JS/JSON) ourselves.
const SW_VERSION = "2026-07-20-home-search-v2";
const SHELL_CACHE = `webai-shell-${SW_VERSION}`;
const MODEL_HOSTS = ["huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co"];
const LIB_HOSTS = ["cdn.jsdelivr.net"];

self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(SHELL_CACHE).catch(() => {}));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          // Delete ONLY our own outdated shell caches. transformers-cache and any library cache are
          // load-bearing model storage — never touch them.
          keys.filter((k) => k.startsWith("webai-shell-") && k !== SHELL_CACHE).map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // Model blobs (HF) + the library JS (jsDelivr): cache-first across ALL caches, never double-store.
  if (MODEL_HOSTS.includes(url.hostname) || LIB_HOSTS.includes(url.hostname)) {
    e.respondWith(
      caches.match(req).then(
        (hit) =>
          hit ||
          fetch(req).then((res) => {
            // Only WE store the small library JS; HF model files are the library's job.
            if (res.ok && LIB_HOSTS.includes(url.hostname)) {
              const copy = res.clone();
              caches.open(SHELL_CACHE).then((c) => c.put(req, copy)).catch(() => {});
            }
            return res;
          }),
      ),
    );
    return;
  }

  // Same-origin app shell: network-first, fall back to cache offline.
  if (url.origin === self.location.origin) {
    e.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(SHELL_CACHE).then((c) => c.put(req, copy)).catch(() => {});
          }
          return res;
        })
        .catch(() => caches.match(req)),
    );
  }
});
