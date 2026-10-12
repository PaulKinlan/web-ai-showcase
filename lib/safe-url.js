// lib/safe-url.js — the one URL policy for externally sourced attribution links.
// Only explicit http(s) URLs or same-origin relative paths may become a live href; everything else
// (javascript:, data:, vbscript:, protocol-relative, backslash tricks, empty, malformed) returns null
// and the caller must render plain text. The scheme is checked on the PARSED URL, so tab/newline/case
// obfuscation is normalised away before the check.
// lib/example-gallery.js, public/image-credit.js and image-credits/ all use this single copy.
export function safeSourceHref(raw, base) {
  if (typeof raw !== "string" || !raw.trim()) return null;
  const value = raw.trim();
  try {
    // The ambient DOM base is only a DEFAULT — web-ai-showcase-94s: `document` does not exist in a
    // Worker, a service worker or Node, and reading it there threw a ReferenceError that the catch
    // below turned into null even for a URL that is already absolute and needs no base at all.
    const domBase = typeof document === "undefined" ? undefined : document.baseURI;
    const resolvedBase = base ?? domBase;
    if (resolvedBase === undefined || resolvedBase === null) {
      // With no base there is no origin to compare against, so only an already-absolute http(s) URL
      // is acceptable; a relative path stays null rather than being resolved against an invented
      // base. Same scheme check as the parsed-URL check below, applied to the parsed absolute URL.
      if (!/^https?:\/\//i.test(value)) return null;
      const absolute = new URL(value);
      return absolute.protocol === "http:" || absolute.protocol === "https:" ? absolute.href : null;
    }
    const url = new URL(value, resolvedBase);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (/^https?:\/\//i.test(value)) return url.href;
    if (/^[a-z][a-z\d+.-]*:/i.test(value) || /^[\\/]{2}/.test(value)) return null;
    return url.origin === new URL(resolvedBase).origin ? url.href : null;
  } catch {
    return null;
  }
}
