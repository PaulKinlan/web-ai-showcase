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
    base ??= document.baseURI;
    const url = new URL(value, base);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (/^https?:\/\//i.test(value)) return url.href;
    if (/^[a-z][a-z\d+.-]*:/i.test(value) || /^[\\/]{2}/.test(value)) return null;
    return url.origin === new URL(base).origin ? url.href : null;
  } catch {
    return null;
  }
}
