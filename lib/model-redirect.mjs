// Final-response trust boundary for Hugging Face model assets and paths-info metadata.
// fetch(redirect:'follow') hides intermediate hops, so this checks the final readable URL before
// streaming bytes to IndexedDB or copying a non-LFS response into Transformers.js Cache Storage.
// Only HF's resolve origin and observed HF LFS/Xet delivery hosts are accepted; never trust a
// suffix such as *.hf.co (which could include unrelated hosted services).
export class UntrustedHFRedirectError extends Error {}

const LEGACY_CDNS = new Set([
  "cdn-lfs.huggingface.co",
  "cdn-lfs-us-1.huggingface.co",
  "cdn-lfs-eu-1.huggingface.co",
  "cas-bridge.xethub.hf.co",
  "cas-bridge.xethub-eu.hf.co",
]);

export function assertTrustedHFResponse(requestUrl, response) {
  let request;
  let final;
  try {
    request = new URL(requestUrl);
    final = new URL(response.url);
  } catch {
    throw new UntrustedHFRedirectError(
      "Untrusted model download redirect: missing or invalid final URL",
    );
  }
  // Existing local resume fixtures use loopback URLs; only HF requests cross this HF-specific boundary.
  if (request.hostname !== "huggingface.co") return;
  if (
    request.origin !== "https://huggingface.co" || final.protocol !== "https:" ||
    (final.port && final.port !== "443")
  ) {
    throw new UntrustedHFRedirectError("Untrusted model download redirect: unexpected origin");
  }
  const host = final.hostname;
  const allowed = final.origin === request.origin || LEGACY_CDNS.has(host) ||
    /^(?:[a-z0-9-]+)\.aws\.cdn\.hf\.co$/.test(host);
  if (!allowed || response.type === "opaque" || response.type === "opaqueredirect") {
    throw new UntrustedHFRedirectError(
      "Untrusted model download redirect: unexpected final origin",
    );
  }
}
