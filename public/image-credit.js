// Contextual image credits — a tiny, dependency-free, reusable annotator.
//
// It reads the site's image-provenance ledger (image-provenance/ledger.json) and, for every <img> on the
// page whose bytes come from a LICENSED source (Wikimedia Commons — CC-BY / CC-BY-SA / CC0 / Public
// Domain), it (a) sets a descriptive title/aria attribution on the image and (b) renders one restrained,
// de-duplicated credit line next to each group of sample images, linking to the full /image-credits/
// page. CC-BY / CC-BY-SA require visible attribution wherever the work is shown — this provides it without
// hand-editing every demo. First-party (procedural) images need no attribution and are left alone.
//
// Include once per page:  <script type="module" src="/web-ai-showcase/public/image-credit.js"></script>
// It is a no-op on pages that display no licensed image, safe with multiple sample strips, and re-runs on
// dynamically-swapped sample images via a MutationObserver.

const REPO_ROOTS = ["/models/", "/media/", "/reports/", "/public/", "/image-provenance/"];

// Derive the site base (…/web-ai-showcase/) and repo-relative path from an absolute image URL.
function toRepoPath(absUrl) {
  let pathname;
  try {
    pathname = new URL(absUrl, location.href).pathname;
  } catch {
    return null;
  }
  for (const root of REPO_ROOTS) {
    const i = pathname.indexOf(root);
    if (i >= 0) return pathname.slice(i + 1); // "models/foo/sample.jpg"
  }
  return null;
}
function siteBase() {
  const p = location.pathname;
  for (const root of REPO_ROOTS) {
    const i = p.indexOf(root);
    if (i >= 0) return p.slice(0, i + 1); // "/web-ai-showcase/"
  }
  return "/";
}

async function loadLedger() {
  const base = siteBase();
  const res = await fetch(base + "image-provenance/ledger.json");
  if (!res.ok) throw new Error("ledger HTTP " + res.status);
  const ledger = await res.json();
  const byPath = new Map();
  for (const e of ledger.entries) {
    if (e.provenance?.kind !== "licensed") continue; // only licensed images need visible credit
    for (const p of e.paths) byPath.set(p, e);
  }
  return { byPath, base };
}

const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
  );

function creditLineHTML(entries, base) {
  // de-dupe by source asset / attribution
  const seen = new Map();
  for (const e of entries) {
    const key = e.provenance.sourceAsset || e.provenance.attribution;
    if (!seen.has(key)) seen.set(key, e.provenance);
  }
  const parts = [...seen.values()].map((p) => {
    const who = esc(p.creator || "web-ai-showcase");
    const lic = esc(p.licenseName || p.license || "");
    const link = p.sourceUrl
      ? ` <a href="${esc(p.sourceUrl)}" rel="noopener" target="_blank">↗</a>`
      : "";
    return `${who} — ${lic}${link}`;
  });
  return `Sample photos: ${parts.join(" · ")} · <a href="${
    esc(base)
  }image-credits/">all image credits</a>`;
}

function annotate(byPath, base) {
  const onPage = [];
  for (const img of document.images) {
    const path = toRepoPath(img.currentSrc || img.src) ||
      toRepoPath(img.getAttribute("data-src") || "");
    if (!path) continue;
    const e = byPath.get(path);
    if (!e) continue;
    onPage.push(e);
    const attr = e.provenance.attribution || "";
    if (attr && img.title !== attr) {
      img.title = attr; // hover/contextual per-image attribution
      if (!img.getAttribute("data-credited")) img.setAttribute("data-credited", "1");
    }
  }
  if (!onPage.length) return;

  // Render one restrained credit line. Prefer to place it right after each sample strip; else once at the
  // end of <main>. Idempotent — reuses an existing node.
  const strips = document.querySelectorAll(".sample-strip");
  const html = creditLineHTML(onPage, base);
  if (strips.length) {
    for (const strip of strips) {
      let credit = strip.parentElement.querySelector(":scope > .img-credit");
      if (!credit) {
        credit = document.createElement("p");
        credit.className = "img-credit";
        credit.style.cssText = "font-size:.72rem;opacity:.72;margin:.35rem 0 0;line-height:1.4;";
        strip.insertAdjacentElement("afterend", credit);
      }
      credit.innerHTML = html;
    }
  } else {
    const main = document.querySelector("main") || document.body;
    let credit = document.getElementById("img-credit-global");
    if (!credit) {
      credit = document.createElement("p");
      credit.id = "img-credit-global";
      credit.className = "img-credit";
      credit.style.cssText = "font-size:.72rem;opacity:.72;margin:1rem 0 0;line-height:1.4;";
      main.appendChild(credit);
    }
    credit.innerHTML = html;
  }
}

try {
  const { byPath, base } = await loadLedger();
  const run = () => annotate(byPath, base);
  run();
  // re-annotate when sample images are swapped in / added (throttled via microtask flag)
  let queued = false;
  const mo = new MutationObserver(() => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      run();
    });
  });
  mo.observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["src"],
  });
} catch (e) {
  // Non-fatal: credits are enhancement. The /image-credits/ page remains the authoritative record.
  console.warn("[image-credit] " + (e?.message || e));
}
