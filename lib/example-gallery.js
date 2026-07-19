// lib/example-gallery.js — a reusable, accessible, rights-safe example-image gallery.
//
// Renders the provenance-tracked assets in media/manifest.json as a selectable
// thumbnail strip so image demos (depth / matting / segmentation / SAM / CLIPSeg /
// panoptic — the "2D→3D / removable-object" family that was underfed with 3 samples)
// get a broad, varied gallery: people, objects, scenes, textures, documents, charts,
// text-heavy images, night/backlit/transparent hard cases.
//
// - Every asset keeps its attribution on screen (creator + license) whenever it is the
//   selected image — rights-safe display, never hotlinked (all assets are local).
// - Thumbnails are <picture> avif→webp→jpg, lazy-loaded, with content-visibility so an
//   offscreen strip costs nothing.
// - onSelect(asset) hands back the resolved LOCAL url of the full image (jpg — broadest
//   decode support in transformers.js workers) plus width/height/description/attribution.
// - Structural CSS only (layout + a focus ring via currentColor); colours inherit from
//   the page's design system so it themes light/dark automatically (invariant 5).
//
// Usage:
//   import { createExampleGallery } from "../../lib/example-gallery.js";
//   const gallery = await createExampleGallery({
//     mount: document.querySelector("#gallery"),
//     manifestUrl: "../../media/manifest.json",   // resolved relative to this page
//     onSelect: (asset) => runInferenceOn(asset.url),   // asset.url is a local path
//     filter: (a) => true,                          // optional
//   });
//   // gallery.selected -> the current asset; gallery.destroy() -> cleanup.

let styleInjected = false;
function injectStyle() {
  if (styleInjected) return;
  styleInjected = true;
  const css = `
.exg { display:flex; flex-direction:column; gap:.5rem; }
.exg-strip { display:grid; grid-template-columns:repeat(auto-fill,minmax(84px,1fr));
  gap:.5rem; content-visibility:auto; contain-intrinsic-size:auto 84px; }
.exg-item { padding:0; border:1px solid; border-color:color-mix(in srgb, currentColor 25%, transparent);
  border-radius:8px; background:transparent; cursor:pointer; overflow:hidden; aspect-ratio:1/1;
  min-inline-size:0; }
.exg-item picture, .exg-item img { display:block; inline-size:100%; block-size:100%; object-fit:cover; }
.exg-item[aria-pressed="true"] { outline:3px solid; outline-offset:-3px; }
.exg-item:focus-visible { outline:3px solid; outline-offset:2px; }
.exg-caption { font-size:.85rem; line-height:1.4; min-block-size:2.6em; }
.exg-caption a { color:inherit; }
@media (prefers-reduced-motion: no-preference) { .exg-item { transition:outline-color .1s; } }
`;
  const el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
}

function resolve(path, base) {
  return new URL(path, base).href;
}

export async function createExampleGallery(
  {
    mount,
    manifestUrl = "../../media/manifest.json",
    onSelect,
    filter,
    limit,
    heading = "Example gallery",
  },
) {
  if (!mount) throw new Error("createExampleGallery: `mount` is required");
  injectStyle();
  const manifestAbs = new URL(manifestUrl, document.baseURI).href;

  let data;
  try {
    data = await fetch(manifestAbs, { cache: "force-cache" }).then((r) => {
      if (!r.ok) throw new Error(`manifest ${r.status}`);
      return r.json();
    });
  } catch (err) {
    // Fail honestly on the page, don't throw the whole demo down.
    mount.innerHTML = "";
    const p = document.createElement("p");
    p.className = "exg-caption";
    p.setAttribute("role", "status");
    p.textContent = `Example gallery unavailable (${String(err.message || err)}).`;
    mount.appendChild(p);
    return { selected: null, destroy() {}, error: err };
  }

  let assets = (data.assets || []).filter((a) =>
    a && a.formats && (a.formats.jpg || a.formats.webp)
  );
  if (filter) assets = assets.filter(filter);
  if (limit) assets = assets.slice(0, limit);

  mount.innerHTML = "";
  const root = document.createElement("div");
  root.className = "exg";

  const strip = document.createElement("div");
  strip.className = "exg-strip";
  strip.setAttribute("role", "group");
  strip.setAttribute("aria-label", heading);

  const caption = document.createElement("p");
  caption.className = "exg-caption";
  caption.setAttribute("role", "status");
  caption.setAttribute("aria-live", "polite");

  const api = { selected: null, destroy() {}, assets };

  function pictureFor(asset) {
    const pic = document.createElement("picture");
    for (const fmt of ["avif", "webp"]) {
      const f = asset.formats[fmt];
      if (!f) continue;
      const s = document.createElement("source");
      s.type = `image/${fmt}`;
      s.srcset = resolve(f.path, manifestAbs);
      pic.appendChild(s);
    }
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.width = 84;
    img.height = 84;
    img.alt = ""; // decorative in the button; the button carries the accessible name
    img.src = resolve((asset.formats.jpg || asset.formats.webp).path, manifestAbs);
    pic.appendChild(img);
    return pic;
  }

  function select(asset, btn) {
    for (const b of strip.querySelectorAll(".exg-item")) b.setAttribute("aria-pressed", "false");
    if (btn) btn.setAttribute("aria-pressed", "true");
    api.selected = asset;
    const url = resolve((asset.formats.jpg || asset.formats.webp).path, manifestAbs);
    // rights-safe attribution, always shown while this image is in use
    caption.innerHTML = "";
    const strong = document.createElement("span");
    strong.textContent = asset.familyRelevance || asset.description?.slice(0, 80) || asset.id;
    caption.appendChild(strong);
    caption.appendChild(document.createElement("br"));
    const attr = document.createElement("small");
    const link = asset.sourceUrl
      ? `<a href="${asset.sourceUrl}" target="_blank" rel="noopener noreferrer">${asset.source}</a>`
      : asset.source;
    attr.innerHTML = `${asset.creator ? asset.creator + " · " : ""}${
      asset.licenseName || asset.license
    }${asset.source ? " · via " + link : ""}`;
    caption.appendChild(attr);
    if (typeof onSelect === "function") {
      onSelect({
        id: asset.id,
        url,
        width: asset.width,
        height: asset.height,
        description: asset.description,
        attribution: asset.attribution,
        asset,
      });
    }
  }

  for (const asset of assets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "exg-item";
    btn.setAttribute("aria-pressed", "false");
    const name = `${
      asset.familyRelevance || asset.description?.slice(0, 60) || asset.id
    }. Source: ${asset.creator ? asset.creator + ", " : ""}${
      asset.licenseName || asset.license
    }, via ${asset.source}.`;
    btn.setAttribute("aria-label", name);
    btn.title = name;
    btn.appendChild(pictureFor(asset));
    btn.addEventListener("click", () => select(asset, btn));
    strip.appendChild(btn);
  }

  root.appendChild(strip);
  root.appendChild(caption);
  mount.appendChild(root);

  api.destroy = () => {
    mount.innerHTML = "";
  };
  api.selectId = (id) => {
    const btn = [...strip.querySelectorAll(".exg-item")].find(
      (b, i) => assets[i]?.id === id,
    );
    const asset = assets.find((a) => a.id === id);
    if (asset) select(asset, btn);
  };
  return api;
}
