# Rights-safe media library

A small, git-tracked corpus of **clearly-licensed** test images with full provenance. It exists so
demo pages can ship real sample inputs without hotlinking someone else's server and without bundling
anything whose license we can't point at.

The initial gallery is curated for the **2D→3D / removable-object family** — depth estimation,
matting, semantic/instance/panoptic segmentation, SAM (point/box prompting) and CLIPSeg
(text-prompted segmentation) — where the existing per-model pages only had three samples each. The
set is deliberately varied: a human portrait, animals, a deep landscape, a busy street, a market
crowd, transparent glass, a backlit silhouette, a flat texture, a chart, a night scene, thin
structures (a bicycle), and web-page screenshots — including the easy cases and the known-hard ones.

## What's here

```
media/
  manifest.json           — the source of truth: one record per asset, with full provenance
  manifest.schema.json    — JSON Schema for a manifest asset record
  assets/<id>.avif        — optimized, modern format (smallest)
  assets/<id>.webp        — optimized fallback
  assets/<id>.jpg         — universal fallback
```

Every asset is stored **locally** (no hotlinking) and shipped in three formats: **AVIF** (best
compression), **WebP** (broad support), and a **JPEG** fallback. Sources are rendered/optimized to a
max of ~1600px on the long edge, sRGB, metadata stripped.

## Provenance rules (non-negotiable)

Each `manifest.json` asset records **source URL, creator, license (SPDX or explicit), retrieval date,
local path(s), dimensions, and a one-line description** — plus a ready-to-render `attribution`
string.

1. **Only clearly-licensed assets are bundled.** Accepted: CC0 / Public Domain, CC-BY, CC-BY-SA
   (Wikimedia Commons); the Unsplash License (Unsplash); and Paul Kinlan's own sites (owner
   permission). If a license is unclear or dubious, **skip the asset and record the gap** in
   `manifest.skipped` — never bundle on a guess. (Example: a Commons file tagged "public domain"
   whose author died recently was dropped as a likely mistag.)
2. **Store locally.** Demos reference `media/assets/…`; they must never hotlink the original source.
3. **Attribute when required.** CC-BY / CC-BY-SA assets must be shown with the recorded
   `attribution` when displayed. CC0 / Public Domain need none, but keep the record anyway.
4. **Keep provenance with the bytes.** If you add an asset, add its full record. If you can't fill in
   the source/creator/license, it doesn't go in.

## Sources used

- **Wikimedia Commons** — CC0 / Public Domain / CC-BY / CC-BY-SA. License, author and description are
  pulled from the file's `extmetadata` via the MediaWiki API, so the provenance matches the file page.
- **Headless-Chrome screenshots** — Paul Kinlan's own sites (`paul.kinlan.me`, owner permission) and
  openly-licensed article pages (Wikipedia article body is CC-BY-SA 4.0; trademarks/logos remain their
  owners' marks). Captured with `google-chrome-stable --headless=new`.
- **Unsplash** — Unsplash License (none included in this first wave; the manifest schema supports it).

## How demos should reference the library (future retrofit wave)

This wave only **builds** the library — it is intentionally **not** wired into any of the current demo
pages yet (that's a later retrofit). When a demo does consume it, the pattern is:

- Read `media/manifest.json`, filter by `familyRelevance` (or `id`), and build a `<picture>` so the
  browser negotiates the best format it supports — AVIF → WebP → JPEG:

  ```html
  <picture>
    <source srcset="../../media/assets/city-street.avif" type="image/avif" />
    <source srcset="../../media/assets/city-street.webp" type="image/webp" />
    <img
      src="../../media/assets/city-street.jpg"
      width="1063"
      height="1600"
      loading="lazy"
      decoding="async"
      alt="Busy street scene — panoptic segmentation / monocular depth test input"
    />
  </picture>
  ```

  Always set `width`/`height` (from the manifest) to reserve space and avoid layout shift, and render
  the asset's `attribution` next to it whenever the license is CC-BY / CC-BY-SA.

## Regenerating / extending

The curation helpers live in `scripts/` and are one-off (not gates):

- `node scripts/fetch-commons.mjs` — search Commons, verify license via the API, pull web-sized
  renderings of licensed originals.
- `node scripts/fetch-screenshots.mjs` — capture web-page screenshots via headless Chrome.
- `node scripts/build-media-manifest.mjs` — optimize every source to avif/webp/jpg and emit
  `manifest.json` with provenance.

To add an asset by hand: drop the optimized `avif`/`webp`/`jpg` into `assets/`, then add a complete
record (all provenance fields) to `manifest.json`. Incomplete provenance = don't add it.
