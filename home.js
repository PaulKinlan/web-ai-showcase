// home.js — homepage controller: grouped catalogue with stable, linkable section fragments + a
// prominent built-only search that reuses the /explore/ search worker (BM25 in a Web Worker; no
// model download for lexical results). Discovery chips + a tiny curated "start here" set round it out.
//
// modern-web-guidance retained + applied (ids):
// - accessibility: semantic landmarks (<header>/<nav>/<main>/<search>), sequential headings, real
//   <label>+<input type=search>, native <a>/<button>, visible focus (styles.css), a SINGLE debounced
//   polite live region for the result count, skip-friendly jump nav, lists for repeated content.
// - search-hidden-content: the "In the pipeline" disclosure stays a native <details> (Find-in-Page +
//   deep-linkable). We never display:none headings that deep links target — the catalogue is only
//   hidden while a search is active (a mode switch), and restored (with fragments) when cleared.
// - size-aware-styling: grids/controls use the design system's fluid layout (no fixed widths); the
//   search box is fluid and the results reuse the responsive .model-grid.
//
// ALL search scoring runs in search/worker.js (CLAUDE.md invariant 15). This file reads DOM, posts
// requests, and paints. The worker/index are loaded LAZILY on first search intent so page load stays
// light; lexical search is usable the instant the index is built — the 25 MB semantic model is never
// downloaded from the homepage (that lives on /explore/).

import { WorkerClient } from "./lib/worker-protocol.js";
import {
  curatedFor,
  homepageSearchPayload,
  INTENT_CHIPS,
  orderedSections,
  readQuery,
  writeQuery,
} from "./home-core.mjs";

const METHOD = { BUILD: "build", SEARCH: "search" };
const CHANNEL = { SEARCH: "search", BUILD: "build" };
const $ = (id) => document.getElementById(id);

// ── device capability line (probe a real adapter — navigator.gpu present ≠ usable) ─────────────────
(async () => {
  const dev = $("device");
  let adapter = false;
  if ("gpu" in navigator) {
    try {
      adapter = !!(await navigator.gpu.requestAdapter());
    } catch { /* no adapter */ }
  }
  dev.textContent = adapter
    ? "✓ WebGPU available — accelerated models will run on your GPU."
    : "gpu" in navigator
    ? "△ WebGPU API present but no usable adapter here — WASM models still run; WebGPU-only models say so."
    : "△ No WebGPU here — WASM-backed models still run (slower); WebGPU-only models say so.";
  if (adapter) dev.classList.add("ok");
})();

// ── fragment scrolling that CONVERGES despite content-visibility height estimation ─────────────────
// Off-screen .model-cards are size-estimated (contain-intrinsic-size); a single scrollIntoView lands
// short because intervening cards realise their true height mid-scroll. Re-assert over a few frames
// until the heading settles at its scroll-margin offset (which clears the sticky search bar).
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  // Force the catalogue to fully render (content-visibility:visible) so ALL card offsets are exact —
  // otherwise an instant fragment jump skips over off-screen cards whose estimated heights never
  // resolve, and the heading lands short. `contain-intrinsic-size: auto` then REMEMBERS each realised
  // size, so restoring content-visibility afterwards won't shift the layout. Perf cost is paid only on
  // an actual fragment navigation (rare).
  const groups = $("catalogue-groups");
  groups?.classList.add("render-all");
  requestAnimationFrame(() => {
    el.scrollIntoView({ block: "start" }); // scroll-margin-top clears the sticky bar; layout is exact now
    el.setAttribute("tabindex", "-1");
    el.focus({ preventScroll: true }); // keyboard users land at the section
    // Keep render-all briefly so any post-scroll de-realise happens only after sizes are remembered.
    setTimeout(() => groups?.classList.remove("render-all"), 400);
  });
}

// ── build a safe model card (all untrusted text via textContent) ──────────────────────────────────
function card(m) {
  const el = document.createElement("a");
  el.className = "model-card";
  el.href = `models/${encodeURIComponent(m.slug)}/`;
  el.style.viewTransitionName = `card-${m.slug}`;

  const h3 = document.createElement("h3");
  h3.textContent = m.name;
  el.append(h3);

  const blurb = document.createElement("p");
  blurb.className = "muted";
  blurb.textContent = m.blurb || m.task;
  el.append(blurb);

  const tags = document.createElement("div");
  tags.className = "tags";
  const tag = (text, cls) => {
    const s = document.createElement("span");
    s.className = "tag" + (cls ? " " + cls : "");
    s.textContent = text;
    return s;
  };
  tags.append(tag(m.task));
  tags.append(m.backend === "webgpu" ? tag("WebGPU", "tag-webgpu") : tag("WASM"));
  if (m.sizeMB) tags.append(tag(`~${m.sizeMB} MB`));
  tags.append(tag("runnable", "tag-run"));
  el.append(tags);

  if (m.unlocks && !String(m.unlocks).startsWith("TODO")) {
    const p = document.createElement("p");
    p.className = "muted";
    p.style.fontSize = ".82rem";
    const strong = document.createElement("strong");
    strong.textContent = "Unlocks: ";
    p.append(strong, document.createTextNode(m.unlocks));
    el.append(p);
  }
  return el;
}

let BUILT = [];
let SECTIONS = [];

// ── render the grouped, linkable catalogue ────────────────────────────────────────────────────────
function renderCatalogue(models) {
  const built = models.filter((m) => m.status === "built");
  const pending = models.filter((m) => m.status !== "built");
  BUILT = built;
  SECTIONS = orderedSections(built);

  // Coverage header — honest denominator.
  const tasks = [...new Set(models.map((m) => m.task))];
  const pct = ((built.length / models.length) * 100).toFixed(1);
  const cov = $("coverage");
  cov.textContent = "";
  const line = document.createElement("p");
  line.style.margin = "0 0 .3rem";
  line.append(
    strongText(String(built.length)),
    document.createTextNode(" interactive demos built · "),
  );
  line.append(
    strongText(String(models.length)),
    document.createTextNode(" browser-runnable model families catalogued across "),
  );
  line.append(strongText(String(tasks.length)), document.createTextNode(" capability tasks."));
  cov.append(line);
  const bar = document.createElement("div");
  bar.className = "bar";
  const barSpan = document.createElement("span");
  barSpan.style.width = `${Math.max(pct, 1.5)}%`;
  bar.append(barSpan);
  cov.append(bar);
  const note = document.createElement("p");
  note.className = "muted";
  note.style.cssText = "font-size:.8rem;margin:0";
  note.textContent =
    `Built ${built.length}/${models.length} (${pct}%). New model pages ship regularly — the routine keeps closing the gap.`;
  cov.append(note);

  // "Jump to" nav — generated from exactly the sections we will render.
  const jump = $("jump");
  jump.innerHTML = "";
  const jh = document.createElement("h2");
  jh.className = "jump-title";
  jh.id = "jump-heading";
  jh.textContent = "Jump to";
  jump.append(jh);
  const jul = document.createElement("ul");
  jul.className = "jump-list";
  for (const s of SECTIONS) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${s.id}`;
    a.textContent = `${s.label} (${s.count})`;
    li.append(a);
    jul.append(li);
  }
  jump.append(jul);
  jump.hidden = false;

  // Grouped sections with stable ids + permalink headings.
  const groups = $("catalogue-groups");
  groups.innerHTML = "";
  for (const s of SECTIONS) {
    const section = document.createElement("section");
    section.className = "cat-section";
    section.setAttribute("aria-labelledby", s.id);

    const h2 = document.createElement("h2");
    h2.id = s.id;
    h2.className = "cat-heading";
    const permalink = document.createElement("a");
    permalink.className = "permalink";
    permalink.href = `#${s.id}`;
    permalink.textContent = s.label;
    const hash = document.createElement("span");
    hash.className = "permalink-hash";
    hash.setAttribute("aria-hidden", "true");
    hash.textContent = "#";
    permalink.append(hash);
    h2.append(permalink);
    section.append(h2);

    if (s.blurb) {
      const p = document.createElement("p");
      p.className = "muted cat-blurb";
      p.textContent = s.blurb;
      section.append(p);
    }

    const grid = document.createElement("div");
    grid.className = "model-grid";
    for (const m of built.filter((x) => (x.modality || "other") === s.key)) grid.append(card(m));
    section.append(grid);
    groups.append(section);
  }

  // Curated "start here".
  const curated = curatedFor(built);
  const startWrap = $("start-here");
  if (curated.length) {
    startWrap.innerHTML = "";
    const h2 = document.createElement("h2");
    h2.id = "start-here-heading";
    h2.textContent = "Good places to start";
    startWrap.append(h2);
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "A representative demo from each major capability — not a popularity ranking.";
    startWrap.append(p);
    const ul = document.createElement("ul");
    ul.className = "curated-list";
    for (const c of curated) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.className = "curated-item";
      a.href = `models/${encodeURIComponent(c.slug)}/`;
      const name = document.createElement("strong");
      name.textContent = c.model.name;
      const why = document.createElement("span");
      why.className = "muted";
      why.textContent = c.why;
      a.append(name, why);
      li.append(a);
      ul.append(li);
    }
    startWrap.append(ul);
    startWrap.hidden = false;
  }

  // Pending universe — a native <details> (searchable + deep-linkable).
  const byTask = new Map();
  for (const m of models) {
    const t = byTask.get(m.task) || { built: 0, total: 0, ex: [] };
    t.total++;
    if (m.status === "built") t.built++;
    else if (t.ex.length < 3) t.ex.push(m.hfId);
    byTask.set(m.task, t);
  }
  const det = document.createElement("details");
  det.className = "pipeline";
  const sum = document.createElement("summary");
  sum.textContent =
    `In the pipeline — ${pending.length} more model families across ${tasks.length} tasks`;
  det.append(sum);
  for (const [task, t] of [...byTask.entries()].sort((a, b) => b[1].total - a[1].total)) {
    const row = document.createElement("div");
    row.className = "task-row";
    const left = document.createElement("span");
    left.append(
      strongText(task),
      document.createTextNode(` — ${t.built} built / ${t.total} eligible`),
    );
    const right = document.createElement("span");
    right.className = "ex";
    right.textContent = t.ex.join(", ") + (t.total > t.built + t.ex.length ? " …" : "");
    row.append(left, right);
    det.append(row);
  }
  $("pipeline-mount").innerHTML = "";
  $("pipeline-mount").append(det);
}

function strongText(text) {
  const s = document.createElement("strong");
  s.textContent = text;
  return s;
}

// ── search: reuse the /explore/ worker for built-only lexical results ─────────────────────────────
const els = {
  form: $("search-form"),
  q: $("q"),
  clear: $("q-clear"),
  count: $("search-count"),
  results: $("results"),
  resultList: $("result-list"),
  catalogueView: $("catalogue-view"),
};

let client = null;
let indexReady = false;
let initPromise = null;
let lastSearchAC = null;

async function ensureSearch() {
  if (indexReady) return true;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    els.count.textContent = "Loading search…";
    client = new WorkerClient({
      url: new URL("./search/worker.js", import.meta.url),
      name: "home-search",
      maxInFlight: 1,
      maxQueue: 4,
    });
    let checksum;
    try {
      const meta = await fetch(new URL("./search/index/meta.json", import.meta.url)).then((r) =>
        r.ok ? r.json() : null
      ).catch(() => null);
      checksum = meta?.checksum;
    } catch { /* build without a checksum hint */ }
    await client.ready;
    await client.request(METHOD.BUILD, { expectChecksum: checksum }, { channel: CHANNEL.BUILD });
    indexReady = true;
    return true;
  })().catch((err) => {
    initPromise = null;
    els.count.textContent = `Search unavailable: ${
      err?.message || err
    }. Browse the catalogue below.`;
    throw err;
  });
  return initPromise;
}

let searchTimer = null;
function scheduleSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => runSearch(false), 160);
}

function setMode(searching) {
  els.results.hidden = !searching;
  els.catalogueView.hidden = searching;
  els.clear.hidden = !els.q.value;
}

async function runSearch(replaceHistory) {
  const q = els.q.value.trim();
  writeState(q, replaceHistory);
  els.clear.hidden = !els.q.value;

  if (!q) {
    setMode(false);
    els.count.textContent = "";
    els.resultList.innerHTML = "";
    return;
  }
  setMode(true);

  let ok = true;
  try {
    await ensureSearch();
  } catch {
    ok = false;
  }
  if (!ok || !indexReady) return;

  lastSearchAC?.abort();
  const ac = new AbortController();
  lastSearchAC = ac;
  try {
    const { result } = await client.request(
      METHOD.SEARCH,
      homepageSearchPayload(q),
      { channel: CHANNEL.SEARCH, signal: ac.signal },
    );
    renderResults(result, q);
  } catch (err) {
    if (err?.name === "AbortError" || err?.name === "SupersededError") return;
    if (err?.name === "QueueOverflowError") return;
    els.count.textContent = `Search failed: ${err?.message || err}`;
  }
}

function renderResults(res, q) {
  // Built-only is enforced in the payload; defend in depth here too.
  const hits = (res.hits || []).filter((h) => h.meta?.status === "built");
  // res.total is the TRUE built-only match count (exact); k=300 ≥ built count so hits === total.
  const total = typeof res.total === "number" ? res.total : hits.length;
  const shownNote = hits.length < total ? ` (showing first ${hits.length})` : "";
  els.count.textContent = `${total} demo${total === 1 ? "" : "s"} for “${q}”${shownNote}`;

  els.resultList.innerHTML = "";
  if (!hits.length) {
    const empty = document.createElement("div");
    empty.className = "no-results";
    const p = document.createElement("p");
    p.append(strongText("No built demos match “"), document.createTextNode(q), strongText("”."));
    empty.append(p);
    const p2 = document.createElement("p");
    p2.className = "muted";
    p2.textContent = "Try a broader phrase or a capability word. Or:";
    empty.append(p2);
    const ul = document.createElement("ul");
    ul.className = "chip-row";
    for (const chip of INTENT_CHIPS.slice(0, 6)) {
      const li = document.createElement("li");
      li.append(chipButton(chip));
      ul.append(li);
    }
    empty.append(ul);
    const explore = document.createElement("p");
    explore.style.marginTop = ".6rem";
    const a = document.createElement("a");
    a.href = `explore/?q=${encodeURIComponent(q)}`;
    a.textContent = "Search all models (including pending) in the explorer →";
    explore.append(a);
    empty.append(explore);
    els.resultList.append(empty);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "model-grid";
  for (const h of hits) grid.append(resultCard(h));
  els.resultList.append(grid);
}

// A search hit carries `meta` (from the index) with demoRoute/blurb/task/backend/sizeMB/name.
function resultCard(h) {
  const m = h.meta || {};
  const model = {
    slug: h.slug,
    name: h.name || m.name,
    blurb: m.blurb,
    task: h.task || m.task,
    backend: m.backend,
    sizeMB: h.sizeMB || m.sizeMB,
    unlocks: m.unlocks,
    // demoRoute is like "models/<slug>/"; fall back to slug.
  };
  const el = card(model);
  if (m.demoRoute) el.href = m.demoRoute;
  return el;
}

function chipButton(chip) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = chip.label;
  b.addEventListener("click", () => {
    els.q.value = chip.q;
    els.q.focus();
    runSearch(false);
  });
  return b;
}

// ── URL <-> state (?q=) ───────────────────────────────────────────────────────────────────────────
function writeState(q, replace) {
  const url = location.pathname + writeQuery({ q }) + location.hash;
  const cur = location.pathname + location.search + location.hash;
  if (url === cur) return;
  history[replace ? "replaceState" : "pushState"]({}, "", url);
}

// ── discovery chips ───────────────────────────────────────────────────────────────────────────────
function renderChips() {
  const mount = $("chips");
  mount.innerHTML = "";
  const ul = document.createElement("ul");
  ul.className = "chip-row";
  for (const chip of INTENT_CHIPS) {
    const li = document.createElement("li");
    li.append(chipButton(chip));
    ul.append(li);
  }
  mount.append(ul);
}

// ── events ────────────────────────────────────────────────────────────────────────────────────────
els.q.addEventListener("input", () => {
  els.clear.hidden = !els.q.value;
  scheduleSearch();
});
els.q.addEventListener("focus", () => {
  ensureSearch().catch(() => {});
}, { once: true });
els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  clearTimeout(searchTimer);
  runSearch(false);
});
els.clear.addEventListener("click", () => {
  els.q.value = "";
  els.clear.hidden = true;
  runSearch(false);
  els.q.focus();
});
window.addEventListener("popstate", () => {
  const { q } = readQuery(location.search);
  els.q.value = q;
  runSearch(true);
  if (!q && location.hash) scrollToSection(location.hash.slice(1));
});

// Intercept in-page section links (jump nav + permalink headings) so the scroll CONVERGES past
// content-visibility estimation, while keeping the URL fragment copyable + back/forward navigable.
document.addEventListener("click", (e) => {
  const a = e.target.closest('a[href^="#cat-"], a[href^="#start-here"]');
  if (!a) return;
  const id = a.getAttribute("href").slice(1);
  if (!document.getElementById(id)) return;
  e.preventDefault();
  if (location.hash.slice(1) !== id) {
    history.pushState(null, "", location.pathname + location.search + "#" + id);
  }
  scrollToSection(id);
});

// ── boot ──────────────────────────────────────────────────────────────────────────────────────────
(async () => {
  renderChips();
  try {
    const { models } = await (await fetch("models.json")).json();
    renderCatalogue(models);
  } catch (e) {
    $("catalogue-groups").innerHTML = "";
    const p = document.createElement("p");
    p.className = "status err";
    p.textContent = `Couldn't load the catalogue: ${e.message}`;
    $("catalogue-groups").append(p);
  }

  // Restore search from ?q= (bookmark/back-forward). A #fragment with no query = catalogue deep link.
  const { q } = readQuery(location.search);
  if (q) {
    els.q.value = q;
    runSearch(true);
  } else if (location.hash) {
    // Sections exist now; scroll (converging past content-visibility estimation).
    scrollToSection(location.hash.slice(1));
  }

  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
})();
