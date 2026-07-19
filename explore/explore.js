// Client-side model explorer — main-thread controller for /explore/.
//
// Talks to search/worker.js over lib/worker-protocol.js (WorkerClient): build/search/facets/explain/
// embedStatus/embedDownload. ALL search compute — BM25, int8 cosine, filtering, facets, AND the query
// embedding — runs in the worker (CLAUDE.md invariant 15); this file only reads DOM, posts requests,
// and paints. Latest-wins on channel "search" + a per-search AbortController give free
// stale-suppression/cancellation. Search state round-trips through the URL (shareable deep links).
//
// The typed contract lives in ../search/search-protocol.ts; browsers can't import .ts, so the runtime
// METHOD/CHANNEL string constants are mirrored here 1:1.
//
// modern-web-guidance retained + applied (ids): accessibility (combobox → listbox, aria-activedescendant,
// role="status" aria-live count, ≥44px targets, :focus-visible, skip link), identify-inp-causes +
// break-up-long-tasks (compute offloaded to the worker; input handler only posts a message; debounced),
// performance (content-visibility on result rows; no main-thread scoring).

import { QueueOverflowError, SupersededError, WorkerClient } from "../lib/worker-protocol.js";

const METHOD = {
  BUILD: "build",
  SEARCH: "search",
  FACETS: "facets",
  EXPLAIN: "explain",
  EMBED_STATUS: "embedStatus",
  EMBED_DOWNLOAD: "embedDownload",
};
const CHANNEL = { SEARCH: "search", BUILD: "build" };

const $ = (id) => document.getElementById(id);
const els = {
  q: $("q"),
  mode: $("mode"),
  alpha: $("alpha"),
  alphaVal: $("alpha-val"),
  expand: $("expand"),
  count: $("count"),
  results: $("results"),
  embedDot: $("embed-dot"),
  embedText: $("embed-text"),
  embedBtn: $("embed-btn"),
  embedProg: $("embed-prog"),
  filters: $("filters"),
  clearFilters: $("clear-filters"),
  copyLink: $("copy-link"),
  sizeMin: $("f-size-min"),
  sizeMax: $("f-size-max"),
  device: $("f-device"),
};

const CHECKBOX_FACETS = [
  "status",
  "task",
  "modality",
  "runtime",
  "backend",
  "license",
  "relKind",
  "tier",
];

// ── URL <-> state ───────────────────────────────────────────────────────────────────────────────────
function readState() {
  const p = new URLSearchParams(location.search);
  const list = (k) => (p.get(k) ? p.get(k).split(",").filter(Boolean) : []);
  const num = (k) => (p.get(k) != null && p.get(k) !== "" ? Number(p.get(k)) : undefined);
  return {
    q: p.get("q") || "",
    mode: p.get("mode") || "hybrid",
    alpha: p.get("alpha") != null ? Number(p.get("alpha")) : 0.5,
    expandAliases: p.get("expand") !== "0",
    filters: {
      status: list("status"),
      task: list("task"),
      modality: list("modality"),
      runtime: list("runtime"),
      backend: list("backend"),
      license: list("license"),
      relKind: list("relKind"),
      tier: list("tier"),
      sizeMinMB: num("sizeMin"),
      sizeMaxMB: num("sizeMax"),
      canonicalFamily: p.get("family") || undefined,
      device: p.get("device") === "1" ? "this-device" : undefined,
    },
  };
}

function writeState(replace) {
  const p = new URLSearchParams();
  if (state.q) p.set("q", state.q);
  if (state.mode !== "hybrid") p.set("mode", state.mode);
  if (state.alpha !== 0.5) p.set("alpha", String(state.alpha));
  if (!state.expandAliases) p.set("expand", "0");
  const f = state.filters;
  for (const k of CHECKBOX_FACETS) if (f[k]?.length) p.set(k, f[k].join(","));
  if (f.sizeMinMB != null) p.set("sizeMin", String(f.sizeMinMB));
  if (f.sizeMaxMB != null) p.set("sizeMax", String(f.sizeMaxMB));
  if (f.canonicalFamily) p.set("family", f.canonicalFamily);
  if (f.device === "this-device") p.set("device", "1");
  const url = location.pathname + (p.toString() ? "?" + p.toString() : "");
  history[replace ? "replaceState" : "pushState"]({}, "", url);
}

function cleanFilters(f) {
  // Strip empty arrays/undefined so the worker sees "no constraint".
  const out = {};
  for (const k of CHECKBOX_FACETS) if (f[k]?.length) out[k] = f[k];
  if (f.sizeMinMB != null && !Number.isNaN(f.sizeMinMB)) out.sizeMinMB = f.sizeMinMB;
  if (f.sizeMaxMB != null && !Number.isNaN(f.sizeMaxMB)) out.sizeMaxMB = f.sizeMaxMB;
  if (f.canonicalFamily) out.canonicalFamily = f.canonicalFamily;
  if (f.device) out.device = f.device;
  return out;
}

let state = readState();

// ── worker ─────────────────────────────────────────────────────────────────────────────────────────
const client = new WorkerClient({
  url: new URL("../search/worker.js", import.meta.url),
  name: "explore",
  maxInFlight: 1,
  maxQueue: 4,
});

let META = null;
let indexReady = false;
let lastSearchAC = null;

function setEmbed(status) {
  const s = status.state;
  els.embedDot.className = "dot " +
    (s === "ready" ? "ready" : s === "error" ? "error" : s === "absent" ? "absent" : "");
  const map = {
    ready: "Semantic ranking on — queries are embedded on-device.",
    absent:
      `Keyword search ready. Add semantic ranking with the on-device intent model (~${status.sizeMB} MB).`,
    downloading: "Downloading intent model…",
    initialising: "Starting intent model…",
    unsupported: "Semantic ranking unavailable on this device — keyword search works fully.",
    error: "Intent model failed to load — keyword search still works.",
  };
  els.embedText.textContent = map[s] || s;
  els.embedBtn.hidden = !(s === "absent" || s === "error");
  els.embedBtn.textContent = s === "error"
    ? "Retry intent model"
    : `Download intent model (~${status.sizeMB} MB)`;
  els.embedBtn.disabled = false;
  // Semantic controls only matter once/if the model is available.
  els.alpha.disabled = s !== "ready" || state.mode === "lexical";
}

async function refreshEmbedStatus() {
  try {
    const { result } = await client.request(METHOD.EMBED_STATUS, {});
    setEmbed(result);
    return result;
  } catch {
    return null;
  }
}

async function boot() {
  renderSkeleton();
  try {
    // meta.json is small (~75 KB) and carries the facet value lists — parse it on the main thread;
    // the ~3 MB index.json is fetched/parsed only in the worker (keeps INP long-task-free).
    META = await fetch(new URL("../search/index/meta.json", import.meta.url)).then((r) =>
      r.ok ? r.json() : null
    ).catch(() => null);
    FACET_VALUES = META?.facetValues || {};
    buildFilterControls();
    applyStateToControls();
    await client.ready;
    els.count.textContent = "Loading the model index…";
    await client.request(
      METHOD.BUILD,
      { expectChecksum: META?.checksum },
      { channel: CHANNEL.BUILD, onProgress: onBuildProgress },
    );
    indexReady = true;
    await refreshEmbedStatus();
    runSearch(true);
  } catch (err) {
    showError(`Couldn't load the search index: ${err?.message || err}`, boot);
  }
}

function onBuildProgress(p) {
  const label = {
    fetching: "Fetching index",
    indexing: "Building index",
    "embedding-index": "Embedding",
    persisting: "Saving index",
  };
  els.count.textContent = `${label[p.status] || p.status}… ${Math.round(p.progress)}%`;
}

// ── search ───────────────────────────────────────────────────────────────────────────────────────────
let searchTimer = null;
function scheduleSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => runSearch(false), 150);
}

async function runSearch(replaceHistory) {
  if (!indexReady) return;
  writeState(replaceHistory);
  const filters = cleanFilters(state.filters);
  lastSearchAC?.abort();
  const ac = new AbortController();
  lastSearchAC = ac;
  const payload = {
    q: state.q,
    filters,
    mode: state.mode,
    alpha: state.alpha,
    expandAliases: state.expandAliases,
    k: 60,
  };
  try {
    const [{ result }, facets] = await Promise.all([
      client.request(METHOD.SEARCH, payload, { channel: CHANNEL.SEARCH, signal: ac.signal }),
      client.request(METHOD.FACETS, { q: state.q, filters }, { channel: "facets" }).then((r) =>
        r.result
      ).catch(() => null),
    ]);
    renderResults(result);
    if (facets) updateFacetCounts(facets);
  } catch (err) {
    if (err instanceof SupersededError || err?.name === "AbortError") return; // newer query won — ignore
    if (err instanceof QueueOverflowError) return; // backpressure — the trailing debounced call will run
    showError(`Search failed: ${err?.message || err}`);
  }
}

// ── rendering ─────────────────────────────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
  );
}
function highlight(text, terms) {
  if (!terms?.length) return esc(text);
  const re = new RegExp(
    "\\b(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")",
    "gi",
  );
  return esc(text).replace(re, "<mark>$1</mark>");
}

function renderSkeleton() {
  els.results.innerHTML = Array.from(
    { length: 5 },
    () => '<div class="skeleton" aria-hidden="true"></div>',
  ).join("");
}

let activeOptions = []; // flat list of option elements for combobox keyboard nav
let activeIdx = -1;

function renderResults(res) {
  activeOptions = [];
  activeIdx = -1;
  currentHits = res.hits;
  els.q.removeAttribute("aria-activedescendant");

  const semanticNote = state.mode !== "lexical" && !res.semanticApplied && state.q.trim()
    ? " · keyword-only (add the intent model for semantic ranking)"
    : "";
  els.count.textContent = `${res.total} model${res.total === 1 ? "" : "s"}${
    state.q ? ` for “${state.q}”` : ""
  }${semanticNote} · ${res.tookMs} ms`;
  els.q.setAttribute("aria-expanded", res.total > 0 ? "true" : "false");

  if (!res.hits.length) {
    els.results.innerHTML = `<div class="state"><p><strong>No models match.</strong></p>
      <p>Try fewer filters, a broader phrase, or turn on “Expand aliases”.</p></div>`;
    return;
  }

  // Group by canonical family in rank order: first-seen = representative; rest = variants (revealable).
  // Models with no known family ("unknown") are NOT collapsed together — each stands alone (a synthetic
  // per-slug key), so "unknown" never masquerades as a real family.
  const groups = [];
  const byFam = new Map();
  for (const h of res.hits) {
    const known = h.canonicalFamily && h.canonicalFamily !== "unknown";
    const fam = known ? h.canonicalFamily : "__" + h.slug;
    let g = byFam.get(fam);
    if (!g) {
      g = { fam, known, rep: h, variants: [] };
      byFam.set(fam, g);
      groups.push(g);
    } else {
      g.variants.push(h);
    }
  }

  const ul = document.createElement("ul");
  ul.className = "listbox";
  ul.setAttribute("role", "listbox");
  ul.setAttribute("aria-label", "Search results");

  let optSeq = 0;
  for (const g of groups) {
    const li = document.createElement("li");
    li.setAttribute("role", "presentation");
    li.className = "fam-group";

    const head = document.createElement("div");
    head.className = "fam-head";
    const totalInGroup = 1 + g.variants.length;
    const title = g.known ? g.rep.canonicalFamily : g.rep.name;
    const meta = g.known
      ? `${g.rep.meta.familyCount} in family · ${totalInGroup} match${
        totalInGroup === 1 ? "" : "es"
      } here`
      : "no shared family";
    head.innerHTML = `<h3>${esc(title)}</h3><span class="fam-meta">${meta}</span>`;
    li.append(head);

    const repOpt = renderOption(g.rep, g.rep.explain, optSeq++);
    li.append(repOpt);

    if (g.variants.length) {
      const varWrap = document.createElement("div");
      varWrap.hidden = true;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "reveal-variants";
      btn.setAttribute("aria-expanded", "false");
      btn.textContent = `Show ${g.variants.length} variant${
        g.variants.length === 1 ? "" : "s"
      } / related`;
      btn.addEventListener("click", () => {
        const open = varWrap.hidden;
        varWrap.hidden = !open;
        btn.setAttribute("aria-expanded", String(open));
        btn.textContent = open
          ? "Hide variants / related"
          : `Show ${g.variants.length} variant${g.variants.length === 1 ? "" : "s"} / related`;
        rebuildActiveOptions(ul);
      });
      for (const v of g.variants) {
        const vo = renderOption(v, v.explain, optSeq++, true);
        varWrap.append(vo);
      }
      head.append(btn);
      li.append(varWrap);
    }
    ul.append(li);
  }

  els.results.innerHTML = "";
  els.results.append(ul);
  rebuildActiveOptions(ul);

  if (res.cursor != null) {
    const more = document.createElement("button");
    more.className = "secondary";
    more.textContent = "Load more";
    more.style.marginTop = "0.5rem";
    more.addEventListener("click", () => loadMore(res.cursor, more));
    els.results.append(more);
  }
}

function renderOption(h, ex, seq, isVariant) {
  const m = h.meta;
  const opt = document.createElement("div");
  opt.className = "result" + (isVariant ? " variant" : "");
  opt.id = `opt-${seq}`;
  opt.setAttribute("role", "option");
  opt.setAttribute("aria-selected", "false");
  opt.dataset.href = m.demoRoute ? new URL("../" + m.demoRoute, import.meta.url).pathname : m.hfUrl;

  const badges = [];
  if (m.status === "built") badges.push(`<span class="badge built">built demo</span>`);
  else if (m.status === "blocked") badges.push(`<span class="badge conf-low">blocked</span>`);
  else badges.push(`<span class="badge">pending</span>`);
  badges.push(`<span class="badge ${m.relKind}">${esc(m.relLabel)}</span>`);
  if (m.tier === "high") badges.push(`<span class="badge tier-high">priority: high</span>`);
  else if (m.tier && m.tier !== "built") {
    badges.push(`<span class="badge">priority: ${esc(m.tier)}</span>`);
  }
  badges.push(
    `<span class="badge ${m.confidence === "low" ? "conf-low" : ""}">confidence: ${
      esc(m.confidence)
    }</span>`,
  );
  badges.push(`<span class="badge">${esc(h.task)}</span>`);
  badges.push(`<span class="badge">${m.backend === "webgpu" ? "WebGPU" : "WASM"}</span>`);
  if (h.sizeMB) badges.push(`<span class="badge">~${h.sizeMB} MB</span>`);
  if (h.license) badges.push(`<span class="badge">${esc(h.license)}</span>`);

  const titleHref = opt.dataset.href;
  const linkText = m.demoRoute ? "Open interactive demo →" : "View model card on Hugging Face →";

  // Why-matched + provenance/duplicate/quality rationale.
  const why = [];
  if (ex.matchedTerms?.length) {
    why.push(`Matched terms: ${ex.matchedTerms.map((t) => `<code>${esc(t)}</code>`).join(" ")}`);
  }
  if (ex.aliasExpanded?.length) {
    why.push(`Alias-expanded: ${ex.aliasExpanded.map((t) => `<code>${esc(t)}</code>`).join(" ")}`);
  }
  why.push(
    `Scores — BM25 <code>${ex.bm25}</code>${
      ex.semantic == null
        ? " · semantic <code>off</code>"
        : ` · semantic <code>${ex.semantic}</code>`
    }`,
  );
  if (ex.filterHits?.length) {
    why.push(`Filters satisfied: ${ex.filterHits.map((t) => `<code>${esc(t)}</code>`).join(" ")}`);
  }
  const prov = [
    `Lineage: <strong>${esc(m.relLabel)}</strong> in the <code>${
      esc(h.canonicalFamily)
    }</code> family (confidence: ${esc(m.confidence)}).`,
  ];
  if (m.relKind !== "canonical") {
    prov.push(
      `Re-presents an existing capability${
        m.canonicalAlternative
          ? ` — canonical alternative: <code>${esc(m.canonicalAlternative)}</code>`
          : ""
      }.`,
    );
  }
  if (m.tierRationale) prov.push(`Priority rationale: ${esc(m.tierRationale)}`);
  const quality = [
    `Popularity (weak signal): ${
      m.downloads != null ? m.downloads.toLocaleString() + " downloads" : "n/a"
    }${m.likes != null ? `, ${m.likes} likes` : ""} · percentile ${
      (m.qualityPercentile * 100).toFixed(0)
    }%.`,
  ];
  if (m.evalPending?.length) {
    quality.push(
      `Eval-pending (honest gaps): <code>${m.evalPending.map(esc).join("</code> <code>")}</code>.`,
    );
  }
  if (m.blockedReason) quality.push(`Blocked: ${esc(m.blockedReason)}`);

  opt.innerHTML = `
    <h4><a href="${esc(titleHref)}"${m.demoRoute ? "" : ' target="_blank" rel="noopener"'}>${
    highlight(h.name, ex.matchedTerms)
  }</a></h4>
    <div class="badges">${badges.join("")}</div>
    ${m.blurb ? `<p class="blurb">${highlight(m.blurb, ex.matchedTerms)}</p>` : ""}
    <details class="why">
      <summary>Why this matched</summary>
      <div class="why-body">
        <p>${why.join(" · ")}</p>
        <p><strong>Provenance / duplicate:</strong> ${prov.join(" ")}</p>
        <p><strong>Quality:</strong> ${quality.join(" ")}</p>
      </div>
    </details>
    <div class="links">
      <a href="${esc(titleHref)}"${
    m.demoRoute ? "" : ' target="_blank" rel="noopener"'
  }>${linkText}</a>
      ${
    m.demoRoute ? `<a href="${esc(m.hfUrl)}" target="_blank" rel="noopener">Model card ↗</a>` : ""
  }
    </div>`;

  opt.addEventListener("click", (e) => {
    if (e.target.closest("a, summary, button")) return; // let real links/toggles work
    navigateTo(opt);
  });
  return opt;
}

function rebuildActiveOptions(ul) {
  activeOptions = [...ul.querySelectorAll('[role="option"]')].filter((o) =>
    o.offsetParent !== null
  );
}

let currentHits = [];
async function loadMore(cursor, btn) {
  btn.disabled = true;
  const accumulated = currentHits;
  try {
    const { result } = await client.request(
      METHOD.SEARCH,
      {
        q: state.q,
        filters: cleanFilters(state.filters),
        mode: state.mode,
        alpha: state.alpha,
        expandAliases: state.expandAliases,
        k: 60,
        cursor,
      },
      {},
    );
    // Re-render the widened window so canonical-family grouping stays correct across pages.
    renderResults({ ...result, hits: [...accumulated, ...result.hits] });
  } catch {
    btn.disabled = false;
    btn.textContent = "Load more (retry)";
  }
}

// ── combobox keyboard navigation (aria-activedescendant over the listbox) ──────────────────────────
function setActive(idx) {
  if (activeIdx >= 0 && activeOptions[activeIdx]) {
    activeOptions[activeIdx].setAttribute("aria-selected", "false");
  }
  activeIdx = idx;
  if (idx < 0 || !activeOptions[idx]) {
    els.q.removeAttribute("aria-activedescendant");
    return;
  }
  const opt = activeOptions[idx];
  opt.setAttribute("aria-selected", "true");
  els.q.setAttribute("aria-activedescendant", opt.id);
  opt.scrollIntoView({ block: "nearest" });
}

function navigateTo(opt) {
  const href = opt.dataset.href;
  const link = opt.querySelector("h4 a");
  if (link && link.target === "_blank") window.open(href, "_blank", "noopener");
  else location.assign(href);
}

els.q.addEventListener("keydown", (e) => {
  if (!activeOptions.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    setActive(Math.min(activeIdx + 1, activeOptions.length - 1));
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    setActive(Math.max(activeIdx - 1, 0));
  } else if (e.key === "Home" && activeIdx >= 0) {
    e.preventDefault();
    setActive(0);
  } else if (e.key === "End" && activeIdx >= 0) {
    e.preventDefault();
    setActive(activeOptions.length - 1);
  } else if (e.key === "Enter" && activeIdx >= 0) {
    e.preventDefault();
    navigateTo(activeOptions[activeIdx]);
  } else if (e.key === "Escape") {
    setActive(-1);
  }
});

// ── filter controls ──────────────────────────────────────────────────────────────────────────────────
const FACET_MOUNT = {
  status: "f-status",
  task: "f-task",
  modality: "f-modality",
  runtime: "f-runtime",
  backend: "f-backend",
  license: "f-license",
  relKind: "f-relKind",
  tier: "f-tier",
};
let FACET_VALUES = {};
function buildFilterControls() {
  // Facet value lists come from meta.json (parsed once, on boot). No large main-thread parse.
  for (const [facet, mountId] of Object.entries(FACET_MOUNT)) {
    const mount = $(mountId);
    if (!mount) continue;
    mount.innerHTML = "";
    for (const val of FACET_VALUES[facet] || []) {
      const row = document.createElement("div");
      row.className = "opt";
      const cid = `chk-${facet}-${String(val).replace(/[^a-z0-9]+/gi, "-")}`;
      row.innerHTML =
        `<label for="${cid}"><input type="checkbox" id="${cid}" data-facet="${facet}" value="${
          esc(val)
        }" /> ${esc(val)}</label>` +
        `<span class="cnt" data-cnt="${facet}:${esc(val)}"></span>`;
      mount.append(row);
    }
  }
}

function updateFacetCounts(facets) {
  for (const facet of Object.keys(FACET_MOUNT)) {
    const counts = facets[facet] || {};
    for (const val of FACET_VALUES[facet] || []) {
      const el = document.querySelector(`[data-cnt="${CSS.escape(facet + ":" + val)}"]`);
      if (el) el.textContent = counts[val] ? String(counts[val]) : "0";
    }
  }
}

function applyStateToControls() {
  els.q.value = state.q;
  els.mode.value = state.mode;
  els.alpha.value = String(state.alpha);
  els.alphaVal.textContent = String(state.alpha);
  els.expand.checked = state.expandAliases;
  els.sizeMin.value = state.filters.sizeMinMB ?? "";
  els.sizeMax.value = state.filters.sizeMaxMB ?? "";
  els.device.checked = state.filters.device === "this-device";
  for (const facet of CHECKBOX_FACETS) {
    const selected = new Set(state.filters[facet] || []);
    document.querySelectorAll(`input[data-facet="${facet}"]`).forEach((cb) => {
      cb.checked = selected.has(cb.value);
    });
  }
}

// ── events ───────────────────────────────────────────────────────────────────────────────────────────
els.q.addEventListener("input", () => {
  state.q = els.q.value;
  scheduleSearch();
});
els.mode.addEventListener("change", () => {
  state.mode = els.mode.value;
  els.alpha.disabled = state.mode === "lexical" ||
    els.embedDot.classList.contains("ready") === false;
  runSearch(false);
});
els.alpha.addEventListener("input", () => {
  state.alpha = Number(els.alpha.value);
  els.alphaVal.textContent = els.alpha.value;
  scheduleSearch();
});
els.expand.addEventListener("change", () => {
  state.expandAliases = els.expand.checked;
  runSearch(false);
});
els.filters.addEventListener("change", (e) => {
  const t = e.target;
  if (t.dataset.facet) {
    const facet = t.dataset.facet;
    const set = new Set(state.filters[facet] || []);
    t.checked ? set.add(t.value) : set.delete(t.value);
    state.filters[facet] = [...set];
    runSearch(false);
  } else if (t === els.sizeMin || t === els.sizeMax) {
    state.filters.sizeMinMB = els.sizeMin.value === "" ? undefined : Number(els.sizeMin.value);
    state.filters.sizeMaxMB = els.sizeMax.value === "" ? undefined : Number(els.sizeMax.value);
    scheduleSearch();
  } else if (t === els.device) {
    state.filters.device = els.device.checked ? "this-device" : undefined;
    runSearch(false);
  }
});
els.clearFilters.addEventListener("click", () => {
  state.filters = {
    status: [],
    task: [],
    modality: [],
    runtime: [],
    backend: [],
    license: [],
    relKind: [],
    tier: [],
  };
  applyStateToControls();
  runSearch(false);
});
els.copyLink.addEventListener("click", async () => {
  writeState(true);
  try {
    await navigator.clipboard.writeText(location.href);
    els.copyLink.textContent = "Copied!";
    setTimeout(() => (els.copyLink.textContent = "Copy link"), 1500);
  } catch {
    els.copyLink.textContent = location.href;
  }
});
els.embedBtn.addEventListener("click", async () => {
  els.embedBtn.disabled = true;
  els.embedProg.hidden = false;
  try {
    await client.request(METHOD.EMBED_DOWNLOAD, {}, {
      onProgress: (p) => {
        els.embedProg.value = p.progress || 0;
        els.embedText.textContent = `${
          p.status === "initialising" ? "Starting" : "Downloading"
        } intent model… ${Math.round(p.progress || 0)}%`;
      },
    });
    els.embedProg.hidden = true;
    await refreshEmbedStatus();
    runSearch(false); // re-rank with semantic now available
  } catch (err) {
    els.embedProg.hidden = true;
    await refreshEmbedStatus();
    els.embedText.textContent = `Intent model failed: ${
      err?.message || err
    }. Keyword search still works.`;
  }
});

window.addEventListener("popstate", () => {
  state = readState();
  applyStateToControls();
  runSearch(true);
});
window.addEventListener("offline", () => {
  if (!indexReady) {
    showError("You're offline and the index isn't cached yet. Reconnect and retry.", boot);
  }
});

function showError(msg, retry) {
  els.count.textContent = "";
  els.results.innerHTML = `<div class="state err"><p>${esc(msg)}</p></div>`;
  if (retry) {
    const b = document.createElement("button");
    b.textContent = "Retry";
    b.style.marginTop = "0.6rem";
    b.addEventListener("click", () => {
      indexReady = false;
      retry();
    });
    els.results.querySelector(".state").append(b);
  }
}

boot();
