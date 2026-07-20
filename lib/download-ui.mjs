// download-ui.mjs — accessible, low-jitter renderer for a multi-file model download.
//
// Consumes snapshots from lib/download-tracker.mjs and paints a calm, honest download panel:
//   • a high-level PHASE (checking → discovering → downloading → verifying → preparing → ready) — a
//     config reaching 100% never implies the whole model is ready;
//   • a byte-weighted aggregate (real transferred / known-total bytes, or an honest "known so far" /
//     indeterminate state) — never a mean of per-file percentages;
//   • an expandable per-file table (name, state, bytes, per-file bar) — concurrent files update
//     independently; and
//   • real controls only when the action genuinely works (Pause/Resume the resumable prefetch, Discard
//     partials, Retry) plus a storage estimate before a large transfer.
//
// modern-web-guidance retained + applied: performance (native <progress>; updates coalesced to one
// rAF; reserved space + no list reordering ⇒ no CLS/jitter; heavy hashing stays in the worker),
// accessibility (native <progress>/<button>/<details>; ONE polite live region for the phase, debounced;
// real labels; keyboard + touch; readable at 390px/200% zoom).

import { formatBytes, PHASE_LABEL } from "./download-tracker.mjs";

export function createDownloadUI({ mount, sizeMB, controls = {} }) {
  mount.innerHTML = `
    <div class="dl-panel panel" data-phase="checking">
      <p class="dl-phase status" role="status" aria-live="polite"></p>
      <progress class="dl-bar" max="100" aria-label="Overall download progress"></progress>
      <p class="dl-agg muted"></p>
      <div class="dl-actions"></div>
      <details class="dl-files">
        <summary>Show file details</summary>
        <div class="dl-file-list" role="group" aria-label="Per-file download status"></div>
      </details>
      <p class="dl-storage muted"></p>
    </div>`;
  const root = mount.querySelector(".dl-panel");
  const phaseEl = root.querySelector(".dl-phase");
  const bar = root.querySelector(".dl-bar");
  const aggEl = root.querySelector(".dl-agg");
  const actionsEl = root.querySelector(".dl-actions");
  const fileList = root.querySelector(".dl-file-list");
  const storageEl = root.querySelector(".dl-storage");

  let lastPhase = "";
  let raf = 0;
  let pending = null;
  const fileRows = new Map(); // id → {row, state, bytes, bar}

  // Debounce the live-region phase text so a burst of updates isn't announced repeatedly.
  let phaseAnnounceTimer = 0;
  function announcePhase(text) {
    clearTimeout(phaseAnnounceTimer);
    phaseAnnounceTimer = setTimeout(() => {
      phaseEl.textContent = text;
    }, 250);
  }

  function paint(snap) {
    root.dataset.phase = snap.phase;
    const label = PHASE_LABEL[snap.phase] || snap.phase;
    if (snap.phase !== lastPhase) {
      lastPhase = snap.phase;
      announcePhase(label);
    } else if (!phaseEl.textContent) {
      phaseEl.textContent = label;
    }

    const a = snap.aggregate;
    // Aggregate bar: definite % only when we truly have it; else indeterminate (no fake number).
    if (snap.phase === "ready") {
      bar.value = 100;
      bar.removeAttribute("data-indeterminate");
    } else if (a.ratio != null) {
      bar.value = Math.round(a.ratio * 100);
      bar.removeAttribute("data-indeterminate");
    } else {
      bar.removeAttribute("value"); // indeterminate native bar
      bar.setAttribute("data-indeterminate", "true");
    }

    // Honest aggregate text: bytes always; % only if known; "known so far" while the set may grow.
    if (snap.phase === "ready") {
      aggEl.textContent = "";
    } else if (a.indeterminate) {
      aggEl.textContent = `${formatBytes(a.loadedBytes)} downloaded so far (sizes still resolving)`;
    } else if (a.totalBytes) {
      const pct = a.ratio != null ? ` (${Math.round(a.ratio * 100)}%)` : "";
      const soFar = a.knownSoFar ? " known so far" : "";
      aggEl.textContent = `${formatBytes(a.loadedBytes)} of ${
        formatBytes(a.totalBytes)
      }${soFar}${pct} · ${a.complete}/${a.fileCount} files`;
    } else {
      aggEl.textContent = `${formatBytes(a.loadedBytes)} downloaded`;
    }
    if (a.failed) aggEl.textContent += ` · ${a.failed} failed`;

    // Per-file rows — reserved order (by first-seen); update in place (no reordering ⇒ no jitter).
    for (const f of snap.files) {
      let r = fileRows.get(f.id);
      if (!r) {
        const row = document.createElement("div");
        row.className = "dl-file";
        row.innerHTML = `
          <span class="dl-fname"></span>
          <span class="dl-fstate"></span>
          <span class="dl-fbytes"></span>
          <progress class="dl-fbar" max="100"></progress>`;
        fileList.append(row);
        r = {
          row,
          name: row.querySelector(".dl-fname"),
          st: row.querySelector(".dl-fstate"),
          by: row.querySelector(".dl-fbytes"),
          fb: row.querySelector(".dl-fbar"),
        };
        r.name.textContent = displayName(f.name || f.id);
        fileRows.set(f.id, r);
      }
      r.st.textContent = f.state;
      r.st.dataset.state = f.state;
      r.by.textContent = f.total
        ? `${formatBytes(f.loaded)} / ${formatBytes(f.total)}`
        : formatBytes(f.loaded);
      if (f.state === "complete" || f.state === "cached") {
        r.fb.value = 100;
      } else if (f.ratio != null) {
        r.fb.value = Math.round(f.ratio * 100);
      } else {
        r.fb.removeAttribute("value");
      }
      r.row.dataset.state = f.state;
    }
  }

  return {
    /** Coalesce paints to one animation frame (throttle → no layout jitter, no live-region spam). */
    update(snap) {
      pending = snap;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const s = pending;
        pending = null;
        if (s) paint(s);
      });
    },
    setActions(buttons) {
      actionsEl.innerHTML = "";
      for (const b of buttons) {
        if (!b) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = b.label;
        if (b.className) btn.className = b.className;
        if (b.disabled) btn.disabled = true;
        if (b.onClick) btn.addEventListener("click", b.onClick);
        actionsEl.append(btn);
      }
    },
    async showStorage() {
      try {
        const est = navigator.storage?.estimate ? await navigator.storage.estimate() : null;
        const need = sizeMB ? `Needs ~${sizeMB} MB. ` : "";
        if (est) {
          const free = Math.max(0, (est.quota || 0) - (est.usage || 0));
          storageEl.textContent = `${need}${formatBytes(free)} free of ${formatBytes(est.quota)} (${
            formatBytes(est.usage)
          } used).`;
        } else if (need) {
          storageEl.textContent = need;
        }
      } catch { /* estimate unavailable */ }
    },
    setStorageMessage(msg) {
      storageEl.textContent = msg;
    },
  };
}

// Path-safe display name (keep the subfolder so "onnx/decoder…" is distinguishable, escape nothing —
// textContent handles it, but trim a huge path for narrow screens).
function displayName(p) {
  const s = String(p);
  return s.length > 48 ? "…" + s.slice(-46) : s;
}
