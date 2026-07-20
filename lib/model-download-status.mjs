// model-download-status.mjs — the reusable <model-download-status> custom element (Task 2b · Phase 3).
//
// The site-wide presentation + control BOUNDARY for a model download. It is deliberately NOT the reducer
// and NOT the download orchestration — those stay in separately-testable modules (lib/download-tracker.mjs,
// lib/model-prefetch.mjs / lib/model-download.js, the runtime adapters in lib/download-adapters.mjs). This
// element only: (a) renders a download-tracker SNAPSHOT into calm, accessible, native semantic UI, and
// (b) emits the user's control intents as events. Any Transformers.js / WebLLM / MediaPipe / raw-ORT demo
// can adopt it by feeding snapshots in and handling the intents out.
//
// LIGHT DOM (deliberate): the download panel reuses the site's shared design-system classes (.panel,
// .status, .muted) from public/styles.css and inherits the light/dark theme from the page. Shadow DOM
// would wall those shared styles + theme tokens off (requiring ::part / re-declaration) for no real
// encapsulation win here — the internals are already class-prefixed (.dl-*). Light DOM also keeps the
// element trivially testable (querySelector works without piercing a shadow root) and accessible (the ONE
// polite live region participates in the page's a11y tree directly). The trade-off — page CSS can reach
// in — is acceptable for a first-party design-system component and is documented here on purpose.
//
// Contract (in): a snapshot from lib/download-tracker.mjs (see its typedefs). Phases:
//   checking · discovering · downloading · paused · verifying · initialising · ready · error
// Contract (out): CustomEvent("mds-action", {bubbles, detail:{action}}) where action ∈
//   download · pause · resume · discard · retry · clear
// Plus imperative helpers (update / setActions / showStorage / setStorageMessage) so it is a near drop-in
// for the previous imperative renderer, easing central adoption.
//
// Robustness: multiple instances are independent (no globals); disconnect cancels pending work and
// suppresses stale snapshots (a generation token); reconnect rebuilds; reset() bumps the generation so a
// clear→re-download can't be corrupted by a late snapshot from the previous load.

import { createDownloadUI } from "./download-ui.mjs";

const AUTO = {
  // phase → the standard controls the element derives when it drives its own controls (auto mode).
  // `canPause` (a per-instance capability, e.g. only the resumable prefetch route) gates Pause.
  derive(phase, { canPause, sizeMB }) {
    switch (phase) {
      case "downloading":
        return canPause ? [{ action: "pause", label: "Pause download", secondary: true }] : [];
      case "paused":
        return [
          { action: "resume", label: "Resume download" },
          { action: "discard", label: "Discard partial downloads", secondary: true },
        ];
      case "verifying":
      case "initialising":
        // The model is being built (from_pretrained / engine load) — not abortable — say so honestly.
        return [{
          action: "none",
          label: "Preparing the model — can't pause",
          secondary: true,
          disabled: true,
        }];
      case "ready":
        return [{ action: "clear", label: "Clear cached model", secondary: true }];
      case "error":
        return [{ action: "retry", label: "Retry", secondary: true }];
      default:
        return []; // checking / discovering — nothing actionable yet
    }
  },
};

export class ModelDownloadStatus extends HTMLElement {
  static VERSION = "1.0.0";
  static get observedAttributes() {
    return ["size-mb", "can-pause", "auto-controls"];
  }

  #ui = null;
  #connected = false;
  #gen = 0; // generation token → suppress stale snapshots after disconnect / reset()
  #lastSnapshot = null;
  #lastPhaseControls = ""; // avoid re-rendering identical control sets (no flicker)

  connectedCallback() {
    if (this.#connected) return;
    this.#connected = true;
    // Replace the declarative fallback (if any) with the live panel.
    this.#ui = createDownloadUI({ mount: this, sizeMB: this.#sizeMB() });
    this.#lastPhaseControls = "";
    if (this.#lastSnapshot) this.update(this.#lastSnapshot); // restore state on reconnect
  }

  disconnectedCallback() {
    this.#connected = false;
    this.#gen++; // any in-flight/late update from before disconnect is now stale
    this.#ui = null;
  }

  attributeChangedCallback(name) {
    if (name === "size-mb" && this.#ui) this.#ui.showStorage();
    if (
      (name === "can-pause" || name === "auto-controls") && this.#lastSnapshot &&
      this.#autoControls()
    ) {
      this.#renderAutoControls(this.#lastSnapshot);
    }
  }

  #sizeMB() {
    const v = Number(this.getAttribute("size-mb"));
    return Number.isFinite(v) && v > 0 ? v : undefined;
  }
  #canPause() {
    return this.hasAttribute("can-pause");
  }
  #autoControls() {
    return this.hasAttribute("auto-controls");
  }

  /** Feed a download-tracker snapshot. No-op (stale-suppressed) when disconnected. */
  update(snapshot) {
    this.#lastSnapshot = snapshot;
    if (!this.#connected || !this.#ui) return; // stale suppression / not yet upgraded
    const gen = this.#gen;
    this.#ui.update(snapshot);
    if (gen !== this.#gen) return; // reset()/disconnect happened during this call
    if (this.#autoControls()) this.#renderAutoControls(snapshot);
  }

  /** Convenience property so hosts can do `el.snapshot = snap`. */
  set snapshot(s) {
    this.update(s);
  }
  get snapshot() {
    return this.#lastSnapshot;
  }

  #renderAutoControls(snap) {
    const key = `${snap.phase}|${this.#canPause()}`;
    if (key === this.#lastPhaseControls) return; // only on transition
    this.#lastPhaseControls = key;
    const derived = AUTO.derive(snap.phase, { canPause: this.#canPause(), sizeMB: this.#sizeMB() });
    this.setActions(derived);
  }

  /**
   * Render a control set. Each item: { action, label, disabled?, secondary?, onClick? }. Clicking emits
   * CustomEvent("mds-action", {detail:{action}}) — and also calls onClick if the host provided one (drop-in
   * compatibility with the previous imperative renderer). `action:"none"` is a non-emitting label button.
   */
  setActions(items) {
    if (!this.#ui) return;
    this.#ui.setActions(
      (items || []).filter(Boolean).map((it) => ({
        label: it.label,
        disabled: it.disabled,
        className: it.secondary || it.className === "secondary" ? "secondary" : it.className,
        onClick: (ev) => {
          if (it.onClick) it.onClick(ev);
          if (it.action && it.action !== "none") {
            this.dispatchEvent(
              new CustomEvent("mds-action", { bubbles: true, detail: { action: it.action } }),
            );
          }
        },
      })),
    );
    this.#lastPhaseControls = ""; // an explicit setActions overrides the derived set
  }

  /** Show a "Download (~N MB)" affordance for the absent state (host decides when — preserves auto-init). */
  showDownloadPrompt({ sizeMB, label } = {}) {
    const mb = sizeMB ?? this.#sizeMB();
    this.setActions([{
      action: "download",
      label: label || `Download model${mb ? ` (~${mb} MB)` : ""}`,
    }]);
  }

  /** Bump the generation so any late snapshot from a previous load is ignored (e.g. clear→re-download). */
  reset() {
    this.#gen++;
    this.#lastPhaseControls = "";
    this.#lastSnapshot = null;
  }

  showStorage() {
    return this.#ui?.showStorage();
  }
  setStorageMessage(msg) {
    this.#ui?.setStorageMessage(msg);
  }
}

if (typeof customElements !== "undefined" && !customElements.get("model-download-status")) {
  customElements.define("model-download-status", ModelDownloadStatus);
}
