// Pure keydown decision logic for the /explore/ search combobox.
//
// Extracted from explore/explore.js so the keyboard contract — including the IME composition
// guard — is deterministically unit-testable without a browser (bead web-ai-showcase-gtr).
//
// modern-web-guidance applied (id: ime-safe-enter-submit, retrieved 2026-10-12 via
// npx -y modern-web-guidance@latest): while an Input Method Editor is composing (Japanese/
// Chinese/Korean), Enter/Arrows/Home/End belong to the IME candidate window — a custom keydown
// handler must early-return on `event.isComposing`, and because Safari delivered compositionend
// BEFORE the confirming Enter keydown in some versions (WebKit bug 165004, so isComposing is
// already false when the keydown handler runs), the guidance's recommended Strategy 1 pairs the
// check with the legacy `keyCode === 229` signal, nested under the key gate. Note on the
// Baseline TODO policy: no TODO(baseline/...) marker is added — KeyboardEvent.isComposing IS
// used directly (Baseline, newly available 2026-09-14), and the 229 check is not a fallback
// replacing it but the documented complement for a browser event-ordering bug.
//
// The function is PURE: no DOM, no navigation, no state mutation. It returns the action for the
// listener to apply, preserving explore.js's pre-gtr semantics exactly for non-composing keys
// (ordinary Enter/Home/End/Arrows/Escape unchanged — the fix must not degrade them).

export const COMBOBOX_ACTIONS = Object.freeze({
  NEXT: "next",
  PREV: "prev",
  HOME: "home",
  END: "end",
  COMMIT: "commit",
  CLEAR: "clear",
});

/**
 * Decide what the combobox should do for one keydown event.
 *
 * @param {KeyboardEvent} event the raw keydown event (only .key, .isComposing, .keyCode are read)
 * @param {{hasOptions: boolean, hasActive: boolean}} state
 *   hasOptions: whether any result option exists (the pre-gtr `activeOptions.length` gate);
 *   hasActive: whether an option is currently highlighted (the pre-gtr `activeIdx >= 0` gate).
 * @returns {{action: string, preventDefault: boolean} | null} null = do nothing (let the
 *   browser/IME have the key).
 */
export function comboboxKeydownDecision(event, { hasOptions, hasActive }) {
  // IME guard FIRST (gtr / ime-safe-enter-submit): during composition every handled key belongs
  // to the IME candidate UI — Enter confirms the composition candidate, Arrows/Home/End move
  // within it. keyCode 229 is the legacy "key not available, IME owns it" signal and ALSO covers
  // Safari's compositionend-before-keydown delivery (isComposing already reset to false).
  if (event.isComposing || event.keyCode === 229) return null;

  // Pre-gtr semantics: with no options on screen the combobox handles nothing.
  if (!hasOptions) return null;

  switch (event.key) {
    case "ArrowDown":
      return { action: COMBOBOX_ACTIONS.NEXT, preventDefault: true };
    case "ArrowUp":
      return { action: COMBOBOX_ACTIONS.PREV, preventDefault: true };
    case "Home":
      return hasActive ? { action: COMBOBOX_ACTIONS.HOME, preventDefault: true } : null;
    case "End":
      return hasActive ? { action: COMBOBOX_ACTIONS.END, preventDefault: true } : null;
    case "Enter":
      return hasActive ? { action: COMBOBOX_ACTIONS.COMMIT, preventDefault: true } : null;
    case "Escape":
      // Pre-gtr semantics: Escape clears the highlight and deliberately does NOT preventDefault.
      return { action: COMBOBOX_ACTIONS.CLEAR, preventDefault: false };
    default:
      return null;
  }
}
