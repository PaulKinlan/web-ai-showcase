// Deterministic event-state tests for the explore combobox keydown decision (web-ai-showcase-gtr).
//
// No browser, no DOM: the decision function is pure. Each test constructs KeyboardEvent-shaped
// objects with exactly the properties real browsers deliver in the state under test — including
// the IME composition states from modern-web-guidance ime-safe-enter-submit (Strategy 1):
//   * composing Enter: isComposing=true (key "Enter", keyCode 13) — the IME confirms a candidate.
//   * Safari race Enter: compositionend already delivered BEFORE the confirming keydown
//     (WebKit bug 165004), so isComposing has been reset to false and only keyCode 229 remains.
//   * ordinary keys: isComposing false/absent, real keyCode — pre-gtr behavior must be unchanged.
import { test } from "node:test";
import assert from "node:assert/strict";
import { comboboxKeydownDecision, COMBOBOX_ACTIONS } from "../explore/combobox-keys.js";

/** A minimal KeyboardEvent-shaped object; only key/isComposing/keyCode are read. */
const key = (k, { isComposing, keyCode } = {}) => ({
  key: k,
  ...(isComposing !== undefined ? { isComposing } : {}),
  ...(keyCode !== undefined ? { keyCode } : {}),
});

const ACTIVE = { hasOptions: true, hasActive: true };
const NO_ACTIVE = { hasOptions: true, hasActive: false };
const NO_OPTIONS = { hasOptions: false, hasActive: false };

test("gtr repro: Enter during IME composition does NOT commit (isComposing=true)", () => {
  // The finding: pressing Enter to confirm a CJK/Japanese/Korean candidate navigated away.
  assert.equal(comboboxKeydownDecision(key("Enter", { isComposing: true, keyCode: 13 }), ACTIVE), null);
});

test("Safari race: Enter after compositionend was delivered first (isComposing reset, keyCode 229)", () => {
  // WebKit bug 165004: compositionend reaches script before the confirming keydown, so
  // isComposing is already false — only the legacy 229 signal distinguishes the IME's Enter.
  assert.equal(comboboxKeydownDecision(key("Enter", { isComposing: false, keyCode: 229 }), ACTIVE), null);
});

test("IME composition owns Arrows/Home/End too, not just Enter", () => {
  for (const k of ["ArrowDown", "ArrowUp", "Home", "End"]) {
    assert.equal(comboboxKeydownDecision(key(k, { isComposing: true }), ACTIVE), null, k);
    assert.equal(comboboxKeydownDecision(key(k, { keyCode: 229 }), ACTIVE), null, `${k} (229)`);
  }
});

test("compositionend race modelled as a sequence: IME Enter ignored, then a real Enter commits", () => {
  // The full Safari timeline, deterministically: composition is active, ends (compositionend
  // delivered), the SAME physical Enter that ended it arrives as keydown 229 -> ignored; the
  // user's NEXT Enter is a genuine combobox commit.
  const duringComposition = comboboxKeydownDecision(key("Enter", { isComposing: true, keyCode: 13 }), ACTIVE);
  const confirmingEnter = comboboxKeydownDecision(key("Enter", { isComposing: false, keyCode: 229 }), ACTIVE);
  const nextRealEnter = comboboxKeydownDecision(key("Enter", { isComposing: false, keyCode: 13 }), ACTIVE);
  assert.equal(duringComposition, null);
  assert.equal(confirmingEnter, null);
  assert.deepEqual(nextRealEnter, { action: COMBOBOX_ACTIONS.COMMIT, preventDefault: true });
});

test("ordinary Enter still commits with preventDefault (no degradation)", () => {
  assert.deepEqual(
    comboboxKeydownDecision(key("Enter", { isComposing: false, keyCode: 13 }), ACTIVE),
    { action: COMBOBOX_ACTIONS.COMMIT, preventDefault: true },
  );
  // isComposing absent entirely (older event shapes): still an ordinary Enter.
  assert.deepEqual(
    comboboxKeydownDecision(key("Enter", { keyCode: 13 }), ACTIVE),
    { action: COMBOBOX_ACTIONS.COMMIT, preventDefault: true },
  );
});

test("ordinary Home/End/Arrows keep their pre-gtr actions (no degradation)", () => {
  assert.deepEqual(comboboxKeydownDecision(key("ArrowDown"), ACTIVE), { action: COMBOBOX_ACTIONS.NEXT, preventDefault: true });
  assert.deepEqual(comboboxKeydownDecision(key("ArrowUp"), ACTIVE), { action: COMBOBOX_ACTIONS.PREV, preventDefault: true });
  assert.deepEqual(comboboxKeydownDecision(key("Home"), ACTIVE), { action: COMBOBOX_ACTIONS.HOME, preventDefault: true });
  assert.deepEqual(comboboxKeydownDecision(key("End"), ACTIVE), { action: COMBOBOX_ACTIONS.END, preventDefault: true });
});

test("pre-gtr gating preserved: Home/End/Enter need an active highlight; Arrows do not", () => {
  assert.equal(comboboxKeydownDecision(key("Home"), NO_ACTIVE), null);
  assert.equal(comboboxKeydownDecision(key("End"), NO_ACTIVE), null);
  assert.equal(comboboxKeydownDecision(key("Enter"), NO_ACTIVE), null);
  assert.deepEqual(comboboxKeydownDecision(key("ArrowDown"), NO_ACTIVE), { action: COMBOBOX_ACTIONS.NEXT, preventDefault: true });
  assert.deepEqual(comboboxKeydownDecision(key("ArrowUp"), NO_ACTIVE), { action: COMBOBOX_ACTIONS.PREV, preventDefault: true });
});

test("Escape still clears WITHOUT preventDefault (pre-gtr semantics)", () => {
  const d = comboboxKeydownDecision(key("Escape"), ACTIVE);
  assert.deepEqual(d, { action: COMBOBOX_ACTIONS.CLEAR, preventDefault: false });
});

test("no options on screen: nothing is handled, composed or not", () => {
  for (const k of ["Enter", "ArrowDown", "Home", "End", "Escape"]) {
    assert.equal(comboboxKeydownDecision(key(k), NO_OPTIONS), null, k);
  }
});

test("unhandled keys and non-Enter modifiers pass through untouched", () => {
  for (const k of ["a", "Tab", "PageDown", " "]) {
    assert.equal(comboboxKeydownDecision(key(k), ACTIVE), null, k);
  }
});
