// Pure acceptance predicates shared by the AST browser driver and its regression tests.
// They are serialized with Function#toString into a CDP Runtime.evaluate expression, so do not
// capture variables outside each function. No browser, DOM, or clock dependency lives here.
export function isFreshMultiTerminal(state) {
  // Tone leaves #rRoute='tagged, no ASR'. A new JFK run resets scores/readout and disables #run;
  // seeing the old route while it is still classifying MUST NOT count as JFK completion.
  if (state.runDisabled || state.status === "AST is listening…") return false;
  if (/Pipeline failed/.test(state.status)) return true; // terminal failure; later assertion fails
  return state.readoutVisible && state.scores > 0 &&
    (state.route === "tagged, no ASR" || (state.route === "→ Whisper" && state.whisperVisible));
}

export function alertWaitTerminal(state) {
  // Wait for the actual visible alert, not merely the first model window. Four complete windows
  // or an explicit page error stop the bounded wait, but can never satisfy the success predicate.
  return !state.hidden || state.count >= 4 || /Classify error/.test(state.status);
}

export function hasSpeechAlert(state) {
  return state.hidden === false && /Heard it/i.test(state.text) && /Speech/i.test(state.text);
}
