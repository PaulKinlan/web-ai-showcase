// Keep the initial selected chip and every subsequent chip change on the same matching path.
export function parseSoundTargets(value) {
  return value.split(",").map((target) => target.trim().toLowerCase());
}

export function matchingSound(labels, minimum, targets) {
  return labels.find((label) => label.score >= minimum &&
    targets.some((target) => label.label.toLowerCase().includes(target)));
}
