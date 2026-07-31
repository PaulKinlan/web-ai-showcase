import { createModelLoader } from "/web-ai-showcase/lib/model-loader.js";
import {
  DEMO_CSS,
  escapeHTML,
  GibberishEngine,
  MODEL_BYTES,
  MODEL_ID,
  pct,
  policy,
  renderScores,
  REVISION,
  SENTIMENT_ID,
  SENTIMENT_REVISION,
  SentimentEngine,
} from "./gibberish.js";

const style = document.createElement("style");
style.textContent = DEMO_CSS;
document.head.append(style);
const $ = (id) => document.getElementById(id);
const mode = document.body.dataset.mode;
let detector = new GibberishEngine();
let detectorReady = false;
let sentiment = null;
let sentimentReady = false;

window.__webaiEvidence = {
  sourceModel: MODEL_ID,
  revision: REVISION,
  bytes: MODEL_BYTES,
  mode,
  runs: [],
  controlsChanged: [],
  lifecycle: [],
};
const evidence = window.__webaiEvidence;
const runButtons = () => [...document.querySelectorAll("[data-run]")];
function setRunEnabled() {
  const enabled = detectorReady && (mode !== "multimodel" || sentimentReady);
  for (const button of runButtons()) button.disabled = !enabled;
}
function lifecycle(event, detail = {}) {
  evidence.lifecycle.push({ event, at: Date.now(), ...detail });
}
function changed(control, value) {
  evidence.controlsChanged.push({ control, value: String(value), at: Date.now() });
}
function status(message, error = false) {
  const node = $("run-status");
  if (!node) return;
  node.hidden = false;
  node.textContent = message;
  node.classList.toggle("err", error);
  node.classList.toggle("ok", !error);
}

createModelLoader({
  mount: $("model-loader"),
  model: {
    modelId: "madhurjindal/autonlp-Gibberish-Detector-492513457",
    revision: REVISION,
    runtime: "transformers.js",
    dtype: "fp32",
    sizeMB: 267.962,
    requiresWebGPU: false,
  },
  load: async (onProgress) => {
    await detector.load(onProgress);
    return detector;
  },
  dispose: async (instance) => instance.close(),
  onReady: () => {
    detectorReady = true;
    lifecycle("detector-ready");
    setRunEnabled();
  },
  onDispose: () => {
    detectorReady = false;
    lifecycle("detector-disposed");
    setRunEnabled();
  },
  onError: (error) => status(`Detector failed: ${error.message}`, true),
});

if (mode === "multimodel") {
  sentiment = new SentimentEngine();
  createModelLoader({
    mount: $("sentiment-loader"),
    model: {
      modelId: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
      revision: SENTIMENT_REVISION,
      runtime: "transformers.js",
      dtype: "q8",
      sizeMB: 67.581,
      requiresWebGPU: false,
    },
    load: async (onProgress) => {
      await sentiment.load(onProgress);
      return sentiment;
    },
    dispose: async (instance) => instance.close(),
    onReady: () => {
      sentimentReady = true;
      lifecycle("sentiment-ready");
      setRunEnabled();
    },
    onDispose: () => {
      sentimentReady = false;
      lifecycle("sentiment-disposed");
      setRunEnabled();
    },
    onError: (error) => status(`Sentiment stage failed: ${error.message}`, true),
  });
}
setRunEnabled();

for (const input of document.querySelectorAll(".controls textarea, [data-evidence-control]")) {
  const recordChange = () =>
    changed(input.id || input.name, input.type === "checkbox" ? input.checked : input.value);
  input.addEventListener("change", recordChange);
  if (input.matches("textarea")) input.addEventListener("input", recordChange);
}
for (const button of document.querySelectorAll("[data-sample]")) {
  button.addEventListener("click", () => {
    $(button.dataset.target || "text").value = button.dataset.sample;
    changed("sample", button.textContent.trim());
  });
}

function maxLength() {
  return Number($("max-length")?.value || 64);
}
function threshold() {
  return Number($("clean-threshold")?.value || 0.7);
}
function allowMild() {
  return Boolean($("allow-mild")?.checked);
}
function renderDecision(result) {
  const decision = policy(result, threshold(), allowMild());
  const node = $("verdict");
  node.hidden = false;
  node.className = `verdict ${decision.accepted ? "ok" : "warn"}`;
  node.innerHTML = `<strong>${
    decision.accepted ? "Accepted by this policy" : "Review / ask for clearer text"
  }</strong><br>
    Top model label: <code>${escapeHTML(result.top.label)}</code> (${
    pct(result.top.score)
  }). Policy used clean ≥ ${pct(decision.threshold)}${
    decision.allowMild ? " or clean + mild above threshold" : ""
  }.`;
  return decision;
}

async function classifyOne(text, routeMode = mode) {
  if (!text.trim()) throw new Error("Enter some text first.");
  status("Running locally in a worker…");
  const result = await detector.classify(text, maxLength());
  const decision = renderDecision(result);
  if ($("scores")) renderScores($("scores"), result);
  if ($("tokens")) {
    $("tokens").innerHTML = result.tokens.map((token, index) =>
      `<code title="token id ${result.tokenIds[index]}">${escapeHTML(token || "∅")}</code>`
    ).join("");
  }
  if ($("internals")) {
    $("internals").textContent = `logits tensor [${
      result.tensorShape.join(", ")
    }] · ${result.tokens.length} tokens${
      result.truncated ? " (at selected limit)" : ""
    } · ${result.ms} ms · ${result.backend.toUpperCase()} · ${result.dtype}`;
  }
  evidence.runs.push({
    routeMode,
    text,
    maxLength: result.maxLength,
    top: result.top,
    scores: result.scores,
    decision,
    tensorShape: result.tensorShape,
    tokens: result.tokens,
    ms: result.ms,
  });
  status(`Finished in ${result.ms} ms on ${result.backend.toUpperCase()}.`);
  return { result, decision };
}

async function guarded(action) {
  document.body.dataset.busy = "true";
  for (const button of runButtons()) button.disabled = true;
  try {
    await action();
  } catch (error) {
    status(`Inference failed: ${error.message}`, true);
  } finally {
    document.body.dataset.busy = "false";
    setRunEnabled();
  }
}

if (mode === "overview" || mode === "basics") {
  $("run").addEventListener("click", () => guarded(() => classifyOne($("text").value)));
}

if (mode === "practical") {
  $("run").addEventListener("click", () =>
    guarded(async () => {
      const texts = $("text-lines").value.split(/\n/).map((line) => line.trim()).filter(Boolean)
        .slice(0, 12);
      if (!texts.length) throw new Error("Add at least one line.");
      status(`Classifying ${texts.length} lines off the main thread…`);
      const rows = await detector.batch(texts, maxLength());
      const decisions = rows.map((result) => ({
        result,
        decision: policy(result, threshold(), allowMild()),
      }));
      $("queue-body").innerHTML = decisions.map(({ result, decision }, index) =>
        `<tr><td>${index + 1}</td><td>${escapeHTML(result.text)}</td><td><code>${
          escapeHTML(result.top.label)
        }</code><br>${pct(result.top.score)}</td><td><strong>${
          decision.accepted ? "accept" : "review"
        }</strong></td><td>${result.ms} ms</td></tr>`
      ).join("");
      $("queue-wrap").hidden = false;
      evidence.runs.push({
        routeMode: mode,
        batch: decisions.map(({ result, decision }) => ({
          text: result.text,
          top: result.top,
          scores: result.scores,
          decision,
        })),
        maxLength: maxLength(),
      });
      status(`Classified ${rows.length} lines locally. No text left this device.`);
    }));
}

if (mode === "wild") {
  const transform = (text) => {
    const strength = Number($("strength").value);
    const kind = $("perturbation").value;
    if (kind === "symbols") {
      return text.split(" ").map((word, index) =>
        index % Math.max(2, 6 - strength) === 0
          ? word.replace(
            /[aeios]/gi,
            (char) => ({ a: "@", e: "3", i: "1", o: "0", s: "$" })[char.toLowerCase()] || char,
          )
          : word
      ).join(" ");
    }
    if (kind === "codes") {
      return `${text} ERR_AUTH_${40 + strength} https://example.com/reset?t=${
        "x".repeat(strength * 3)
      }`;
    }
    const words = text.split(/\s+/);
    return words.map((_, index) => words[(index * (strength + 2)) % words.length]).join(" ");
  };
  $("run").addEventListener("click", () =>
    guarded(async () => {
      const original = $("text").value.trim();
      const altered = transform(original);
      $("altered").textContent = altered;
      const before = await detector.classify(original, maxLength());
      const after = await detector.classify(altered, maxLength());
      $("before-label").textContent = `${before.top.label} ${pct(before.top.score)}`;
      $("after-label").textContent = `${after.top.label} ${pct(after.top.score)}`;
      $("wild-results").hidden = false;
      evidence.runs.push({
        routeMode: mode,
        original,
        altered,
        perturbation: $("perturbation").value,
        strength: Number($("strength").value),
        before,
        after,
      });
      status(`Compared two real inferences in ${before.ms + after.ms} ms total.`);
    }));
}

if (mode === "multimodel") {
  $("run").addEventListener("click", () =>
    guarded(async () => {
      const text = $("text").value.trim();
      if (!text) throw new Error("Enter some text first.");
      status("Stage 1: checking coherence locally…");
      const first = await detector.classify(text, maxLength());
      const decision = policy(first, threshold(), allowMild());
      $("gate-stage").dataset.state = decision.accepted ? "pass" : "stop";
      $("gate-output").textContent = `${first.top.label} ${pct(first.top.score)} · clean ${
        pct(decision.clean)
      } · ${decision.accepted ? "passed" : "stopped"}`;
      let second = null;
      if (decision.accepted) {
        status("Stage 2: accepted text is now reaching the sentiment worker…");
        second = await sentiment.classify(text);
        $("sentiment-stage").dataset.state = "pass";
        $("sentiment-output").textContent = second.scores.map((row) =>
          `${row.label} ${pct(row.score)}`
        ).join(" · ");
      } else {
        $("sentiment-stage").dataset.state = "stop";
        $("sentiment-output").textContent =
          "Not run — the detector policy stopped this input. No substituted or canned sentiment result.";
      }
      evidence.runs.push({
        routeMode: mode,
        text,
        maxLength: maxLength(),
        detector: first,
        decision,
        sentiment: second,
      });
      status(
        second
          ? `Both real model stages finished locally (${first.ms + second.ms} ms).`
          : `Gate stopped the pipeline after a real detector inference (${first.ms} ms).`,
      );
    }));
}

window.addEventListener("pagehide", () => {
  void detector.close();
  if (sentiment) void sentiment.close();
});
