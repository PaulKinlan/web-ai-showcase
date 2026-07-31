import {
  fileToDataURL,
  mountLoader,
  parseLabels,
  renderRanking,
} from "/web-ai-showcase/models/fashion-clip/fashion.js";

const CONFIG = {
  type: {
    labels: "a white sneaker, a sports t-shirt, a red evening dress, a leather handbag",
    help:
      "Compare broad garment and accessory types. The label vocabulary is supplied at run time.",
  },
  attributes: {
    labels:
      "white canvas footwear, black leather footwear, a printed white cotton shirt, a red silk garment",
    help:
      "Probe combinations of colour, material and garment form rather than a fixed product class.",
  },
  catalog: {
    labels: "footwear / sneakers, tops / sportswear, dresses / occasionwear, bags / accessories",
    help:
      "Map a product photo into a retailer-owned taxonomy without uploading it or training a new head.",
  },
  search: {
    labels:
      "heritage white tennis shoe, signed team sports shirt, formal red evening look, structured leather work bag",
    help:
      "Treat catalogue descriptions as queries and rank which one best matches the product image.",
  },
  brief: {
    labels:
      "minimal monochrome weekend, archival sports nostalgia, dramatic evening statement, polished office essential",
    help:
      "Use expressive merchandising briefs as labels. The percentages are relative to this exact set of briefs.",
  },
  audit: {
    labels:
      "clean isolated product photo, product worn by a person, busy lifestyle background, text-heavy product image",
    help:
      "Triage whether imagery resembles FashionCLIP's centred product-photo training distribution.",
  },
  domain: {
    labels: "a white sneaker, a sports t-shirt, a red evening dress, a leather handbag",
    help:
      "Run the same labels through FashionCLIP and general CLIP. Divergence reveals the value—and risk—of domain adaptation.",
  },
};

const $ = (id) => document.getElementById(id);
const mode = document.body.dataset.demo;
const config = CONFIG[mode] ?? CONFIG.type;
const isMulti = mode === "domain";
let fashionEngine;
let generalEngine;
let currentImage = "/web-ai-showcase/media/assets/fashion-sneaker.jpg";

$("labels").value = config.labels;
$("demo-help").textContent = config.help;

function setRunnable() {
  $("run").disabled = !(fashionEngine && (!isMulti || generalEngine));
}
mountLoader({
  mount: $("model-loader"),
  kind: "fashion",
  onReady: (engine) => {
    fashionEngine = engine;
    setRunnable();
  },
  onDispose: () => {
    fashionEngine = null;
    setRunnable();
  },
});
if (isMulti) {
  $("general-loader-wrap").hidden = false;
  mountLoader({
    mount: $("general-loader"),
    kind: "general",
    onReady: (engine) => {
      generalEngine = engine;
      setRunnable();
    },
    onDispose: () => {
      generalEngine = null;
      setRunnable();
    },
  });
}

async function chooseSample(button) {
  for (const candidate of document.querySelectorAll("[data-sample]")) {
    candidate.setAttribute("aria-pressed", String(candidate === button));
  }
  currentImage = button.dataset.sample;
  $("preview").src = currentImage;
  $("preview").alt = button.dataset.alt;
  $("run-status").textContent = `Selected ${button.dataset.alt}.`;
}
for (const button of document.querySelectorAll("[data-sample]")) {
  button.addEventListener("click", () => chooseSample(button));
}

$("upload").addEventListener("change", async () => {
  const file = $("upload").files?.[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    $("run-status").textContent = "Choose an image file.";
    $("run-status").classList.add("err");
    return;
  }
  currentImage = await fileToDataURL(file);
  $("preview").src = currentImage;
  $("preview").alt = `Uploaded image: ${file.name}`;
  for (const button of document.querySelectorAll("[data-sample]")) {
    button.setAttribute("aria-pressed", "false");
  }
  $("run-status").textContent = `Selected ${file.name}.`;
  $("run-status").classList.remove("err");
});

$("reset-labels").addEventListener("click", () => {
  $("labels").value = config.labels;
});

function renderInternals(result) {
  $("internals-empty").hidden = true;
  $("internals").hidden = false;
  $("latency").textContent = `${result.elapsedMs} ms`;
  $("backend").textContent = result.device.toUpperCase();
  $("image-shape").textContent = result.imageDimensions
    ? `[${result.imageDimensions.join(" × ")}]`
    : "projection tensor";
  $("text-shape").textContent = result.textDimensions
    ? `[${result.textDimensions.join(" × ")}]`
    : "one projection per label";
  const body = $("tensor-rows");
  body.replaceChildren();
  const order = result.labels.map((_, index) => index).sort((a, b) =>
    result.probabilities[b] - result.probabilities[a]
  );
  for (const index of order) {
    const row = document.createElement("tr");
    for (
      const value of [
        result.labels[index],
        result.cosines[index].toFixed(4),
        result.logits[index].toFixed(3),
        `${(result.probabilities[index] * 100).toFixed(2)}%`,
      ]
    ) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
}

$("run").addEventListener("click", async () => {
  const labels = parseLabels($("labels").value);
  $("run-status").classList.remove("err", "ok");
  if (labels.length < 2) {
    $("run-status").textContent = "Enter at least two distinct labels.";
    $("run-status").classList.add("err");
    return;
  }
  $("run").disabled = true;
  $("run-status").textContent = isMulti
    ? "Running FashionCLIP, then general CLIP…"
    : "Ranking labels in the worker…";
  try {
    const fashion = await fashionEngine.classify(currentImage, labels);
    renderRanking($("results"), fashion, "FashionCLIP ranking");
    renderInternals(fashion);
    if (isMulti) {
      const general = await generalEngine.classify(currentImage, labels);
      renderRanking($("general-results"), general, "General CLIP ranking");
      $("comparison-note").hidden = false;
      const fashionTop =
        fashion.labels[fashion.probabilities.indexOf(Math.max(...fashion.probabilities))];
      const generalTop =
        general.labels[general.probabilities.indexOf(Math.max(...general.probabilities))];
      $("comparison-note").textContent = fashionTop === generalTop
        ? `Both models rank “${fashionTop}” first; inspect the score distribution to see calibration differences.`
        : `The models disagree: FashionCLIP ranks “${fashionTop}” first; general CLIP ranks “${generalTop}” first.`;
    }
    $("run-status").textContent = "Real local inference complete.";
    $("run-status").classList.add("ok");
  } catch (error) {
    $("run-status").textContent = `Inference failed: ${error.message}`;
    $("run-status").classList.add("err");
  } finally {
    setRunnable();
  }
});
