// Grounding DINO (tiny) phrase-grounding worker — runs ALL inference off the main thread. Unlike a
// fixed-class detector, Grounding DINO grounds FREE-TEXT PHRASES to boxes: give it a caption made of
// phrases ("a cat. a remote control. a laptop.") and it boxes every region matching each phrase. One
// forward pass per prompt returns every candidate above a low floor; the page filters that cached list
// by the score slider client-side, so dragging the slider never re-runs the model.
//
// Model: onnx-community/grounding-dino-tiny-ONNX (task: zero-shot-object-detection), WASM backend, q8.
// This is IDEA-Research/grounding-dino-tiny exported to ONNX for Transformers.js. Boxes come back in
// ORIGINAL image pixel coordinates (percentage:false). Shared loader from webai.js.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// Run detection at this floor once per prompt; the UI slider filters the cached result upward. Grounding
// DINO's own default is 0.3; we floor lower so the slider has candidates to reveal.
const FLOOR = 0.1;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "zero-shot-object-detection",
    model: "onnx-community/grounding-dino-tiny-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function run(id, imageURL, phrases) {
  await ensureLoaded();
  // Grounding DINO grounds ONE caption at a time: it tokenises a single string of period-separated
  // phrases (batch size 1) and grounds each phrase within it. Passing an array of phrases batches the
  // text input (batch = phrase count) which the ONNX export rejects ("input_ids ... Got: N Expected:
  // 1"). So join the phrases into a single lowercased, period-terminated caption. parseQueries()
  // already lowercased + stripped periods, so re-add exactly one "." per phrase.
  const caption = phrases.map((p) => (p.endsWith(".") ? p : p + ".")).join(" ");
  const t0 = performance.now();
  // percentage:false → boxes in absolute pixel coordinates of the source image.
  const output = await pipe(imageURL, [caption], { threshold: FLOOR, percentage: false });
  const detections = output.map((d) => {
    const rawLabel = String(d.label).replace(/\.+$/, "").trim();
    // Grounding DINO labels a box with the UNION of the caption tokens it grounded (e.g. "a cat.
    // remote"), because one region can activate several phrase tokens. For clean per-phrase colouring
    // and counts we attribute each box to the input phrase whose words overlap the label most; the
    // honest raw union is kept in rawLabel for the "See inside" table.
    let label = rawLabel, best = -1;
    const lw = new Set(rawLabel.toLowerCase().split(/\W+/).filter(Boolean));
    for (const p of phrases) {
      const pw = p.toLowerCase().split(/\W+/).filter(Boolean);
      if (!pw.length) continue;
      const overlap = pw.filter((w) => lw.has(w)).length / pw.length;
      if (overlap > best) {
        best = overlap;
        label = p;
      }
    }
    return {
      label, // clean input phrase (best word-overlap) — used for colour, counts, summaries
      rawLabel, // exactly what the model returned (union of grounded tokens)
      score: d.score,
      box: { xmin: d.box.xmin, ymin: d.box.ymin, xmax: d.box.xmax, ymax: d.box.ymax },
    };
  });
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, detections, queries: phrases, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.queries);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
