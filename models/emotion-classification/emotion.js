// Front-end helpers for the 7-class Ekman emotion pages. Keeps each page thin: it owns the worker
// handshake, the softmax bar chart, and the SVG "emotion wheel" (radar). All inference lives in
// worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/emotion-classification/worker.js";

// The 7 Ekman emotions this DistilRoBERTa head emits, in a stable display order for the radar wheel
// (so the polygon shape is comparable between inputs). Each maps to an affective group (drives colour)
// and a fixed chart colour derived from the design tokens — no raw hex.
export const EMOTION_ORDER = ["joy", "surprise", "neutral", "sadness", "fear", "disgust", "anger"];

export const EMOTION_META = {
  joy: { group: "pos", color: "var(--good)", emoji: "😊" },
  surprise: { group: "amb", color: "var(--warn)", emoji: "😮" },
  neutral: { group: "neu", color: "var(--muted)", emoji: "😐" },
  sadness: {
    group: "neg",
    color: "color-mix(in srgb, var(--accent) 65%, var(--muted))",
    emoji: "😢",
  },
  fear: { group: "neg", color: "var(--accent)", emoji: "😨" },
  disgust: { group: "neg", color: "color-mix(in srgb, var(--bad) 55%, var(--warn))", emoji: "🤢" },
  anger: { group: "neg", color: "var(--bad)", emoji: "😠" },
};

export const GROUP_NAME = { pos: "positive", neg: "negative", amb: "ambiguous", neu: "neutral" };

export function groupOf(label) {
  return EMOTION_META[label]?.group || "neu";
}
export function colorOf(label) {
  return EMOTION_META[label]?.color || "var(--muted)";
}
export function emojiOf(label) {
  return EMOTION_META[label]?.emoji || "";
}

export class EmotionEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result" || msg.type === "batch") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Score one text → { scores:[{label,score}×7 sorted], ms, device }. Softmax: scores sum to ~1. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Score many texts in one pass → { results:[[{label,score}×7]…], ms, device }. */
  classifyBatch(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "batch", id, texts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Split text into sentences/lines for the arc. Keeps it simple + punctuation-aware. */
export function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .match(/[^.!?]+[.!?]*/g)?.map((s) => s.trim()).filter((s) => s.length > 1) ?? [];
}

/** Turn a sorted score array into a {label:score} map for stable-order rendering. */
export function scoreMap(scores) {
  const m = {};
  for (const { label, score } of scores) m[label] = score;
  return m;
}

/**
 * Render a full ranked bar chart of all 7 class scores into `container`. The winner (index 0) is
 * marked `.pass`. Colour encodes the emotion. Because it's a softmax, the bars sum to ~100%.
 */
export function renderScores(container, scores, limit = 7) {
  const rows = scores.slice(0, limit).map(({ label, score }, i) => {
    const pct = (score * 100).toFixed(1);
    return `
      <div class="emo-row${i === 0 ? " pass" : ""}">
        <span class="emo-name">${emojiOf(label)} ${escapeHTML(label)}</span>
        <span class="emo-bar"><span class="emo-fill" style="inline-size:${pct}%;background:${
      colorOf(label)
    }"></span></span>
        <span class="emo-score">${pct}%</span>
      </div>`;
  });
  container.innerHTML = rows.join("");
}

const SVGNS = "http://www.w3.org/2000/svg";
function pt(cx, cy, r, angle) {
  return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
}

/**
 * Render the "emotion wheel": a 7-axis radar (spider) chart of the softmax distribution into an
 * <svg> element. This is the SEE-INSIDE viz — the shape of the whole distribution at a glance, not
 * just the winner. Uses a viewBox so it scales fluidly with no fixed pixel size (no overflow).
 */
export function renderRadar(svg, scores) {
  const size = 240, cx = size / 2, cy = size / 2, R = 92;
  const padX = 46, padY = 18; // room so edge axis labels aren't clipped by the viewBox
  const map = scoreMap(scores);
  const n = EMOTION_ORDER.length;
  svg.setAttribute("viewBox", `${-padX} ${-padY} ${size + 2 * padX} ${size + 2 * padY}`);
  svg.setAttribute("role", "img");
  const winner = scores[0]?.label;
  svg.setAttribute(
    "aria-label",
    `Emotion wheel. Winning emotion ${winner} at ${
      ((scores[0]?.score ?? 0) * 100).toFixed(0)
    } percent.`,
  );
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const angleFor = (i) => (-Math.PI / 2) + (i * 2 * Math.PI / n);

  // grid rings
  for (const frac of [0.25, 0.5, 0.75, 1]) {
    const ring = document.createElementNS(SVGNS, "polygon");
    ring.setAttribute(
      "points",
      EMOTION_ORDER.map((_, i) =>
        pt(cx, cy, R * frac, angleFor(i)).map((v) => v.toFixed(1)).join(",")
      )
        .join(" "),
    );
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "var(--border)");
    ring.setAttribute("stroke-width", "1");
    svg.appendChild(ring);
  }
  // axes + labels
  EMOTION_ORDER.forEach((label, i) => {
    const [ax, ay] = pt(cx, cy, R, angleFor(i));
    const axis = document.createElementNS(SVGNS, "line");
    axis.setAttribute("x1", cx);
    axis.setAttribute("y1", cy);
    axis.setAttribute("x2", ax.toFixed(1));
    axis.setAttribute("y2", ay.toFixed(1));
    axis.setAttribute("stroke", "var(--border)");
    axis.setAttribute("stroke-width", "1");
    svg.appendChild(axis);
    const [lx, ly] = pt(cx, cy, R + 16, angleFor(i));
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", lx.toFixed(1));
    t.setAttribute("y", ly.toFixed(1));
    t.setAttribute("text-anchor", Math.abs(lx - cx) < 6 ? "middle" : lx < cx ? "end" : "start");
    t.setAttribute("dominant-baseline", "middle");
    t.setAttribute("font-size", "11");
    t.setAttribute("fill", "var(--muted)");
    t.textContent = label;
    svg.appendChild(t);
  });
  // the data polygon
  const poly = document.createElementNS(SVGNS, "polygon");
  poly.setAttribute(
    "points",
    EMOTION_ORDER.map((label, i) => {
      const v = map[label] ?? 0;
      return pt(cx, cy, R * Math.min(1, v), angleFor(i)).map((x) => x.toFixed(1)).join(",");
    }).join(" "),
  );
  poly.setAttribute("fill", "color-mix(in srgb, var(--accent) 28%, transparent)");
  poly.setAttribute("stroke", "var(--accent)");
  poly.setAttribute("stroke-width", "2");
  svg.appendChild(poly);
  // vertices
  EMOTION_ORDER.forEach((label, i) => {
    const v = map[label] ?? 0;
    const [x, y] = pt(cx, cy, R * Math.min(1, v), angleFor(i));
    const dot = document.createElementNS(SVGNS, "circle");
    dot.setAttribute("cx", x.toFixed(1));
    dot.setAttribute("cy", y.toFixed(1));
    dot.setAttribute("r", label === winner ? "4" : "2.5");
    dot.setAttribute("fill", colorOf(label));
    svg.appendChild(dot);
  });
}

export const EMOTION_CSS = `
.emo-list { display:flex; flex-direction:column; gap:.32rem; margin-top:.6rem; }
.emo-row { display:grid; grid-template-columns:7.5rem 1fr 3.4rem; gap:.55rem; align-items:center; }
.emo-name { font-size:.85rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.emo-bar { block-size:.72rem; background:var(--bg-secondary); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; min-inline-size:0; }
.emo-fill { display:block; block-size:100%; border-radius:999px; transition:inline-size .18s ease; }
.emo-score { font-family:var(--font-mono); font-size:.76rem; color:var(--muted); text-align:end; }
.emo-row.pass .emo-name { color:var(--color); font-weight:600; }
.emo-row.pass .emo-score { color:var(--color); }
.emo-pills { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }
.emo-pill { font-family:var(--font-mono); font-size:.8rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); }
.emo-pill b { font-weight:700; }
.emo-pill.pos { color:var(--good); border-color:var(--good); }
.emo-pill.neg { color:var(--bad); border-color:var(--bad); }
.emo-pill.amb { color:var(--warn); border-color:var(--warn); }
.emo-pill.neu { color:var(--muted); }
.verdict-row { display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin-top:.6rem; }
.verdict-label { font-family:var(--font-display); font-size:1.8rem; text-transform:capitalize; }
.verdict-label.pos { color:var(--good); } .verdict-label.neg { color:var(--bad); }
.verdict-label.amb { color:var(--warn); } .verdict-label.neu { color:var(--muted); }
.verdict-conf { font-family:var(--font-mono); color:var(--muted); font-size:.9rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.3rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.2rem; }
.chip:hover { border-color:var(--accent); }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.wheel-wrap { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:1.2rem;
  align-items:center; margin-top:.6rem; }
.wheel-wrap svg { inline-size:100%; max-inline-size:280px; block-size:auto; margin-inline:auto; display:block; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; text-transform:capitalize; }
.ticket { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.55rem .75rem;
  display:flex; justify-content:space-between; gap:.6rem; align-items:center; margin-top:.5rem; flex-wrap:wrap; }
.ticket .msg { flex:1 1 200px; min-inline-size:0; }
.ticket.route-escalate { border-inline-start:4px solid var(--bad); }
.ticket.route-empathy { border-inline-start:4px solid var(--accent); }
.ticket.route-celebrate { border-inline-start:4px solid var(--good); }
.ticket.route-standard { border-inline-start:4px solid var(--muted); }
.ticket .route { font-family:var(--font-mono); font-size:.72rem; white-space:nowrap; }
.arc-step { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .7rem;
  margin-top:.5rem; border-inline-start:4px solid var(--muted); }
.arc-step .meta { font-family:var(--font-mono); font-size:.72rem; margin-bottom:.2rem; }
.arc-sentence { font-size:.95rem; }
`;
