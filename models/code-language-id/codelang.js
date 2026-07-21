// Front-end helpers for the Code language ID page. Owns the worker handshake and renders the 6-language
// confidence bars. All inference lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/code-language-id/worker.js";

// The 6 CodeSearchNet languages → a display name + accent hue (colour is decorative; the name is the signal).
export const LANGS = {
  go: { name: "Go", hue: 190 },
  java: { name: "Java", hue: 20 },
  javascript: { name: "JavaScript", hue: 45 },
  php: { name: "PHP", hue: 260 },
  python: { name: "Python", hue: 210 },
  ruby: { name: "Ruby", hue: 0 },
};
export const langName = (l) => (LANGS[l]?.name ?? l);
export const langHue = (l) => (LANGS[l]?.hue ?? 210);

// Verified idiomatic sample snippets — each classifies to its language with high confidence.
export const SAMPLES = {
  python: `import math

def area(r):
    return math.pi * r ** 2

print(area(2))`,
  javascript: `function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}`,
  go: `package main

import "fmt"

func main() {
	nums := []int{1, 2, 3}
	sum := 0
	for _, n := range nums {
		sum += n
	}
	fmt.Printf("sum = %d\\n", sum)
}`,
  java: `import java.util.List;

public class Sum {
  static int total(List<Integer> xs) {
    return xs.stream().mapToInt(Integer::intValue).sum();
  }
}`,
  ruby: `class Stack
  def initialize
    @items = []
  end

  def push(x)
    @items << x
  end
end`,
  php: `<?php
function slugify($s) {
  return strtolower(preg_replace('/[^a-z0-9]+/i', '-', $s));
}
echo slugify('Hello World');`,
};

export function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

export class CodeLangEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  classify(code) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "classify", id, code });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render language labels as accessible confidence bars (name + score), top first. */
export function renderBars(container, labels) {
  container.replaceChildren(...labels.map(({ label, score }, i) => {
    const pct = (score * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "cl-bar" + (i === 0 ? " top" : "");
    row.style.setProperty("--cl-hue", String(langHue(label)));
    const name = document.createElement("span");
    name.className = "cl-name";
    name.textContent = langName(label);
    const track = document.createElement("span");
    track.className = "cl-track";
    const fill = document.createElement("i");
    fill.style.width = Math.max(1.5, score * 100) + "%";
    track.append(fill);
    const val = document.createElement("span");
    val.className = "cl-pct";
    val.textContent = pct + "%";
    row.append(name, track, val);
    return row;
  }));
}

export const CODELANG_CSS = `
.cl-input { font-family: var(--font-mono, monospace); font-size: .86rem; inline-size: 100%; padding: .7rem .8rem;
  border-radius: 8px; min-block-size: 11rem; border: 1px solid var(--border); background: var(--bg-raised);
  color: var(--color); resize: vertical; tab-size: 2; white-space: pre; overflow-wrap: normal; overflow-x: auto; }
.cl-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.cl-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.cl-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.cl-chip:hover, .cl-chip:focus-visible { border-color: var(--accent); }
.cl-verdict { font-size: 1.3rem; font-weight: 700; margin: .4rem 0 .6rem; min-height: 1.6rem; }
.cl-bars { display: flex; flex-direction: column; gap: .3rem; max-width: 28rem; }
.cl-bar { display: grid; grid-template-columns: 7rem 1fr 3rem; align-items: center; gap: .5rem; font-size: .86rem; }
.cl-bar.top .cl-name { font-weight: 700; }
.cl-name { white-space: nowrap; }
.cl-track { height: 9px; border-radius: 5px; background: color-mix(in srgb, var(--color) 12%, transparent); overflow: hidden; }
.cl-track > i { display: block; height: 100%; border-radius: 5px; background: hsl(var(--cl-hue) 70% 55%); }
.cl-pct { font-family: var(--font-mono, monospace); text-align: right; font-size: .8rem; color: var(--muted); }
.cl-bar.top .cl-pct { color: var(--color); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .7rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
