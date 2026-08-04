// Front-end helpers for the Protein mutation oracle. Owns the worker handshake, sequence sanitising, the
// clickable residue grid, and the two result views (predicted residues + per-substitution effect). All
// inference (ESM-2 masked-LM) lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/protein-mutation-oracle/worker.js";

export const AA_NAMES = {
  A: "Alanine", R: "Arginine", N: "Asparagine", D: "Aspartate", C: "Cysteine",
  E: "Glutamate", Q: "Glutamine", G: "Glycine", H: "Histidine", I: "Isoleucine",
  L: "Leucine", K: "Lysine", M: "Methionine", F: "Phenylalanine", P: "Proline",
  S: "Serine", T: "Threonine", W: "Tryptophan", Y: "Tyrosine", V: "Valine",
};
// property class per amino acid → a subtle colour on the residue grid (decorative; the letter is the datum)
const AA_CLASS = {
  D: "acidic", E: "acidic",
  K: "basic", R: "basic", H: "basic",
  S: "polar", T: "polar", N: "polar", Q: "polar", Y: "polar", C: "polar",
  G: "special", P: "special",
  A: "hydro", V: "hydro", L: "hydro", I: "hydro", M: "hydro", F: "hydro", W: "hydro",
};
export const aaClass = (a) => AA_CLASS[a] || "hydro";

// Reference protein sequences (UniProt — sequence data are facts). Mature chains, no initiator Met, so the
// 1-based position matches conventional residue numbering.
export const SAMPLES = {
  hbb: {
    name: "Hemoglobin β (human, HBB)",
    note: "Click position 6 (E). Substituting V is the sickle-cell mutation — watch it score deeply negative.",
    seq:
      "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
  },
  insulin: {
    name: "Insulin (human, INS — proinsulin)",
    note: "The conserved cysteines (C) form the disulphide bonds that hold insulin together — mutating them scores badly.",
    seq:
      "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
  },
  ubiquitin: {
    name: "Ubiquitin (human, UBB)",
    note: "One of the most conserved proteins in biology — the model is confident about almost every position.",
    seq:
      "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  },
  lysozyme: {
    name: "Lysozyme C (human, LYZ)",
    note: "An antibacterial enzyme in tears and saliva. Try its catalytic and buried hydrophobic residues.",
    seq:
      "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
  },
};

export class ProteinEngine {
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
  scan(seq, pos) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "scan", id, seq, pos });
    });
  }
}

/** Keep only the 20 standard amino-acid letters (uppercased); report how many characters were dropped. */
export function sanitizeSequence(text) {
  const up = (text || "").toUpperCase();
  const seq = (up.match(/[ACDEFGHIKLMNPQRSTVWY]/g) || []).join("");
  const dropped = (up.replace(/\s/g, "").length) - seq.length;
  return { seq, dropped };
}

/** Per-substitution log-likelihood ratio vs wild-type, sorted from most→least favourable. */
export function mutationRatios(probs, wtAA) {
  const lnWt = Math.log(probs[wtAA]);
  return Object.keys(probs)
    .map((aa) => ({ aa, p: probs[aa], llr: Math.log(probs[aa]) - lnWt }))
    .sort((a, b) => b.llr - a.llr);
}

/** Render the sequence as a clickable residue grid with a position ruler; marks the selected position. */
export function renderSequenceGrid(container, seq, selectedPos, onPick) {
  container.replaceChildren();
  for (let i = 0; i < seq.length; i++) {
    if (i % 10 === 0) {
      const ruler = document.createElement("span");
      ruler.className = "pg-ruler";
      ruler.textContent = String(i + 1);
      container.append(ruler);
    }
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = `pg-res pg-${aaClass(seq[i])}${i === selectedPos ? " pg-sel" : ""}`;
    cell.textContent = seq[i];
    cell.title = `${AA_NAMES[seq[i]]} ${i + 1}`;
    cell.setAttribute("aria-label", `Position ${i + 1}, ${AA_NAMES[seq[i]]}`);
    cell.addEventListener("click", () => onPick(i));
    container.append(cell);
  }
}

/** Top predicted residues at the masked position (bars); the wild-type residue is marked. */
export function renderPredictions(container, probs, wtAA) {
  const ranked = Object.keys(probs).map((a) => ({ a, p: probs[a] })).sort((x, y) => y.p - x.p).slice(0, 8);
  const max = ranked[0].p || 1;
  container.replaceChildren(...ranked.map(({ a, p }) => {
    const row = document.createElement("div");
    row.className = "pg-bar-row";
    row.innerHTML =
      `<span class="pg-bar-aa${a === wtAA ? " pg-wt" : ""}">${a}</span>` +
      `<span class="pg-bar-track"><span class="pg-bar-fill" style="width:${(p / max * 100).toFixed(1)}%"></span></span>` +
      `<span class="pg-bar-val">${(p * 100).toFixed(1)}%${a === wtAA ? " · wild-type" : ""}</span>`;
    return row;
  }));
}

/** Diverging per-substitution effect bars (green ≥0 tolerated, red <0 likely-damaging); wild-type centred. */
export function renderMutations(container, probs, wtAA, pos) {
  const rows = mutationRatios(probs, wtAA);
  const span = Math.max(1, ...rows.map((r) => Math.abs(r.llr)));
  container.replaceChildren(...rows.map(({ aa, llr }) => {
    const row = document.createElement("div");
    row.className = "pg-mut-row";
    const isWt = aa === wtAA;
    const pct = (Math.abs(llr) / span * 50).toFixed(1);
    const bar = isWt
      ? `<span class="pg-mut-track"><span class="pg-mut-wt"></span></span>`
      : `<span class="pg-mut-track"><span class="pg-mut-fill ${llr < 0 ? "neg" : "pos"}" ` +
        `style="width:${pct}%;${llr < 0 ? "right" : "left"}:50%"></span></span>`;
    row.innerHTML =
      `<span class="pg-mut-aa">${wtAA}${pos + 1}${aa}</span>` + bar +
      `<span class="pg-mut-val">${isWt ? "wild-type" : (llr >= 0 ? "+" : "") + llr.toFixed(2)}</span>`;
    return row;
  }));
}

export const PROTEIN_CSS = `
.pg-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .4rem 0; }
.pg-btn { font: inherit; font-size: .82rem; padding: .3rem .7rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.pg-btn:hover:not([disabled]), .pg-btn:focus-visible { border-color: var(--accent); }
.pg-btn[disabled] { opacity: .5; cursor: default; }
.pg-seqin { width: 100%; box-sizing: border-box; font-family: var(--font-mono, monospace); font-size: .8rem;
  min-height: 4.5rem; resize: vertical; padding: .5rem .6rem; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-raised); color: var(--color); letter-spacing: .04em; }
.pg-grid { display: flex; flex-wrap: wrap; gap: 2px; margin: .6rem 0; padding: .5rem; border: 1px solid var(--border);
  border-radius: 10px; background: var(--bg-raised); max-height: 220px; overflow-y: auto; }
.pg-ruler { flex: 0 0 100%; font-family: var(--font-mono, monospace); font-size: .62rem; color: var(--muted);
  margin: .35rem 0 .05rem; }
.pg-ruler:first-child { margin-top: 0; }
.pg-res { font-family: var(--font-mono, monospace); font-size: .74rem; font-weight: 700; width: 1.35rem; height: 1.5rem;
  border: 1px solid transparent; border-radius: 4px; cursor: pointer; color: #1a1a1a; padding: 0; line-height: 1.5rem; }
.pg-res:hover, .pg-res:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.pg-acidic { background: #f7b7b0; } .pg-basic { background: #a9c8f5; } .pg-polar { background: #a9e5d6; }
.pg-hydro { background: #f0ddac; } .pg-special { background: #d9c2ee; }
.pg-sel { outline: 3px solid var(--accent); outline-offset: 1px; box-shadow: 0 0 0 2px var(--bg) inset; }
.pg-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-top: .8rem; }
@media (max-width: 560px) { .pg-panel { grid-template-columns: 1fr; } }
.pg-panel h4 { margin: 0 0 .5rem; font-size: .9rem; }
.pg-sub { font-size: .75rem; color: var(--muted); margin: 0 0 .5rem; }
.pg-bar-row, .pg-mut-row { display: grid; grid-template-columns: 2.6rem 1fr 5.5rem; align-items: center; gap: .5rem;
  font-family: var(--font-mono, monospace); font-size: .72rem; margin: .18rem 0; }
.pg-bar-aa { font-weight: 700; text-align: center; }
.pg-bar-aa.pg-wt, .pg-mut-aa { color: var(--accent); }
.pg-bar-track { height: .7rem; background: color-mix(in srgb, var(--muted) 22%, transparent); border-radius: 4px; overflow: hidden; }
.pg-bar-fill { display: block; height: 100%; background: var(--accent); }
.pg-bar-val, .pg-mut-val { color: var(--muted); text-align: right; }
.pg-mut-track { position: relative; height: .7rem; background: color-mix(in srgb, var(--muted) 18%, transparent);
  border-radius: 4px; }
.pg-mut-track::before { content: ""; position: absolute; left: 50%; top: -1px; bottom: -1px; width: 1px; background: var(--muted); }
.pg-mut-fill { position: absolute; top: 0; height: 100%; }
.pg-mut-fill.neg { background: #d94c4c; border-radius: 4px 0 0 4px; }
.pg-mut-fill.pos { background: #3fa66a; border-radius: 0 4px 4px 0; }
.pg-mut-wt { position: absolute; left: calc(50% - 3px); top: -2px; width: 6px; height: calc(100% + 4px);
  background: var(--accent); border-radius: 2px; }
.pg-callout { margin: .8rem 0 0; padding: .6rem .8rem; border-left: 3px solid var(--accent);
  background: color-mix(in srgb, var(--accent) 8%, transparent); border-radius: 0 8px 8px 0; font-size: .82rem; }
.pg-legend { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .72rem; color: var(--muted); margin: .4rem 0; }
.pg-legend span { display: inline-flex; align-items: center; gap: .3rem; }
.pg-swatch { width: .8rem; height: .8rem; border-radius: 3px; display: inline-block; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
