import { WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

const ROOT = "/web-ai-showcase/models/autonlp-gibberish-detector-492513457/";
export const MODEL_ID = "madhurjindal/autonlp-Gibberish-Detector-492513457";
export const REVISION = "76672dd7d3575f68ab980705bcec975cc62de71c";
export const MODEL_BYTES = 267961863;
export const SENTIMENT_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
export const SENTIMENT_REVISION = "0b6928efcb76139cae2c6881d49cda67fe119f42";

class Engine {
  constructor(url, name) {
    this.url = url;
    this.name = name;
    this.client = null;
  }
  async load(onProgress) {
    if (this.client) return this;
    const client = new WorkerClient({
      url: this.url,
      name: this.name,
      maxInFlight: 1,
      maxQueue: 8,
    });
    try {
      await client.ready;
      await client.request("load", {}, { onProgress });
      this.client = client;
      return this;
    } catch (error) {
      await client.terminate().catch(() => {});
      throw error;
    }
  }
  async close() {
    if (!this.client) return;
    await this.client.terminate();
    this.client = null;
  }
}

export class GibberishEngine extends Engine {
  constructor() {
    super(`${ROOT}worker.js`, "gibberish-detector");
  }
  async classify(text, maxLength = 64, options = {}) {
    if (!this.client) throw new Error("Model not loaded");
    const { result } = await this.client.request(
      "classify",
      { text, maxLength },
      { channel: options.channel ?? "classify", signal: options.signal },
    );
    return result;
  }
  async batch(texts, maxLength = 64, options = {}) {
    if (!this.client) throw new Error("Model not loaded");
    const { result } = await this.client.request(
      "batch",
      { texts, maxLength },
      { signal: options.signal, onProgress: options.onProgress },
    );
    return result.rows;
  }
}

export class SentimentEngine extends Engine {
  constructor() {
    super(`${ROOT}sentiment-worker.js`, "gibberish-sentiment");
  }
  async classify(text, options = {}) {
    if (!this.client) throw new Error("Sentiment model not loaded");
    const { result } = await this.client.request("classify", { text }, { signal: options.signal });
    return result;
  }
}

export function policy(result, cleanThreshold = 0.7, allowMild = false) {
  const clean = result.scores.find((row) => row.label === "clean")?.score ?? 0;
  const mild = result.scores.find((row) => row.label === "mild gibberish")?.score ?? 0;
  return {
    accepted: clean >= cleanThreshold || (allowMild && clean + mild >= cleanThreshold),
    clean,
    mild,
    threshold: cleanThreshold,
    allowMild,
  };
}

export function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}
export function escapeHTML(value) {
  return String(value).replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
}

export function renderScores(container, result) {
  container.innerHTML = result.scores.map((row) =>
    `<div class="score-row">
    <span>${escapeHTML(row.label)}</span><progress max="1" value="${row.score}" aria-label="${
      escapeHTML(row.label)
    } ${pct(row.score)}"></progress>
    <strong>${pct(row.score)}</strong><code>${row.logit.toFixed(3)}</code>
  </div>`
  ).join("");
}

export const DEMO_CSS = `
.demo-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(16rem,.8fr);gap:1rem;align-items:start}
.controls{display:grid;gap:.75rem}.controls label{display:grid;gap:.3rem;font-weight:650}.controls textarea{inline-size:100%;box-sizing:border-box;resize:vertical;min-block-size:7rem}
.control-row,.chips,.metrics{display:flex;flex-wrap:wrap;gap:.55rem;align-items:center}.control-row>*{min-inline-size:0}.controls button,.controls select,.controls input[type="range"],.model-loader button{min-block-size:44px}.controls label:has(input[type="checkbox"]) span:last-child{display:flex;align-items:center;gap:.5rem;min-block-size:44px}.controls input[type="checkbox"]{inline-size:24px;block-size:24px;margin:0}
.score-row{display:grid;grid-template-columns:minmax(7rem,1fr) minmax(7rem,2fr) 4.2rem 4.5rem;gap:.55rem;align-items:center;margin-block:.45rem}.score-row progress{inline-size:100%}
.verdict{border-inline-start:.35rem solid var(--accent);padding:.75rem;margin-block:.75rem;background:var(--surface)}.verdict.ok{border-color:var(--good)}.verdict.warn{border-color:var(--bad)}
.token-list{display:flex;flex-wrap:wrap;gap:.35rem}.token-list code{padding:.25rem .4rem;background:var(--surface);overflow-wrap:anywhere}
.results-table{inline-size:100%;border-collapse:collapse}.results-table th,.results-table td{text-align:start;padding:.55rem;border-block-end:1px solid var(--border);vertical-align:top}.table-wrap{overflow-x:auto}
.stage{padding:.8rem;border:1px solid var(--border);border-radius:.6rem}.stage[data-state="pass"]{border-color:var(--good)}.stage[data-state="stop"]{border-color:var(--bad)}
[data-busy="true"]{cursor:progress}.limitations{border-inline-start:.35rem solid var(--bad);padding-inline-start:1rem}
@media(max-width:760px){.demo-grid{grid-template-columns:1fr}.score-row{grid-template-columns:minmax(6.5rem,1fr) minmax(6rem,1.4fr) 3.8rem}.score-row code{grid-column:2/4}.control-row>*{flex:1 1 10rem}.results-table{min-inline-size:38rem}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
`;
