// all-mpnet-base-v2 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/all-mpnet-base-v2 (ONNX export of sentence-transformers/all-mpnet-base-v2), pipeline task
// feature-extraction, WASM backend, int8-quantized (q8), pinned to an immutable revision.
//
// Uses the shared, versioned worker-protocol.js (serveWorker): a single in-flight request, a bounded
// queue, cooperative cancellation via AbortSignal, stale-response suppression, latest-wins channels,
// terminal fatal errors, and deterministic dispose (onDispose). The model download runs as a `load`
// request so per-file progress flows through the protocol's onProgress into the shared loader panel.
// The pooled 768-d vectors are returned as a TRANSFERRED Float64Array buffer (ownership moves; no clone).
//
// The canonical model card specifies attention-mask-aware MEAN pooling followed by L2 normalization,
// with no instruction prefix — query and document are embedded identically (symmetric). We request
// normalize:false so "See inside" can report the real pre-normalization magnitude, then L2-normalize
// here so cosine similarity becomes a plain dot product.

import { serveWorker } from "/web-ai-showcase/lib/worker-protocol.js";
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function l2norm(vec, base, dim) {
  let s = 0;
  for (let j = 0; j < dim; j++) {
    const v = vec[base + j];
    s += v * v;
  }
  return Math.sqrt(s);
}

serveWorker({
  // init is intentionally light — the heavy model download is a `load` request so its progress reaches
  // the shared <model-download-status> panel via the protocol's per-request onProgress.
  async init() {},
  methods: {
    // Download + validate the pinned q8 model on the WASM backend. Progress flows through onProgress.
    async load(_payload, { onProgress }) {
      const loaded = await loadPipeline({
        task: "feature-extraction",
        model: "Xenova/all-mpnet-base-v2",
        backend: "wasm",
        dtype: "q8",
        revision: "e086c5e0b3a57b0ce46dd6d9c0662948860b35f3",
        onProgress: (p) => onProgress?.(p),
      });
      pipe = loaded.pipe;
      device = loaded.device;
      return { result: { device } };
    },
    // payload: { texts:string[] }. Returns { flat:Float64Array (n*dim, unit vectors), norms, dim, n, ms, device }.
    // The flat buffer is TRANSFERRED (not cloned).
    async embed({ texts }, { signal }) {
      if (!pipe) throw new Error("Model not loaded");
      if (signal.aborted) {
        throw signal.reason ?? new DOMException("Aborted", "AbortError");
      }
      const t0 = performance.now();
      // pooling:"mean" → mask-aware average of the per-token vectors (MPNet's trained representation).
      // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
      const out = await pipe(texts, {
        pooling: "mean",
        normalize: false,
        truncation: true,
        max_length: 384,
      });
      // Cooperative cancellation checkpoint after the long inference step.
      if (signal.aborted) {
        throw signal.reason ?? new DOMException("Aborted", "AbortError");
      }
      const dim = out.dims[out.dims.length - 1];
      const src = out.data;
      const n = texts.length;
      const flat = new Float64Array(n * dim);
      const norms = new Array(n);
      for (let i = 0; i < n; i++) {
        const base = i * dim;
        const mag = l2norm(src, base, dim);
        norms[i] = mag;
        const inv = mag || 1;
        for (let j = 0; j < dim; j++) flat[base + j] = src[base + j] / inv;
      }
      const ms = Math.round(performance.now() - t0);
      return {
        result: { flat, norms, dim, n, ms, device },
        transfer: [flat.buffer],
      };
    },
    // k-means clustering over the embeddings, run OFF the main thread. payload: { texts, k }.
    // Returns { assign, centroids(unit), central (index of most-central member per cluster),
    //          cohesion (mean intra-cluster cosine, -1..1), stability (fraction of points that keep
    //          their cluster across two differently-seeded runs, 0..1), ms }.
    async cluster({ texts, k }, { signal }) {
      if (!pipe) throw new Error("Model not loaded");
      if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const t0 = performance.now();
      const out = await pipe(texts, {
        pooling: "mean",
        normalize: true,
        truncation: true,
        max_length: 384,
      });
      if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const dim = out.dims[out.dims.length - 1];
      const src = out.data;
      const n = texts.length;
      const vecs = [];
      for (let i = 0; i < n; i++) {
        const row = new Float64Array(dim);
        for (let j = 0; j < dim; j++) row[j] = src[i * dim + j];
        vecs.push(row);
      }
      const cos = (a, b) => {
        let s = 0;
        for (let j = 0; j < dim; j++) s += a[j] * b[j];
        return s;
      };
      // Deterministic k-means++ with a fixed LCG seed (results reproducible across clicks).
      function runKmeans(seed) {
        const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
        const kk = Math.min(k, n);
        const centroids = [vecs[Math.floor(rnd() * n)].slice()];
        while (centroids.length < kk) {
          const d2 = vecs.map((v) => Math.min(...centroids.map((c) => 1 - cos(v, c))));
          const sum = d2.reduce((a, b) => a + b, 0) || 1;
          let r = rnd() * sum, pick = 0;
          for (let i = 0; i < n; i++) {
            r -= d2[i];
            if (r <= 0) {
              pick = i;
              break;
            }
          }
          centroids.push(vecs[pick].slice());
        }
        let assign = new Array(n).fill(0);
        for (let it = 0; it < 25; it++) {
          let moved = false;
          for (let i = 0; i < n; i++) {
            let best = 0, bs = -Infinity;
            for (let g = 0; g < kk; g++) {
              const s = cos(vecs[i], centroids[g]);
              if (s > bs) {
                bs = s;
                best = g;
              }
            }
            if (assign[i] !== best) {
              assign[i] = best;
              moved = true;
            }
          }
          for (let g = 0; g < kk; g++) {
            const members = [];
            for (let i = 0; i < n; i++) if (assign[i] === g) members.push(i);
            if (!members.length) continue;
            const c = new Float64Array(dim);
            for (const i of members) for (let j = 0; j < dim; j++) c[j] += vecs[i][j];
            const norm = Math.sqrt(c.reduce((a, x) => a + x * x, 0)) || 1;
            for (let j = 0; j < dim; j++) c[j] /= norm;
            centroids[g] = c;
          }
          if (!moved && it > 0) break;
        }
        return assign;
      }
      const a1 = runKmeans(42);
      const a2 = runKmeans(1337);
      // stability: fraction of points assigned identically across two differently-seeded runs.
      let same = 0;
      for (let i = 0; i < n; i++) if (a1[i] === a2[i]) same++;
      const stability = n ? same / n : 0;
      // Recompute centroids for a1 and the most-central member + mean intra-cluster cosine (cohesion).
      const kk = Math.min(k, n);
      const centroids = [];
      const central = [];
      let cohSum = 0, cohCount = 0;
      for (let g = 0; g < kk; g++) {
        const members = [];
        for (let i = 0; i < n; i++) if (a1[i] === g) members.push(i);
        if (!members.length) {
          centroids.push(null);
          central.push(-1);
          continue;
        }
        const c = new Float64Array(dim);
        for (const i of members) for (let j = 0; j < dim; j++) c[j] += vecs[i][j];
        const norm = Math.sqrt(c.reduce((a, x) => a + x * x, 0)) || 1;
        for (let j = 0; j < dim; j++) c[j] /= norm;
        centroids.push(c);
        // most-central member = highest cosine to the centroid.
        let bestI = members[0], bestS = -Infinity;
        for (const i of members) {
          const s = cos(vecs[i], c);
          if (s > bestS) {
            bestS = s;
            bestI = i;
          }
        }
        central.push(bestI);
        // intra-cluster pairwise cosine for cohesion.
        for (let x = 0; x < members.length; x++) {
          for (let y = x + 1; y < members.length; y++) {
            cohSum += cos(vecs[members[x]], vecs[members[y]]);
            cohCount++;
          }
        }
      }
      const cohesion = cohCount ? cohSum / cohCount : 1;
      const ms = Math.round(performance.now() - t0);
      return { result: { assign: a1, k: kk, central, cohesion, stability, ms, device } };
    },
  },
  onDispose() {
    // transformers.js owns its Cache Storage; releasing the pipeline reference lets the WASM heap free.
    pipe = null;
  },
});
