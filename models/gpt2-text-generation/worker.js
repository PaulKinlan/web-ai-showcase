// GPT-2 worker — autoregressive text generation off the main thread, with the next-token distribution
// exposed at every step. Model: Xenova/gpt2 (task: text-generation), WASM, int8.
//
// GPT-2 is the original decoder-only transformer: it models P(next token | all previous tokens) over a
// 50,257-token byte-level BPE vocabulary, and generates by sampling from that distribution one token at a
// time, feeding each choice back in. We run our OWN decode loop (rather than the pipeline's generate) so
// we can read the FULL probability distribution at each step — that's what powers the "see inside" surface
// (watch the model pick) and the interactive "choose the next token" game. Every number is the model's
// real output; nothing is faked.
//
// Operations:
//   generate     → stream a completion token-by-token; each step reports the chosen token, its probability,
//                  and the top-k candidates it was chosen from. Greedy or temperature+top-k sampling.
//   distribution → a single forward pass over the current text; returns the top-k next-token candidates
//                  with probabilities (used by the interactive branching demo).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/gpt2";
let tokenizer = null;
let model = null;
let Tensor = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForCausalLM, Tensor: T, env } = await import(TRANSFORMERS_URL);
  Tensor = T;
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the service worker.
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: "int8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

function makeInputs(ids) {
  const big = BigInt64Array.from(ids, (v) => BigInt(v));
  const ones = BigInt64Array.from(ids, () => 1n);
  return {
    input_ids: new Tensor("int64", big, [1, ids.length]),
    attention_mask: new Tensor("int64", ones, [1, ids.length]),
  };
}

// Softmax the final-position logits row and return {topk, probsForSampling, vocab}.
function lastRow(logits) {
  const [, seq, vocab] = logits.dims;
  const data = logits.data;
  const start = (seq - 1) * vocab;
  const row = data.subarray(start, start + vocab);
  return { row, vocab };
}

// Top-k indices of a row (descending), without sorting the whole vocab.
function topkIndices(row, vocab, k) {
  const best = [];
  for (let i = 0; i < vocab; i++) {
    const v = row[i];
    if (best.length < k) {
      best.push(i);
      if (best.length === k) best.sort((a, b) => row[a] - row[b]);
    } else if (v > row[best[0]]) {
      best[0] = i;
      let j = 0;
      while (j < k - 1 && row[best[j]] > row[best[j + 1]]) {
        const t = best[j];
        best[j] = best[j + 1];
        best[j + 1] = t;
        j++;
      }
    }
  }
  best.sort((a, b) => row[b] - row[a]);
  return best;
}

function softmaxOver(row, indices, temperature = 1) {
  const t = Math.max(1e-6, temperature);
  let max = -Infinity;
  for (const i of indices) if (row[i] / t > max) max = row[i] / t;
  let sum = 0;
  const exps = indices.map((i) => {
    const e = Math.exp(row[i] / t - max);
    sum += e;
    return e;
  });
  return exps.map((e) => e / sum);
}

// Full-distribution probability of a single index (for the "chosen token" confidence, temperature 1).
function probOf(row, vocab, idx) {
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  for (let i = 0; i < vocab; i++) sum += Math.exp(row[i] - max);
  return Math.exp(row[idx] - max) / sum;
}

function decode(id) {
  return tokenizer.decode([id]);
}

async function generate(id, prompt, opts) {
  await ensureLoaded();
  const maxNew = Math.min(80, Math.max(1, opts.maxNew ?? 40));
  const greedy = !!opts.greedy;
  const temperature = opts.temperature ?? 0.9;
  const topK = Math.min(50, Math.max(1, opts.topK ?? 40));
  const enc = await tokenizer(prompt);
  let ids = Array.from(enc.input_ids.data, Number);
  const eos = model.config?.eos_token_id ?? 50256;
  const t0 = performance.now();
  let generated = "";
  for (let step = 0; step < maxNew; step++) {
    const { logits } = await model(makeInputs(ids));
    const { row, vocab } = lastRow(logits);
    const kIdx = topkIndices(row, vocab, Math.max(topK, 8));
    let chosen;
    if (greedy) {
      chosen = kIdx[0];
    } else {
      const cand = kIdx.slice(0, topK);
      const probs = softmaxOver(row, cand, temperature);
      let r = Math.random(), acc = 0;
      chosen = cand[cand.length - 1];
      for (let i = 0; i < cand.length; i++) {
        acc += probs[i];
        if (r <= acc) {
          chosen = cand[i];
          break;
        }
      }
    }
    const chosenProb = probOf(row, vocab, chosen);
    // top-8 candidates (full-distribution probs) for the see-inside panel
    const showIdx = kIdx.slice(0, 8);
    const showProbs = showIdx.map((i) => probOf(row, vocab, i));
    ids.push(chosen);
    const piece = decode(chosen);
    generated += piece;
    post({
      type: "token",
      id,
      step,
      piece,
      generated,
      chosen: { token: piece, prob: chosenProb },
      topk: showIdx.map((i, n) => ({ token: decode(i), prob: showProbs[n], chosen: i === chosen })),
    });
    if (chosen === eos) break;
  }
  post({
    type: "done",
    id,
    text: prompt + generated,
    generated,
    ms: Math.round(performance.now() - t0),
    tokens: ids.length,
    device,
  });
}

async function distribution(id, text, topK) {
  await ensureLoaded();
  const enc = await tokenizer(text);
  const ids = Array.from(enc.input_ids.data, Number);
  const t0 = performance.now();
  const { logits } = await model(makeInputs(ids));
  const { row, vocab } = lastRow(logits);
  const k = Math.min(20, Math.max(1, topK ?? 12));
  const idx = topkIndices(row, vocab, k);
  const probs = idx.map((i) => probOf(row, vocab, i));
  post({
    type: "dist",
    id,
    topk: idx.map((i, n) => ({ token: decode(i), prob: probs[n], logit: row[i] })),
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "generate") await generate(id, e.data.prompt, e.data.opts || {});
    else if (type === "distribution") await distribution(id, e.data.text, e.data.topK);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
