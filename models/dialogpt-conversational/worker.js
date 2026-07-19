// DialoGPT (small) conversational worker — multi-turn dialogue generation off the main thread.
// Model: onnx-community/DialoGPT-small-ONNX (task: text-generation / conversational), WASM, q8.
//
// DialoGPT is a GPT-2-based conversational model from Microsoft, trained on 147M Reddit dialogue
// exchanges. Its ONE defining mechanism: a whole multi-turn conversation is fed to the model as a
// SINGLE sequence where each turn is separated by the end-of-text token (`<|endoftext|>`, id 50256).
// The model then continues that sequence — its continuation, up to the next EOS, is the reply. We build
// that EOS-joined context here and report it so the "see inside" surface can show the real mechanism.
//
// Why a manual decode loop (not pipeline().generate()): this merged single-file ONNX export has no
// separate with-past decoder, and transformers.js 3.7.5's generate()/text-generation pipeline throws in
// `_prepare_inputs_for_generation` for it (verified). Running our own forward-pass greedy/sampling loop
// with AutoModelForCausalLM — exactly like the GPT-2 demo — works and, as a bonus, lets us stream every
// token and expose the per-token distribution. Every number is the model's real output; nothing faked.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/DialoGPT-small-ONNX";
let tokenizer = null;
let model = null;
let Tensor = null;
const device = "wasm";
let EOS = 50256;
let EOS_STR = "<|endoftext|>";

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
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  EOS = model.config?.eos_token_id ?? 50256;
  EOS_STR = tokenizer.decode([EOS]);
  post({ type: "ready", device, eos: EOS, eosStr: EOS_STR });
}

function makeInputs(ids) {
  const big = BigInt64Array.from(ids, (v) => BigInt(v));
  const ones = BigInt64Array.from(ids, () => 1n);
  return {
    input_ids: new Tensor("int64", big, [1, ids.length]),
    attention_mask: new Tensor("int64", ones, [1, ids.length]),
  };
}

function lastRow(logits) {
  const [, seq, vocab] = logits.dims;
  const start = (seq - 1) * vocab;
  return { row: logits.data.subarray(start, start + vocab), vocab };
}

// Top-k indices of a row (descending) without fully sorting the vocab.
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

function probOf(row, vocab, idx) {
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  for (let i = 0; i < vocab; i++) sum += Math.exp(row[i] - max);
  return Math.exp(row[idx] - max) / sum;
}

// Build the EOS-joined dialogue context: every turn (persona seed + user/bot history + new message)
// concatenated with the end-of-text token between and after each turn — the DialoGPT input format.
function buildContext(turns) {
  const parts = turns.filter((t) => t != null && String(t).length > 0);
  const text = parts.map((t) => String(t)).join(EOS_STR) + EOS_STR;
  const ids = Array.from(tokenizer(text).input_ids.data, Number);
  return { text, ids };
}

async function chat(id, turns, opts) {
  await ensureLoaded();
  const maxNew = Math.min(80, Math.max(1, opts.maxNew ?? 48));
  const greedy = !!opts.greedy;
  const temperature = opts.temperature ?? 0.9;
  const topK = Math.min(80, Math.max(1, opts.topK ?? 50));
  const { text: contextText, ids: contextIds } = buildContext(turns);
  const promptLen = contextIds.length;
  const ids = contextIds.slice();
  const t0 = performance.now();
  let reply = "";
  let steps = 0;
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
    ids.push(chosen);
    steps++;
    if (chosen === EOS) break; // the model closed its turn
    const piece = tokenizer.decode([chosen]);
    const chosenProb = probOf(row, vocab, chosen);
    reply += piece;
    const showIdx = kIdx.slice(0, 6);
    const showProbs = showIdx.map((i) => probOf(row, vocab, i));
    post({
      type: "token",
      id,
      step,
      piece,
      reply,
      chosen: { prob: chosenProb },
      topk: showIdx.map((i, n) => ({
        token: tokenizer.decode([i]),
        prob: showProbs[n],
        chosen: i === chosen,
      })),
    });
  }
  const ms = Math.round(performance.now() - t0);
  const replyTokens = steps;
  post({
    type: "done",
    id,
    reply: reply.trim(),
    contextText,
    promptTokens: promptLen,
    replyTokens,
    endedOnEos: ids[ids.length - 1] === EOS,
    ms,
    device,
    eosStr: EOS_STR,
    eos: EOS,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "chat") await chat(id, e.data.turns, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
