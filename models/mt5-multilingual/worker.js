// mT5-small multilingual span-corruption worker — inference off the main thread so the UI stays smooth.
// Model: Xenova/mt5-small (task: text2text-generation), WASM, q8.
//
// mT5 is the MULTILINGUAL T5: the same encoder-decoder architecture as T5, but pretrained on the mC4
// corpus covering 101 languages. Crucially it was pretrained ONLY on the span-corruption (denoising)
// objective and — unlike FLAN-T5 (instruction-tuned) or the English T5 checkpoints (which ship
// downstream task heads/prefixes) — it was NOT fine-tuned on any downstream task. So the one thing this
// raw checkpoint genuinely knows how to do is RECONSTRUCT MASKED SPANS:
//   input :  "Thank you <extra_id_0> me to your party <extra_id_1> week."
//   output:  "<extra_id_0> for inviting <extra_id_1> last <extra_id_2>"   (sentinel-delimited fills)
// It does this across all 101 languages from a single 250k-token multilingual SentencePiece vocab — mask
// a Spanish, German, Chinese or English sentence and it fills the blank in that same language. That is
// the honest capability we showcase: the raw multilingual foundation, warts and all (a small q8 model
// gives rough fills), never a canned answer. We also reconstruct the sentence by slotting the fills back
// into the blanks, and timestamp each decoded token so the "see inside" surface shows the real cadence.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let tokenizer = null;
let TextStreamer = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, TextStreamer: TS, env } = await import(TRANSFORMERS_URL);
  TextStreamer = TS;
  env.allowLocalModels = false;
  pipe = await pipeline("text2text-generation", "Xenova/mt5-small", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

/** Encode a string → { count, tokens:string[] }. Token strings expose T5's SentencePiece ▁ word marks. */
function tokenize(text) {
  const enc = tokenizer(text);
  const ids = enc.input_ids;
  const arr = ids?.tolist ? ids.tolist()[0] : (ids?.data ? Array.from(ids.data, Number) : []);
  let tokens = [];
  try {
    tokens = arr.map((id) => tokenizer.decode([id], { skip_special_tokens: false }));
  } catch {
    tokens = arr.map((id) => String(id));
  }
  return { count: arr.length, tokens };
}

/**
 * Reconstruct the original sentence by slotting each sentinel fill back into its blank.
 * mT5's output is sentinel-delimited: "<extra_id_0> fill0 <extra_id_1> fill1 …". We split on the
 * sentinel tokens to recover [fill0, fill1, …] and substitute them for the matching <extra_id_n> in the
 * input. This is exactly the span-corruption format — showing it makes the objective legible.
 */
function reconstruct(input, output) {
  const parts = output.split(/<extra_id_\d+>/).map((s) => s.trim());
  // parts[0] is whatever precedes the first sentinel (usually empty); fills start at parts[1].
  const fills = parts.slice(1);
  let recon = input;
  let usedAny = false;
  recon = recon.replace(/<extra_id_(\d+)>/g, (m, n) => {
    const fill = fills[Number(n)];
    if (fill != null && fill !== "") {
      usedAny = true;
      return `⟦${fill}⟧`;
    }
    return m;
  });
  return usedAny ? recon : null;
}

async function generate(id, input, opts) {
  await ensureLoaded();
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const sample = !!opts.doSample && beams === 1;
  const gen = {
    max_new_tokens: Math.max(1, opts.maxNewTokens | 0 || 48),
    num_beams: beams,
    do_sample: sample,
    ...(sample ? { temperature: opts.temperature ?? 0.9, top_k: opts.topK ?? 50 } : {}),
    no_repeat_ngram_size: 3,
  };

  const times = [];
  let partial = "";
  // Per-token streaming/timing is only clean for a single sequence (greedy or sampling, beams === 1);
  // beam search reorders tokens at the end, so we skip the streamer and report aggregate timing honestly.
  if (beams === 1) {
    gen.streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false, // keep <extra_id_n> sentinels visible — they ARE the output format
      callback_function: (t) => {
        partial += t;
        post({ type: "stream", id, text: partial });
      },
      token_callback_function: () => times.push(performance.now()),
    });
  }

  const inTok = tokenize(input);
  const t0 = performance.now();
  const out = await pipe(input, gen);
  const ms = Math.round(performance.now() - t0);
  const output = (out[0]?.generated_text ?? "").trim();
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTokens = times.length || tokenize(output).count;
  const sentinelCount = (input.match(/<extra_id_\d+>/g) || []).length;

  post({
    type: "result",
    id,
    input,
    output,
    reconstruction: reconstruct(input, output),
    sentinelCount,
    inTokens: inTok.count,
    inTokenStrings: inTok.tokens,
    outTokens,
    intervals,
    beams,
    sampled: sample,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await generate(e.data.id, e.data.input, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
