// The tool layer for the voice-agent demo: schemas the Qwen chat template understands, local
// executors that really change page state, a no-eval arithmetic evaluator, and a parser for the
// <tool_call> blocks Qwen2.5 emits.
//
// Nothing here touches the network or the DOM. Executors receive a `ctx` bag supplied by the page,
// so this module stays pure and unit-testable in Node (scripts/validate-qwen-voice-tools.mjs).

/**
 * Tool schemas in the OpenAI-function shape the Qwen2.5 chat template serialises into its
 * <tools></tools> block. Keep the set SMALL and the descriptions blunt — a 0.5B model picks better
 * from six sharp tools than from twenty vague ones.
 */
export const TOOL_SCHEMAS = [
  {
    type: "function",
    function: {
      name: "get_time",
      description:
        "Get the current date and time, optionally in another city's time zone. Use for any question about what time or date it is.",
      parameters: {
        type: "object",
        properties: {
          timezone: {
            type: "string",
            description:
              'IANA time zone such as "Europe/London", "America/New_York" or "Asia/Tokyo". Omit for the local time zone.',
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "start_timer",
      description:
        "Start a countdown timer that rings when it finishes. Use whenever the user asks to be timed or reminded in N seconds or minutes.",
      parameters: {
        type: "object",
        properties: {
          seconds: { type: "number", description: "Duration of the timer in seconds." },
          label: { type: "string", description: "Short name for the timer, e.g. \"pasta\"." },
        },
        required: ["seconds"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description:
        "Evaluate an arithmetic expression. Use for any sum, percentage, or number question.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: 'Arithmetic only, e.g. "18 * 7", "(120 + 45) / 3", "sqrt(144)".',
          },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "convert_units",
      description:
        "Convert a value between units of length, mass, temperature, volume, or speed.",
      parameters: {
        type: "object",
        properties: {
          value: { type: "number", description: "The number to convert." },
          from: { type: "string", description: 'Source unit, e.g. "km", "kg", "celsius", "mph".' },
          to: { type: "string", description: 'Target unit, e.g. "miles", "lb", "fahrenheit", "kph".' },
        },
        required: ["value", "from", "to"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "add_note",
      description: "Save a short note to the on-page notepad. Use when the user asks to remember, note, or write something down.",
      parameters: {
        type: "object",
        properties: { text: { type: "string", description: "The note text." } },
        required: ["text"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_notes",
      description: "Read back every note saved so far.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
];

export const TOOL_NAMES = TOOL_SCHEMAS.map((t) => t.function.name);

// ---------------------------------------------------------------------------
// Safe arithmetic — a recursive-descent parser, NOT eval()/Function(). A model-authored string is
// untrusted input; it never becomes code. Anything outside the grammar throws.
// ---------------------------------------------------------------------------

const FUNCS = {
  sqrt: Math.sqrt,
  abs: Math.abs,
  round: Math.round,
  floor: Math.floor,
  ceil: Math.ceil,
  min: Math.min,
  max: Math.max,
  pow: Math.pow,
};
const CONSTS = { pi: Math.PI, e: Math.E };

function tokenize(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    if (/[0-9.]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9._]/.test(src[j])) j++;
      const raw = src.slice(i, j).replace(/_/g, "");
      const num = Number(raw);
      if (!Number.isFinite(num)) throw new Error(`not a number: "${raw}"`);
      tokens.push({ t: "num", v: num });
      i = j;
      continue;
    }
    if (/[a-zA-Z]/.test(c)) {
      let j = i;
      while (j < src.length && /[a-zA-Z0-9]/.test(src[j])) j++;
      tokens.push({ t: "name", v: src.slice(i, j).toLowerCase() });
      i = j;
      continue;
    }
    if ("+-*/%^(),".includes(c)) { tokens.push({ t: c }); i++; continue; }
    // '×' and '÷' show up in spoken-then-transcribed maths often enough to be worth accepting.
    if (c === "×") { tokens.push({ t: "*" }); i++; continue; }
    if (c === "÷") { tokens.push({ t: "/" }); i++; continue; }
    throw new Error(`unexpected character "${c}"`);
  }
  return tokens;
}

/** Evaluate an arithmetic expression safely. Throws on anything that isn't plain maths. */
export function evaluateExpression(src) {
  if (typeof src !== "string" || !src.trim()) throw new Error("empty expression");
  if (src.length > 200) throw new Error("expression too long");
  const tokens = tokenize(src);
  let pos = 0;
  const peek = () => tokens[pos];
  const eat = (t) => {
    if (!tokens[pos] || tokens[pos].t !== t) throw new Error(`expected "${t}"`);
    return tokens[pos++];
  };

  // expr := term (('+'|'-') term)*
  function expr() {
    let left = term();
    while (peek() && (peek().t === "+" || peek().t === "-")) {
      const op = tokens[pos++].t;
      const right = term();
      left = op === "+" ? left + right : left - right;
    }
    return left;
  }
  // term := power (('*'|'/'|'%') power)*
  function term() {
    let left = power();
    while (peek() && (peek().t === "*" || peek().t === "/" || peek().t === "%")) {
      const op = tokens[pos++].t;
      const right = power();
      if ((op === "/" || op === "%") && right === 0) throw new Error("division by zero");
      left = op === "*" ? left * right : op === "/" ? left / right : left % right;
    }
    return left;
  }
  // power := unary ('^' power)?   (right-associative)
  function power() {
    const base = unary();
    if (peek() && peek().t === "^") { pos++; return Math.pow(base, power()); }
    return base;
  }
  // unary := ('-'|'+') unary | primary
  function unary() {
    if (peek() && peek().t === "-") { pos++; return -unary(); }
    if (peek() && peek().t === "+") { pos++; return unary(); }
    return primary();
  }
  // primary := num | const | func '(' args ')' | '(' expr ')'
  function primary() {
    const tok = peek();
    if (!tok) throw new Error("unexpected end of expression");
    if (tok.t === "num") { pos++; return tok.v; }
    if (tok.t === "(") { pos++; const v = expr(); eat(")"); return v; }
    if (tok.t === "name") {
      pos++;
      if (Object.hasOwn(CONSTS, tok.v)) return CONSTS[tok.v];
      const fn = Object.hasOwn(FUNCS, tok.v) ? FUNCS[tok.v] : null;
      if (!fn) throw new Error(`unknown name "${tok.v}"`);
      eat("(");
      const args = [expr()];
      while (peek() && peek().t === ",") { pos++; args.push(expr()); }
      eat(")");
      return fn(...args);
    }
    throw new Error(`unexpected token "${tok.t}"`);
  }

  const value = expr();
  if (pos !== tokens.length) throw new Error("trailing input after the expression");
  if (!Number.isFinite(value)) throw new Error("result is not a finite number");
  return value;
}

// ---------------------------------------------------------------------------
// Unit conversion — an explicit table. Unknown units fail loudly rather than guessing.
// ---------------------------------------------------------------------------

// Everything linear converts through a base unit: value * factor = base.
const LINEAR = {
  length: {
    base: "m",
    units: {
      mm: 0.001, cm: 0.01, m: 1, km: 1000,
      in: 0.0254, inch: 0.0254, inches: 0.0254,
      ft: 0.3048, foot: 0.3048, feet: 0.3048,
      yd: 0.9144, yard: 0.9144, yards: 0.9144,
      mi: 1609.344, mile: 1609.344, miles: 1609.344,
    },
  },
  mass: {
    base: "kg",
    units: {
      mg: 1e-6, g: 0.001, gram: 0.001, grams: 0.001, kg: 1, kilogram: 1, kilograms: 1,
      t: 1000, tonne: 1000, tonnes: 1000,
      oz: 0.028349523125, ounce: 0.028349523125, ounces: 0.028349523125,
      lb: 0.45359237, lbs: 0.45359237, pound: 0.45359237, pounds: 0.45359237,
      st: 6.35029318, stone: 6.35029318,
    },
  },
  volume: {
    base: "l",
    units: {
      ml: 0.001, l: 1, litre: 1, litres: 1, liter: 1, liters: 1,
      cup: 0.2365882365, cups: 0.2365882365,
      pt: 0.473176473, pint: 0.473176473, pints: 0.473176473,
      gal: 3.785411784, gallon: 3.785411784, gallons: 3.785411784,
    },
  },
  speed: {
    base: "mps",
    units: {
      mps: 1, kph: 0.2777777778, kmh: 0.2777777778, "km/h": 0.2777777778,
      mph: 0.44704, knot: 0.514444, knots: 0.514444,
    },
  },
};

const TEMP = new Set(["c", "celsius", "centigrade", "f", "fahrenheit", "k", "kelvin"]);

function normUnit(u) {
  return String(u ?? "").trim().toLowerCase().replace(/^degrees?\s+/, "").replace(/\.$/, "");
}

function toCelsius(v, u) {
  if (u === "c" || u === "celsius" || u === "centigrade") return v;
  if (u === "f" || u === "fahrenheit") return (v - 32) * (5 / 9);
  return v - 273.15; // kelvin
}
function fromCelsius(v, u) {
  if (u === "c" || u === "celsius" || u === "centigrade") return v;
  if (u === "f" || u === "fahrenheit") return v * (9 / 5) + 32;
  return v + 273.15;
}

/** Convert between units. Returns { value, from, to, dimension }. Throws on unknown/mismatched units. */
export function convert(value, from, to) {
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(num)) throw new Error(`"${value}" is not a number`);
  const f = normUnit(from);
  const t = normUnit(to);
  if (!f || !t) throw new Error("both a source and a target unit are required");
  if (TEMP.has(f) || TEMP.has(t)) {
    if (!TEMP.has(f) || !TEMP.has(t)) throw new Error(`can't convert ${f} to ${t}`);
    return { value: fromCelsius(toCelsius(num, f), t), from: f, to: t, dimension: "temperature" };
  }
  for (const [dimension, spec] of Object.entries(LINEAR)) {
    const a = spec.units[f];
    const b = spec.units[t];
    // Both units must live in the SAME dimension — a half match (km → kg) keeps looking, then fails.
    if (a != null && b != null) return { value: (num * a) / b, from: f, to: t, dimension };
  }
  throw new Error(`don't know how to convert "${from}" to "${to}"`);
}

// ---------------------------------------------------------------------------
// Parsing the model's output
// ---------------------------------------------------------------------------

/** Pull the first balanced {...} object out of `text` starting at `from`. Returns [json, endIndex]. */
function firstObject(text, from = 0) {
  const start = text.indexOf("{", from);
  if (start === -1) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return [text.slice(start, i + 1), i + 1];
  }
  return null;
}

/**
 * Extract tool calls from raw model output. Qwen2.5 is *supposed* to emit
 * `<tool_call>{"name":…,"arguments":{…}}</tool_call>`, and a 0.5B model often nearly does — so we
 * also accept a fenced ```json block or a bare object, as long as it has a name we published.
 * Returns [] when the model answered directly; the page reports that honestly rather than retrying.
 */
export function parseToolCalls(text, names = TOOL_NAMES) {
  const out = [];
  const seen = new Set();
  const consider = (raw) => {
    let obj;
    try { obj = JSON.parse(raw); } catch { return; }
    if (!obj || typeof obj !== "object") return;
    const name = obj.name ?? obj.function?.name ?? obj.tool ?? obj.tool_name;
    if (typeof name !== "string" || !names.includes(name)) return;
    let args = obj.arguments ?? obj.function?.arguments ?? obj.parameters ?? obj.args ?? {};
    if (typeof args === "string") {
      try { args = JSON.parse(args); } catch { args = {}; }
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) args = {};
    const key = name + JSON.stringify(args);
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ name, arguments: args });
  };

  const src = String(text ?? "");
  // 1. The canonical <tool_call> blocks (closing tag optional — small models truncate it).
  const tagged = /<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g;
  let m;
  while ((m = tagged.exec(src)) !== null) {
    const found = firstObject(m[1]);
    if (found) consider(found[0]);
  }
  if (out.length) return out;

  // 2. A fenced code block.
  const fenced = /```(?:json|tool_call)?\s*([\s\S]*?)```/g;
  while ((m = fenced.exec(src)) !== null) {
    const found = firstObject(m[1]);
    if (found) consider(found[0]);
  }
  if (out.length) return out;

  // 3. A bare object anywhere in the text.
  let idx = 0;
  for (let guard = 0; guard < 8; guard++) {
    const found = firstObject(src, idx);
    if (!found) break;
    consider(found[0]);
    idx = found[1];
  }
  return out;
}

/** Strip tool-call markup so what's left is the model's prose, if any. */
export function stripToolCalls(text) {
  return String(text ?? "")
    .replace(/<tool_call>[\s\S]*?(?:<\/tool_call>|$)/g, "")
    .replace(/```(?:json|tool_call)?[\s\S]*?```/g, "")
    .trim();
}

// ---------------------------------------------------------------------------
// Executors — each returns a JSON-serialisable result that goes back to the model as a
// `role: "tool"` message, plus a human-readable `display` line for the page.
// ---------------------------------------------------------------------------

function fmtNumber(n) {
  if (!Number.isFinite(n)) return String(n);
  const rounded = Math.round(n * 1e6) / 1e6;
  return String(rounded);
}

export const EXECUTORS = {
  get_time({ timezone } = {}, ctx = {}) {
    const now = ctx.now ? new Date(ctx.now) : new Date();
    const local = Intl.DateTimeFormat().resolvedOptions().timeZone;
    let zone = typeof timezone === "string" && timezone.trim() ? timezone.trim() : local;
    let fmt;
    try {
      fmt = new Intl.DateTimeFormat("en-GB", {
        timeZone: zone,
        dateStyle: "full",
        timeStyle: "short",
      });
    } catch {
      // An unknown zone is a real failure — say so instead of silently answering for somewhere else.
      throw new Error(`"${zone}" is not a time zone I know`);
    }
    const formatted = fmt.format(now);
    return { result: { timezone: zone, datetime: formatted, iso: now.toISOString() }, display: formatted };
  },

  start_timer({ seconds, label } = {}, ctx = {}) {
    const secs = Math.round(Number(seconds));
    if (!Number.isFinite(secs) || secs <= 0) throw new Error("a timer needs a positive number of seconds");
    if (secs > 3600) throw new Error("timers are capped at one hour in this demo");
    const name = typeof label === "string" && label.trim() ? label.trim().slice(0, 40) : "timer";
    const id = ctx.startTimer ? ctx.startTimer(secs, name) : null;
    return {
      result: { started: true, seconds: secs, label: name, id },
      display: `${name} — ${secs}s, counting down on the page`,
    };
  },

  calculate({ expression } = {}) {
    const value = evaluateExpression(expression);
    return { result: { expression: String(expression), value }, display: `${expression} = ${fmtNumber(value)}` };
  },

  convert_units({ value, from, to } = {}) {
    const c = convert(value, from, to);
    return {
      result: { value: c.value, from: c.from, to: c.to, dimension: c.dimension },
      display: `${fmtNumber(Number(value))} ${c.from} = ${fmtNumber(c.value)} ${c.to}`,
    };
  },

  add_note({ text } = {}, ctx = {}) {
    const note = String(text ?? "").trim();
    if (!note) throw new Error("nothing to note down");
    const notes = ctx.notes ?? [];
    notes.push(note.slice(0, 200));
    ctx.onNotesChanged?.(notes);
    return { result: { saved: true, note, total: notes.length }, display: `saved "${note}" (${notes.length} total)` };
  },

  list_notes(_args, ctx = {}) {
    const notes = ctx.notes ?? [];
    return {
      result: { notes, count: notes.length },
      display: notes.length ? notes.map((n, i) => `${i + 1}. ${n}`).join(" · ") : "the notepad is empty",
    };
  },
};

/**
 * Run one parsed call. Never throws: a tool failure is a real result the model should see and
 * explain, so it comes back as { ok:false, error } rather than blowing up the turn.
 */
export function runTool(call, ctx = {}) {
  const fn = Object.hasOwn(EXECUTORS, call?.name) ? EXECUTORS[call.name] : null;
  const t0 = (globalThis.performance?.now?.() ?? 0);
  if (!fn) return { ok: false, name: call?.name ?? "(none)", error: `no such tool: ${call?.name}`, ms: 0 };
  try {
    const { result, display } = fn(call.arguments ?? {}, ctx);
    return { ok: true, name: call.name, arguments: call.arguments ?? {}, result, display, ms: Math.round((globalThis.performance?.now?.() ?? 0) - t0) };
  } catch (err) {
    return { ok: false, name: call.name, arguments: call.arguments ?? {}, error: String(err?.message ?? err), ms: Math.round((globalThis.performance?.now?.() ?? 0) - t0) };
  }
}

/** The content string handed back to the model as the `tool` message. */
export function toolMessageContent(outcome) {
  return JSON.stringify(outcome.ok ? outcome.result : { error: outcome.error });
}

/** The system prompt. Deliberately terse: a 0.5B model follows short rules far better than long ones. */
export const SYSTEM_PROMPT =
  "You are a voice assistant running entirely in the user's browser. The user speaks; their words " +
  "reach you as a transcript, so expect the odd mis-heard word. When a tool can answer, call exactly " +
  "one tool. When you get a tool result, reply with one short spoken-style sentence stating the " +
  "answer. Never invent a tool result.";
