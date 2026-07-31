// Pure Turkish NER helpers shared by the browser worker and Node regression tests.
// Keep this module free of DOM/Worker globals so BIO/alignment and text transforms are testable.

const SPECIAL = /^\[(?:CLS|SEP|PAD|MASK)\]$/;

export function alignPieces(pieces, text) {
  let cursor = 0;
  return pieces.map((piece) => {
    if (SPECIAL.test(piece)) return { start: null, end: null, special: true };
    while (/\s/u.test(text[cursor] || "")) cursor++;
    let surface = piece.replace(/^##/, "");
    if (piece === "[UNK]") {
      const match = text.slice(cursor).match(/^[^\s.,!?;:'"()]+/u);
      surface = match?.[0] || text[cursor] || "";
    }
    const start = text.indexOf(surface, cursor);
    if (start < 0) {
      throw new Error(`Could not align tokenizer piece “${piece}” after character ${cursor}`);
    }
    const end = start + surface.length;
    cursor = end;
    return { start, end, special: false };
  });
}

export function mergeWords(tokens, text) {
  const words = [];
  for (const token of tokens) {
    if (token.special) continue;
    const continuation = token.word.startsWith("##") && words.length > 0;
    if (continuation) {
      const word = words.at(-1);
      word.end = token.end;
      word.surface = text.slice(word.start, word.end);
      word.pieces.push(token);
      // Match token-classification's "first" strategy: a noisy continuation must not split a word.
      word.score = word.pieces.reduce((sum, item) => sum + item.score, 0) / word.pieces.length;
    } else {
      const type = token.entity === "O" ? null : token.entity.slice(2);
      words.push({
        surface: text.slice(token.start, token.end),
        entity: token.entity,
        type,
        score: token.score,
        start: token.start,
        end: token.end,
        pieces: [token],
      });
    }
  }
  return words.map(({ pieces: _pieces, ...word }) => word);
}

export function entitySpans(words, text) {
  const spans = [];
  let open = null;
  for (const word of words) {
    if (!word.type) {
      open = null;
      continue;
    }
    const bio = word.entity.slice(0, 1);
    // Repair an orphan I-* as a new span and split any type transition.
    if (bio === "B" || !open || open.type !== word.type) {
      open = { type: word.type, start: word.start, end: word.end, scores: [word.score] };
      spans.push(open);
    } else {
      open.end = word.end;
      open.scores.push(word.score);
    }
  }
  return spans.map((span) => ({
    type: span.type,
    text: text.slice(span.start, span.end),
    start: span.start,
    end: span.end,
    score: span.scores.reduce((a, b) => a + b, 0) / span.scores.length,
  }));
}

export async function sha256Hex(bytes) {
  const view = bytes instanceof ArrayBuffer
    ? bytes
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const digest = await crypto.subtle.digest("SHA-256", view);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function assertArtifactIntegrity(bytes, { expectedBytes, expectedSha256, label }) {
  const byteLength = bytes.byteLength;
  if (byteLength !== expectedBytes) {
    throw new Error(
      `${label} integrity check failed: expected ${expectedBytes} bytes, received ${byteLength}`,
    );
  }
  const actualSha256 = await sha256Hex(bytes);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `${label} integrity check failed: expected SHA-256 ${expectedSha256}, received ${actualSha256}`,
    );
  }
  return { byteLength, sha256: actualSha256 };
}

export function replaceEntitySpans(text, entities, replacements) {
  const edits = entities
    .map((entity, index) => ({ entity, replacement: replacements.get(index) }))
    .filter(({ entity, replacement }) =>
      replacement != null && entity.start != null && entity.end != null
    )
    .sort((a, b) => b.entity.start - a.entity.start);
  let output = text;
  for (const { entity, replacement } of edits) {
    output = output.slice(0, entity.start) + replacement + output.slice(entity.end);
  }
  return output;
}

export function turkishEntityKey(type, text) {
  return `${type}|${text.toLocaleLowerCase("tr")}`;
}
