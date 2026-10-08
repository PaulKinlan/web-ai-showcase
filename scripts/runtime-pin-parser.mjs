// runtime-pin-parser.mjs — the SINGLE entry point for all JavaScript/HTML syntax
// knowledge used by the runtime-pin currency gate.
//
// Design: design/runtime-pin-ast-parser.md (bead web-ai-showcase-j6i).
// Hard rules:
//  - ONE parser, ONE entry point: no other module may tokenize/parse JS or HTML
//    for the currency gate. Syntax knowledge lives here only.
//  - FAIL CLOSED: a parse failure on a file with marker/runtime-sink evidence is
//    an error (RuntimePinParseError), never "inert".
//  - RAW tokens only: candidates come from source.slice(node.start, node.end)
//    (UTF-16 offsets). Never decode escapes, never truncate a numeric prefix.
//  - No network, no writes: this module is pure parsing/classification.

import * as acorn from "acorn";
import * as acornWalk from "acorn-walk";
import * as parse5 from "parse5";

export const ACORN_VERSION = "8.19.0";
export const ACORN_WALK_VERSION = "8.3.5";
export const PARSE5_VERSION = "8.0.1";

// Raw contiguous package markers (byte-level; discovery only, never approval).
export const RAW_MARKERS = ["onnxruntime-web@", "@huggingface/transformers@"];
// Regex-literal escaped variant seen in tooling sources ( census/unsupported
// classification only — never decoded into a URL candidate).
export const ESCAPED_MARKER_VARIANTS = ["@huggingface\\/transformers@"];
// Cooked-escape marker forms (e.g. `onnxruntime-web\x40` / `\x40huggingface`)
// are discovered by the cooked-literal census pass, not by raw byte search.

export class RuntimePinParseError extends Error {
  constructor(filePath, kind, detail, pos = undefined) {
    super(`${filePath}: ${kind}: ${detail}`);
    this.name = "RuntimePinParseError";
    this.filePath = filePath;
    this.kind = kind; // "js-parse" | "html-parse" | "unsupported-context"
    this.pos = pos;
  }
}

// ---------------------------------------------------------------------------
// JavaScript
// ---------------------------------------------------------------------------

/**
 * Parse a JS/MJS source string with acorn. Tries sourceType "module", then
 * "script", then module with allowHashBang. Returns
 * { ast, comments, tokens, sourceType } or throws RuntimePinParseError.
 */
export function parseJavaScript(source, filePath) {
  const attempts = [
    { sourceType: "module", allowHashBang: true },
    { sourceType: "script", allowHashBang: true },
  ];
  const errors = [];
  for (const attempt of attempts) {
    const comments = [];
    const tokens = [];
    try {
      const ast = acorn.parse(source, {
        ecmaVersion: "latest",
        sourceType: attempt.sourceType,
        allowHashBang: attempt.allowHashBang,
        allowReturnOutsideFunction: true,
        allowAwaitOutsideFunction: true,
        allowImportExportEverywhere: false,
        locations: true,
        ranges: true,
        onComment: comments,
        onToken: tokens,
      });
      return { ast, comments, tokens, sourceType: attempt.sourceType };
    } catch (err) {
      errors.push(`${attempt.sourceType}: ${err.message}`);
    }
  }
  throw new RuntimePinParseError(filePath, "js-parse", errors.join(" | "));
}

/**
 * Build the ordered span table for a parsed JS source: every string Literal,
 * TemplateLiteral (with quasi ranges), comment and regex token, with exact
 * UTF-16 start/end offsets. Spans carry a `kind` used for marker-context
 * classification. Raw text is always recoverable via
 * source.slice(span.start, span.end).
 */
// acorn-walk's default base skips NON-computed property/member keys (they are
// usually Identifiers). Object keys in this corpus include full string URLs
// (e.g. the generated sw.js runtime-integrity manifest), so visit Literal keys
// explicitly or their markers would misclassify as bare "code".
const WALK_BASE = { ...acornWalk.base };
const withLiteralKey = (kind) => (node, state, c) => {
  if (node.computed || node.key.type === "Literal") c(node.key, state, "Expression");
  if (kind === "Property") c(node.value, state, "Expression");
};
WALK_BASE.Property = withLiteralKey("Property");
WALK_BASE.PropertyDefinition = withLiteralKey("PropertyDefinition");
WALK_BASE.MethodDefinition = withLiteralKey("MethodDefinition");

export function jsSyntaxSpans(source, parsed) {
  const { ast, comments, tokens } = parsed;
  const spans = [];
  for (const c of comments) {
    spans.push({
      kind: c.type === "Line" ? "comment-line" : "comment-block",
      start: c.start,
      end: c.end,
    });
  }
  for (const t of tokens) {
    if (t.type && t.type.label === "regexp") {
      spans.push({ kind: "regex", start: t.start, end: t.end });
    }
  }
  acornWalk.fullAncestor(ast, (node) => {
    if (node.type === "Literal" && typeof node.value === "string") {
      spans.push({ kind: "string", start: node.start, end: node.end });
    } else if (node.type === "TemplateLiteral") {
      spans.push({
        kind: "template",
        start: node.start,
        end: node.end,
        quasis: node.quasis.map((q) => ({ start: q.start, end: q.end })),
        expressions: node.expressions.map((e) => ({
          start: e.start,
          end: e.end,
          type: e.type,
        })),
      });
    }
  }, WALK_BASE);
  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  return spans;
}

/**
 * Classify a byte/UTF-16 offset inside JS source into an executable-context
 * kind using the span table. Nested template literals are handled naturally:
 * the innermost (smallest) containing span wins, and a marker inside a
 * template EXPRESSION is classified from the expression's own spans
 * (e.g. an inner template) or as template-expression if bare code.
 */
export function classifyJsOffset(spans, offset) {
  let best = null;
  for (const span of spans) {
    if (span.start <= offset && offset < span.end) {
      if (!best || span.end - span.start <= best.end - best.start) best = span;
    }
  }
  if (!best) return { kind: "code", span: null };
  if (best.kind === "template") {
    const inQuasi = best.quasis.some((q) => q.start <= offset && offset < q.end);
    return { kind: inQuasi ? "template-quasi" : "template-expression", span: best };
  }
  return { kind: best.kind, span: best };
}

/**
 * Bounded lexical-scope constant binding (vs2 scope). Resolves EXACTLY the
 * pattern used by the three ORT workers:
 *   const NAME = "<primitive string>";
 *   ... import(`…${NAME}…`) / wasmPaths = `…${NAME}…`
 * Returns a Map name → { raw, value, declaratorStart, declaratorEnd } for
 * top-level `const` bindings whose value is a plain string Literal and whose
 * Identifier is never reassigned, never shadowed in a nested scope, and never
 * referenced through anything but a bare Identifier read. Anything else is
 * NOT returned (the caller must fail closed on the unresolved sink).
 */
export function resolveTopLevelStringConstants(ast) {
  const bindings = new Map();
  for (const stmt of ast.body) {
    if (stmt.type !== "VariableDeclaration" || stmt.kind !== "const") continue;
    for (const decl of stmt.declarations) {
      if (
        decl.id &&
        decl.id.type === "Identifier" &&
        decl.init &&
        decl.init.type === "Literal" &&
        typeof decl.init.value === "string"
      ) {
        bindings.set(decl.id.name, {
          raw: decl.init.raw,
          value: decl.init.value,
          declaratorStart: decl.start,
          declaratorEnd: decl.end,
          nameStart: decl.id.start,
          nameEnd: decl.id.end,
        });
      }
    }
  }
  // Invalidate on any reassignment, update, or shadowing declaration.
  acornWalk.fullAncestor(ast, (node, ancestors) => {
    const invalidate = (name) => bindings.delete(name);
    if (node.type === "AssignmentExpression" && node.left.type === "Identifier") {
      invalidate(node.left.name);
    } else if (node.type === "UpdateExpression" && node.argument.type === "Identifier") {
      invalidate(node.argument.name);
    } else if (
      (node.type === "FunctionDeclaration" ||
        node.type === "FunctionExpression" ||
        node.type === "ArrowFunctionExpression") &&
      ancestors.length > 1
    ) {
      // Shadowing: a param or inner declaration reusing the name.
      for (const param of node.params || []) {
        if (param.type === "Identifier") invalidate(param.name);
      }
      acornWalk.simple(node.body ?? node, {
        VariableDeclaration(inner) {
          for (const d of inner.declarations) {
            if (d.id.type === "Identifier") invalidate(d.id.name);
          }
        },
      });
    }
  });
  return bindings;
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

const EXECUTABLE_SCRIPT_TYPES = new Set([
  "", // classic script
  "module",
  "text/javascript",
  "application/javascript",
]);

/**
 * Parse an HTML document with parse5 (sourceCodeLocationInfo on). Returns
 * { document } or throws RuntimePinParseError on parse5 exceptions. parse5 is
 * error-recovering by spec, so malformed markup does not throw; callers must
 * rely on location data, not exception, for classification.
 */
export function parseHtmlDocument(source, filePath) {
  try {
    const document = parse5.parse(source, { sourceCodeLocationInfo: true });
    return { document };
  } catch (err) {
    throw new RuntimePinParseError(filePath, "html-parse", err.message);
  }
}

/**
 * Extract every script-bearing context from a parsed HTML document.
 * Returns an array of entries:
 *  { kind: "script-src",      src, startOffset, endOffset }   quoted src only
 *  { kind: "script-inline",   text, startOffset, endOffset, scriptType }
 *  { kind: "script-nonexec",  scriptType, startOffset, endOffset } importmap/json/etc
 *  { kind: "unsupported-event-handler", attrName, startOffset, endOffset }
 *  { kind: "unsupported-javascript-url", attrName, startOffset, endOffset }
 *  { kind: "unsupported-unquoted-src", startOffset, endOffset }
 * Unsupported kinds are returned, NOT silently ignored — the gate fails closed.
 */
export function htmlScriptContexts(document) {
  const contexts = [];
  const walk = (node, ancestors) => {
    if (node.tagName === "script" && node.sourceCodeLocation) {
      const loc = node.sourceCodeLocation;
      const attrs = Object.fromEntries(
        (node.attrs || []).map((a) => [a.name.toLowerCase(), a.value]),
      );
      const srcAttr = (node.attrs || []).find((a) => a.name.toLowerCase() === "src");
      if (srcAttr) {
        const attrLoc = loc.attrs && loc.attrs[srcAttr.name.toLowerCase()];
        if (attrLoc && attrLoc.startOffset !== undefined) {
          contexts.push({
            kind: "script-src",
            src: srcAttr.value,
            startOffset: attrLoc.startOffset,
            endOffset: attrLoc.endOffset,
          });
        } else {
          contexts.push({
            kind: "unsupported-unquoted-src",
            startOffset: loc.startOffset,
            endOffset: loc.endOffset,
          });
        }
      } else {
        const scriptType = (attrs.type || "").toLowerCase();
        const textNode = (node.childNodes || []).find((c) => c.nodeName === "#text");
        if (EXECUTABLE_SCRIPT_TYPES.has(scriptType)) {
          contexts.push({
            kind: "script-inline",
            scriptType: scriptType || "classic",
            text: textNode ? textNode.value : "",
            startOffset: textNode && textNode.sourceCodeLocation
              ? textNode.sourceCodeLocation.startOffset
              : loc.startOffset,
            endOffset: textNode && textNode.sourceCodeLocation
              ? textNode.sourceCodeLocation.endOffset
              : loc.endOffset,
          });
        } else {
          contexts.push({
            kind: "script-nonexec",
            scriptType: scriptType || "unknown",
            startOffset: loc.startOffset,
            endOffset: loc.endOffset,
          });
        }
      }
    }
    // Event handlers and javascript: URLs on ANY element: unsupported contexts.
    if (node.attrs) {
      const loc = node.sourceCodeLocation;
      for (const a of node.attrs) {
        const name = a.name.toLowerCase();
        const attrLoc = loc && loc.attrs && loc.attrs[a.name.toLowerCase()];
        const offsets = attrLoc
          ? { startOffset: attrLoc.startOffset, endOffset: attrLoc.endOffset }
          : { startOffset: loc ? loc.startOffset : 0, endOffset: loc ? loc.endOffset : 0 };
        if (name.startsWith("on")) {
          contexts.push({ kind: "unsupported-event-handler", attrName: name, ...offsets });
        } else if (typeof a.value === "string" && /^\s*javascript:/i.test(a.value)) {
          contexts.push({ kind: "unsupported-javascript-url", attrName: name, ...offsets });
        }
      }
    }
    for (const child of node.childNodes || []) walk(child, [...ancestors, node]);
  };
  walk(document, []);
  return contexts;
}

/** True when a UTF-16 offset lies inside a <pre> or <code> element. */
export function htmlOffsetInDisplayRegion(document, offset) {
  let inDisplay = false;
  const walk = (node, displayAncestor) => {
    const display =
      displayAncestor || node.tagName === "pre" || node.tagName === "code";
    if (
      display &&
      node.sourceCodeLocation &&
      node.sourceCodeLocation.startOffset <= offset &&
      offset < node.sourceCodeLocation.endOffset &&
      (node.tagName === "pre" || node.tagName === "code")
    ) {
      inDisplay = true;
    }
    for (const child of node.childNodes || []) walk(child, display);
  };
  walk(document, false);
  return inDisplay;
}
