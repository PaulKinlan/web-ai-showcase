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
// Cooked-escape marker forms (package name followed by an escaped at-sign,
// e.g. the JS escape \x40 or the HTML entity &#64;) are discovered by the
// cooked-literal census pass, not by raw byte search.

/** A file is binary (fail-closed binary-NUL pass) iff it contains a NUL byte. */
export function isBinarySource(source) {
  return source.includes("\u0000");
}

/**
 * TOTAL marker discovery: every raw contiguous marker, the regex-literal
 * escaped variant, and cooked/entity/percent escape forms, with exact UTF-16
 * offsets. JS escapes covered: \x40 \u0040 \u{40} and octal \100 (non-strict
 * strings); HTML entities: &#64; &#x40; &#X40; &commat;; URL percent: %40.
 * Discovery NEVER approves: hits are classified and reconciled, and residual
 * unenumerated forms fail closed via the decoded-value scan + unsupported
 * rules in design/runtime-pin-ast-parser.md §3.
 */
export function markerOffsets(source) {
  const hits = [];
  const variants = [
    ...RAW_MARKERS.map((m) => ({ needle: m, escaped: false })),
    ...ESCAPED_MARKER_VARIANTS.map((m) => ({ needle: m, escaped: true })),
  ];
  for (const { needle, escaped } of variants) {
    let idx = source.indexOf(needle);
    while (idx !== -1) {
      hits.push({ offset: idx, marker: needle, escaped });
      idx = source.indexOf(needle, idx + 1);
    }
  }
  const bases = ["onnxruntime-web", "@huggingface/transformers", "@huggingface\\/transformers"];
  const escapes = ["\\x40", "\\u0040", "\\u{40}", "\\100", "%40", "&#64;", "&#x40;", "&#X40;", "&commat;"];
  for (const base of bases) {
    for (const esc of escapes) {
      const needle = base + esc;
      let idx = source.indexOf(needle);
      while (idx !== -1) {
        hits.push({ offset: idx, marker: needle, escaped: true, cookedEscape: true });
        idx = source.indexOf(needle, idx + 1);
      }
    }
  }
  hits.sort((a, b) => a.offset - b.offset);
  return hits;
}

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
// explicitly AND still visit node.value (the default behaviour for all three
// node kinds) — otherwise method bodies, getters and class-field initializers
// would silently drop out of the span table and their markers would
// misclassify as bare "code".
const WALK_BASE = { ...acornWalk.base };
const withLiteralKey = () => (node, state, c) => {
  if (node.computed || node.key.type === "Literal") c(node.key, state, "Expression");
  if (node.value) c(node.value, state, "Expression");
};
WALK_BASE.Property = withLiteralKey();
WALK_BASE.PropertyDefinition = withLiteralKey();
WALK_BASE.MethodDefinition = withLiteralKey();

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
 *   const NAME = "<primitive string>";   // top level
 *   ... import(`…${NAME}…`) / wasmPaths = `…${NAME}…`
 * Returns a Map name → { raw, value, declaratorStart, declaratorEnd } for
 * top-level `const` bindings whose value is a plain string Literal. A name is
 * INVALIDATED (absent from the result → the caller must fail closed on the
 * unresolved sink) when ANY of these holds anywhere in the file:
 *  - the name is reassigned (AssignmentExpression) or updated (++/--);
 *  - the name is re-bound in ANY nested scope: block `let/const/function`,
 *    function params (any pattern incl. destructuring), catch param,
 *    for-in/of loop bindings, named function/class expressions, imports;
 *  - the name is referenced through anything but a bare Identifier that IS the
 *    whole template expression (`${NAME}`) — member reads, call args,
 *    concatenation (`${NAME + "x"}`), shorthand properties etc. all invalidate.
 */
export function resolveTopLevelStringConstants(ast) {
  const bindings = new Map();
  const declaratorIds = new Set(); // Identifier nodes that ARE the top-level declarator
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
        declaratorIds.add(decl.id);
      }
    }
  }
  if (bindings.size === 0) return bindings;

  // Collect every Identifier bound by a binding pattern (params, declarator
  // ids, catch params, import locals …) at ANY depth.
  const patternIds = (pattern, out) => {
    if (!pattern) return;
    switch (pattern.type) {
      case "Identifier":
        out.push(pattern);
        break;
      case "ObjectPattern":
        for (const p of pattern.properties) {
          patternIds(p.type === "RestElement" ? p.argument : p.value, out);
        }
        break;
      case "ArrayPattern":
        for (const el of pattern.elements) patternIds(el, out);
        break;
      case "RestElement":
        patternIds(pattern.argument, out);
        break;
      case "AssignmentPattern":
        patternIds(pattern.left, out);
        break;
    }
  };

  const invalidate = (name) => bindings.delete(name);

  acornWalk.fullAncestor(
    ast,
    // acorn-walk callback signature: (node, state, ancestors, type).
    (node, _state, ancestors) => {
      // 1. Writes.
      if (node.type === "AssignmentExpression") {
        const ids = [];
        patternIds(node.left, ids);
        for (const id of ids) invalidate(id.name);
        return;
      }
      if (node.type === "UpdateExpression" && node.argument.type === "Identifier") {
        invalidate(node.argument.name);
        return;
      }
      // 2. Re-bindings in any scope (skip the top-level declarator itself).
      if (node.type === "VariableDeclarator" && !declaratorIds.has(node.id)) {
        const ids = [];
        patternIds(node.id, ids);
        for (const id of ids) if (!declaratorIds.has(id)) invalidate(id.name);
        return;
      }
      if (
        node.type === "FunctionDeclaration" ||
        node.type === "FunctionExpression" ||
        node.type === "ArrowFunctionExpression"
      ) {
        // Any function id (named expression, or a nested/top-level declaration
        // sharing the name) re-binds or conflicts with the const — invalidate.
        if (node.id) invalidate(node.id.name);
        for (const param of node.params || []) {
          const ids = [];
          patternIds(param, ids);
          for (const id of ids) invalidate(id.name);
        }
        return;
      }
      if (node.type === "CatchClause" && node.param) {
        const ids = [];
        patternIds(node.param, ids);
        for (const id of ids) invalidate(id.name);
        return;
      }
      if (
        node.type === "ImportSpecifier" ||
        node.type === "ImportDefaultSpecifier" ||
        node.type === "ImportNamespaceSpecifier"
      ) {
        invalidate(node.local.name);
        return;
      }
      if (node.type === "ClassDeclaration" || node.type === "ClassExpression") {
        if (node.id) invalidate(node.id.name);
        return;
      }
      // 3. Bare-read rule: every remaining Identifier reference to a bound
      //    name must BE an entire template expression (`${NAME}`), nothing
      //    else. Identifier roles that are not reads (member properties,
      //    property keys, labels) are ignored; everything else invalidates.
      if (node.type === "Identifier" && bindings.has(node.name) && !declaratorIds.has(node)) {
        const parent = ancestors.length >= 2 ? ancestors[ancestors.length - 2] : null;
        if (!parent) return;
        if (
          (parent.type === "MemberExpression" && parent.property === node && !parent.computed) ||
          (parent.type === "Property" && parent.key === node && !parent.computed) ||
          parent.type === "LabeledStatement" ||
          parent.type === "BreakStatement" ||
          parent.type === "ContinueStatement" ||
          parent.type === "PropertyDefinition" ||
          parent.type === "MethodDefinition"
        ) {
          return; // not a binding read
        }
        const isBareTemplateRead =
          parent.type === "TemplateLiteral" && parent.expressions.includes(node);
        if (!isBareTemplateRead) invalidate(node.name);
      }
    },
    WALK_BASE,
  );
  return bindings;
}

/**
 * Return the innermost TemplateLiteral containing `offset`, or null.
 * `{ node, tagged }` — `tagged` is true when the template is the quasi of a
 * TaggedTemplateExpression; a tagged template's cooked text is NOT necessarily
 * the runtime string (the tag function may build anything from the raw parts),
 * so callers doing constant binding MUST treat tagged templates as
 * unresolvable (fail closed) rather than judge the cooked text (bead vs2,
 * design review P2-3).
 */
export function templateLiteralAt(parsed, offset) {
  let found = null;
  acornWalk.fullAncestor(
    parsed.ast,
    (node, _state, ancestors) => {
      if (node.type !== "TemplateLiteral") return;
      if (!(node.start <= offset && offset < node.end)) return;
      if (found && node.end - node.start >= found.node.end - found.node.start) return;
      const parent = ancestors[ancestors.length - 2];
      found = {
        node,
        tagged: parent?.type === "TaggedTemplateExpression" && parent.quasi === node,
      };
    },
    WALK_BASE,
  );
  return found;
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

// Full JavaScript MIME essence list per the HTML spec (plus "module" and the
// empty classic default). Anything else is non-executable and fails closed on
// a marker rather than being misread as runnable JS.
const EXECUTABLE_SCRIPT_TYPES = new Set([
  "", // classic script
  "module",
  "text/javascript",
  "application/javascript",
  "text/ecmascript",
  "application/ecmascript",
  "text/jscript",
  "text/livescript",
  "text/x-javascript",
  "application/x-javascript",
  "text/javascript1.0",
  "text/javascript1.1",
  "text/javascript1.2",
  "text/javascript1.3",
  "text/javascript1.4",
  "text/javascript1.5",
]);

// Browsers execute a script when the MIME ESSENCE is a JavaScript type;
// parameters (`; charset=utf-8`) do not change that.
function scriptTypeEssence(typeAttr) {
  return (typeAttr || "").split(";")[0].trim().toLowerCase();
}

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
 *  { kind: "script-src",      src, attrName, startOffset, endOffset }  // src / SVG href / xlink:href, quoted or unquoted
 *  { kind: "script-inline",   text, startOffset, endOffset, scriptType }
 *  { kind: "script-nonexec",  scriptType, text, startOffset, endOffset } // importmap/json/etc
 *  { kind: "unsupported-event-handler", attrName, startOffset, endOffset }
 *  { kind: "unsupported-javascript-url", attrName, startOffset, endOffset }
 *  { kind: "unsupported-unlocated-src", attrName, startOffset, endOffset }
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
      const srcAttr = (node.attrs || []).find(
        (a) => a.name.toLowerCase() === "src" || a.name.toLowerCase() === "href",
      );
      if (srcAttr) {
        const qualified = srcAttr.prefix ? `${srcAttr.prefix}:${srcAttr.name.toLowerCase()}` : srcAttr.name.toLowerCase();
        const attrLoc = loc.attrs && (loc.attrs[qualified] || loc.attrs[srcAttr.name.toLowerCase()]);
        if (attrLoc && attrLoc.startOffset !== undefined) {
          contexts.push({
            kind: "script-src",
            src: srcAttr.value,
            attrName: qualified,
            startOffset: attrLoc.startOffset,
            endOffset: attrLoc.endOffset,
          });
        } else {
          // parse5 always supplies attr locations for quoted AND unquoted
          // attrs; a missing location means a construct this design has not
          // classified — fail closed.
          contexts.push({
            kind: "unsupported-unlocated-src",
            attrName: qualified,
            startOffset: loc.startOffset,
            endOffset: loc.endOffset,
          });
        }
      } else {
        const scriptType = scriptTypeEssence(attrs.type);
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
            scriptType: (attrs.type || "").toLowerCase() || "unknown",
            text: textNode ? textNode.value : "",
            startOffset: textNode && textNode.sourceCodeLocation
              ? textNode.sourceCodeLocation.startOffset
              : loc.startOffset,
            endOffset: textNode && textNode.sourceCodeLocation
              ? textNode.sourceCodeLocation.endOffset
              : loc.endOffset,
          });
        }
      }
    }
    // Event handlers and javascript: URLs on ANY element: unsupported contexts.
    // The URL parser strips ASCII TAB/LF/CR before scheme matching, so the
    // javascript: test must do the same ("java\nscript:" is a real URL).
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
        } else if (
          typeof a.value === "string" &&
          /^\s*javascript:/i.test(a.value.replace(/[\t\n\r]/g, ""))
        ) {
          contexts.push({ kind: "unsupported-javascript-url", attrName: name, ...offsets });
        }
      }
    }
    // <template> content lives under node.content in parse5 — without this the
    // walker is blind to an entire executable subtree.
    if (node.content && node.content.childNodes) {
      for (const child of node.content.childNodes) walk(child, [...ancestors, node]);
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
    if (node.content && node.content.childNodes) {
      for (const child of node.content.childNodes) walk(child, display);
    }
    for (const child of node.childNodes || []) walk(child, display);
  };
  walk(document, false);
  return inDisplay;
}
