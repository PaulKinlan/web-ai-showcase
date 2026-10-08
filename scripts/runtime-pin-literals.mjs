// Additive source-aware audit of complete literal CDN runtime URLs. The numeric scan remains intact.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const TJS = "@huggingface/transformers" + "@";
const ESCAPED_TJS = "@huggingface\\/transformers" + "@";
const ORT = "onnxruntime-web" + "@";
const MARKERS = [TJS, ESCAPED_TJS, ORT];
const TARGETS = ["models", "lib", "public", "scripts", "search", "models.json", "sw.js", "runtime-integrity.json"];
const LEDGER = "inventory/runtime-pin-marker-ledger.json";

function filesAt(root) {
  const files = [];
  const visit = (name) => {
    const path = join(root, name);
    if (!existsSync(path)) return;
    const stat = statSync(path);
    if (stat.isDirectory()) {
      for (const child of readdirSync(path).sort()) visit(join(name, child));
    } else if (stat.isFile()) files.push(name.split("\\").join("/"));
  };
  for (const name of TARGETS) visit(name);
  return files.sort();
}

function lexicalJs(text, offset = 0) {
  const spans = [];
  let i = 0;
  while (i < text.length) {
    const start = i;
    const c = text[i];
    if (c === "/" && text[i + 1] === "/") {
      i = text.indexOf("\n", i + 2);
      if (i < 0) i = text.length;
      spans.push({ start: offset + start, end: offset + i, kind: "line-comment" });
      continue;
    }
    if (c === "/" && text[i + 1] === "*") {
      const end = text.indexOf("*/", i + 2);
      i = end < 0 ? text.length : end + 2;
      spans.push({ start: offset + start, end: offset + i, kind: "block-comment" });
      continue;
    }
    if (c !== "'" && c !== '"' && c !== "`") { i++; continue; }
    const quote = c;
    let escaped = false;
    let interpolated = false;
    i++;
    while (i < text.length) {
      if (text[i] === "\\") { escaped = true; i += 2; continue; }
      if (quote === "`" && text[i] === "$" && text[i + 1] === "{") interpolated = true;
      if (text[i++] === quote) break;
    }
    const closed = text[i - 1] === quote;
    spans.push({
      start: offset + start, end: offset + i,
      kind: quote === "`" && interpolated ? "template" : "string",
      value: text.slice(start + 1, closed ? i - 1 : i),
      escaped, closed,
    });
  }
  return spans;
}

function lexicalHtml(text) {
  const spans = [];
  let i = 0;
  let displayDepth = 0;
  while (i < text.length) {
    const start = text.indexOf("<", i);
    if (start < 0) break;
    if (text.startsWith("<!--", start)) {
      const end = text.indexOf("-->", start + 4);
      i = end < 0 ? text.length : end + 3;
      spans.push({ start, end: i, kind: "html-comment" });
      continue;
    }
    let end = start + 1;
    let quote = null;
    for (; end < text.length; end++) {
      if (quote) { if (text[end] === quote) quote = null; }
      else if (text[end] === "'" || text[end] === '"') quote = text[end];
      else if (text[end] === ">") break;
    }
    if (end >= text.length) break;
    const tag = text.slice(start, end + 1);
    const name = /^<\s*(\/?)\s*([a-z][a-z0-9-]*)/i.exec(tag);
    i = end + 1;
    if (!name) continue;
    const closing = Boolean(name[1]);
    const tagName = name[2].toLowerCase();
    if (["pre", "code", "style"].includes(tagName)) {
      displayDepth = Math.max(0, displayDepth + (closing ? -1 : 1));
      continue;
    }
    if (displayDepth || closing || tagName !== "script") continue;
    // Attribute parsing is local to the complete tag, not arbitrary HTML prose.
    const attrs = new Map();
    let p = name[0].length;
    while (p < tag.length - 1) {
      while (/\s/.test(tag[p] ?? "")) p++;
      const attrStart = p;
      while (p < tag.length && /[\w:-]/.test(tag[p])) p++;
      if (p === attrStart) { p++; continue; }
      const key = tag.slice(attrStart, p).toLowerCase();
      while (/\s/.test(tag[p] ?? "")) p++;
      if (tag[p] !== "=") continue;
      p++;
      while (/\s/.test(tag[p] ?? "")) p++;
      if (tag[p] !== "'" && tag[p] !== '"') {
        const from = p;
        while (p < tag.length && !/[\s>]/.test(tag[p])) p++;
        attrs.set(key, { value: tag.slice(from, p), quoted: false, start: start + from, end: start + p });
        continue;
      }
      const delimiter = tag[p++];
      const from = p;
      while (p < tag.length && tag[p] !== delimiter) p++;
      attrs.set(key, { value: tag.slice(from, p), quoted: true, start: start + from, end: start + p });
      p++;
    }
    const type = attrs.get("type")?.value.toLowerCase() ?? "";
    const executable = !type || type === "module" || type === "text/javascript" || type === "application/javascript";
    if (!executable) continue;
    const src = attrs.get("src");
    if (src) spans.push({ ...src, kind: src.quoted ? "html-attribute" : "html-unquoted", escaped: /&/.test(src.value), closed: src.quoted });
    const lower = text.toLowerCase();
    const close = lower.indexOf("</script", i);
    const bodyEnd = close < 0 ? text.length : close;
    spans.push(...lexicalJs(text.slice(i, bodyEnd), i));
    i = bodyEnd;
  }
  return spans;
}

function markerHits(line) {
  const hits = [];
  for (const marker of MARKERS) {
    let at = 0;
    while ((at = line.indexOf(marker, at)) >= 0) {
      hits.push({ at, marker });
      at += marker.length;
    }
  }
  return hits.sort((a, b) => a.at - b.at);
}

function numericPrefix(s) { return /^\d+\.\d+\.\d+/.test(s); }
function hash(s) { return createHash("sha256").update(s).digest("hex"); }
function sourceContext(path, span, line) {
  if (path.endsWith("_questions.json")) return "json-prose";
  if (path === "runtime-integrity.json") return "json-generated-note";
  if (path === "scripts/runtime-integrity.mjs" && line.includes("not listed above")) return "wildcard-coverage-note";
  if (span?.kind === "line-comment") return "line-comment";
  if (span?.kind === "template" && path === "scripts/audit-model-currency.mjs") return "tool-template";
  if (span?.kind === "template") return "template";
  if (["scripts/raw-ort-inventory.mjs", "scripts/measure-raw-ort-dual-runtime.mjs"].includes(path) && line.includes("/")) return "tool-regex";
  return span?.kind ?? "code";
}

function validateUrl(raw, marker, file, line, allowlist, isDerivedInventoryHit) {
  const errors = [];
  if (raw.includes("\\") || raw.includes("${") || raw.includes("*")) {
    return { errors: [`unsupported dynamic/escaped runtime URL in ${file}:${line}: ${raw}`], lib: null };
  }
  let url;
  try { url = new URL(raw); } catch {
    return { errors: [`invalid runtime URL in ${file}:${line}: ${raw}`], lib: null };
  }
  if (url.protocol !== "https:" || !["cdn.jsdelivr.net", "unpkg.com", "esm.sh"].includes(url.hostname)) {
    return { errors: [`unsupported runtime CDN URL in ${file}:${line}: ${raw}`], lib: null };
  }
  const packagePath = marker === TJS ? "/@huggingface/transformers" + "@" : "/onnxruntime-web" + "@";
  const pos = raw.indexOf(packagePath);
  if (pos < 0 || (url.hostname === "cdn.jsdelivr.net" && !raw.slice(0, pos).endsWith("/npm"))) {
    return { errors: [`unsupported runtime package URL in ${file}:${line}: ${raw}`], lib: null };
  }
  const remainder = raw.slice(pos + packagePath.length);
  const version = remainder.slice(0, remainder.indexOf("/") < 0 ? undefined : remainder.indexOf("/"));
  const lib = marker === TJS ? "transformers" : "onnxruntimeWeb";
  if (!version) return { errors: [`unsupported empty runtime specifier in ${file}:${line}: ${raw}`], lib };
  if (isDerivedInventoryHit(file, line, marker === TJS ? "@huggingface/transformers" : "onnxruntime-web", version)) return { errors, lib };
  if (lib === "onnxruntimeWeb") {
    const versions = (allowlist.onnxruntimeWeb?.allowedVersions ?? []).map((v) => v.version);
    if (!versions.includes(version)) errors.push(`unauthorized onnxruntime-web version "${version}" in ${file}:${line} — not in scripts/runtime-pin-allowlist.json`);
  } else if (version !== allowlist.transformers.shared) {
    const override = allowlist.transformers.allowedLocalOverrides.find((o) => o.version === version);
    const slug = /^models\/([^/]+)\//.exec(file)?.[1];
    if (!override || !slug || !override.slugs.includes(slug)) {
      errors.push(`unauthorized @huggingface/transformers version "${version}" in ${file}:${line} — not authorized for this route`);
    }
  }
  return { errors, lib };
}

/** Return additive errors + separate literal discovery counts; never alter the legacy scanner. */
export function inspectLiteralRuntimePins(root, allowlist, isDerivedInventoryHit) {
  const errors = [];
  const counts = { transformers: 0, onnxruntimeWeb: 0 };
  const ledger = JSON.parse(readFileSync(join(root, LEDGER), "utf8"));
  const discovered = [];
  for (const file of filesAt(root)) {
    const bytes = readFileSync(join(root, file));
    if (bytes.includes(0)) {
      if (MARKERS.some((marker) => bytes.includes(Buffer.from(marker)))) {
        errors.push(`runtime pin in binary-classified file ${file} (raw package marker, including floating specifiers)`);
      }
      continue;
    }
    const text = bytes.toString("utf8");
    const sourceFile = /\.(?:js|mjs|html)$/.test(file);
    const spans = sourceFile ? (file.endsWith(".html") ? lexicalHtml(text) : lexicalJs(text)) : [];
    let fileOffset = 0;
    for (const [index, line] of text.split("\n").entries()) {
      const hits = markerHits(line);
      for (let ordinal = 0; ordinal < hits.length; ordinal++) {
        const { at, marker } = hits[ordinal];
        const pos = fileOffset + at;
        const span = spans.find((s) => s.start <= pos && pos < s.end);
        if (!numericPrefix(line.slice(at + marker.length))) {
          discovered.push({
            path: file, line: index + 1, ordinalOnLine: ordinal + 1,
            expectedCountOnLine: hits.length, marker, context: sourceContext(file, span, line),
            disposition: ledger.entries.find((entry) => entry.path === file && entry.line === index + 1 && entry.marker === marker)?.disposition ?? "unreviewed",
            sha256: hash(line),
          });
        }
      }
      fileOffset += line.length + 1;
    }
    if (!sourceFile) continue;
    for (const span of spans) {
      if (!span.value?.startsWith("http://") && !span.value?.startsWith("https://")) continue;
      for (const marker of [TJS, ORT]) {
        if (!span.value.includes(marker)) continue;
        const line = text.slice(0, span.start).split("\n").length;
        if (!span.closed || span.escaped || span.kind === "template" || span.kind === "html-unquoted" || span.value.includes("${")) {
          const reviewed = ledger.entries.some((entry) =>
            entry.path === file && entry.line === line && entry.disposition === "vs2-dynamic" &&
            entry.sha256 === hash(text.split("\n")[line - 1])
          );
          if (!reviewed) errors.push(`unsupported runtime URL literal in ${file}:${line}`);
          continue;
        }
        const result = validateUrl(span.value, marker, file, line, allowlist, isDerivedInventoryHit);
        errors.push(...result.errors);
        if (result.lib) counts[result.lib]++;
      }
    }
  }
  const expected = JSON.stringify(ledger.entries);
  if (JSON.stringify(discovered) !== expected) {
    const missing = ledger.entries.filter((entry) => !discovered.some((got) => JSON.stringify(got) === JSON.stringify(entry)));
    const added = discovered.filter((got) => !ledger.entries.some((entry) => JSON.stringify(got) === JSON.stringify(entry)));
    errors.push(`runtime marker golden ledger drift: ${added.length} new/changed, ${missing.length} missing/changed; review inventory/runtime-pin-marker-ledger.json manually (no auto-approval)`);
  }
  for (const pair of ledger.splitAssemblies) {
    for (const fragment of pair.fragments) {
      const line = readFileSync(join(root, pair.path), "utf8").split("\n")[fragment.line - 1];
      if (!line || hash(line) !== fragment.sha256) errors.push(`reviewed split-marker vs2 expression changed in ${pair.path}:${fragment.line}`);
    }
  }
  if (counts.transformers < 1) errors.push("literal runtime-pin scan saw no executable @huggingface/transformers URLs");
  if (counts.onnxruntimeWeb < 1) errors.push("literal runtime-pin scan saw no executable onnxruntime-web URLs");
  return { errors, counts };
}
