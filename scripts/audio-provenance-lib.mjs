// Constants shared by scripts/check-audio-provenance.mjs and test/audio-provenance.test.mjs.
//
// WHY THIS FILE EXISTS (bead web-ai-showcase-eba, fix pass 3): the gate's scope rule used to be an
// ALLOWLIST of audio extensions, so every extension NOT on the list was skipped silently. That failed
// OPEN — a .mkv (the same Matroska container as the .webm the gate admits), a .mov, a .caf, a .w64 all
// carried real audio and passed with rc=0. The rule is now INVERTED: every git-tracked file is in scope
// UNLESS it is on the explicit "cannot contain audio" allowlist below. An unknown, new, or no-extension
// file therefore FAILS CLOSED until a human adds it here with a justification — one spelling of the
// scope rule, imported by both the gate and its tests.
//
// FALSE-POSITIVE COST, stated honestly: adding a legitimate non-audio file type (a new font format, a
// new doc format, a project file) makes the gate red until its extension is added below. That is the
// intended direction — the failure is loud, explicit and reviewable in the diff, which is far better
// than a real audio asset shipping silently because nobody had added its extension to a list.

// Extensions that carry (or can carry) an audio track. This set does NOT decide scope — scope comes
// from NON_AUDIO_EXT below — it only lets the gate say "this is a known media file" instead of "this is
// a known audio format" when it reports an undeclared file. Keep the common cases complete so the
// message is useful.
export const KNOWN_MEDIA_EXT = new Set([
  // audio
  "wav", "wave", "mp3", "mp2", "ogg", "oga", "opus", "m4a", "m4b", "m4p", "flac", "aac", "aiff",
  "aif", "aifc", "alac", "wma", "amr", "ac3", "dts", "ape", "wv", "mka", "caf", "w64", "au", "snd",
  "ra", "rm", "gsm", "dss", "msv", "dvf", "vox", "tta", "spx", "shn", "mid", "midi", "kar",
  // media containers that can carry an audio track
  "mp4", "m4v", "mov", "qt", "mkv", "webm", "ogv", "avi", "wmv", "flv", "f4v", "mpg", "mpeg", "mpe",
  "m2v", "vob", "3gp", "3g2", "ts", "m2ts", "mts", "rmvb", "asf", "amv", "mxf", "y4m", "dv",
]);

// KNOWN COLLISION, documented rather than hidden: `ts` is classified as TypeScript source here because
// that is what a `.ts` file is in this repository, while `.ts` is also the MPEG transport-stream
// container. An MPEG-TS file renamed `.ts` is therefore out of scope; if the repo ever bundles one, the
// classification has to change (and that is a reviewable diff).
//
// Extensions whose files cannot carry a playable audio track: source code, markup, styles, text and
// documentation, config, still images, fonts, structured data/model weights, archives and other
// non-media binary formats. Category strings are documentation for a reviewer.
export const NON_AUDIO_EXT = new Map(Object.entries({
  // code
  ...Object.fromEntries(["js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts", "py", "pyi", "rb", "go",
    "rs", "java", "kt", "kts", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "cs", "php", "lua", "pl",
    "pm", "r", "swift", "dart", "scala", "sh", "bash", "zsh", "fish", "ksh", "ps1", "psm1", "bat",
    "cmd", "asm", "s", "v", "sv", "vhd", "vhdl", "ex", "exs", "erl", "hs", "ml", "mli", "clj", "cljs",
    "groovy", "gradle", "vb", "f90", "for", "jl", "nim", "zig", "sql", "wat"].map((e) => [e, "source code"])),
  // markup + styles
  ...Object.fromEntries(["html", "htm", "xhtml", "xml", "svg", "vue", "svelte", "astro", "njk",
    "hbs", "handlebars", "mustache", "ejs", "pug", "jade", "liquid", "jinja", "j2", "tmpl", "erb",
    "css", "scss", "sass", "less", "styl"].map((e) => [e, "markup or stylesheet"])),
  // text + documentation + subtitles
  ...Object.fromEntries(["md", "markdown", "mdx", "txt", "text", "rst", "adoc", "asciidoc", "org", "tex",
    "bib", "man", "rtf", "csv", "tsv", "psv", "log", "nfo", "srt", "vtt", "ass", "ssa", "sub", "po",
    "pot", "diff", "patch"].map((e) => [e, "text or documentation"])),
  // config
  ...Object.fromEntries(["json", "jsonc", "json5", "ndjson", "jsonl", "yaml", "yml", "toml", "ini",
    "cfg", "conf", "config", "properties", "env", "lock", "gitignore", "gitattributes", "gitmodules",
    "editorconfig", "npmrc", "nvmrc", "dockerignore", "eslintignore", "prettierignore", "babelrc",
    "browserslistrc", "npmignore", "ignore", "rules", "watchmanconfig", "csproj", "fsproj", "vbproj",
    "sln", "cmake", "mk", "make", "plist", "webmanifest", "manifest"].map((e) => [e, "configuration"])),
  // still images — a still image carries pixels, not an audio track
  ...Object.fromEntries(["jpg", "jpeg", "jpe", "png", "apng", "gif", "webp", "avif", "bmp", "ico",
    "cur", "tif", "tiff", "heic", "heif", "jxl", "jp2", "j2k", "psd", "xcf", "exr", "tga", "dds",
    "pbm", "pgm", "ppm", "pnm", "svgz", "eps", "ai", "sketch", "fig"].map((e) => [e, "still image"])),
  // fonts
  ...Object.fromEntries(["woff", "woff2", "ttf", "otf", "eot", "ttc", "pfb", "pfm", "afm"].map((e) => [e, "font"])),
  // structured data, model weights and other non-media binary blobs
  ...Object.fromEntries(["parquet", "arrow", "feather", "orc", "avro", "tfrecord", "h5", "hdf5",
    "pt", "pth", "onnx", "safetensors", "gguf", "ggml", "npy", "npz", "pkl", "pickle", "msgpack",
    "msgpck", "cbor", "bson", "sqlite", "sqlite3", "db", "mdb", "wasm", "map", "proto", "graphql",
    "sol", "abi", "prof", "cpuprofile", "heapsnapshot", "trace", "etl", "pcap"].map((e) => [e, "data or model blob"])),
  // archives, containers and compiled artefacts
  ...Object.fromEntries(["zip", "tar", "gz", "tgz", "bz2", "xz", "lz", "lzma", "zst", "7z", "rar",
    "jar", "war", "whl", "deb", "rpm", "apk", "aab", "ipa", "dmg", "iso", "img", "so", "dylib", "dll",
    "a", "o", "obj", "lib", "exe", "class", "pyc", "pyo", "pyd", "gem", "crate", "vsix"].map((e) => [e, "archive or compiled artefact"])),
  // certificates, keys and integrity sidecars
  ...Object.fromEntries(["pem", "crt", "cer", "der", "key", "pub", "p12", "pfx", "asc", "gpg", "sig",
    "md5", "sha1", "sha256", "sha512", "sum", "sfv", "torrent"].map((e) => [e, "certificate, key or checksum"])),
}));

// Extension-less tracked files that cannot contain audio are listed BY EXACT PATH, so a NEW
// extension-less file is in scope and fails closed rather than being waved through.
export const NON_AUDIO_PATH = new Map(Object.entries({
  ".beads/hooks/post-checkout": "git hook script (extension-less)",
  ".beads/hooks/post-merge": "git hook script (extension-less)",
  ".beads/hooks/pre-commit": "git hook script (extension-less)",
  ".beads/hooks/pre-push": "git hook script (extension-less)",
  ".beads/hooks/prepare-commit-msg": "git hook script (extension-less)",
  // Opaque .bin blobs are listed by exact path rather than by extension: model weights and a vector
  // index cannot carry an audio track, but a bare .bin CAN be anything, so a NEW .bin anywhere else
  // stays in scope and must be justified here.
  "models/speecht5-tts/speakers/awb.bin": "SpeechT5 speaker embedding (model weight)",
  "models/speecht5-tts/speakers/bdl.bin": "SpeechT5 speaker embedding (model weight)",
  "models/speecht5-tts/speakers/clb.bin": "SpeechT5 speaker embedding (model weight)",
  "models/speecht5-tts/speakers/ksp.bin": "SpeechT5 speaker embedding (model weight)",
  "models/speecht5-tts/speakers/rms.bin": "SpeechT5 speaker embedding (model weight)",
  "models/speecht5-tts/speakers/slt.bin": "SpeechT5 speaker embedding (model weight)",
  "search/index/vectors.i8.bin": "int8 search-vector index (binary data blob)",
}));

// The extension is everything after the last dot of the basename, lower-cased; a leading-dot filename
// like .gitignore counts as extension "gitignore", and a name with no dot (Makefile, a git hook) has
// extension "" and is therefore in scope unless listed in NON_AUDIO_PATH.
export function extensionOf(path) {
  const base = path.slice(path.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  return dot < 0 ? "" : base.slice(dot + 1).toLowerCase();
}

/** Fail-closed scope: a tracked file is in scope unless it is explicitly known not to carry audio. */
export function inScope(path) {
  if (NON_AUDIO_PATH.has(path)) return false;
  const ext = extensionOf(path);
  if (!ext) return true; // unknown extension-less file → in scope
  return !NON_AUDIO_EXT.has(ext);
}

// The audio-provenance legacy baseline is anchored to this commit. The anchor is CODE on purpose: it
// lives here, not in the ledger, so repointing legacyBaseline.sha at some other commit (or at "HEAD")
// and regenerating the hashes is a code change that shows up in a reviewable diff and FAILS the gate
// with BASELINE ANCHOR MOVED. Moving it is a deliberate act of review, not a ledger edit.
export const BASELINE_SHA = "e9c20b75ba5b83866ad4367461a8100a07dc5afc";
