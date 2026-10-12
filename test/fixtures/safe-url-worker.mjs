// Runs the attribution URL policy in a real worker thread: a Worker has no `document` (Node's worker
// has no `location` either), which is exactly the environment web-ai-showcase-94s broke in.
import { parentPort } from "node:worker_threads";
import { safeSourceHref } from "../../lib/safe-url.js";

parentPort.postMessage({
  noDocument: typeof document === "undefined",
  noLocation: typeof location === "undefined",
  absolute: safeSourceHref("https://example.com"),
  absoluteWithPath: safeSourceHref("https://example.com/a/b?q=1#f"),
  absoluteUpper: safeSourceHref("HTTPS://EXAMPLE.COM"),
  relative: safeSourceHref("/a/b"),
  relativeDot: safeSourceHref("./a"),
  protocolRelative: safeSourceHref("//evil.test/x"),
  javascript: safeSourceHref("javascript:alert(1)"),
  data: safeSourceHref("data:text/html,<script>1</script>"),
  ftp: safeSourceHref("ftp://a/b"),
});
