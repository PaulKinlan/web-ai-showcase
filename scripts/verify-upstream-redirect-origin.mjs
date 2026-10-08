// Verifies the runtime semantics that server.ts's upstream redirect guard depends on, against real
// local servers and real redirects rather than a fake fetch.
//
// Why this exists: the guard in server.ts can only refuse an off-origin redirect hop if fetch with
// redirect: "manual" exposes the 3xx status and the Location header. In a BROWSER, redirect: "manual"
// yields an opaque-redirect response with status 0 and no readable headers, and a guard written the
// same way would silently pass the redirect through. Deno's server-side fetch does expose them; this
// script pins that, so a future runtime change that made "manual" opaque fails loudly here instead of
// quietly reopening the hole.
//
// It also demonstrates the original defect: with redirect: "follow" the foreign origin IS reached and
// its bytes ARE returned, which is exactly what the proxy would have republished under the canonical
// origin.
//
// Run: deno run --allow-net scripts/verify-upstream-redirect-origin.mjs
// Not part of `deno task gate`: it binds local sockets, so it is deliberately kept out of the
// permission-free server test command.

import { CANONICAL_ORIGIN, createHandler, SITE_PREFIX } from "../server.ts";

const hits = { steal: 0, target: 0 };

const foreign = Deno.serve({ port: 0, hostname: "127.0.0.1" }, (request) => {
  const path = new URL(request.url).pathname;
  if (path === "/steal") {
    hits.steal++;
    return new Response("EXTERNAL BYTES", { headers: { "content-type": "text/html" } });
  }
  return new Response("nope", { status: 404 });
});
// Differing ports are differing origins for the same scheme and host, so this is genuinely off-origin.
const FOREIGN_ORIGIN = `http://127.0.0.1:${foreign.addr.port}`;

const upstream = Deno.serve({ port: 0, hostname: "127.0.0.1" }, (request) => {
  const path = new URL(request.url).pathname;
  // endsWith so the same fixture serves both the direct fetch checks below and the SITE_PREFIX-prefixed
  // paths the handler builds when it proxies.
  if (path.endsWith("/cross")) {
    return new Response(null, { status: 302, headers: { location: `${FOREIGN_ORIGIN}/steal` } });
  }
  if (path.endsWith("/same")) {
    return new Response(null, {
      status: 301,
      headers: { location: `${SITE_PREFIX}/models/target` },
    });
  }
  if (path.endsWith("/target")) {
    hits.target++;
    return new Response("<html>local target</html>", { headers: { "content-type": "text/html" } });
  }
  return new Response("nope", { status: 404 });
});
const UPSTREAM = `http://127.0.0.1:${upstream.addr.port}`;
console.log(`upstream=${UPSTREAM} foreign=${FOREIGN_ORIGIN}`);

let ok = true;
function check(label, condition, detail) {
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}${detail ? `  ${detail}` : ""}`);
  if (!condition) ok = false;
}

const manual = await fetch(`${UPSTREAM}/cross`, { redirect: "manual" });
check(
  "manual: status is the real 3xx, not opaque 0",
  manual.status === 302,
  `status=${manual.status}`,
);
check(
  "manual: Location is readable",
  manual.headers.get("location") === `${FOREIGN_ORIGIN}/steal`,
  `location=${manual.headers.get("location")}`,
);
check(
  "manual: the foreign origin was never contacted",
  hits.steal === 0,
  `stealHits=${hits.steal}`,
);

const followed = await fetch(`${UPSTREAM}/cross`, { redirect: "follow" });
const body = await followed.text();
check(
  "follow: reaches the FOREIGN body (the reported defect, before the fix)",
  body.includes("EXTERNAL BYTES"),
  `status=${followed.status} body=${JSON.stringify(body.slice(0, 20))}`,
);
check(
  "follow: the final response.url shows the escape",
  new URL(followed.url).origin === FOREIGN_ORIGIN,
  `url=${followed.url}`,
);

const same = await fetch(`${UPSTREAM}/same`, { redirect: "follow" });
check(
  "same-origin redirects still resolve, so the fix does not break legitimate proxying",
  (await same.text()).includes("local target"),
  `status=${same.status} targetHits=${hits.target}`,
);

// -------------------------------------------------------------------------------------------------
// End-to-end through the REAL handler. The checks above prove fetch semantics; these prove the proxy
// itself, which the earlier fixture could not do because UPSTREAM_ORIGIN was a module constant with no
// way to point the handler at a local stand-in. Now the origin is injectable for exactly this.
const handlerHits = { steal: 0, target: 0 };
const foreignBefore = hits.steal;
const foreignE2E = Deno.serve({ port: 0, hostname: "127.0.0.1" }, (request) => {
  if (new URL(request.url).pathname === "/steal") {
    handlerHits.steal++;
    return new Response("EXTERNAL BYTES", { headers: { "content-type": "text/html" } });
  }
  return new Response("nope", { status: 404 });
});
const FOREIGN_E2E = `http://127.0.0.1:${foreignE2E.addr.port}`;
const upstreamE2E = Deno.serve({ port: 0, hostname: "127.0.0.1" }, (request) => {
  const path = new URL(request.url).pathname;
  if (path.endsWith("/models/cross")) {
    return new Response(null, { status: 302, headers: { location: `${FOREIGN_E2E}/steal` } });
  }
  if (path.endsWith("/models/same")) {
    return new Response(null, {
      status: 301,
      headers: { location: `${SITE_PREFIX}/models/target` },
    });
  }
  if (path.endsWith("/models/target")) {
    handlerHits.target++;
    return new Response("<html>E2E target</html>", { headers: { "content-type": "text/html" } });
  }
  return new Response("nope", { status: 404 });
});
const UPSTREAM_E2E = `http://127.0.0.1:${upstreamE2E.addr.port}`;
// The production default is untouched; only this call passes an origin, and the real fetch is used.
const handler = createHandler(fetch, UPSTREAM_E2E);

const crossResponse = await handler(new Request(`${CANONICAL_ORIGIN}/models/cross`));
check(
  "handler: refuses a cross-origin redirect with 502",
  crossResponse.status === 502,
  `status=${crossResponse.status}`,
);
check(
  "handler: the foreign origin was NEVER CONTACTED (counter, not status)",
  handlerHits.steal === 0,
  `foreignHits=${handlerHits.steal}`,
);
check(
  "handler: no foreign bytes in the refusal body",
  !(await crossResponse.text()).includes("EXTERNAL BYTES"),
  `body withheld`,
);

const sameResponse = await handler(new Request(`${CANONICAL_ORIGIN}/models/same`));
const sameBody = await sameResponse.text();
check(
  "handler: still follows a same-origin redirect and returns the target body",
  sameResponse.status === 200 && sameBody.includes("E2E target"),
  `status=${sameResponse.status} targetHits=${handlerHits.target}`,
);

foreignE2E.shutdown();
upstreamE2E.shutdown();
check(
  "handler: the direct-fetch foreign counter was not disturbed by the handler run",
  hits.steal === foreignBefore,
  `before=${foreignBefore} after=${hits.steal}`,
);

foreign.shutdown();
upstream.shutdown();
console.log(ok ? "\nALL FIXTURE CHECKS PASS" : "\nFIXTURE CHECKS FAILED");
Deno.exit(ok ? 0 : 1);
