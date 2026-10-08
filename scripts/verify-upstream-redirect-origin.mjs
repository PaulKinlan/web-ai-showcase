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
  if (path === "/cross") {
    return new Response(null, { status: 302, headers: { location: `${FOREIGN_ORIGIN}/steal` } });
  }
  if (path === "/same") {
    return new Response(null, { status: 301, headers: { location: "/target" } });
  }
  if (path === "/target") {
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

foreign.shutdown();
upstream.shutdown();
console.log(ok ? "\nALL FIXTURE CHECKS PASS" : "\nFIXTURE CHECKS FAILED");
Deno.exit(ok ? 0 : 1);
