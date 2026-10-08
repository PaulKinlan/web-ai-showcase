// Local two-origin HTTP fixture: the HF-shaped request is redirected to a foreign origin.
// No real model/network service or browser is needed for the trust-boundary regression.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { prefetchModel } from "../lib/model-prefetch.mjs";
import { assertTrustedHFResponse } from "../lib/model-redirect.mjs";

const modelId = "example/model";
const file = "config.json";
const requested = `https://huggingface.co/${modelId}/resolve/main/${file}`;

test("accepts direct HF and observed Xet/LFS CDN final origins; rejects lookalikes and opaque", () => {
  const accepted = [
    requested,
    "https://us.aws.cdn.hf.co/xet-bridge-us/a/b",
    "https://eu.aws.cdn.hf.co/xet-bridge-eu/a/b",
    "https://cas-bridge.xethub-eu.hf.co/xet-bridge-eu/a/b",
    "https://cas-bridge.xethub.hf.co/xet-bridge/a/b",
    "https://cdn-lfs.huggingface.co/a",
    "https://cdn-lfs-us-1.huggingface.co/a",
  ];
  for (const url of accepted) assert.doesNotThrow(() => assertTrustedHFResponse(requested, { url }));
  for (const url of [
    "https://huggingface.co.evil.example/a", "http://huggingface.co/a",
    "https://evil.hf.co/a", "https://us.aws.cdn.hf.co.evil.example/a",
    "https://us.aws.cdn.hf.co:8443/a", "data:text/plain,fake", "",
  ]) {
    assert.throws(() => assertTrustedHFResponse(requested, { url }), /Untrusted/);
  }
  assert.throws(() => assertTrustedHFResponse(requested, { url: requested, type: "opaque" }), /Untrusted/);
  // The existing localhost browser resume fixture is deliberately out of this HF-only policy's scope.
  assert.doesNotThrow(() => assertTrustedHFResponse("http://127.0.0.1:1234/model", { url: "http://127.0.0.1:1234/model" }));
});

async function listen(server) {
  server.listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
  return `http://127.0.0.1:${server.address().port}`;
}

test("a foreign final origin is rejected before a non-LFS config reaches Cache Storage", async () => {
  let cacheWrites = 0;
  let finalUrl;
  const foreign = createServer((_req, res) => {
    res.writeHead(200, { "Content-Type": "application/json", "Content-Length": "2" });
    res.end("{}");
  });
  const foreignOrigin = await listen(foreign);
  const source = createServer((_req, res) => {
    res.writeHead(302, { Location: `${foreignOrigin}/injected-config` });
    res.end();
  });
  const sourceOrigin = await listen(source);
  try {
    await assert.rejects(
      prefetchModel({
        modelId,
        files: [file],
        deps: {
          resolveInfo: async () => [{ file, url: requested, size: 2, oid: null }],
          cacheOpen: async () => ({ put: async () => { cacheWrites++; } }),
          existsInCache: async () => false,
          simpleFetch: async () => {
            const response = await fetch(`${sourceOrigin}/resolve/main/${file}`);
            finalUrl = response.url;
            return response;
          },
        },
      }),
      /unexpected|untrusted|origin|redirect/i,
    );
    assert.equal(finalUrl, `${foreignOrigin}/injected-config`);
    assert.equal(cacheWrites, 0);
  } finally {
    await Promise.all([
      new Promise((resolve) => source.close(resolve)),
      new Promise((resolve) => foreign.close(resolve)),
    ]);
  }
});
