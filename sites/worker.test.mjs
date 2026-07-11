import assert from "node:assert/strict";
import test from "node:test";

import worker from "./worker.mjs";

function assetEnvironment(responses) {
  const requests = [];
  return {
    requests,
    env: {
      ASSETS: {
        async fetch(request) {
          const pathname = new URL(request.url).pathname;
          requests.push(pathname);
          return responses[pathname] ?? new Response("missing", { status: 404 });
        },
      },
    },
  };
}

test("serves the app shell with cross-origin isolation headers", async () => {
  const { env, requests } = assetEnvironment({
    "/index.html": new Response("<html>obamify</html>", {
      headers: { "content-type": "text/html; charset=utf-8" },
    }),
  });

  const response = await worker.fetch(new Request("https://example.test/"), env);

  assert.equal(response.status, 200);
  assert.deepEqual(requests, ["/index.html"]);
  assert.equal(response.headers.get("cross-origin-opener-policy"), "same-origin");
  assert.equal(response.headers.get("cross-origin-embedder-policy"), "require-corp");
  assert.equal(response.headers.get("origin-agent-cluster"), "?1");
});

test("serves hashed WASM assets without rewriting their paths", async () => {
  const { env, requests } = assetEnvironment({
    "/obamify-hash_bg.wasm": new Response("wasm", {
      headers: { "content-type": "application/wasm" },
    }),
  });

  const response = await worker.fetch(
    new Request("https://example.test/obamify-hash_bg.wasm"),
    env,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(requests, ["/obamify-hash_bg.wasm"]);
  assert.equal(response.headers.get("cross-origin-resource-policy"), "same-origin");
  assert.equal(response.headers.get("cross-origin-opener-policy"), null);
});

test("falls back to index.html for client-side routes", async () => {
  const { env, requests } = assetEnvironment({
    "/index.html": new Response("<html>obamify</html>", {
      headers: { "content-type": "text/html" },
    }),
  });

  const response = await worker.fetch(
    new Request("https://example.test/studio/session"),
    env,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(requests, ["/studio/session", "/index.html"]);
  assert.equal(response.headers.get("cross-origin-embedder-policy"), "require-corp");
});
