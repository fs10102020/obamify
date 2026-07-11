function assetRequest(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  return new Request(url, request);
}

function isDocumentResponse(request, response) {
  return (
    request.mode === "navigate" ||
    response.headers.get("content-type")?.includes("text/html") === true
  );
}

function withBrowserHeaders(request, response) {
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("X-Content-Type-Options", "nosniff");

  if (isDocumentResponse(request, response)) {
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Origin-Agent-Cluster", "?1");
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function shouldUseAppShell(request, pathname) {
  return (
    (request.method === "GET" || request.method === "HEAD") &&
    !pathname.split("/").pop()?.includes(".")
  );
}

export default {
  async fetch(request, env) {
    if (!env?.ASSETS || typeof env.ASSETS.fetch !== "function") {
      return new Response("Static asset binding is unavailable", { status: 503 });
    }

    const url = new URL(request.url);
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    let response = await env.ASSETS.fetch(assetRequest(request, pathname));

    if (response.status === 404 && shouldUseAppShell(request, pathname)) {
      response = await env.ASSETS.fetch(assetRequest(request, "/index.html"));
    }

    return withBrowserHeaders(request, response);
  },
};
