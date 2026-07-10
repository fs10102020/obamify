var CACHE_NAME = "obamify-pwa-v3"

var filesToCache = [
    "./",
    "./index.html",
    "./manifest.json",
    "./worker.js",
]

function isNavigation(request) {
    return request.mode === "navigate"
}

function isSameOrigin(url) {
    return url.origin === self.location.origin
}

// Install: precache core app shell — let errors fail the install
self.addEventListener("install", function (e) {
    e.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            return cache.addAll(filesToCache)
        })
    )
})

// Activate: delete only old obamify-pwa-* caches
self.addEventListener("activate", function (e) {
    e.waitUntil(
        caches.keys().then(function (names) {
            return Promise.all(
                names.filter(function (name) {
                    return name.startsWith("obamify-pwa-") && name !== CACHE_NAME
                }).map(function (name) {
                    return caches.delete(name)
                })
            )
        }).then(function () {
            return self.clients.claim()
        })
    )
})

// Fetch: network-first for navigations, cache-first for other same-origin GETs
self.addEventListener("fetch", function (e) {
    var url = new URL(e.request.url)

    if (e.request.method !== "GET" || !isSameOrigin(url)) {
        return
    }

    if (isNavigation(e.request)) {
        e.respondWith(networkFirst(e.request))
    } else {
        e.respondWith(cacheFirst(e.request))
    }
})

function networkFirst(request) {
    return fetch(request).then(function (response) {
        response = withIsolationHeaders(response)
        if (response && response.status === 200) {
            var clone = response.clone()
            caches.open(CACHE_NAME).then(function (cache) {
                cache.put(request, clone)
            })
        }
        return response
    }).catch(function () {
        return caches.match(request).then(function (response) {
            return response ? withIsolationHeaders(response) : response
        })
    })
}

function withIsolationHeaders(response) {
    var headers = new Headers(response.headers)
    headers.set("Cross-Origin-Opener-Policy", "same-origin")
    headers.set("Cross-Origin-Embedder-Policy", "require-corp")
    return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: headers,
    })
}

function cacheFirst(request) {
    return caches.match(request).then(function (cached) {
        if (cached) {
            return cached
        }
        return fetch(request).then(function (response) {
            if (response && response.status === 200) {
                var clone = response.clone()
                caches.open(CACHE_NAME).then(function (cache) {
                    cache.put(request, clone)
                })
            }
            return response
        })
    })
}
