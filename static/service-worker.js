const CACHE_NAME = "cough-detection-v2";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.add("/");
    })
  );

  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  const url = new URL(event.request.url);

  if (url.origin !== self.location.origin) return;
  if (url.pathname !== "/") return;

  event.respondWith(
    fetch(event.request).catch(async () => {
      const cached = await caches.match("/");
      return cached || new Response(
        "You are offline. Please reconnect to use Cough Detection AI.",
        {
          headers: {
            "Content-Type": "text/plain"
          }
        }
      );
    })
  );
});
