const CACHE_NAME = "cough-detection-v1";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.add("/"))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
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
        "You are offline. Please reconnect to use cough detection.",
        { headers: { "Content-Type": "text/plain" } }
      );
    })
  );
});
