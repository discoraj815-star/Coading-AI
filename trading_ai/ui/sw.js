// Trading AI Assistant — Private PWA Safe Service Worker
// CRITICAL SECURITY & SAFETY POLICY:
// 1. NEVER cache private API responses (/api/*).
// 2. NEVER cache WebSocket connections (/ws/*).
// 3. NEVER cache broker credentials or sensitive authenticated records.
// 4. Stale/offline cached data MUST NEVER masquerade as fresh live market data.

const CACHE_NAME = 'trading-ai-shell-v1';
const STATIC_SHELL = [
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/favicon.ico'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_SHELL);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // 1. NEVER CACHE API, WEBSOCKETS, OR AUTH REQUESTS
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/') || event.request.method !== 'GET') {
    // Network-only, fail directly to protect sensitive real-time data
    event.respondWith(fetch(event.request));
    return;
  }

  // 2. App Navigation (GET /) -> Network First with safe fallback
  if (event.request.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      fetch(event.request).catch(async () => {
        const cached = await caches.match(event.request);
        if (cached) return cached;
        return new Response(
          '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Trading AI — Offline</title>' +
          '<style>body{background:#0a0d14;color:#f8fafc;font-family:sans-serif;padding:32px;text-align:center;}' +
          '.box{max-width:480px;margin:80px auto;padding:24px;background:#121824;border:1px solid #e11d48;border-radius:12px;}' +
          'h2{color:#f43f5e;margin-bottom:12px;}p{color:#94a3b8;line-height:1.6;}</style></head>' +
          '<body><div class="box"><h2>⚠️ OFFLINE MODE</h2>' +
          '<p>The server at your desktop is currently unreachable.</p>' +
          '<p><strong>Safety Invariant:</strong> Offline cached views do NOT display live market data or active trading supervision.</p>' +
          '<p>Please ensure your desktop server is running and your device is on the private network.</p>' +
          '</div></body></html>',
          { headers: { 'Content-Type': 'text/html' } }
        );
      })
    );
    return;
  }

  // 3. Static Assets -> Stale-while-revalidate
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const fetchPromise = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseToCache));
        }
        return networkResponse;
      }).catch(() => cachedResponse);

      return cachedResponse || fetchPromise;
    })
  );
});
