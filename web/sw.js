// eMIPS offline: this service worker keeps the site in the
// browser.  web/build_site.py writes it at the site's root with the build's
// files and version filled in.
//  - Installing caches the eMIPS page (its files, emips.zip, every program of
//    the list), the guide's pages and Pyodide's core files.
//  - The site's own files: the network first (a new release is never hidden
//    by an old cache), the cache when there is no network; whatever else of
//    the site is fetched (the guide's pictures) is kept as it comes.
//  - Pyodide (its pinned version's URL never changes): from the cache.
"use strict";

const VERSION = "@VERSION@";
const CACHE = `emips-${VERSION}`;
const PYODIDE = "@PYODIDE@";
const PYODIDE_CACHE = "emips-pyodide-@PYODIDE_VERSION@";
const PYODIDE_FILES = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm",
                       "python_stdlib.zip", "pyodide-lock.json"];
const SITE = @SITE@;

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const site = await caches.open(CACHE);
    await site.addAll(SITE.map((u) => new Request(u, { cache: "reload" })));
    const py = await caches.open(PYODIDE_CACHE);
    for (const f of PYODIDE_FILES) {
      if (!await py.match(PYODIDE + f)) await py.add(PYODIDE + f);
    }
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {        // (an older release's caches)
      if (name.startsWith("emips-") && name !== CACHE && name !== PYODIDE_CACHE) {
        await caches.delete(name);
      }
    }
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (req.url.startsWith(PYODIDE)) {
    event.respondWith((async () => {
      const py = await caches.open(PYODIDE_CACHE);
      const hit = await py.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) py.put(req, res.clone());
      return res;
    })());
    return;
  }
  if (url.origin !== self.location.origin) return;   // (the visit counter: as it is)
  event.respondWith((async () => {
    const site = await caches.open(CACHE);
    try {
      const res = await fetch(req);
      if (res.ok) site.put(req, res.clone());
      return res;
    } catch (err) {                                  // (no network: what was kept)
      const nav = req.mode === "navigate";
      const hit = await site.match(req, { ignoreSearch: nav })
        || (url.pathname.endsWith("/")              // (run/ is run/index.html)
            && await site.match(new URL("index.html", url).href, { ignoreSearch: true }));
      if (hit) return hit;
      throw err;
    }
  })());
});
