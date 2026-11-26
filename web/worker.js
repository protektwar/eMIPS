// eMIPS in the browser: the Web Worker.  Python (Pyodide) and bridge.py run
// here, off the page's thread, so the page never freezes.  The page sends
// {id, method, args}; the worker answers {id, result} (the bridge's JSON) or
// {id, error}.  "run" is a loop of short slices: after each one the worker
// posts the state ({type: "state"}) and yields, so a "pause" message (a stop
// flag: GitHub Pages sends no COOP / COEP headers, so no SharedArrayBuffer
// interrupt) stops it within one slice.
//: Pyodide, pinned (moving it is a change with its own test run)
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

const SLICE_MS = 30;           // a Run slice: about this long, then messages are read
let web = null;                // bridge.WEB
let running = null;            // the id of the run in progress (null: none)
let slice = 500;               // steps a slice (adapted to SLICE_MS)

function call(method, args) {
  return web[method](...args);
}

async function boot() {
  // (a module worker: Pyodide 0.31 refuses classic ones)
  const { loadPyodide } = await import(PYODIDE + "pyodide.mjs");
  const py = await loadPyodide({ indexURL: PYODIDE });
  const zip = await (await fetch("emips.zip")).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/emips" });
  py.runPython("import sys; sys.path.insert(0, '/emips')");
  py.runPython(await (await fetch("bridge.py")).text());
  web = py.globals.get("WEB");
  return JSON.stringify({ version: py.runPython("import emips; emips.__version__") });
}

function checkSlices(id) {     // the checks: as Run, a slice at a time
  if (running !== id) return;
  const t = performance.now();
  const st = JSON.parse(web.check_advance(slice));
  const dt = performance.now() - t;
  slice = Math.max(50, Math.min(200000, Math.round(slice * SLICE_MS / Math.max(dt, 1))));
  if (st.done) {
    running = null;
    postMessage({ id, result: JSON.stringify(st) });
  } else {
    postMessage({ type: "checks", id, state: st });
    setTimeout(() => checkSlices(id), 0);
  }
}

function runSlices(id) {
  if (running !== id) return;
  const t = performance.now();
  const st = JSON.parse(web.run(slice));
  const dt = performance.now() - t;
  slice = Math.max(50, Math.min(200000, Math.round(slice * SLICE_MS / Math.max(dt, 1))));
  const stop = st.halted || st.needs_input || (st.stopped && st.stopped !== "") || st.error;
  if (stop) {
    running = null;
    postMessage({ id, result: JSON.stringify(st) });
  } else {
    postMessage({ type: "state", id, state: st });
    setTimeout(() => runSlices(id), 0);     // (messages — a pause — are read here)
  }
}

self.onmessage = async (e) => {
  const { id, method, args = [] } = e.data;
  try {
    if (method === "boot") {
      postMessage({ id, result: await boot() });
    } else if (method === "run") {
      running = id;
      runSlices(id);
    } else if (method === "check") {
      const st = JSON.parse(web.check_start(...args));
      if (st.done) {
        postMessage({ id, result: JSON.stringify(st) });
      } else {
        running = id;
        checkSlices(id);
      }
    } else if (method === "pause") {
      const was = running;
      running = null;
      postMessage({ id, result: web.state(), paused: was });
    } else {
      postMessage({ id, result: call(method, args) });
    }
  } catch (err) {
    running = null;
    postMessage({ id, error: String(err) });
  }
};
