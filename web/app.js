// eMIPS in the browser: the page.  Python (Pyodide) runs eMIPS itself in a Web
// Worker (worker.js); bridge.py is its side: WEB.open / step / run / reset ...
// return the state as JSON.  Every call is asynchronous: the page never waits.
"use strict";

const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), programs: $("programs"), release: $("release"), mode: $("mode"),
  assemble: $("assemble"), step: $("step"), run: $("run"), pause: $("pause"),
  back: $("back"), reset: $("reset"), messages: $("messages"), file: $("file"),
  where: $("where"), gutter: $("gutter"), current: $("current"), source: $("source"),
  listing: document.querySelector("#listing tbody"), regs: document.querySelector("#regs tbody"),
  console: $("console"), input: $("input"), keys: $("keys"), dbgout: $("dbgout"), dbg: $("dbg"),
  settingsBtn: $("settingsBtn"), settings: $("settings"), settingsBody: $("settingsBody"),
  settingsDefaults: $("settingsDefaults"), newProg: $("newProg"), openFile: $("openFile"),
  fileInput: $("fileInput"), saveProg: $("saveProg"), download: $("download"),
  deleteProg: $("deleteProg"), memSegment: $("memSegment"), memAddress: $("memAddress"),
  memPhysical: $("memPhysical"), memBytes: $("memBytes"), memDoubles: $("memDoubles"),
  memDoublesLabel: $("memDoublesLabel"), memory: $("memory"), chart: $("chart"),
  chartHint: $("chartHint"), screen: $("screen"), displayInfo: $("displayInfo"),
  fileButtons: $("fileButtons"), fileLines: $("fileLines"), noteLines: $("noteLines"),
  playWav: $("playWav"), saveWav: $("saveWav"), saveMid: $("saveMid"), playLive: $("playLive"),
  msgDialog: $("msgDialog"), msgTitle: $("msgTitle"), msgText: $("msgText"),
  askDialog: $("askDialog"), askTitle: $("askTitle"), askText: $("askText"),
  askValue: $("askValue"), askButtons: $("askButtons"), main: document.querySelector("main"),
  checkBtn: $("checkBtn"), checks: document.querySelector("#checks tbody"),
  checksHint: $("checksHint"),
};
// the window's layout: three groups of tabs (Console's, the registers', the views')
const PANE_TITLES = { special: "PC / CP0", fpu: "FPU", dsp: "DSP", msa: "MSA", vector: "Vector",
  interrupts: "Interrupts", debug: "Debug", caches: "Caches", tlb: "TLB", counters: "Counters",
  trace: "Trace", board: "Board", system: "System", mt: "MT", tools: "Tools" };
const REG_PANES = ["special", "fpu"];
const RIGHT_PANES = ["dsp", "msa", "vector", "interrupts", "debug", "caches", "tlb", "counters",
  "trace", "board", "system", "mt", "tools"];
const GROUPS = {
  left: ["console", "memory", "checks", "debugger"],
  regs: ["regs", ...REG_PANES.map((p) => `pane-${p}`)],
  right: ["datapath", "instr", "chart", "display", "files", "sound", ...RIGHT_PANES.map((p) => `pane-${p}`)],
};
const DEVICE_TABS = ["display", "files", "sound"];   // (shown when the program has one)
const current = { left: "console", regs: "regs", right: "datapath",
                  ...JSON.parse(localStorage.getItem("emips.tabs") || "{}") };
let worker = null;             // eMIPS, in worker.js
let ready = false;
let program = { name: "program.asm", path: null };
let assembled = null;          // the text last assembled (null: none)
let running = false, rows = new Map(), lineOf = new Map(), bpLines = new Set();
let programs = [];
let schema = [];               // the Settings dialog's sections (bridge.settings_schema)
let settings = JSON.parse(localStorage.getItem("emips.settings") || "{}");   // dest -> value
// the user's programs and the files programs read, kept in this browser
let mine = JSON.parse(localStorage.getItem("emips.programs") || "{}");       // name -> text
let companions = JSON.parse(localStorage.getItem("emips.files") || "{}");   // name -> text
let loadedText = "";           // the editor's text as opened / saved (the unsaved mark)
const MINE = "my:";            // the program list's value for one of the user's programs

// -- talking to the worker -------------------------------------------------------------

let nextId = 1;
const pending = new Map();     // id -> {resolve, reject}

function call(method, ...args) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, method, args });
  });
}

function answered(m) {
  if (m.type === "state") {                  // a Run slice's state: shown, the run goes on
    render(m.state, true);
    return;
  }
  if (m.type === "checks") {                 // a check in progress
    setStatus(`checking… run ${m.state.run} of ${m.state.runs}, ${m.state.steps} steps`);
    fillChecks(m.state);
    return;
  }
  for (const id of [m.id, m.paused]) {       // (a pause also ends the run it stopped)
    const p = pending.get(id);
    if (!p) continue;
    pending.delete(id);
    if (m.error) p.reject(new Error(m.error));
    else p.resolve(m.result === undefined || m.result === null ? null : JSON.parse(m.result));
  }
}

function setStatus(text) { ui.status.textContent = text; }

function enable(on) {
  for (const b of [ui.assemble, ui.step, ui.run, ui.back, ui.reset]) b.disabled = !on || running;
  ui.checkBtn.disabled = !ready || running;
  ui.settingsBtn.disabled = !ready || running;
  for (const b of [ui.newProg, ui.openFile, ui.saveProg, ui.download]) b.disabled = !ready || running;
  ui.deleteProg.hidden = !(program.path || "").startsWith(MINE);
  ui.deleteProg.disabled = !ready || running;
  ui.pause.disabled = !running;
  ui.programs.disabled = running;
  ui.dbg.disabled = !on || running;
}

async function act(method, ...args) {       // a button's call, its state shown
  if (!ready) return;
  render(await call(method, ...args));
}

// -- loading -----------------------------------------------------------------------

async function boot() {
  const t0 = performance.now();
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (e) => answered(e.data);
  worker.onerror = (e) => setStatus("could not start: " + e.message);
  const [{ version }, list] = await Promise.all([
    call("boot"), fetch("programs.json").then((r) => r.json())]);
  schema = await call("settings_schema");
  timbres = await call("timbres");
  for (const [i, name] of (await call("memory_segments")).entries()) ui.memSegment.add(new Option(name, i));
  for (const g of Object.keys(GROUPS)) showGroup(g);
  await showTab(current.right, false);
  ready = true;
  markSettings();
  programs = list;
  fillPrograms();
  setStatus(`eMIPS ${version} ready (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  for (const [name, text] of Object.entries(companions)) await call("save", name, text);
  const wanted = new URLSearchParams(location.search).get("open")
    || localStorage.getItem("emips.last");
  const gone = wanted && wanted.startsWith(MINE) && mine[wanted.slice(MINE.length)] === undefined;
  await openProgram(wanted && !gone ? wanted : programs[0].items[0].path);
  document.body.dataset.ready = "1";        // (booted, the program open: tests wait for it)
}

function fillPrograms() {
  ui.programs.innerHTML = "";
  const names = Object.keys(mine).sort();
  if (names.length) {
    const og = document.createElement("optgroup");
    og.label = "My programs (in this browser)";
    for (const n of names) og.appendChild(new Option(n, MINE + n));
    ui.programs.appendChild(og);
  }
  for (const g of programs) {
    const og = document.createElement("optgroup");
    og.label = g.group;
    for (const item of g.items) {
      const o = document.createElement("option");
      o.value = item.path;
      o.textContent = item.title;
      og.appendChild(o);
    }
    ui.programs.appendChild(og);
  }
  ui.programs.disabled = false;
}

function find(path) {
  for (const g of programs) for (const item of g.items) if (item.path === path) return item;
  return null;
}

async function openProgram(path) {
  if (!confirmDiscard()) { ui.programs.value = program.path || ""; return; }
  const item = find(path) || { path, files: [], title: path };
  const own = path.startsWith(MINE);
  const text = own ? mine[path.slice(MINE.length)] ?? null : await fetchText(path);
  if (text === null) { showMessages([`could not load ${path}`], false); return; }
  for (const f of item.files || []) {
    const companion = await fetchText(f);
    if (companion !== null) await call("save", f.split("/").pop(), companion);
  }
  program = { name: own ? path.slice(MINE.length) : path.split("/").pop(), path };
  diskHandle = own ? await handleGet(program.name) || null : null;
  ui.programs.value = path;
  ui.source.value = text;
  loadedText = text;
  markUnsaved();
  ui.console.textContent = "";
  resetDevices();
  // (what the program reads, ready in the boxes: Enter sends it)
  ui.input.value = item.input || "";
  ui.keys.value = (item.keys || "").replace(/\\n$/, "");
  updateGutter();
  history.replaceState(null, "", `?open=${encodeURIComponent(path)}`);
  localStorage.setItem("emips.last", path);
  enable(ready);
  countEvent(own ? "open: my program" : `open: ${path}`);
  await assemble();
  document.body.dataset.program = path;     // (opened and assembled: tests wait for it)
}

// -- the visit counter's events (GoatCounter, when the site has it: build_site) ----------

function countEvent(name) {
  // (no cookies, no personal data: a program of the site by its path, the
  //  user's own only as "my program"; nothing without the counter's script)
  const gc = window.goatcounter;
  if (!gc || typeof gc.count !== "function") return;
  try { gc.count({ path: name, title: name, event: true }); } catch { /* never in the way */ }
}

// -- the user's programs and files -------------------------------------------------------

function markUnsaved() {
  const dirty = ui.source.value !== loadedText;
  ui.file.textContent = program.name + (dirty ? " •" : "");
  ui.file.title = (dirty ? "not saved (Save keeps it in this browser)" : "")
    + (diskHandle ? `${dirty ? "; " : ""}on your computer: ${diskHandle.name}` : "");
}

function confirmDiscard() {
  return ui.source.value === loadedText || !program.path
    || confirm(`${program.name} has changes that are not saved. Leave them?`);
}

function keepMine() {
  localStorage.setItem("emips.programs", JSON.stringify(mine));
}

function uniqueName(base) {
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base, ext = dot > 0 ? base.slice(dot) : ".asm";
  let name = base, n = 2;
  while (mine[name] !== undefined) name = `${stem}${n++}${ext}`;
  return name;
}

async function saveProgram() {
  const own = (program.path || "").startsWith(MINE);
  const name = own ? program.name : uniqueName(program.name);
  mine[name] = ui.source.value;
  keepMine();
  fillPrograms();
  program = { name, path: MINE + name };
  ui.programs.value = program.path;
  loadedText = ui.source.value;
  markUnsaved();
  enable(ready);
  history.replaceState(null, "", `?open=${encodeURIComponent(program.path)}`);
  localStorage.setItem("emips.last", program.path);
  let disk = "";
  if (own && diskHandle) {
    try {
      await writeDisk(diskHandle, ui.source.value);
      disk = ` and in ${diskHandle.name} on your computer`;
    } catch (e) {
      disk = ` (not on your computer: ${e.message})`;
    }
  }
  showMessages([own ? `saved ${name} in this browser${disk}`
                    : `saved as My programs / ${name} (the original is unchanged)`]);
}

const NEW_PROGRAM = `# My program.  Check (F8) runs it and tests the "# check:" lines.
# check: output == "Hello!\\n"
# check: $v0 == 10

        .data
msg:    .asciiz "Hello!\\n"

        .text
main:   li    $v0, 4              # print_string
        la    $a0, msg
        syscall
        li    $v0, 10             # exit
        syscall
`;

async function newProgram() {
  if (!confirmDiscard()) return;
  const name = uniqueName("program.asm");
  mine[name] = NEW_PROGRAM;
  keepMine();
  fillPrograms();
  loadedText = ui.source.value;            // (nothing to lose: just confirmed)
  await openProgram(MINE + name);
}

// -- files on the computer: a program's file, written again by Save -----------------------
// (the File System Access API where the browser has it — Chrome, Edge —; the
// handles kept in IndexedDB, so Save still writes the file after a reload;
// elsewhere Save to disk downloads)

const DISK_TYPES = [{ description: "Assembly programs", accept: { "text/plain": [".asm", ".s", ".S"] } }];
let diskHandle = null;          // the open program's file on the computer (or null)

function idb(mode, work) {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open("emips", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("handles");
    r.onerror = () => reject(r.error);
    r.onsuccess = () => {
      const tx = r.result.transaction("handles", mode);
      const req = work(tx.objectStore("handles"));
      tx.oncomplete = () => resolve(req && req.result);
      tx.onerror = () => reject(tx.error);
    };
  });
}
const handleGet = (name) => idb("readonly", (st) => st.get(name)).catch(() => null);
const handleSet = (name, h) => idb("readwrite", (st) => st.put(h, name)).catch(() => null);
const handleDel = (name) => idb("readwrite", (st) => st.delete(name)).catch(() => null);

async function writeDisk(handle, text) {
  if (handle.queryPermission && await handle.queryPermission({ mode: "readwrite" }) !== "granted"
      && await handle.requestPermission({ mode: "readwrite" }) !== "granted") {
    throw new Error("the browser was not allowed to write the file");
  }
  const w = await handle.createWritable();
  await w.write(text);
  await w.close();
}

function downloadText(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function saveToDisk() {
  if (!window.showSaveFilePicker) {        // (no file access: the browser's Downloads)
    downloadText(program.name, ui.source.value);
    showMessages([`downloaded ${program.name} (your browser puts it in its Downloads `
                  + "folder; Open… reads it back)"]);
    return;
  }
  let handle;
  try {
    handle = await window.showSaveFilePicker({ suggestedName: program.name, types: DISK_TYPES });
  } catch (e) {
    if (e.name !== "AbortError") showMessages([`could not save: ${e.message}`], false);
    return;
  }
  try {
    await writeDisk(handle, ui.source.value);
  } catch (e) {
    showMessages([`could not save ${handle.name}: ${e.message}`], false);
    return;
  }
  if (!(program.path || "").startsWith(MINE) || program.name !== handle.name) {
    // (kept in the browser as one of mine too, under the file's name)
    const name = mine[handle.name] !== undefined && program.name !== handle.name
      ? uniqueName(handle.name) : handle.name;
    if ((program.path || "").startsWith(MINE) && program.name !== name) delete mine[program.name];
    mine[name] = ui.source.value;
    keepMine();
    fillPrograms();
    program = { name, path: MINE + name };
    ui.programs.value = program.path;
    history.replaceState(null, "", `?open=${encodeURIComponent(program.path)}`);
    localStorage.setItem("emips.last", program.path);
  } else {
    mine[program.name] = ui.source.value;
    keepMine();
  }
  diskHandle = handle;
  await handleSet(program.name, handle);
  loadedText = ui.source.value;
  markUnsaved();
  enable(ready);
  showMessages([`saved ${handle.name} on your computer (Save writes it again)`]);
}

async function deleteProgram() {
  if (!program.path.startsWith(MINE) || !confirm(`Delete ${program.name} from this browser?`)) return;
  delete mine[program.name];
  await handleDel(program.name);            // (the file on the computer stays)
  keepMine();
  fillPrograms();
  loadedText = ui.source.value;            // (deleted on purpose)
  await openProgram(programs[0].items[0].path);
}

const PROGRAM_FILES = /\.(asm|s|S|hex)$/;

async function openFromDisk() {
  if (!window.showOpenFilePicker) { ui.fileInput.click(); return; }
  let handle;
  try {
    [handle] = await window.showOpenFilePicker();
  } catch (e) {
    if (e.name !== "AbortError") showMessages([`could not open: ${e.message}`], false);
    return;
  }
  await openLocalFile(await handle.getFile(), handle);
}

async function openLocalFile(file, handle = null) {
  const text = await file.text();
  if (PROGRAM_FILES.test(file.name)) {      // a program: one of mine
    if (!confirmDiscard()) return;
    // (the same file again: the same program; another one of that name: a copy)
    const same = handle && mine[file.name] !== undefined
      && await handleGet(file.name).then((h) => h && h.isSameEntry && h.isSameEntry(handle));
    const name = mine[file.name] === undefined || same ? file.name : uniqueName(file.name);
    mine[name] = text;
    keepMine();
    fillPrograms();
    if (handle) await handleSet(name, handle);
    loadedText = ui.source.value;
    await openProgram(MINE + name);
  } else {                                  // a file a program reads (a disk, a --file)
    companions[file.name] = text;
    localStorage.setItem("emips.files", JSON.stringify(companions));
    await call("save", file.name, text);
    showMessages([`${file.name}: kept for programs to read (--file ${file.name}, `
                  + `--disk ${file.name}); assemble again to use it`]);
  }
}

async function fetchText(path) {          // paths are relative to the site's root
  try {
    const r = await fetch(`../${path}`);
    return r.ok ? await r.text() : null;
  } catch { return null; }
}

function quote(v) {                         // (a shell word: the bridge splits with shlex)
  return /^[\w.:@,+=\/-]+$/.test(v) ? v : `'${v.replace(/'/g, "'\\''")}'`;
}

function settingsFlags(values = settings) {
  const out = [];
  for (const sec of schema) for (const f of sec.fields) {
    const v = values[f.dest];
    if (f.kind === "bool") { if (v) out.push(f.flag); }
    else if (v !== undefined && v !== "") out.push(`${f.flag} ${quote(String(v))}`);
  }
  return out.join(" ");
}

function flags() {
  return [settingsFlags(), ui.release.value ? `--release ${ui.release.value}` : "",
          ui.mode.value].filter(Boolean).join(" ");
}

// -- the views' tabs: Memory, Pipeline, Debugger ---------------------------------------

function viewSpec() {
  const spec = {};
  if (current.left === "memory") {
    spec.memory = { segment: Number(ui.memSegment.value || 0), address: ui.memAddress.value,
                    physical: ui.memPhysical.checked, bytes: ui.memBytes.checked,
                    doubles: ui.memDoubles.checked };
  }
  if (current.right === "chart") spec.chart = true;
  if (current.right === "instr") spec.instr = true;
  if (current.right === "datapath") spec.datapath = true;
  spec.panes = {};
  for (const g of ["regs", "right"]) {
    if (current[g].startsWith("pane-")) {
      const name = current[g].slice(5);
      spec.panes[name] = paneOpts[name] || {};
    }
  }
  return spec;
}

function groupOf(name) {
  return Object.keys(GROUPS).find((g) => GROUPS[g].includes(name));
}

function showGroup(g) {                    // (the group's buttons and panels)
  for (const t of GROUPS[g]) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === current[g]));
    $(`panel-${t}`).hidden = t !== current[g];
  }
}

async function showTab(name, refresh = true) {
  const g = groupOf(name);
  if (!g) return;
  current[g] = $(`tab-${name}`).hidden ? GROUPS[g][0] : name;
  localStorage.setItem("emips.tabs", JSON.stringify(current));
  showGroup(g);
  if (current.right === "datapath" && !dp) await buildDatapath();
  if (current.right === "datapath" && !dp24) await build24k();
  const st = await call("set_views", JSON.stringify(viewSpec()));
  if (refresh && !running && assembled !== null) render(st);
}

function showPaneTabs(avail) {             // (DSP / MSA / Vector: a core that has them)
  for (const name of [...REG_PANES, ...RIGHT_PANES]) {
    $(`tab-pane-${name}`).hidden = !avail.includes(name);
  }
  for (const g of ["regs", "right"]) {
    if ($(`tab-${current[g]}`).hidden) { current[g] = GROUPS[g][0]; showGroup(g); }
  }
}

function buildPaneTabs() {
  const add = (name, bar, before, pane) => {
    const b = document.createElement("button");
    b.setAttribute("role", "tab");
    b.id = `tab-pane-${name}`;
    b.setAttribute("aria-controls", `panel-pane-${name}`);
    b.setAttribute("aria-selected", "false");
    b.textContent = PANE_TITLES[name];
    b.addEventListener("click", () => showTab(`pane-${name}`));
    bar.insertBefore(b, before);
    const p = document.createElement("div");
    p.className = "panel";
    p.id = `panel-pane-${name}`;
    p.setAttribute("role", "tabpanel");
    p.hidden = true;
    p.innerHTML = '<div class="pane-bar"></div><div class="scroll pane-body"></div>';
    pane.appendChild(p);
  };
  const regs = document.querySelector(".regs-pane");
  for (const name of REG_PANES) add(name, regs.querySelector(".tabs"), null, regs);
  const right = document.querySelector(".tabs-pane");
  for (const name of RIGHT_PANES) add(name, $("rightTabs"), $("enlarge"), right);
}

function fillMemory(mem) {
  const head = ui.memory.tHead, body = ui.memory.tBodies[0];
  if (mem.error) { head.innerHTML = ""; body.innerHTML = `<tr><td>${mem.error}</td></tr>`; return; }
  ui.memDoublesLabel.hidden = !mem.wide || ui.memBytes.checked;
  head.innerHTML = "<tr><th>address</th>" + mem.heads.map((h) => `<th>${h}</th>`).join("") + "</tr>";
  body.innerHTML = mem.rows.map(([addr, cells]) =>
    `<tr><td>${addr}</td>${cells.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("");
}

// -- the 5-stage datapath (emips.datapathdraw: the window's drawing, in SVG) -------------

let dp = null;                 // the SVG's elements by key (built once)
const SVGNS = "http://www.w3.org/2000/svg";
const PT = 1.33;               // (Qt's font points at 96 dpi, in the drawing's pixels)

function svgEl(tag, attrs, parent, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) setLines(e, text);
  if (parent) parent.appendChild(e);
  return e;
}

function setLines(e, text) {   // (a multi-line text: tspans one under the other)
  if (e.dataset.text === text) return;
  e.dataset.text = text;
  e.textContent = "";
  const x = e.getAttribute("x");
  String(text).split("\n").forEach((line, i) => {
    const t = document.createElementNS(SVGNS, "tspan");
    t.setAttribute("x", x);
    if (i) t.setAttribute("dy", "1.2em");
    t.textContent = line;
    e.appendChild(t);
  });
}

function tokenColour(tok) {
  if (!tok) return null;
  const [kind, st] = tok.split(":");
  if (kind === "stage") return `var(--dp-stage-${st})`;
  if (kind === "band") return `var(--dp-band-${st})`;
  return `var(--dp-${tok.replace("_", "-")})`;
}

async function buildDatapath() {
  const L = await call("datapath_layout");
  const svg = svgEl("svg", { viewBox: `0 0 ${L.W} ${L.H}`, preserveAspectRatio: "xMidYMid meet" });
  const g = (name) => svgEl("g", { class: name }, svg);
  const bands = g("bands"), bars = g("bars"), wires = g("wires"), parts = g("parts"),
    texts = g("texts"), values = g("values"), badges = g("badges");
  dp = { svg, parts: {}, wires: {}, values: {}, titles: {}, bands: {}, ports: {}, wiresG: wires,
         badges };
  for (const [st, [x0, x1]] of Object.entries(L.stages)) {
    dp.bands[st] = svgEl("rect", { x: x0, y: 0, width: x1 - x0, height: L.H - 40,
                                   fill: "var(--dp-bubble)" }, bands);
    svgEl("text", { x: x0 + 8, y: 6, "dominant-baseline": "hanging", "font-size": 13 * PT,
                    "font-weight": "bold", fill: `var(--dp-stage-${st})` }, texts, st);
    dp.titles[st] = svgEl("text", { x: x0 + 8, y: 30, "dominant-baseline": "hanging",
                                    "font-size": 10 * PT, "font-weight": "bold",
                                    fill: "var(--dp-text)" }, texts, "");
  }
  for (const [name, x] of Object.entries(L.bars)) {
    const r = svgEl("rect", { x, y: 60, width: 12, height: 680, fill: "var(--dp-bar-fill)",
                              stroke: "var(--dp-bar-stroke)" }, bars);
    svgEl("title", {}, r).textContent = `${name} pipeline register: holds what the next `
      + "stage needs, latched at the end of every cycle";
    svgEl("text", { x: x - 18, y: 744, "dominant-baseline": "hanging", "font-size": 8 * PT,
                    fill: "var(--dp-muted)" }, texts, name);
  }
  for (const [key, [pts, tip]] of Object.entries(L.wires)) {
    const w = svgEl("g", {}, wires);
    const line = svgEl("polyline", { points: pts.map((p) => p.join(",")).join(" "),
                                     fill: "none" }, w);
    const [[x0, y0], [x1, y1]] = pts.slice(-2);
    let tri;
    if (x1 !== x0) { const d = x1 > x0 ? 1 : -1; tri = [[x1, y1], [x1 - 7 * d, y1 - 4], [x1 - 7 * d, y1 + 4]]; }
    else { const d = y1 > y0 ? 1 : -1; tri = [[x1, y1], [x1 - 4, y1 - 7 * d], [x1 + 4, y1 - 7 * d]]; }
    const head = svgEl("polygon", { points: tri.map((p) => p.join(",")).join(" ") }, w);
    if (tip) svgEl("title", {}, w).textContent = tip;
    dp.wires[key] = { g: w, line, head };
  }
  for (const [key, [kind, x, y, w, h, label, tip]] of Object.entries(L.parts)) {
    const grp = svgEl("g", { class: "part", "data-key": key }, parts);
    let shape;
    if (kind === "ellipse") shape = svgEl("ellipse", { cx: x + w / 2, cy: y + h / 2, rx: w / 2, ry: h / 2 }, grp);
    else if (kind === "alu") {
      const P = [[0, 0], [1, 0.3], [1, 0.7], [0, 1], [0, 0.6], [0.3, 0.5], [0, 0.4]];
      shape = svgEl("polygon", { points: P.map(([a, b]) => `${x + w * a},${y + h * b}`).join(" ") }, grp);
    } else shape = svgEl("rect", { x, y, width: w, height: h, rx: kind === "mux" ? w / 2 : 0 }, grp);
    const lines = label.split("\n").length, size = (kind === "mux" ? 8 : 9) * PT;
    const text = svgEl("text", { x: x + w / 2, y: y + h / 2 - (lines - 1) * size * 0.6,
                                 "text-anchor": "middle", "dominant-baseline": "central",
                                 "font-size": size, "font-weight": kind === "mux" ? "normal" : "bold" },
                       grp, label);
    const title = svgEl("title", {}, grp);
    title.textContent = tip;
    grp.addEventListener("click", () => showPart(key));
    dp.parts[key] = { g: grp, shape, text, title, h, y, size };
  }
  for (const [name, [x, y]] of Object.entries(L.ports)) {
    dp.ports[name] = svgEl("text", { x, y, "dominant-baseline": "hanging", "font-size": 7 * PT,
                                     fill: "var(--dp-idle-text)" }, texts, name);
  }
  for (const [key, [x, y, size]] of Object.entries(L.values)) {
    const grp = svgEl("g", {}, values);
    const bg = svgEl("rect", { x: x - 2, y: y - 1, width: 0, height: 0, fill: "var(--dp-value-bg)" }, grp);
    const text = svgEl("text", { x, y, "dominant-baseline": "hanging", "font-size": size * PT }, grp, "");
    dp.values[key] = { bg, text };
  }
  for (const [key, [number, title]] of Object.entries(L.numbers)) {
    const [, x0, y0] = L.parts[key];
    const grp = svgEl("g", { class: "part" }, badges);
    svgEl("circle", { cx: x0, cy: y0, r: 11, fill: "var(--dp-red-fill)", stroke: "var(--dp-red)",
                      "stroke-width": 1.5 }, grp);
    svgEl("text", { x: x0, y: y0, "text-anchor": "middle", "dominant-baseline": "central",
                    "font-size": 9 * PT, "font-weight": "bold", fill: "var(--dp-badge-text)" },
          grp, String(number));
    svgEl("title", {}, grp).textContent = `${number}: ${title} (click: what happens inside)`;
    grp.addEventListener("click", () => showPart(key));
  }
  badges.style.display = $("dpNumbers").checked ? "" : "none";
  let x = 1000;                                 // the legend
  for (const st of ["IF", "ID", "EX", "MEM", "WB"]) {
    svgEl("rect", { x, y: 768, width: 14, height: 14, fill: `var(--dp-band-${st})`,
                    stroke: `var(--dp-stage-${st})` }, texts);
    svgEl("text", { x: x + 18, y: 767, "dominant-baseline": "hanging", "font-size": 8 * PT,
                    fill: "var(--dp-text)" }, texts, st);
    x += 62;
  }
  svgEl("text", { x, y: 767, "dominant-baseline": "hanging", "font-size": 8 * PT,
                  fill: "var(--dp-red)" }, texts, "red: forwarding / stall");
  $("datapath").innerHTML = "";
  $("datapath").appendChild(svg);
}

// (the pipeline model's own view: the 5-stage datapath, the 24K's nine
//  stages, the P8700's window — as the window's Datapath dock switches)
function showModelView(d) {
  if (d.error) { $("datapath").dataset.error = d.error; return; }
  const model = d.model || "5-stage";
  if (dp) dp.svg.style.display = model === "5-stage" ? "" : "none";
  if (dp24) dp24.svg.style.display = model === "24k" ? "" : "none";
  $("core8700").hidden = model !== "p8700";
  if (model === "5-stage" && dp) applyDatapath(d);
  else if (model === "24k" && dp24) apply24k(d.view);
  else if (model === "p8700" && d.view) fillP8700(d.view);
}

let dp24 = null;

async function build24k() {
  const L = await call("core_layout");
  const svg = svgEl("svg", { viewBox: `0 0 ${L.W} ${L.H}`, preserveAspectRatio: "xMidYMid meet" });
  svg.style.display = "none";
  const g = (name) => svgEl("g", { class: name }, svg);
  const bands = g("bands"), wires = g("wires"), parts = g("parts"), texts = g("texts"),
    values = g("values"), regions = g("regions"), badges = g("badges");
  dp24 = { svg, bands: {}, titles: {}, parts: {}, wires: {}, values: {}, ib: [], blocks: {},
           badges };
  for (const [st, x] of Object.entries(L.stages)) {
    dp24.bands[st] = svgEl("rect", { x, y: 0, width: L.col - 8, height: 290, fill: "var(--dp-bubble)" }, bands);
    svgEl("text", { x: x + 6, y: 6, "dominant-baseline": "hanging", "font-size": 13 * PT,
                    "font-weight": "bold", fill: `var(--dp-stage-${st})` }, texts, st);
    dp24.titles[st] = svgEl("text", { x: x + 6, y: 30, "dominant-baseline": "hanging",
                                      "font-size": 8 * PT, "font-weight": "bold",
                                      fill: "var(--dp-text)" }, texts, "");
    const [px, py, w, h, label, tip] = L.parts[st];
    const pg = svgEl("g", {}, parts);
    const shape = svgEl("rect", { x: px, y: py, width: w, height: h }, pg);
    const lines = label.split("\n").length, size = 9 * PT;
    const text = svgEl("text", { x: px + w / 2, y: py + h / 2 - (lines - 1) * size * 0.6,
                                 "text-anchor": "middle", "dominant-baseline": "central",
                                 "font-size": size, "font-weight": "bold" }, pg, label);
    svgEl("title", {}, pg).textContent = tip;
    dp24.parts[st] = { shape, text };
  }
  for (const [key, [pts, tip]] of Object.entries(L.wires)) {
    const w = svgEl("g", {}, wires);
    const line = svgEl("polyline", { points: pts.map((q) => q.join(",")).join(" "), fill: "none" }, w);
    const [[x0, y0], [x1, y1]] = pts.slice(-2);
    let tri;
    if (x1 !== x0) { const d = x1 > x0 ? 1 : -1; tri = [[x1, y1], [x1 - 7 * d, y1 - 4], [x1 - 7 * d, y1 + 4]]; }
    else { const d = y1 > y0 ? 1 : -1; tri = [[x1, y1], [x1 - 4, y1 - 7 * d], [x1 + 4, y1 - 7 * d]]; }
    const head = svgEl("polygon", { points: tri.map((q) => q.join(",")).join(" ") }, w);
    if (tip) svgEl("title", {}, w).textContent = tip;
    dp24.wires[key] = { line, head };
  }
  for (const [key, [x, y, size]] of Object.entries(L.values)) {
    const grp = svgEl("g", {}, values);
    const bg = svgEl("rect", { x: x - 2, y: y - 1, width: 0, height: 0, fill: "var(--dp-value-bg)" }, grp);
    dp24.values[key] = { bg, text: svgEl("text", { x, y, "dominant-baseline": "hanging",
                                                   "font-size": size * PT }, grp, "") };
  }
  for (const [x, y, w, h] of L.ib) {
    const box = svgEl("rect", { x, y, width: w, height: h, stroke: "var(--dp-idle-line)",
                                fill: "var(--dp-part-fill)" }, parts);
    const text = svgEl("text", { x: x + 6, y: y + 3, "dominant-baseline": "hanging",
                                 "font-size": 8 * PT, fill: "var(--dp-text)" }, texts, "");
    dp24.ib.push({ box, text });
  }
  for (const [, [x, y, size, bold, col, text]] of Object.entries(L.texts)) {
    svgEl("text", { x, y, "dominant-baseline": "hanging", "font-size": size * PT,
                    "font-weight": bold ? "bold" : "normal",
                    fill: tokenColour(col) || "var(--dp-text)" }, texts, text);
  }
  for (const [key, [x, y, size, bold]] of Object.entries(L.blocks)) {
    dp24.blocks[key] = svgEl("text", { x, y, "dominant-baseline": "hanging", "font-size": size * PT,
                                       "font-weight": bold ? "bold" : "normal" }, texts, "");
  }
  for (const [key, [bx, by, rx, ry, rw, rh]] of Object.entries(L.regions)) {
    const [number, title] = L.numbers[key];
    const hit = svgEl("rect", { x: rx, y: ry, width: rw, height: rh, fill: "transparent",
                                class: "part", "data-key": key }, regions);
    svgEl("title", {}, hit).textContent = `${number}: ${title} (click: what happens inside)`;
    hit.addEventListener("click", () => showPart(key));
    const grp = svgEl("g", { class: "part" }, badges);
    svgEl("circle", { cx: bx, cy: by, r: 11, fill: "var(--dp-red-fill)", stroke: "var(--dp-red)",
                      "stroke-width": 1.5 }, grp);
    svgEl("text", { x: bx, y: by, "text-anchor": "middle", "dominant-baseline": "central",
                    "font-size": 9 * PT, "font-weight": "bold", fill: "var(--dp-badge-text)" },
          grp, String(number));
    grp.addEventListener("click", () => showPart(key));
  }
  badges.style.display = $("dpNumbers").checked ? "" : "none";
  $("datapath").appendChild(svg);
}

function apply24k(d) {
  for (const [st, band] of Object.entries(d.bands)) dp24.bands[st].setAttribute("fill", tokenColour(band));
  for (const [st, title] of Object.entries(d.titles)) setLines(dp24.titles[st], title);
  for (const [st, [color, fill]] of Object.entries(d.parts)) {
    const p = dp24.parts[st];
    p.shape.setAttribute("stroke", tokenColour(color) || "var(--dp-idle-line)");
    p.shape.setAttribute("stroke-width", color ? 2.4 : 1.2);
    p.shape.setAttribute("fill", tokenColour(fill) || "var(--dp-part-fill)");
    p.text.setAttribute("fill", color ? "var(--dp-text)" : "var(--dp-idle-text)");
  }
  for (const [key, [color, width]] of Object.entries(d.wires)) {
    const w = dp24.wires[key], c = tokenColour(color) || "var(--dp-idle-line)";
    w.line.setAttribute("stroke", c);
    w.line.setAttribute("stroke-width", color ? width || 2.2 : 1.1);
    w.head.setAttribute("fill", c);
    w.head.setAttribute("stroke", c);
  }
  for (const [key, [text, color]] of Object.entries(d.values)) {
    const v = dp24.values[key];
    setLines(v.text, text);
    v.text.setAttribute("fill", tokenColour(color) || "var(--dp-text)");
    if (text) {
      const b = v.text.getBBox();
      v.bg.setAttribute("width", b.width + 4);
      v.bg.setAttribute("height", b.height + 2);
      v.bg.style.display = "";
    } else v.bg.style.display = "none";
  }
  d.ib.forEach(([text, fill], i) => {
    setLines(dp24.ib[i].text, text);
    dp24.ib[i].box.setAttribute("fill", tokenColour(fill) || "var(--dp-part-fill)");
  });
  for (const [key, [text, color]] of Object.entries(d.blocks)) {
    setLines(dp24.blocks[key], text);
    dp24.blocks[key].setAttribute("fill", tokenColour(color) || "var(--dp-text)");
  }
}

function fillP8700(v) {
  $("core8700Title").textContent = v.title;
  const t = $("core8700Table");
  t.tHead.innerHTML = "<tr>" + v.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr>";
  t.tBodies[0].innerHTML = v.rows.map(([cells, band]) => {
    const bg = band === "error" ? "var(--dp-red-fill)" : tokenColour(band);
    return `<tr style="background: ${bg}">` + cells.map((c) => `<td>${escapeHtml(c)}</td>`).join("") + "</tr>";
  }).join("");
  $("core8700State").textContent = v.state;
}

function applyDatapath(d) {
  if (d.error) { $("datapath").dataset.error = d.error; return; }
  for (const [key, [color, fill, label, tip, shown]] of Object.entries(d.parts)) {
    const p = dp.parts[key];
    p.g.style.display = shown ? "" : "none";
    p.shape.setAttribute("stroke", tokenColour(color) || "var(--dp-idle-line)");
    p.shape.setAttribute("stroke-width", color ? 2.4 : 1.2);
    p.shape.setAttribute("fill", tokenColour(fill) || "var(--dp-part-fill)");
    p.text.setAttribute("fill", color ? "var(--dp-text)" : "var(--dp-idle-text)");
    if (p.text.dataset.text !== label) {
      const lines = label.split("\n").length;
      p.text.setAttribute("y", p.y + p.h / 2 - (lines - 1) * p.size * 0.6);
      setLines(p.text, label);
    }
    p.title.textContent = tip;
  }
  const lit = [], red = [];
  for (const [key, [color, shown]] of Object.entries(d.wires)) {
    const w = dp.wires[key];
    w.g.style.display = shown ? "" : "none";
    const c = tokenColour(color) || "var(--dp-idle-line)";
    w.line.setAttribute("stroke", c);
    w.line.setAttribute("stroke-width", color ? 2.2 : 1.1);
    w.head.setAttribute("fill", c);
    w.head.setAttribute("stroke", c);
    if (color === "red") red.push(w.g); else if (color) lit.push(w.g);
  }
  for (const g of [...lit, ...red]) dp.wiresG.appendChild(g);    // (lit wires on top)
  for (const [key, [text, color]] of Object.entries(d.values)) {
    const v = dp.values[key];
    setLines(v.text, text);
    v.text.setAttribute("fill", tokenColour(color) || "var(--dp-text)");
    if (text) {
      const b = v.text.getBBox();
      v.bg.setAttribute("width", b.width + 4);
      v.bg.setAttribute("height", b.height + 2);
      v.bg.style.display = "";
    } else {
      v.bg.style.display = "none";
    }
  }
  for (const [st, title] of Object.entries(d.titles)) setLines(dp.titles[st], title);
  for (const [st, band] of Object.entries(d.bands)) dp.bands[st].setAttribute("fill", tokenColour(band));
  for (const [k, name] of Object.entries(d.ports)) setLines(dp.ports[k], name);
}

let partKey = null, partTab = 0;

async function showPart(key) {
  if (running) return;
  if (partKey !== key) { partTab = 0; $("partBit").value = 0; }
  partKey = key;
  const info = await call("part_info", key, Number($("partBit").value) || 0);
  $("partTitle").textContent = info.heading;
  $("partWhere").textContent = info.instruction ? `In ${info.stage} now: ${info.instruction}`
    : `${info.stage} holds a bubble`;
  $("partNow").innerHTML = info.now.map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  $("partText").innerHTML = info.html;        // (emips.partlogic's: escaped there)
  $("partInside").textContent = info.inside;
  $("partBitLabel").hidden = !info.width;
  $("partBit").max = Math.max(0, info.width - 1);
  $("partGates").hidden = !info.drawings.length;
  const tabs = $("partTabs");
  tabs.innerHTML = "";
  partTab = Math.min(partTab, Math.max(0, info.drawings.length - 1));
  info.drawings.forEach(([name], i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.textContent = name;
    b.setAttribute("aria-selected", String(i === partTab));
    b.addEventListener("click", () => { partTab = i; showPart(key); });
    tabs.appendChild(b);
  });
  if (!$("partDialog").open) $("partDialog").showModal();
  const box = $("partDrawing");
  box.innerHTML = "";
  if (info.drawings.length) box.appendChild(sceneSvg(info.drawings[partTab][1]));
}

const SCENE_COLOURS = { idle_line: "var(--dp-idle-line)", idle_text: "var(--dp-idle-text)",
  text: "var(--dp-text)", muted: "var(--dp-muted)", part_fill: "var(--dp-part-fill)",
  bar_stroke: "var(--dp-bar-stroke)" };

function gatePath(kind, x, y, w, h) {
  if (kind === "and") return `M${x} ${y} L${x + w / 2} ${y} A${w / 2} ${h / 2} 0 0 1 ${x + w / 2} ${y + h} L${x} ${y + h} Z`;
  if (kind === "or" || kind === "xor") {
    const body = `M${x} ${y} Q${x + w * 0.6} ${y} ${x + w} ${y + h / 2} Q${x + w * 0.6} ${y + h} ${x} ${y + h} Q${x + w * 0.3} ${y + h / 2} ${x} ${y} Z`;
    return kind === "xor" ? body + ` M${x - 6} ${y} Q${x - 6 + w * 0.3} ${y + h / 2} ${x - 6} ${y + h}` : body;
  }
  if (kind === "not") return `M${x} ${y} L${x + w - 6} ${y + h / 2} L${x} ${y + h} Z`;
  if (kind === "mux") return `M${x} ${y} L${x + w} ${y + h * 0.18} L${x + w} ${y + h * 0.82} L${x} ${y + h} Z`;
  return null;
}

function sceneSvg(scene) {
  // emips.partlogic.Scene: the window's drawing operations, in SVG
  const svg = svgEl("svg", {});
  const layers = [0, 1, 2, 3, 4].map(() => svgEl("g", {}, svg));
  const col = (tok) => (tok === "lit" ? tokenColour(scene.lit) : SCENE_COLOURS[tok] || "var(--dp-text)");
  for (const op of scene.ops) {
    const [kind] = op;
    if (kind === "line") {
      const [, x0, y0, x1, y1, c, w, z] = op;
      svgEl("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: col(c), "stroke-width": w }, layers[z]);
    } else if (kind === "text") {
      const [, x, y, s, size, bold, c, center, rot] = op;
      svgEl("text", { transform: `translate(${x},${y}) rotate(${rot})`, "dominant-baseline": "hanging",
                      "font-size": size * PT, "font-weight": bold ? "bold" : "normal", fill: col(c),
                      "text-anchor": center ? "middle" : "start" }, layers[4]).textContent = s;
    } else if (kind === "dot") {
      const [, x, y, r, c] = op;
      svgEl("circle", { cx: x, cy: y, r, fill: col(c), stroke: col(c) }, layers[3]);
    } else if (kind === "rect") {
      const [, x, y, w, h, c] = op;
      svgEl("rect", { x, y, width: w, height: h, fill: "none", stroke: col(c),
                      "stroke-dasharray": "4 3" }, layers[0]);
    } else if (kind === "gate") {
      const [, g, x, y, w, h, c, width] = op;
      const attrs = { fill: col("part_fill"), stroke: col(c), "stroke-width": width };
      const d = gatePath(g, x, y, w, h);
      if (d) svgEl("path", { ...attrs, d }, layers[2]);
      else svgEl("rect", { ...attrs, x, y, width: w, height: h, rx: 6 }, layers[2]);
      if (g === "not") svgEl("circle", { ...attrs, cx: x + w - 3, cy: y + h / 2, r: 3 }, layers[2]);
    }
  }
  // (the drawing's extent: as Qt's itemsBoundingRect, measured once shown)
  requestAnimationFrame(() => {
    const b = svg.getBBox();
    svg.setAttribute("viewBox", `${b.x - 8} ${b.y - 8} ${b.width + 16} ${b.height + 16}`);
    svg.style.maxWidth = `${(b.width + 16) * 1.25}px`;
  });
  return svg;
}

// -- the Panes tab (emips.panedata: the window's other panes) -----------------------------

const paneOpts = JSON.parse(localStorage.getItem("emips.paneOpts") || "{}");

function setPaneOpt(name, key, value) {
  paneOpts[name] = { ...(paneOpts[name] || {}), [key]: value };
  localStorage.setItem("emips.paneOpts", JSON.stringify(paneOpts));
  showTab(`pane-${name}`);
}

async function paneAct(name, action, args = {}) {
  if (running) return;
  render(await call("pane_action", name, action, JSON.stringify(args)));
}

function fillCell(td, fill, bold) {
  if (fill) td.className = `fill-${fill.startsWith("band:") ? "valid" : fill}`;
  if (bold) td.classList.add("bold");
}

function paneBlock(name, b) {
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const frag = document.createDocumentFragment();
  if (b.title) frag.appendChild(el("h4", "", b.title));
  if (b.kind === "note") frag.appendChild(el("p", "note", b.text));
  else if (b.kind === "lines") frag.appendChild(el("pre", "", b.lines.join("\n")));
  else if (b.kind === "fields") {
    const box = el("div", "fields");
    b.cells.forEach(([name, where, bits, value], i) => {
      const f = el("div", "field");
      f.style.background = `var(--f${i % 8})`;
      f.innerHTML = `<b>${escapeHtml(name)}</b> ${escapeHtml(where)}<br><span class="bits">${escapeHtml(bits)}</span><br>${escapeHtml(value)}`;
      box.appendChild(f);
    });
    frag.appendChild(box);
  } else if (b.kind === "board") {
    const leds = el("div", "leds");
    leds.append("LEDs ");
    for (const on of b.leds) leds.appendChild(el("span", on ? "led on" : "led"));
    leds.appendChild(el("span", "malta-display", b.display));
    leds.appendChild(el("span", "", "  " + b.switches));
    frag.appendChild(leds);
  } else if (b.kind === "table" || b.kind === "core") {
    const t = el("table");
    const head = el("tr");
    for (const h of b.heads) head.appendChild(el("th", "", h));
    t.appendChild(el("thead")).appendChild(head);
    const body = t.appendChild(el("tbody"));
    b.rows.forEach((cells, r) => {
      const tr = el("tr");
      if (b.fills && b.fills[r]) tr.className = `fill-${b.fills[r].startsWith("band:") ? "valid" : b.fills[r]}`;
      if (b.bold && b.bold[r] === true) tr.classList.add("bold");
      cells.forEach((c, k) => {
        const td = el("td", "", c);
        if (b.cellfills) fillCell(td, b.cellfills[r][k], b.cellbold && b.cellbold[r][k]);
        if (b.celltips && b.celltips[r][k]) td.title = b.celltips[r][k];
        tr.appendChild(td);
      });
      if (b.tips && b.tips[r]) tr.title = b.tips[r];
      if (b.action) {                         // (a click: show that core / VP everywhere)
        tr.classList.add("pick");
        const cpu = b.pick ? b.pick[r] : r;
        tr.addEventListener("click", () => paneAct(name, b.action, { cpu }));
      }
      body.appendChild(tr);
    });
    frag.appendChild(t);
  }
  return frag;
}

function paneControl(name, c) {
  const span = document.createElement("span");
  const label = (text, input) => {
    const l = document.createElement("label");
    l.append(text + " ", input);
    if (c.tip) l.title = c.tip;
    return l;
  };
  if (c.kind === "select") {
    const sel = document.createElement("select");
    for (const [v, text] of c.items) sel.add(new Option(text, v));
    sel.value = c.value;
    sel.addEventListener("change", () => setPaneOpt(name, c.key, sel.value));
    span.appendChild(label(c.label, sel));
  } else if (c.kind === "check") {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!c.value;
    cb.disabled = !!c.disabled;
    cb.addEventListener("change", () => setPaneOpt(name, c.key, cb.checked));
    const l = document.createElement("label");
    l.append(cb, " " + c.label);
    if (c.tip) l.title = c.tip;
    span.appendChild(l);
  } else if (c.kind === "text") {
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = c.value ?? "";
    const go = () => (c.action ? paneAct(name, c.action, { value: inp.value }) : setPaneOpt(name, c.key, inp.value));
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    span.appendChild(label(c.label, inp));
    if (c.action) {
      const b = document.createElement("button");
      b.textContent = "Set";
      b.addEventListener("click", go);
      span.appendChild(b);
    }
  } else if (c.kind === "button") {
    const b = document.createElement("button");
    b.textContent = c.label;
    b.disabled = !!c.disabled;
    if (c.tip) b.title = c.tip;
    b.addEventListener("click", () => paneAct(name, c.action));
    span.appendChild(b);
  } else if (c.kind === "counter") {               // (PerfCtl: event, modes, interrupt)
    span.className = "counter-row";
    const sel = document.createElement("select");
    for (const [v, text] of c.items) sel.add(new Option(text, v));
    sel.value = c.event;
    const boxes = {};
    const modes = document.createElement("span");
    for (const [k, name] of [["u", "U"], ["s", "S"], ["k", "K"], ["x", "EXL"]]) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = c.modes.includes(k);
      cb.disabled = k === "s" && !c.supervisor;
      boxes[k] = cb;
      const l = document.createElement("label");
      l.append(cb, name + " ");
      modes.appendChild(l);
    }
    const ie = document.createElement("input");
    ie.type = "checkbox";
    ie.checked = c.ie;
    const apply = () => paneAct(name, "counter", {
      i: c.i, event: sel.value, ie: ie.checked,
      modes: Object.entries(boxes).filter(([, b]) => b.checked).map(([k]) => k).join("") });
    for (const e of [sel, ie, ...Object.values(boxes)]) e.addEventListener("change", apply);
    const l = document.createElement("label");
    l.append(ie, " interrupt on overflow");
    span.append(`PerfCnt${c.i} `, sel, modes, l);
  } else if (c.kind === "trace") {                 // (the TCB / the trace encoder's options)
    const on = document.createElement("input");
    on.type = "checkbox";
    on.checked = c.on;
    const opts = {};
    const l = document.createElement("label");
    l.append(on, " Trace");
    span.appendChild(l);
    let proto = null;
    if (c.rv) {
      proto = document.createElement("select");
      proto.add(new Option("E-Trace", "e"));
      proto.add(new Option("N-Trace", "n"));
      proto.value = c.protocol;
      span.appendChild(proto);
    }
    for (const [k, text, tip, value] of c.options) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = value;
      opts[k] = cb;
      const ol = document.createElement("label");
      ol.append(cb, " " + text);
      ol.title = tip;
      span.appendChild(ol);
    }
    const apply = () => paneAct(name, "trace", { on: on.checked, protocol: proto ? proto.value : "e",
      options: Object.fromEntries(Object.entries(opts).map(([k, cb]) => [k, cb.checked])) });
    for (const e of [on, ...Object.values(opts), ...(proto ? [proto] : [])]) e.addEventListener("change", apply);
    if (c.rv) {
      const pk = document.createElement("input");
      pk.type = "checkbox";
      pk.checked = c.packets;
      pk.addEventListener("change", () => setPaneOpt(name, "packets", pk.checked));
      const pl = document.createElement("label");
      pl.append(pk, " Packets");
      span.appendChild(pl);
    }
  }
  return span;
}

function fillPane(name, p) {
  const panel = $(`panel-pane-${name}`);
  const controls = panel.querySelector(".pane-bar");
  controls.innerHTML = "";
  for (const c of p.controls) controls.appendChild(paneControl(name, c));
  controls.hidden = !p.controls.length;
  const body = panel.querySelector(".pane-body");
  const keep = body.scrollTop;
  body.innerHTML = "";
  for (const b of p.blocks) body.appendChild(paneBlock(name, b));
  body.scrollTop = keep;
}

function fillInstr(d) {
  const box = $("instr");
  if (!d) {
    box.innerHTML = '<p class="hint">Step to see what happens inside an instruction: its bit '
      + "fields, the ALU, the address, the branch and what it changed.</p>";
    return;
  }
  if (d.error) { box.textContent = d.error; return; }
  const val = (t) => escapeHtml(t).replace(/(0x[0-9a-f]+)/g, '<span class="val">$1</span>');
  let h = `<h3>${escapeHtml(d.title)}</h3>` + d.format.map((l) => `<p>${val(l)}</p>`).join("");
  h += '<div class="fields">' + d.fields.map(([name, msb, lsb, bits, meaning], i) =>
    `<div class="field" style="background: var(--f${i % 8})"><b>${escapeHtml(name)}</b> `
    + `[${msb}:${lsb}]<br><span class="bits">${bits}</span><br>${escapeHtml(meaning)}</div>`).join("")
    + "</div>";
  for (const [title, lines] of d.sections) {
    h += `<h4>${escapeHtml(title)}</h4>` + lines.map((l) => `<p>${val(l)}</p>`).join("");
  }
  const lanes = d.lanes;
  if (lanes && lanes.lines) {
    h += `<h4>${escapeHtml(lanes.title)}</h4>`
      + lanes.lines.map((l) => `<p class="val">${escapeHtml(l)}</p>`).join("");
  } else if (lanes) {
    h += `<h4>${escapeHtml(lanes.title)}</h4><table><tr><th></th>`
      + lanes.head.map((x) => `<th>${x}</th>`).join("") + "</tr>"
      + lanes.rows.map(([label, cells]) => `<tr><td>${escapeHtml(label)}</td>`
        + cells.map((x) => `<td>${x}</td>`).join("") + "</tr>").join("") + "</table>";
  }
  if (d.exception) h += `<h4 class="exception">Exception: ${escapeHtml(d.exception)}</h4>`;
  box.innerHTML = h;
}

const CHART_STAGES = new Set(["IF", "ID", "EX", "MEM", "WB"]);

function fillChart(chart) {
  ui.chartHint.hidden = chart.rows.length > 0;
  const head = ui.chart.tHead, body = ui.chart.tBodies[0];
  head.innerHTML = "<tr><th>instruction</th>" + chart.cycles.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  body.innerHTML = chart.rows.map(([pc, text, cells]) =>
    `<tr><td class="label">${pc}  ${escapeHtml(text)}</td>` + cells.map((c) => {
      const cls = !c ? "" : c === "--" || c === "xx" ? "stall"
        : CHART_STAGES.has(c) ? `stage-${c}` : "stage-other";
      return `<td class="${cls}">${c}</td>`;
    }).join("") + "</tr>").join("");
  const sc = ui.chart.parentElement;
  sc.scrollLeft = sc.scrollWidth;
}

function escapeHtml(t) {
  return String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
}

// -- Check: the program's checks (emips.check, as the window's Run -> Check) -------------

let checkRun = null;           // a check in progress (its call's promise)

async function checkProgram() {
  if (running || !ready) return;
  countEvent("check");
  running = true;
  enable(true);
  showTab("checks", false);
  ui.checks.innerHTML = "";
  ui.checksHint.hidden = true;
  setStatus("checking…");
  const st = await call("check", program.name, ui.source.value, flags());
  running = false;
  enable(true);
  if (st && st.lines === undefined && !st.error) {
    setStatus("check stopped");              // (Pause)
    return;
  }
  fillChecks(st);
  setStatus(st.error ? "check: " + st.error.split("\n")[0]
            : `check: ${st.passed} of ${st.total} passed`);
}

function fillChecks(st) {
  ui.checks.innerHTML = "";
  const add = (kind, text, line) => {
    const tr = document.createElement("tr");
    tr.className = kind;
    const td = document.createElement("td");
    td.textContent = text;
    tr.appendChild(td);
    if (line) {
      tr.dataset.line = line;
      tr.title = `line ${line}`;
      tr.addEventListener("click", () => goToLine(line));
    }
    ui.checks.appendChild(tr);
  };
  if (st.source) add("info", `checks from ${st.source}`, 0);
  for (const [kind, text, line] of st.lines || []) {
    add(kind, kind === "output" ? `output: ${text}` : text, line);
  }
  if (st.error) add("fail", st.error, 0);
  if (st.total !== undefined) {
    add(st.passed === st.total ? "total pass" : "total fail",
        `${st.passed} of ${st.total} checks passed`, 0);
  }
}

function goToLine(line) {
  const lines = ui.source.value.split("\n");
  const start = lines.slice(0, line - 1).reduce((n, l) => n + l.length + 1, 0);
  ui.source.focus();
  ui.source.setSelectionRange(start, start + (lines[line - 1] || "").length);
  ui.source.scrollTop = Math.max(0, (line - 5) * 20);
  updateGutter();
}

// -- M27's devices: the program's dialogs, the display, files, sound ---------------------

let dialogQueue = [];          // message dialogs not shown yet
let pendingAsk = null;         // the input dialog the program waits in
let dialogBusy = false;
let continueRun = false;       // a run stopped for a dialog / input: it goes on after
let waitingInput = false;      // the last state: the program waits for an input line
const have = {};               // the device tabs this program has
let timbres = {};              // emips.midi.TIMBRES (live playback)
let audio = null, clockBase = null;

function resetDevices() {
  for (const t of DEVICE_TABS) { have[t] = false; $(`tab-${t}`).hidden = true; }
  if (DEVICE_TABS.includes(current.right)) showTab("datapath", false);
  dialogQueue = []; pendingAsk = null; continueRun = false; clockBase = null;
}

function showDevice(name, select) {
  if (have[name]) return;
  have[name] = true;
  $(`tab-${name}`).hidden = false;
  if (select) showTab(name);
}

function devices(st) {
  if (st.display) { showDevice("display", true); drawFrame(st.display); }
  if (st.files) { showDevice("files", false); fillFiles(st.files); }
  if (st.notes) {
    if (st.notes.count) showDevice("sound", false);
    if (!st.notes.count) clockBase = null;
    ui.noteLines.textContent = st.notes.lines.join("\n");
    if (ui.playLive.checked && audio) for (const n of st.notes.new) playNote(n);
  }
  for (const d of st.dialogs || []) {        // (each in the console, as the command line)
    const out = ui.console.textContent;
    ui.console.textContent += (out && !out.endsWith("\n") ? "\n" : "") + d.line + "\n";
    if (!d.input) dialogQueue.push(d);
  }
  pendingAsk = st.dialog || null;
  if (running && dialogQueue.length) {       // (the run waits while they are shown)
    continueRun = true;
    call("pause");
  } else if (!running) {
    drainDialogs();
  }
}

async function drainDialogs() {
  if (dialogBusy) return;
  dialogBusy = true;
  try {
    for (;;) {
      if (dialogQueue.length) { await showMessage(dialogQueue.shift()); continue; }
      if (pendingAsk) {
        const d = pendingAsk;
        pendingAsk = null;
        render(await call("give_input", await ask(d)));
        continue;
      }
      break;
    }
  } finally {
    dialogBusy = false;
  }
  // (a run goes on unless the program still waits for a console line)
  if (continueRun && !running && !waitingInput) { continueRun = false; run(); }
}

function modal(dlg) {
  return new Promise((resolve) => {
    dlg.addEventListener("close", () => resolve(dlg.returnValue), { once: true });
    dlg.showModal();
  });
}

async function showMessage(d) {
  ui.msgTitle.textContent = d.title;
  ui.msgText.textContent = d.text;
  ui.msgText.className = d.icon;
  ui.msgDialog.returnValue = "";
  await modal(ui.msgDialog);
}

async function ask(d) {
  ui.askTitle.textContent = d.title;
  ui.askText.textContent = d.text;
  ui.askText.className = "question";
  const confirmBox = d.kind === "confirm";
  ui.askValue.hidden = confirmBox;
  ui.askValue.value = "";
  ui.askButtons.innerHTML = "";
  const buttons = confirmBox ? [["Yes", "y"], ["No", "n"], ["Cancel", "!cancel"]]
    : [["OK", "ok"], ["Cancel", "!cancel"]];
  for (const [label, value] of buttons) {
    const b = document.createElement("button");
    b.textContent = label;
    b.value = value;
    ui.askButtons.appendChild(b);
  }
  ui.askDialog.returnValue = "";
  setTimeout(() => (confirmBox ? ui.askButtons.firstChild : ui.askValue).focus(), 0);
  const v = await modal(ui.askDialog);
  if (v === "ok") return ui.askValue.value;
  if (v === "") return confirmBox ? "!close" : "!cancel";      // (Escape: closed)
  return v;
}

let frameSize = null;          // [columns, rows, unit_w, unit_h]
const offscreen = document.createElement("canvas");

function drawFrame(d) {
  const bin = atob(d.rgb), W = d.columns, H = d.rows;
  const img = new ImageData(W, H);
  for (let i = 0, j = 0; i < W * H; i++, j += 3) {
    img.data[4 * i] = bin.charCodeAt(j);
    img.data[4 * i + 1] = bin.charCodeAt(j + 1);
    img.data[4 * i + 2] = bin.charCodeAt(j + 2);
    img.data[4 * i + 3] = 255;
  }
  offscreen.width = W; offscreen.height = H;
  offscreen.getContext("2d").putImageData(img, 0, 0);
  ui.screen.width = W * d.unit_w; ui.screen.height = H * d.unit_h;
  const ctx = ui.screen.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, ui.screen.width, ui.screen.height);
  frameSize = [W, H, d.unit_w, d.unit_h];
  ui.displayInfo.dataset.describe = d.describe + (d.on ? "" : " — off (CONTROL.ON = 0)");
  ui.displayInfo.textContent = ui.displayInfo.dataset.describe;
}

let hovered = "";
async function hoverPixel(e) {
  if (!frameSize || running) return;
  const [W, H, uw, uh] = frameSize;
  const r = ui.screen.getBoundingClientRect();
  const k = Math.min(r.width / (W * uw), r.height / (H * uh));
  const x = (e.clientX - r.left - (r.width - W * uw * k) / 2) / k;
  const y = (e.clientY - r.top - (r.height - H * uh * k) / 2) / k;
  const col = Math.floor(x / uw), row = Math.floor(y / uh);
  if (col < 0 || row < 0 || col >= W || row >= H) return;
  if (`${col},${row}` === hovered) return;
  hovered = `${col},${row}`;
  const text = await call("display_word", col, row);
  ui.displayInfo.textContent = text ? `${text}   — ${ui.displayInfo.dataset.describe}`
    : ui.displayInfo.dataset.describe;
}

function fillFiles(f) {
  ui.fileLines.textContent = f.lines.join("\n");
  const names = f.names.join("\n");
  if (ui.fileButtons.dataset.names === names) return;
  ui.fileButtons.dataset.names = names;
  ui.fileButtons.innerHTML = "";
  for (const name of f.names) {
    const b = document.createElement("button");
    b.textContent = `Download ${name}`;
    b.title = "The file as the run has left it";
    b.addEventListener("click", async () => saveBase64(name, await call("file_bytes", name),
                                                       "application/octet-stream"));
    ui.fileButtons.appendChild(b);
  }
}

function blobOf(b64, type) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

function saveBase64(name, b64, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blobOf(b64, type));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  return a.href;
}

function wakeAudio() {           // (browsers start sound only after a click)
  if (!ui.playLive.checked) return;
  if (!audio) { try { audio = new AudioContext(); } catch { return; } }
  if (audio.state === "suspended") audio.resume();
}

function playNote([start, pitch, duration, instrument, volume]) {
  const t = timbres[Math.floor(instrument / 8)];
  if (!t || duration <= 0) return;
  const [harm, attack, decay, sustain, release, percussive] = t;
  const now = audio.currentTime;
  // the program's clock: a note keeps its place after the one before, unless late
  if (clockBase === null || clockBase + start / 1000 < now) clockBase = now + 0.05 - start / 1000;
  const t0 = clockBase + start / 1000, held = duration / 1000, amp = volume / 127 * 0.25;
  const real = new Float32Array(harm.length + 1), imag = new Float32Array(harm.length + 1);
  harm.forEach((h, k) => { imag[k + 1] = h; });
  const osc = audio.createOscillator();
  osc.setPeriodicWave(audio.createPeriodicWave(real, imag));
  osc.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
  const g = audio.createGain();
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(amp, t0 + attack);
  const level = amp * sustain * (percussive ? Math.exp(-3 * Math.min(held, decay)) : 1);
  g.gain.linearRampToValueAtTime(level, t0 + Math.min(attack + decay, Math.max(held, attack)));
  g.gain.setValueAtTime(percussive ? level * Math.exp(-3 * Math.max(held - decay, 0)) : level,
                        t0 + Math.max(held, attack + decay));
  g.gain.linearRampToValueAtTime(0, t0 + Math.max(held, attack + decay) + release);
  osc.connect(g).connect(audio.destination);
  osc.start(t0);
  osc.stop(t0 + Math.max(held, attack + decay) + release + 0.05);
}

// -- the Settings dialog ---------------------------------------------------------------

function markSettings() {
  ui.settingsBtn.classList.toggle("changed", settingsFlags() !== "");
  ui.settingsBtn.title = settingsFlags() || "The machine: architecture, core, caches, MMU, board, display ...";
}

function dialogValues() {
  const values = {};
  for (const el of ui.settingsBody.querySelectorAll("[data-dest]")) {
    const v = el.type === "checkbox" ? el.checked : el.value.trim();
    if (v !== false && v !== "") values[el.dataset.dest] = v;
  }
  return values;
}

function buildSettings(values) {
  ui.settingsBody.innerHTML = "";
  for (const sec of schema) {
    const fs = document.createElement("fieldset");
    const lg = document.createElement("legend");
    lg.textContent = sec.title;
    fs.appendChild(lg);
    for (const f of sec.fields) {
      const row = document.createElement("div");
      row.className = "setting";
      const id = `set-${f.dest}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      label.textContent = f.flag;
      let el;
      if (f.kind === "bool") {
        el = document.createElement("input");
        el.type = "checkbox";
        el.checked = !!values[f.dest];
      } else if (f.kind === "choice") {
        el = document.createElement("select");
        el.add(new Option("(default)", ""));
        for (const c of f.choices) el.add(new Option(c, c));
        el.value = values[f.dest] ?? "";
      } else {
        el = document.createElement("input");
        el.type = "text";
        el.placeholder = "(default)";
        el.value = values[f.dest] ?? "";
      }
      el.id = id;
      el.dataset.dest = f.dest;
      el.addEventListener("change", greySettings);
      const help = document.createElement("div");
      help.className = "help";
      help.id = `${id}-help`;
      help.textContent = f.help;
      el.setAttribute("aria-describedby", help.id);
      row.append(label, el, help);
      fs.appendChild(row);
    }
    ui.settingsBody.appendChild(fs);
  }
}

async function greySettings() {
  const r = await call("settings", settingsFlags(dialogValues()));
  const why = r.forbidden || {};
  for (const sel of ui.settingsBody.querySelectorAll("select[data-dest]")) {
    for (const o of sel.options) {
      const reason = (why[sel.dataset.dest] || {})[o.value];
      o.disabled = !!reason;
      o.title = reason || "";
    }
  }
}

async function openSettings() {
  if (!ready || running) return;
  buildSettings(settings);
  await greySettings();
  ui.settings.showModal();
}

async function assemble() {
  const text = ui.source.value;
  const st = await call("open", program.name, text, flags());
  assembled = st.loaded ? text : null;
  ui.console.textContent = "";
  render(st);
  return st.loaded;
}

async function ensureAssembled() {          // the window's "assemble when the source changed"
  return assembled === ui.source.value || await assemble();
}

// -- running -----------------------------------------------------------------------

async function step() {
  if (running || !await ensureAssembled()) return;
  wakeAudio();
  render(await call("step"));
}

async function run() {
  if (running || !await ensureAssembled()) return;
  wakeAudio();
  running = true;
  enable(true);
  setStatus("running…");
  const st = await call("run");            // (slices: states arrive as it goes)
  running = false;
  enable(true);
  // (it waits for input: the run goes on once it is given — as the window's)
  if (st.needs_input) continueRun = true;
  render(st);
}

async function pause() {
  if (!running) return;
  continueRun = false;
  await call("pause");                     // (the run's own call ends with its state)
}

// -- showing it ----------------------------------------------------------------------

function showMessages(list, note = true) {
  ui.messages.hidden = !list.length;
  ui.messages.textContent = list.join("\n");
  ui.messages.classList.toggle("note", note);     // (an error: the program did not load)
}

function render(st, light = false) {
  showMessages(st.messages || [], !!st.loaded);
  enable(st.loaded);
  if (!st.loaded) {
    setStatus("assembly errors: fix the program, then Assemble");
    ui.listing.innerHTML = ""; ui.regs.innerHTML = "";
    rows = new Map(); lineOf = new Map(); bpLines = new Set();
    currentLine = null; positionBar(); ui.where.textContent = ""; updateGutter();
    return;
  }
  if (st.listing) fillListing(st.listing);
  if (st.reset_output) ui.console.textContent = "";
  if (st.output) {
    ui.console.textContent += st.output;
    ui.console.scrollTop = ui.console.scrollHeight;
  }
  waitingInput = !!st.needs_input;
  devices(st);
  if (st.memory) fillMemory(st.memory);
  if (st.chart) fillChart(st.chart);
  if ("instr" in st) fillInstr(st.instr);
  if (st.panes_available) showPaneTabs(st.panes_available);
  for (const [name, p] of Object.entries(st.panes || {})) fillPane(name, p);
  if (st.datapath) showModelView(st.datapath);
  if (st.debugger !== undefined) {
    ui.dbgout.textContent += st.debugger;
    ui.dbgout.scrollTop = ui.dbgout.scrollHeight;
  }
  ui.input.disabled = !st.needs_input;
  ui.keys.disabled = false;
  if (st.needs_input && !running) ui.input.focus();
  if (light) { setStatus(`running… ${st.steps} steps`); return; }
  fillRegs(st);
  markListing(st);
  markLine(st.line);
  bpLines = new Set(st.breakpoints.map((a) => lineOf.get(a)).filter((l) => l));
  updateGutter();
  const unit = st.pipeline ? "cycles" : "steps";
  let what = st.halted ? "the program has ended" : st.needs_input ? "waiting for input"
    : st.stopped === "breakpoint" ? "breakpoint"
    : st.stopped === "end" ? "the end of the program (its last instruction ran)" : "";
  if (st.error) what = st.error;
  setStatus(`${st.steps} ${unit}${st.pc !== null ? `, pc 0x${st.pc.toString(16).padStart(8, "0")}` : ""}`
            + (what ? ` — ${what}` : ""));
  ui.where.textContent = st.line ? `line ${st.line}` : "";
}

function fillListing(listing) {
  ui.listing.innerHTML = "";
  rows = new Map();
  lineOf = new Map();
  for (const [addr, hex, word, text, line] of listing) {
    const tr = document.createElement("tr");
    tr.dataset.addr = addr;
    for (const v of ["", hex, word, text, line ?? ""]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => !running && act("toggle_breakpoint", addr));
    ui.listing.appendChild(tr);
    rows.set(addr, tr);
    if (line) lineOf.set(addr, line);
  }
}

function markListing(st) {
  for (const tr of rows.values()) {
    tr.className = "";
    tr.firstChild.textContent = "";
    tr.firstChild.classList.toggle("bp", st.breakpoints.includes(Number(tr.dataset.addr)));
  }
  let first = null;
  if (st.pipeline) {
    for (const [stage, addr] of Object.entries(st.stages)) {
      const tr = rows.get(addr);
      if (!tr) continue;
      tr.classList.add(`stage-${stage}`);
      tr.firstChild.textContent = (tr.firstChild.textContent ? tr.firstChild.textContent + " " : "")
        + stage;
      if (stage === "EX" || !first) first = tr;
    }
  } else if (rows.get(st.pc)) {
    first = rows.get(st.pc);
    first.classList.add("current");
  }
  if (first) first.scrollIntoView({ block: "nearest" });
}

function fillRegs(st) {
  const rowsHtml = [];
  for (const [name, hex, dec, changed] of st.regs) {
    rowsHtml.push(`<tr><td>${name}</td><td class="${changed ? "changed" : ""}">${hex}</td>`
                  + `<td>${dec}</td></tr>`);
  }
  for (const [name, hex] of st.special) rowsHtml.push(`<tr><td>${name}</td><td>${hex}</td><td></td></tr>`);
  ui.regs.innerHTML = rowsHtml.join("");
}

// the editor: line numbers (breakpoints), the current line's bar

function updateGutter() {
  const n = ui.source.value.split("\n").length;
  const html = [];
  for (let i = 1; i <= n; i++) html.push(`<div data-line="${i}"${bpLines.has(i) ? ' class="bp"' : ""}>${i}</div>`);
  ui.gutter.innerHTML = html.join("");
  ui.gutter.scrollTop = ui.source.scrollTop;
}

let currentLine = null;
function markLine(line) {
  currentLine = line;
  positionBar();
  if (line) {                                // keep it in view
    const top = (line - 1) * 20, view = ui.source.clientHeight;
    if (top < ui.source.scrollTop || top + 20 > ui.source.scrollTop + view) {
      ui.source.scrollTop = Math.max(0, top - view / 3);
    }
  }
}

function positionBar() {
  ui.current.hidden = !currentLine;
  if (currentLine) ui.current.style.top = `${4 + (currentLine - 1) * 20 - ui.source.scrollTop}px`;
}

// -- the inputs ----------------------------------------------------------------------

ui.source.addEventListener("scroll", () => { ui.gutter.scrollTop = ui.source.scrollTop; positionBar(); });
ui.source.addEventListener("input", () => { updateGutter(); markUnsaved(); });
ui.newProg.addEventListener("click", newProgram);

function enlarge(on) {
  ui.main.classList.toggle("expanded", on);
  $("enlarge").setAttribute("aria-pressed", String(on));
  $("enlarge").textContent = on ? "⤡" : "⤢";
  $("enlarge").setAttribute("aria-label", on ? "Back" : "Enlarge");
}
$("enlarge").addEventListener("click", () => enlarge(!ui.main.classList.contains("expanded")));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && ui.main.classList.contains("expanded")
      && !document.querySelector("dialog[open]")) enlarge(false);
});
$("partBit").addEventListener("change", () => partKey && showPart(partKey));
$("dpNumbers").addEventListener("change", () => {
  if (dp) dp.badges.style.display = $("dpNumbers").checked ? "" : "none";
  if (dp24) dp24.badges.style.display = $("dpNumbers").checked ? "" : "none";
  localStorage.setItem("emips.dpNumbers", $("dpNumbers").checked ? "1" : "");
});
$("dpNumbers").checked = !!localStorage.getItem("emips.dpNumbers");
ui.screen.addEventListener("mousemove", hoverPixel);
ui.screen.addEventListener("mouseleave", () => {
  hovered = "";
  ui.displayInfo.textContent = ui.displayInfo.dataset.describe || "";
});
ui.playWav.addEventListener("click", async () => {
  const url = URL.createObjectURL(blobOf(await call("midi_download", "wav"), "audio/wav"));
  const player = new Audio(url);
  player.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
  player.play();
});
ui.saveWav.addEventListener("click", async () =>
  saveBase64(`${program.name.replace(/\.\w+$/, "")}.wav`, await call("midi_download", "wav"), "audio/wav"));
ui.saveMid.addEventListener("click", async () =>
  saveBase64(`${program.name.replace(/\.\w+$/, "")}.mid`, await call("midi_download", "mid"), "audio/midi"));
ui.playLive.addEventListener("change", wakeAudio);
buildPaneTabs();
for (const g of Object.keys(GROUPS)) {            // (a tab kept from another layout)
  if (!GROUPS[g].includes(current[g])) current[g] = GROUPS[g][0];
}
for (const t of Object.values(GROUPS).flat()) {
  if (!t.startsWith("pane-")) $(`tab-${t}`).addEventListener("click", () => showTab(t));
}

// the borders between the panes: drag to resize (kept in the browser)
const layout = JSON.parse(localStorage.getItem("emips.layout") || "{}");
for (const [k, v] of Object.entries(layout)) ui.main.style.setProperty(k, v);
for (const bar of document.querySelectorAll(".split")) {
  bar.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    const horizontal = bar.classList.contains("h");
    const before = bar.previousElementSibling, box = bar.parentElement.getBoundingClientRect();
    const start = before.getBoundingClientRect();
    const total = horizontal ? box.height : box.width;
    const x0 = horizontal ? e.clientY : e.clientX;
    const size0 = horizontal ? start.height : start.width;
    bar.setPointerCapture(e.pointerId);
    bar.classList.add("dragging");
    const move = (ev) => {
      const size = Math.max(80, Math.min(total - 120, size0 + (horizontal ? ev.clientY : ev.clientX) - x0));
      const pct = `${(100 * size / total).toFixed(2)}%`;
      ui.main.style.setProperty(bar.dataset.var, pct);
      layout[bar.dataset.var] = pct;
    };
    const up = () => {
      bar.classList.remove("dragging");
      bar.removeEventListener("pointermove", move);
      localStorage.setItem("emips.layout", JSON.stringify(layout));
      if (dp && current.right === "datapath") showTab("datapath");   // (the drawing refits)
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up, { once: true });
  });
  bar.addEventListener("dblclick", () => {         // (a double click: the default size)
    ui.main.style.removeProperty(bar.dataset.var);
    delete layout[bar.dataset.var];
    localStorage.setItem("emips.layout", JSON.stringify(layout));
  });
}
for (const el of [ui.memSegment, ui.memPhysical, ui.memBytes, ui.memDoubles]) {
  el.addEventListener("change", () => {
    if (el === ui.memSegment) ui.memAddress.value = "";
    showTab("memory");
  });
}
ui.memAddress.addEventListener("keydown", (e) => { if (e.key === "Enter") showTab("memory"); });
ui.openFile.addEventListener("click", openFromDisk);
ui.fileInput.addEventListener("change", async () => {
  const f = ui.fileInput.files[0];
  ui.fileInput.value = "";
  if (f) await openLocalFile(f);
});
ui.saveProg.addEventListener("click", saveProgram);
ui.download.addEventListener("click", saveToDisk);
ui.checkBtn.addEventListener("click", checkProgram);
ui.deleteProg.addEventListener("click", deleteProgram);
window.addEventListener("beforeunload", (e) => {
  if (program.path && ui.source.value !== loadedText) { e.preventDefault(); e.returnValue = ""; }
});
ui.gutter.addEventListener("click", async (e) => {
  const line = Number(e.target.dataset.line);
  if (!line || running || !await ensureAssembled()) return;
  act("toggle_line", line);
});
ui.programs.addEventListener("change", () => openProgram(ui.programs.value));
ui.release.addEventListener("change", () => !running && assemble());
ui.mode.addEventListener("change", () => !running && assemble());
ui.assemble.addEventListener("click", assemble);
ui.settingsBtn.addEventListener("click", openSettings);
ui.settingsDefaults.addEventListener("click", () => { buildSettings({}); greySettings(); });
ui.settings.addEventListener("close", () => {
  if (ui.settings.returnValue !== "apply") return;
  settings = dialogValues();
  localStorage.setItem("emips.settings", JSON.stringify(settings));
  markSettings();
  assemble();
});
ui.step.addEventListener("click", step);
ui.run.addEventListener("click", () => { countEvent("run"); run(); });
ui.pause.addEventListener("click", pause);
ui.back.addEventListener("click", () => act("step_back"));
ui.reset.addEventListener("click", () => act("reset"));
ui.input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const line = ui.input.value;
  ui.input.value = "";
  ui.console.textContent += line + "\n";
  call("give_input", line).then((st) => {
    render(st);
    if (continueRun && !st.needs_input) { continueRun = false; run(); }
  });
});
ui.keys.addEventListener("keydown", (e) => {   // (also while running: the worker reads it
  if (e.key !== "Enter" || !ready || assembled === null) return;   //  between slices)
  const keys = ui.keys.value;
  ui.keys.value = "";
  call("type_keys", keys + "\n").then((st) => { if (!running) render(st); });
});
ui.dbg.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  const line = ui.dbg.value.trim();
  ui.dbg.value = "";
  if (!line || !await ensureAssembled()) return;
  ui.dbgout.textContent += `(emips) ${line}\n`;
  act("debugger", line);
});
document.addEventListener("keydown", (e) => {
  if (!ready) return;
  if (e.key === "F5") { e.preventDefault(); if (!running && !ui.run.disabled) { countEvent("run"); run(); } }
  else if (e.key === "F8") { e.preventDefault(); if (!ui.checkBtn.disabled) checkProgram(); }
  else if (e.key === "F10") { e.preventDefault(); if (!ui.step.disabled) step(); }
  else if (e.key === "b" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!running) assemble(); }
  else if (e.key.toLowerCase() === "s" && (e.ctrlKey || e.metaKey) && e.shiftKey) {
    e.preventDefault(); if (!running) saveToDisk();
  }
  else if (e.key === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!running) saveProgram(); }
});

boot().catch((e) => { setStatus("could not start: " + e); console.error(e); });
