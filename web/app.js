// eMIPS in the browser: the page.  Python (Pyodide) runs eMIPS itself; bridge.py
// is its side: WEB.open / step / run / reset ... return the state as JSON.
"use strict";

const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), programs: $("programs"), release: $("release"), mode: $("mode"),
  assemble: $("assemble"), step: $("step"), run: $("run"), pause: $("pause"),
  back: $("back"), reset: $("reset"), messages: $("messages"), file: $("file"),
  where: $("where"), gutter: $("gutter"), current: $("current"), source: $("source"),
  listing: document.querySelector("#listing tbody"), regs: document.querySelector("#regs tbody"),
  console: $("console"), input: $("input"), keys: $("keys"), dbgout: $("dbgout"), dbg: $("dbg"),
};
const SLICE_MS = 60;           // a Run slice: about this long, then the page breathes

let web = null;                // bridge.WEB
let program = { name: "program.asm", path: null };
let assembled = null;          // the text last assembled (null: none)
let running = false, slice = 500, rows = new Map(), lineOf = new Map(), bpLines = new Set();
let programs = [];

function call(method, ...args) {
  return JSON.parse(web[method](...args));
}

function setStatus(text) { ui.status.textContent = text; }

function enable(on) {
  for (const b of [ui.assemble, ui.step, ui.run, ui.back, ui.reset]) b.disabled = !on || running;
  ui.pause.disabled = !running;
  ui.programs.disabled = running;
  ui.dbg.disabled = !on || running;
}

// -- loading -----------------------------------------------------------------------

async function boot() {
  const t0 = performance.now();
  const py = await loadPyodide();
  setStatus("Loading eMIPS…");
  const zip = await (await fetch("emips.zip")).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/emips" });
  py.runPython("import sys; sys.path.insert(0, '/emips')");
  py.runPython(await (await fetch("bridge.py")).text());
  web = py.globals.get("WEB");
  const version = py.runPython("import emips; emips.__version__");
  programs = await (await fetch("programs.json")).json();
  fillPrograms();
  setStatus(`eMIPS ${version} ready (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  const wanted = new URLSearchParams(location.search).get("open");
  await openProgram(wanted || programs[0].items[0].path);
}

function fillPrograms() {
  ui.programs.innerHTML = "";
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
  const item = find(path) || { path, files: [], title: path };
  const text = await fetchText(path);
  if (text === null) { showMessages([`could not load ${path}`]); return; }
  for (const f of item.files || []) {
    const companion = await fetchText(f);
    if (companion !== null) web.save(f.split("/").pop(), companion);
  }
  program = { name: path.split("/").pop(), path };
  ui.programs.value = path;
  ui.file.textContent = program.name;
  ui.source.value = text;
  ui.console.textContent = "";
  updateGutter();
  history.replaceState(null, "", `?open=${encodeURIComponent(path)}`);
  assemble();
}

async function fetchText(path) {          // paths are relative to the site's root
  try {
    const r = await fetch(`../${path}`);
    return r.ok ? await r.text() : null;
  } catch { return null; }
}

function flags() {
  return [ui.release.value ? `--release ${ui.release.value}` : "", ui.mode.value]
    .filter(Boolean).join(" ");
}

function assemble() {
  const st = call("open", program.name, ui.source.value, flags());
  assembled = st.loaded ? ui.source.value : null;
  ui.console.textContent = "";
  render(st);
  return st.loaded;
}

function ensureAssembled() {                // the window's "assemble when the source changed"
  return assembled === ui.source.value || assemble();
}

// -- running -----------------------------------------------------------------------

function step() {
  if (!ensureAssembled()) return;
  render(call("step"));
}

function run() {
  if (!ensureAssembled()) return;
  running = true;
  enable(true);
  setStatus("running…");
  setTimeout(tick, 0);
}

function tick() {
  if (!running) return;
  const t = performance.now();
  const st = call("run", slice);
  const dt = performance.now() - t;
  slice = Math.max(50, Math.min(200000, Math.round(slice * SLICE_MS / Math.max(dt, 1))));
  const stop = st.halted || st.needs_input || (st.stopped && st.stopped !== "") || st.error;
  if (stop) stopRunning(st); else { render(st, true); setTimeout(tick, 0); }
}

function stopRunning(st) {
  running = false;
  enable(true);
  render(st);
}

function pause() {
  if (!running) return;
  running = false;
  enable(true);
  render(call("state"));
}

// -- showing it ----------------------------------------------------------------------

function showMessages(list) {
  ui.messages.hidden = !list.length;
  ui.messages.textContent = list.join("\n");
}

function render(st, light = false) {
  showMessages(st.messages || []);
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
    tr.addEventListener("click", () => render(call("toggle_breakpoint", addr)));
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
ui.source.addEventListener("input", updateGutter);
ui.gutter.addEventListener("click", (e) => {
  const line = Number(e.target.dataset.line);
  if (!line || running || !ensureAssembled()) return;
  render(call("toggle_line", line));
});
ui.programs.addEventListener("change", () => openProgram(ui.programs.value));
ui.release.addEventListener("change", () => !running && assemble());
ui.mode.addEventListener("change", () => !running && assemble());
ui.assemble.addEventListener("click", assemble);
ui.step.addEventListener("click", step);
ui.run.addEventListener("click", run);
ui.pause.addEventListener("click", pause);
ui.back.addEventListener("click", () => render(call("step_back")));
ui.reset.addEventListener("click", () => render(call("reset")));
ui.input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const line = ui.input.value;
  ui.input.value = "";
  ui.console.textContent += line + "\n";
  render(call("give_input", line));
});
ui.keys.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !web || assembled === null) return;
  render(call("type_keys", ui.keys.value + "\n"));
  ui.keys.value = "";
});
ui.dbg.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const line = ui.dbg.value.trim();
  ui.dbg.value = "";
  if (!line || !ensureAssembled()) return;
  ui.dbgout.textContent += `(emips) ${line}\n`;
  render(call("debugger", line));
});
document.addEventListener("keydown", (e) => {
  if (!web) return;
  if (e.key === "F5") { e.preventDefault(); if (!running && !ui.run.disabled) run(); }
  else if (e.key === "F10") { e.preventDefault(); if (!ui.step.disabled) step(); }
  else if (e.key === "b" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!running) assemble(); }
});

boot().catch((e) => { setStatus("could not start: " + e); console.error(e); });
