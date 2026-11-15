"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "—" : (x > 0.99 && x < 1 ? (100 * x).toFixed(1) : Math.round(100 * x)) + "%");
const clock = (m) => { const t = Math.floor(m) + 480, d = Math.floor(t / 1440) + 1;
  return `Day ${d} · ${String(Math.floor((t % 1440) / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };
const ICON = { human: "👷", agent: "🤖", robot: "🦾" };
const STATUS = { waiting: "Waiting for a worker", in_progress: "In progress", done: "Completed", failed: "Failed" };
function showError(e) { const el = $("#error"); el.textContent = e.message; el.hidden = false; setTimeout(() => (el.hidden = true), 6000); }

let CLIENT;
try { CLIENT = sessionStorage.getItem("portal-client"); if (!CLIENT) { CLIENT = "c" + Math.random().toString(36).slice(2, 10); sessionStorage.setItem("portal-client", CLIENT); } }
catch { CLIENT = "c" + Math.random().toString(36).slice(2, 10); }

let CAT = null, SEL = null, PREF = "any";
const open = new Set();

function renderCatalog() {
  $("#p-catalog").innerHTML = CAT.services.map((s) => `<button class="svc ${SEL === s.id ? "sel" : ""}" data-svc="${s.id}">
      <span class="svc-name">${esc(s.name)}</span>
      <span class="badge ${s.grounding.split(" ")[0]}">${esc(s.grounding)}</span>
      <span class="svc-types">${s.types.map((t) => `${ICON[t]} ${esc(CAT.types[t])} ×${s.workers[t]}`).join(" · ")}</span></button>`).join("");
}

function selectService(id) {
  SEL = id; PREF = "any";
  const s = CAT.services.find((x) => x.id === id);
  $("#p-form-card").hidden = false;
  $("#p-svc-name").textContent = s.name;
  $("#p-svc-note").textContent = s.note;
  $("#p-repo-wrap").hidden = $("#p-diff-wrap").hidden = id !== "bug_fix";
  const opts = [["any", "Let the system decide"], ...s.types.map((t) => [t, `${ICON[t]} ${CAT.types[t]}`])];
  $("#p-pref").innerHTML = opts.map(([v, l]) => `<button class="seg-btn ${v === PREF ? "on" : ""}" data-pref="${v}">${esc(l)}</button>`).join("");
  renderCatalog();
}

async function submit() {
  const body = { customer: $("#p-name").value.trim() || "Customer", service: SEL, preference: PREF, priority: $("#p-priority").value,
                 repo: $("#p-repo").value, difficulty: +$("#p-diff").value, client: CLIENT };
  const r = await fetch("/api/portal/request", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "request failed");
  open.add(j.id);
  await refresh();
}

function progress(a, now) {
  if (!a || a.success !== undefined) return "";
  const f = Math.min(1, Math.max(0, (now - a.start) / (a.end - a.start)));
  return `<div class="progress tight"><div style="width:${100 * f}%"></div></div><p class="hint">About ${Math.max(0, Math.ceil(a.end - now))} min left (simulated time)</p>`;
}

function candidatesTable(a) {
  return `<table class="tbl"><tr><th>Worker</th><th>Type</th><th>Data</th><th>Caliber</th><th>Cost</th><th>Free</th></tr>${a.candidates.map((c) =>
    `<tr class="${c.worker === a.worker ? "chosen" : ""}"><td>${c.worker === a.worker ? "➜ " : ""}${esc(c.name)}</td><td>${ICON[c.type]} ${esc(CAT.types[c.type])}</td>
     <td>${esc(c.source)}</td><td>${pct(c.caliber)}</td><td>${c.cost}</td><td>${c.free ? "yes" : "busy"}</td></tr>`).join("")}</table>`;
}

function renderMine(st) {
  const mine = st.requests.slice().reverse();
  $("#p-count").textContent = mine.length;
  if (!mine.length) return;
  $("#p-mine").innerHTML = mine.map((r) => {
    const a = r.attempts[r.attempts.length - 1];
    const who = a ? `${ICON[a.type]} <b>${esc(a.name)}</b> · ${esc(CAT.types[a.type])} · caliber ${pct(a.caliber)}` : "Looking for a free worker…";
    return `<div class="req ${r.status}">
      <div class="req-head"><span><b>#${r.id} ${esc(r.service_name)}</b>${r.task ? ` <span class="hint inline">matched to recorded task ${esc(r.task.id)} (${esc(r.task.difficulty_label)})</span>` : ""}</span>
        <span class="pill ${r.status}">${STATUS[r.status]}</span></div>
      <div class="req-who">${who}</div>
      ${progress(a, st.clock)}
      <ol class="timeline">${r.timeline.map((t) => `<li><span class="when">${clock(t.t).slice(-5)}</span> ${esc(t.text)}</li>`).join("")}</ol>
      ${a ? `<details data-id="${r.id}" ${open.has(r.id) ? "open" : ""}><summary>Why this worker?</summary><p class="hint">${esc(a.reason)}</p>${candidatesTable(a)}</details>` : ""}
    </div>`;
  }).join("");
}

async function refresh() {
  const st = await fetch(`/api/portal/state?client=${CLIENT}`).then((r) => r.json());
  $("#p-clock").textContent = clock(st.clock) + " (simulated clock)";
  renderMine(st);
}

$("#p-catalog").onclick = (e) => { const b = e.target.closest("[data-svc]"); if (b) selectService(b.dataset.svc); };
$("#p-pref").onclick = (e) => { const b = e.target.closest("[data-pref]"); if (!b) return; PREF = b.dataset.pref;
  document.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("on", x === b)); };
$("#p-submit").onclick = () => submit().catch(showError);
$("#p-mine").addEventListener("toggle", (e) => { const id = +e.target.dataset.id; if (!id) return; e.target.open ? open.add(id) : open.delete(id); }, true);

(async () => {
  try {
    CAT = await fetch("/api/portal/catalog").then((r) => r.json());
    $("#p-repo").innerHTML = CAT.repos.map((r) => `<option>${esc(r)}</option>`).join("");
    $("#p-diff").innerHTML = Object.entries(CAT.difficulty).map(([k, v]) => `<option value="${k}" ${k === "1" ? "selected" : ""}>${esc(v)}</option>`).join("");
    renderCatalog();
    await refresh();
    setInterval(() => refresh().catch(() => {}), 1000);
  } catch (e) { showError(e); }
})();
