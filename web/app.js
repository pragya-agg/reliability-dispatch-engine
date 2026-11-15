"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "—" : (100 * x).toFixed(d) + "%");
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const POLICY_COLOR = { random: "--p-random", "track record": "--p-track", model: "--p-model", learning: "--p-learning" };

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function showError(e) { const el = $("#error"); el.textContent = "Error: " + e.message; el.hidden = false; setTimeout(() => (el.hidden = true), 6000); }

// ---------- playback --------------------------------------------------------------------------
class Player {
  constructor(btn, speedSel, stepFn) {
    Object.assign(this, { btn, speedSel, stepFn, playing: false, busy: false });
    btn.onclick = () => (this.playing ? this.pause() : this.play());
    setInterval(() => this.tick(), 250);
  }
  play() { this.playing = true; this.btn.textContent = "❚❚ Pause"; }
  pause() { this.playing = false; this.btn.textContent = "▶ Play"; }
  async tick() {
    if (!this.playing || this.busy) return;
    this.busy = true;
    try { if (await this.stepFn(Math.max(1, Math.round(+this.speedSel.value / 4)))) this.pause(); }
    catch (e) { this.pause(); showError(e); }
    finally { this.busy = false; }
  }
}

// ---------- tiny line chart -------------------------------------------------------------------
function lineChart(canvas, series, { yMin = 0, yMax = 1, fmtY = (v) => v, fmtX = (v) => v, xMax } = {}) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.height / (canvas._dpr || 1);
  canvas._dpr = dpr; canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 44, R = 10, T = 10, B = 40;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const x0 = xs.length ? Math.min(...xs) : 0, x1 = Math.max(xMax ?? 0, xs.length ? Math.max(...xs) : 1, x0 + 1);
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R), Y = (y) => T + (1 - (y - yMin) / (yMax - yMin)) * (H - T - B);
  g.font = "11px system-ui"; g.fillStyle = css("--muted"); g.strokeStyle = css("--line"); g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = yMin + ((yMax - yMin) * i) / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.textAlign = "right"; g.fillText(fmtY(v), L - 5, y + 4);
  }
  g.textAlign = "center";
  for (let i = 0; i <= 4; i++) { const v = x0 + ((x1 - x0) * i) / 4; g.fillText(fmtX(v), X(v), H - B + 15); }
  let lx = L;
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []);
    g.beginPath(); s.points.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.stroke();
    g.setLineDash([]); g.fillStyle = s.color; g.fillRect(lx, H - 14, 10, 3);
    g.fillStyle = css("--muted"); g.textAlign = "left"; g.fillText(s.label, lx + 14, H - 10);
    lx += g.measureText(s.label).width + 30;
  }
}

function table(rows, cols) {
  if (!rows || !rows.length) return '<p class="hint">Not available. Run python run_all.py.</p>';
  cols = cols || Object.keys(rows[0]);
  const f = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : esc(v));
  return `<div class="tblwrap"><table class="tbl"><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows
    .map((r) => `<tr>${cols.map((c) => `<td>${f(r[c])}</td>`).join("")}</tr>`).join("")}</table></div>`;
}
const tile = (k, v, s = "") => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;

let META = null;

// =========================== AI AGENT DISPATCH ===================================================
const A = { sid: null, jobs: [], pool: [], state: null, assign: {}, outcome: {}, feed: [], series: {}, flash: {}, sel: null };
const simClock = (m) => { const t = m + 8 * 60, d = Math.floor(t / 1440) + 1, h = Math.floor((t % 1440) / 60), mi = t % 60;
  return `Day ${d} · ${String(h).padStart(2, "0")}:${String(mi).padStart(2, "0")}`; };
const agentName = (k) => A.pool[k]?.name ?? k;

async function agentsReset() {
  const cfg = { pool_size: +$("#a-pool").value, slots: +$("#a-slots").value, load: +$("#a-load").value,
                batch_minutes: +$("#a-batch").value, seed: +$("#a-seed").value };
  const r = await api("/api/agents/new", cfg);
  Object.assign(A, { batch: r.config.batch_minutes, sid: r.sid, jobs: r.jobs, pool: r.pool, state: r.state, assign: {}, outcome: {}, feed: [], flash: {}, sel: null });
  A.series = Object.fromEntries([...META.policies.map((p) => p.id), "oracle"].map((k) => [k, []]));
  $("#a-inspect").innerHTML = '<p class="hint">Click a job in the queue or feed, or an agent card.</p>';
  renderAgents();
}

async function agentsStep(min) {
  const r = await api(`/api/agents/${A.sid}/step?minutes=${min}`);
  A.flash = {};
  for (const e of r.events) {
    if (e.type === "assign") {
      A.assign[e.job] = e; A.flash[e.agent] = "assign";
      A.feed.push({ cls: "assign", t: e.t, job: e.job, html: `<b>${esc(A.jobs[e.job].id)}</b> → ${esc(agentName(e.agent))} <span class="hint inline">caliber ${pct(e.score, 0)}, waited ${e.wait} min</span>` });
    } else if (e.type === "complete") {
      A.outcome[e.job] = e; A.flash[e.agent] = e.resolved ? "ok" : "fail";
      A.feed.push({ cls: e.resolved ? "ok" : "bad", t: e.t, job: e.job, html: `${e.resolved ? "✓ resolved" : "✗ failed"} <b>${esc(A.jobs[e.job].id)}</b> by ${esc(agentName(e.agent))}` });
    }
  }
  if (A.feed.length > 300) A.feed = A.feed.slice(-300);
  A.state = r.state;
  for (const p of r.state.policies) if (p.rate != null) A.series[p.policy].push([r.now, p.rate]);
  A.series.oracle.push([r.now, r.state.oracle_rate_so_far]);
  renderAgents();
  if (A.sel) inspect(A.sel);
  return r.finished;
}

function renderAgents() {
  const s = A.state, L = s.policies.find((p) => p.policy === "learning");
  $("#a-clock").textContent = simClock(s.now) + " (simulated clock)";
  const busy = s.agents.reduce((a, x) => a + x.busy, 0), slots = s.agents.reduce((a, x) => a + x.slots, 0);
  const nextRound = (A.batch - (s.now % A.batch)) % A.batch;
  $("#a-strip").innerHTML = tile("Jobs arrived", `${s.arrived} / ${s.total}`)
    + tile("Waiting in queue", s.queue.length, `next dispatch round in ${nextRound} min`)
    + tile("Slots busy", `${busy} / ${slots}`) + tile("Completed", L.done, `${L.resolved} resolved`)
    + tile("Success rate", pct(L.rate), "learning dispatcher") + tile("Avg. wait", L.avg_wait == null ? "—" : `${Math.round(L.avg_wait)} min`);
  $("#a-progress").style.width = `${(100 * L.done) / s.total}%`;
  $("#a-qcount").textContent = s.queue.length;
  $("#a-queue").innerHTML = s.queue.slice(0, 60).map((j) => { const J = A.jobs[j];
    return `<div class="job" data-job="${j}"><div class="t">${esc(J.id)}</div><div class="m">${esc(J.difficulty)} · ~${J.work_min} min · waiting ${s.now - J.arrival} min</div></div>`; }).join("")
    || '<p class="hint">Queue empty.</p>';
  $("#a-agents").innerHTML = s.agents.map((a) => `<div class="agent ${A.flash[a.idx] ? "flash-" + A.flash[a.idx] : ""}" data-agent="${a.idx}">
      <div class="n">${esc(a.name)}</div>
      <div class="r">track record ${pct(a.rating, 0)} · live ${a.resolved}/${a.done}</div>
      <div class="slots">${Array.from({ length: a.slots }, (_, i) => `<span class="slot ${i < a.busy ? "busy" : ""}"></span>`).join("")}</div>
      <div class="chips">${a.running.map((j) => `<span class="chip">${esc(A.jobs[j].repo.split("/")[1])}</span>`).join("")}</div></div>`).join("");
  $("#a-feed").innerHTML = A.feed.slice(-120).reverse().map((f) => `<div class="ev ${f.cls}" data-job="${f.job}"><span class="when">${simClock(f.t).slice(-5)}</span>${f.html}</div>`).join("")
    || '<p class="hint">Press Play.</p>';
  $("#a-score").innerHTML = s.policies.map((p) => `<div class="score-row"><span class="swatch" style="background:${css(POLICY_COLOR[p.policy])}"></span>
      <span>${esc(p.label)}</span><span class="num">${p.resolved}/${p.done}</span><span class="num"><b>${pct(p.rate)}</b></span>
      <div class="bar"><div style="width:${100 * (p.rate || 0)}%;background:${css(POLICY_COLOR[p.policy])}"></div></div></div>`).join("")
    + `<div class="score-row"><span class="swatch" style="background:${css("--p-oracle")}"></span><span>Oracle: share of arrived jobs any pool agent can solve</span><span></span><span class="num">${pct(s.oracle_rate_so_far)}</span></div>`;
  lineChart($("#a-chart"), [...META.policies.map((p) => ({ label: p.label, color: css(POLICY_COLOR[p.id]), points: A.series[p.id] })),
    { label: "Oracle", color: css("--p-oracle"), dash: [5, 4], width: 1.5, points: A.series.oracle }],
    { yMin: 0, yMax: 1, fmtY: (v) => pct(v, 0), fmtX: (v) => simClock(Math.round(v)).slice(-5) });
}

function inspect(sel) {
  A.sel = sel;
  const el = $("#a-inspect");
  if (sel.agent != null) {
    const a = A.state.agents[sel.agent];
    const rows = Object.entries(a.caliber_by_repo).sort((x, y) => y[1] - x[1]);
    el.innerHTML = `<p><b>${esc(a.name)}</b><br><span class="hint inline">${esc(A.pool[sel.agent].full_name)}</span></p>
      <p class="hint">Track record on training tasks: ${pct(a.rating)}. Live: ${a.resolved} resolved of ${a.done}. Busy slots: ${a.busy}/${a.slots}.</p>
      <table><tr><th>Repository</th><th>Current caliber (model + what it learned live)</th></tr>${rows.map(([r, v]) =>
        `<tr><td>${esc(r)}</td><td><span class="pbar" style="width:${Math.round(v * 140)}px"></span>${pct(v, 0)}</td></tr>`).join("")}</table>`;
    return;
  }
  const j = sel.job, J = A.jobs[j], as = A.assign[j], oc = A.outcome[j];
  const status = oc ? (oc.resolved ? '<span class="yes">resolved</span>' : '<span class="no">failed</span>') : as ? "in progress" : "waiting";
  let html = `<p><b>${esc(J.id)}</b><br><span class="hint inline">${esc(J.repo)} · difficulty ${esc(J.difficulty)} · assumed work ${J.work_min} min · arrived ${simClock(J.arrival)} · status: ${status}</span></p>`;
  if (as) {
    const order = as.scores.map((v, k) => [k, v]).sort((x, y) => y[1] - x[1]);
    html += `<p class="hint">Dispatched at ${simClock(as.t)} after waiting ${as.wait} min. Caliber of every pool agent at that moment; only agents with a free slot could take it.${oc ? " The last column is what each agent actually did on this task (recorded)." : ""}</p>
      <table><tr><th>Agent</th><th>Caliber</th>${oc ? "<th>Recorded result</th>" : ""}</tr>${order.map(([k, v]) =>
        `<tr class="${k === as.agent ? "chosen" : ""}"><td>${k === as.agent ? "➜ " : ""}${esc(agentName(k))}</td><td><span class="pbar" style="width:${Math.round(v * 120)}px"></span>${pct(v, 0)}</td>${oc ? `<td>${oc.all_outcomes[k] ? '<span class="yes">✓</span>' : '<span class="no">✗</span>'}</td>` : ""}</tr>`).join("")}</table>`;
  } else html += '<p class="hint">Not dispatched yet.</p>';
  el.innerHTML = html;
}

async function whatIf() {
  const r = await api(`/api/agents/${A.sid}/whatif`, { repo: $("#w-repo").value, difficulty: +$("#w-diff").value,
    statement_len: +$("#w-len").value, year: 2023 });
  $("#w-out").innerHTML = `<table class="tbl"><tr><th>Rank</th><th>Agent</th><th>Predicted success</th><th>Free slot now</th></tr>${r.ranking.map((x, i) =>
    `<tr class="${i === 0 ? "chosen" : ""}"><td>${i + 1}</td><td>${esc(x.name)}</td><td><span class="pbar" style="width:${Math.round(x.p * 120)}px"></span>${pct(x.p, 0)}</td><td>${x.has_free_slot ? "yes" : "no"}</td></tr>`).join("")}</table>`;
}

// =========================== COURIERS ============================================================
const C = { sid: null, meta: null, status: null, feed: [], series: { late: [], caught: [], flagged: [] }, stats: null, now: 0 };
const courierClock = (m) => { const d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60), mi = ((m % 60) + 60) % 60;
  return `${C.meta ? C.meta.day : ""} ${String(h).padStart(2, "0")}:${String(mi).padStart(2, "0")}${d > 0 ? ` (+${d} day)` : ""}`; };

async function couriersReset() {
  const r = await api(`/api/couriers/new?city=${encodeURIComponent($("#c-city").value)}`);
  Object.assign(C, { sid: r.sid, meta: r.meta, status: new Int8Array(r.meta.jobs), feed: [], series: { late: [], caught: [], flagged: [] }, stats: null, now: r.meta.start });
  $("#c-mapnote").textContent = `${r.meta.city}, ${r.meta.day}: ${r.meta.jobs.toLocaleString()} real pickups by ${r.meta.couriers} couriers`;
  $("#c-note").textContent = `High risk = model score ≥ ${pct(r.meta.threshold, 2)} (the riskiest 10% of all test-period jobs; set without looking at outcomes). This day's real late rate: ${pct(r.meta.base_rate, 2)}.`;
  renderCouriers();
}

async function couriersStep(min) {
  const r = await api(`/api/couriers/${C.sid}/step?minutes=${min}`);
  const S = C.status;
  for (const e of r.events) {
    if (e.type === "accept") {
      S[e.j] = e.flag ? 2 : 1;
      if (e.flag || $("#c-filter").value === "all") C.feed.push({ alert: e.flag, cls: e.flag ? "warn" : "", t: e.t, html: `${e.flag ? "⚠ <b>High-risk job</b>" : "Job accepted"} by courier #${e.courier}: risk ${pct(e.risk, 1)}, window ends ${courierClock(e.due).slice(-5)}, courier track record ${pct(e.track, 1)} late, ${e.open_jobs} other open jobs` });
    } else if (e.type === "overdue") {
      S[e.j] = 5;
      C.feed.push({ alert: true, cls: "bad", t: e.t, html: `⏰ <b>Overdue</b>: courier #${e.courier} missed the pickup window. ${e.flag ? '<span class="yes">Flagged at acceptance</span>' : '<span class="no">Not flagged</span>'} (risk ${pct(e.risk, 1)})` });
    } else if (e.type === "pickup") {
      S[e.j] = e.late ? 4 : 3;
      if (e.late) C.feed.push({ alert: true, cls: "bad", t: e.t, html: `Late pickup by courier #${e.courier}, ${e.late_by} min after the window. ${e.flag ? '<span class="yes">Flagged</span>' : '<span class="no">Not flagged</span>'}` });
      else if ($("#c-filter").value === "all") C.feed.push({ alert: false, cls: "ok", t: e.t, html: `✓ On-time pickup by courier #${e.courier}` });
    }
  }
  if (C.feed.length > 400) C.feed = C.feed.slice(-400);
  C.stats = r.stats; C.now = r.now;
  C.series.late.push([r.now, r.stats.late]); C.series.caught.push([r.now, r.stats.caught]);
  renderCouriers();
  return r.finished;
}

function drawMap() {
  const cv = $("#c-map"), m = C.meta; if (!m) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = 460;
  cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr);
  g.fillStyle = css("--bg"); g.fillRect(0, 0, W, H);
  const [x0, x1] = m.bounds.lng, [y0, y1] = m.bounds.lat;
  const kx = (W - 20) / (x1 - x0), ky = (H - 20) / (y1 - y0), k = Math.min(kx, ky);
  const ox = (W - (x1 - x0) * k) / 2, oy = (H - (y1 - y0) * k) / 2;
  const P = (j) => [ox + (m.points.lng[j] - x0) * k, H - (oy + (m.points.lat[j] - y0) * k)];
  const layers = [[3, css("--ok"), 1.6, 0.35], [1, "#7f8c9d", 2.2, 0.9], [4, css("--bad"), 2.6, 0.9], [2, css("--warn"), 3.2, 1]];
  for (const [st, color, r, a] of layers) {
    g.fillStyle = color; g.globalAlpha = a;
    for (let j = 0; j < C.status.length; j++) if (C.status[j] === st) { const [x, y] = P(j); g.fillRect(x - r / 2, y - r / 2, r, r); }
  }
  g.globalAlpha = 1; g.strokeStyle = css("--bad"); g.lineWidth = 1.5;
  for (let j = 0; j < C.status.length; j++) if (C.status[j] === 5) { const [x, y] = P(j); g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.stroke(); }
}

function renderCouriers() {
  $("#c-clock").textContent = courierClock(C.now);
  const s = C.stats || { known: 0, flagged_known: 0, accepted: 0, open: 0, picked: 0, late: 0, flagged: 0, caught: 0, overdue_open: 0, watchlist: [] };
  $("#c-strip").innerHTML = tile("Jobs accepted", s.accepted.toLocaleString(), `of ${C.meta.jobs.toLocaleString()} today`)
    + tile("Open now", s.open.toLocaleString(), `${s.overdue_open} overdue`) + tile("Picked up", s.picked.toLocaleString())
    + tile("Late so far", s.late, s.known ? pct(s.late / s.known, 1) + " of jobs with a known outcome" : "")
    + tile("Flagged high risk", s.flagged.toLocaleString(), s.accepted ? pct(s.flagged / s.accepted, 1) + " of accepted" : "")
    + tile("Late jobs flagged in advance", s.late ? pct(s.caught / s.late, 0) : "—", `${s.caught} of ${s.late}`)
    + tile("Flags confirmed late", s.flagged_known ? pct(s.caught / s.flagged_known, 1) : "—",
           `of ${s.flagged_known} flagged jobs with a known outcome; day's late rate ${pct(C.meta.base_rate, 1)}`);
  const all = $("#c-filter").value === "all";
  $("#c-feed").innerHTML = C.feed.filter((f) => all || f.alert).slice(-150).reverse()
    .map((f) => `<div class="ev ${f.cls}"><span class="when">${courierClock(f.t).slice(-5)}</span>${f.html}</div>`).join("") || '<p class="hint">Press Play.</p>';
  $("#c-watch").innerHTML = s.watchlist.length ? `<div class="tblwrap"><table class="tbl"><tr><th>Courier</th><th>Open jobs</th><th>High-risk open</th><th>Overdue</th><th>Highest risk</th></tr>${s.watchlist.map((w) =>
    `<tr><td>#${w.courier}</td><td>${w.open}</td><td>${w.flagged}</td><td>${w.overdue ? `<span class="no">${w.overdue}</span>` : 0}</td><td>${pct(w.max_risk, 1)}</td></tr>`).join("")}</table></div>`
    : '<p class="hint">No open jobs.</p>';
  lineChart($("#c-chart"), [{ label: "Late pickups so far", color: css("--bad"), points: C.series.late },
    { label: "…of which flagged at acceptance", color: css("--warn"), points: C.series.caught }],
    { yMin: 0, yMax: Math.max(10, ...C.series.late.map((p) => p[1])) * 1.1, fmtY: (v) => Math.round(v), fmtX: (v) => courierClock(Math.round(v)).slice(-5) });
  drawMap();
}

// =========================== EVIDENCE & WIRING ===================================================
function renderEvidence() {
  const ev = META.evaluation;
  $("#e-live").innerHTML = ev ? table(ev.map((r) => ({ "Load": r.load, "Random": pct(r.random), "Best track record": pct(r["track record"]),
    "Model + optimal": pct(r.model), "Model + learning": pct(r.learning), "Oracle": pct(r.oracle),
    "Model − track record (95% CI)": r["model - track record (95% CI)"], "Learning − model (95% CI)": r["learning - model (95% CI)"],
    "Avg wait (min)": Math.round(r.avg_wait_min) }))) : table(null);
  const o = META.offline || {};
  $("#e-couriers").innerHTML = table(o.couriers);
  $("#e-agents").innerHTML = table(o.agents);
  $("#e-assign").innerHTML = table(o.assignment);
}

document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tabpanel").forEach((p) => (p.hidden = p.id !== "tab-" + b.dataset.tab));
  if (b.dataset.tab === "couriers") renderCouriers(); else if (b.dataset.tab === "agents") renderAgents();
}));
$("#a-queue").onclick = $("#a-feed").onclick = (e) => { const el = e.target.closest("[data-job]"); if (el) inspect({ job: +el.dataset.job }); };
$("#a-agents").onclick = (e) => { const el = e.target.closest("[data-agent]"); if (el) inspect({ agent: +el.dataset.agent }); };
$("#a-reset").onclick = () => { aPlayer.pause(); agentsReset().catch(showError); };
$("#a-step").onclick = () => agentsStep(30).catch(showError);
$("#w-go").onclick = () => whatIf().catch(showError);
$("#c-reset").onclick = $("#c-city").onchange = () => { cPlayer.pause(); couriersReset().catch(showError); };
$("#c-filter").onchange = () => renderCouriers();
window.addEventListener("resize", () => { if (A.state) renderAgents(); if (C.meta) renderCouriers(); });
const aPlayer = new Player($("#a-play"), $("#a-speed"), agentsStep);
const cPlayer = new Player($("#c-play"), $("#c-speed"), couriersStep);

(async () => {
  try {
    META = await fetch("/api/meta").then((r) => r.json());
    $("#m-train").textContent = `${META.train_tasks} older tasks (${META.train_rows.toLocaleString()} recorded agent attempts)`;
    $("#m-replay").textContent = META.replay_tasks;
    $("#w-repo").innerHTML = META.repos.map((r) => `<option>${esc(r)}</option>`).join("");
    $("#w-diff").innerHTML = Object.entries(META.difficulty).map(([k, v]) => `<option value="${k}" ${k === "1" ? "selected" : ""}>${esc(v)}</option>`).join("");
    $("#c-city").innerHTML = META.cities.map((c) => `<option>${esc(c)}</option>`).join("");
    renderEvidence();
    await agentsReset();
    await couriersReset();
  } catch (e) { showError(e); }
})();
