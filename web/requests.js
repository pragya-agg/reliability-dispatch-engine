"use strict";
(() => {
  const ICON = { human: "👷", agent: "🤖", robot: "🦾" };
  const TYPE = { human: "Human", agent: "AI agent", robot: "Robot" };
  const p0 = (x) => (x == null ? "—" : (x > 0.99 && x < 1 ? (100 * x).toFixed(1) : Math.round(100 * x)) + "%");
  const clock = (m) => { const t = Math.floor(m) + 480, d = Math.floor(t / 1440) + 1;
    return `Day ${d} · ${String(Math.floor((t % 1440) / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };
  let ST = null, SEL = null, CAT = null, myClient = null;
  try { myClient = sessionStorage.getItem("portal-client"); } catch {}
  const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }).then((r) => r.json());
  const visible = () => !$("#tab-requests").hidden;

  function card(r) {
    const a = r.attempts[r.attempts.length - 1];
    const who = a ? `${ICON[a.type]} ${esc(a.name)}` : `wants: ${r.preference === "any" ? "any" : TYPE[r.preference]}`;
    let bar = "";
    if (a && a.success === undefined) bar = `<div class="progress tight"><div style="width:${100 * Math.min(1, (ST.clock - a.start) / (a.end - a.start))}%"></div></div>`;
    const res = r.status === "done" ? ' <span class="yes">✓</span>' : r.status === "failed" ? ' <span class="no">✗</span>' : "";
    return `<div class="mini-req req ${r.status} ${r.auto ? "" : "mine-req"}" data-req="${r.id}"><b>#${r.id}</b> ${esc(r.service_name)}${res}<br>
      <span class="hint inline">${esc(r.customer)} · ${who}${r.attempts.length > 1 ? " · re-dispatched" : ""}</span>${bar}</div>`;
  }

  function inspectReq() {
    const r = ST && ST.requests.find((x) => x.id === SEL);
    if (!r) return;
    const parts = r.attempts.map((a, i) => `<p><b>Attempt ${i + 1}:</b> ${ICON[a.type]} ${esc(a.name)} (${esc(TYPE[a.type])}), caliber ${p0(a.caliber)}${a.success === undefined ? " · working" : a.success ? ' · <span class="yes">succeeded</span>' : ' · <span class="no">did not complete</span>'}${a.outcome_source ? ` <span class="hint inline">(${esc(a.outcome_source)})</span>` : ""}</p>
      <p class="hint">${esc(a.reason)}</p>
      <table><tr><th>Candidate</th><th>Data</th><th>Caliber</th><th>Cost</th><th>Score</th><th>Free</th></tr>${a.candidates.map((c) =>
        `<tr class="${c.worker === a.worker ? "chosen" : ""}"><td>${c.worker === a.worker ? "➜ " : ""}${ICON[c.type]} ${esc(c.name)}</td><td>${esc(c.source)}</td><td><span class="pbar" style="width:${Math.round(c.caliber * 80)}px"></span>${p0(c.caliber)}</td><td>${c.cost}</td><td>${c.score.toFixed(3)}</td><td>${c.free ? "yes" : "busy"}</td></tr>`).join("")}</table>`).join("");
    $("#r-inspect").innerHTML = `<p><b>#${r.id} ${esc(r.service_name)}</b> for ${esc(r.customer)}<br><span class="hint inline">preference: ${r.preference === "any" ? "let the system decide" : esc(TYPE[r.preference])} · priority: ${esc(r.priority)}${r.task ? ` · matched to recorded task ${esc(r.task.id)} (${esc(r.task.difficulty_label)})` : ""}</span></p>
      ${parts || '<p class="hint">Still waiting: every eligible worker is busy.</p>'}
      <ol class="timeline">${r.timeline.map((t) => `<li><span class="when">${clock(t.t).slice(-5)}</span> ${esc(t.text)}</li>`).join("")}</ol>`;
  }

  function render() {
    const s = ST, T = s.totals;
    $("#r-clock").textContent = clock(s.clock) + " (simulated clock)";
    $("#r-strip").innerHTML = tile("Requests", T.requests) + tile("Waiting", T.waiting) + tile("In progress", T.in_progress)
      + tile("Completed", T.done) + tile("Failed", T.failed) + tile("Done on first try", T.closed ? p0(T.first_try / T.closed) : "—", `${T.first_try} of ${T.closed} closed`);
    $("#r-types").innerHTML = Object.entries(s.by_type).map(([t, v]) => tile(`${ICON[t]} ${TYPE[t]} workers`, `${v.busy} / ${v.capacity} busy`,
      `${v.ok} of ${v.jobs} jobs succeeded${v.jobs ? " (" + p0(v.ok / v.jobs) + ")" : ""}`)).join("");
    const rs = s.requests.slice().reverse();
    const col = (f, n) => rs.filter(f).slice(0, n).map(card).join("") || '<p class="hint">None.</p>';
    $("#r-waiting").innerHTML = col((r) => r.status === "waiting", 12);
    $("#r-progress").innerHTML = col((r) => r.status === "in_progress", 14);
    $("#r-closed").innerHTML = col((r) => r.status === "done" || r.status === "failed", 14);
    $("#r-roster").innerHTML = ["human", "agent", "robot"].map((t) => `<div><h3>${ICON[t]} ${TYPE[t]}s</h3>${s.workers.filter((w) => w.type === t).map((w) =>
      `<div class="wrow"><span class="st ${w.busy.length >= w.capacity ? "busy" : ""}"></span><span><b>${esc(w.name)}</b> <span class="badge ${w.source === "simulated" ? "simulated" : "measured"}">${w.source === "simulated" ? "simulated" : "real"}</span><br>
        <span class="hint inline">${Object.entries(w.services).map(([k, v]) => `${esc(CAT ? CAT.services.find((x) => x.id === k).name : k)} ${v == null ? "(per task)" : p0(v)}`).join(" · ")}</span></span>
        <span class="num">${w.busy.length}/${w.capacity} · ${w.ok}/${w.done} ok</span></div>`).join("")}</div>`).join("");
    if (SEL) inspectReq();
  }

  async function poll() {
    if (!visible()) return;
    ST = await fetch("/api/portal/state").then((r) => r.json());
    render();
  }

  $("#r-speed").onchange = () => post("/api/portal/settings", { speed: +$("#r-speed").value });
  $("#r-auto").onchange = () => post("/api/portal/settings", { auto: $("#r-auto").checked });
  $("#r-rate").onchange = () => post("/api/portal/settings", { rate: +$("#r-rate").value });
  $("#r-reset").onclick = () => post("/api/portal/reset").then(() => { SEL = null; $("#r-inspect").innerHTML = '<p class="hint">Click a request to see who was considered and why.</p>'; poll(); });
  $("#tab-requests").addEventListener("click", (e) => { const el = e.target.closest("[data-req]"); if (el) { SEL = +el.dataset.req; inspectReq(); } });

  (async () => {
    CAT = await fetch("/api/portal/catalog").then((r) => r.json());
    $("#r-assume").innerHTML = table(CAT.assumptions.caliber.map((a) => ({ "Service": a.service, "Worker type": a.type, "Assumed caliber range": a.range })))
      + `<p class="hint">Assumed cost per job: ${Object.entries(CAT.assumptions.cost).map(([k, v]) => `${esc(k)} ${v}`).join(", ")} units. Parallel jobs per worker: ${Object.entries(CAT.assumptions.capacity).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}.</p>`;
    const s0 = await fetch("/api/portal/state").then((r) => r.json());
    $("#r-speed").value = String(s0.speed); $("#r-auto").checked = s0.auto; $("#r-rate").value = s0.rate;
    const fromHash = () => { if (location.hash === "#requests") document.querySelector('[data-tab="requests"]').click(); };
    window.addEventListener("hashchange", fromHash);
    fromHash();
    setInterval(() => poll().catch(() => {}), 1000);
    poll();
  })();
})();
