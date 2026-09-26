/* Shared page chrome and helpers. No framework, no build step, no network
   request outside this app. */
"use strict";

const DS = (() => {
  const NAV = [
    ["/fleet", "Fleet"], ["/alerts", "Alerts"], ["/monitor", "Live monitor"],
    ["/board", "Board live"], ["/hardware", "Hardware"], ["/about", "About"],
    ["/settings", "Settings"],
  ];
  const BRAND = {name: "DriveSentinel", footer: "Prepared for KONE Elevate'26 · Problem Statement 06"};
  let META = null;

  const ICON = {
    mark: `<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><rect x="1" y="1" width="30" height="30" rx="8" fill="#2a4a94"/>
      <path d="M5 17h5l2.5-6 4 12 3-9 2 3H27" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    menu: `<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
    wrench: `<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.1L3.5 17.2a1.8 1.8 0 0 0 2.5 2.5l5.8-5.8a4 4 0 0 0 5.1-5.4l-2.4 2.4-2.3-.6-.6-2.3z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>`,
    check: `<svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    arrow: `<svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    bell: `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 16V11a6 6 0 0 1 12 0v5l1.5 2h-15zM10 20a2 2 0 0 0 4 0" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`,
  };

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  }

  function toSignIn() {
    location.href = "/signin?next=" + encodeURIComponent(location.pathname + location.search);
  }

  async function json(url, opts) {
    const r = await fetch(url, Object.assign({credentials: "same-origin"}, opts || {}));
    if (r.status === 401) { toSignIn(); throw new Error("signed out"); }
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  const stClass = status => "st-" + String(status || "NO DATA").replace(" ", "");
  const badge = (status, solid) =>
    `<span class="badge ${solid ? "solid " : ""}${stClass(status)}">${esc(status || "NO DATA")}</span>`;
  const rating = label => `<span class="rating${label === "ADVISORY" ? " adv" : ""}">${esc(label)}</span>`;

  function clock(unix) {
    if (!unix) return "—";
    return new Date(unix * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  }
  function ago(unix) {
    if (!unix) return "—";
    const s = Math.max(0, Math.round(Date.now() / 1000 - unix));
    if (s < 5) return "just now";
    if (s < 60) return `${s} s ago`;
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    return clock(unix);
  }
  const pct = x => x == null ? "—" : `${Math.round(x * 100)} %`;
  const fmt = (x, d = 4) => x == null ? "—" : Number(x).toFixed(d);
  const int = x => x == null ? "—" : Number(x).toLocaleString("en-GB");

  // One alert card, the same on every page. `opts.fresh` animates arrival;
  // `opts.link` adds the Why? link; `opts.time` overrides the timestamp.
  function alertCard(a, opts = {}) {
    const facts = [["Where", a.where], ["How bad", a.how_bad]].filter(([, v]) => v);
    const time = opts.time !== undefined ? opts.time : a.time_unix;
    return `<article class="alert-card ${stClass(a.status)}${opts.fresh ? " fresh" : ""}">
      <div class="alert-top">${badge(a.status, true)}
        <span class="alert-meta"><b>${esc(a.lift)}</b> · ${esc(a.stage_name)}</span>
        <span class="time">${clock(time)}</span></div>
      <div class="alert-headline">${esc(a.headline)}</div>
      <div class="alert-meta">${esc(a.alert_type)}${opts.site === false ? "" : ` · ${esc(a.site)}`}</div>
      ${facts.length ? `<div class="facts">${facts.map(([k, v]) =>
        `<div><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join("")}</div>` : `<div style="height:12px"></div>`}
      <div class="action">${ICON.wrench}<div><b>Suggested action</b> · ${esc(a.action)}</div></div>
      <div class="alert-foot"><span class="muted">Confidence ${pct(a.confidence)} · ${a.n_obs} observations pooled</span>
        ${opts.link === false || !a.id ? "" : `<a href="/alerts/${encodeURIComponent(a.id)}">Why? ${ICON.arrow}</a>`}</div>
    </article>`;
  }

  function emptyState(text) {
    return `<div class="card empty">${ICON.bell}<div>${esc(text)}</div></div>`;
  }

  async function chrome(active) {
    try { META = await json("/api/meta"); } catch (e) { META = null; }
    const b = META ? META.brand : BRAND;
    const header = document.createElement("header");
    header.className = "site-header";
    header.innerHTML = `<div class="inner">
        <a class="brand" href="/fleet"><span class="name">${esc(b.name)}</span></a>
        <button class="menu-toggle" aria-label="Menu" aria-expanded="false">${ICON.menu}</button>
        <nav class="main" aria-label="Main">${NAV.map(([h, t]) =>
          `<a href="${h}"${h === active ? ' class="active" aria-current="page"' : ""}>${t}</a>`).join("")}</nav>
      </div>`;
    const toggle = header.querySelector(".menu-toggle");
    toggle.onclick = () => {
      const open = header.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    };
    const footer = document.createElement("footer");
    footer.className = "site-footer";
    footer.innerHTML = `<div class="inner"><span>${esc(b.footer)}</span>
        <a href="/engineering">Engineering view</a></div>`;
    document.body.prepend(header);
    document.body.append(footer);
    return META;
  }

  function meta() { return META; }

  // Poll without piling up requests, and pause while the tab is hidden.
  function every(ms, fn) {
    let busy = false;
    const run = async () => {
      if (busy || document.hidden) return;
      busy = true;
      try { await fn(); } catch (e) { /* next tick retries */ } finally { busy = false; }
    };
    return setInterval(run, ms);
  }

  return {esc, json, badge, rating, stClass, clock, ago, pct, fmt, int, alertCard,
          emptyState, chrome, meta, every, ICON};
})();
