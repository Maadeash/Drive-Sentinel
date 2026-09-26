/* DSChart -- a small SVG line-chart module, written for this app.
   Local, dependency-free, no network. Draws line series over a shared x axis,
   optional vertical markers (fault frequencies, the alert moment), an optional
   horizontal band (a normal range) and an optional shaded x-span.

     DSChart.line(el, {
       series: [{name, values, x?, color?, area?}],   x defaults to 0..n-1
       xLabel, yLabel, height, xMin, xMax, yMin, yMax,
       markers: [{x, label}], band: [y0, y1], shade: [x0, x1], cursor: x,
       sparkline: true                          no axes, no legend
     })
*/
"use strict";

const DSChart = (() => {
  const PALETTE = ["#1d3572", "#4f7cc4", "#8aa4cf", "#6b7280", "#b45309"];
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    return e;
  }

  function niceTicks(lo, hi, n) {
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    const span = hi - lo, step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function label(v) {
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString("en-GB", {maximumFractionDigits: 0});
    if (a >= 10) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    if (a === 0) return "0";
    return v.toPrecision(2);
  }

  function line(container, o) {
    // The drawing is sized to the container's real pixel width, so labels are
    // the same size in a half-width card as in a full-width one.
    const W = Math.max(260, Math.round(container.clientWidth || 640)), H = o.height || (o.sparkline ? 60 : 220);
    const pad = o.sparkline ? {l: 2, r: 2, t: 4, b: 4}
                           : {l: 46, r: 12, t: o.yLabel ? 26 : 12, b: 34};   // y label gets its own strip
    const series = (o.series || []).filter(s => s.values && s.values.length);
    container.classList.add("chart");
    container.innerHTML = "";
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
                           "aria-label": o.ariaLabel || o.yLabel || "chart"});
    container.appendChild(svg);
    if (!series.length) {
      const t = el("text", {x: W / 2, y: H / 2, "text-anchor": "middle", class: "tick"});
      t.textContent = "no signal";
      svg.appendChild(t);
      return;
    }
    const xs = series.map(s => s.x || s.values.map((_, i) => i));
    let xMin = o.xMin ?? Math.min(...xs.map(a => a[0]));
    let xMax = o.xMax ?? Math.max(...xs.map(a => a[a.length - 1]));
    const all = series.flatMap(s => s.values).filter(v => v != null && isFinite(v));
    if (o.band) all.push(...o.band);
    let yMin = o.yMin ?? Math.min(...all), yMax = o.yMax ?? Math.max(...all);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const m = (yMax - yMin) * 0.06; if (o.yMin == null) yMin -= m; if (o.yMax == null) yMax += m;
    if (xMin === xMax) xMax = xMin + 1;
    const X = v => pad.l + (v - xMin) / (xMax - xMin) * (W - pad.l - pad.r);
    const Y = v => H - pad.b - (v - yMin) / (yMax - yMin) * (H - pad.t - pad.b);

    if (o.shade) svg.appendChild(el("rect", {class: "shade", x: X(o.shade[0]), y: pad.t,
      width: Math.max(0, X(o.shade[1]) - X(o.shade[0])), height: H - pad.t - pad.b}));
    if (o.band) svg.appendChild(el("rect", {class: "band", x: pad.l, y: Y(o.band[1]),
      width: W - pad.l - pad.r, height: Math.max(0, Y(o.band[0]) - Y(o.band[1]))}));

    if (!o.sparkline) {
      for (const v of niceTicks(yMin, yMax, 4)) {
        svg.appendChild(el("line", {class: "grid-line", x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v)}));
        const t = el("text", {class: "tick", x: pad.l - 6, y: Y(v) + 3, "text-anchor": "end"});
        t.textContent = label(v); svg.appendChild(t);
      }
      for (const v of niceTicks(xMin, xMax, 6)) {
        const t = el("text", {class: "tick", x: X(v), y: H - pad.b + 14, "text-anchor": "middle"});
        t.textContent = label(v); svg.appendChild(t);
      }
      svg.appendChild(el("line", {class: "axis", x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b}));
      svg.appendChild(el("line", {class: "axis", x1: pad.l, x2: pad.l, y1: pad.t, y2: H - pad.b}));
      if (o.xLabel) { const t = el("text", {class: "tick", x: W - pad.r, y: H - 4, "text-anchor": "end"});
                      t.textContent = o.xLabel; svg.appendChild(t); }
      if (o.yLabel) { const t = el("text", {class: "tick", x: 4, y: 12});
                      t.textContent = o.yLabel; svg.appendChild(t); }
    }

    // Label rows alternate in x order, so neighbouring markers never share a row.
    const rank = new Map((o.markers || []).map((mk, i) => [mk, i])
      .sort((a, b) => a[0].x - b[0].x).map(([mk], r) => [mk, r]));
    (o.markers || []).forEach(mk => {
      if (mk.x < xMin || mk.x > xMax) return;
      svg.appendChild(el("line", {class: "marker", x1: X(mk.x), x2: X(mk.x), y1: pad.t, y2: H - pad.b}));
      if (mk.label && !o.sparkline) {
        const t = el("text", {class: "marker-label", x: X(mk.x) + 3, y: pad.t + 10 + (rank.get(mk) % 2) * 11});
        t.textContent = mk.label; svg.appendChild(t);
      }
    });

    series.forEach((s, k) => {
      const x = xs[k];
      const colour = s.color || PALETTE[k % PALETTE.length];
      let d = "", pen = false, runs = [], run = [];
      s.values.forEach((v, i) => {
        if (v == null || !isFinite(v)) { pen = false; if (run.length) runs.push(run); run = []; return; }
        d += `${pen ? "L" : "M"}${X(x[i]).toFixed(1)},${Y(v).toFixed(1)}`;
        run.push([X(x[i]), Y(v)]);
        pen = true;
      });
      if (run.length) runs.push(run);
      if (s.area) {                      // a faint fill under the line, per unbroken run
        const base = H - pad.b;
        runs.filter(r => r.length > 1).forEach(r => {
          const a = `M${r[0][0].toFixed(1)},${base}` + r.map(([px, py]) => `L${px.toFixed(1)},${py.toFixed(1)}`).join("")
                    + `L${r[r.length - 1][0].toFixed(1)},${base}Z`;
          svg.appendChild(el("path", {d: a, class: "area", fill: colour}));
        });
      }
      svg.appendChild(el("path", {d, fill: "none", stroke: colour,
        "stroke-width": o.sparkline ? 1.6 : 1.8, "stroke-linejoin": "round"}));
    });

    if (o.cursor != null && o.cursor >= xMin && o.cursor <= xMax)
      svg.appendChild(el("line", {class: "cursor", x1: X(o.cursor), x2: X(o.cursor), y1: pad.t, y2: H - pad.b}));

    if (!o.sparkline && series.length > 1) {
      const lg = document.createElement("div");
      lg.className = "legend";
      lg.innerHTML = series.map((s, k) =>
        `<span><i style="background:${s.color || PALETTE[k % PALETTE.length]}"></i>${DS.esc(s.name)}</span>`).join("");
      container.appendChild(lg);
    }
  }

  return {line, PALETTE};
})();
