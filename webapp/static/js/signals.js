/* How each stage's signal is drawn. One place, used by Lift health, the Why?
   page and Live monitor, so a stage looks the same everywhere.

   bearing   vibration envelope spectrum against shaft orders, with the
             outer- and inner-race fault orders marked
   supply    current RMS per supply phase, L1/L2/L3
   inverter  Ia-Ib current imbalance, with the normal range shaded
   winding   negative-sequence current ratio
*/
"use strict";

const DSSignal = (() => {
  const PHASE_COLORS = ["#1d3572", "#4f7cc4", "#b45309"];

  function spectrumX(meta, n) {
    return Array.from({length: n}, (_, i) => (i + 0.5) * meta.x_max_order / n);
  }

  // The current frame, or a short live history.
  function live(el, kind, meta, frames, opts = {}) {
    const spark = !!opts.sparkline;
    const h = opts.height || (spark ? 70 : 220);
    if (!frames || !frames.length || !meta) { DSChart.line(el, {series: [], sparkline: spark, height: h}); return; }
    const last = frames[frames.length - 1];
    if (kind === "bearing") {
      const v = ((last.frame || last) || {}).spectrum || [];
      DSChart.line(el, {series: [{name: "vibration envelope", values: v, x: spectrumX(meta, v.length), area: true}],
        markers: spark ? [] : (meta.markers || []).map(m => ({x: m.order, label: m.label})),
        xMin: 0, xMax: meta.x_max_order, xLabel: spark ? "" : meta.x_label,
        yLabel: spark ? "" : "envelope level", sparkline: spark, height: h, ariaLabel: meta.view});
      return;
    }
    const t = frames.map(f => f.t);
    const fr = f => (f.frame || {});
    if (kind === "supply") {
      DSChart.line(el, {series: ["L1", "L2", "L3"].map((p, k) => ({name: p, x: t,
          values: frames.map(f => (fr(f).rms || [])[k]), color: PHASE_COLORS[k]})),
        xLabel: spark ? "" : "s", yLabel: spark ? "" : "A", sparkline: spark, height: h,
        yMin: 0, ariaLabel: meta.view});
      return;
    }
    if (kind === "inverter") {
      DSChart.line(el, {series: [{name: "Ia–Ib imbalance", x: t, values: frames.map(f => fr(f).imbalance), area: true}],
        band: meta.normal_band, xLabel: spark ? "" : "s", yLabel: spark ? "" : "imbalance",
        sparkline: spark, height: h, ariaLabel: meta.view});
      return;
    }
    if (kind === "winding") {
      DSChart.line(el, {series: [{name: "negative-sequence ratio", x: t, values: frames.map(f => fr(f).neg_seq_ratio), area: true}],
        xLabel: spark ? "" : "s", yLabel: spark ? "" : "ratio", sparkline: spark,
        height: h, ariaLabel: meta.view});
    }
  }

  // The stage's whole signal with the alert moment marked (Why? page).
  function whole(el, kind, meta, trigger, opts = {}) {
    const h = opts.height || 240;
    if (kind === "bearing") { live(el, kind, meta, [trigger], {height: h}); return; }
    const s = meta.series || {};
    const names = Object.keys(s);
    const x = meta.series_t || (s[names[0]] || []).map((_, i) => i);
    const tMark = kind === "winding" ? null : trigger && trigger.t;
    DSChart.line(el, {
      series: names.map((n, k) => ({name: n, values: s[n], x,
        color: kind === "supply" ? PHASE_COLORS[k] : undefined, area: kind !== "supply"})),
      band: kind === "inverter" ? meta.normal_band : undefined,
      markers: tMark != null ? [{x: tMark, label: "alert"}] : [],
      xLabel: meta.x_label, yLabel: meta.y_label, height: h, yMin: kind === "supply" ? 0 : undefined,
      ariaLabel: meta.view});
  }

  function caption(kind, meta) {
    if (!meta) return "";
    const base = DS.esc(meta.view || "");
    const cap = base ? base.charAt(0).toUpperCase() + base.slice(1) : "";
    if (kind === "bearing") return `${cap}. Dashed lines mark the outer-race and inner-race fault frequencies and their second harmonics.`;
    if (kind === "inverter") return `${cap}. The green band is the normal operating range.`;
    if (kind === "supply") return `${cap}. A collapsed line is a lost phase.`;
    return cap + ".";
  }

  return {live, whole, caption};
})();
