"""
Render artifacts/multistage/*/ JSON into docs/results_multistage.md.

Per workflow_v2.md §9 rule 2, results markdown is GENERATED, never hand-typed.
Anything not run reads NOT RUN.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivesentinel import config as C
from drivesentinel import fusion as FU
from drivesentinel import trip as TR
from scripts._render_supply import render_supply, render_cnn_comparison
from scripts._render_inverter import render_inverter

BRANCHES = [
    ("winding", "B-S4", "S4", "D2 KAIST PMSM"),
    ("supply", "B-S1", "S1", "D4 Thomas"),
    ("inverter_telemetry", "B-S2/S3", "S2/S3", "D3 Bacha"),
]
OUT = os.path.join(C.PROJECT_ROOT, "docs", "results_multistage.md")


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.{nd}f}"
    return str(v)


def render_winding(r):
    L = []
    L.append(f"**Task: {r['task']}.** {r['n_windows']:,} windows, "
             f"{r['window_s']} s / {r['hop_s']} s hop, "
             f"{len(r['feature_names'])} ratio features.\n")

    L.append("### `healthy`: NOT MEASURABLE\n")
    L.append(f"> {r['healthy_reason']}\n")
    L.append("No healthy number is reported here, on any metric card, or as a fusion "
             "input. The confound is total rather than partial, so a caveated number "
             "would be worse than none.\n")

    # -- V4 first: it is the headline -----------------------------------
    L.append("### V4 — batch control (headline)\n")
    L.append("Same features, predicting the acquisition variable instead of the fault "
             "class.\n")
    L.append("| Target | Split | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|")
    for k, v in r["V4"]["targets"].items():
        if "accuracy" not in v:
            continue
        leaky = k.endswith("_shuffled_LEAKY")
        split = "shuffled **(leaky reference)**" if leaky else "leave-one-motor-out"
        name = k.replace("_shuffled_LEAKY", "")
        L.append(f"| {name} | {split} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} "
                 f"| {fmt(v['majority_baseline'])} |")
    L.append("")

    p = r["V4_probe"]
    L.append("#### Model-free probe — 1-NN on `i0rel_residual` alone\n")
    L.append(f"> {p['note']}\n")
    L.append("| Subset | n | → session | → batch | → chassis | → **fault class** | fault baseline |")
    L.append("|---|---|---|---|---|---|---|")
    for sub, d in p["subsets"].items():
        L.append(f"| {sub} | {d['n']} | {fmt(d['session']['loo_1nn_accuracy'],3)} "
                 f"| {fmt(d['batch']['loo_1nn_accuracy'],3)} "
                 f"| {fmt(d['chassis']['loo_1nn_accuracy'],3)} "
                 f"| **{fmt(d['fault_class']['loo_1nn_accuracy'],3)}** "
                 f"| {fmt(d['fault_class']['majority_baseline'],3)} |")
    L.append("")
    sob = p["subsets"].get("1000W_1500W_only", {}).get("fault_class", {})
    L.append(f"**Session-only baseline = {fmt(sob.get('loo_1nn_accuracy'),3)}** "
             f"(fault class, 1000/1500 W, from a scalar that cannot contain winding "
             f"information). Any fault-class score that does not clearly beat this is "
             f"**not demonstrating winding diagnosis**.\n")

    # -- V1 vs V5 --------------------------------------------------------
    L.append("### V1 vs V5 — the gap is the result\n")
    L.append("| Protocol | Scheme | Groups scored | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|---|")
    for key, label in (("V1", "leave-one-motor-out"), ("V5", "leave-one-session-out")):
        d = r[key]["pooled"]
        L.append(f"| **{key}** | {label} | {r[key]['n_groups_scored']} "
                 f"| {fmt(d['accuracy'])} | {fmt(d['macro_f1'])} "
                 f"| {fmt(d['majority_baseline'])} |")
    v3 = r["V3"]["pooled"]
    L.append(f"| V3 **(leaky reference)** | shuffled windows | — "
             f"| {fmt(v3['accuracy'])} | {fmt(v3['macro_f1'])} "
             f"| {fmt(v3['majority_baseline'])} |")
    L.append(f"| — | **session-only baseline** | — | {fmt(sob.get('loo_1nn_accuracy'),3)} "
             f"| — | {fmt(sob.get('majority_baseline'),3)} |")
    L.append("")

    for key, label in (("V1", "V1 — leave-one-motor-out"),
                       ("V5", "V5 — leave-one-session-out")):
        L.append(f"#### {label}\n")
        if r[key].get("excluded_reason"):
            L.append(f"> {r[key]['excluded_reason']}\n")
        L.append("| Fold | Held out | Accuracy | Macro-F1 | Baseline | Train windows | Test windows |")
        L.append("|---|---|---|---|---|---|---|")
        for f in r[key]["folds"]:
            if f.get("excluded"):
                L.append(f"| {f['fold']} | {', '.join(f['held_out'])} | — | — | — | — | "
                         f"**EXCLUDED**: {f['reason']} |")
            else:
                L.append(f"| {f['fold']} | {', '.join(f['held_out'])} "
                         f"| {fmt(f['accuracy'])} | {fmt(f['macro_f1'])} "
                         f"| {fmt(f['majority_baseline'])} "
                         f"| {f['train_windows']} | {f['test_windows']} |")
        L.append("")

    # -- fusion ----------------------------------------------------------
    fu = r["fusion"]
    L.append("### Fusion authority\n")
    L.append(f"| Field | Value |")
    L.append(f"|---|---|")
    L.append(f"| Protocol used | {fu['protocol_used']} |")
    L.append(f"| Validation groups scored | {fu['n_validation_groups']} "
             f"(floor: {fu['floor']['min_validation_groups']}) |")
    L.append(f"| Honest macro-F1 | {fmt(fu['honest_macro_f1'])} "
             f"(floor: {fu['floor']['min_macro_f1']}) |")
    L.append(f"| Floor fixed at | {fu['floor']['fixed_at']}, commit "
             f"`{fu['floor']['fixed_at_commit'][:8]}` |")
    L.append(f"| **Status** | **{fu['status']}** |")
    L.append("")
    L.append(f"{fu['reason']} The floor was pre-registered before any multi-stage "
             f"branch existed and has not been adjusted.\n")
    return L


def main():
    lines = ["# DriveSentinel v2 — multi-stage results",
             "",
             "Generated by `scripts/60_render_multistage_results.py` from the JSON in",
             "`artifacts/multistage/`. Do not edit by hand.",
             "",
             "Every number carries its protocol. Leaky numbers are labelled",
             "**(leaky reference)** in the heading as well as the body.",
             "", "---", ""]

    for slug, code, stage, dataset in BRANCHES:
        path = os.path.join(C.ARTIFACT_DIR, "multistage", slug, f"{slug}_results.json")
        lines.append(f"## {code} `{slug}` — stage {stage}, {dataset}")
        lines.append("")
        if not os.path.exists(path):
            lines += ["**NOT RUN**", ""]
            continue
        with open(path) as fh:
            r = json.load(fh)
        if slug == "winding":
            lines += render_winding(r)
            cnn_path = os.path.join(C.ARTIFACT_DIR, "multistage", "winding",
                                    "winding_cnn_results.json")
            if os.path.exists(cnn_path):
                with open(cnn_path) as fh:
                    lines += render_cnn_comparison(json.load(fh), r, fmt)
        elif slug == "supply":
            lines += render_supply(r, fmt)
        elif slug == "inverter_telemetry":
            lines += render_inverter(r, fmt)
        else:
            lines += ["**NOT RUN**", ""]
        lines.append(f"_Source: `{os.path.relpath(path, C.PROJECT_ROOT)}`, "
                     f"{r.get('seconds', '?')} s._")
        lines.append("")
        lines.append("---")
        lines.append("")

    # -- fusion: thresholds and authority, generated from the same JSON ----
    lines.append("## Fusion")
    lines.append("")
    fa = C.FUSION_CONFIG["fault_authority"]
    lines.append(f"Pre-registered **{fa['fixed_at']}**, commit "
                 f"`{fa['fixed_at_commit'][:8]}`, before any multi-stage branch "
                 f"existed. Not adjusted since.")
    lines.append("")
    lines.append("### Thresholds")
    lines.append("")
    lines.append("| Parameter | Value | Meaning |")
    lines.append("|---|---|---|")
    fc = C.FUSION_CONFIG
    for k, meaning in (
            ("rolling_window", "windows in the rolling probability mean"),
            ("tau_fault", "p_fault at or above this counts towards Fault"),
            ("tau_warning", "p_fault at or above this counts towards Warning"),
            ("hysteresis", "clearing needs p_fault below tau_warning minus this"),
            ("k_consecutive", "consecutive updates required to change status")):
        lines.append(f"| `{k}` | {fc[k]} | {meaning} |")
    lines.append(f"| `min_validation_groups` | {fa['min_validation_groups']} "
                 f"| independent validation groups required for Fault authority |")
    lines.append(f"| `min_macro_f1` | {fa['min_macro_f1']} "
                 f"| honest macro-F1 required for Fault authority |")
    lines.append("")
    lines.append("Raising takes `k_consecutive` updates; clearing takes roughly")
    lines.append("`rolling_window + k_consecutive`, because `p_fault` is a rolling mean")
    lines.append("and the high samples must flush out first. The asymmetry is deliberate.")
    lines.append("")
    lines.append("### Branch authority")
    lines.append("")
    lines.append("Read from each branch's results JSON at load time, never hardcoded.")
    lines.append("")
    lines.append("| Branch | Stage | Protocol | Groups | Macro-F1 | Groups ≥ "
                 f"{fa['min_validation_groups']} | F1 ≥ {fa['min_macro_f1']} | Tier |")
    lines.append("|---|---|---|---|---|---|---|---|")
    tick = {True: "yes", False: "**no**", None: "—"}
    for r in FU.authority_table(FU.load_branch_metrics()):
        f1 = "—" if r["macro_f1"] is None else f"{r['macro_f1']:.4f}"
        lines.append(f"| `{r['branch']}` | {r['stage']} | {r['protocol'] or '—'} "
                     f"| {r['n_validation_groups'] if r['n_validation_groups'] is not None else '—'} "
                     f"| {f1} | {tick[r['groups_test']]} | {tick[r['metric_test']]} "
                     f"| **{r['tier']}** |")
    lines.append("")
    lines.append("An **INDICATIVE** branch may raise `Warning` and contribute to the")
    lines.append("evidence string, but never `Fault`, however confident it is. Its")
    lines.append("waveform and spectrum panels stay live — capped, not hidden.")
    lines.append("A **NOT MEASURED** branch has no authority at all.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Trip gating")
    lines.append("")
    lines.append(f"**Gating is `enabled: {C.TRIP_CONFIG['enabled']}`.** No real dataset in")
    lines.append("the roster has a speed ramp — Paderborn is fixed at 900/1500 rpm, KAIST")
    lines.append("at 200.00 Hz, Bacha at 10 rad/s, Thomas mains-fed at 49.96–50.04 Hz.")
    lines.append("Segmentation on any of them returns `cruise` for every window, so gating")
    lines.append("would change nothing. It is implemented and unit-tested against synthetic")
    lines.append("ramps with known ground truth, and left off until there is a speed profile")
    lines.append("to gate. **No claim is made that it gates anything on real data.**")
    lines.append("")
    lines.append("### Order resolution vs shaft speed")
    lines.append("")
    lines.append("Analytic, not measured. `delta_order = 60 / (T * rpm)`.")
    lines.append("")
    lines.append("| Shaft rpm | f_shaft (Hz) | revs/window | Order resolution | Source |")
    lines.append("|---|---|---|---|---|")
    for r in TR.resolution_table():
        src = ("Paderborn, **measured**" if r["shaft_rpm"] in (900.0, 1500.0)
               else "**extrapolation**" + (" — gearless sheave range"
                                           if r["shaft_rpm"] <= 50 else ""))
        lines.append(f"| {r['shaft_rpm']:.0f} | {r['f_shaft_hz']:.2f} "
                     f"| {r['shaft_revs_in_window']:.2f} "
                     f"| {r['order_resolution']:.4f} | {src} |")
    gap = C.FAULT_ORDERS_NOMINAL["bpfi"] - C.FAULT_ORDERS_NOMINAL["bpfo"]
    lines.append("")
    lines.append(f"BPFO ({C.FAULT_ORDERS_NOMINAL['bpfo']:.2f}) and BPFI "
                 f"({C.FAULT_ORDERS_NOMINAL['bpfi']:.2f}) are **{gap:.2f} orders** apart.")
    lines.append(f"At 900 rpm a {C.WINDOW_SECONDS:g} s window resolves "
                 f"{TR.order_resolution(900.0)['order_resolution']:.4f} orders — ample.")
    lines.append(f"At 20 rpm it resolves "
                 f"{TR.order_resolution(20.0)['order_resolution']:.2f} orders, wider than")
    lines.append("the gap, so the two lines merge. The fix is a longer window, not a")
    lines.append("cleverer algorithm: resolution is 1/T, and holding 0.067 orders at 20 rpm")
    lines.append("needs a 45 s window — longer than many elevator trips. That trade is the")
    lines.append("real constraint on porting this to a sheave, and it is arithmetic.")
    lines.append("")
    lines.append("---")
    lines.append("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
