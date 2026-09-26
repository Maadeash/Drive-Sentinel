"""
Supply-branch section for docs/results_multistage.md.

Split out of 60_render_multistage_results.py only because it is long; it is
imported by that script and is not a separate entry point.
"""


def render_supply(r, fmt):
    L = []
    L.append(f"**Task: {r['task']}.** Deliverable: **{r['deliverable']}**.")
    L.append("")
    L.append(f"> {r['rule_statement']}")
    L.append("")
    L.append("### Why a rule and not a learned model")
    L.append("")
    L.append(f"> {r['why_not_learned']}")
    L.append("")

    L.append("### R1 — the rule, every recording")
    L.append("")
    L.append("| File | Motor | Scenario | True | Rule verdict | Latency |")
    L.append("|---|---|---|---|---|---|")
    for rec in r["R1_rule_per_recording"]["recordings"]:
        lat = rec.get("latency_s")
        ok = "✅" if rec["verdict"] == rec["true_label"] else "❌"
        lat_txt = "—" if lat is None else f"{lat:.2f} s"
        L.append(f"| `{rec['file']}` | {rec['motor']} | {rec['scenario']} "
                 f"| {rec['true_label']} | {ok} {rec['verdict']} | {lat_txt} |")
    sc = r["R1_rule_per_recording"]["scores"]
    L.append("")
    L.append(f"**{round(sc['accuracy'] * sc['n'])}/{sc['n']} recordings correct** — "
             f"accuracy {fmt(sc['accuracy'])}, macro-F1 {fmt(sc['macro_f1'])}, "
             f"baseline {fmt(sc['majority_baseline'])}.")
    L.append("")
    lat = r["detection_latency"]
    L.append(f"Detection latency is measured against an amplitude crossing "
             f"independent of the rule's own threshold, and cannot be better than "
             f"the {lat['hop_s']} s hop.")
    L.append("")

    L.append("### R2 vs L1 vs L3 — the protocol is the whole story")
    L.append("")
    L.append("| Protocol | Scheme | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|")
    p2 = r["R2_rule_leave_one_motor_out"]["pooled"]
    L.append(f"| **R2 rule** | leave-one-motor-out | **{fmt(p2['accuracy'])}** "
             f"| **{fmt(p2['macro_f1'])}** | {fmt(p2['majority_baseline'])} |")
    l1 = r["L1_learned_leave_one_motor_out"]["pooled"]
    L.append(f"| L1 learned | leave-one-motor-out | {fmt(l1['accuracy'])} "
             f"| {fmt(l1['macro_f1'])} | {fmt(l1['majority_baseline'])} |")
    l3 = r["L3_learned_shuffled_LEAKY"]["pooled"]
    L.append(f"| L3 learned **(leaky reference)** | shuffled windows "
             f"| {fmt(l3['accuracy'])} | {fmt(l3['macro_f1'])} "
             f"| {fmt(l3['majority_baseline'])} |")
    L.append("")
    gap = (l3["accuracy"] - l1["accuracy"]) * 100
    L.append(f"The learned model drops from {fmt(l3['accuracy'])} under a leaky "
             f"split to {fmt(l1['accuracy'])} under leave-one-motor-out — a gap of "
             f"**{gap:.0f} points** on the same features. The rule, which has "
             f"nothing fitted to any recording, is unaffected by the split.")
    L.append("")

    n1 = r["N1_bearing_confound"]
    L.append("### N1 — bearing confound, a documented NEGATIVE RESULT")
    L.append("")
    L.append(f"Predicting **motor identity** from the same features scores "
             f"**{fmt(n1['scores']['accuracy'])}** against a baseline of "
             f"{fmt(n1['scores']['majority_baseline'])}.")
    L.append("")
    L.append(f"> {n1['statement']}")
    L.append("")

    lr = r["label_reconstruction"]
    L.append("### Label vector — reconstructed, not validated, not used")
    L.append("")
    L.append(f"The paper's 1000-sample / step-500 reconstruction was tested against "
             f"the measured phase-current collapse boundaries: "
             f"**{lr['file_boundary_hits']}** label-run boundaries coincide with a "
             f"file boundary at {lr['windows_per_file']} windows per file, and "
             f"**{lr['event_boundary_hits']}** coincide with a measured phase-loss "
             f"event.")
    L.append("")
    L.append("The file structure reconstructs; the within-file event structure does "
             "not. **Labels come from the collapse rule instead**, which was the "
             "primary source either way.")
    L.append("")

    fu = r["fusion"]
    L.append("### Fusion authority")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| Protocol used | {fu['protocol_used']} |")
    L.append(f"| Validation groups | {fu['n_validation_groups']} "
             f"(floor: {fu['floor']['min_validation_groups']}) |")
    L.append(f"| Honest macro-F1 | {fmt(fu['honest_macro_f1'])} "
             f"(floor: {fu['floor']['min_macro_f1']}) |")
    L.append(f"| **Status** | **{fu['status']}** |")
    L.append("")
    L.append(f"**A perfect macro-F1 of {fmt(fu['honest_macro_f1'])} still does not "
             f"earn `Fault` authority**, because two motors is below the "
             f"pre-registered minimum of {fu['floor']['min_validation_groups']} "
             f"validation groups. The floor was fixed before this branch existed and "
             f"has not been adjusted. This is the floor working as intended: a "
             f"branch can be perfectly right and still not be trusted alone.")
    L.append("")
    return L


def render_cnn_comparison(cnn, gbm, fmt):
    """
    CNN vs gradient boosting on B-S4, same task and same splits.

    Lives here rather than in the winding renderer because it is a POST-HOC
    experiment, not part of the branch's declared protocol set. It is rendered
    only when winding_cnn_results.json exists.
    """
    L = []
    v = cnn["verdict"]
    L.append("### Post-hoc experiment — CNN vs gradient boosting")
    L.append("")
    L.append(f"**{cnn['status']}.** Recipe and acceptance criteria were committed "
             f"before the run (`scripts/experiments/d2_cnn.py`, commit `05cf8d7`). "
             f"One recipe, no sweep, {len(cnn['recipe']['seeds'])} seeds.")
    L.append("")
    L.append("| Protocol | CNN (mean ± std over seeds) | Gradient boosting | Baseline |")
    L.append("|---|---|---|---|")
    rows = [("V1 leave-one-motor-out", "V1_leave_one_motor_out",
             gbm["V1"]["pooled"]["accuracy"], gbm["V1"]["pooled"]["majority_baseline"]),
            ("**V5 leave-one-session-out**", "V5_leave_one_session_out",
             gbm["V5"]["pooled"]["accuracy"], gbm["V5"]["pooled"]["majority_baseline"]),
            ("V3 shuffled **(leaky reference)**", "V3_shuffled_LEAKY",
             gbm["V3"]["pooled"]["accuracy"], gbm["V3"]["pooled"]["majority_baseline"])]
    for name, key, g, base in rows:
        a = cnn[key]["accuracy"]
        L.append(f"| {name} | {fmt(a['mean'])} ± {fmt(a['std'], 4)} | {fmt(g)} "
                 f"| {fmt(base)} |")
    probe = cnn["acceptance"]["session_only_baseline"]
    L.append(f"| — | **session-only baseline** | {fmt(probe, 3)} | 0.467 |")
    L.append("")
    L.append("#### Acceptance criteria, as declared")
    L.append("")
    L.append("| Criterion | Threshold | Result | Outcome |")
    L.append("|---|---|---|---|")
    L.append(f"| 1. V5 beats the session-only baseline | > {fmt(probe, 3)} "
             f"| {fmt(v['v5_accuracy_mean'])} "
             f"| {'**PASS**' if v['criterion_1_beats_session_baseline'] else '**FAIL**'} |")
    L.append(f"| 2. V5 beats the GBM by more than the seed spread "
             f"| > {fmt(cnn['acceptance']['gbm_v5_accuracy'])} + "
             f"{fmt(v['seed_spread'], 4)} | margin {v['criterion_2_margin']:+.4f} "
             f"| {'**PASS**' if v['criterion_2_beats_gbm_by_more_than_seed_spread'] else '**FAIL**'} |")
    L.append("")
    # The V3 observation is the reason this negative result earns slide space.
    c3 = cnn["V3_shuffled_LEAKY"]["accuracy"]["mean"]
    g3v = gbm["V3"]["pooled"]["accuracy"]
    L.append("#### A fourth independent line on the D2 confound")
    L.append("")
    L.append(f"On the **leaky** shuffled split the CNN scores {fmt(c3)} against the "
             f"GBM's {fmt(g3v)} — **{100*(g3v-c3):.1f} points lower**, on the same "
             f"windows and the same task.")
    L.append("")
    L.append("A leaky split rewards memorising individual recordings. The GBM, with "
             "23 scalars it can combine freely, memorises them almost perfectly. A "
             "convolutional model of comparable capacity, constrained to look for "
             "*shape* in the spectrum, cannot reach the same score — because the "
             "thing being rewarded is not spectral shape, it is per-recording "
             "identity.")
    L.append("")
    L.append("So the GBM's 0.9990 was **largely per-recording memorisation**, and the "
             "CNN's inability to match it is evidence of that rather than a weakness. "
             "This is a fourth independent line supporting the same finding:")
    L.append("")
    L.append("| # | Line of evidence | Where |")
    L.append("|---|---|---|")
    L.append("| 1 | DAQ chassis and module split the 3000 W motor's fault types | `data_notes_d2.md` §7A |")
    L.append("| 2 | Probe polarity forms three patterns that track session | `data_notes_d2.md` §6.2 |")
    L.append("| 3 | Session-clustered residual recovers the session at 1.000 | `data_notes_d2.md` §7B |")
    L.append("| 4 | **The leaky score collapses when the model cannot memorise recordings** | **this section** |")
    L.append("")
    L.append(f"**Outcome: {v['outcome']}.** Shipped branch: "
             f"**{v['shipped_branch']}**.")
    L.append("")
    L.append(f"{v['tier_unchanged']}")
    L.append("")
    return L
