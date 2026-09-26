"""Inverter-telemetry section for docs/results_multistage.md."""


def _conf_table(labels, matrix):
    L = ["| true \\ pred | " + " | ".join(f"`{c}`" for c in labels) + " |",
         "|---" * (len(labels) + 1) + "|"]
    for c, row in zip(labels, matrix):
        L.append(f"| **`{c}`** | " + " | ".join(str(v) for v in row) + " |")
    return L


def render_inverter(r, fmt):
    L = []
    L.append(f"**Task: {r['task']}** (4-class family). {r['n_windows']:,} windows, "
             f"{r['window_s']} s / {r['hop_s']} s hop, purge "
             f"{r['purge_windows']} windows.")
    L.append("")

    L.append("### The confound, stated before any number")
    L.append("")
    L.append(f"> {r['confound']}")
    L.append("")

    # -- V3 leads -----------------------------------------------------------
    v3 = r["V3_electrical_only"]
    b3 = v3["block_split"]
    L.append("### V3 — electrical-only ablation (headline)")
    L.append("")
    L.append(f"> {v3['question']}")
    L.append("")
    L.append(f"Temperature channels removed: {v3['n_features']} features from "
             f"`Ia`/`Ib` alone.")
    L.append("")
    L.append("| Split | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|")
    L.append(f"| contiguous block | **{fmt(b3['accuracy'])}** "
             f"| **{fmt(b3['macro_f1'])}** | {fmt(b3['majority_baseline'])} |")
    s3 = v3["shuffled_LEAKY"]
    L.append(f"| random **(leaky reference)** | {fmt(s3['accuracy'])} "
             f"| {fmt(s3['macro_f1'])} | {fmt(s3['majority_baseline'])} |")
    L.append("")
    L.append("Per-class F1 under the block split:")
    L.append("")
    L.append("| Class | F1 | Test windows |")
    L.append("|---|---|---|")
    for c in b3["labels"]:
        L.append(f"| `{c}` | {fmt(b3['per_class_f1'][c], 3)} | {b3['support'][c]} |")
    L.append("")

    ce = r.get("confound_evidence")
    if ce:
        L.append("#### What that table actually shows")
        L.append("")
        L.append(f"> {ce['statement']}")
        L.append("")
        L.append(f"> {ce['open_circuit_note']}")
        L.append("")

    v1 = r["V1_block_split"]["scores"]
    v2 = r["V2_shuffled_LEAKY"]["scores"]

    rid = r.get("run_identification_control")
    if rid:
        L.append("### The run-identification control — the inference, measured")
        L.append("")
        L.append(f"> {rid['question']}")
        L.append("")
        L.append("| Predicting | Features | Accuracy | Macro-F1 | Baseline |")
        L.append("|---|---|---|---|---|")
        L.append(f"| **which run** | with temperature | **{fmt(rid['with_temperature']['accuracy'])}** "
                 f"| {fmt(rid['with_temperature']['macro_f1'])} "
                 f"| {fmt(rid['with_temperature']['majority_baseline'])} |")
        L.append(f"| **which run** | electrical only | **{fmt(rid['electrical_only']['accuracy'])}** "
                 f"| {fmt(rid['electrical_only']['macro_f1'])} "
                 f"| {fmt(rid['electrical_only']['majority_baseline'])} |")
        L.append("")
        L.append("Set beside the condition scores on the same split and the same features:")
        L.append("")
        L.append("| Features | 4-class condition | **which run** | Gap |")
        L.append("|---|---|---|---|")
        L.append(f"| with temperature | {fmt(v1['accuracy'])} "
                 f"| {fmt(rid['with_temperature']['accuracy'])} "
                 f"| {v1['accuracy'] - rid['with_temperature']['accuracy']:+.4f} |")
        L.append(f"| electrical only | {fmt(b3['accuracy'])} "
                 f"| {fmt(rid['electrical_only']['accuracy'])} "
                 f"| {b3['accuracy'] - rid['electrical_only']['accuracy']:+.4f} |")
        L.append("")
        L.append(f"> {rid['interpretation']}")
        L.append("")
        L.append("**This is the measurement, not the inference.** The earlier "
                 "`over_temp` observation — F1 0.796 with no temperature sensor — "
                 "argued that the model must be reading run identity. This trains "
                 "the same features to predict run identity directly and finds it "
                 f"recoverable at {fmt(rid['with_temperature']['accuracy'])}. The "
                 "4-class number and the run-identity number are within a few points "
                 "of each other on both feature sets, which is what you would expect "
                 "if they are largely the same quantity.")
        L.append("")

    # -- V1 / V2 ------------------------------------------------------------
    L.append("### V1 and V2 — with temperature")
    L.append("")
    L.append("| Protocol | Split | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|")
    L.append(f"| **V1** | contiguous block, 70/30 per run | **{fmt(v1['accuracy'])}** "
             f"| **{fmt(v1['macro_f1'])}** | {fmt(v1['majority_baseline'])} |")
    L.append(f"| V2 **(leaky reference)** | random windows | {fmt(v2['accuracy'])} "
             f"| {fmt(v2['macro_f1'])} | {fmt(v2['majority_baseline'])} |")
    L.append("")
    L.append(f"> {r['temperature_note']}")
    L.append("")
    L.append(f"V1 and V2 agree to {fmt(abs(v1['accuracy'] - v2['accuracy']), 4)} — "
             f"the leaky split gains nothing, because with temperature present the "
             f"task is already saturated. That is not a sign the block split is "
             f"safe; it is a sign the task is trivial once a thermometer is in the "
             f"feature set.")
    L.append("")

    # -- 9-class, qualitative ----------------------------------------------
    loc = r["location_9class_QUALITATIVE_ONLY"]
    L.append("### 9-class location view — qualitative only, NOT a metric")
    L.append("")
    L.append(f"> {loc['reason']}")
    L.append("")
    L.append("Test-window support per class: " +
             ", ".join(f"`{k}` {v}" for k, v in loc["support"].items()) + ".")
    L.append("")
    L += _conf_table(loc["labels"], loc["confusion"])
    L.append("")
    L.append("**No per-class number is reported from this matrix.**")
    L.append("")

    # -- dropped channels and calibration ----------------------------------
    L.append("### Channels dropped, and why the numbers survive a bad calibration")
    L.append("")
    L.append(f"Dropped: {', '.join('`' + c + '`' for c in r['dropped_channels'])}. "
             f"{r['dropped_reason']}")
    L.append("")
    L.append(f"> {r['calibration_independence']}")
    L.append("")

    # -- fusion -------------------------------------------------------------
    fu = r["fusion"]
    L.append("### Fusion authority")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| Protocol used | {fu['protocol_used']} |")
    L.append(f"| Validation groups | **{fu['n_validation_groups']}** "
             f"(floor: {fu['floor']['min_validation_groups']}) |")
    L.append(f"| Honest macro-F1 | {fmt(fu['honest_macro_f1'])} "
             f"(floor: {fu['floor']['min_macro_f1']}) |")
    L.append(f"| **Status** | **{fu['status']}** |")
    L.append("")
    L.append(f"{fu['reason']}")
    L.append("")
    return L
