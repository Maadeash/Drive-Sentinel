"""
Render artifacts/runs/lobo_summary.json into docs/results.md.

Keeps the written-up numbers mechanically tied to the run that produced them,
so the document cannot drift from the JSON the way a hand-typed table does.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivesentinel import config as C


def render(summary: dict, export_report: dict = None) -> str:
    p = summary["pooled"]
    labels = summary["labels"]
    L = []

    L.append("# DriveSentinel v5 — results\n")
    L.append("Leave-one-bearing-out over all 29 bearings, pooled. Every bearing is")
    L.append("held out exactly once, so concatenating the folds gives one")
    L.append("out-of-sample prediction per window across the whole dataset.\n")
    L.append("> Per-**fold** macro-F1 and per-fold baselines are omitted deliberately:")
    L.append("> a single-bearing test set contains one class, so its macro-F1 caps at")
    L.append("> 1/3 and its majority baseline is always 1.0. Only the pooled set has")
    L.append("> all three classes in it.\n")

    L.append("## Headline\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Window accuracy | **{p['window_acc']:.4f}** |")
    L.append(f"| Macro-F1 | **{p['macro_f1']:.4f}** |")
    L.append(f"| Per-recording accuracy (majority vote) | **{p['recording_acc']:.4f}** |")
    L.append(f"| Majority-class baseline | {p['majority_baseline']:.4f} |")
    L.append(f"| Windows / recordings | {p['n_windows']:,} / {p['n_recordings']:,} |")
    L.append(f"| Per-bearing accuracy | {summary['per_bearing_acc_mean']:.4f} "
             f"± {summary['per_bearing_acc_std']:.4f} "
             f"(SEM {summary['per_bearing_acc_sem']:.4f}, n={summary['n_folds']}) |")
    L.append(f"| Sweep time | {summary['seconds'] / 60:.1f} min |\n")

    L.append("## Confusion matrix\n")
    L.append("Rows are truth, columns are prediction.\n")
    L.append("| | " + " | ".join(labels) + " | recall |")
    L.append("|---|" + "---|" * (len(labels) + 1))
    for lab, row in zip(labels, summary["confusion"]):
        total = sum(row)
        recall = row[labels.index(lab)] / total if total else 0.0
        L.append(f"| **{lab}** | " + " | ".join(f"{v:,}" for v in row) +
                 f" | {recall:.3f} |")
    L.append("")

    L.append("## By damage origin\n")
    L.append("Never pooled into one figure — v4 measured artificial→real transfer at")
    L.append("37.9 %. Healthy bearings join both groups as the reference class.\n")
    L.append("| Population | Bearings | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|")
    for k, v in summary["by_damage_origin"].items():
        L.append(f"| {k} | {v['n_bearings']} | {v['window_acc']:.4f} | "
                 f"{v['macro_f1']:.4f} | {v['majority_baseline']:.4f} |")
    L.append("")

    L.append("## Per-bearing\n")
    L.append("| Bearing | Class | Origin | Accuracy | Recording acc | Epochs |")
    L.append("|---|---|---|---|---|---|")
    for f in sorted(summary["folds"], key=lambda r: r["window_acc"]):
        L.append(f"| `{f['test_bearing']}` | {f['test_label']} | {f['origin']} | "
                 f"{f['window_acc']:.4f} | {f['recording_acc']:.4f} | "
                 f"{f['best_epoch']} |")
    L.append("")

    if summary.get("aggregation_curve"):
        L.append("## Evidence accumulation\n")
        L.append("Accuracy when N four-second observations of the same bearing are")
        L.append("pooled by averaging their per-window softmax. Still leave-one-")
        L.append("bearing-out — every probability is out-of-sample. A lift makes dozens")
        L.append("of trips a day, so a health verdict does not have to be called from a")
        L.append("single look at the machine.\n")
        L.append("| Observations pooled | Accuracy |")
        L.append("|---|---|")
        for n, acc in summary["aggregation_curve"].items():
            L.append(f"| {n} | {acc:.4f} |")
        L.append("")

    if export_report:
        L.append("## INT8 export\n")
        L.append(f"- Weights: {export_report['n_weights']:,} int8, "
                 f"{export_report['n_biases']:,} int32 biases")
        if "agreement" in export_report:
            a = export_report["agreement"]
            L.append(f"- INT8 vs float argmax agreement: "
                     f"**{a['argmax_agreement'] * 100:.2f} %** "
                     f"(bar: ≥ 98 %), n={a['n']:,}")
            L.append(f"- Max |logit| difference: {a['max_abs_logit_diff']:.4f}")
        L.append("\n| Layer | Terms | Peak \\|acc\\| | Bits | int32 |")
        L.append("|---|---|---|---|---|")
        for name, acc in export_report["accumulator"].items():
            L.append(f"| {name} | {acc['terms']} | {acc['peak_abs']:,} | "
                     f"{acc['bits_needed']} | {'ok' if acc['fits_int32'] else 'OVERFLOW'} |")
        L.append("")

    L.append("---\n")
    L.append(f"Generated from `artifacts/runs/lobo_summary.json`. "
             f"DSP contract: {C.summary()}")
    return "\n".join(L) + "\n"


def main() -> None:
    summary_path = os.path.join(C.RUN_DIR, "lobo_summary.json")
    if not os.path.exists(summary_path):
        sys.exit(f"FATAL: {summary_path} not found -- run scripts/02_train_lobo.py")
    with open(summary_path) as fh:
        summary = json.load(fh)

    export_report = None
    vpath = os.path.join(C.EXPORT_DIR, "verification.json")
    if os.path.exists(vpath):
        with open(vpath) as fh:
            export_report = json.load(fh)

    docs = os.path.join(C.PROJECT_ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    out = os.path.join(docs, "results.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(summary, export_report))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
