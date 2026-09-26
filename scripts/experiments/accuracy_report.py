"""
Collect every experiment run so far into docs/accuracy.md.

Reads whatever exists in artifacts/runs/ -- sweep.json (v1 recipes), sweep2.json
(v2 feature set), sweep3.json (v1 + ensembling), task_variants.json (binary and
the leaky shuffled reference) -- and renders one table per question asked, so
the answer to "can we reach 85-90 %?" is visible in one place rather than
scattered across logs.

Deliberately does NOT pick a winner by sorting on accuracy alone.  With a
per-bearing standard deviation near 0.36 over 29 bearings, the standard error on
any of these numbers is about 0.067, and cuDNN's non-deterministic convolution
backward means re-running the identical config moves the result by several
points.  Differences smaller than roughly 8 points are not evidence.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C

RUNS = C.RUN_DIR


def load(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def sem(summary):
    return summary.get("per_bearing_acc_sem", float("nan"))


def row(name, s, extra=""):
    p = s["pooled"]
    return (f"| {name} | {p['window_acc']:.4f} | {p['macro_f1']:.4f} | "
            f"{p['recording_acc']:.4f} | ±{sem(s):.4f} | {extra} |")


def main() -> None:
    sweep1 = load("sweep.json") or {}
    sweep2 = load("sweep2.json") or {}
    sweep3 = load("sweep3.json") or {}
    variants = load("task_variants.json") or {}

    L = ["# DriveSentinel v5 — how far accuracy actually goes\n"]
    L.append("Every number here is **leave-one-bearing-out over all 29 bearings, "
             "pooled**, unless a row says otherwise. Baseline (always predict the "
             "majority class) is **0.4153** for the 3-class task.\n")

    L.append("## The headline\n")
    best = sweep3.get("sched20_ens3_tta")
    if best:
        p = best["pooled"]
        L.append(f"| | Window | Macro-F1 | Per-recording |")
        L.append("|---|---|---|---|")
        L.append(f"| **Best honest 3-class** | **{p['window_acc']:.4f}** | "
                 f"**{p['macro_f1']:.4f}** | **{p['recording_acc']:.4f}** |")
        L.append(f"| Majority baseline | {p['majority_baseline']:.4f} | — | — |")
        L.append("")
    if variants.get("binary_lobo"):
        b = variants["binary_lobo"]
        p = b["pooled"]
        margin = p["window_acc"] - p["majority_baseline"]
        L.append("## Binary healthy vs damaged — reads as a win, mostly isn't\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Window accuracy | {p['window_acc']:.4f} |")
        L.append(f"| **Majority baseline** | **{p['majority_baseline']:.4f}** |")
        L.append(f"| Margin over baseline | **{margin:+.4f}** |")
        L.append(f"| Macro-F1 | {p['macro_f1']:.4f} |")
        L.append(f"| Per-recording | {p['recording_acc']:.4f} |")
        L.append("")
        L.append("Only 6 of 29 bearings are healthy, so *always predicting damaged*")
        L.append(f"already scores {p['majority_baseline']:.4f}. The model beats that by")
        L.append(f"{margin * 100:.1f} points on accuracy — which is why accuracy is the")
        L.append("wrong headline for this framing. Macro-F1 of "
                 f"{p['macro_f1']:.4f} against roughly 0.44 for the")
        L.append("always-damaged predictor is the honest evidence that it has learned")
        L.append("something; the accuracy figure mostly reflects class imbalance.\n")
        if b.get("aggregation_curve"):
            L.append("Pooling observations does help here, unlike the 3-class task:\n")
            L.append("| Observations pooled | Accuracy |")
            L.append("|---|---|")
            for n, acc in b["aggregation_curve"].items():
                L.append(f"| {n} | {acc:.4f} |")
            L.append("")
            top = list(b["aggregation_curve"].values())[-1]
            L.append(f"At N=40 it reaches {top:.4f}, genuinely above the "
                     f"{p['majority_baseline']:.4f} baseline — so the binary signal is")
            L.append("real, just weak per observation.\n")

    L.append("## Why the numbers move between runs\n")
    L.append("Per-bearing accuracy has a standard deviation near 0.36 across 29")
    L.append("bearings, so the standard error on any pooled figure is about 0.067.")
    L.append("cuDNN's convolution backward is non-deterministic, so re-running an")
    L.append("identical config moves the result by several points on its own: the")
    L.append("`v50_exact` recipe scored 0.7773 once and 0.6972 on a later identical")
    L.append("run. **Differences smaller than roughly 8 points are not evidence.**")
    L.append("Only variance-reducing changes — seed ensembling, TTA — hold up on")
    L.append("re-running, which is why the shipped recipe uses them.\n")

    if sweep1 or sweep3:
        L.append("## Training recipe (2-channel input)\n")
        L.append("| Recipe | Window | Macro-F1 | Per-recording | SEM | Notes |")
        L.append("|---|---|---|---|---|---|")
        for name, s in sorted(sweep3.items(),
                              key=lambda kv: -kv[1]["pooled"]["window_acc"]):
            note = ("3 seeds + TTA — variance reduction, the only change that held"
                    if "ens3" in name else "single model")
            L.append(row(name, s, note))
        for name, s in sorted(sweep1.items(),
                              key=lambda kv: -kv[1]["pooled"]["window_acc"]):
            L.append(row(name + " (earlier run)", s, "single model"))
        L.append("")

    if sweep2:
        L.append("## Input representation: 5 channels vs 2 — rejected\n")
        L.append("The v2 cache added a current raw-order spectrum, split the")
        L.append("vibration envelope into low and high resonance bands, and added a")
        L.append("vibration raw log-frequency spectrum. On the identical recipe it")
        L.append("was **worse**, and a bearing-level nearest-centroid check found no")
        L.append("improvement either (16/29 correct with both).\n")
        L.append("| Recipe | Window | Macro-F1 | Per-recording | SEM | Feature set |")
        L.append("|---|---|---|---|---|---|")
        for name, s in sorted(sweep2.items(),
                              key=lambda kv: -kv[1]["pooled"]["window_acc"]):
            L.append(row(name, s, s.get("feature_set", "v2") + " (5 channels)"))
        L.append("")
        L.append("More input channels gave the model more to overfit without giving")
        L.append("it more to learn from. The 2-channel contract ships.\n")

    curve = (sweep3.get("sched20_ens3_tta") or {}).get("aggregation_curve")
    if curve:
        L.append("## Evidence accumulation does not rescue it\n")
        L.append("Pooling N four-second observations of the same bearing by averaging")
        L.append("their softmax. Still strictly leave-one-bearing-out.\n")
        L.append("| Observations pooled | Accuracy |")
        L.append("|---|---|")
        for n, acc in curve.items():
            L.append(f"| {n} | {acc:.4f} |")
        L.append("")
        L.append("**It saturates by N=5 and is flat to N=40.** The errors are")
        L.append("bearing-level, not window-level: when the model is wrong about a")
        L.append("bearing it is wrong about *every* window of it, so averaging more")
        L.append("looks at the same bearing cannot help. K002 scores 0.0000 — forty")
        L.append("observations of K002 still give the wrong answer.\n")

    if variants.get("3class_shuffled_LEAKY"):
        r = variants["3class_shuffled_LEAKY"]
        L.append("## Calibration: what a leaky split would report\n")
        L.append("Windows split at random, ignoring bearing identity, so the seven")
        L.append("near-identical windows of one recording land on both sides. This is")
        L.append("the protocol behind most published Paderborn results.\n")
        L.append(f"| Protocol | Window accuracy | Macro-F1 |")
        L.append("|---|---|---|")
        L.append(f"| Shuffled windows (**leaky — not a generalisation estimate**) | "
                 f"{r['window_acc']:.4f} | {r['macro_f1']:.4f} |")
        if best:
            gap = r["window_acc"] - best["pooled"]["window_acc"]
            L.append(f"| Leave-one-bearing-out (honest) | "
                     f"{best['pooled']['window_acc']:.4f} | "
                     f"{best['pooled']['macro_f1']:.4f} |")
            L.append("")
            L.append(f"**Gap: {gap * 100:.1f} points.** That gap is the entire reason")
            L.append("the honest number looks unimpressive next to the literature.")
            L.append("Reporting the shuffled figure would clear 90 % immediately and")
            L.append("would mean nothing about an unseen bearing.\n")

    L.append("## What limits it\n")
    L.append("- **29 bearings is the sample size**, not 16,127 windows. The healthy")
    L.append("  class has six, and leave-one-bearing-out leaves five to learn from.")
    L.append("- At bearing level, **12 of 29 bearings sit closest to the wrong class**")
    L.append("  centroid on mean spectrum alone. The classes are genuinely entangled")
    L.append("  for a substantial minority of specimens.")
    L.append("- Errors are all-or-nothing per bearing, which is why aggregation")
    L.append("  saturates and why the per-bearing spread is so wide.\n")

    out = os.path.join(C.PROJECT_ROOT, "docs", "accuracy.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
