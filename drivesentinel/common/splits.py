"""
Group-holdout splits for the multi-stage branches, and the assertions that prove
they held.

WHAT THIS IS FOR
----------------
`folds.py` already does leave-one-bearing-out for D1 and does it correctly.  This
module is the generalisation the other three datasets need, plus the two split
shapes D1 never required:

  leave_one_group_out      D2 by motor (3), D4 by motor (2)
  holdout_condition        D2 lowest-severity holdout -- train on higher
                           severities, test on the lowest
  contiguous_block         D3, where there is exactly ONE run per condition and
                           therefore no group axis at all
  shuffled                 the leaky reference, generated deliberately and
                           labelled as such

THE PURGE GAP IS NOT OPTIONAL FOR CONTIGUOUS SPLITS
---------------------------------------------------
A rolling window of length W with hop H means consecutive windows share W-H
samples.  Cutting a run at 70 % without a gap puts windows that physically
overlap on both sides of the split.  On D3 that is 50-sample windows at hop 10:
the window straddling the cut shares 80 % of its samples with a training window.
`contiguous_block` therefore takes a purge in WINDOWS and drops them, and refuses
to run with a purge smaller than one window's worth of overlap.

WHY EVERY FUNCTION RETURNS INDICES, NOT DATA
--------------------------------------------
So that `assert_no_group_overlap` can be called on the result by the caller, by
the tests, and by the branch scripts, against the same `groups` array the split
was built from.  A split that cannot be audited after the fact is a split you
have to trust.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Fold = Dict[str, object]


# ===========================================================================
# assertions
# ===========================================================================

def assert_no_group_overlap(groups: Sequence, train_idx: Sequence[int],
                            test_idx: Sequence[int], context: str = "") -> None:
    """
    Raise if any group appears on both sides.  This is the whole point.

    Cheap enough to call on every fold of every branch, and it is -- see the
    branch scripts.  `folds.py:341` does the same thing for D1 and that assertion
    has never fired, which is the argument for keeping it rather than removing it.
    """
    g = np.asarray(groups)
    tr = set(g[np.asarray(train_idx, dtype=int)].tolist())
    te = set(g[np.asarray(test_idx, dtype=int)].tolist())
    both = tr & te
    if both:
        where = f" [{context}]" if context else ""
        raise AssertionError(f"GROUP LEAK{where}: {sorted(both)} on both sides")


def assert_disjoint(train_idx: Sequence[int], test_idx: Sequence[int],
                    context: str = "") -> None:
    """Raise if any index appears on both sides. Catches off-by-one purges."""
    both = set(map(int, train_idx)) & set(map(int, test_idx))
    if both:
        where = f" [{context}]" if context else ""
        n = len(both)
        raise AssertionError(
            f"INDEX LEAK{where}: {n} index(es) on both sides, e.g. {sorted(both)[:5]}"
        )


def group_counts(groups: Sequence, labels: Sequence) -> Dict[str, Dict[str, int]]:
    """
    label -> {group -> n}.  Printed by every branch script before it trains, so
    that "3 folds over 3 motors with 1 healthy recording each" is visible in the
    log rather than discovered in the results.
    """
    out: Dict[str, Dict[str, int]] = {}
    for g, l in zip(groups, labels):
        out.setdefault(str(l), {}).setdefault(str(g), 0)
        out[str(l)][str(g)] += 1
    return out


def describe(groups: Sequence, labels: Sequence) -> str:
    counts = group_counts(groups, labels)
    n_groups = len(set(map(str, groups)))
    lines = [f"{n_groups} group(s), {len(set(map(str, labels)))} class(es), "
             f"{len(list(groups))} rows"]
    for lab in sorted(counts):
        per = counts[lab]
        lines.append(f"  {lab:24} {sum(per.values()):6} rows over "
                     f"{len(per):2} group(s): "
                     + ", ".join(f"{g}={n}" for g, n in sorted(per.items())))
    return "\n".join(lines)


# ===========================================================================
# splits
# ===========================================================================

def leave_one_group_out(groups: Sequence,
                        min_groups: int = 2) -> List[Fold]:
    """
    One fold per group, in sorted order.  Deterministic, no RNG.

    `min_groups` guards the case that actually bit this project: a split with
    fewer groups than classes cannot produce a meaningful per-class number, and
    silently returning one fold is worse than failing.
    """
    uniq = sorted(set(map(str, groups)))
    if len(uniq) < min_groups:
        raise ValueError(
            f"leave-one-group-out needs >= {min_groups} groups, got {len(uniq)}: "
            f"{uniq}. If this dataset has one run per condition, use "
            f"contiguous_block() and label the result within-run."
        )

    g = np.asarray(list(map(str, groups)))
    folds: List[Fold] = []
    for i, held in enumerate(uniq):
        is_test = g == held
        folds.append({
            "fold": i,
            "scheme": "leave-one-group-out",
            "test_groups": [held],
            "train_idx": np.flatnonzero(~is_test),
            "test_idx": np.flatnonzero(is_test),
        })
    return folds


def holdout_condition(conditions: Sequence, groups: Sequence,
                      held_out, name: str = "condition") -> Fold:
    """
    Train on everything except the named condition value(s); test on them.

    This is D2's V2: train on higher severities, test on the lowest.  It is NOT a
    group split -- the same motors appear on both sides -- so the number it
    produces answers "does this transfer to a smaller fault?", not "does this
    transfer to a new machine".  The returned fold says so in `caveat`, and the
    branch scripts copy that string onto the metric card.
    """
    held = {held_out} if not isinstance(held_out, (list, tuple, set)) else set(held_out)
    c = np.asarray([str(x) for x in conditions])
    is_test = np.isin(c, [str(h) for h in held])
    if not is_test.any():
        raise ValueError(f"no rows with {name} in {sorted(map(str, held))}")
    if is_test.all():
        raise ValueError(f"every row has {name} in {sorted(map(str, held))}")

    return {
        "fold": 0,
        "scheme": f"holdout-{name}",
        "held_out": sorted(map(str, held)),
        "train_idx": np.flatnonzero(~is_test),
        "test_idx": np.flatnonzero(is_test),
        "caveat": (f"the same groups appear on both sides; this measures transfer "
                   f"to an unseen {name}, not to an unseen machine"),
    }


def contiguous_block(n: int, train_frac: float = 0.70,
                     purge: int = 1, order: Optional[Sequence[int]] = None) -> Fold:
    """
    First `train_frac` train, last part test, with `purge` windows dropped between.

    For D3, where there is one run per condition and no group axis.  `order` lets
    a caller pass the within-run time order explicitly when the row order is not
    already chronological.

    purge >= 1 is enforced.  With a rolling window of length W and hop H, adjacent
    windows share W-H samples, so a zero-gap cut puts overlapping windows on both
    sides.  Pass purge = ceil(W/H) to guarantee no shared samples at all; the
    default of 1 removes the single straddling window and is the minimum that is
    defensible.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac={train_frac} must be in (0, 1)")
    if purge < 1:
        raise ValueError(
            "purge must be >= 1: adjacent rolling windows share samples, so a "
            "zero-gap contiguous split leaks by construction"
        )

    idx = np.arange(n) if order is None else np.asarray(order, dtype=int)
    cut = int(round(n * train_frac))
    train = idx[:max(cut - purge, 0)]
    test = idx[cut:]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"n={n} train_frac={train_frac} purge={purge} leaves "
            f"{len(train)} train / {len(test)} test rows"
        )
    return {
        "fold": 0,
        "scheme": "contiguous-block",
        "train_frac": train_frac,
        "purge": purge,
        "train_idx": train,
        "test_idx": test,
        "caveat": "one run per condition: this is a within-run estimate",
    }


def contiguous_block_per_group(groups: Sequence, train_frac: float = 0.70,
                               purge: int = 1) -> Fold:
    """
    `contiguous_block` applied within each group, then concatenated.

    D3's actual shape: nine condition files, each a separate continuous run, each
    needing its own 70/30 cut.  A single global cut would put whole conditions on
    one side.
    """
    g = np.asarray([str(x) for x in groups])
    train: List[np.ndarray] = []
    test: List[np.ndarray] = []
    for name in sorted(set(g.tolist())):
        rows = np.flatnonzero(g == name)
        f = contiguous_block(len(rows), train_frac, purge)
        train.append(rows[f["train_idx"]])
        test.append(rows[f["test_idx"]])
    return {
        "fold": 0,
        "scheme": "contiguous-block-per-group",
        "train_frac": train_frac,
        "purge": purge,
        "train_idx": np.concatenate(train),
        "test_idx": np.concatenate(test),
        "caveat": "one run per condition: this is a within-run estimate",
    }


def shuffled(n: int, n_folds: int = 5, seed: int = 0,
             labels: Optional[Sequence] = None) -> List[Fold]:
    """
    THE LEAKY REFERENCE, generated deliberately.

    Rows are split at random with no regard to group, so windows of one recording
    land on both sides.  Every fold carries `leaky: True` and a `caveat`, and the
    reporting code refuses to render a leaky fold without the label.  This exists
    so the gap between it and the honest number can be shown, which is the single
    most useful chart in this project.
    """
    rng = np.random.default_rng(seed)
    if labels is None:
        parts = np.array_split(rng.permutation(n), n_folds)
    else:
        y = np.asarray(labels)
        parts = [[] for _ in range(n_folds)]
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            for i, chunk in enumerate(np.array_split(idx, n_folds)):
                parts[i].extend(chunk.tolist())
        parts = [np.asarray(sorted(p)) for p in parts]

    folds: List[Fold] = []
    for k in range(n_folds):
        test = np.asarray(parts[k], dtype=int)
        train = np.concatenate([np.asarray(parts[j], dtype=int)
                                for j in range(n_folds) if j != k])
        folds.append({
            "fold": k,
            "scheme": "shuffled-rows",
            "leaky": True,
            "train_idx": train,
            "test_idx": test,
            "caveat": "(leaky reference) rows split at random, ignoring group",
        })
    return folds


def iter_checked(folds: List[Fold], groups: Sequence) -> Iterator[Fold]:
    """
    Yield folds, asserting group separation on each one first.

    Leaky folds are passed through unchecked -- they are leaky on purpose -- but
    only if they say so. A fold that leaks without `leaky: True` is a bug and
    raises.
    """
    for f in folds:
        assert_disjoint(f["train_idx"], f["test_idx"], context=f"fold {f['fold']}")
        if not f.get("leaky"):
            assert_no_group_overlap(groups, f["train_idx"], f["test_idx"],
                                    context=f"{f['scheme']} fold {f['fold']}")
        yield f


def n_validation_groups(folds: List[Fold], groups: Sequence) -> int:
    """
    How many distinct groups are held out across the whole protocol.

    Feeds `config.fault_authority()`: the fusion floor requires >= 3 independent
    validation groups before a branch may raise Fault on its own.  Computed from
    the folds rather than declared by hand, so a branch cannot claim authority it
    does not have.
    """
    g = np.asarray([str(x) for x in groups])
    held = set()
    for f in folds:
        if f.get("leaky"):
            continue
        held.update(g[np.asarray(f["test_idx"], dtype=int)].tolist())
    return len(held)
