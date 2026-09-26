"""
Split utilities: prove zero group overlap, and prove the purge gap actually purges.

These are the tests the P0 gate names ("split tests prove zero group overlap").
They are deliberately paranoid about the two failure modes this project has
already hit once: a split made over rows instead of specimens, and a contiguous
cut with no gap between overlapping windows.
"""

import numpy as np
import pytest

from drivesentinel.common import splits as S


# ---------------------------------------------------------------------------
# fixtures: a small dataset with an obvious group structure
# ---------------------------------------------------------------------------

def make_rows(n_groups=3, per_group=10):
    """groups = ['m0' x10, 'm1' x10, ...], labels cycling over 3 classes."""
    groups, labels = [], []
    classes = ["healthy", "inter_turn", "inter_coil"]
    for g in range(n_groups):
        for i in range(per_group):
            groups.append(f"m{g}")
            labels.append(classes[i % len(classes)])
    return np.array(groups), np.array(labels)


# ---------------------------------------------------------------------------
# leave-one-group-out
# ---------------------------------------------------------------------------

def test_logo_makes_one_fold_per_group():
    groups, _ = make_rows(n_groups=3)
    folds = S.leave_one_group_out(groups)
    assert len(folds) == 3
    assert [f["test_groups"] for f in folds] == [["m0"], ["m1"], ["m2"]]


def test_logo_has_zero_group_overlap():
    groups, _ = make_rows(n_groups=4, per_group=7)
    for f in S.leave_one_group_out(groups):
        S.assert_no_group_overlap(groups, f["train_idx"], f["test_idx"])
        S.assert_disjoint(f["train_idx"], f["test_idx"])


def test_logo_covers_every_row_exactly_once_as_test():
    groups, _ = make_rows(n_groups=3, per_group=5)
    seen = np.concatenate([f["test_idx"] for f in S.leave_one_group_out(groups)])
    assert sorted(seen.tolist()) == list(range(len(groups)))


def test_logo_train_and_test_partition_the_data():
    groups, _ = make_rows(n_groups=3)
    for f in S.leave_one_group_out(groups):
        n = len(f["train_idx"]) + len(f["test_idx"])
        assert n == len(groups)


def test_logo_refuses_too_few_groups():
    groups = np.array(["only_one"] * 10)
    with pytest.raises(ValueError, match="needs >= 2 groups"):
        S.leave_one_group_out(groups)


def test_logo_is_deterministic():
    groups, _ = make_rows(n_groups=3)
    a = S.leave_one_group_out(groups)
    b = S.leave_one_group_out(groups)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa["test_idx"], fb["test_idx"])


# ---------------------------------------------------------------------------
# the leak detector itself must actually detect leaks
# ---------------------------------------------------------------------------

def test_assert_no_group_overlap_fires_on_a_real_leak():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError, match="GROUP LEAK"):
        S.assert_no_group_overlap(groups, [0, 2], [1, 3])


def test_assert_no_group_overlap_passes_on_a_clean_split():
    groups = np.array(["a", "a", "b", "b"])
    S.assert_no_group_overlap(groups, [0, 1], [2, 3])


def test_assert_disjoint_fires_on_a_shared_index():
    with pytest.raises(AssertionError, match="INDEX LEAK"):
        S.assert_disjoint([0, 1, 2], [2, 3])


# ---------------------------------------------------------------------------
# condition holdout (D2 lowest-severity)
# ---------------------------------------------------------------------------

def test_holdout_condition_splits_on_the_named_value():
    sev = np.array([0.0, 0.68, 0.81, 1.34, 0.68])
    groups = np.array(["m0"] * 5)
    f = S.holdout_condition(sev, groups, held_out=0.68)
    assert sorted(f["test_idx"].tolist()) == [1, 4]
    assert sorted(f["train_idx"].tolist()) == [0, 2, 3]


def test_holdout_condition_carries_its_caveat():
    sev = np.array([0.0, 0.68, 1.34])
    f = S.holdout_condition(sev, np.array(["m0"] * 3), held_out=0.68)
    assert "not to an unseen machine" in f["caveat"]


def test_holdout_condition_refuses_an_absent_value():
    sev = np.array([0.0, 0.68])
    with pytest.raises(ValueError, match="no rows with"):
        S.holdout_condition(sev, np.array(["m0"] * 2), held_out=99.0)


def test_holdout_condition_refuses_to_hold_out_everything():
    sev = np.array([0.68, 0.68])
    with pytest.raises(ValueError, match="every row"):
        S.holdout_condition(sev, np.array(["m0"] * 2), held_out=0.68)


# ---------------------------------------------------------------------------
# contiguous block + purge  (D3)
# ---------------------------------------------------------------------------

def test_contiguous_block_purges_the_straddling_window():
    f = S.contiguous_block(n=100, train_frac=0.70, purge=1)
    assert f["train_idx"].max() == 68     # index 69 purged
    assert f["test_idx"].min() == 70
    S.assert_disjoint(f["train_idx"], f["test_idx"])


def test_contiguous_block_purge_gap_scales():
    f = S.contiguous_block(n=100, train_frac=0.70, purge=5)
    assert f["train_idx"].max() == 64
    assert f["test_idx"].min() == 70
    assert f["test_idx"].min() - f["train_idx"].max() == 6


def test_contiguous_block_refuses_zero_purge():
    with pytest.raises(ValueError, match="purge must be >= 1"):
        S.contiguous_block(n=100, purge=0)


def test_contiguous_block_is_ordered_not_shuffled():
    f = S.contiguous_block(n=50)
    assert np.all(np.diff(f["train_idx"]) > 0)
    assert np.all(np.diff(f["test_idx"]) > 0)
    assert f["train_idx"].max() < f["test_idx"].min()


def test_contiguous_block_refuses_a_degenerate_split():
    with pytest.raises(ValueError, match="leaves"):
        S.contiguous_block(n=3, train_frac=0.99, purge=5)


def test_contiguous_block_per_group_cuts_inside_each_group():
    groups = np.array(["f0"] * 100 + ["f1"] * 50)
    f = S.contiguous_block_per_group(groups, train_frac=0.70, purge=1)
    S.assert_disjoint(f["train_idx"], f["test_idx"])
    # both groups must appear on BOTH sides -- that is the point of a
    # within-run split, and it is why the caveat says "within-run"
    g = groups
    assert set(g[f["train_idx"]]) == {"f0", "f1"}
    assert set(g[f["test_idx"]]) == {"f0", "f1"}
    assert "within-run" in f["caveat"]


# ---------------------------------------------------------------------------
# the leaky reference must be labelled
# ---------------------------------------------------------------------------

def test_shuffled_folds_are_flagged_leaky():
    folds = S.shuffled(n=100, n_folds=5, seed=0)
    assert all(f["leaky"] for f in folds)
    assert all("leaky reference" in f["caveat"] for f in folds)


def test_shuffled_covers_every_row_once():
    folds = S.shuffled(n=97, n_folds=5, seed=1)
    seen = np.concatenate([f["test_idx"] for f in folds])
    assert sorted(seen.tolist()) == list(range(97))


def test_shuffled_stratifies_when_given_labels():
    labels = np.array(["a"] * 50 + ["b"] * 50)
    folds = S.shuffled(n=100, n_folds=5, seed=0, labels=labels)
    for f in folds:
        counts = np.unique(labels[f["test_idx"]], return_counts=True)[1]
        assert counts.tolist() == [10, 10]


def test_shuffled_is_reproducible_for_a_seed():
    a = S.shuffled(n=60, seed=7)
    b = S.shuffled(n=60, seed=7)
    assert all(np.array_equal(x["test_idx"], y["test_idx"]) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# iter_checked: the gate every branch script runs through
# ---------------------------------------------------------------------------

def test_iter_checked_passes_clean_group_folds():
    groups, _ = make_rows(n_groups=3)
    assert len(list(S.iter_checked(S.leave_one_group_out(groups), groups))) == 3


def test_iter_checked_lets_a_declared_leaky_fold_through():
    groups, _ = make_rows(n_groups=3)
    folds = S.shuffled(n=len(groups), n_folds=3, seed=0)
    assert len(list(S.iter_checked(folds, groups))) == 3


def test_iter_checked_raises_on_an_undeclared_leak():
    groups = np.array(["a", "a", "b", "b"])
    bad = [{"fold": 0, "scheme": "hand-made",
            "train_idx": np.array([0, 2]), "test_idx": np.array([1, 3])}]
    with pytest.raises(AssertionError, match="GROUP LEAK"):
        list(S.iter_checked(bad, groups))


# ---------------------------------------------------------------------------
# validation-group count -> fusion fault authority
# ---------------------------------------------------------------------------

def test_n_validation_groups_counts_held_out_groups():
    groups, _ = make_rows(n_groups=3)
    folds = S.leave_one_group_out(groups)
    assert S.n_validation_groups(folds, groups) == 3


def test_n_validation_groups_ignores_leaky_folds():
    groups, _ = make_rows(n_groups=3)
    folds = S.shuffled(n=len(groups), n_folds=3, seed=0)
    assert S.n_validation_groups(folds, groups) == 0


def test_two_group_dataset_fails_the_fusion_floor():
    """D4 has 2 motors. It must not be able to raise Fault on its own."""
    from drivesentinel import config as C
    groups = np.array(["healthy_motor"] * 10 + ["faulty_motor"] * 10)
    folds = S.leave_one_group_out(groups)
    n = S.n_validation_groups(folds, groups)
    assert n == 2
    assert C.fault_authority(n, macro_f1=0.99) is False


def test_three_group_dataset_with_a_good_score_passes_the_floor():
    from drivesentinel import config as C
    groups, _ = make_rows(n_groups=3)
    n = S.n_validation_groups(S.leave_one_group_out(groups), groups)
    assert C.fault_authority(n, macro_f1=0.80) is True
    assert C.fault_authority(n, macro_f1=0.74) is False


def test_fault_authority_is_false_on_missing_inputs():
    from drivesentinel import config as C
    assert C.fault_authority(None, 0.9) is False
    assert C.fault_authority(5, None) is False


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def test_group_counts_reports_per_class_per_group():
    groups = np.array(["m0", "m0", "m1"])
    labels = np.array(["healthy", "fault", "fault"])
    c = S.group_counts(groups, labels)
    assert c["fault"] == {"m0": 1, "m1": 1}
    assert c["healthy"] == {"m0": 1}


def test_describe_surfaces_a_single_group_class():
    """The D2 shape: healthy exists on every motor but as one recording each."""
    groups = np.array(["m0"] * 3 + ["m1"] * 3)
    labels = np.array(["healthy", "a", "b", "healthy", "a", "b"])
    text = S.describe(groups, labels)
    assert "2 group(s)" in text
    assert "healthy" in text
