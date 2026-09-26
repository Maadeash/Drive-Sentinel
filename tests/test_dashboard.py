"""
Dashboard panels.

`panels.py` is deliberately free of Streamlit so the rules it enforces can be
tested without a browser. The rule that matters is `assert_out_of_sample`: a
bearing scenario that cannot prove the model never saw it must not render.
"""

import json
import os

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import fusion as FU
from dashboard import panels as P


# ---------------------------------------------------------------------------
# the out-of-sample rule
# ---------------------------------------------------------------------------

def test_bearing_scenario_without_the_flag_is_refused():
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": False, "train_bearings": np.array(["K001"])}
    with pytest.raises(ValueError, match="not marked out-of-sample"):
        P.assert_out_of_sample(sc)


def test_bearing_scenario_training_on_itself_is_refused():
    """The violation the rule exists to catch."""
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": True, "train_bearings": np.array(["K001", "KA04"])}
    with pytest.raises(ValueError, match="OUT-OF-SAMPLE VIOLATION"):
        P.assert_out_of_sample(sc)


def test_a_clean_bearing_scenario_passes():
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": True, "train_bearings": np.array(["K001", "K002"])}
    P.assert_out_of_sample(sc)


def test_non_bearing_scenarios_are_not_subject_to_the_rule():
    """The winding ramp is not a held-out-unit replay and does not claim to be."""
    P.assert_out_of_sample({"branch": "winding", "out_of_sample": False})


# ---------------------------------------------------------------------------
# stage rows and tier badges
# ---------------------------------------------------------------------------

def test_every_stage_gets_a_row():
    rows = P.stage_rows(FU.load_branch_metrics())
    assert [r["stage"] for r in rows] == ["S1", "S2", "S3", "S4", "S5"]


def test_each_row_carries_its_own_badge_not_a_legend():
    for r in P.stage_rows(FU.load_branch_metrics()):
        assert r["badge"] in ("FAULT-CAPABLE", "INDICATIVE", "NOT MEASURED")
        assert r["badge_colour"].startswith("#")


def test_bearing_stage_is_fault_capable_as_measured():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["bearing"]["measured"]:
        pytest.skip("lobo_summary.json absent")
    assert rows["bearing"]["can_fault"] is True
    assert rows["bearing"]["badge"] == "FAULT-CAPABLE"


def test_winding_stage_is_indicative_as_measured():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    assert rows["winding"]["can_fault"] is False
    assert rows["winding"]["badge"] == "INDICATIVE"


def test_all_five_stages_are_always_shown():
    """
    Never hide a stage. All four branches are measured now, so this asserts the
    structural property rather than relying on one being absent.
    """
    rows = {r["stage"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    assert set(rows) == {"S1", "S2", "S3", "S4", "S5"}
    assert all(r["metric_line"] for r in rows.values())


def test_an_unmeasured_branch_still_gets_a_row_and_a_reason():
    """Constructed directly, so it holds even when every real branch is measured."""
    m = dict(FU.load_branch_metrics())
    m["supply"] = FU.BranchMetric("supply", "S1", measured=False,
                                  note="results JSON absent")
    rows = {r["branch"]: r for r in P.stage_rows(m)}
    assert rows["supply"]["badge"] == "NOT MEASURED"
    assert rows["supply"]["can_fault"] is False
    assert "absent" in rows["supply"]["metric_line"]


def test_supply_stage_is_indicative_at_a_perfect_score():
    """
    B-S1 scores macro-F1 1.0000 and is still INDICATIVE, because two motors is
    below the pre-registered three-group minimum. If this ever reads
    Fault-capable, the floor has been moved.
    """
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["supply"]["measured"]:
        pytest.skip("supply_results.json absent")
    assert rows["supply"]["badge"] == "INDICATIVE"
    assert rows["supply"]["can_fault"] is False
    assert "1.0000" in rows["supply"]["metric_line"]


def test_stage_row_metric_line_names_the_protocol():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if rows["bearing"]["measured"]:
        assert "leave-one-bearing-out" in rows["bearing"]["metric_line"]


def test_status_defaults_to_not_measured_for_absent_branches():
    m = dict(FU.load_branch_metrics())
    m["inverter_telemetry"] = FU.BranchMetric("inverter_telemetry", "S3",
                                              measured=False, note="absent")
    rows = {r["branch"]: r for r in P.stage_rows(m)}
    assert rows["inverter_telemetry"]["status"] == "NOT MEASURED"


# ---------------------------------------------------------------------------
# metric cards
# ---------------------------------------------------------------------------

def test_cards_exist_for_every_branch():
    names = {c["branch"] for c in P.metric_cards(FU.load_branch_metrics())}
    assert names == {"supply", "inverter_telemetry", "winding", "bearing"}


def test_cards_carry_the_floor_for_comparison():
    fa = C.FUSION_CONFIG["fault_authority"]
    for c in P.metric_cards(FU.load_branch_metrics()):
        assert c["groups_floor"] == fa["min_validation_groups"]
        assert c["metric_floor"] == fa["min_macro_f1"]


def test_winding_card_carries_the_session_only_baseline():
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    if not cards["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    labels = [e["label"] for e in cards["winding"]["extra"]]
    assert "Session-only baseline" in labels
    assert any("leaky reference" in l for l in labels)


def test_winding_card_says_healthy_is_not_measurable():
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    if not cards["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    entry = [e for e in cards["winding"]["extra"] if e["label"] == "healthy"]
    assert entry and entry[0]["value"] == "NOT MEASURABLE"


def test_unmeasured_card_has_no_metric_and_says_why():
    m = dict(FU.load_branch_metrics())
    m["inverter_telemetry"] = FU.BranchMetric("inverter_telemetry", "S3",
                                              measured=False, note="absent")
    cards = {c["branch"]: c for c in P.metric_cards(m)}
    assert cards["inverter_telemetry"]["metric"] is None
    assert cards["inverter_telemetry"]["note"]


def test_inverter_branch_is_indicative_with_zero_groups():
    """
    B-S2/S3 has NO group axis -- one run per condition. Zero validation groups
    must fail the floor regardless of score.
    """
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["inverter_telemetry"]["measured"]:
        pytest.skip("inverter_telemetry_results.json absent")
    assert rows["inverter_telemetry"]["badge"] == "INDICATIVE"
    assert rows["inverter_telemetry"]["can_fault"] is False


def test_every_branch_now_has_a_row_with_a_tier():
    for r in P.stage_rows(FU.load_branch_metrics()):
        assert r["tier"] in ("Fault-capable", "INDICATIVE", "NOT MEASURED")


def test_supply_scenario_declares_it_is_a_rule_not_a_model():
    man = P.load_manifest()
    sup = [s for s in man["scenarios"] if s.get("branch") == "supply" and s.get("file")]
    if not sup:
        pytest.skip("supply scenario not built")
    assert sup[0]["deliverable"] == "threshold rule"
    assert sup[0]["out_of_sample"] is False       # nothing fitted, nothing to hold out
    assert sup[0]["verdict"] == sup[0]["true_label"]


# ---------------------------------------------------------------------------
# trip panel
# ---------------------------------------------------------------------------

def test_trip_panel_reports_gating_disabled():
    assert P.trip_panel()["enabled"] is False


def test_trip_panel_carries_the_resolution_argument():
    tp = P.trip_panel()
    assert tp["gap"] == pytest.approx(1.89, abs=0.02)
    assert tp["at_900"] < tp["gap"] < tp["at_20"]
    assert tp["needed_window_at_20"] > 40


def test_trip_panel_caption_states_no_ramp_exists():
    assert "no dataset" in P.trip_panel()["caption"].lower()


def test_resolution_table_marks_extrapolation():
    rows = P.trip_panel()["resolution"]
    assert any(r["shaft_rpm"] == 900.0 for r in rows)
    assert any(r["shaft_rpm"] <= 20.0 for r in rows)


# ---------------------------------------------------------------------------
# replay through fusion
# ---------------------------------------------------------------------------

def _fake_scenario(p_fault=0.95, n=30, branch="bearing"):
    probs = np.tile([[1 - p_fault, p_fault, 0.0]], (n, 1)).astype(np.float32)
    return {"branch": branch, "stage": "S5", "bearing": "KA04",
            "scenario": "bearing_KA04", "out_of_sample": True,
            "train_bearings": np.array(["K001"]),
            "labels": np.array(C.LABELS), "probs": probs}


def test_replay_produces_a_step_per_window():
    v = P.replay(_fake_scenario(n=25), FU.load_branch_metrics())
    assert len(v["trace"]) == 25
    assert len(v["p_fault"]) == 25


def test_replay_of_a_capable_branch_can_reach_fault():
    m = FU.load_branch_metrics()
    if not m["bearing"].fault_authority:
        pytest.skip("bearing metric absent")
    v = P.replay(_fake_scenario(0.99, 30), m)
    assert v["final"]["status"] == "Fault"


def test_replay_of_an_indicative_branch_caps_at_warning():
    """End-to-end through the dashboard path, not just the fusion unit test."""
    m = FU.load_branch_metrics()
    if not m["winding"].measured:
        pytest.skip("winding_results.json absent")
    sc = _fake_scenario(0.99, 30, branch="winding")
    sc["stage"] = "S4"
    v = P.replay(sc, m)
    assert v["final"]["status"] == "Warning"
    assert "LOW CONFIDENCE BRANCH" in v["evidence"]


def test_replay_refuses_an_in_sample_scenario():
    sc = _fake_scenario()
    sc["train_bearings"] = np.array(["KA04"])
    with pytest.raises(ValueError, match="OUT-OF-SAMPLE VIOLATION"):
        P.replay(sc, FU.load_branch_metrics())


def test_evidence_string_names_the_stage_and_the_protocol():
    v = P.replay(_fake_scenario(0.9, 15), FU.load_branch_metrics())
    assert "S5" in v["evidence"] and "pooled obs" in v["evidence"]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_loads_or_returns_an_empty_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DEMO_DIR", str(tmp_path))
    assert P.load_manifest() == {"scenarios": [], "built": None}


def test_built_manifest_marks_skipped_stages():
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    unbuilt = [s for s in man["scenarios"] if not s.get("file")]
    assert all(s.get("reason") for s in unbuilt), \
        "an unbuilt scenario must say why"


def test_built_bearing_scenarios_declare_out_of_sample():
    man = P.load_manifest()
    for s in man["scenarios"]:
        if s.get("branch") == "bearing" and s.get("file"):
            assert s["out_of_sample"] is True


# ---------------------------------------------------------------------------
# scenario files must be safe to load
# ---------------------------------------------------------------------------

def test_every_scenario_loads_without_pickling():
    """
    Regression. `filenames` came from a pandas column, so it was dtype=object,
    and numpy refuses to read an object array with allow_pickle=False. The
    dashboard disables pickling on purpose: a scenario file is something a judge
    might be handed, and loading it must not be able to execute code.
    """
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    for s in man["scenarios"]:
        if not s.get("file"):
            continue
        sc = P.load_scenario(s["file"])          # allow_pickle=False inside
        assert sc


def test_no_scenario_array_is_an_object_array():
    import numpy as _np
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    for s in man["scenarios"]:
        if not s.get("file"):
            continue
        with _np.load(os.path.join(P.DEMO_DIR, s["file"]), allow_pickle=True) as z:
            for k in z.files:
                assert z[k].dtype != object, f"{s['file']}:{k} is an object array"


def test_stored_window_count_is_capped():
    man = P.load_manifest()
    for s in man["scenarios"]:
        if s.get("branch") == "bearing" and s.get("file"):
            assert s["n_windows_stored"] <= 120
            assert s["n_windows_stored"] < s["n_windows_in_fold"]


# ---------------------------------------------------------------------------
# card rows must be TEXT, not canvas
#
# st.dataframe paints to a <canvas>: nothing in it is selectable, copyable,
# findable with ctrl-F, or reachable by a screen reader. The rows on these cards
# carry the argument, so they render as markdown instead. These tests guard the
# rows by name -- if one is dropped or the renderer regresses to a grid, the
# claim silently stops being readable on the slide.
# ---------------------------------------------------------------------------

def test_markdown_table_escapes_pipes_and_newlines():
    t = P.markdown_table(["a", "b"], [["x|y", "one" + chr(10) + "two"]])
    lines = t.split(chr(10))
    assert len(lines) == 3, "a newline in a cell broke the table apart"
    assert P.PIPE_ESCAPE + "|" not in t     # escaped, not doubled
    structural = lines[2].replace(P.PIPE_ESCAPE, "")
    assert structural.count("|") == 3      # two borders, one separator
    assert "one two" in lines[2]


def test_markdown_table_has_a_header_separator():
    t = P.markdown_table(["a", "b", "c"], [])
    assert t.split(chr(10))[1] == "|---|---|---|"


def _extras_text(branch):
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    c = cards[branch]
    if not c["measured"] or not c["extra"]:
        pytest.skip(f"{branch} results JSON absent")
    return P.extras_table(c["extra"])


def test_winding_card_rows_are_readable_text():
    t = _extras_text("winding")
    for claim in ("Session-only baseline", "NOT MEASURABLE", "leaky reference",
                  "CNN experiment", "NEGATIVE RESULT"):
        assert claim in t, claim


def test_inverter_card_rows_are_readable_text():
    t = _extras_text("inverter_telemetry")
    for claim in ("Run identity, with temperature", "Run identity, electrical only",
                  "electrical-only ablation", "leaky reference"):
        assert claim in t, claim


def test_inverter_card_carries_the_run_identification_control():
    """
    The control is what makes the 4-class number quotable. It reads from the
    same JSON the docs read, so the card cannot drift from the report.
    """
    import json as _json
    p = os.path.join(C.ARTIFACT_DIR, "multistage", "inverter_telemetry",
                     "inverter_telemetry_results.json")
    if not os.path.exists(p):
        pytest.skip("inverter branch not run")
    with open(p) as fh:
        rid = _json.load(fh)["run_identification_control"]
    t = _extras_text("inverter_telemetry")
    assert f"{rid['with_temperature']['accuracy']:.4f}" in t
    assert f"{rid['electrical_only']['accuracy']:.4f}" in t


def _app_source():
    return open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "dashboard", "app.py"),
        encoding="utf-8").read()


def _strip_comments(block):
    return chr(10).join(l for l in block.split(chr(10))
                        if not l.strip().startswith("#"))


def test_card_extras_are_not_rendered_with_st_dataframe():
    """The regression this replaced. A grid here is a canvas here."""
    src = _app_source()
    block = src.split("panel 5 -- metric cards")[1]
    assert "st.dataframe" not in _strip_comments(block)
    assert "P.extras_table(card[" in block


def test_the_order_resolution_table_is_selectable_text():
    """
    Panel 3's order-resolution table is the answer to the low-speed sheave
    question, and its last column is a per-row EXTRAPOLATION label. Painted to a
    canvas by st.dataframe those labels cannot be selected, copied, found with
    ctrl-F or read by a screen reader -- so a caveat that exists only as pixels
    is a caveat nobody can check. Same argument as the metric cards, one panel
    over, and it was left behind when those were converted.
    """
    src = _app_source()
    block = src.split("Order resolution vs shaft speed")[1].split("st.divider()")[0]
    assert "st.dataframe" not in _strip_comments(block)
    assert "P.markdown_table(" in block
    assert "EXTRAPOLATION" in block


def test_no_panel_renders_a_table_with_st_dataframe():
    """
    The whole file, not just the two panels that had the problem. Every table in
    this dashboard carries a protocol label or a caveat in one of its columns, so
    there is no table here for which a canvas grid is the right choice.
    """
    assert "st.dataframe" not in _strip_comments(_app_source())
