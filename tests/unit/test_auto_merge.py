"""The conditions under which the evaluator merges the top-ranked result itself."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pr_bot", Path(__file__).resolve().parents[2] / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_bot)

SHA = "a" * 40
ENTRY = {"status": "frontier", "tier": "L", "head": SHA, "kind": "manifest", "gain": 0.004}
DETAIL = {"head": {"sha": SHA}, "draft": False, "merged": False, "labels": [],
          "mergeable": True, "mergeable_state": "clean", "title": "feat(manifests): deep qkv FP8"}


def test_a_measured_paying_clean_result_is_merged():
    assert pr_bot.merge_blockers(ENTRY, DETAIL) == []


def test_a_force_push_after_evaluation_stops_the_merge():
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "head": {"sha": "b" * 40}}) == \
        ["the head moved after it was evaluated"]


def test_unpaid_unmeasured_and_provisional_results_are_never_merged():
    assert pr_bot.merge_blockers({**ENTRY, "status": "provisional"}, DETAIL)   # no private holdout PASS
    assert pr_bot.merge_blockers({**ENTRY, "status": "dominated", "tier": "none"}, DETAIL)
    assert pr_bot.merge_blockers({}, DETAIL)


def test_failing_checks_conflicts_and_drafts_stop_the_merge():
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "mergeable_state": "unstable"})   # a check is red or running
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "mergeable": False, "mergeable_state": "dirty"})
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "mergeable": None})               # not computed yet
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "draft": True})
    assert pr_bot.merge_blockers(ENTRY, {**DETAIL, "merged": True})


def test_contributed_code_still_needs_the_maintainers_approval():
    code = {**ENTRY, "kind": "code"}
    assert pr_bot.merge_blockers(code, DETAIL) == [f"contributed code without {pr_bot.APPROVED}"]
    assert pr_bot.merge_blockers(code, {**DETAIL, "labels": [{"name": pr_bot.APPROVED}]}) == []
