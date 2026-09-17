import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pr_bot", Path(__file__).resolve().parents[2] / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_bot)


def test_classify_manifest_code_and_evaluator_prs():
    assert pr_bot.classify(["manifests/gdn-deep.yaml"]) == ("manifest", ["manifests/gdn-deep.yaml"])
    assert pr_bot.classify(["manifests/a.yaml", "manifests/b.yaml"])[0] == "other"
    assert pr_bot.classify(["bittrellis/quantizers/gptq.py", "manifests/a.yaml", "docs/quantizers.md"])[0] == "code"
    assert pr_bot.classify(["manifests/a.yaml", "configs/hpc01.yaml"])[0] == "evaluator"
    assert pr_bot.classify(["bittrellis/eval/logits.py"])[0] == "evaluator"
    assert pr_bot.classify(["README.md"])[0] == "other"


GATES = {"rp_kl_max": 0.30, "top1_min": 0.80, "long_context_guard": {"required_success": {"long-8k": 1.0, "long-16k": 1.0}}}


def test_queue_runs_in_first_seen_order_not_pr_number():
    prs = [{"number": 5, "head": {"sha": "a"}}, {"number": 9, "head": {"sha": "b"}}]
    seen = {(5, "a"): "2026-09-17T12:00:00Z", (9, "b"): "2026-09-17T11:00:00Z"}  # #5 force-pushed later
    assert [p["number"] for p in pr_bot.queue_order(prs, seen)] == [9, 5]


def test_quality_gates_stop_before_speed_runs():
    ok = {"rp_kl": 0.12, "top1": 0.91, "nonfinite_logprobs": 0,
          "needles_by_length": {"long-8k": {"required": 3, "retrieved": 3}, "long-16k": {"required": 3, "retrieved": 3}}}
    assert pr_bot.quality_gate_failures(ok, GATES) == []
    bad = {**ok, "rp_kl": 0.4, "needles_by_length": {**ok["needles_by_length"], "long-16k": {"required": 3, "retrieved": 2}}}
    fails = pr_bot.quality_gate_failures(bad, GATES)
    assert len(fails) == 2 and "long-16k 2/3" in fails[1]


def test_references_are_earlier_live_prs_by_other_authors():
    def entry(pr, author, t, status="frontier", head="h"):
        return {"pr": pr, "author": author, "first_seen": t, "status": status, "artifact": f"/a/{pr}", "head": head}

    state = {"_merged": {}, "1-h": entry(1, "alice", "t1"), "2-h": entry(2, "bob", "t2"), "3-h": entry(3, "carol", "t4"),
             "4-h": entry(4, "dave", "t0", status="duplicate"), "5-h": entry(5, "erin", "t1", head="old")}
    me = {"pr": 6, "author": "bob", "first_seen": "t3"}
    live = {1: "h", 2: "h", 3: "h", 4: "h", 5: "new"}
    assert pr_bot.reference_entries(me, state, live) == ["1-h"]  # not own, not later, not unmeasured, not a stale head
    assert pr_bot.reference_entries(me, state, {2: "h", 3: "h"}) == []  # #1 closed unmerged


def test_status_and_comment():
    row = {"name": "x", "valid": True, "frontier": True, "frontier_gain": 0.004, "rp_kl": 0.12, "decode_tps": 94.0,
           "prefill_tps": 14000.0, "peak_gpu_gib": 22.0, "holdout": "PASS", "gate_failures": []}
    assert pr_bot.status_from_row(row) == "frontier"
    assert pr_bot.status_from_row({**row, "frontier_gain": 0.0}) == "dominated"
    assert pr_bot.status_from_row({**row, "valid": False}) == "gate"
    frontier = {"evaluator_epoch": "hpc01-e2", "incumbent": "V0", "internal": [row]}
    body = pr_bot.render_comment("x", "id", frontier, None, "frontier", ["note"], {"manifest": {"outcome": "pass"}},
                                 {"quality_seconds": 360.0})
    assert "bt:frontier" in body and "quality 6.0 min" in body and "Claude" not in body
