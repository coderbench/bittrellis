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
