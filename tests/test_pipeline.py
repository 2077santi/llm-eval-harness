"""End-to-end with the offline mock judge: no API key, no cost."""

import pytest

from evalh import analyze, judge
from evalh.dataset import load_eval_set
from evalh.judge import JudgeScore, build_prompt


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    mp = pytest.MonkeyPatch()
    mp.setattr(judge, "RESULTS_DIR", tmp / "results")
    mp.setattr(analyze, "results_path", judge.results_path)
    judge.run(mock=True)
    yield analyze.run(out_dir=tmp)
    mp.undo()


def test_eval_set_is_balanced_enough():
    items = load_eval_set()
    assert len(items) == 40
    assert items["id"].is_unique
    assert items["category"].value_counts().min() == 8
    assert (items["planted_issue"] == "none").mean() < 0.6  # enough flawed responses to test a judge


def test_rubric_prompt_wraps_inputs():
    prompt = build_prompt("Q?", "A.")
    assert "<user_prompt>\nQ?\n</user_prompt>" in prompt
    assert "<assistant_response>\nA.\n</assistant_response>" in prompt


def test_schema_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        JudgeScore(accuracy=6, helpfulness=3, safety_pass=True, primary_issue="none", confidence=0.9, rationale="x")


def test_mock_run_is_labeled_as_mock(result):
    assert result["is_mock"] is True
    assert result["summary"]["scored"] == 40


def test_review_queue_catches_disagreements(result):
    queued = {q["id"] for q in result["review_queue"]}
    for item in result["items"]:
        if abs(item["j_helpfulness"] - item["h_helpfulness"]) >= 2:
            assert item["id"] in queued


def test_resume_does_not_rescore(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "RESULTS_DIR", tmp_path)
    out = judge.run(mock=True, limit=3)
    judge.run(mock=True, limit=5)
    assert len(out.read_text().splitlines()) == 5
