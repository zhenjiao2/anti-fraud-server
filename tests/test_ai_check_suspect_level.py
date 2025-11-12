import json
import os

import ai


def test_returns_parsed_json_from_sample_file(monkeypatch):
    # Load the sample agent output produced earlier in the repo
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "langchain_fraud_analysis_result.json")
    # Fallback path for alternate layouts
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "src", "langchain_fraud_analysis_result.json")

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_analysis_str = data.get("final_analysis")
    assert final_analysis_str, "test fixture must include final_analysis"

    def fake_run(message):
        return {"final_analysis": final_analysis_str, "status": data.get("status", "completed")}

    monkeypatch.setattr(ai, "_run_workflow_sync", fake_run)

    res = ai.check_suspect_level("dummy message")
    assert isinstance(res, dict)
    # sample has rating 3
    assert res.get("rating") == 3
    assert "fraud_type" in res
    assert "advice" in res
    assert isinstance(res.get("risk_keywords"), list)


def test_returns_error_on_exception(monkeypatch):
    def fake_run(message):
        raise RuntimeError("boom")

    monkeypatch.setattr(ai, "_run_workflow_sync", fake_run)

    res = ai.check_suspect_level("dummy")
    assert isinstance(res, dict)
    assert res.get("error") == "agent_failure"
    assert "boom" in res.get("error_message", "")


import pytest


# @pytest.mark.skipif(
#     not os.environ.get("RUN_AGENT_INTEGRATION_TESTS"),
#     reason="integration tests disabled by default",
# )
def test_integration_runs_workflow():
    # This test will run the real workflow. It is skipped by default unless
    # RUN_AGENT_INTEGRATION_TESTS=1 is set in the environment.
    # It assumes the repository's config or environment variables for model
    # credentials are already configured.
    res = ai.check_suspect_level("测试短信：请不要点击任何可疑链接。")
    data = json.loads(remove_codeblock_markers(res))

    # The result for integration should be a dict containing parsed analysis
    assert isinstance(data, dict)
    # final structure should include advice or reasoning
    assert ("advice" in data) and ("reasoning" in data) and ("rating" in data)

def remove_codeblock_markers(text):
    # Remove the starting marker "```json"
    if text.startswith("```json"):
        text = text[len("```json"):].lstrip()
    # Remove the ending marker "```"
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text        

