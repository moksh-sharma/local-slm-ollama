import pytest

from local_slm.benchmark.metrics import summarize_prompt
from local_slm.benchmark.quality import evaluate_task
from local_slm.models import GenerationResult


def test_quality_math_pass():
    task = {
        "id": "math",
        "category": "reasoning",
        "prompt": "17*23",
        "checks": [{"type": "contains", "value": "391"}],
    }
    result = evaluate_task(task, "The answer is 391.")
    assert result.passed


def test_quality_json_fail():
    task = {
        "id": "json",
        "checks": [{"type": "json_has", "keys": ["status", "count"]}],
    }
    result = evaluate_task(task, "not json")
    assert not result.passed


def test_summarize_prompt_averages():
    runs = [
        GenerationResult(
            model="m",
            prompt="p",
            response="r",
            time_to_first_token_ms=100,
            total_latency_ms=500,
            tokens_per_second=40,
            completion_tokens=20,
        ),
        GenerationResult(
            model="m",
            prompt="p",
            response="r",
            time_to_first_token_ms=200,
            total_latency_ms=700,
            tokens_per_second=60,
            completion_tokens=30,
        ),
    ]
    summary = summarize_prompt(runs, "id1", "gen")
    assert summary.avg_ttft_ms == 150
    assert summary.avg_latency_ms == 600
    assert summary.avg_tokens_per_second == 50
