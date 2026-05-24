import pytest

from local_slm.ollama_client import OllamaClient


def test_mock_generate_returns_timing():
    client = OllamaClient(mock=True)
    result = client.generate("llama3.2:1b", "Hello", max_tokens=32)
    assert result.response
    assert result.time_to_first_token_ms is not None
    assert result.total_latency_ms > 0
    assert result.tokens_per_second > 0


def test_mock_quality_answers_vary_by_model():
    client = OllamaClient(mock=True)
    small = client.generate("llama3.2:1b", "What is 17 * 23? Reply with only the number.", max_tokens=16)
    large = client.generate("qwen2.5:3b", "What is 17 * 23? Reply with only the number.", max_tokens=16)
    assert "391" in large.response
    assert small.response != large.response or "391" in small.response
