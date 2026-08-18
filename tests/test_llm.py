"""Tests for LLM clients."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as ge
from vertexai.generative_models import (Candidate,
                                         Content, Part, GenerationResponse)

from truthclf.llm import (ResponseCache, ResponseShapeError, VertexClient,
                          make_client)
from truthclf.predictors.zeroshot import prob_from_logprobs


@pytest.fixture
def vertex_client():
    """A VertexClient with a mocked-out inner `_client`."""
    with patch("vertexai.generative_models.GenerativeModel") as mock:
        client = VertexClient("gemini-2.5-flash")
        client._client = mock
        # Set env vars so _ensure_client() doesn't complain
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "test-location"
        yield client
        del os.environ["GOOGLE_CLOUD_PROJECT"]
        del os.environ["GOOGLE_CLOUD_LOCATION"]


def test_make_client_vertex():
    """make_client(backend='vertex') should produce a VertexClient."""
    client = make_client("gemini-2.5-flash", backend="vertex")
    assert isinstance(client, VertexClient)


def test_vertex_client_contents_conversion():
    """Test conversion from OpenAI message format to Vertex `contents`."""
    client = VertexClient("gemini-2.5-flash")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    system_instruction, contents = client._contents(messages)
    assert system_instruction == "You are a helpful assistant."
    assert contents == [
        {"role": "user", "parts": [{"text": "Hello!"}]},
        {"role": "model", "parts": [{"text": "Hi there!"}]},
    ]


def test_vertex_client_classify_one_success(vertex_client):
    """A successful classify call should parse logprobs correctly."""
    # Assemble a realistic-looking mock response using MagicMock
    mock_response = MagicMock(spec=GenerationResponse)
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.finish_reason.name = "MAX_TOKENS"
    mock_candidate.text = "True"

    mock_candidate.logprobs_result = MagicMock()
    mock_cand_true = MagicMock(token_id=4339, log_probability=-0.1)
    mock_cand_false = MagicMock(token_id=9277, log_probability=-2.3)
    mock_top_candidate = MagicMock(candidates=[mock_cand_true, mock_cand_false])
    mock_candidate.logprobs_result.top_candidates = [mock_top_candidate]

    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 1

    vertex_client._client.return_value.generate_content.return_value = mock_response

    messages = [{"role": "user", "content": "Is this true?"}]
    result = vertex_client.classify_one(messages)

    assert "top_logprobs" in result
    assert "True" in result["top_logprobs"]
    assert "False" in result["top_logprobs"]
    assert result["top_logprobs"]["True"] == -0.1
    assert result["top_logprobs"]["False"] == -2.3

    # Check that it was cached
    generation_config = {
        "max_output_tokens": 1,
        "top_k": 5,
        "thinking_config": {"thinking_budget": 0},
    }
    cache_key = vertex_client._cache_key(messages, "classify", **generation_config)
    assert vertex_client.cache.get(cache_key) is not None


def test_vertex_client_classify_one_handles_stop_reason(vertex_client):
    """finish_reason=STOP is also a success condition."""
    mock_response = MagicMock(spec=GenerationResponse)
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.finish_reason.name = "STOP"
    mock_candidate.text = "True"
    mock_candidate.logprobs_result = MagicMock()
    mock_cand_true = MagicMock(token_id=4339, log_probability=-0.1)
    mock_cand_false = MagicMock(token_id=9277, log_probability=-2.3)
    mock_top_candidate = MagicMock(candidates=[mock_cand_true, mock_cand_false])
    mock_candidate.logprobs_result.top_candidates = [mock_top_candidate]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 1
    vertex_client._client.return_value.generate_content.return_value = mock_response

    messages = [{"role": "user", "content": "Is this true?"}]
    vertex_client.classify_one(messages)  # Should not raise


def test_vertex_client_classify_one_failure_reason(vertex_client):
    """A failure finish_reason should raise ResponseShapeError."""
    mock_response = MagicMock(spec=GenerationResponse)
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.finish_reason.name = "SAFETY"
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 0
    vertex_client._client.return_value.generate_content.return_value = mock_response

    with pytest.raises(ResponseShapeError, match="Generation failed with reason: SAFETY"):
        vertex_client.classify_one([{"role": "user", "content": "..."}])


def test_vertex_client_score_one_success(vertex_client):
    """A successful score call should return the model's text response."""
    mock_response = MagicMock(spec=GenerationResponse)
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.text = "85"
    mock_response.candidates = [mock_candidate]
    mock_response.text = "85"  # The response object also has a text property shortcut
    mock_response.usage_metadata.prompt_token_count = 12
    mock_response.usage_metadata.candidates_token_count = 1
    vertex_client._client.generate_content.return_value = mock_response

    messages = [{"role": "user", "content": "What is the score?"}]
    result = vertex_client.score_one(messages)

    assert result == "85"
    cache_key = vertex_client._cache_key(
        messages, "score", max_tokens=vertex_client.max_output_tokens
    )
    assert vertex_client.cache.get(cache_key) is not None


def test_prob_from_logprobs_softmax():
    """Verify p(True) calculation from logprobs."""
    logprobs = {"True": -0.1, "False": -2.3}
    p = prob_from_logprobs(logprobs)
    # math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.3))
    # 0.904837 / (0.904837 + 0.10025) = 0.9002
    assert p == pytest.approx(0.9002, abs=1e-4)

    # Only True
    logprobs = {"True": -0.1}
    p = prob_from_logprobs(logprobs)
    assert p == pytest.approx(0.9048, abs=1e-4)
    
    # Only False
    logprobs = {"False": -2.3}
    p = prob_from_logprobs(logprobs)
    assert p == pytest.approx(1.0 - 0.10025, abs=1e-4)

    # Neither
    logprobs = {"Maybe": -0.5}
    p = prob_from_logprobs(logprobs)
    assert p is None
