"""Tests for LLM client functions.

Covers thinking-model robustness (multiple provider patterns)
and retry logic for transient failures.
"""

from unittest.mock import MagicMock, patch

import openai
import pytest

from simple_agent.config import ModelConfig
from simple_agent.llm import _retry_llm_call, _strip_think_tags, reason


class TestStripThinkTags:
    def test_no_tags_passthrough(self):
        assert _strip_think_tags("Normal text") == "Normal text"

    def test_think_tags_with_content_after(self):
        text = "<think>Let me analyze this carefully.</think>The answer is 42."
        assert _strip_think_tags(text) == "The answer is 42."

    def test_think_tags_content_inside_only(self):
        text = "<think>All the reasoning is here and there's nothing after.</think>"
        assert _strip_think_tags(text) == "All the reasoning is here and there's nothing after."

    def test_think_tags_with_whitespace(self):
        text = "<think>\n  Reasoning here.\n</think>\n\nThe actual answer."
        assert _strip_think_tags(text) == "The actual answer."

    def test_think_tags_case_insensitive(self):
        text = "<Think>reasoning</Think>Answer here."
        assert _strip_think_tags(text) == "Answer here."


class TestReason:
    def test_normal_content_returned(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "Normal analysis result."
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        config = ModelConfig(model="test")
        result = reason(client, "Analyze this", config)
        assert result == "Normal analysis result."

    def test_empty_content_uses_reasoning_content(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        msg.reasoning_content = "Deep analysis from reasoning_content field."
        msg.reasoning = None
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        config = ModelConfig(model="test")
        result = reason(client, "Analyze this", config)
        assert result == "Deep analysis from reasoning_content field."

    def test_empty_content_uses_reasoning_field(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        msg.reasoning = "Analysis from reasoning field."
        # reasoning_content not present
        del msg.reasoning_content
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        config = ModelConfig(model="test")
        result = reason(client, "Analyze this", config)
        assert result == "Analysis from reasoning field."

    def test_content_with_think_tags_stripped(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "<think>Internal reasoning.</think>The real answer is here."
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        config = ModelConfig(model="test")
        result = reason(client, "Analyze this", config)
        assert result == "The real answer is here."

    def test_empty_content_no_reasoning_fields(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        del msg.reasoning_content
        del msg.reasoning
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        config = ModelConfig(model="test")
        result = reason(client, "Analyze this", config)
        assert result == ""


class TestRetryLlmCall:
    @patch("simple_agent.llm.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        fn = MagicMock(side_effect=[
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            "success",
        ])
        result = _retry_llm_call(fn, max_retries=2, base_delay=1.0)
        assert result == "success"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("simple_agent.llm.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        fn = MagicMock(side_effect=[
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            "success",
        ])
        _retry_llm_call(fn, max_retries=2, base_delay=1.0)
        # First retry: 1.0 * 2^0 = 1.0, second: 1.0 * 2^1 = 2.0
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

    def test_non_retryable_error_raises_immediately(self):
        fn = MagicMock(side_effect=openai.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body=None,
        ))
        with pytest.raises(openai.AuthenticationError):
            _retry_llm_call(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 1

    @patch("simple_agent.llm.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        fn = MagicMock(side_effect=openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None,
        ))
        with pytest.raises(openai.RateLimitError):
            _retry_llm_call(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3

    def test_zero_retries_disables_retry(self):
        fn = MagicMock(side_effect=openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None,
        ))
        with pytest.raises(openai.RateLimitError):
            _retry_llm_call(fn, max_retries=0, base_delay=0.01)
        assert fn.call_count == 1

    def test_success_on_first_try(self):
        fn = MagicMock(return_value="immediate success")
        result = _retry_llm_call(fn, max_retries=2, base_delay=1.0)
        assert result == "immediate success"
        assert fn.call_count == 1
