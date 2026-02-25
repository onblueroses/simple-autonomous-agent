"""Tests for async LLM client functions.

Mirrors test_llm.py structure with AsyncMock. Covers retry with exponential
backoff, non-retryable error propagation, and basic async function return values.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from simple_agent.config import ModelConfig
from simple_agent.llm import _async_retry_llm_call, acreate_client, adraft, areason, ascore


class TestAsyncRetryLlmCall:
    @patch("simple_agent.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_rate_limit(self, mock_sleep):
        fn = AsyncMock(side_effect=[
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            "success",
        ])
        result = await _async_retry_llm_call(fn, max_retries=2, base_delay=1.0)
        assert result == "success"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("simple_agent.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff(self, mock_sleep):
        fn = AsyncMock(side_effect=[
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            "success",
        ])
        await _async_retry_llm_call(fn, max_retries=2, base_delay=1.0)
        # First retry: 1.0 * 2^0 = 1.0, second: 1.0 * 2^1 = 2.0
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

    async def test_non_retryable_error_raises_immediately(self):
        fn = AsyncMock(side_effect=openai.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body=None,
        ))
        with pytest.raises(openai.AuthenticationError):
            await _async_retry_llm_call(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 1

    @patch("simple_agent.llm.asyncio.sleep", new_callable=AsyncMock)
    async def test_exhausted_retries_raises(self, mock_sleep):
        fn = AsyncMock(side_effect=openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None,
        ))
        with pytest.raises(openai.RateLimitError):
            await _async_retry_llm_call(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3

    async def test_zero_retries_disables_retry(self):
        fn = AsyncMock(side_effect=openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None,
        ))
        with pytest.raises(openai.RateLimitError):
            await _async_retry_llm_call(fn, max_retries=0, base_delay=0.01)
        assert fn.call_count == 1

    async def test_success_on_first_try(self):
        fn = AsyncMock(return_value="immediate success")
        result = await _async_retry_llm_call(fn, max_retries=2, base_delay=1.0)
        assert result == "immediate success"
        assert fn.call_count == 1


class TestAscore:
    async def test_returns_content(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = '{"score": 0.85, "reason": "relevant"}'
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await ascore(client, "Rate this", config)
        assert result == '{"score": 0.85, "reason": "relevant"}'


class TestAreason:
    async def test_normal_content_returned(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "Normal analysis result."
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await areason(client, "Analyze this", config)
        assert result == "Normal analysis result."

    async def test_content_with_think_tags_stripped(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "<think>Internal reasoning.</think>The real answer is here."
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await areason(client, "Analyze this", config)
        assert result == "The real answer is here."

    async def test_empty_content_uses_reasoning_content(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        msg.reasoning_content = "Deep analysis from reasoning_content field."
        msg.reasoning = None
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await areason(client, "Analyze this", config)
        assert result == "Deep analysis from reasoning_content field."

    async def test_empty_content_uses_reasoning_field(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        msg.reasoning = "Analysis from reasoning field."
        del msg.reasoning_content
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await areason(client, "Analyze this", config)
        assert result == "Analysis from reasoning field."

    async def test_empty_content_no_reasoning_fields(self):
        client = MagicMock()
        msg = MagicMock(spec=[])
        msg.content = ""
        del msg.reasoning_content
        del msg.reasoning
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await areason(client, "Analyze this", config)
        assert result == ""


class TestAdraft:
    async def test_returns_content(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "A well-crafted draft response."
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=msg)])
        )
        config = ModelConfig(model="test")
        result = await adraft(client, "You are an expert.", "Write about this", config)
        assert result == "A well-crafted draft response."


class TestAcreateClient:
    def test_returns_async_openai(self):
        client = acreate_client("https://api.example.com/v1", "sk-test")
        assert isinstance(client, openai.AsyncOpenAI)
