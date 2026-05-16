"""Tests for the AsyncAgent (arun)."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from simple_agent.agent import AsyncAgent


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _resp(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_msg(content, tool_calls))])


def _tc(call_id: str, name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _async_client(*responses):
    c = MagicMock()
    c.chat.completions.create = AsyncMock(side_effect=list(responses))
    return c


class TestAsyncAgentLoop:
    async def test_async_done_no_tools(self):
        c = _async_client(_resp(content="hi"))
        agent = AsyncAgent(client=c, model="m")
        r = await agent.arun("say")
        assert r.output == "hi" and r.terminated == "done" and r.steps == 1

    async def test_async_tool_call_then_final(self):
        def double(x: int) -> int:
            """Double x."""
            return x * 2

        c = _async_client(
            _resp(tool_calls=[_tc("c1", "double", '{"x": 21}')]),
            _resp(content="42"),
        )
        agent = AsyncAgent(client=c, model="m", tools=[double])
        r = await agent.arun("double")
        assert r.output == "42" and r.steps == 2

    async def test_async_max_steps(self):
        def keep(x: int) -> int:
            return x

        c = _async_client(
            _resp(tool_calls=[_tc("a", "keep", '{"x": 1}')]),
            _resp(tool_calls=[_tc("b", "keep", '{"x": 2}')]),
        )
        agent = AsyncAgent(client=c, model="m", tools=[keep], max_steps=2)
        r = await agent.arun("loop")
        assert r.terminated == "max_steps" and r.steps == 2

    async def test_async_no_progress(self):
        c = _async_client(_resp(content=None))
        agent = AsyncAgent(client=c, model="m")
        r = await agent.arun("?")
        assert r.terminated == "no_progress"

    async def test_async_memory_in_place(self):
        c = _async_client(_resp(content="ok"))
        agent = AsyncAgent(client=c, model="m")
        mem: list[dict] = []
        r = await agent.arun("hi", memory=mem)
        assert r.messages is mem and mem[0]["role"] == "user"

    async def test_async_retry_on_rate_limit(self):
        rl_err = openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None
        )
        c = MagicMock()
        c.chat.completions.create = AsyncMock(side_effect=[rl_err, _resp(content="ok")])
        agent = AsyncAgent(client=c, model="m", retry_base_delay=0.0)
        with patch("simple_agent.llm.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            r = await agent.arun("hi")
        assert r.output == "ok"
        sleep_mock.assert_awaited()


class TestAsyncToolDispatch:
    async def test_async_def_tool_awaited(self):
        async def aplus(x: int, y: int) -> int:
            return x + y

        c = _async_client(
            _resp(tool_calls=[_tc("c", "aplus", '{"x": 2, "y": 3}')]),
            _resp(content="5"),
        )
        agent = AsyncAgent(client=c, model="m", tools=[aplus])
        r = await agent.arun("add")
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "5"

    async def test_sync_def_tool_called_inline(self):
        def square(x: int) -> int:
            return x * x

        c = _async_client(
            _resp(tool_calls=[_tc("c", "square", '{"x": 4}')]),
            _resp(content="16"),
        )
        agent = AsyncAgent(client=c, model="m", tools=[square])
        r = await agent.arun("sq")
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "16"

    async def test_functools_partial_around_async_tool_awaited(self):
        async def underlying(x: int) -> int:
            return x + 100

        wrapped = functools.partial(underlying)
        c = _async_client(
            _resp(tool_calls=[_tc("c", "underlying", '{"x": 5}')]),
            _resp(content="105"),
        )
        agent = AsyncAgent(client=c, model="m", tools=[wrapped])
        r = await agent.arun("partial")
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "105"

    async def test_sync_tool_returning_awaitable_is_awaited(self):
        async def inner(x: int) -> int:
            return x - 1

        def make_awaitable(x: int) -> int:  # annotation lies; returns coroutine
            return inner(x)  # type: ignore[return-value]

        c = _async_client(
            _resp(tool_calls=[_tc("c", "make_awaitable", '{"x": 10}')]),
            _resp(content="9"),
        )
        agent = AsyncAgent(client=c, model="m", tools=[make_awaitable])
        r = await agent.arun("trick")
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "9"
