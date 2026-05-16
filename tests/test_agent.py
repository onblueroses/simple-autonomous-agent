"""Tests for the sync Agent (run)."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import openai
import pytest

from simple_agent.agent import (
    Agent,
    AgentResult,
    _build_tool_specs,
    _format_tool_result,
    tool_spec,
)


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _resp(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_msg(content, tool_calls))])


def _tc(call_id: str, name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _client(*responses):
    c = MagicMock()
    c.chat.completions.create.side_effect = list(responses)
    return c


class TestToolSpec:
    def test_basic_int_str_schema(self):
        def add(a: int, b: str) -> str: ...

        spec = tool_spec(add)
        params = spec["function"]["parameters"]
        assert params["properties"] == {
            "a": {"type": "integer"},
            "b": {"type": "string"},
        }
        assert sorted(params["required"]) == ["a", "b"]

    def test_optional_param_not_required(self):
        def f(a: int, b: Optional[str] = None) -> str: ...

        spec = tool_spec(f)
        assert spec["function"]["parameters"]["required"] == ["a"]
        assert spec["function"]["parameters"]["properties"]["b"] == {"type": "string"}

    def test_pep604_none_union_not_required(self):
        def f(a: int, b: str | None = None) -> str: ...

        spec = tool_spec(f)
        assert spec["function"]["parameters"]["required"] == ["a"]

    def test_default_value_not_required(self):
        def f(a: int, b: int = 7) -> int: ...

        spec = tool_spec(f)
        assert spec["function"]["parameters"]["required"] == ["a"]

    def test_list_param_schema(self):
        def f(items: list[str]) -> int: ...

        spec = tool_spec(f)
        assert spec["function"]["parameters"]["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_untyped_param_raises_TypeError(self):
        def f(a, b: int): ...

        with pytest.raises(TypeError, match="no type annotation"):
            tool_spec(f)

    def test_unsupported_type_dict_raises_TypeError(self):
        def f(a: dict) -> str: ...

        with pytest.raises(TypeError, match="not supported"):
            tool_spec(f)

    def test_partial_without_dunder_name_uses_underlying(self):
        def underlying(x: int) -> int:
            """Underlying tool."""
            return x

        wrapped = functools.partial(underlying)
        spec = tool_spec(wrapped)
        assert spec["function"]["name"] == "underlying"

    def test_description_from_docstring_first_line(self):
        def f(a: int) -> str:
            """First line of doc.

            Second paragraph ignored.
            """
            return ""

        spec = tool_spec(f)
        assert spec["function"]["description"] == "First line of doc."


class TestBuildToolSpecs:
    def test_callable_list_builds_specs_and_lookup(self):
        def alpha(x: int) -> int: ...
        def beta(y: str) -> str: ...

        specs, by_name = _build_tool_specs([alpha, beta])
        assert [s["function"]["name"] for s in specs] == ["alpha", "beta"]
        assert by_name["alpha"] is alpha and by_name["beta"] is beta

    def test_non_callable_raises_TypeError(self):
        with pytest.raises(TypeError, match="not callable"):
            _build_tool_specs([{"type": "function"}])  # type: ignore[list-item]


class TestFormatToolResult:
    def test_str_passthrough(self):
        assert _format_tool_result("hi") == "hi"

    def test_bytes_decoded(self):
        assert _format_tool_result(b"hello") == "hello"

    def test_none_empty_string(self):
        assert _format_tool_result(None) == ""

    def test_dict_json_dumps(self):
        assert _format_tool_result({"a": 1}) == '{"a": 1}'

    def test_list_json_dumps(self):
        assert _format_tool_result([1, "x"]) == '[1, "x"]'

    def test_int_json_dumps(self):
        assert _format_tool_result(42) == "42"

    def test_unsupported_falls_back_to_str(self):
        class Foo:
            def __repr__(self):
                return "Foo()"

        assert _format_tool_result(Foo()) == "Foo()"


class TestAgentLoop:
    def test_done_no_tools_one_step(self):
        c = _client(_resp(content="hello"))
        agent = Agent(client=c, model="m")
        r = agent.run("say hi")
        assert r.output == "hello" and r.steps == 1 and r.terminated == "done"

    def test_one_tool_call_then_final(self):
        def double(x: int) -> int:
            """Double x."""
            return x * 2

        c = _client(
            _resp(tool_calls=[_tc("c1", "double", '{"x": 21}')]),
            _resp(content="42"),
        )
        agent = Agent(client=c, model="m", tools=[double])
        r = agent.run("double 21")
        assert r.output == "42" and r.steps == 2 and r.terminated == "done"
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "42" and tool_msgs[0]["tool_call_id"] == "c1"

    def test_multi_tool_call_one_response_counts_one_step(self):
        def echo(x: str) -> str:
            return x

        c = _client(
            _resp(
                tool_calls=[
                    _tc("a", "echo", '{"x": "A"}'),
                    _tc("b", "echo", '{"x": "B"}'),
                ]
            ),
            _resp(content="done"),
        )
        agent = Agent(client=c, model="m", tools=[echo])
        r = agent.run("multi")
        assert r.steps == 2 and r.terminated == "done"
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert [m["content"] for m in tool_msgs] == ["A", "B"]

    def test_two_separate_tool_responses_then_final(self):
        def echo(x: str) -> str:
            return x

        c = _client(
            _resp(tool_calls=[_tc("a", "echo", '{"x": "1"}')]),
            _resp(tool_calls=[_tc("b", "echo", '{"x": "2"}')]),
            _resp(content="ok"),
        )
        agent = Agent(client=c, model="m", tools=[echo])
        r = agent.run("two")
        assert r.steps == 3 and r.terminated == "done"

    def test_tool_error_becomes_observation_loop_continues(self):
        def boom(x: int) -> int:
            raise RuntimeError("nope")

        c = _client(
            _resp(tool_calls=[_tc("c", "boom", '{"x": 1}')]),
            _resp(content="recovered"),
        )
        agent = Agent(client=c, model="m", tools=[boom])
        r = agent.run("try")
        assert r.terminated == "done" and r.output == "recovered"
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"].startswith("Error: ")

    def test_max_steps_termination(self):
        def loop_tool(x: int) -> int:
            return x

        c = _client(
            _resp(tool_calls=[_tc("a", "loop_tool", '{"x": 1}')]),
            _resp(tool_calls=[_tc("b", "loop_tool", '{"x": 2}')]),
        )
        agent = Agent(client=c, model="m", tools=[loop_tool], max_steps=2)
        r = agent.run("never finish")
        assert r.steps == 2 and r.terminated == "max_steps"

    def test_no_progress_termination(self):
        c = _client(_resp(content=None))
        agent = Agent(client=c, model="m")
        r = agent.run("?")
        assert r.steps == 1 and r.terminated == "no_progress" and r.output == ""

    def test_system_prompt_first_message(self):
        c = _client(_resp(content="ok"))
        agent = Agent(client=c, model="m", system_prompt="be terse")
        r = agent.run("hi")
        assert r.messages[0] == {"role": "system", "content": "be terse"}
        assert r.messages[1] == {"role": "user", "content": "hi"}

    def test_memory_appended_in_place_and_is_result_messages(self):
        c = _client(_resp(content="ok"))
        agent = Agent(client=c, model="m")
        mem: list[dict] = []
        r = agent.run("hello", memory=mem)
        assert r.messages is mem
        assert mem[0] == {"role": "user", "content": "hello"}
        assert mem[-1]["role"] == "assistant"

    def test_repeated_runs_share_memory_dont_duplicate_system(self):
        c = _client(_resp(content="one"), _resp(content="two"))
        agent = Agent(client=c, model="m", system_prompt="S")
        mem: list[dict] = []
        agent.run("first", memory=mem)
        agent.run("second", memory=mem)
        sys_count = sum(1 for m in mem if m.get("role") == "system")
        assert sys_count == 1
        user_count = sum(1 for m in mem if m.get("role") == "user")
        assert user_count == 2

    def test_memory_with_existing_system_message_not_overridden(self):
        c = _client(_resp(content="ok"))
        agent = Agent(client=c, model="m", system_prompt="agent_system")
        mem = [{"role": "system", "content": "caller_system"}]
        agent.run("go", memory=mem)
        sys_msgs = [m for m in mem if m.get("role") == "system"]
        assert len(sys_msgs) == 1 and sys_msgs[0]["content"] == "caller_system"

    def test_max_steps_zero_raises_ValueError(self):
        with pytest.raises(ValueError, match="max_steps"):
            Agent(client=MagicMock(), model="m", max_steps=0)

    def test_max_steps_negative_raises_ValueError(self):
        with pytest.raises(ValueError, match="max_steps"):
            Agent(client=MagicMock(), model="m", max_steps=-1)

    def test_retry_on_rate_limit_then_success(self):
        rl_err = openai.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None
        )
        c = MagicMock()
        c.chat.completions.create.side_effect = [rl_err, _resp(content="ok")]
        agent = Agent(client=c, model="m", retry_base_delay=0.0)
        with patch("simple_agent.llm.time.sleep") as sleep_mock:
            r = agent.run("hi")
        assert r.output == "ok" and r.terminated == "done"
        sleep_mock.assert_called()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_sync_agent_rejects_awaitable_result(self):
        async def async_tool(x: int) -> int:
            return x

        c = _client(_resp(tool_calls=[_tc("c", "async_tool", '{"x": 1}')]))
        agent = Agent(client=c, model="m", tools=[async_tool])
        with pytest.raises(TypeError, match="awaitable"):
            agent.run("oops")


class TestAgentResultDataclass:
    def test_fields(self):
        r = AgentResult(output="x", messages=[], steps=1, terminated="done")
        assert r.output == "x" and r.steps == 1
