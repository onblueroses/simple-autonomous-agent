"""Minimal Agent: model + callable tools + bounded loop.

Each `run(task, memory=...)` (or async `arun(...)`) builds `[system?] + memory[...] +
user(task)`, calls the model, dispatches tool_calls if present, and loops until the
model returns final content or `max_steps` is hit. `run` and `arun` share core
helpers; the divergence is the LLM call boundary and the per-tool await step.

Termination reasons: "done" (content + no tool_calls), "max_steps", "no_progress".
Memory is caller-owned and mutated in place. Tool errors become observations.
`run` (sync) rejects awaitable tool results; `arun` (async) awaits them.
"""

from __future__ import annotations

import functools
import inspect
import json
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable

from .llm import _async_retry_llm_call, _retry_llm_call


@dataclass
class AgentResult:
    output: str
    messages: list[dict]
    steps: int
    terminated: str


_SCALAR_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_for_hints(fn: Callable) -> Callable:
    if isinstance(fn, functools.partial):
        return fn.func
    return getattr(fn, "__wrapped__", fn)


def _split_optional(annotation: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(args) == 2 and len(non_none) == 1:
            return True, non_none[0]
    return False, annotation


def _schema_for(annotation: Any, fn_name: str, p_name: str) -> dict:
    _, inner = _split_optional(annotation)
    if inner in _SCALAR_TYPES:
        return {"type": _SCALAR_TYPES[inner]}
    if typing.get_origin(inner) is list:
        args = typing.get_args(inner)
        if not args:
            return {"type": "array"}
        _, item = _split_optional(args[0])
        if item in _SCALAR_TYPES:
            return {"type": "array", "items": {"type": _SCALAR_TYPES[item]}}
    raise TypeError(
        f"Tool {fn_name!r} parameter {p_name!r}: type {annotation!r} is not supported."
    )


def _tool_name(fn: Callable) -> str:
    name = getattr(fn, "__name__", None)
    if name:
        return name
    unwrapped = _unwrap_for_hints(fn)
    name = getattr(unwrapped, "__name__", None)
    if name:
        return name
    raise TypeError(
        f"Tool {fn!r} has no __name__. Wrap it in a def or assign __name__ explicitly."
    )


def tool_spec(fn: Callable) -> dict:
    """Build an OpenAI tool spec dict for a single callable (D4, D9)."""
    sig = inspect.signature(fn)
    name = _tool_name(fn)
    try:
        hints = typing.get_type_hints(_unwrap_for_hints(fn))
    except Exception:
        hints = {}
    properties: dict[str, dict] = {}
    required: list[str] = []
    for p_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p_name not in hints:
            raise TypeError(
                f"Tool {name!r} parameter {p_name!r} has no type annotation."
            )
        annotation = hints[p_name]
        properties[p_name] = _schema_for(annotation, name, p_name)
        is_opt, _ = _split_optional(annotation)
        if param.default is inspect.Parameter.empty and not is_opt:
            required.append(p_name)
    description = ""
    if fn.__doc__:
        for line in fn.__doc__.strip().splitlines():
            if line.strip():
                description = line.strip()
                break
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _build_tool_specs(
    tools: list[Callable],
) -> tuple[list[dict], dict[str, Callable]]:
    specs: list[dict] = []
    by_name: dict[str, Callable] = {}
    for t in tools:
        if not callable(t):
            raise TypeError(
                f"Tool {t!r} is not callable. v0.3.0 accepts callables only."
            )
        spec = tool_spec(t)
        by_name[spec["function"]["name"]] = t
        specs.append(spec)
    return specs, by_name


def _format_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if result is None:
        return ""
    if isinstance(result, (dict, list, tuple, int, float, bool)):
        try:
            return json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


def _format_tool_error(exc: BaseException) -> str:
    return f"Error: {exc!r}"


def _initial_messages(
    system_prompt: str | None, memory: list[dict] | None, task: str
) -> list[dict]:
    if memory is None:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": task})
        return msgs
    if system_prompt and (not memory or memory[0].get("role") != "system"):
        memory.insert(0, {"role": "system", "content": system_prompt})
    memory.append({"role": "user", "content": task})
    return memory


def _assistant_dict(msg: Any) -> dict:
    out: dict = {"role": "assistant", "content": msg.content}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ]
    return out


def _tool_message(tc: Any, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc.id, "content": content}


def _resolve_tool_call(
    call: Any, by_name: dict[str, Callable]
) -> tuple[Any, BaseException | None]:
    fn = by_name.get(call.function.name)
    if fn is None:
        return None, KeyError(f"Tool {call.function.name!r} not registered.")
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return None, e
    try:
        return fn(**args), None
    except Exception as e:  # noqa: BLE001
        return None, e


class _AgentBase:
    def __init__(
        self,
        client: Any,
        model: str,
        tools: list[Callable] | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ):
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._specs, self._tools = _build_tool_specs(tools or [])

    def _call_kwargs(self, messages: list[dict]) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self._specs:
            kwargs["tools"] = self._specs
        return kwargs


class Agent(_AgentBase):
    def run(self, task: str, *, memory: list[dict] | None = None) -> AgentResult:
        messages = _initial_messages(self.system_prompt, memory, task)
        last_msg: Any = None
        for step in range(self.max_steps):
            resp = _retry_llm_call(
                self.client.chat.completions.create,
                max_retries=self.max_retries,
                base_delay=self.retry_base_delay,
                **self._call_kwargs(messages),
            )
            msg = resp.choices[0].message
            last_msg = msg
            messages.append(_assistant_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    result, err = _resolve_tool_call(tc, self._tools)
                    if err is not None:
                        content = _format_tool_error(err)
                    elif inspect.isawaitable(result):
                        raise TypeError(
                            f"Tool {tc.function.name!r} returned awaitable; use AsyncAgent."
                        )
                    else:
                        content = _format_tool_result(result)
                    messages.append(_tool_message(tc, content))
                continue
            if msg.content:
                return AgentResult(msg.content, messages, step + 1, "done")
            return AgentResult("", messages, step + 1, "no_progress")
        return AgentResult(
            (last_msg.content if last_msg else "") or "",
            messages,
            self.max_steps,
            "max_steps",
        )


class AsyncAgent(_AgentBase):
    async def arun(self, task: str, *, memory: list[dict] | None = None) -> AgentResult:
        messages = _initial_messages(self.system_prompt, memory, task)
        last_msg: Any = None
        for step in range(self.max_steps):
            resp = await _async_retry_llm_call(
                self.client.chat.completions.create,
                max_retries=self.max_retries,
                base_delay=self.retry_base_delay,
                **self._call_kwargs(messages),
            )
            msg = resp.choices[0].message
            last_msg = msg
            messages.append(_assistant_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    result, err = _resolve_tool_call(tc, self._tools)
                    if err is not None:
                        content = _format_tool_error(err)
                    else:
                        if inspect.isawaitable(result):
                            result = await result
                        content = _format_tool_result(result)
                    messages.append(_tool_message(tc, content))
                continue
            if msg.content:
                return AgentResult(msg.content, messages, step + 1, "done")
            return AgentResult("", messages, step + 1, "no_progress")
        return AgentResult(
            (last_msg.content if last_msg else "") or "",
            messages,
            self.max_steps,
            "max_steps",
        )
