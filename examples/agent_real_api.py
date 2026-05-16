"""Run a bare-bones Agent loop against a real LLM via OpenRouter.

Setup:
    1. Free API key at https://openrouter.ai/keys
    2. export OPENROUTER_API_KEY="<your-key>"
    3. python examples/agent_real_api.py

Demonstrates two tools: a sandboxed arithmetic evaluator and a constant lookup.
No `eval`/`exec` — the calculator uses an AST walker that only permits constants
and unary/binary numeric ops.
"""

import ast
import operator
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_agent import Agent, create_client

AGENT_MODEL = "google/gemma-4-31b-it:free"  # verified 2026-05-15

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a numeric arithmetic expression (e.g. '2 * (3 + 4)')."""
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree.body))


_CONSTANTS = {
    "speed_of_light": "2.998e8 m/s",
    "planck": "6.626e-34 J*s",
    "avogadro": "6.022e23 /mol",
    "gravity": "9.80665 m/s^2",
    "pi": "3.141592653589793",
}


def lookup_constant(name: str) -> str:
    """Return the value of a known physical or mathematical constant."""
    return _CONSTANTS.get(name, f"unknown constant: {name}")


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.", file=sys.stderr)
        return 1
    client = create_client(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    agent = Agent(
        client=client,
        model=AGENT_MODEL,
        tools=[calculate, lookup_constant],
        system_prompt="Use tools when you need to compute or look up values.",
        max_steps=6,
    )
    result = agent.run("What is Planck's constant times 1e34, plus pi?")
    print(f"Output:     {result.output}")
    print(f"Steps:      {result.steps}")
    print(f"Terminated: {result.terminated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
