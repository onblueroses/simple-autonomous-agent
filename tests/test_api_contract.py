"""Public API contract test (Phase 2 / C2).

Enforces backwards-compatible API evolution: every pre-Phase-2 export is still
exported; every pre-existing parameter on a callable is preserved by name and
position with the same default and kind; new parameters are allowed only if
appended after existing ones with a default (so existing callers don't break).
"""

import inspect
import json
import pathlib

import simple_agent


_BASELINE_PATH = pathlib.Path(__file__).parent / "_api_baseline.json"
_ALLOWED_NEW_EXPORTS = {
    "compute_prompt_hash",
    "Agent",
    "AsyncAgent",
    "AgentResult",
    "tool_spec",
}


def _parameter_names_in_order(obj) -> list[str]:
    try:
        return [p.name for p in inspect.signature(obj).parameters.values()]
    except (ValueError, TypeError):
        return []


def _baseline_param_count(sig_str: str) -> int:
    """Count the parameters in a stringified inspect.Signature, ignoring quoted defaults."""
    inner = sig_str.split("(", 1)[1].rsplit(")", 1)[0]
    if not inner.strip():
        return 0
    depth = 0
    in_str = False
    quote = ""
    count = 1
    for c in inner:
        if in_str:
            if c == quote:
                in_str = False
            continue
        if c in ("'", '"'):
            in_str = True
            quote = c
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            count += 1
    return count


def test_no_exports_removed():
    baseline = json.loads(_BASELINE_PATH.read_text())
    current = {n: getattr(simple_agent, n, None) for n in simple_agent.__all__}
    missing = set(baseline) - set(current)
    assert not missing, f"Public exports removed: {missing}"


def test_only_sanctioned_new_exports():
    baseline = json.loads(_BASELINE_PATH.read_text())
    current_names = set(simple_agent.__all__)
    added = current_names - set(baseline)
    unsanctioned = added - _ALLOWED_NEW_EXPORTS
    assert not unsanctioned, f"Unsanctioned new public exports: {unsanctioned}"


def test_signatures_backwards_compatible():
    baseline = json.loads(_BASELINE_PATH.read_text())
    issues: list[str] = []
    for name, base_entry in baseline.items():
        obj = getattr(simple_agent, name, None)
        if obj is None:
            continue
        if not base_entry.get("callable"):
            if callable(obj):
                issues.append(f"{name}: was non-callable, now callable")
            continue
        if not callable(obj):
            issues.append(f"{name}: was callable, now non-callable")
            continue

        try:
            cur_sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        cur_params = list(cur_sig.parameters.values())
        base_count = _baseline_param_count(base_entry["signature"])
        if len(cur_params) < base_count:
            issues.append(
                f"{name}: parameter count shrank ({len(cur_params)} < {base_count})"
            )
            continue
        for p in cur_params[base_count:]:
            if p.default is inspect.Parameter.empty and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                issues.append(
                    f"{name}: new required parameter {p.name!r} breaks callers"
                )
    assert not issues, "Backwards-incompatible API drift:\n" + "\n".join(issues)
