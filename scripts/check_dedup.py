"""C3 verifier for sync/async dedup.

Uses `ast.unparse` to canonicalize each function body before comparing, so
formatter-induced line splits don't count as logic divergence. Normalizes
the inherent sync vs async differences (the `await` keyword and the
`a`-prefixed function names), then counts the lines that still differ.
A pair passes if the larger side has at most MAX_UNIQUE_LINES lines not
present in the partner.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

MAX_UNIQUE_LINES = 4

PAIRS = [
    ("simple_agent/llm.py", "score", "ascore"),
    ("simple_agent/llm.py", "reason", "areason"),
    ("simple_agent/llm.py", "draft", "adraft"),
    ("simple_agent/llm.py", "_retry_llm_call", "_async_retry_llm_call"),
    ("simple_agent/pipeline.py", "run_pipeline", "arun_pipeline"),
    ("simple_agent/pipeline.py", "run_batch", "arun_batch"),
    ("simple_agent/agent.py", "run", "arun"),
]


class _AwaitStripper(ast.NodeTransformer):
    """Drop the `await` keyword; the inner expression stays."""

    def visit_Await(self, node: ast.Await):
        self.generic_visit(node)
        return node.value


_NAME_REWRITES = {
    "ascore": "score",
    "areason": "reason",
    "adraft": "draft",
    "_async_retry_llm_call": "_retry_llm_call",
    "_batch_iter_async": "_batch_iter_sync",
    "AsyncPipelineConfig": "PipelineConfig",
    "arun_pipeline": "run_pipeline",
    "arun_batch": "run_batch",
    "acreate_client": "create_client",
}


class _NameRewriter(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name):
        if node.id in _NAME_REWRITES:
            return ast.copy_location(
                ast.Name(id=_NAME_REWRITES[node.id], ctx=node.ctx), node
            )
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if node.attr in _NAME_REWRITES:
            return ast.copy_location(
                ast.Attribute(
                    value=node.value, attr=_NAME_REWRITES[node.attr], ctx=node.ctx
                ),
                node,
            )
        return node


class _AsyncStmtNormalizer(ast.NodeTransformer):
    """`async def` -> `def`, `async with` -> `with`, `async for` -> `for`."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        return ast.copy_location(
            ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self.generic_visit(node)
        return ast.copy_location(ast.With(items=node.items, body=node.body), node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.generic_visit(node)
        return ast.copy_location(
            ast.For(
                target=node.target, iter=node.iter, body=node.body, orelse=node.orelse
            ),
            node,
        )


def _find_func(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


def _normalize_body(func_node) -> list[str]:
    """Return the function body as a list of canonicalized, dedented lines."""
    body = list(func_node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    module = ast.Module(body=body, type_ignores=[])
    module = _AsyncStmtNormalizer().visit(module)
    module = _AwaitStripper().visit(module)
    module = _NameRewriter().visit(module)
    ast.fix_missing_locations(module)
    src = ast.unparse(module)
    out = []
    for line in src.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        out.append(s)
    return out


def _count_unique(a: list[str], b: list[str]) -> tuple[int, int, list[str], list[str]]:
    set_a, set_b = set(a), set(b)
    only_a = [x for x in a if x not in set_b]
    only_b = [x for x in b if x not in set_a]
    return len(only_a), len(only_b), only_a, only_b


def main() -> int:
    failures = 0
    print(f"check_dedup: MAX_UNIQUE_LINES = {MAX_UNIQUE_LINES}")
    print("-" * 60)
    for rel, sync_name, async_name in PAIRS:
        path = REPO_ROOT / rel
        tree = ast.parse(path.read_text())
        sync_fn = _find_func(tree, sync_name)
        async_fn = _find_func(tree, async_name)
        if sync_fn is None or async_fn is None:
            print(f"[ERROR] {rel}::{sync_name}/{async_name}: function(s) not found.")
            failures += 1
            continue
        sync_lines = _normalize_body(sync_fn)
        async_lines = _normalize_body(async_fn)
        n_only_sync, n_only_async, only_sync, only_async = _count_unique(
            sync_lines, async_lines
        )
        worst = max(n_only_sync, n_only_async)
        status = "PASS" if worst <= MAX_UNIQUE_LINES else "FAIL"
        print(
            f"[{status}] {sync_name}/{async_name}: "
            f"{n_only_sync} unique sync line(s), {n_only_async} unique async line(s); worst = {worst}."
        )
        if status == "FAIL":
            failures += 1
            print(f"  only in sync:  {only_sync}")
            print(f"  only in async: {only_async}")
    print("-" * 60)
    if failures:
        print(f"check_dedup: {failures} pair(s) FAILED.")
        return 1
    print(f"check_dedup: all {len(PAIRS)} pairs within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
