"""Tests for _extract_json sanitization passes (Phase 1.7)."""

import json

from simple_agent.pipeline import _extract_json


def test_strips_markdown_fence():
    raw = '```json\n{"score": 0.7}\n```'
    assert json.loads(_extract_json(raw)) == {"score": 0.7}


def test_handles_smart_quotes():
    raw = "{\u201cscore\u201d: 0.7}"
    assert json.loads(_extract_json(raw)) == {"score": 0.7}


def test_strips_line_comments():
    raw = '{\n  // a comment\n  "score": 0.7\n}'
    assert json.loads(_extract_json(raw)) == {"score": 0.7}


def test_removes_trailing_commas():
    raw = '{"score": 0.7, "reason": "ok",}'
    assert json.loads(_extract_json(raw)) == {"score": 0.7, "reason": "ok"}


def test_combined_defects():
    raw = '```json\n{\n  \u201cscore\u201d: 0.7,  // line comment\n  "reason": "ok",\n}\n```'
    assert json.loads(_extract_json(raw)) == {"score": 0.7, "reason": "ok"}


def test_url_with_double_slash_in_string_preserved():
    raw = '{"link": "https://example.com/path"}'
    assert json.loads(_extract_json(raw)) == {"link": "https://example.com/path"}
