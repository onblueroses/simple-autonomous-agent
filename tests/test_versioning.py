"""Tests for simple_agent.versioning (Phase 2 / C6)."""

import string


from simple_agent import compute_prompt_hash


def test_deterministic():
    assert compute_prompt_hash("x") == compute_prompt_hash("x")


def test_differs_for_differing_inputs():
    assert compute_prompt_hash("x") != compute_prompt_hash("y")


def test_length_and_charset():
    h = compute_prompt_hash("anything")
    assert len(h) == 16
    assert set(h) <= set(string.hexdigits.lower())


def test_unicode_safe():
    h = compute_prompt_hash("\u201ccurly\u201d \u00fcber")
    assert len(h) == 16


def test_empty_input():
    assert compute_prompt_hash("") == compute_prompt_hash("")
    assert len(compute_prompt_hash("")) == 16
