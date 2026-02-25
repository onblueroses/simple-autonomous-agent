"""Output quality rules and input sanitization.

Why regex over LLM-based quality checks: Quality validation runs on every
output, so it must be deterministic, instant, and free. Regex rules catch
the most common AI writing patterns (em dashes, "delve/crucial/landscape",
exactly-three-item lists, filler openings and closings) without an API call.
These patterns are empirically the strongest signals of AI authorship - a
human reader spots them in seconds.

Why sanitize_input matters: When untrusted text (user content, scraped data)
goes into LLM prompts, prompt injection is a real risk. The sanitization
strips common injection patterns ("ignore previous instructions", fake XML
tags, role-switching attempts) before the text reaches any model. It's not
foolproof, but it catches the obvious attacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityRule:
    """A quality rule backed by a regex pattern."""

    name: str
    pattern: str
    description: str
    severity: str = "warning"


@dataclass
class Violation:
    """A quality rule match found in text."""

    rule: str
    matched: str
    severity: str


# These are a starting point. Tune for your domain and language.
DEFAULT_RULES: tuple[QualityRule, ...] = (
    QualityRule("em_dash", r"[\u2014\u2013]", "Em/en dashes signal AI authorship."),
    QualityRule("ai_vocabulary", r"\b(?:delve|crucial|landscape|tapestry|game-changer|paradigm shift|synergy)\b", "High-frequency AI vocabulary."),
    QualityRule("three_point_list", r"(?:^|\n)\s*(?:1\.|[-*])\s+.+\n\s*(?:2\.|[-*])\s+.+\n\s*(?:3\.|[-*])\s+.+\s*$", "Exactly three items is the strongest AI structural signal."),
    QualityRule("filler_opening", r"(?i)^(?:that'?s a (?:great|good|interesting|excellent) question|great question|interesting question)", "Filler openings."),
    QualityRule("filler_closing", r"(?i)(?:i hope this helps|feel free to (?:ask|reach out)|don'?t hesitate to|let me know if you (?:need|have|want))", "Reflexive LLM closings."),
    QualityRule("summary_closer", r"(?i)(?:^|\n)\s*(?:in (?:summary|conclusion)|to (?:sum|wrap) (?:up|it up)|overall,)", "Restates what was already said."),
    QualityRule("hedge_pile", r"(?i)\b(?:it'?s (?:important|worth) (?:to note|noting|mentioning)|it should be noted)\b", "Hedging that pads without adding info."),
    QualityRule("excessive_transition", r"(?i)\b(?:furthermore|moreover|additionally|in addition to this|it is also worth)\b", "Overused LLM transition words."),
)


def default_rules() -> list[QualityRule]:
    """Return a mutable copy of the built-in quality rules."""
    return list(DEFAULT_RULES)


def check_quality(text: str, rules: list[QualityRule]) -> list[Violation]:
    """Run rules against text. Returns all violations."""
    violations = []
    for rule in rules:
        for match in re.findall(rule.pattern, text, re.MULTILINE):
            violations.append(Violation(
                rule=rule.name,
                matched=match if isinstance(match, str) else str(match),
                severity=rule.severity,
            ))
    return violations


# --- Input sanitization ---

_INJECTION_RE = re.compile("|".join([
    r"ignore (?:all )?(?:previous |prior )?instructions",
    r"forget (?:all )?(?:your )?(?:previous )?instructions",
    r"disregard (?:all )?(?:previous |prior )?(?:instructions|context)",
    r"(?:print|show|reveal|output) (?:your )?system (?:prompt|message|instructions)",
    r"what (?:are|is) your (?:system )?(?:prompt|instructions)",
    r"act as (?:a |an )?",
    r"you are now (?:a |an )?",
    r"new (?:persona|role|identity|character):",
    r"from now on,? (?:you |your )",
    r"</?(?:system|assistant|instruction|prompt)>",
]), re.IGNORECASE)


def sanitize_input(text: str) -> str:
    """Strip prompt injection patterns from untrusted input."""
    return _INJECTION_RE.sub("[...]", text)


def validate_output(
    text: str,
    rules: list[QualityRule] | None = None,
    min_words: int = 20,
    max_words: int = 500,
) -> tuple[bool, list[str]]:
    """Check output length and quality. Returns (passed, reasons)."""
    reasons = []
    word_count = len(text.split())

    if word_count < min_words:
        reasons.append(f"Too short: {word_count} words (minimum {min_words})")
    if word_count > max_words:
        reasons.append(f"Too long: {word_count} words (maximum {max_words})")

    if rules:
        for v in check_quality(text, rules):
            reasons.append(f"[{v.severity}] {v.rule}: matched '{v.matched}'")

    return (len(reasons) == 0, reasons)
