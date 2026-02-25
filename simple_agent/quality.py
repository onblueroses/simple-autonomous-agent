"""Output quality rules and input sanitization.

The module most agent frameworks skip entirely. This handles two problems:
1. LLM output that reads like AI slop (em dashes, "delve", three-point lists)
2. Prompt injection in untrusted input that reaches the LLM

Ships with sensible defaults that catch the most common AI writing patterns.
These are a starting point - configure rules for your specific use case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityRule:
    """A single output quality rule backed by a regex pattern."""

    name: str
    pattern: str
    description: str
    severity: str = "warning"  # "warning" or "error"


@dataclass
class Violation:
    """A quality rule violation found in output text."""

    rule: str
    matched: str
    severity: str


def default_rules() -> list[QualityRule]:
    """Built-in anti-AI-slop rules. A starting point, not a final config.

    These catch the most recognizable AI writing patterns. Extend or
    replace them for your domain and language.
    """
    return [
        QualityRule(
            name="em_dash",
            pattern=r"[\u2014\u2013]",
            description="Em/en dashes are a strong AI writing signal. Use hyphens.",
        ),
        QualityRule(
            name="ai_vocabulary",
            pattern=r"\b(?:delve|crucial|landscape|tapestry|game-changer|paradigm shift|synergy)\b",
            description="Vocabulary that flags text as AI-generated.",
        ),
        QualityRule(
            name="three_point_list",
            pattern=r"(?:^|\n)\s*(?:1\.|[-*])\s+.+\n\s*(?:2\.|[-*])\s+.+\n\s*(?:3\.|[-*])\s+.+\s*$",
            description="Exactly three bullet points or numbered items is the strongest AI structural signal.",
        ),
        QualityRule(
            name="filler_opening",
            pattern=r"(?i)^(?:that'?s a (?:great|good|interesting|excellent) question|great question|interesting question)",
            description="Filler openings that add nothing and signal AI authorship.",
        ),
        QualityRule(
            name="filler_closing",
            pattern=r"(?i)(?:i hope this helps|feel free to (?:ask|reach out)|don'?t hesitate to|let me know if you (?:need|have|want))",
            description="Filler closings that LLMs append reflexively.",
        ),
        QualityRule(
            name="summary_closer",
            pattern=r"(?i)(?:^|\n)\s*(?:in (?:summary|conclusion)|to (?:sum|wrap) (?:up|it up)|overall,)",
            description="Summary closers that restate what was already said.",
        ),
        QualityRule(
            name="hedge_pile",
            pattern=r"(?i)\b(?:it'?s (?:important|worth) (?:to note|noting|mentioning)|it should be noted)\b",
            description="Hedging phrases that pad without adding information.",
        ),
        QualityRule(
            name="excessive_transition",
            pattern=r"(?i)\b(?:furthermore|moreover|additionally|in addition to this|it is also worth)\b",
            description="Transition words that LLMs overuse to connect paragraphs.",
        ),
    ]


def check_quality(text: str, rules: list[QualityRule]) -> list[Violation]:
    """Run quality rules against text. Returns all violations found."""
    violations = []
    for rule in rules:
        matches = re.findall(rule.pattern, text, re.MULTILINE)
        for match in matches:
            violations.append(Violation(
                rule=rule.name,
                matched=match if isinstance(match, str) else str(match),
                severity=rule.severity,
            ))
    return violations


# --- Input sanitization ---

_INJECTION_PATTERNS = [
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
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_input(text: str) -> str:
    """Strip prompt injection patterns from untrusted input."""
    return _INJECTION_RE.sub("[...]", text)


def validate_output(
    text: str,
    rules: list[QualityRule] | None = None,
    min_words: int = 20,
    max_words: int = 500,
) -> tuple[bool, list[str]]:
    """Validate output text for quality and length.

    Returns (passed, list_of_reasons). Passed is True only when
    no errors are found and word count is within bounds.
    """
    reasons = []
    word_count = len(text.split())

    if word_count < min_words:
        reasons.append(f"Too short: {word_count} words (minimum {min_words})")
    if word_count > max_words:
        reasons.append(f"Too long: {word_count} words (maximum {max_words})")

    if rules:
        violations = check_quality(text, rules)
        for v in violations:
            reasons.append(f"[{v.severity}] {v.rule}: matched '{v.matched}'")

    return (len(reasons) == 0, reasons)
