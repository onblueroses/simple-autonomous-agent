"""Output quality rules and input sanitization.

Regex over LLM-based quality checks: validation runs on every output, so it
must be deterministic, instant, and free.
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


# AI-vocabulary words occur at 5-25x human baseline frequency in LLM output.
# Verified against PubMed/GPTZero corpora as of 2026-05. Tune for your domain.
_AI_VOCABULARY = (
    "delve",
    "crucial",
    "landscape",
    "tapestry",
    "game-changer",
    "paradigm shift",
    "synergy",
    "additionally",
    "comprehensive",
    "enhance",
    "facilitate",
    "testament",
    "underscore",
    "showcase",
    "vibrant",
    "beacon",
    "pivotal",
    "meticulous",
    "intricate",
    "noteworthy",
    "palpable",
    "multifaceted",
    "realm",
    "harness",
    "navigate",
    "leverage",
    "robust",
    "seamless",
    "cornerstone",
    "embark",
)


_AI_VOCABULARY_PATTERN = (
    r"\b(?:" + "|".join(re.escape(w) for w in _AI_VOCABULARY) + r")\b"
)


# Em-dash sentinel used by the in-process counter; never matches in text.
_EM_DASH_COUNT_SENTINEL = r"(?!x)x"


def _em_dash_violations(text: str) -> list[str]:
    """Flag paragraphs (separated by blank lines) containing 3+ em or en dashes.

    One or two dashes per paragraph is normal human writing.
    """
    out: list[str] = []
    for para in text.split("\n\n"):
        dashes = re.findall(r"[\u2014\u2013]", para)
        if len(dashes) >= 3:
            out.append("".join(dashes))
    return out


DEFAULT_RULES: tuple[QualityRule, ...] = (
    QualityRule(
        "em_dash",
        _EM_DASH_COUNT_SENTINEL,
        "3+ em/en dashes per paragraph signals AI authorship.",
    ),
    QualityRule(
        "ai_vocabulary", _AI_VOCABULARY_PATTERN, "High-frequency AI vocabulary."
    ),
    QualityRule(
        "three_point_list",
        r"(?:^|\n)\s*(?:1\.|[-*])\s+.+\n\s*(?:2\.|[-*])\s+.+\n\s*(?:3\.|[-*])\s+.+\s*$",
        "Exactly three items is the strongest AI structural signal.",
    ),
    QualityRule(
        "filler_opening",
        r"(?i)^(?:that'?s a (?:great|good|interesting|excellent) question|great question|interesting question)",
        "Filler openings.",
    ),
    QualityRule(
        "filler_closing",
        r"(?i)(?:i hope this helps|feel free to (?:ask|reach out)|don'?t hesitate to|let me know if you (?:need|have|want))",
        "Reflexive LLM closings.",
    ),
    QualityRule(
        "summary_closer",
        r"(?i)(?:^|\n)\s*(?:in (?:summary|conclusion)|to (?:sum|wrap) (?:up|it up)|overall,)",
        "Restates what was already said.",
    ),
    QualityRule(
        "hedge_pile",
        r"(?i)\b(?:it'?s (?:important|worth) (?:to note|noting|mentioning)|it should be noted)\b",
        "Hedging that pads without adding info.",
    ),
    QualityRule(
        "excessive_transition",
        r"(?i)\b(?:furthermore|moreover|in addition to this|it is also worth)\b",
        "Overused LLM transition words.",
    ),
    QualityRule(
        "vague_attribution",
        r"(?i)\b(?:experts? (?:argue|say|claim)|industry reports|several sources|some observers|studies show|it is believed)\b",
        "Unsourced authority appeals.",
    ),
    QualityRule(
        "negative_parallelism",
        r"(?i)\bit'?s not (?:just|merely|simply|about)\b.{1,80}\bit'?s\b",
        "'It's not X, it's Y' is a strong AI structural tell.",
    ),
    QualityRule(
        "copula_avoidance",
        r"(?i)\b(?:serves as|stands as|marks a|represents a|exemplifies)\b",
        "Inflated copula substitutes.",
    ),
    QualityRule(
        "knowledge_cutoff_disclaimer",
        r"(?i)(?:as of my (?:last )?(?:update|training)|while (?:specific )?details are (?:limited|scarce)|based on (?:my )?available information)",
        "Model self-reference / cutoff disclaimer.",
    ),
    QualityRule(
        "generic_positive_conclusion",
        r"(?i)(?:the future looks (?:bright|promising)|exciting times (?:lie ahead|are ahead)|only time will tell|a step in the right direction)",
        "Vacuous upbeat closings.",
    ),
    QualityRule(
        "prompt_leak",
        r"(?i)(?:I am an AI|as an AI(?: language model)?|I cannot (?:help|assist)|I don'?t have access|</s>|\[INST\]|\[/INST\]|<\|im_(?:start|end)\|>|my training data|my knowledge cutoff)",
        "Model self-reference / system-prompt fragment leaked into output.",
    ),
)


def default_rules() -> list[QualityRule]:
    """Mutable copy so callers can append without mutating the module constant."""
    return list(DEFAULT_RULES)


def check_quality(text: str, rules: list[QualityRule]) -> list[Violation]:
    violations: list[Violation] = []
    for rule in rules:
        if rule.name == "em_dash":
            for matched in _em_dash_violations(text):
                violations.append(
                    Violation(rule=rule.name, matched=matched, severity=rule.severity)
                )
            continue
        for match in re.findall(rule.pattern, text, re.MULTILINE):
            violations.append(
                Violation(
                    rule=rule.name,
                    matched=match if isinstance(match, str) else str(match),
                    severity=rule.severity,
                )
            )
    return violations


_INJECTION_RE = re.compile(
    "|".join(
        [
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
    ),
    re.IGNORECASE,
)


def sanitize_input(text: str) -> str:
    return _INJECTION_RE.sub("[...]", text)


def validate_output(
    text: str,
    rules: list[QualityRule] | None = None,
    min_words: int = 20,
    max_words: int = 500,
) -> list[str]:
    """Check output length and quality. Returns failure reasons (empty = passed)."""
    reasons = []
    word_count = len(text.split())

    if word_count < min_words:
        reasons.append(f"Too short: {word_count} words (minimum {min_words})")
    if word_count > max_words:
        reasons.append(f"Too long: {word_count} words (maximum {max_words})")

    if rules:
        for v in check_quality(text, rules):
            reasons.append(f"[{v.severity}] {v.rule}: matched '{v.matched}'")

    return reasons
