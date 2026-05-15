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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CAUSAL_CONNECTOR_RE = re.compile(r"(?i)\b(?:because|since|so|therefore|hence|thus)\b")


def _sentence_word_counts(text: str) -> list[int]:
    return [len(s.split()) for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _burstiness_violations(text: str) -> list[str]:
    """Flag 5-sentence windows with no short (<8 words) AND no long (>20 words) sentence.

    Real human prose has bursty length distribution. AI prose tends toward uniformity.
    """
    counts = _sentence_word_counts(text)
    out: list[str] = []
    for i in range(len(counts) - 4):
        window = counts[i : i + 5]
        if not any(w < 8 for w in window) or not any(w > 20 for w in window):
            out.append(f"sentences {i + 1}-{i + 5}: lengths {window}")
    return out


def _causal_connector_violations(text: str) -> list[str]:
    """Flag any 200-word window with fewer than 1 causal connector.

    AI text underuses 'because/since/so/therefore' by ~5x vs human baseline.
    """
    tokens = text.split()
    if len(tokens) < 200:
        return []
    out: list[str] = []
    step = 100
    for start in range(0, len(tokens) - 199, step):
        window = " ".join(tokens[start : start + 200])
        if len(_CAUSAL_CONNECTOR_RE.findall(window)) < 1:
            out.append(f"words {start + 1}-{start + 200}: 0 causal connectors")
    return out


# Sentinels used as the QualityRule pattern slot; the dispatch in check_quality
# routes by rule name to a real function.
_STATISTICAL_SENTINEL = r"(?!x)x"


def statistical_rules() -> list[QualityRule]:
    """Opt-in statistical rules. Slower than regex; not included in DEFAULT_RULES.

    Use when you have ~200+ words of output and want metrics beyond pattern matching.
    """
    return [
        QualityRule(
            "burstiness_check",
            _STATISTICAL_SENTINEL,
            "Sentence-length distribution is uniformly bland (AI signature).",
        ),
        QualityRule(
            "causal_connector_ratio",
            _STATISTICAL_SENTINEL,
            "Under-uses 'because/since/so/therefore' vs human baseline.",
        ),
    ]


def default_rules_de() -> list[QualityRule]:
    """Generic German AI-tell rules. Opt-in; not included in DEFAULT_RULES.

    All patterns are language-level (not domain-specific). Suitable for any
    German content; tune for your corpus.
    """
    return [
        QualityRule(
            "de_eroeffnungsformel",
            r"(?im)^\s*(?:in der heutigen|in einer welt,|stell dir vor,)",
            "Generic German LLM opening formula.",
        ),
        QualityRule(
            "de_schlussformel",
            r"(?i)(?:fazit:|zusammenfassend|abschlie\u00dfend l\u00e4sst sich)",
            "Generic German LLM closing formula.",
        ),
        QualityRule(
            "de_orientiert_basiert",
            r"\b\w+(?:orientiert|basiert)\w*\b",
            "-orientiert/-basiert Komposita overused in German LLM output.",
        ),
        QualityRule(
            "de_genitivkette",
            r"\b\w+s\s+\w+s\s+\w+s\b",
            "Three+ chained genitives reads as machine-translated.",
        ),
        QualityRule(
            "de_nominalisierung",
            r"\b\w+(?:ung|heit|keit|tät)\b.{1,40}\b\w+(?:ung|heit|keit|tät)\b.{1,40}\b\w+(?:ung|heit|keit|tät)\b",
            "Three+ abstract nominalizations in close proximity.",
        ),
        QualityRule(
            "de_dreierregel",
            r"(?i)\b\w+,\s+\w+\s+und\s+\w+\b",
            "Tight three-item enumeration; check for AI rule-of-three.",
        ),
        QualityRule(
            "de_anglizismus_buzz",
            r"(?i)\b(?:game[-\s]?changer|next[-\s]?level|key[-\s]?learnings?|state[-\s]?of[-\s]?the[-\s]?art)\b",
            "English buzz-phrases dropped into German LLM output.",
        ),
    ]


_STATISTICAL_DISPATCH = {
    "burstiness_check": _burstiness_violations,
    "causal_connector_ratio": _causal_connector_violations,
}


def check_quality(text: str, rules: list[QualityRule]) -> list[Violation]:
    violations: list[Violation] = []
    for rule in rules:
        if rule.name == "em_dash":
            for matched in _em_dash_violations(text):
                violations.append(
                    Violation(rule=rule.name, matched=matched, severity=rule.severity)
                )
            continue
        if rule.name in _STATISTICAL_DISPATCH:
            for matched in _STATISTICAL_DISPATCH[rule.name](text):
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
