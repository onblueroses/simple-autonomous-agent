"""Phase 3: opt-in statistical rules + German rules (not in DEFAULT_RULES)."""

from simple_agent.quality import (
    check_quality,
    default_rules,
    default_rules_de,
    statistical_rules,
)


class TestStatisticalRules:
    def test_not_in_default_rules(self):
        default = {r.name for r in default_rules()}
        for r in statistical_rules():
            assert r.name not in default

    def test_burstiness_flags_uniform_length(self):
        # 5 sentences, all 12-15 words: no <8-word AND no >20-word -> violation
        text = " ".join(
            [
                "The system processes data through multiple stages every minute carefully and methodically with care.",
                "Each step in the workflow is independently validated against schema rules and tests defined upstream.",
                "Configuration files describe the model parameters needed for the next stage of processing tasks reliably.",
                "Errors are surfaced with descriptive messages and structured payloads to allow downstream alerting systems.",
                "Tests run against mocked LLM calls so no external API key is required during development cycles or CI.",
            ]
        )
        v = check_quality(text, statistical_rules())
        assert any(x.rule == "burstiness_check" for x in v)

    def test_burstiness_passes_bursty_text(self):
        # Every 5-sentence window must contain at least one <8-word AND one >20-word sentence.
        short = "Tiny."
        long_sentence = (
            "Here is a much longer sentence that contains well over twenty words "
            "to make this paragraph bursty and pass the variance check easily."
        )
        text = " ".join([short, long_sentence] * 5)
        v = check_quality(text, statistical_rules())
        assert not any(x.rule == "burstiness_check" for x in v)

    def test_causal_connector_flags_zero_connectors(self):
        text = " ".join(["alpha"] * 220)
        v = check_quality(text, statistical_rules())
        assert any(x.rule == "causal_connector_ratio" for x in v)

    def test_causal_connector_passes_with_connectors(self):
        chunks = (
            ["alpha"] * 100 + ["because"] + ["alpha"] * 100 + ["since"] + ["alpha"] * 20
        )
        text = " ".join(chunks)
        v = check_quality(text, statistical_rules())
        assert not any(x.rule == "causal_connector_ratio" for x in v)


class TestGermanRules:
    def test_not_in_default_rules(self):
        default = {r.name for r in default_rules()}
        for r in default_rules_de():
            assert r.name not in default

    def test_eroeffnungsformel(self):
        v = check_quality(
            "In der heutigen schnelllebigen Welt ist alles digital.", default_rules_de()
        )
        assert any(x.rule == "de_eroeffnungsformel" for x in v)

    def test_schlussformel(self):
        v = check_quality(
            "Abschließend lässt sich sagen, dass das Projekt erfolgreich war.",
            default_rules_de(),
        )
        assert any(x.rule == "de_schlussformel" for x in v)

    def test_orientiert_basiert(self):
        v = check_quality(
            "Ein kundenorientierter und datenbasierter Ansatz.", default_rules_de()
        )
        assert any(x.rule == "de_orientiert_basiert" for x in v)

    def test_genitivkette(self):
        v = check_quality(
            "Die Auswirkungen des Berichts des Vorstands des Unternehmens.",
            default_rules_de(),
        )
        assert any(x.rule == "de_genitivkette" for x in v)

    def test_anglizismus_buzz(self):
        v = check_quality(
            "Das ist ein echter Game-Changer für unsere Industrie.", default_rules_de()
        )
        assert any(x.rule == "de_anglizismus_buzz" for x in v)

    def test_clean_german_passes(self):
        text = "Der Bericht zeigt eine Umsatzsteigerung von drei Prozent im Vergleich zum Vorjahr."
        v = check_quality(text, default_rules_de())
        assert all(x.rule != "de_eroeffnungsformel" for x in v)
        assert all(x.rule != "de_schlussformel" for x in v)
