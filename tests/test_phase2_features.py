"""Phase 2 feature tests: needs_search gate (C4) and per-persona rule override (C7)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from simple_agent.config import (
    AsyncPipelineConfig,
    ModelConfig,
    PipelineConfig,
)
from simple_agent.persona import Persona
from simple_agent.pipeline import arun_pipeline, run_pipeline
from simple_agent.quality import QualityRule


def _mock_sync_clients(score: str, reason: str, draft: str):
    mock_scorer = MagicMock()
    mock_scorer.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=score))]
    )
    mock_writer = MagicMock()
    mock_writer.chat.completions.create.side_effect = [
        MagicMock(
            choices=[MagicMock(message=MagicMock(content=reason, reasoning=None))]
        ),
        MagicMock(choices=[MagicMock(message=MagicMock(content=draft))]),
    ]
    return mock_scorer, mock_writer


def _mock_async_clients(score: str, reason: str, draft: str):
    mock_scorer = MagicMock()
    mock_scorer.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=score))])
    )
    mock_writer = MagicMock()
    mock_writer.chat.completions.create = AsyncMock(
        side_effect=[
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=reason, reasoning=None))]
            ),
            MagicMock(choices=[MagicMock(message=MagicMock(content=draft))]),
        ]
    )
    return mock_scorer, mock_writer


_LONG_DRAFT = " ".join(["software"] * 30)
_ITEM = {"id": "demo", "text": "Compare a few audit tools."}


class TestNeedsSearch:
    def test_sync_gate_false_skips_grounding(self):
        scorer, writer = _mock_sync_clients('{"score": 0.8}', "analysis", _LONG_DRAFT)
        ground_fn = MagicMock(return_value="GROUND_DATA")
        gate = MagicMock(return_value=False)

        config = PipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
            ground_fn=ground_fn,
            needs_search=gate,
        )
        result = run_pipeline(_ITEM, config)

        gate.assert_called_once()
        ground_fn.assert_not_called()
        assert result.grounding == ""

    def test_sync_gate_true_calls_grounding(self):
        scorer, writer = _mock_sync_clients('{"score": 0.8}', "analysis", _LONG_DRAFT)
        ground_fn = MagicMock(return_value="GROUND_DATA")
        gate = MagicMock(return_value=True)

        config = PipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
            ground_fn=ground_fn,
            needs_search=gate,
        )
        result = run_pipeline(_ITEM, config)

        gate.assert_called_once()
        ground_fn.assert_called_once()
        assert result.grounding == "GROUND_DATA"

    def test_sync_no_gate_preserves_default(self):
        scorer, writer = _mock_sync_clients('{"score": 0.8}', "analysis", _LONG_DRAFT)
        ground_fn = MagicMock(return_value="GROUND_DATA")
        config = PipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
            ground_fn=ground_fn,
        )
        result = run_pipeline(_ITEM, config)
        ground_fn.assert_called_once()
        assert result.grounding == "GROUND_DATA"

    @pytest.mark.asyncio
    async def test_async_gate_false_skips_grounding(self):
        scorer, writer = _mock_async_clients('{"score": 0.8}', "analysis", _LONG_DRAFT)
        ground_fn = AsyncMock(return_value="GROUND_DATA")
        gate = MagicMock(return_value=False)

        config = AsyncPipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
            ground_fn=ground_fn,
            needs_search=gate,
        )
        result = await arun_pipeline(_ITEM, config)
        ground_fn.assert_not_called()
        assert result.grounding == ""

    @pytest.mark.asyncio
    async def test_async_gate_supports_awaitable_predicate(self):
        scorer, writer = _mock_async_clients('{"score": 0.8}', "analysis", _LONG_DRAFT)
        ground_fn = AsyncMock(return_value="GROUND_DATA")
        gate = AsyncMock(return_value=False)

        config = AsyncPipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
            ground_fn=ground_fn,
            needs_search=gate,
        )
        result = await arun_pipeline(_ITEM, config)
        gate.assert_called_once()
        ground_fn.assert_not_called()
        assert result.grounding == ""


class TestPerPersonaQualityRules:
    def _config_for_persona(self, draft_text: str):
        scorer, writer = _mock_sync_clients('{"score": 0.8}', "analysis", draft_text)
        return PipelineConfig(
            scorer=ModelConfig("s"),
            reasoner=ModelConfig("r"),
            writer=ModelConfig("w"),
            scorer_client=scorer,
            writer_client=writer,
        )

    def test_persona_rules_override_default(self):
        persona = Persona(
            name="strict",
            identity="Strict reviewer.",
            voice="Terse.",
            expertise=["nothing"],
            quality_rules=[
                QualityRule(
                    name="always_fail",
                    pattern=r".+",
                    description="catches everything",
                )
            ],
        )
        clean_text = (
            "The quarterly report shows a 3% revenue increase year over year. "
            + " ".join(["word"] * 25)
        )
        config = self._config_for_persona(clean_text)
        result = run_pipeline(_ITEM, config, personas=[persona])

        assert result.passed_quality is False
        assert any("always_fail" in e for e in result.errors)
        assert all("ai_vocabulary" not in e for e in result.errors)
        assert all("prompt_leak" not in e for e in result.errors)
        assert result.draft == clean_text

    def test_persona_without_rules_uses_default(self):
        persona = Persona(
            name="default",
            identity="ordinary",
            voice="ordinary",
            expertise=["nothing"],
        )
        text = "Let me delve into this crucial landscape. " + " ".join(["word"] * 25)
        config = self._config_for_persona(text)
        result = run_pipeline(_ITEM, config, personas=[persona])
        assert any("ai_vocabulary" in e for e in result.errors)
