"""Tests for the pipeline orchestration.

All LLM calls are mocked - no real API calls. Tests focus on:
- Score threshold filtering
- Fault tolerance (stages failing independently)
- Full flow producing populated results
- Quality violations being recorded
"""

import json
from unittest.mock import MagicMock

from simple_agent.config import ModelConfig, PipelineConfig
from simple_agent.persona import Persona
from simple_agent.pipeline import _extract_json, run_batch, run_pipeline
from simple_agent.quality import default_rules
from simple_agent.state import StateStore


def _make_config(
    score_response: str = '{"score": 0.8, "reason": "relevant"}',
    reason_response: str = "Core question is about investment returns.",
    draft_response: str = "The yield at 4.2% is below market average for comparable assets.",
    score_threshold: float = 0.6,
) -> PipelineConfig:
    """Create a PipelineConfig with mocked LLM clients."""
    mock_scorer = MagicMock()
    mock_scorer.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=score_response))]
    )

    mock_writer = MagicMock()

    # Writer client handles both reason() and draft() calls
    # We use side_effect to return different responses for sequential calls
    mock_writer.chat.completions.create.side_effect = [
        # First call: reason()
        MagicMock(choices=[MagicMock(message=MagicMock(content=reason_response, reasoning=None))]),
        # Second call: draft()
        MagicMock(choices=[MagicMock(message=MagicMock(content=draft_response))]),
    ]

    return PipelineConfig(
        scorer=ModelConfig(model="test-scorer"),
        reasoner=ModelConfig(model="test-reasoner"),
        writer=ModelConfig(model="test-writer"),
        scorer_client=mock_scorer,
        writer_client=mock_writer,
        score_threshold=score_threshold,
    )


SAMPLE_ITEM = {"id": "test-1", "text": "What is a good investment yield in 2026?"}


class TestExtractJson:
    def test_clean_json_passthrough(self):
        raw = '{"score": 0.8, "reason": "relevant"}'
        assert _extract_json(raw) == raw

    def test_json_fenced_with_language(self):
        raw = '```json\n{"score": 0.8, "reason": "relevant"}\n```'
        assert json.loads(_extract_json(raw)) == {"score": 0.8, "reason": "relevant"}

    def test_json_fenced_bare(self):
        raw = '```\n{"score": 0.8}\n```'
        assert json.loads(_extract_json(raw)) == {"score": 0.8}

    def test_json_with_preamble(self):
        raw = 'Here is the result:\n{"score": 0.8, "reason": "good"}'
        assert json.loads(_extract_json(raw)) == {"score": 0.8, "reason": "good"}

    def test_non_json_passthrough(self):
        raw = "This is just plain text with no JSON."
        assert _extract_json(raw) == raw

    def test_nested_json_objects(self):
        raw = '{"outer": {"inner": 1}, "score": 0.5}'
        assert json.loads(_extract_json(raw)) == {"outer": {"inner": 1}, "score": 0.5}

    def test_whitespace_around_fences(self):
        raw = '  ```json\n  {"score": 0.9}  \n```  '
        assert json.loads(_extract_json(raw)) == {"score": 0.9}


class TestRunPipeline:
    def test_full_successful_flow(self):
        config = _make_config()
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.item_id == "test-1"
        assert result.score == 0.8
        assert result.draft != ""
        assert result.reasoning != ""

    def test_score_below_threshold_returns_early(self):
        config = _make_config(
            score_response='{"score": 0.3, "reason": "not relevant"}',
            score_threshold=0.6,
        )
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.score == 0.3
        assert result.draft == ""
        assert result.passed_quality is False

    def test_grounding_failure_doesnt_block_draft(self):
        def failing_ground(query: str) -> str:
            raise ConnectionError("Search API is down")

        config = _make_config()
        config.ground_fn = failing_ground
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.draft != ""
        assert any("grounding" in e for e in result.errors)

    def test_reasoning_failure_doesnt_block_draft(self):
        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))]
        )

        mock_writer = MagicMock()
        # First call (reason) raises, second call (draft) succeeds
        mock_writer.chat.completions.create.side_effect = [
            Exception("Reasoning model timeout"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Draft without reasoning context."))]),
        ]

        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.draft == "Draft without reasoning context."
        assert result.reasoning == ""
        assert any("reasoning" in e for e in result.errors)

    def test_quality_violations_recorded(self):
        config = _make_config(
            draft_response="Let me delve into this. I hope this helps! " + " ".join(["word"] * 30),
        )
        config.quality_rules = default_rules()
        result = run_pipeline(SAMPLE_ITEM, config)
        assert any("ai_vocabulary" in e for e in result.errors)

    def test_persona_selection_with_multiple(self):
        personas = [
            Persona(name="Analyst", identity="Analyst.", voice="Analytical.", expertise=["Finance"]),
            Persona(name="Researcher", identity="Researcher.", voice="Academic.", expertise=["Research"]),
        ]

        mock_scorer = MagicMock()
        # First call: score, second call: persona selection
        mock_scorer.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"persona": "Researcher", "reason": "academic topic"}'))]),
        ]

        mock_writer = MagicMock()
        mock_writer.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Reasoning output.", reasoning=None))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="The evidence suggests a moderate effect."))]),
        ]

        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        result = run_pipeline(SAMPLE_ITEM, config, personas=personas)
        assert result.persona == "Researcher"

    def test_persists_to_state_store(self):
        config = _make_config()
        store = StateStore(":memory:")
        _result = run_pipeline(SAMPLE_ITEM, config, state=store)
        assert store.has_item("test-1")
        drafts = store.get_pending_drafts()
        assert len(drafts) == 1
        store.close()

    def test_fenced_json_score_response(self):
        config = _make_config(
            score_response='```json\n{"score": 0.85, "reason": "very relevant"}\n```',
        )
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.score == 0.85
        assert result.draft != ""

    def test_custom_scorer_prompt_template(self):
        custom_template = "Is this about finance? {content}\nReturn JSON: {\"score\": 1.0}"
        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))]
        )
        mock_writer = MagicMock()
        mock_writer.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Reasoning.", reasoning=None))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="A draft with enough words to pass validation checks easily here."))]),
        ]
        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
            scorer_prompt_template=custom_template,
        )
        result = run_pipeline(SAMPLE_ITEM, config)
        # Verify the custom prompt was actually used
        call_args = mock_scorer.chat.completions.create.call_args
        prompt_used = call_args.kwargs["messages"][0]["content"]
        assert "Is this about finance?" in prompt_used
        assert result.score == 0.9

    def test_scoring_failure_handled_gracefully(self):
        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create.side_effect = Exception("API error")

        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=MagicMock(),
        )
        result = run_pipeline(SAMPLE_ITEM, config)
        assert result.score == 0.0
        assert any("scoring" in e for e in result.errors)


class TestRunBatch:
    def test_processes_all_items(self):
        items = [
            {"id": "a", "text": "First item"},
            {"id": "b", "text": "Second item"},
        ]

        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))]
        )

        mock_writer = MagicMock()
        mock_writer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A well-reasoned response with concrete details and specific numbers.", reasoning=None))]
        )

        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        results = run_batch(items, config, delay=0)
        assert len(results) == 2
        assert all(r.draft != "" for r in results)

    def test_logs_run_to_state(self):
        items = [{"id": "a", "text": "Test"}]

        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))]
        )
        mock_writer = MagicMock()
        mock_writer.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Draft output text here.", reasoning=None))]
        )

        config = PipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        store = StateStore(":memory:")
        run_batch(items, config, state=store, delay=0)

        row = store._conn.execute("SELECT * FROM runs").fetchone()
        assert row is not None
        assert row["items_processed"] == 1
        store.close()
