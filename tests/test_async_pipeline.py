from unittest.mock import AsyncMock, MagicMock

from simple_agent.config import AsyncPipelineConfig, ModelConfig
from simple_agent.pipeline import arun_batch, arun_pipeline
from simple_agent.quality import default_rules
from simple_agent.state import StateStore


def _make_async_config(
    score_response: str = '{"score": 0.8, "reason": "relevant"}',
    reason_response: str = "Core question is about investment returns.",
    draft_response: str = "The yield at 4.2% is below market average for comparable assets.",
    score_threshold: float = 0.6,
) -> AsyncPipelineConfig:
    mock_scorer = MagicMock()
    mock_scorer.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=score_response))])
    )

    mock_writer = MagicMock()
    mock_writer.chat.completions.create = AsyncMock(side_effect=[
        # First call: areason()
        MagicMock(choices=[MagicMock(message=MagicMock(content=reason_response, reasoning=None))]),
        # Second call: adraft()
        MagicMock(choices=[MagicMock(message=MagicMock(content=draft_response))]),
    ])

    return AsyncPipelineConfig(
        scorer=ModelConfig(model="test-scorer"),
        reasoner=ModelConfig(model="test-reasoner"),
        writer=ModelConfig(model="test-writer"),
        scorer_client=mock_scorer,
        writer_client=mock_writer,
        score_threshold=score_threshold,
    )


SAMPLE_ITEM = {"id": "test-1", "text": "What is a good investment yield in 2026?"}


class TestArunPipeline:
    async def test_full_successful_flow(self):
        config = _make_async_config()
        result = await arun_pipeline(SAMPLE_ITEM, config)
        assert result.item_id == "test-1"
        assert result.score == 0.8
        assert result.draft == "The yield at 4.2% is below market average for comparable assets."
        assert result.reasoning == "Core question is about investment returns."

    async def test_score_below_threshold_returns_early(self):
        config = _make_async_config(
            score_response='{"score": 0.3, "reason": "not relevant"}',
            score_threshold=0.6,
        )
        result = await arun_pipeline(SAMPLE_ITEM, config)
        assert result.score == 0.3
        assert result.draft == ""
        assert result.passed_quality is False

    async def test_async_grounding(self):
        async def mock_ground(query: str) -> str:
            return "Berlin rental yields average 3.5% in 2026."

        config = _make_async_config()
        config.ground_fn = mock_ground
        result = await arun_pipeline(SAMPLE_ITEM, config)
        assert result.grounding == "Berlin rental yields average 3.5% in 2026."
        assert result.draft != ""

    async def test_grounding_failure_doesnt_block_draft(self):
        async def failing_ground(query: str) -> str:
            raise ConnectionError("Search API is down")

        config = _make_async_config()
        config.ground_fn = failing_ground
        result = await arun_pipeline(SAMPLE_ITEM, config)
        assert result.draft != ""
        assert any("grounding" in e for e in result.errors)

    async def test_persists_to_state_store(self):
        config = _make_async_config()
        store = StateStore(":memory:")
        _result = await arun_pipeline(SAMPLE_ITEM, config, state=store)
        assert store.has_item("test-1")
        drafts = store.get_pending_drafts()
        assert len(drafts) == 1
        store.close()

    async def test_quality_violations_recorded(self):
        config = _make_async_config(
            draft_response="Let me delve into this. I hope this helps! " + " ".join(["word"] * 30),
        )
        config.quality_rules = default_rules()
        result = await arun_pipeline(SAMPLE_ITEM, config)
        assert any("ai_vocabulary" in e for e in result.errors)


class TestArunBatch:
    async def test_processes_all_items(self):
        items = [
            {"id": "a", "text": "First item"},
            {"id": "b", "text": "Second item"},
        ]

        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))])
        )

        mock_writer = MagicMock()
        mock_writer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(
                content="A well-reasoned response with concrete details and specific numbers.",
                reasoning=None,
            ))])
        )

        config = AsyncPipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        results = await arun_batch(items, config)
        assert len(results) == 2
        assert all(r.draft == "A well-reasoned response with concrete details and specific numbers." for r in results)

    async def test_preserves_order(self):
        items = [{"id": f"item-{i}", "text": f"Item {i}"} for i in range(5)]

        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))])
        )

        mock_writer = MagicMock()
        mock_writer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(
                content="Draft output text here with enough words for validation.",
                reasoning=None,
            ))])
        )

        config = AsyncPipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        results = await arun_batch(items, config, max_concurrency=2)
        assert [r.item_id for r in results] == [f"item-{i}" for i in range(5)]

    async def test_logs_run_to_state(self):
        items = [{"id": "a", "text": "Test"}]

        mock_scorer = MagicMock()
        mock_scorer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "reason": "yes"}'))])
        )
        mock_writer = MagicMock()
        mock_writer.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(
                content="Draft output text here.",
                reasoning=None,
            ))])
        )

        config = AsyncPipelineConfig(
            scorer=ModelConfig(model="test"),
            reasoner=ModelConfig(model="test"),
            writer=ModelConfig(model="test"),
            scorer_client=mock_scorer,
            writer_client=mock_writer,
        )
        store = StateStore(":memory:")
        await arun_batch(items, config, state=store)

        row = store._conn.execute("SELECT * FROM runs").fetchone()
        assert row is not None
        assert row["items_processed"] == 1
        store.close()
