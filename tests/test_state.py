"""Tests for SQLite state persistence."""

import time

from simple_agent.state import StateStore


class TestStateStore:
    def setup_method(self):
        self.store = StateStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_save_and_retrieve_item(self):
        self.store.save_item("item-1", {"title": "Test"}, score=0.8)
        pending = self.store.get_pending_items()
        assert len(pending) == 1
        assert pending[0]["id"] == "item-1"
        assert pending[0]["score"] == 0.8

    def test_duplicate_item_ignored(self):
        self.store.save_item("item-1", {"title": "First"}, score=0.5)
        self.store.save_item("item-1", {"title": "Second"}, score=0.9)
        pending = self.store.get_pending_items()
        assert len(pending) == 1
        # First insert wins
        assert pending[0]["data"]["title"] == "First"

    def test_has_item(self):
        assert not self.store.has_item("item-1")
        self.store.save_item("item-1", {"title": "Test"})
        assert self.store.has_item("item-1")

    def test_item_status_update(self):
        self.store.save_item("item-1", {"title": "Test"})
        self.store.update_item_status("item-1", "drafted")
        pending = self.store.get_pending_items()
        assert len(pending) == 0

    def test_save_and_retrieve_draft(self):
        self.store.save_item("item-1", {"title": "Test"})
        draft_id = self.store.save_draft("item-1", "analyst", "Draft text here")
        assert draft_id is not None
        drafts = self.store.get_pending_drafts()
        assert len(drafts) == 1
        assert drafts[0]["persona"] == "analyst"

    def test_draft_status_update(self):
        self.store.save_item("item-1", {"title": "Test"})
        draft_id = self.store.save_draft("item-1", "analyst", "Draft text")
        self.store.update_draft_status(draft_id, "approved")
        pending = self.store.get_pending_drafts()
        assert len(pending) == 0

    def test_expire_stale_drafts(self):
        self.store.save_item("item-1", {"title": "Test"})
        self.store.save_draft("item-1", "analyst", "Old draft")
        # Manually backdate the draft
        self.store._conn.execute(
            "UPDATE drafts SET created_at = '2020-01-01T00:00:00+00:00'"
        )
        self.store._conn.commit()
        expired = self.store.expire_stale_drafts(hours=48)
        assert expired == 1
        pending = self.store.get_pending_drafts()
        assert len(pending) == 0

    def test_run_logging(self):
        run_id = self.store.start_run()
        assert run_id is not None
        self.store.finish_run(run_id, items_processed=5, drafts_created=2, errors=["test error"])
        row = self.store._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert row["items_processed"] == 5
        assert row["drafts_created"] == 2
        assert "test error" in row["errors"]

    def test_pending_items_sorted_by_score(self):
        self.store.save_item("low", {"title": "Low"}, score=0.2)
        self.store.save_item("high", {"title": "High"}, score=0.9)
        self.store.save_item("mid", {"title": "Mid"}, score=0.5)
        pending = self.store.get_pending_items()
        scores = [p["score"] for p in pending]
        assert scores == sorted(scores, reverse=True)
