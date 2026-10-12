"""Unit tests for Step 3: Universal Learning, User Commands, and Background Job State."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from trading_ai.core.command_parser import UserCommandParser
from trading_ai.core.command_service import UserCommandService
from trading_ai.core.source_resolver import SourceResolver
from trading_ai.schemas import (
    BackgroundJobState,
    CommandIntent,
    CorrectionRequest,
    JobStatus,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    PipelineStage,
    RuleSaveRequest,
    SourceMetadata,
    SourceResolutionRequest,
    SourceStatus,
    SourceType,
    UserCommand,
)
from trading_ai.storage import DatabaseManager, Repository


@pytest.fixture
def in_memory_repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def parser():
    return UserCommandParser()


@pytest.fixture
def command_service(in_memory_repo):
    return UserCommandService(repository=in_memory_repo)


@pytest.fixture
def source_resolver():
    return SourceResolver()


# =====================================================================
# 1. COMMAND PARSER TESTS
# =====================================================================
class TestCommandParser:
    def test_parse_save_rule(self, parser):
        text = "Save this rule: Always wait for 15m candle close before entry when market is in trend unless high-impact news."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.SAVE_RULE
        assert res.suggested_action == "save_rule"
        assert "Always wait for 15m candle close" in res.command.parameters.get("rule", "")
        assert res.command.parameters.get("conditions") is not None
        assert res.command.parameters.get("exceptions") is not None

    def test_parse_correction(self, parser):
        text = "No, correct it. Rule item_fvg123 applies only when 15m FVG has clear displacement."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.CORRECT
        assert res.suggested_action == "apply_correction"
        assert res.command.target_knowledge_id == "item_fvg123"

    def test_parse_you_are_wrong(self, parser):
        text = "You are wrong here. Premium vs Discount requires swing high to swing low."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.CORRECT

    def test_parse_remember(self, parser):
        text = "Remember this: Asian session high and low act as key liquidity pools."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.REMEMBER
        assert res.suggested_action == "remember"

    def test_parse_add_trading_knowledge(self, parser):
        text = "Add this to my trading knowledge: London open killzone starts at 02:00 NY time."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.ADD_KNOWLEDGE
        assert res.suggested_action == "add_knowledge"

    def test_parse_mark_exception(self, parser):
        text = "Mark this as an exception: During FOMC releases, standard stop hunts can exceed 20 pips."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.MARK_EXCEPTION
        assert res.suggested_action == "mark_exception"

    def test_parse_explain(self, parser):
        text = "Explain why this rule applies: item_rule_orderblock_45"
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.EXPLAIN
        assert res.command.target_knowledge_id == "item_rule_orderblock_45"

    def test_parse_forget(self, parser):
        text = "Forget this incorrect interpretation: item_rule_obsolete"
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.FORGET
        assert res.command.target_knowledge_id == "item_rule_obsolete"

    def test_parse_use_rule(self, parser):
        text = "Use this rule in future analysis for EURUSD."
        res = parser.parse(text)
        assert res.parsed_success is True
        assert res.command.intent == CommandIntent.USE_RULE

    def test_parse_empty_text(self, parser):
        res = parser.parse("   ")
        assert res.parsed_success is False
        assert res.command.intent == CommandIntent.UNKNOWN


# =====================================================================
# 2. SAVE RULE & KNOWLEDGE CREATION TESTS
# =====================================================================
class TestSaveRule:
    def test_save_rule_with_full_metadata(self, command_service, in_memory_repo):
        req = RuleSaveRequest(
            rule="Wait for break of structure before taking entry in direction of trend",
            conditions=["Higher timeframe bias is bullish", "15m candle closes above swing high"],
            exceptions=["High impact NFP release"],
            source="user",
            evidence="Lecturer stated in timestamp 14:20",
            job_id="job_test_101",
            confidence=0.95,
        )
        res = command_service.save_rule(req)
        assert res["status"] == "success"
        assert res["action"] == "saved_rule"
        assert res["version"] == 1
        assert res["confidence"] == 0.95

        # Verify in repository
        item_id = res["item_id"]
        items = in_memory_repo.get_knowledge("job_test_101", only_committed=False)
        assert len(items) == 1
        assert items[0].item_id == item_id

        # Verify version entry
        versions = in_memory_repo.get_knowledge_versions(item_id)
        assert len(versions) == 1
        assert versions[0].version_number == 1
        assert versions[0].is_active is True
        assert versions[0].conditions == req.conditions
        assert versions[0].exceptions == req.exceptions

        # Verify memory embedding
        emb = in_memory_repo.get_embedding_by_source("job_test_101", item_id)
        assert emb is not None
        assert "Wait for break of structure" in emb.text_content

    def test_save_rule_with_missing_parts_marked(self, command_service, in_memory_repo):
        # User provides rule without specifying conditions or exceptions
        req = RuleSaveRequest(
            rule="Always risk 1% per trade",
            conditions=None,
            exceptions=None,
            job_id="job_test_102",
        )
        res = command_service.save_rule(req)
        assert res["status"] == "success"
        # Must explicitly mark missing parts rather than inventing them
        assert res["conditions"] == ["[MISSING]"]
        assert res["exceptions"] == ["[NONE_SPECIFIED]"]

    def test_duplicate_save_command_prevents_duplicate_memory(self, command_service, in_memory_repo):
        req = RuleSaveRequest(
            rule="Fair Value Gap requires 3-candle sequence",
            job_id="job_test_dup",
        )
        res1 = command_service.save_rule(req)
        assert res1["status"] == "success"

        # Attempt to save exact same rule
        res2 = command_service.save_rule(req)
        assert res2["status"] == "exists"
        assert res2["action"] == "deduplicated"
        assert res2["item_id"] == res1["item_id"]

        # Verify only 1 item exists in repository
        items = in_memory_repo.get_knowledge("job_test_dup", only_committed=False)
        assert len(items) == 1


# =====================================================================
# 3. USER CORRECTION & AUDIT VERSIONING TESTS
# =====================================================================
class TestUserCorrection:
    def test_user_correction_versioning_and_provenance(self, command_service, in_memory_repo):
        # 1. First, save an initial rule (Version 1)
        init_req = RuleSaveRequest(
            rule="Rule X means price always reverses at previous daily high.",
            job_id="job_corr_test",
            created_by="initial_ai",
        )
        init_res = command_service.save_rule(init_req)
        item_id = init_res["item_id"]

        # 2. User says: "No, correct it. Rule X applies only when liquidity sweep occurs."
        corr_req = CorrectionRequest(
            target_knowledge_id=item_id,
            correction_statement="Rule X applies only when liquidity sweep occurs and MSS follows on 1m.",
            reason="Original interpretation ignored sweep and market structure shift prerequisite.",
            conditions=["Liquidity sweep above PDH", "1m MSS confirmation"],
            exceptions=["Major trend continuation days"],
            job_id="job_corr_test",
            corrected_by="trader_bob",
        )
        corr_res = command_service.apply_correction(corr_req)

        assert corr_res["status"] == "success"
        assert corr_res["action"] == "corrected"
        assert corr_res["version"] == 2
        assert corr_res["prior_statement"] == "Rule X means price always reverses at previous daily high."
        assert "liquidity sweep occurs" in corr_res["new_statement"]
        assert corr_res["corrected_by"] == "trader_bob"

        # 3. Verify audit history in repository: both versions exist
        versions = in_memory_repo.get_knowledge_versions(item_id)
        assert len(versions) == 2

        v1 = versions[0]
        assert v1.version_number == 1
        assert v1.is_active is False
        assert v1.superseded_by_version == 2
        assert "always reverses" in v1.statement

        v2 = versions[1]
        assert v2.version_number == 2
        assert v2.is_active is True
        assert v2.superseded_by_version is None
        assert "liquidity sweep occurs" in v2.statement
        assert v2.provenance["previously_believed"] == v1.statement
        assert v2.provenance["corrected_by"] == "trader_bob"

        # 4. Active item in knowledge table reflects Version 2
        active_v = in_memory_repo.get_active_knowledge_version(item_id)
        assert active_v is not None
        assert active_v.version_number == 2


# =====================================================================
# 4. SOURCE RESOLVER TESTS
# =====================================================================
class TestSourceResolver:
    def test_resolve_youtube_valid_url(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        res = source_resolver.resolve(req)
        assert res.is_accessible is True
        assert res.status == SourceStatus.AVAILABLE
        assert res.source_type == SourceType.YOUTUBE
        assert res.details.get("video_id") == "dQw4w9WgXcQ"

    def test_resolve_youtube_short_url(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://youtu.be/dQw4w9WgXcQ")
        res = source_resolver.resolve(req)
        assert res.is_accessible is True
        assert res.source_type == SourceType.YOUTUBE
        assert res.details.get("video_id") == "dQw4w9WgXcQ"

    def test_resolve_youtube_missing_id_returns_unavailable(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://www.youtube.com/watch")
        res = source_resolver.resolve(req)
        assert res.is_accessible is False
        assert res.status == SourceStatus.SOURCE_UNAVAILABLE

    def test_resolve_web_page(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://www.investopedia.com/trading-strategy-guide")
        res = source_resolver.resolve(req)
        assert res.is_accessible is True
        assert res.source_type == SourceType.WEB_PAGE
        assert res.status == SourceStatus.AVAILABLE

    def test_resolve_pdf_url(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://example.com/books/ict_handbook.pdf")
        res = source_resolver.resolve(req)
        assert res.is_accessible is True
        assert res.source_type == SourceType.PDF

    def test_resolve_blocked_domain_returns_unavailable(self, source_resolver):
        req = SourceResolutionRequest(source_location="https://malicious-site.org/lecture")
        res = source_resolver.resolve(req)
        assert res.is_accessible is False
        assert res.status == SourceStatus.SOURCE_UNAVAILABLE
        assert "restricted" in res.error_reason.lower()

    def test_resolve_local_file_exists(self, source_resolver):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"video content")
            tmp_path = tmp.name

        try:
            req = SourceResolutionRequest(source_location=tmp_path)
            res = source_resolver.resolve(req)
            assert res.is_accessible is True
            assert res.source_type == SourceType.LOCAL_VIDEO
            assert res.status == SourceStatus.AVAILABLE
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_resolve_local_file_not_found_returns_unavailable(self, source_resolver):
        req = SourceResolutionRequest(source_location="C:/nonexistent_folder/missing_video.mp4")
        res = source_resolver.resolve(req)
        assert res.is_accessible is False
        assert res.status == SourceStatus.SOURCE_UNAVAILABLE
        assert "does not exist" in res.error_reason.lower()

    def test_resolve_raw_user_text(self, source_resolver):
        req = SourceResolutionRequest(source_location="Order block is the last down candle before up move.")
        res = source_resolver.resolve(req)
        assert res.is_accessible is True
        assert res.source_type == SourceType.USER_TEXT
        assert res.status == SourceStatus.AVAILABLE


# =====================================================================
# 5. BACKGROUND JOB & PROCESS TRACKING TESTS
# =====================================================================
class TestBackgroundJobState:
    def test_job_state_transitions_and_progress(self, in_memory_repo):
        bg_job = BackgroundJobState(
            job_id="job_bg_1",
            source_id="src_test_1",
            status=JobStatus.QUEUED,
            progress=0.0,
            current_stage=PipelineStage.SOURCE,
            total_units=10,
            completed_units=0,
        )
        in_memory_repo.save_background_job(bg_job)

        # Transition to Processing
        bg_job.status = JobStatus.PROCESSING
        bg_job.current_stage = PipelineStage.AUDIO
        bg_job.current_serial = 1
        bg_job.completed_units = 1
        bg_job.progress = 10.0
        in_memory_repo.save_background_job(bg_job)

        loaded = in_memory_repo.get_background_job("job_bg_1")
        assert loaded is not None
        assert loaded.status == JobStatus.PROCESSING
        assert loaded.current_stage == PipelineStage.AUDIO
        assert loaded.progress == 10.0
        assert loaded.completed_units == 1

        # Pause job
        bg_job.status = JobStatus.PAUSED
        in_memory_repo.save_background_job(bg_job)
        assert in_memory_repo.get_background_job("job_bg_1").status == JobStatus.PAUSED

        # Resume and complete
        bg_job.status = JobStatus.PROCESSING
        bg_job.completed_units = 10
        bg_job.progress = 100.0
        bg_job.status = JobStatus.COMPLETED
        bg_job.current_stage = PipelineStage.COMPLETED
        in_memory_repo.save_background_job(bg_job)

        final_job = in_memory_repo.get_background_job("job_bg_1")
        assert final_job.status == JobStatus.COMPLETED
        assert final_job.progress == 100.0
        assert final_job.completed_units == 10


# =====================================================================
# 6. CONCEPTS & DOUBTS IDENTIFICATION TESTS
# =====================================================================
class TestConceptsAndDoubts:
    def test_concepts_and_doubts_retrieval(self, in_memory_repo):
        job_id = "job_concepts_doubts"

        # 1. Create a concept
        k_concept = KnowledgeItem(
            item_id="item_c1",
            job_id=job_id,
            item_type=KnowledgeType.CONCEPT,
            title="Fair Value Gap (FVG)",
            statement="A three-candle imbalance in price delivery.",
            validation_status=KnowledgeValidationStatus.VALIDATED,
            start_time=10.0,
            end_time=25.0,
        )
        in_memory_repo.save_knowledge_item(k_concept, serial_number=1, is_committed=True)

        # 2. Create an uncertain knowledge item (doubt)
        k_doubt = KnowledgeItem(
            item_id="item_d1",
            job_id=job_id,
            item_type=KnowledgeType.RULE,
            title="Unconfirmed Silver Bullet Setup",
            statement="Entries without NY lunch pause may have erratic fills.",
            reasoning="Speaker expressed uncertainty during timestamp 45:10",
            validation_status=KnowledgeValidationStatus.UNCERTAIN,
            confidence=0.4,
            start_time=40.0,
            end_time=55.0,
        )
        in_memory_repo.save_knowledge_item(k_doubt, serial_number=1, is_committed=True)

        # Query concepts
        concepts = in_memory_repo.get_concepts(job_id)
        assert len(concepts) >= 1
        assert any(c["title"] == "Fair Value Gap (FVG)" for c in concepts)

        # Query doubts
        doubts = in_memory_repo.get_doubts(job_id)
        assert len(doubts) >= 1
        assert any(d["title"] == "Unconfirmed Silver Bullet Setup" for d in doubts)
        assert any(d["status"] == "UNCERTAIN" for d in doubts)
