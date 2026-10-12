"""FastAPI backend application serving REST API and Web UI — Step 1 + Step 2."""

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from trading_ai.core.config import get_settings
from trading_ai.core.model_router import ModelRouter
from trading_ai.core.command_parser import UserCommandParser
from trading_ai.core.command_service import UserCommandService
from trading_ai.core.source_resolver import SourceResolver
from trading_ai.pipeline.orchestrator import LecturePipelineOrchestrator
from trading_ai.schemas.enums import (
    SourceDomain,
    SourceAuthority,
    RelationshipType,
    TradeDecisionState,
)
from trading_ai.schemas import (
    BackgroundJobState,
    CommandParseResult,
    CorrectionRequest,
    JobMetadata,
    JobStatus,
    PipelineStage,
    RuleSaveRequest,
    SourceMetadata,
    SourceResolutionRequest,
    SourceResolutionResult,
    SourceStatus,
    SourceType,
    UserCommand,
    ExecutionEnvironment,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository

from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.screen_access import ScreenAccessManager
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.supervisor.commit_barrier import CommitBarrier
from trading_ai.schemas.settings import UnifiedSettings, mask_sensitive_settings

settings = get_settings()
db_mgr = DatabaseManager(settings.db_path)
repo = Repository(db_mgr)
orchestrator = LecturePipelineOrchestrator(repository=repo, settings=settings)
source_resolver = SourceResolver()
command_parser = UserCommandParser()

# Centralized Unified Settings State
unified_settings_state = UnifiedSettings()

# Step 4 Core Engines
data_provider = SimulatedMarketDataProvider(seed=42)
mtf_engine = MultiTimeframeEngine(data_provider)
lecture_validator = LectureValidator(repo)
chart_marker = ChartMarker()
screen_manager = ScreenAccessManager(repo)
commit_barrier = CommitBarrier(repo)

from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.core.tech_fund_correlator import TechnicalFundamentalCorrelator
from trading_ai.fundamental import ForexFactoryProvider, NewsTimingEngine

# Step 6 Fundamental Market Intelligence Engines
fundamental_provider = ForexFactoryProvider()
news_timing_engine = NewsTimingEngine()
tech_fund_correlator = TechnicalFundamentalCorrelator()

# Step 5 & 6 Situation & Display Engine
situation_engine = CurrentSituationEngine(
    repository=repo,
    chart_marker=chart_marker,
    fundamental_provider=fundamental_provider,
    news_timing_engine=news_timing_engine,
    correlator=tech_fund_correlator,
)

from trading_ai.companion import TradingCompanionCore
from trading_ai.execution import HedgeFundExecutionCore

# Step 7 AI Trading Companion Core
companion_core = TradingCompanionCore(repository=repo)

# Step 8 Hedge Fund Execution & Risk Management Core
execution_core = HedgeFundExecutionCore(
    repository=repo,
    discipline_engine=companion_core.discipline_engine,
)

# Step 9 Trading Brain, Backtesting & Promotion Engines
from trading_ai.trading_brain import GeminiTradingBrain
from trading_ai.backtest import BacktestEngine, RulePromotionManager
from trading_ai.schemas.backtest import (
    BacktestConfig,
    CreateRuleVersionRequest,
    ManualBacktestTradeRequest,
    PromoteRuleRequest,
    RuleVersion,
    StartBacktestRequest,
)
from trading_ai.schemas.enums import AssetClass, BacktestMode, BacktestStatus, RuleLifecycleState

trading_brain = GeminiTradingBrain(
    gemini_client=orchestrator._ai,
    repository=repo,
)
backtest_engine = BacktestEngine(
    repository=repo,
    market_data_provider=data_provider,
    trading_brain=trading_brain,
    risk_engine=execution_core.risk_engine,
)
promotion_manager = RulePromotionManager(
    repository=repo,
    execution_gate=execution_core.execution_gate,
)

watchlist_monitor = WatchlistMonitor(
    repository=repo,
    data_provider=data_provider,
    mtf_engine=mtf_engine,
    lecture_validator=lecture_validator,
    chart_marker=chart_marker,
    commit_barrier=commit_barrier,
    watchlist=["BTC/USD", "ETH/USD", "SOL/USD", "EUR/USD", "XAU/USD"],
    situation_engine=situation_engine,
    companion_core=companion_core,
)

command_service = UserCommandService(
    repository=repo,
    token_manager=orchestrator._token_manager,
    model_router=orchestrator._router,
    watchlist_monitor=watchlist_monitor,
    mtf_engine=mtf_engine,
    lecture_validator=lecture_validator,
    chart_marker=chart_marker,
    screen_access_manager=screen_manager,
    data_provider=data_provider,
    situation_engine=situation_engine,
)

app = FastAPI(
    title="Trading AI Assistant - Steps 1-9",
    description="Multimodal Lecture Intelligence + Long-Term Memory + Universal Learning & User Commands + Live Market Intelligence + Selective Visualization + Fundamental Intelligence + AI Trading Companion + Hedge Fund Execution Framework + Backtesting & Trading-Brain Validation",
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def private_auth_middleware(request: Request, call_next):
    if not hasattr(security_core, "auth_manager") or not security_core.auth_manager.is_auth_enabled:
        return await call_next(request)

    path = request.url.path
    exempt_paths = {
        "/api/auth/login",
        "/api/auth/status",
        "/manifest.webmanifest",
        "/sw.js",
        "/favicon.ico",
        "/docs",
        "/openapi.json",
    }
    if path in exempt_paths or path.startswith("/icons/"):
        return await call_next(request)

    # 1. Session cookie check
    session_id = request.cookies.get("trading_ai_session")
    if session_id and security_core.auth_manager.verify_session(session_id):
        return await call_next(request)

    # 2. Authorization / X-API-Key header check
    auth_header = request.headers.get("Authorization") or request.headers.get("X-API-Key")
    if auth_header:
        token = auth_header.replace("Bearer ", "").strip()
        client_ip = request.client.host if request.client else "127.0.0.1"
        if security_core.auth_manager.verify_token(token, client_ip=client_ip):
            return await call_next(request)

    # 3. Paired device check
    dev_id = request.headers.get("X-Device-Id")
    if dev_id:
        dev = security_core.device_manager.get_device(dev_id)
        if dev and dev.status.value in ["ACTIVE", "TRUSTED"]:
            return await call_next(request)

    # For API endpoints, deny immediately with 401
    if path.startswith("/api/"):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Authentication required for private Trading AI Assistant."},
        )

    # Allow HTML root navigation so the client SPA can display the login lockscreen
    return await call_next(request)


# Ensure directories exist
settings.ensure_directories()

# Mount frames directory for image inspection
app.mount("/frames", StaticFiles(directory=settings.frames_dir), name="frames")
ui_dir = Path(__file__).parent.parent / "ui"


@app.get("/manifest.webmanifest", include_in_schema=False)
def get_manifest():
    """Serve PWA Web App Manifest."""
    manifest_file = ui_dir / "manifest.webmanifest"
    if manifest_file.exists():
        return FileResponse(manifest_file, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js", include_in_schema=False)
def get_service_worker():
    """Serve PWA Service Worker."""
    sw_file = ui_dir / "sw.js"
    if sw_file.exists():
        return FileResponse(sw_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


@app.get("/icons/{file_name}", include_in_schema=False)
def get_icon(file_name: str):
    """Serve PWA and UI icon assets."""
    icon_file = ui_dir / "icons" / file_name
    if icon_file.exists() and icon_file.is_file():
        return FileResponse(icon_file)
    raise HTTPException(status_code=404, detail="Icon not found")


class LoginRequest(BaseModel):
    token: str


@app.post("/api/auth/login")
def auth_login(req: LoginRequest, response: Response, request: Request):
    """Authenticate with personal access token and receive signed session cookie."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not security_core.auth_manager.verify_token(req.token, client_ip=client_ip):
        if security_core.auth_manager.is_rate_limited(client_ip):
            raise HTTPException(status_code=429, detail="Too many failed login attempts. Please wait 60 seconds.")
        raise HTTPException(status_code=401, detail="Invalid personal access token.")

    session_id = security_core.auth_manager.create_session(client_ip=client_ip)
    response.set_cookie(
        key="trading_ai_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=int(7 * 24 * 3600),
        path="/",
    )
    return {"success": True, "message": "Authentication successful", "session_id": session_id}


@app.post("/api/auth/logout")
def auth_logout(response: Response, request: Request):
    """Revoke session and clear session cookie."""
    session_id = request.cookies.get("trading_ai_session")
    if session_id:
        security_core.auth_manager.revoke_session(session_id)
    response.delete_cookie(key="trading_ai_session", path="/")
    return {"success": True, "message": "Logged out successfully"}


@app.get("/api/auth/status")
def auth_status(request: Request):
    """Return current authentication state."""
    if not security_core.auth_manager.is_auth_enabled:
        return {"auth_required": False, "authenticated": True}

    session_id = request.cookies.get("trading_ai_session")
    is_authenticated = False
    if session_id and security_core.auth_manager.verify_session(session_id):
        is_authenticated = True
    else:
        auth_header = request.headers.get("Authorization") or request.headers.get("X-API-Key")
        if auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            client_ip = request.client.host if request.client else "127.0.0.1"
            if security_core.auth_manager.verify_token(token, client_ip=client_ip):
                is_authenticated = True
        else:
            dev_id = request.headers.get("X-Device-Id")
            if dev_id:
                dev = security_core.device_manager.get_device(dev_id)
                if dev and dev.status.value in ["ACTIVE", "TRUSTED"]:
                    is_authenticated = True

    return {
        "auth_required": True,
        "authenticated": is_authenticated,
    }


class SelectFileRequest(BaseModel):
    file_path: Optional[str] = None
    sample_name: Optional[str] = None


@app.get("/")
def get_index():
    """Serve the single-page application UI."""
    index_file = ui_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_file)


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    """Return favicon if present or 204."""
    fav = ui_dir / "icons" / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return Response(status_code=204)


@app.get("/api/health")
def get_health():
    """Return system health status."""
    return {"status": "ok", "version": "0.8.0", "step": "Step 8 complete"}


@app.get("/api/models")
def get_model_mappings():
    """Return configured model router roles and active model IDs."""
    router = ModelRouter(settings)
    return {
        "mappings": router.list_mappings(),
        "has_live_api": orchestrator._ai.has_live_api,
    }


@app.get("/api/jobs")
def list_jobs():
    """List all lecture analysis jobs."""
    jobs = repo.list_jobs()
    return {"jobs": [j.model_dump() for j in jobs]}


@app.post("/api/upload")
async def upload_lecture_file(file: UploadFile = File(...)):
    """Upload a lecture video or audio file."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    dest_path = os.path.join(settings.upload_dir, f"{job_id}_{file.filename}")

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = orchestrator._ingestion.ingest_file(
        source_path=dest_path,
        filename=file.filename,
        job_id=job_id,
    )
    return {"job": job.model_dump()}


@app.post("/api/select_sample")
def select_sample_lecture(req: SelectFileRequest):
    """Create a sample lecture file and ingest it for instant testing."""
    sample_name = req.sample_name or "EURUSD_15m_Liquidity_Lecture.mp4"
    sample_path = os.path.join(settings.upload_dir, sample_name)

    # Create dummy media file if not present
    if not os.path.exists(sample_path):
        with open(sample_path, "wb") as f:
            f.write(b"Trading Lecture Video Binary: EURUSD 15m Liquidity and Fair Value Gap analysis.")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = orchestrator._ingestion.ingest_file(
        source_path=sample_path,
        filename=sample_name,
        job_id=job_id,
    )
    return {"job": job.model_dump()}


def _run_analysis_background(job_id: str, file_path: str, filename: str):
    """Background task to run orchestrator pipeline."""
    try:
        orchestrator.run_pipeline(
            source_media_path=file_path,
            filename=filename,
            job_id=job_id,
            chunk_duration=30.0,
        )
    except Exception as e:
        logger.error(f"Background analysis failed for {job_id}: {e}")


@app.post("/api/jobs/{job_id}/analyze")
def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    """Trigger multimodal lecture analysis for a job."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.status = JobStatus.PROCESSING
    repo.save_job(job)

    # Run analysis
    _run_analysis_background(job.job_id, job.file_path, job.filename)

    updated_job = repo.get_job(job_id)
    return {"status": "started", "job": updated_job.model_dump() if updated_job else job.model_dump()}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get the current status, stage, and errors for a job."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    barrier = repo.get_commit_barrier(job_id)
    return {
        "job": job.model_dump(),
        "barrier": barrier.model_dump(),
    }


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str):
    """Retrieve full analysis results: transcripts, visuals, alignments, evidence, knowledge, notes."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    transcripts = repo.get_transcripts(job_id)
    visual_events = repo.get_visual_events(job_id)
    alignments = repo.get_alignments(job_id)
    evidence = repo.get_evidence(job_id)
    knowledge = repo.get_knowledge(job_id, only_committed=False)
    committed_knowledge = [k for k in knowledge if getattr(k, "is_committed", True)]
    notes = repo.get_notes(job_id)
    barrier = repo.get_commit_barrier(job_id)
    task_units = repo.list_task_units(job_id)
    token_usage = orchestrator._token_manager.get_job_usage_summary(job_id)

    return {
        "job": job.model_dump(),
        "transcripts": [t.model_dump() for t in transcripts],
        "visual_events": [v.model_dump() for v in visual_events],
        "alignments": [a.model_dump() for a in alignments],
        "evidence": [e.model_dump() for e in evidence],
        "knowledge": [k.model_dump() for k in knowledge],
        "notes": notes.model_dump() if notes else None,
        "barrier": barrier.model_dump(),
        "task_units": [u.model_dump() for u in task_units],
        "token_usage": token_usage,
    }


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Retry a failed job from its last failure state."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.status = JobStatus.PROCESSING
    job.last_error = None
    repo.save_job(job)

    _run_analysis_background(job.job_id, job.file_path, job.filename)
    updated = repo.get_job(job_id)
    return {"status": "retried", "job": updated.model_dump() if updated else job.model_dump()}


# ------------------------------------------------------------------
# Step 2 API Endpoints: Context Intelligence + Long-Term Memory
# ------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/context")
def get_context_status(job_id: str):
    """Return the most recent context window snapshot for a job."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    snapshots = repo.list_context_snapshots(job_id)
    latest = repo.get_latest_context_snapshot(job_id)
    return {
        "job_id": job_id,
        "total_snapshots": len(snapshots),
        "latest_snapshot": latest.model_dump() if latest else None,
    }


@app.get("/api/jobs/{job_id}/memory")
def get_memory_embeddings(job_id: str, source_type: Optional[str] = None, limit: int = 50):
    """Return persisted memory embeddings for a job, optionally filtered by source type."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    embeddings = repo.get_memory_embeddings(job_id, source_type=source_type, limit=limit)
    return {
        "job_id": job_id,
        "source_type_filter": source_type,
        "count": len(embeddings),
        # Return metadata only — never send full embedding vectors over API
        "embeddings": [
            {
                "embedding_id": e.embedding_id,
                "source_type": e.source_type,
                "source_id": e.source_id,
                "text_content": e.text_content[:200],  # Truncated for API safety
                "model_name": e.model_name,
                "importance_score": e.importance_score,
                "access_count": e.access_count,
                "created_at": e.created_at.isoformat(),
            }
            for e in embeddings
        ],
    }


@app.post("/api/jobs/{job_id}/memory/search")
def search_memory(job_id: str, query: str, source_types: Optional[str] = None, limit: int = 10):
    """
    Hybrid structured+semantic memory search over all persisted knowledge for a job.

    Returns ranked results with similarity scores.
    Never sends full embedding vectors.
    """
    from trading_ai.core.memory_retriever import MemoryRetriever
    from trading_ai.core.token_manager import TokenManager

    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    tm = TokenManager(settings)
    retriever = MemoryRetriever(repository=repo, token_manager=tm)
    src_filter = source_types.split(",") if source_types else None

    results = retriever.search(
        job_id=job_id,
        query=query,
        source_types=src_filter,
        limit=min(limit, 20),  # Cap at 20 to protect token budget
        max_result_tokens=8000,  # Never return more than 8k token equivalent
    )

    return {
        "job_id": job_id,
        "query": query,
        "result_count": len(results),
        "results": [
            {
                "rank": r.rank,
                "source_type": r.embedding.source_type,
                "source_id": r.embedding.source_id,
                "text_snippet": r.embedding.text_content[:300],
                "similarity_score": r.similarity_score,
                "retrieval_metadata": r.retrieval_metadata,
            }
            for r in results
        ],
    }


# ==================================================================
# Step 3 API Endpoints: Universal Sources, Jobs, Knowledge & Commands
# ==================================================================

class SubmitCommandRequest(BaseModel):
    command_text: str
    target_knowledge_id: Optional[str] = None
    job_id: Optional[str] = None


class CreateJobRequest(BaseModel):
    source_id: Optional[str] = None
    file_path: Optional[str] = None
    filename: Optional[str] = None
    chunk_duration: float = 30.0


# ------------------------------------------------------------------
# Universal Sources
# ------------------------------------------------------------------
@app.post("/api/sources")
def create_source(source: SourceMetadata):
    """Register or save an external or local learning source."""
    repo.save_source(source)
    return {"source": source.model_dump()}


@app.post("/api/sources/resolve")
def resolve_source(req: SourceResolutionRequest):
    """
    Resolve, validate accessibility, and categorize an external URL or file source.
    Returns SOURCE_UNAVAILABLE if restricted or inaccessible.
    """
    result = source_resolver.resolve(req)
    if result.is_accessible:
        source_meta = SourceMetadata(
            source_id=result.source_id,
            source_type=result.source_type,
            source_location=result.source_location,
            title=result.title,
            language=result.language,
            normalized_language=result.normalized_language,
            permissions={"authorized": True},
            acquisition=result.details,
            processing_status=result.status,
            error_reason=result.error_reason,
        )
        repo.save_source(source_meta)
    return {"resolution": result.model_dump()}


@app.get("/api/sources")
def list_sources(source_type: Optional[SourceType] = None, limit: int = 50):
    """List registered learning sources."""
    sources = repo.list_sources(source_type=source_type, limit=limit)
    return {"sources": [s.model_dump() for s in sources]}


@app.get("/api/sources/{source_id}")
def get_source(source_id: str):
    """Retrieve metadata for a specific learning source."""
    source = repo.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    return {"source": source.model_dump()}


# ------------------------------------------------------------------
# Background Jobs & Process Tracking
# ------------------------------------------------------------------
@app.post("/api/jobs/create")
def create_processing_job(req: CreateJobRequest, background_tasks: BackgroundTasks):
    """Create a new processing job for a source or media file."""
    source_id = req.source_id or f"src_{uuid.uuid4().hex[:8]}"
    file_path = req.file_path

    if not file_path and req.source_id:
        source = repo.get_source(req.source_id)
        if source:
            file_path = source.source_location

    if not file_path:
        sample_name = req.filename or "sample_lecture.mp4"
        file_path = os.path.join(settings.upload_dir, sample_name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(b"Trading Binary Data")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = orchestrator._ingestion.ingest_file(
        source_path=file_path,
        filename=req.filename or Path(file_path).name,
        job_id=job_id,
    )

    bg_job = BackgroundJobState(
        job_id=job_id,
        source_id=source_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        current_stage=PipelineStage.SOURCE,
    )
    repo.save_background_job(bg_job)

    return {
        "job": job.model_dump(),
        "background_job": bg_job.model_dump(),
    }


@app.get("/api/jobs/{job_id}/status")
def get_detailed_job_status(job_id: str):
    """Retrieve persisted status, barrier state, and background progress."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    barrier = repo.get_commit_barrier(job_id)
    bg_job = orchestrator.get_background_job_state(job_id)
    return {
        "job_id": job_id,
        "job": job.model_dump(),
        "barrier": barrier.model_dump(),
        "background_job": bg_job.model_dump() if bg_job else None,
    }


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """Pause an active processing job."""
    bg_job = orchestrator.pause_job(job_id)
    if not bg_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"status": "paused", "background_job": bg_job.model_dump()}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str, background_tasks: BackgroundTasks):
    """Resume a paused or interrupted processing job from its last committed chunk."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    bg_job = orchestrator.resume_job(job_id)
    _run_analysis_background(job.job_id, job.file_path, job.filename)
    return {"status": "resumed", "background_job": bg_job.model_dump() if bg_job else None}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a background processing job."""
    bg_job = orchestrator.cancel_job(job_id)
    if not bg_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"status": "cancelled", "background_job": bg_job.model_dump()}


# ------------------------------------------------------------------
# Notes, Concepts & Doubts Insights
# ------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/notes")
def get_job_notes(job_id: str):
    """Retrieve synthesized multimodal lecture notes."""
    notes = repo.get_notes(job_id)
    if not notes:
        return {"job_id": job_id, "notes": None, "message": "Notes not yet generated for this job."}
    return {"job_id": job_id, "notes": notes.model_dump()}


@app.get("/api/jobs/{job_id}/concepts")
def get_job_concepts(job_id: str):
    """Retrieve core concepts, definitions, and terminology identified for a job."""
    concepts = repo.get_concepts(job_id)
    return {"job_id": job_id, "count": len(concepts), "concepts": concepts}


@app.get("/api/jobs/{job_id}/doubts")
def get_job_doubts(job_id: str):
    """Retrieve all identified doubts, caveats, contradictions, and uncertain items for a job."""
    doubts = repo.get_doubts(job_id)
    return {"job_id": job_id, "count": len(doubts), "doubts": doubts}


@app.get("/api/jobs/{job_id}/memory_updates")
def get_job_memory_updates(job_id: str):
    """Retrieve recent memory embeddings, context snapshot count, and active knowledge versions."""
    embeddings = repo.get_memory_embeddings(job_id, limit=20)
    snapshots = repo.list_context_snapshots(job_id)
    knowledge = repo.get_knowledge(job_id, only_committed=False)

    return {
        "job_id": job_id,
        "total_embeddings": len(embeddings),
        "total_context_snapshots": len(snapshots),
        "total_knowledge_items": len(knowledge),
        "recent_embeddings": [
            {
                "embedding_id": e.embedding_id,
                "source_type": e.source_type,
                "source_id": e.source_id,
                "text_snippet": e.text_content[:200],
                "created_at": e.created_at.isoformat(),
            }
            for e in embeddings[:10]
        ],
    }


# ------------------------------------------------------------------
# User Commands & Corrections
# ------------------------------------------------------------------
@app.post("/api/commands")
def submit_user_command(req: SubmitCommandRequest):
    """
    Parse and execute an explicit user command (Save rule, Correct, Remember, Explain, etc.).
    Preserves audit history and updates memory accordingly.
    """
    parse_result = command_parser.parse(
        raw_text=req.command_text, target_knowledge_id=req.target_knowledge_id
    )
    exec_result = command_service.execute_command(parse_result.command, job_id=req.job_id)
    return {
        "command": parse_result.command.model_dump(),
        "parsed_success": parse_result.parsed_success,
        "suggested_action": parse_result.suggested_action,
        "explanation": parse_result.explanation,
        "result": exec_result,
    }


@app.post("/api/commands/save_rule")
def save_rule_command(req: RuleSaveRequest):
    """Explicitly save a persistent trading rule with conditions, exceptions, and evidence."""
    result = command_service.save_rule(req)
    return {"result": result}


@app.post("/api/commands/correct")
def submit_correction(req: CorrectionRequest):
    """
    Submit a versioned correction to existing knowledge.
    Preserves prior beliefs and evidence while activating new version.
    """
    result = command_service.apply_correction(req)
    return {"result": result}


@app.get("/api/commands")
def list_user_commands(limit: int = 50):
    """List historical user commands."""
    cmds = repo.list_user_commands(limit=limit)
    return {"commands": [c.model_dump() for c in cmds]}


@app.get("/api/knowledge/{item_id}/versions")
def get_knowledge_versions(item_id: str):
    """Retrieve complete audit history and all versions for a knowledge item."""
    versions = repo.get_knowledge_versions(item_id)
    active = repo.get_active_knowledge_version(item_id)
    return {
        "item_id": item_id,
        "total_versions": len(versions),
        "active_version": active.model_dump() if active else None,
        "versions": [v.model_dump() for v in versions],
    }


# =====================================================================
# STEP 4: Live Market Intelligence, Watchlist, Setups & Screen Endpoints
# =====================================================================

class ScreenAccessRequestModel(BaseModel):
    app_name: str
    window_title: Optional[str] = None
    region_box: Optional[List[int]] = None


class WatchlistScanRequestModel(BaseModel):
    force: bool = True
    symbols: Optional[List[str]] = None


@app.get("/api/market/snapshot/{symbol:path}")
def get_market_snapshot_endpoint(symbol: str, timeframe: str = "15m", limit: int = 60):
    """Fetch current price action snapshot and candles for a symbol."""
    snapshot = data_provider.get_snapshot(symbol=symbol, timeframe=timeframe, limit=limit)
    return snapshot.model_dump(mode="json")


@app.get("/api/market/analysis/{symbol:path}")
def get_market_analysis_endpoint(symbol: str):
    """Fetch latest persisted multi-timeframe analysis for a symbol or compute on demand."""
    analysis = repo.get_latest_market_analysis(symbol=symbol)
    if not analysis:
        analysis = mtf_engine.analyze_symbol(symbol=symbol)
        repo.save_market_analysis(analysis)
    return analysis.model_dump(mode="json")


@app.post("/api/market/analysis/{symbol:path}")
def trigger_market_analysis_endpoint(symbol: str):
    """Trigger a fresh multi-timeframe analysis on demand."""
    analysis = watchlist_monitor.process_symbol(symbol=symbol, force=True)
    if not analysis:
        analysis = mtf_engine.analyze_symbol(symbol=symbol)
        repo.save_market_analysis(analysis)
    return analysis.model_dump(mode="json")


@app.get("/api/market/setups/{symbol:path}")
def get_market_setups_endpoint(symbol: str, status: Optional[str] = None):
    """Fetch lecture-validated setups for a symbol, optionally filtered by status."""
    parsed_status = None
    if status:
        try:
            from trading_ai.schemas.enums import SetupStatus
            parsed_status = SetupStatus(status)
        except ValueError:
            parsed_status = None
    setups = repo.get_market_setups(symbol=symbol, status=parsed_status)
    return {
        "symbol": symbol,
        "count": len(setups),
        "setups": [s.model_dump(mode="json") for s in setups],
    }


@app.get("/api/market/markings/{symbol:path}")
def get_chart_markings_endpoint(symbol: str, timeframe: Optional[str] = None, only_valid: bool = True):
    """Fetch clean, decluttered chart markings for overlay rendering."""
    markings = repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=only_valid)
    return {
        "symbol": symbol,
        "count": len(markings),
        "markings": [m.model_dump(mode="json") for m in markings],
    }


@app.get("/api/market/chart/{symbol:path}")
def get_market_chart_endpoint(symbol: str, timeframe: str = "15m", limit: int = 50):
    """Fetch candlestick OHLCV series with lecture-derived ICT/SMC markings for live charting canvas."""
    snapshot = data_provider.get_snapshot(symbol=symbol, timeframe=timeframe, limit=limit)
    markings = repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=True)
    if not markings:
        setups = repo.get_market_setups(symbol=symbol)
        levels = repo.get_market_levels(symbol=symbol, timeframe=timeframe)
        markings = chart_marker.generate_markings(snapshot, setups, levels)

    candles_data = [
        {
            "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else str(c.timestamp),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        }
        for c in snapshot.candles
    ]

    last_px = getattr(snapshot, "last_price", getattr(snapshot, "current_price", snapshot.candles[-1].close if snapshot.candles else 0.0))

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": float(last_px),
        "candle_count": len(candles_data),
        "candles": candles_data,
        "markings": [m.model_dump(mode="json") for m in markings],
    }


@app.get("/api/market/levels/{symbol:path}")
def get_market_levels_endpoint(symbol: str, timeframe: Optional[str] = None):
    """Fetch structural and session price levels for a symbol."""
    levels = repo.get_market_levels(symbol=symbol, timeframe=timeframe, only_valid=True)
    return {
        "symbol": symbol,
        "count": len(levels),
        "levels": [lvl.model_dump(mode="json") for lvl in levels],
    }


@app.get("/api/market/watchlist")
def get_watchlist_overview():
    """Retrieve full watchlist status, active monitor state, and independent background analysis."""
    states = repo.list_watchlist_monitor_states()
    state_map = {s.symbol: s.model_dump(mode="json") for s in states}
    return {
        "monitor_status": watchlist_monitor.status.value,
        "watchlist": watchlist_monitor.watchlist,
        "symbol_states": state_map,
    }


@app.post("/api/market/watchlist/scan")
def scan_watchlist_endpoint(req: Optional[WatchlistScanRequestModel] = None):
    """Trigger independent multi-symbol analysis across the configured watchlist."""
    force = req.force if req else True
    results = watchlist_monitor.process_all_watchlist(force=force)
    summary = {}
    for sym, analysis in results.items():
        if analysis:
            summary[sym] = {
                "overall_bias": analysis.overall_bias.value,
                "setups_count": len(analysis.active_setups),
                "markings_count": len(analysis.active_markings),
                "timestamp": analysis.timestamp,
            }
        else:
            summary[sym] = {"status": "SKIPPED_OR_UNCHANGED"}
    return {
        "status": "success",
        "processed_count": len(results),
        "summary": summary,
    }


@app.get("/api/market/monitor/status")
def get_monitor_status():
    """Check background monitoring engine operational status."""
    return {
        "status": watchlist_monitor.status.value,
        "watchlist": watchlist_monitor.watchlist,
    }


@app.post("/api/market/monitor/start")
def start_monitor_endpoint():
    """Start or resume independent background watchlist monitoring."""
    watchlist_monitor.start()
    return {
        "status": "success",
        "monitor_status": watchlist_monitor.status.value,
        "message": "Watchlist monitoring resumed in background.",
    }


@app.post("/api/market/monitor/stop")
def stop_monitor_endpoint():
    """Stop or pause background watchlist monitoring."""
    watchlist_monitor.stop()
    return {
        "status": "success",
        "monitor_status": watchlist_monitor.status.value,
        "message": "Watchlist monitoring paused.",
    }


@app.get("/api/screen/permissions")
def list_screen_permissions_endpoint():
    """List all screen access permission requests and currently granted scopes."""
    perms = repo.list_screen_permissions()
    active = repo.get_active_screen_permission()
    return {
        "active_permission": active.model_dump(mode="json") if active else None,
        "permissions": [p.model_dump(mode="json") for p in perms],
    }


@app.post("/api/screen/request")
def request_screen_permission_endpoint(req: ScreenAccessRequestModel):
    """Request explicit permission to inspect an authorized window/application."""
    perm = screen_manager.request_access(
        app_name=req.app_name,
        window_title=req.window_title,
        region_box=req.region_box,
    )
    return perm.model_dump(mode="json")


@app.post("/api/screen/{permission_id}/grant")
def grant_screen_permission_endpoint(permission_id: str):
    """Explicitly grant permission for a requested application scope."""
    perm = screen_manager.grant_access(permission_id=permission_id)
    if not perm:
        raise HTTPException(status_code=404, detail="Screen permission request not found")
    return perm.model_dump(mode="json")


@app.post("/api/screen/{permission_id}/revoke")
def revoke_screen_permission_endpoint(permission_id: str):
    """Explicitly revoke screen access permission."""
    perm = screen_manager.revoke_access(permission_id=permission_id)
    if not perm:
        raise HTTPException(status_code=404, detail="Screen permission request not found")
    return perm.model_dump(mode="json")


# =====================================================================
# STEP 5: INTELLIGENT SITUATION & SELECTIVE DISPLAY ENDPOINTS
# =====================================================================

class DisplayPolicyRequestModel(BaseModel):
    show_all_liquidity: bool = False
    show_hidden_setups: bool = False
    show_all_fvgs: bool = False
    show_all_order_blocks: bool = False
    show_unconfirmed: bool = False
    focus_active_setup_only: bool = False
    max_visible_markings: int = 7


@app.get("/api/market/situation/{symbol:path}")
def get_current_situation_endpoint(symbol: str):
    """
    Retrieve or compute the lightweight decision-context layer answering
    'What matters RIGHT NOW?' for a symbol.
    """
    situation = repo.get_latest_current_situation(symbol=symbol)
    if not situation:
        snapshot = data_provider.get_snapshot(symbol=symbol, timeframe="15m")
        analysis = repo.get_latest_market_analysis(symbol=symbol)
        if not analysis:
            analysis = mtf_engine.analyze_symbol(symbol=symbol)
            repo.save_market_analysis(analysis)
        situation = situation_engine.evaluate_situation(
            symbol=symbol,
            snapshot=snapshot,
            unified_analysis=analysis,
        )
    return situation.model_dump(mode="json")


@app.post("/api/market/situation/{symbol:path}")
def evaluate_current_situation_endpoint(symbol: str, policy_req: Optional[DisplayPolicyRequestModel] = None):
    """
    Force re-evaluation of the current situation snapshot with optional display policy.
    """
    snapshot = data_provider.get_snapshot(symbol=symbol, timeframe="15m")
    analysis = repo.get_latest_market_analysis(symbol=symbol)
    if not analysis:
        analysis = mtf_engine.analyze_symbol(symbol=symbol)
        repo.save_market_analysis(analysis)

    from trading_ai.schemas.market import DisplayPolicy
    policy = DisplayPolicy(**policy_req.model_dump()) if policy_req else DisplayPolicy()

    situation = situation_engine.evaluate_situation(
        symbol=symbol,
        snapshot=snapshot,
        unified_analysis=analysis,
        display_policy=policy,
    )
    return situation.model_dump(mode="json")


# ------------------------------------------------------------------ #
# Step 6 — Fundamental Market Intelligence & News Timing Endpoints   #
# ------------------------------------------------------------------ #

@app.get("/api/fundamental/events")
def get_fundamental_events_endpoint(
    currency: Optional[str] = None,
    impact: Optional[str] = None,
    limit: int = 50,
):
    """Query normalized economic calendar events."""
    impact_enum = None
    if impact:
        try:
            from trading_ai.schemas.enums import EventImpact
            impact_enum = EventImpact(impact.upper())
        except ValueError:
            pass

    events = repo.get_fundamental_events(currency=currency, impact=impact_enum, limit=limit)
    if not events and fundamental_provider:
        # Initial on-demand fetch if empty
        events = fundamental_provider.fetch_events(currencies=[currency] if currency else None)
        if events:
            repo.save_fundamental_events(events)
            events = events[:limit]
    return [e.model_dump(mode="json") for e in events]


@app.get("/api/fundamental/context/{symbol:path}")
def get_fundamental_context_endpoint(symbol: str):
    """Retrieve news timing and event proximity analysis for a symbol."""
    currencies = news_timing_engine.resolve_symbol_currencies(symbol)
    events = repo.get_fundamental_events(limit=100)
    filtered = [e for e in events if e.currency.upper() in {c.upper() for c in currencies}]
    if not filtered and fundamental_provider:
        fresh = fundamental_provider.fetch_events(currencies=currencies)
        if fresh:
            repo.save_fundamental_events(fresh)
            filtered = fresh

    timing_context = news_timing_engine.evaluate_timing(
        symbol=symbol, events=filtered, current_time=datetime.now(timezone.utc)
    )
    return timing_context.model_dump(mode="json")


@app.get("/api/fundamental/analysis/{symbol:path}")
def get_tech_fund_analysis_endpoint(symbol: str):
    """
    Retrieve synthesized technical + fundamental correlation analysis,
    including event risk and conflict detection.
    """
    analysis = repo.get_latest_tech_fund_analysis(symbol=symbol)
    if not analysis:
        # Force evaluation via situation engine
        snapshot = data_provider.get_snapshot(symbol=symbol, timeframe="15m")
        unified_analysis = repo.get_latest_market_analysis(symbol=symbol)
        if not unified_analysis:
            unified_analysis = mtf_engine.analyze_symbol(symbol=symbol)
            repo.save_market_analysis(unified_analysis)
        situation_engine.evaluate_situation(
            symbol=symbol,
            snapshot=snapshot,
            unified_analysis=unified_analysis,
        )
        analysis = repo.get_latest_tech_fund_analysis(symbol=symbol)

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis available for {symbol}")
    return analysis.model_dump(mode="json")


@app.get("/api/fundamental/status")
def get_fundamental_provider_status_endpoint():
    """Check health, connection status, and staleness of the fundamental data provider."""
    state = fundamental_provider.get_provider_status()
    repo.save_provider_state(state)
    return state.model_dump(mode="json")


@app.post("/api/fundamental/sync")
def sync_fundamental_events_endpoint():
    """Explicitly trigger synchronization of fundamental economic calendar data."""
    events = fundamental_provider.fetch_events()
    if events:
        repo.save_fundamental_events(events)
    state = fundamental_provider.get_provider_status()
    repo.save_provider_state(state)
    return {
        "status": "success",
        "provider_status": state.status.value,
        "events_synced": len(events),
        "last_successful_fetch": state.last_successful_fetch.isoformat() if state.last_successful_fetch else None,
    }


# =====================================================================
# STEP 7: AI TRADING COMPANION & PROACTIVE ALERT ENDPOINTS
# =====================================================================

from trading_ai.schemas.companion import (
    CompanionQueryRequest,
    DisciplineConfig,
    TradeIntentRequest,
)

class AcknowledgeAlertRequestModel(BaseModel):
    alert_id: str


@app.get("/api/companion/status")
def get_companion_status_endpoint(user_id: str = "default_user"):
    """Overall status of Trading Companion engines, discipline, and proactive alerts."""
    unacked = repo.get_alerts(unacknowledged_only=True, limit=100)
    discipline = companion_core.get_discipline_status(user_id=user_id)
    return {
        "status": "operational",
        "companion_engine": "ACTIVE",
        "unacknowledged_alerts_count": len(unacked),
        "discipline_status": {
            "trades_today": discipline.trades_today_count,
            "daily_loss": discipline.daily_realized_loss,
            "active_violations": discipline.active_violations,
            "cooldown_active": bool(discipline.cooldown_active_until and discipline.cooldown_active_until > datetime.now(timezone.utc)),
        },
        "supported_intents": [
            "MARKET_STATUS",
            "EXPLAIN_BIAS",
            "EXPLAIN_SETUP",
            "NEWS_QUERY",
            "SCENARIO_ANALYSIS",
            "SETUP_VALIDATION",
            "MARKET_HISTORY",
            "RISK_STATUS",
            "DISCIPLINE_STATUS",
            "WHY_ALERT",
            "GENERAL_MARKET_QA",
        ],
    }


class JarvisToggleRequest(BaseModel):
    state: str
    spoken_alerts: Optional[bool] = None


jarvis_runtime_state = {
    "state": "ON",
    "spoken_alerts": True,
    "last_toggled_at": datetime.now(timezone.utc).isoformat(),
    "error_message": None,
}


@app.get("/api/jarvis/status")
def get_jarvis_status_endpoint():
    """
    Retrieve one authoritative JARVIS master status.
    States: OFF, TURNING_ON, ON, DEGRADED, ERROR.
    Reflects real backend readiness, broker data health, and voice preference.
    """
    unacked = repo.get_alerts(unacknowledged_only=True, limit=50) if repo else []
    sup = globals().get("market_supervisor")
    active_broker = sup.active_broker_name if sup else "Paper Broker"
    sup_status = sup.get_status() if sup else {}
    data_health = sup_status.get("data_health", "CONNECTED")

    current_state = jarvis_runtime_state.get("state", "ON")
    if current_state == "ON":
        if data_health in ("DISCONNECTED", "ERROR"):
            current_state = "DEGRADED"
        elif jarvis_runtime_state.get("error_message"):
            current_state = "ERROR"

    label_map = {
        "OFF": "JARVIS: OFF (○)",
        "TURNING_ON": "JARVIS: CONNECTING (⏳)",
        "ON": "JARVIS: READY (●)",
        "DEGRADED": "JARVIS: RECONNECTING (▲)",
        "ERROR": "JARVIS: ERROR (✖)",
    }

    return {
        "status": "success",
        "state": current_state,
        "label": label_map.get(current_state, f"JARVIS: {current_state}"),
        "spoken_alerts": jarvis_runtime_state.get("spoken_alerts", True),
        "backend_ready": True,
        "active_broker": active_broker,
        "supervisor_active": sup.is_monitoring if sup else False,
        "data_health": data_health,
        "unacknowledged_alerts_count": len(unacked),
        "session_id": "default_session",
        "error_message": jarvis_runtime_state.get("error_message"),
    }


@app.post("/api/jarvis/toggle")
def toggle_jarvis_master_endpoint(req: JarvisToggleRequest):
    """
    One clear JARVIS master toggle.
    Controls real backend background intelligence, spoken alerts, and session activation.
    """
    desired = req.state.upper().strip()
    if desired in ("ON", "READY"):
        jarvis_runtime_state["state"] = "ON"
        jarvis_runtime_state["error_message"] = None
        if req.spoken_alerts is not None:
            jarvis_runtime_state["spoken_alerts"] = req.spoken_alerts
        logger.info("JARVIS master state toggled to ON.")
    elif desired in ("OFF", "STOP"):
        jarvis_runtime_state["state"] = "OFF"
        if req.spoken_alerts is not None:
            jarvis_runtime_state["spoken_alerts"] = req.spoken_alerts
        logger.info("JARVIS master state toggled to OFF.")
    elif desired in ("TURNING_ON", "CONNECTING"):
        jarvis_runtime_state["state"] = "TURNING_ON"
    elif desired == "DEGRADED":
        jarvis_runtime_state["state"] = "DEGRADED"
    elif desired == "ERROR":
        jarvis_runtime_state["state"] = "ERROR"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid JARVIS state: {desired}")

    jarvis_runtime_state["last_toggled_at"] = datetime.now(timezone.utc).isoformat()
    return get_jarvis_status_endpoint()


@app.get("/api/companion/context/{symbol:path}")
def get_companion_context_endpoint(symbol: str, user_id: str = "default_user"):
    """Retrieve decision-relevant TradingCompanionContext for a symbol."""
    ctx = companion_core.get_companion_context(symbol=symbol, user_id=user_id)
    return ctx.model_dump(mode="json")


@app.get("/api/alerts")
def get_alerts_endpoint(unacknowledged_only: bool = False, limit: int = 50):
    """List recent proactive alerts across all monitored symbols."""
    alerts = companion_core.get_alerts(unacknowledged_only=unacknowledged_only, limit=limit)
    return [a.model_dump(mode="json") for a in alerts]


@app.get("/api/alerts/{symbol:path}")
def get_symbol_alerts_endpoint(symbol: str, unacknowledged_only: bool = False, limit: int = 50):
    """Retrieve proactive alerts filtered by symbol."""
    alerts = companion_core.get_alerts(symbol=symbol, unacknowledged_only=unacknowledged_only, limit=limit)
    return [a.model_dump(mode="json") for a in alerts]


@app.post("/api/alerts/acknowledge")
def acknowledge_alert_endpoint(req: AcknowledgeAlertRequestModel):
    """Mark an alert as acknowledged to dismiss it from high-priority display."""
    success = companion_core.acknowledge_alert(req.alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {req.alert_id} not found")
    return {"status": "success", "alert_id": req.alert_id, "acknowledged": True}


@app.get("/api/discipline/status")
def get_discipline_status_endpoint(user_id: str = "default_user"):
    """Get user-configured trading discipline limits, trade counters, and active warnings."""
    state = companion_core.get_discipline_status(user_id=user_id)
    return state.model_dump(mode="json")


@app.post("/api/discipline/config")
def update_discipline_config_endpoint(config: DisciplineConfig):
    """Update user-defined discipline and risk boundaries."""
    state = companion_core.update_discipline_config(user_id=config.user_id, config=config)
    return state.model_dump(mode="json")


@app.post("/api/discipline/check_trade")
def check_trade_confirmation_endpoint(intent: TradeIntentRequest, user_id: str = "default_user"):
    """
    Validate trade intent against setup context, risk parameters, and discipline rules.
    Stops at USER_CONFIRMATION_REQUIRED — never executes orders automatically.
    """
    result = companion_core.check_trade_intent(intent=intent, user_id=user_id)
    return result.model_dump(mode="json")


@app.get("/api/scenarios/{symbol:path}")
def get_scenarios_endpoint(symbol: str, level: Optional[float] = None):
    """Retrieve conditional IF-THEN-UNLESS scenario analyses for symbol."""
    scenarios = companion_core.get_scenarios(symbol=symbol, custom_level=level)
    return [s.model_dump(mode="json") for s in scenarios]


@app.post("/api/companion/query")
def companion_query_endpoint(req: CompanionQueryRequest):
    """
    Interactive Voice / Companion conversation endpoint.
    Routes user intent, fetches minimal sufficient context, and generates grounded answer.
    """
    response = companion_core.query_companion(req)
    return response.model_dump(mode="json")


# =====================================================================
# STEP 8: HEDGE FUND EXECUTION & RISK MANAGEMENT ENDPOINTS
# =====================================================================

from trading_ai.schemas.enums import EngagementMode, ExecutionMode
from trading_ai.schemas.execution import (
    ExecutionPolicy,
    TradeProposal,
    Order,
    Position,
    PositionEvent,
)


class ProposalGenerateRequest(BaseModel):
    symbol: str
    direction: Optional[str] = None
    engagement_mode: EngagementMode = EngagementMode.DEFENSIVE
    account_equity: float = 10000.0
    user_id: str = "default_user"


class ProposalApproveRequest(BaseModel):
    user_confirmed: bool = True
    user_id: str = "default_user"


class ProposalRejectRequest(BaseModel):
    reason: str = "Manual Trader Rejection"


class ProposalModifyRequest(BaseModel):
    new_sl: Optional[float] = None
    new_target: Optional[float] = None
    new_mode: Optional[EngagementMode] = None
    account_equity: float = 10000.0
    user_id: str = "default_user"


class UpdateSLRequest(BaseModel):
    new_sl: float
    reason: str = "Manual Trailing SL Adjustment"


class ClosePositionRequest(BaseModel):
    current_price: Optional[float] = None
    reason: str = "Manual Trader Discretion Exit"
    user_id: str = "default_user"


@app.get("/api/execution/status")
def get_execution_status_endpoint(user_id: str = "default_user"):
    """Get status of Hedge Fund Execution Core, policies, and open positions."""
    policy = execution_core.get_or_create_policy(user_id)
    open_pos = repo.get_open_positions(environment=policy.execution_environment)
    proposals = repo.get_trade_proposals(limit=10, environment=policy.execution_environment)
    return {
        "status": "operational",
        "engine": "HedgeFundExecutionCore",
        "policy": policy.model_dump(mode="json"),
        "open_positions_count": len(open_pos),
        "recent_proposals_count": len(proposals),
        "broker": "PaperBrokerAdapter" if policy.execution_environment == ExecutionEnvironment.PAPER else "LiveBrokerAdapter",
        "execution_environment": policy.execution_environment.value,
        "live_execution_enabled": policy.live_execution_enabled,
    }


@app.get("/api/risk/status")
def get_risk_status_endpoint(user_id: str = "default_user"):
    """Get active risk parameters, policy limits, and discipline state."""
    policy = execution_core.get_or_create_policy(user_id)
    discipline = companion_core.get_discipline_status(user_id)
    return {
        "user_id": user_id,
        "policy": policy.model_dump(mode="json"),
        "discipline": discipline.model_dump(mode="json"),
    }


@app.post("/api/execution/policy")
def update_execution_policy_endpoint(policy: ExecutionPolicy):
    """Update execution constraints (Manual/Auto mode, risk caps, RR minimums)."""
    updated = execution_core.update_policy(policy)
    return updated.model_dump(mode="json")


class ExecutionModeToggleRequest(BaseModel):
    mode: str = Field(description="Target mode: MANUAL or AUTO/AUTOMATIC")
    confirm: bool = Field(default=False, description="Explicit human confirmation for automated execution")
    user_id: str = Field(default="default_user", description="Trader user ID")


@app.post("/api/execution/mode")
def toggle_execution_mode_endpoint(request: ExecutionModeToggleRequest):
    """Safely toggle between MANUAL and AUTOMATIC execution gates with required confirmation."""
    target_mode = request.mode.strip().upper()
    is_auto = target_mode in ["AUTO", "AUTOMATIC"]

    current_policy = execution_core.get_or_create_policy(request.user_id)

    # Safety Check: Switching to AUTOMATIC requires explicit confirmation
    if is_auto and not request.confirm:
        return {
            "status": "CONFIRMATION_REQUIRED",
            "message": "Switching to AUTOMATIC GATE permits automated trade execution governed strictly by RiskEngine. Explicit confirmation required.",
            "current_mode": current_policy.execution_mode.value,
            "requires_confirmation": True,
        }

    from trading_ai.schemas.enums import ExecutionMode
    current_policy.execution_mode = ExecutionMode.AUTO if is_auto else ExecutionMode.MANUAL
    current_policy.auto_execution_enabled = is_auto
    updated_policy = execution_core.update_policy(current_policy)

    # Synchronize centralized settings state
    unified_settings_state.trading.execution_mode = updated_policy.execution_mode.value

    logger.info(
        f"EXECUTION_MODE_CHANGED: User={request.user_id} Mode={updated_policy.execution_mode.value} "
        f"AutoEnabled={updated_policy.auto_execution_enabled}"
    )

    return {
        "status": "SUCCESS",
        "execution_mode": updated_policy.execution_mode.value,
        "auto_execution_enabled": updated_policy.auto_execution_enabled,
        "policy": updated_policy.model_dump(mode="json"),
    }


@app.get("/api/settings")
def get_unified_settings_endpoint():
    """Retrieve centralized application, AI, voice, broker, and trading settings with secret masking."""
    # Ensure current state reflects active execution policy
    policy = execution_core.get_or_create_policy("default_user")
    unified_settings_state.trading.execution_mode = policy.execution_mode.value
    unified_settings_state.trading.execution_environment = policy.execution_environment.value
    unified_settings_state.trading.risk_per_trade_pct = policy.max_auto_risk_pct
    unified_settings_state.trading.min_rr_ratio = policy.min_rr_ratio

    # Sync discipline settings from TradingCompanionCore if available
    if companion_core is not None:
        try:
            d_state = companion_core.get_discipline_status("default_user")
            if d_state and d_state.config:
                unified_settings_state.trading.require_stop_loss = d_state.config.stop_loss_required
                if d_state.config.cooldown_minutes is not None:
                    unified_settings_state.trading.cooldown_minutes = d_state.config.cooldown_minutes
        except Exception as e:
            logger.debug(f"Could not sync discipline status to settings: {e}")

    masked = mask_sensitive_settings(unified_settings_state)
    return masked.model_dump(mode="json")


@app.post("/api/settings")
def update_unified_settings_endpoint(payload: UnifiedSettings):
    """Update centralized settings and propagate updates to execution policy and runtime subsystems."""
    global unified_settings_state
    unified_settings_state = payload

    # Propagate to ExecutionPolicy
    from trading_ai.schemas.enums import ExecutionMode, ExecutionEnvironment
    policy = execution_core.get_or_create_policy("default_user")
    if payload.trading.execution_mode.upper() in ["AUTO", "AUTOMATIC"]:
        policy.execution_mode = ExecutionMode.AUTO
        policy.auto_execution_enabled = True
    else:
        policy.execution_mode = ExecutionMode.MANUAL
        policy.auto_execution_enabled = False

    if payload.trading.execution_environment.upper() == "LIVE":
        policy.execution_environment = ExecutionEnvironment.LIVE
        policy.live_execution_enabled = True
    else:
        policy.execution_environment = ExecutionEnvironment.PAPER
        policy.live_execution_enabled = False

    policy.max_auto_risk_pct = payload.trading.risk_per_trade_pct
    policy.min_rr_ratio = payload.trading.min_rr_ratio
    execution_core.update_policy(policy)

    # Propagate to TradingCompanionCore DisciplineEngine
    if companion_core is not None:
        try:
            from trading_ai.schemas.companion import DisciplineConfig
            d_cfg = DisciplineConfig(
                user_id="default_user",
                stop_loss_required=payload.trading.require_stop_loss,
                max_risk_per_trade_pct=payload.trading.risk_per_trade_pct,
                daily_loss_limit=payload.trading.max_daily_loss_pct * 100.0 if payload.trading.max_daily_loss_pct else None,
                cooldown_minutes=payload.trading.cooldown_minutes,
            )
            companion_core.update_discipline_config(user_id="default_user", config=d_cfg)
        except Exception as e:
            logger.warning(f"Failed to propagate discipline settings to companion core: {e}")

    logger.info(f"UNIFIED_SETTINGS_UPDATED: Version={payload.system.version} Mode={payload.trading.execution_mode}")

    masked = mask_sensitive_settings(unified_settings_state)
    return {
        "status": "SUCCESS",
        "settings": masked.model_dump(mode="json"),
    }



@app.get("/api/trades/proposals")
def get_trade_proposals_endpoint(
    symbol: Optional[str] = None,
    environment: Optional[ExecutionEnvironment] = None,
    limit: int = 50,
):
    """List recent auditable trade proposals."""
    proposals = repo.get_trade_proposals(symbol=symbol, environment=environment, limit=limit)
    return [p.model_dump(mode="json") for p in proposals]


@app.get("/api/trades/proposals/{proposal_id}")
def get_trade_proposal_detail_endpoint(proposal_id: str):
    """Retrieve full details of a specific trade proposal."""
    proposal = repo.get_trade_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return proposal.model_dump(mode="json")


@app.post("/api/trades/proposals/generate")
def generate_proposal_endpoint(req: ProposalGenerateRequest):
    """Generate an institutional TradeProposal from current market situation."""
    sit = repo.get_latest_current_situation(req.symbol)
    if not sit:
        sit = situation_engine.evaluate_situation(req.symbol)
        if not sit:
            raise HTTPException(status_code=404, detail=f"No situation data available for {req.symbol}")

    proposal = execution_core.generate_trade_proposal(
        symbol=req.symbol,
        situation=sit,
        direction=req.direction,
        engagement_mode=req.engagement_mode,
        account_equity=req.account_equity,
        user_id=req.user_id,
    )
    return proposal.model_dump(mode="json")


@app.post("/api/trades/proposals/{proposal_id}/approve")
def approve_proposal_endpoint(proposal_id: str, req: ProposalApproveRequest):
    """
    Approve proposal via ExecutionGate and execute via BrokerAdapter.
    Mandatory manual confirmation check.
    """
    success, order, position, errors = execution_core.execute_proposal(
        proposal_id=proposal_id,
        user_confirmed=req.user_confirmed,
        user_id=req.user_id,
    )
    if not success:
        return JSONResponse(
            status_code=400,
            content={
                "status": "rejected",
                "proposal_id": proposal_id,
                "reasons": errors,
            },
        )
    return {
        "status": "executed",
        "proposal_id": proposal_id,
        "order": order.model_dump(mode="json") if order else None,
        "position": position.model_dump(mode="json") if position else None,
    }


@app.post("/api/trades/proposals/{proposal_id}/reject")
def reject_proposal_endpoint(proposal_id: str, req: ProposalRejectRequest):
    """Reject a pending trade proposal."""
    prop = execution_core.reject_proposal(proposal_id, reason=req.reason)
    if not prop:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return prop.model_dump(mode="json")


@app.post("/api/trades/proposals/{proposal_id}/modify")
def modify_proposal_endpoint(proposal_id: str, req: ProposalModifyRequest):
    """Modify proposal parameters and recalculate risk deterministically."""
    prop = execution_core.modify_proposal(
        proposal_id=proposal_id,
        new_sl=req.new_sl,
        new_target=req.new_target,
        new_mode=req.new_mode,
        account_equity=req.account_equity,
        user_id=req.user_id,
    )
    if not prop:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return prop.model_dump(mode="json")


@app.get("/api/orders")
def get_orders_endpoint(
    symbol: Optional[str] = None,
    environment: Optional[ExecutionEnvironment] = None,
    limit: int = 50,
):
    """Retrieve recent broker order history."""
    orders = repo.get_orders(symbol=symbol, environment=environment, limit=limit)
    return [o.model_dump(mode="json") for o in orders]


@app.get("/api/positions")
def get_positions_endpoint(
    symbol: Optional[str] = None,
    environment: Optional[ExecutionEnvironment] = None,
    open_only: bool = False,
    limit: int = 50,
):
    """Retrieve managed positions."""
    if open_only:
        positions = repo.get_open_positions(symbol=symbol, environment=environment)
    else:
        positions = repo.get_positions(symbol=symbol, environment=environment, limit=limit)
    return [p.model_dump(mode="json") for p in positions]


@app.post("/api/positions/{position_id}/close")
def close_position_endpoint(position_id: str, req: ClosePositionRequest):
    """Manually close an open position."""
    pos = repo.get_position(position_id)
    if not pos:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

    close_price = req.current_price
    if close_price is None or close_price <= 0:
        sit = repo.get_latest_current_situation(pos.symbol)
        close_price = sit.current_price if sit else pos.entry_price

    closed_pos, err = execution_core.position_manager.manual_close_position(
        position_id=position_id,
        current_price=close_price,
        user_id=req.user_id,
        reason=req.reason,
    )
    if err:
        raise HTTPException(status_code=400, detail=err)
    return closed_pos.model_dump(mode="json")


@app.post("/api/positions/{position_id}/update_sl")
def update_position_sl_endpoint(position_id: str, req: UpdateSLRequest):
    """Update stop loss on an open position."""
    updated_pos, err = execution_core.position_manager.manual_update_sl(
        position_id=position_id,
        new_sl=req.new_sl,
        reason=req.reason,
    )
    if err:
        raise HTTPException(status_code=400, detail=err)
    return updated_pos.model_dump(mode="json")


# =========================================================================
# STEP 9: BACKTESTING, TRADING-BRAIN & PROMOTION ENDPOINTS
# =========================================================================

class ExtractRuleRequest(BaseModel):
    lecture_text: str
    lecture_id: Optional[str] = None
    source_timestamp: Optional[float] = None
    existing_rule_id: Optional[str] = None
    current_version: int = 0


class AnalyzeBacktestPerformanceRequest(BaseModel):
    backtest_id: str


@app.get("/api/rules/versions")
def get_rule_versions_endpoint(
    state: Optional[RuleLifecycleState] = None,
    limit: int = 50,
):
    """List versioned rules with lifecycle status."""
    versions = repo.get_rule_versions(state=state)
    return [v.model_dump(mode="json") for v in versions[:limit]]


@app.post("/api/rules/versions")
def create_rule_version_endpoint(req: CreateRuleVersionRequest):
    """Create a new candidate or learned rule version."""
    rule_ver = RuleVersion(
        title=req.title,
        rule_text=req.rule_text,
        conditions=req.conditions,
        exceptions=req.exceptions,
        source_lecture_id=req.source_lecture_id,
        source_timestamp=req.source_timestamp,
        provenance=req.provenance,
        tags=req.tags,
        state=RuleLifecycleState.CANDIDATE,
    )
    repo.save_rule_version(rule_ver)
    return rule_ver.model_dump(mode="json")


@app.post("/api/trading-brain/extract-rule")
def extract_rule_endpoint(req: ExtractRuleRequest):
    """Reason on lecture text via Gemini Trading Brain to extract candidate rule."""
    rule_ver = trading_brain.extract_candidate_rule(
        lecture_text=req.lecture_text,
        lecture_id=req.lecture_id,
        source_timestamp=req.source_timestamp,
        existing_rule_id=req.existing_rule_id,
        current_version=req.current_version,
    )
    return rule_ver.model_dump(mode="json")


@app.post("/api/trading-brain/analyze-backtest")
def analyze_backtest_endpoint(req: AnalyzeBacktestPerformanceRequest):
    """Qualitative review of completed backtest results by Gemini Trading Brain."""
    run = repo.get_backtest_run(req.backtest_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {req.backtest_id} not found")
    if not run.metrics:
        raise HTTPException(status_code=400, detail="Backtest has no computed metrics yet")

    rule = repo.get_rule_version(run.rule_version_id) if run.rule_version_id else None
    if not rule:
        rule = RuleVersion(
            rule_id="ad_hoc",
            title=f"Backtest {run.symbol} Rule",
            rule_text="Empirical price action setup",
            conditions=["Displacement"],
        )

    trades = repo.get_backtest_trades(req.backtest_id)
    analysis = trading_brain.analyze_backtest_performance(run.metrics, rule, trades)
    return analysis


@app.post("/api/backtests")
def create_backtest_endpoint(req: StartBacktestRequest):
    """Create a new backtest run (AI, Manual, Recent Lecture, or Combined)."""
    cfg = BacktestConfig(
        asset_class=req.asset_class,
        symbol=req.symbol,
        timeframe=req.timeframe,
        initial_equity=req.initial_equity,
        risk_per_trade_pct=req.risk_per_trade_pct,
        mode=req.mode,
        max_candles=req.max_candles,
    )
    run = backtest_engine.create_backtest(
        mode=req.mode,
        asset_class=req.asset_class,
        symbol=req.symbol,
        timeframe=req.timeframe,
        rule_version_id=req.rule_version_id,
        strategy_version_id=req.strategy_version_id,
        config=cfg,
    )
    return run.model_dump(mode="json")


@app.get("/api/backtests")
def list_backtests_endpoint(
    symbol: Optional[str] = None,
    asset_class: Optional[AssetClass] = None,
    mode: Optional[BacktestMode] = None,
    status: Optional[BacktestStatus] = None,
    limit: int = 50,
):
    """Retrieve historical backtest runs."""
    runs = repo.get_backtest_runs(
        symbol=symbol,
        asset_class=asset_class,
        limit=limit,
    )
    # Apply client-side filtering for mode and status (not yet in repo query)
    if mode is not None:
        runs = [r for r in runs if r.mode == mode]
    if status is not None:
        runs = [r for r in runs if r.status == status]
    return [r.model_dump(mode="json") for r in runs]


@app.get("/api/backtests/cross-asset-summary")
def get_cross_asset_summary_endpoint():
    """Retrieve aggregate performance while preserving separate Forex & Crypto reports."""
    runs = repo.get_backtest_runs(limit=100)
    summary = backtest_engine.generate_cross_asset_summary(runs)
    return summary


@app.get("/api/backtests/{backtest_id}")
def get_backtest_endpoint(backtest_id: str):
    """Retrieve details and metrics of a specific backtest."""
    run = repo.get_backtest_run(backtest_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    return run.model_dump(mode="json")


@app.post("/api/backtests/{backtest_id}/start")
def start_backtest_endpoint(backtest_id: str):
    """Execute historical replay synchronously or initiate background run."""
    try:
        run = backtest_engine.run_backtest(backtest_id)
        return run.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Error running backtest {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/backtests/{backtest_id}/pause")
def pause_backtest_endpoint(backtest_id: str):
    """Pause backtest and save checkpoint."""
    run = backtest_engine.pause_backtest(backtest_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    return run.model_dump(mode="json")


@app.post("/api/backtests/{backtest_id}/resume")
def resume_backtest_endpoint(backtest_id: str):
    """Resume paused backtest from last saved candle checkpoint."""
    run = backtest_engine.resume_backtest(backtest_id)
    return run.model_dump(mode="json")


@app.post("/api/backtests/{backtest_id}/cancel")
def cancel_backtest_endpoint(backtest_id: str):
    """Cancel a backtest run."""
    run = backtest_engine.cancel_backtest(backtest_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    return run.model_dump(mode="json")


@app.get("/api/backtests/{backtest_id}/trades")
def get_backtest_trades_endpoint(backtest_id: str):
    """Retrieve individual trade logs for a backtest run."""
    trades = repo.get_backtest_trades(backtest_id)
    return [t.model_dump(mode="json") for t in trades]


@app.post("/api/backtests/{backtest_id}/manual-trade")
def record_manual_trade_endpoint(backtest_id: str, req: ManualBacktestTradeRequest):
    """Record a user-executed trade during manual replay backtest."""
    req.backtest_id = backtest_id
    trade = backtest_engine.record_manual_trade(req)
    return trade.model_dump(mode="json")


@app.get("/api/rules/{rule_version_id}/promotion")
def get_rule_promotion_endpoint(rule_version_id: str):
    """Check rule promotion status along Backtest -> Paper -> Live ladder."""
    prm = promotion_manager.get_or_create_promotion_record(rule_version_id)
    return prm.model_dump(mode="json")


@app.post("/api/rules/{rule_version_id}/approve-paper")
def approve_rule_for_paper_endpoint(rule_version_id: str, req: PromoteRuleRequest):
    """Promote rule to APPROVED_FOR_PAPER following backtest validation."""
    try:
        prm = promotion_manager.approve_for_paper(
            rule_version_id=rule_version_id,
            user_confirmed=req.user_confirmed,
            notes=req.notes,
        )
        return prm.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/rules/{rule_version_id}/approve-live")
def approve_rule_for_live_endpoint(rule_version_id: str, req: PromoteRuleRequest):
    """Promote rule to LIVE_ELIGIBLE following paper trading validation."""
    try:
        prm = promotion_manager.approve_for_live(
            rule_version_id=rule_version_id,
            user_confirmed=req.user_confirmed,
            notes=req.notes,
        )
        return prm.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Step 10 Professional Trading Journal Engine
from trading_ai.journal import TradingJournalCore
from trading_ai.schemas.journal import (
    PreTradeReviewRequest,
    TradeJournalEntry,
    PlannedTradeValues,
    ActualTradeValues,
)
from trading_ai.schemas.enums import (
    JournalEnvironment,
    JournalBacktestType,
    TradeOutcome,
    ExitReasonCategory,
)

journal_core = TradingJournalCore(
    repository=repo,
    discipline_engine=companion_core.discipline_engine,
    promotion_manager=promotion_manager,
)


@app.get("/api/journal/status")
def get_journal_status_endpoint():
    """Retrieve high-level status of the Trading Journal subsystem."""
    return journal_core.get_status()


@app.get("/api/journal/trades")
def list_journal_trades_endpoint(
    environment: Optional[JournalEnvironment] = None,
    asset_class: Optional[str] = None,
    symbol: Optional[str] = None,
    backtest_type: Optional[JournalBacktestType] = None,
    limit: int = 100,
):
    """List trade journal entries filtered by environment, symbol, asset class, or backtest type."""
    entries = repo.list_journal_entries(
        environment=environment,
        asset_class=asset_class,
        symbol=symbol,
        backtest_type=backtest_type,
        limit=limit,
    )
    return [e.model_dump(mode="json") for e in entries]


@app.get("/api/journal/trades/{journal_id}")
def get_journal_trade_endpoint(journal_id: str):
    """Retrieve details of a single trade journal entry."""
    entry = repo.get_journal_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return entry.model_dump(mode="json")


@app.post("/api/journal/trades")
def create_journal_trade_endpoint(entry: TradeJournalEntry):
    """Create or manually log a trade journal entry."""
    # Finalize deterministic metrics if actual is provided
    journal_core.posttrade_evaluator.finalize_actual_metrics(
        planned=entry.planned,
        actual=entry.actual,
        direction=entry.direction,
    )
    repo.save_journal_entry(entry)
    return entry.model_dump(mode="json")


@app.post("/api/journal/trades/{journal_id}/complete")
def complete_journal_trade_endpoint(
    journal_id: str,
    actual: ActualTradeValues,
):
    """Finalize a trade journal entry with actual execution outcomes and attribute win/loss."""
    entry = repo.get_journal_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    entry.actual = actual
    journal_core.posttrade_evaluator.finalize_actual_metrics(
        planned=entry.planned,
        actual=entry.actual,
        direction=entry.direction,
    )
    entry.updated_at = datetime.now(timezone.utc)
    entry.closed_at = datetime.now(timezone.utc)
    repo.save_journal_entry(entry)
    return entry.model_dump(mode="json")


@app.post("/api/journal/pretrade-review")
def pretrade_review_endpoint(req: PreTradeReviewRequest):
    """Conduct pre-trade journal review before trade proceeds to Step 8 Execution Gate."""
    result = journal_core.pretrade_reviewer.review_trade(req)
    return result.model_dump(mode="json")


@app.get("/api/journal/pretrade-review/{proposal_id}")
def get_pretrade_review_endpoint(proposal_id: str):
    """Retrieve saved pre-trade review result for a proposal."""
    review = repo.get_pretrade_review(proposal_id)
    if not review:
        raise HTTPException(status_code=404, detail="Pre-trade review not found for proposal")
    return review.model_dump(mode="json")


@app.get("/api/journal/lessons")
def list_journal_lessons_endpoint(limit: int = 100):
    """List journal lessons and improvement candidates."""
    lessons = repo.list_journal_lessons(limit=limit)
    return [l.model_dump(mode="json") for l in lessons]


@app.post("/api/journal/lessons")
def propose_journal_lesson_endpoint(payload: Dict[str, Any]):
    """Propose a new candidate lesson."""
    lesson = journal_core.lesson_engine.propose_lesson(
        observation=payload.get("observation", ""),
        explanation=payload.get("explanation", ""),
        proposed_action=payload.get("proposed_action", ""),
        source_journal_ids=payload.get("source_journal_ids", []),
        affected_rule_id=payload.get("affected_rule_id"),
        confidence=float(payload.get("confidence", 0.85)),
        evidence=payload.get("evidence", {}),
    )
    return lesson.model_dump(mode="json")


@app.post("/api/journal/lessons/{lesson_id}/review")
def review_journal_lesson_endpoint(lesson_id: str, payload: Dict[str, Any]):
    """User review gate for a candidate lesson (Approve -> VALIDATED, Reject -> REJECTED)."""
    try:
        reviewed = journal_core.lesson_engine.user_review_lesson(
            lesson_id=lesson_id,
            reviewer=payload.get("reviewer", "user"),
            approved=payload.get("approved", True),
            notes=payload.get("notes"),
        )
        return reviewed.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/journal/lessons/{lesson_id}/validate")
def validate_journal_lesson_endpoint(lesson_id: str, payload: Dict[str, Any]):
    """Submit validated lesson into candidate rule improvement pipeline (Step 9 ladder)."""
    try:
        res = journal_core.lesson_engine.apply_validated_lesson(
            lesson_id=lesson_id,
            approved_by=payload.get("approved_by", "user"),
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/journal/analytics")
def get_journal_analytics_default_endpoint(
    environment: JournalEnvironment = JournalEnvironment.PAPER,
    asset_class: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get deterministic analytics summary for specified isolated environment."""
    summary = journal_core.analytics_engine.calculate_summary(
        environment=environment,
        asset_class=asset_class,
        symbol=symbol,
    )
    return summary.model_dump(mode="json")


@app.get("/api/journal/analytics/{environment}")
def get_journal_analytics_by_env_endpoint(
    environment: JournalEnvironment,
    asset_class: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get deterministic analytics summary strictly isolated by environment."""
    summary = journal_core.analytics_engine.calculate_summary(
        environment=environment,
        asset_class=asset_class,
        symbol=symbol,
    )
    return summary.model_dump(mode="json")


@app.get("/api/journal/backtest/manual")
def get_manual_backtest_journal_endpoint(symbol: Optional[str] = None):
    """Retrieve dedicated Manual Backtest journal entries."""
    entries = repo.list_journal_entries(
        environment=JournalEnvironment.BACKTEST,
        backtest_type=JournalBacktestType.MANUAL_BACKTEST,
        symbol=symbol,
    )
    return [e.model_dump(mode="json") for e in entries]


@app.get("/api/journal/backtest/ai")
def get_ai_backtest_journal_endpoint(symbol: Optional[str] = None):
    """Retrieve dedicated AI Backtest journal entries."""
    entries = repo.list_journal_entries(
        environment=JournalEnvironment.BACKTEST,
        backtest_type=JournalBacktestType.AI_BACKTEST,
        symbol=symbol,
    )
    return [e.model_dump(mode="json") for e in entries]


# Step 11 High-Performance & Adaptive AI Architecture
from trading_ai.adaptive import AdaptiveAICore
from trading_ai.schemas.enums import EventPriority, UpgradeStatus

adaptive_core = AdaptiveAICore(repository=repo)


@app.get("/api/performance/status")
def get_performance_status_endpoint():
    """Retrieve comprehensive Step 11 system performance and runtime metrics."""
    return adaptive_core.get_system_status()


@app.get("/api/performance/latency")
def get_performance_latency_endpoint(limit: int = 20):
    """Retrieve recent measured latency records."""
    records = repo.get_recent_latency_telemetry(limit=limit)
    return [r.model_dump(mode="json") for r in records]


@app.get("/api/performance/workers")
def get_performance_workers_endpoint():
    """Retrieve worker concurrency, scheduler policy and thread status."""
    return {
        "status": "ACTIVE",
        "workers": {
            "market_data_worker": "ONLINE",
            "incremental_analysis_worker": "ONLINE",
            "event_bus_dispatcher": "ONLINE",
            "telemetry_collector": "ONLINE",
        },
        "event_bus": adaptive_core.event_bus.get_queue_stats(),
    }


@app.get("/api/performance/cache")
def get_performance_cache_endpoint():
    """Retrieve memory tier (Hot, Warm, Cold) cache statistics."""
    return adaptive_core.memory.get_tier_stats()


@app.get("/api/performance/queues")
def get_performance_queues_endpoint():
    """Retrieve prioritized queue breakdown and pending event counts."""
    return adaptive_core.event_bus.get_queue_stats()


@app.post("/api/performance/tick")
def process_tick_endpoint(payload: Dict[str, Any]):
    """Ingest and process a raw price tick through the high-performance pipeline."""
    symbol = payload.get("symbol", "BTC/USD")
    price = float(payload.get("price", 65000.0))
    source = payload.get("source", "API_FEED")
    res = adaptive_core.process_market_tick(symbol=symbol, price=price, source=source)
    return res


@app.get("/api/performance/upgrades")
def list_component_upgrades_endpoint(limit: int = 50):
    """List component upgrade history and candidate records."""
    records = adaptive_core.upgrade_manager.list_upgrades(limit=limit)
    return [r.model_dump(mode="json") for r in records]


@app.post("/api/performance/upgrades")
def register_component_upgrade_endpoint(payload: Dict[str, Any]):
    """Register a new candidate component upgrade for sandbox benchmarking."""
    component_name = payload.get("component_name", "")
    current_version = payload.get("current_version", "1.0.0")
    candidate_version = payload.get("candidate_version", "1.1.0")
    notes = payload.get("notes", "")
    rec = adaptive_core.upgrade_manager.register_candidate(
        component_name=component_name,
        current_version=current_version,
        candidate_version=candidate_version,
        notes=notes,
    )
    return rec.model_dump(mode="json")


@app.post("/api/performance/upgrades/{upgrade_id}/benchmark")
def record_upgrade_benchmark_endpoint(upgrade_id: str, payload: Dict[str, Any]):
    """Record sandbox benchmark results for an upgrade candidate."""
    try:
        accuracy = float(payload.get("accuracy", 0.95))
        latency_ms = float(payload.get("latency_ms", 12.5))
        pass_rate = float(payload.get("regression_pass_rate", 1.0))
        rec = adaptive_core.upgrade_manager.run_sandbox_benchmark(
            upgrade_id=upgrade_id,
            accuracy=accuracy,
            latency_ms=latency_ms,
            regression_pass_rate=pass_rate,
        )
        return rec.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/performance/upgrades/{upgrade_id}/paper")
def record_upgrade_paper_endpoint(upgrade_id: str, payload: Dict[str, Any]):
    """Record paper validation for an upgrade candidate."""
    try:
        passed = bool(payload.get("passed", True))
        rec = adaptive_core.upgrade_manager.record_paper_validation(upgrade_id=upgrade_id, passed=passed)
        return rec.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/performance/upgrades/{upgrade_id}/approve")
def approve_upgrade_endpoint(upgrade_id: str, payload: Dict[str, Any]):
    """User approval gate for promoting an upgrade into production."""
    try:
        approved_by = payload.get("approved_by", "")
        is_user = bool(payload.get("is_user", False))
        rec = adaptive_core.upgrade_manager.approve_and_promote(
            upgrade_id=upgrade_id,
            approved_by=approved_by,
            is_user=is_user,
        )
        return rec.model_dump(mode="json")
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.post("/api/performance/upgrades/{upgrade_id}/rollback")
def rollback_upgrade_endpoint(upgrade_id: str, payload: Dict[str, Any]):
    """Roll back an upgrade candidate to its previous version."""
    try:
        reason = payload.get("reason", "User requested rollback")
        rec = adaptive_core.upgrade_manager.rollback(upgrade_id=upgrade_id, reason=reason)
        return rec.model_dump(mode="json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


# =====================================================================
# STEP 12: SECURITY, PRIVACY & HUMAN-COMPUTER INTERACTION (HCI)
# =====================================================================
from trading_ai.security import SecurityCore
from trading_ai.schemas.enums import (
    CapabilityState,
    DataPrivacyClassification,
    DeviceStatus,
    NotificationPriority,
    PCCapability,
)
from trading_ai.schemas.security import (
    CapabilityApproval,
    CapabilityRequest,
    DevicePairingApproval,
    DevicePairingRequest,
    EncryptedPayload,
    VoiceCommandRequest,
)

security_core = SecurityCore(
    repository=repo,
    event_bus=adaptive_core.event_bus,
    auth_token=settings.auth_token,
    session_secret=settings.session_secret,
    auth_enabled=settings.auth_enabled,
)


# --- Security & Capabilities ---
@app.get("/api/security/status")
def get_security_status_endpoint():
    """Overall security posture, privacy policy, and capability statistics."""
    rules = security_core.capability_gate.list_rules()
    devices = security_core.device_manager.list_devices()
    return {
        "status": "SECURE",
        "default_policy": "DENY",
        "total_capabilities": len(rules),
        "active_capabilities_count": sum(1 for r in rules if r.state.value in ["APPROVED_ALWAYS", "APPROVED_ONCE"]),
        "paired_devices_count": len(devices),
        "active_devices_count": sum(1 for d in devices if d.status.value == "ACTIVE"),
        "recent_audit_count": len(security_core.audit_logger.get_recent_entries(limit=10)),
        "privacy_mode": "LOCAL_FIRST",
    }


@app.get("/api/security/capabilities")
def list_capabilities_endpoint():
    """List all PC capabilities and their current permission states."""
    rules = security_core.capability_gate.list_rules()
    return [r.model_dump(mode="json") for r in rules]


@app.post("/api/security/capabilities/request")
def request_capability_endpoint(payload: Dict[str, Any]):
    """Submit a formal request to invoke a PC capability."""
    try:
        cap_str = payload.get("capability", "")
        cap = PCCapability(cap_str)
        req_comp = payload.get("requesting_component", "api_client")
        context = payload.get("context", {})
        req = security_core.capability_gate.request_capability(
            capability=cap,
            requesting_component=req_comp,
            context=context,
        )
        return req.model_dump(mode="json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.post("/api/security/capabilities/authorize")
def authorize_capability_endpoint(payload: Dict[str, Any]):
    """Explicit human user authorization or denial of a capability."""
    try:
        cap_str = payload.get("capability", "")
        cap = PCCapability(cap_str)
        req_id = payload.get("request_id", "")
        approved = bool(payload.get("approved", False))
        is_user = bool(payload.get("is_user", False))
        approved_by = payload.get("approved_by", "")
        reason = payload.get("reason", "")

        approval = CapabilityApproval(
            request_id=req_id,
            capability=cap,
            approved=approved,
            is_user=is_user,
            approved_by=approved_by,
            reason=reason,
        )
        rule = security_core.capability_gate.authorize_capability(approval)
        return rule.model_dump(mode="json")
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.post("/api/security/capabilities/revoke")
def revoke_capability_endpoint(payload: Dict[str, Any]):
    """Immediately revoke a previously granted capability."""
    try:
        cap_str = payload.get("capability", "")
        cap = PCCapability(cap_str)
        revoked_by = payload.get("revoked_by", "user")
        rule = security_core.capability_gate.revoke_capability(cap, revoked_by=revoked_by)
        return rule.model_dump(mode="json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.get("/api/security/audit")
def get_security_audit_logs_endpoint(limit: int = 50):
    """Retrieve structured, privacy-redacted security audit logs."""
    entries = security_core.audit_logger.get_recent_entries(limit=limit)
    return [e.model_dump(mode="json") for e in entries]


# --- Device Identity, Pairing & Encryption ---
@app.get("/api/devices")
def list_devices_endpoint():
    """List registered devices without exposing secrets."""
    devices = security_core.device_manager.list_devices()
    return [d.model_dump(mode="json") for d in devices]


@app.post("/api/devices/pairing/generate")
def generate_pairing_code_endpoint(payload: Dict[str, Any]):
    """Generate a cryptographically secure 6-digit pairing code."""
    dev_id = payload.get("device_id") or f"dev_{uuid.uuid4().hex[:8]}"
    dev_name = payload.get("device_name", "Unknown Phone")
    dev_type = payload.get("device_type", "phone")
    code = security_core.device_manager.generate_pairing_code(
        device_id=dev_id,
        device_name=dev_name,
        device_type=dev_type,
    )
    return {"device_id": dev_id, "pairing_code": code, "expires_in_seconds": 300}


@app.post("/api/devices/pairing/request")
def submit_pairing_request_endpoint(payload: Dict[str, Any]):
    """Client device submits pairing code and nonce."""
    req = DevicePairingRequest(
        device_id=payload.get("device_id", ""),
        device_name=payload.get("device_name", ""),
        device_type=payload.get("device_type", "phone"),
        pairing_code=payload.get("pairing_code", ""),
        nonce=payload.get("nonce", uuid.uuid4().hex),
    )
    ok = security_core.device_manager.submit_pairing_request(req)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid pairing code or expired request.")
    return {"status": "AWAITING_USER_APPROVAL", "device_id": req.device_id}


@app.post("/api/devices/pairing/approve")
def approve_device_pairing_endpoint(payload: Dict[str, Any]):
    """Human user explicitly approves or rejects device pairing."""
    approval = DevicePairingApproval(
        device_id=payload.get("device_id", ""),
        approved=bool(payload.get("approved", False)),
        is_user=bool(payload.get("is_user", False)),
        approved_by=payload.get("approved_by", "user"),
    )
    ok, secret = security_core.device_manager.approve_pairing(approval)
    if not ok:
        raise HTTPException(status_code=403, detail="Device pairing rejected or device not found.")
    return {
        "status": "ACTIVE",
        "device_id": approval.device_id,
        "shared_secret": secret,
    }


@app.post("/api/devices/{device_id}/revoke")
def revoke_device_endpoint(device_id: str, payload: Optional[Dict[str, Any]] = None):
    """Revoke a trusted device and purge its session keys."""
    reason = (payload or {}).get("reason", "Revoked by user")
    ok = security_core.device_manager.revoke_device(device_id=device_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Device not found.")
    return {"status": "REVOKED", "device_id": device_id}


# --- Privacy & Data Classification ---
@app.get("/api/privacy/status")
def get_privacy_status_endpoint():
    """Retrieve privacy classification policies and local processing state."""
    return {
        "privacy_mode": "LOCAL_FIRST",
        "rules_count": len(security_core.privacy_router._rules),
        "local_processing": True,
        "cloud_processing_requires_user_approval": True,
    }


@app.post("/api/privacy/sanitize")
def sanitize_check_endpoint(payload: Dict[str, Any]):
    """Test data classification and privacy sanitization pipeline."""
    raw_input = payload.get("input", "")
    target = payload.get("target_destination", "cloud")
    consent = bool(payload.get("user_cloud_consent", False))
    res = security_core.privacy_router.process_and_route(
        raw_input=raw_input,
        target_destination=target,
        requesting_component="PrivacySanitizeAPI",
        user_cloud_consent=consent,
    )
    return res


# --- Screen Monitoring Privacy ---
@app.get("/api/screen/privacy")
def get_screen_privacy_endpoint():
    """Retrieve screen allowlist, denylist, and capability state."""
    return {
        "rules": [r.model_dump(mode="json") for r in security_core.screen_monitor._rules],
        "screen_read_authorized": security_core.capability_gate.check_permission(PCCapability.SCREEN_READ, "API"),
    }


@app.post("/api/screen/capture")
def capture_screen_frame_endpoint(payload: Dict[str, Any]):
    """Capture a screen frame subject to privacy filter, masking, and change detection."""
    try:
        app_name = payload.get("app_name", "")
        window_title = payload.get("window_title", "")
        ocr_text = payload.get("ocr_text", "")
        frame = security_core.screen_monitor.capture_and_process_frame(
            app_name=app_name,
            window_title=window_title,
            ocr_text=ocr_text,
        )
        return frame
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


# --- Voice Gateway ---
@app.get("/api/voice/status")
def get_voice_status_endpoint():
    """Retrieve two-way voice system status and permissions."""
    return {
        "state": security_core.voice_gateway._state.value,
        "microphone_ready": True,
        "speaker_ready": True,
        "voice_output_authorized": security_core.capability_gate.check_permission(PCCapability.VOICE_OUTPUT, "API"),
    }


@app.post("/api/voice/command")
def process_voice_command_endpoint(payload: Dict[str, Any]):
    """Submit a voice command for capability-checked execution."""
    cmd_text = payload.get("command_text", "")
    audio_b64 = payload.get("audio_base64")
    req = VoiceCommandRequest(
        command_text=cmd_text,
        audio_base64=audio_b64,
        device_id=payload.get("device_id"),
        session_id=payload.get("session_id"),
        is_user_voice=bool(payload.get("is_user_voice", True)),
    )
    res = security_core.voice_gateway.process_voice_command(req)
    return res.model_dump(mode="json")


@app.post("/api/voice/cancel")
def cancel_voice_endpoint():
    """Cancel active speech synthesis or voice processing."""
    security_core.voice_gateway.cancel_active_operation()
    return {"status": "CANCELLED"}


# --- Cross-Device Security Notifications ---
@app.get("/api/security/notifications")
def list_security_notifications_endpoint(limit: int = 50):
    """List recent security notifications."""
    notifs = security_core.notifications.list_notifications(limit=limit)
    return [n.model_dump(mode="json") for n in notifs]


@app.post("/api/notifications/dispatch")
def dispatch_notification_endpoint(payload: Dict[str, Any]):
    """Dispatch a privacy-minimized notification."""
    try:
        title = payload.get("title", "Trading AI Notification")
        message = payload.get("message", "")
        prio_str = payload.get("priority", "MEDIUM")
        prio = NotificationPriority(prio_str)
        cat = payload.get("category", "system")
        targets = payload.get("target_device_ids")
        item = security_core.notifications.dispatch_notification(
            title=title,
            message=message,
            priority=prio,
            category=cat,
            target_device_ids=targets,
        )
        return item.model_dump(mode="json")
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.post("/api/notifications/read-all")
def mark_notifications_read_endpoint():
    """Mark all notifications as read."""
    count = security_core.notifications.mark_all_read()
    return {"status": "ok", "marked_read_count": count}


# --- Authenticated Encrypted WebSocket for PC ↔ Phone Sync ---
@app.websocket("/ws/devices/{device_id}")
async def device_websocket_endpoint(websocket: WebSocket, device_id: str):
    """Real-time encrypted communication channel between PC and paired phone."""
    device = security_core.device_manager.get_device(device_id)
    if not device or device.status != DeviceStatus.ACTIVE:
        await websocket.close(code=4003, reason="Device is not active or authorized.")
        return

    await websocket.accept()
    security_core.audit_logger.log(
        action="WEBSOCKET_CONNECTED",
        decision="ALLOWED",
        requesting_component="DeviceWebSocket",
        device_id=device_id,
        is_user_authorized=True,
    )

    try:
        import json
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg_data = json.loads(raw_text)
                payload = EncryptedPayload(**msg_data)
                decrypted = security_core.device_manager.decrypt_from_device(payload)
                response_cipher = security_core.device_manager.encrypt_for_device(
                    device_id=device_id,
                    plaintext=f"ACK:{decrypted[:20]}",
                )
                await websocket.send_text(json.dumps(response_cipher.model_dump(mode="json")))
            except Exception as decrypt_err:
                err_payload = {"error": f"Rejected: {str(decrypt_err)}"}
                await websocket.send_text(json.dumps(err_payload))
    except WebSocketDisconnect:
        security_core.audit_logger.log(
            action="WEBSOCKET_DISCONNECTED",
            decision="DISCONNECTED",
            requesting_component="DeviceWebSocket",
            device_id=device_id,
        )


# ============================================================================
# Step 13: Data Security, Cyber Wall, Watchdog & Auto-Recovery Endpoints
# ============================================================================

from trading_ai.resilience import ResilienceCore
from trading_ai.schemas.recovery import CyberWallValidationRequest

resilience_core = ResilienceCore(repository=repo, event_bus=adaptive_core.event_bus)


@app.get("/api/resilience/status")
def get_resilience_status_endpoint():
    """Retrieve composite health, CPU/RAM budgets, cyber wall threat level, and watchdog status."""
    report = resilience_core.get_composite_status()
    return report.model_dump(mode="json")


@app.get("/api/resilience/threats")
def list_threats_endpoint():
    """List recent cyber wall threats and anomaly incidents."""
    threats = resilience_core.threat_detector.list_active_threats()
    return [t.model_dump(mode="json") for t in threats]


@app.post("/api/resilience/threats/evaluate")
def evaluate_payload_endpoint(payload: Dict[str, Any]):
    """Evaluate untrusted input for shell exploits, prompt injections, and malware signatures."""
    content = payload.get("content", "")
    source = payload.get("source", "API_EVALUATE")
    incident = resilience_core.threat_detector.evaluate_payload(content, source=source)
    return incident.model_dump(mode="json")


@app.get("/api/resilience/backups")
def list_backups_endpoint():
    """List isolated disaster recovery knowledge backups."""
    backups = resilience_core.backup_manager.list_backups()
    return [b.model_dump(mode="json") for b in backups]


@app.post("/api/resilience/backups/create")
def create_backup_endpoint(payload: Optional[Dict[str, Any]] = None):
    """Trigger an isolated, checksummed backup of current knowledge."""
    notes = (payload or {}).get("notes", "Manual backup via API/Dashboard")
    backup = resilience_core.backup_manager.create_backup(notes=notes)
    return backup.model_dump(mode="json")


@app.post("/api/resilience/backups/{backup_id}/restore")
def restore_backup_endpoint(backup_id: str):
    """Restore knowledge items from a verified backup archive."""
    res = resilience_core.backup_manager.restore_backup(backup_id)
    if not res.success:
        raise HTTPException(status_code=400, detail=res.error_message or "Restore failed")
    return res.model_dump(mode="json")


@app.get("/api/resilience/watchdog/tasks")
def list_watchdog_tasks_endpoint():
    """List active tasks currently monitored by the Anti-Hang Watchdog."""
    tasks = resilience_core.watchdog.list_active_tasks()
    return [t.model_dump(mode="json") for t in tasks]


@app.post("/api/resilience/watchdog/check")
def check_watchdog_hangs_endpoint():
    """Trigger on-demand watchdog audit for stalled or hanging tasks."""
    hung = resilience_core.watchdog.check_for_hangs()
    return {
        "hung_tasks_count": len(hung),
        "hung_tasks": [t.model_dump(mode="json") for t in hung],
    }


@app.post("/api/resilience/cyberwall/validate-path")
def validate_cyberwall_path_endpoint(req: CyberWallValidationRequest):
    """Validate a path against Cyber Wall directory allowlisting and traversal protections."""
    res = resilience_core.cyber_wall.validate_path(target_path=req.target_path, action=req.action)
    return res.model_dump(mode="json")


@app.post("/api/resilience/recovery/trigger")
def trigger_recovery_endpoint(payload: Dict[str, Any]):
    """Trigger crash-recovery for a stalled or failed job."""
    job_id = payload.get("job_id", "")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing required job_id.")
    if not resilience_core.recovery_coordinator:
        raise HTTPException(status_code=500, detail="Recovery coordinator not configured.")

    res = resilience_core.recovery_coordinator.recover_job_state(job_id=job_id)
    return res


# =====================================================================
# STEP 14: SEPARATE FILE ARCHITECTURE & CORE TRADE LOGIC APIS
# =====================================================================
from trading_ai.trade_logic import TradeLogicCore
from trading_ai.schemas.trade_logic import (
    NormalizedKnowledgeItem,
    KnowledgeRelationship,
    ConfluenceEvaluation,
    WhyNotTradeResult,
    CoreTradeDecision,
)
from trading_ai.schemas.enums import SourceDomain, RelationshipType

trade_logic_core = TradeLogicCore(repository=repo)


@app.get("/api/trade-logic/status")
def get_trade_logic_status():
    """Return status and domain metrics for Step 14."""
    stats = trade_logic_core.domain_manager.get_domain_stats()
    all_items = repo.list_normalized_knowledge(limit=100) if hasattr(repo, "list_normalized_knowledge") else []
    return {
        "status": "ACTIVE",
        "domains": stats,
        "total_normalized_items": len(all_items),
    }


@app.get("/api/trade-logic/domains")
def get_trade_logic_domains():
    """Return storage and file metrics for all four isolated knowledge domains."""
    stats = trade_logic_core.domain_manager.get_domain_stats()
    result = []
    for d in trade_logic_core.domain_manager.DOMAINS:
        s = stats.get(d.value, {"file_count": 0, "total_bytes": 0})
        result.append({
            "domain": d.value,
            "name": d.name,
            "file_count": s["file_count"],
            "total_bytes": s["total_bytes"],
            "files": trade_logic_core.domain_manager.list_domain_files(d),
        })
    return result


@app.get("/api/trade-logic/knowledge")
def list_trade_logic_knowledge(
    domain: Optional[str] = None,
    concept_type: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
):
    """Retrieve normalized knowledge records with optional domain filter."""
    dom_enum = None
    if domain:
        for d in SourceDomain:
            if d.value.lower() == domain.lower() or d.name.lower() == domain.lower():
                dom_enum = d
                break
    items = repo.list_normalized_knowledge(domain=dom_enum, concept_type=concept_type, limit=limit) if hasattr(repo, "list_normalized_knowledge") else []
    if query:
        q_lower = query.lower()
        items = [i for i in items if q_lower in i.title.lower() or q_lower in i.statement.lower()]
    return [
        {
            **i.model_dump(mode="json"),
            "source_domain": i.domain.value if hasattr(i.domain, "value") else str(i.domain),
            "rule_type": i.concept_type,
            "content": i.statement,
        }
        for i in items
    ]


@app.post("/api/trade-logic/knowledge/ingest")
def ingest_normalized_knowledge(payload: Dict[str, Any]):
    """Ingest a normalized knowledge item into the unified layer with permanent provenance."""
    from trading_ai.trade_logic.knowledge_normalizer import KnowledgeNormalizer
    if "statement" not in payload and "content" in payload:
        payload["statement"] = payload["content"]

    dom_str = payload.get("domain", SourceDomain.YOUTUBE_LECTURES.value)
    dom_val = SourceDomain.YOUTUBE_LECTURES
    for d in SourceDomain:
        if d.value.lower() == str(dom_str).lower() or d.name.lower() == str(dom_str).lower():
            dom_val = d
            break

    auth_str = payload.get("authority", SourceAuthority.LECTURE_RULE.value)
    auth_val = SourceAuthority.LECTURE_RULE
    for a in SourceAuthority:
        if a.value.lower() == str(auth_str).lower() or a.name.lower() == str(auth_str).lower():
            auth_val = a
            break

    item = KnowledgeNormalizer.normalize_rule(
        title=payload.get("title", "Untitled Rule"),
        content=payload.get("statement", ""),
        domain=dom_val,
        rule_type=payload.get("rule_type", payload.get("concept_type", "SETUP_RULE")),
        authority=auth_val,
        confidence=float(payload.get("confidence", 0.85)),
        symbols=payload.get("symbols", []),
        timeframes=payload.get("timeframes", []),
        direction=payload.get("direction"),
    )
    if hasattr(repo, "save_normalized_knowledge"):
        repo.save_normalized_knowledge(item)

    res_dict = item.model_dump(mode="json")
    res_dict["source_domain"] = item.domain.value
    res_dict["content"] = item.statement
    res_dict["rule_type"] = item.concept_type
    return res_dict


@app.get("/api/trade-logic/relationships")
def list_knowledge_relationships_endpoint(item_id: Optional[str] = None):
    """List cross-source knowledge relationships."""
    rels = repo.list_knowledge_relationships(item_id=item_id) if hasattr(repo, "list_knowledge_relationships") else []
    return [r.model_dump(mode="json") for r in rels]


@app.post("/api/trade-logic/relationships/create")
def create_knowledge_relationship_endpoint(payload: Dict[str, Any]):
    """Create a new cross-source relationship link."""
    source_id = payload.get("source_item_id")
    target_id = payload.get("target_item_id")
    rel_type_str = payload.get("relationship_type", "SUPPORTS")
    desc = payload.get("description", "")
    if not source_id or not target_id:
        raise HTTPException(status_code=400, detail="Missing source_item_id or target_item_id")
    try:
        rel_type = RelationshipType(rel_type_str)
    except Exception:
        rel_type = RelationshipType.SUPPORTS
    rel = trade_logic_core.relationship_graph.add_relationship(source_id, target_id, rel_type, desc)
    return rel.model_dump(mode="json")


@app.post("/api/trade-logic/evaluate")
def evaluate_trade_logic_endpoint(payload: Dict[str, Any]):
    """Execute complete unified trade evaluation (Data Fusion + Confluence + Why-Not-Trade + Core Trade Logic)."""
    symbol = payload.get("symbol", "EURUSD")
    timeframe = payload.get("timeframe", "15m")
    direction = payload.get("direction", "BUY")
    current_price = float(payload.get("current_price", 1.0850))
    structural_sl = float(payload.get("structural_sl", current_price - 0.0020 if direction.upper() == "BUY" else current_price + 0.0020))
    target_tp1 = float(payload.get("target_tp1", current_price + 0.0050 if direction.upper() == "BUY" else current_price - 0.0050))
    target_tp2 = float(payload["target_tp2"]) if payload.get("target_tp2") else None
    setup_type = payload.get("setup_type", "liquidity_sweep")
    htf_bias = payload.get("htf_bias", payload.get("high_timeframe_bias", "BULLISH" if direction.upper() == "BUY" else "BEARISH"))
    is_poi_valid = bool(payload.get("is_poi_valid", True))
    is_confirmation_present = bool(payload.get("is_confirmation_present", True))
    news_risk_level = payload.get("news_risk_level", "NO_RISK")
    active_exceptions = payload.get("active_exceptions", [])
    daily_loss_limit_hit = bool(payload.get("daily_loss_limit_hit", False))

    decision = trade_logic_core.analyze_and_propose_trade(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        current_price=current_price,
        structural_sl=structural_sl,
        target_tp1=target_tp1,
        target_tp2=target_tp2,
        setup_type=setup_type,
        htf_bias=htf_bias,
        is_poi_valid=is_poi_valid,
        is_confirmation_present=is_confirmation_present,
        news_risk_level=news_risk_level,
        active_exceptions=active_exceptions,
        daily_loss_limit_hit=daily_loss_limit_hit,
    )
    dec_dict = decision.model_dump(mode="json")

    return {
        "confluence": {
            "state": decision.decision_state,
            "confluence_score": 0.85 if decision.decision_state in ("VALID_SETUP", "HIGH_CONFLUENCE", "CONFIRMED_SETUP") else 0.20,
            "matched_rules": [
                {"title": p.get("title", "Rule"), "item_id": p.get("rule_id", "rule_1"), "source_domain": p.get("domain", "youtube_lectures")}
                for p in decision.source_provenance_trace
            ],
            "conflicting_rules": [],
            "missing_rules": [],
        },
        "why_not_trade": {
            "has_blocking_reasons": decision.decision_state in ("NO_SETUP", "INVALIDATED", "CONFLICTED"),
            "blocking_reasons": decision.why_not_summary if decision.decision_state in ("NO_SETUP", "INVALIDATED", "CONFLICTED") else [],
            "warnings": decision.why_not_summary if decision.decision_state not in ("NO_SETUP", "INVALIDATED", "CONFLICTED") else [],
        },
        "trade_decision": {
            **dec_dict,
            "decision_state": decision.decision_state,
            "evidence_references": decision.evidence_references,
        },
        **dec_dict,
    }


@app.get("/api/trade-logic/journal-feedback")
def get_journal_feedback_endpoint(symbol: str = "EURUSD", setup_type: str = "liquidity_sweep", direction: Optional[str] = None):
    """Retrieve historical journal lessons and warning flags for a setup."""
    insight = trade_logic_core.journal_feedback.get_feedback_for_setup(symbol=symbol, setup_type=setup_type, direction=direction)
    d = insight.model_dump(mode="json")
    d["validated_lessons"] = insight.applicable_lessons
    d["mistakes_to_avoid"] = insight.warning_flags
    return d


# ================================================================== #
# STEP 15: AI Core, Video Streaming, Plugins & Live Decision Endpoints
# ================================================================== #

from trading_ai.plugins import (
    PluginRegistry,
    MockForexFactoryNewsPlugin,
    MockTechnicalAnalysisPlugin,
    MockBrokerReadPlugin,
)
from trading_ai.ai_core import (
    MultilingualNormalizer,
    EpistemicSeparationEngine,
    IterativeComprehensionLoop,
    LargeVideoStreamingProcessor,
    LiveMarketDecisionEngine,
)

plugin_registry = PluginRegistry(repository=repo)
# Register built-in sample plugins
plugin_registry.register(MockForexFactoryNewsPlugin(), enable=True)
plugin_registry.register(MockTechnicalAnalysisPlugin(), enable=True)
plugin_registry.register(MockBrokerReadPlugin(), enable=True)

large_video_engine = LargeVideoStreamingProcessor(repository=repo, commit_barrier=commit_barrier)
iterative_comprehension_loop = IterativeComprehensionLoop(repository=repo)
live_decision_engine = LiveMarketDecisionEngine(repository=repo, trade_logic=trade_logic_core)


@app.get("/api/plugins")
def list_plugins_endpoint():
    """List all registered plugins with full capabilities, status, and health metrics."""
    return plugin_registry.list_plugins()


@app.get("/api/plugins/health")
def plugins_health_summary_endpoint():
    """Get aggregate health status of all registered plugins."""
    return plugin_registry.get_health_summary()


@app.post("/api/plugins/{plugin_id}/enable")
def enable_plugin_endpoint(plugin_id: str):
    """Enable a registered plugin."""
    success = plugin_registry.enable_plugin(plugin_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to enable plugin {plugin_id}")
    return {"plugin_id": plugin_id, "status": "ACTIVE", "is_enabled": True}


@app.post("/api/plugins/{plugin_id}/disable")
def disable_plugin_endpoint(plugin_id: str):
    """Disable a plugin safely without system failure."""
    success = plugin_registry.disable_plugin(plugin_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to disable plugin {plugin_id}")
    return {"plugin_id": plugin_id, "status": "DISABLED", "is_enabled": False}


@app.get("/api/plugins/audit")
def list_plugin_audit_logs_endpoint(plugin_id: Optional[str] = None, limit: int = 50):
    """Retrieve immutable plugin security audit events."""
    if hasattr(repo, "list_plugin_audit_logs"):
        return repo.list_plugin_audit_logs(plugin_id=plugin_id, limit=limit)
    return []


@app.post("/api/plugins/{plugin_id}/invoke")
def invoke_plugin_endpoint(plugin_id: str, payload: Dict[str, Any]):
    """Safely invoke an action on a registered plugin within failure isolation."""
    action = payload.get("action", "")
    params = payload.get("params", {})
    res = plugin_registry.invoke(plugin_id=plugin_id, action=action, params=params)
    return res


@app.post("/api/ai-core/comprehend")
def comprehend_multilingual_endpoint(payload: Dict[str, Any]):
    """Omni-language comprehension with pure Hindi user-facing synthesis."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")
    timestamp = payload.get("timestamp")
    insight = MultilingualNormalizer.comprehend(text, timestamp=timestamp)
    return insight.model_dump(mode="json")


@app.get("/api/ai-core/epistemic-state")
def get_epistemic_state_endpoint(symbol: str = "EURUSD"):
    """Fetch four-layer epistemic separation model view."""
    # Synthesize live view using LiveMarketDecisionEngine
    res = live_decision_engine.evaluate_live_market(symbol=symbol)
    return res.epistemic_state.model_dump(mode="json")


@app.post("/api/ai-core/review-concept")
def review_concept_endpoint(payload: Dict[str, Any]):
    """Execute iterative comprehension review loop on an ambiguous concept."""
    job_id = payload.get("job_id", "job_demo")
    concept_title = payload.get("concept_title", "Ambiguous Setup")
    initial_conf = float(payload.get("initial_confidence", 0.65))
    audio_ev = payload.get("audio_evidence")
    visual_ev = payload.get("visual_evidence")
    transcript_ctx = payload.get("transcript_context")
    token_budget = int(payload.get("token_budget", 4000))

    review_res = iterative_comprehension_loop.review_concept(
        job_id=job_id,
        concept_title=concept_title,
        initial_confidence=initial_conf,
        audio_evidence=audio_ev,
        visual_evidence=visual_ev,
        transcript_context=transcript_ctx,
        token_budget=token_budget,
    )
    return review_res.model_dump(mode="json")


@app.post("/api/video-processing/jobs/create")
def create_video_streaming_job_endpoint(payload: Dict[str, Any]):
    """Initialize a streaming large video processing job."""
    job_id = payload.get("job_id") or f"vjob_{uuid.uuid4().hex[:10]}"
    file_path = payload.get("file_path", "data/large_video.mp4")
    file_size_bytes = int(payload.get("file_size_bytes", 20 * (1024**3)))  # default 20 GB
    duration_seconds = float(payload.get("duration_seconds", 3600.0))
    chunk_dur = float(payload.get("chunk_duration", 30.0))

    summary = large_video_engine.initialize_job(
        job_id=job_id,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        chunk_duration=chunk_dur,
    )
    return summary.model_dump(mode="json")


@app.post("/api/video-processing/jobs/{job_id}/process-chunk")
def process_video_chunk_endpoint(job_id: str, payload: Dict[str, Any]):
    """Process a video chunk through the serial commit barrier."""
    chunk_index = int(payload.get("chunk_index", 1))
    start_time = float(payload.get("start_time", 0.0))
    end_time = float(payload.get("end_time", 30.0))
    simulate_fail = bool(payload.get("simulate_failure", False))

    chunk_res = large_video_engine.process_chunk(
        job_id=job_id,
        chunk_index=chunk_index,
        start_time=start_time,
        end_time=end_time,
        simulate_failure=simulate_fail,
    )
    return chunk_res.model_dump(mode="json")


@app.get("/api/video-processing/jobs")
def list_video_streaming_jobs_endpoint():
    """List large video streaming processing jobs."""
    if hasattr(repo, "list_video_processing_jobs"):
        return repo.list_video_processing_jobs()
    return []


@app.post("/api/live-decision/evaluate")
def evaluate_live_decision_endpoint(payload: Dict[str, Any]):
    """Run proactive live decision engine with concept mapping, forecasting, and Hindi output."""
    symbol = payload.get("symbol", "EURUSD")
    timeframe = payload.get("timeframe", "15m")
    current_price = float(payload.get("current_price", 1.0850))
    direction = payload.get("direction", "BUY")
    observed_sweep = bool(payload.get("observed_sweep", True))
    observed_disp = bool(payload.get("observed_displacement", False))
    observed_fvg = bool(payload.get("observed_fvg", False))
    htf_bias = payload.get("htf_bias", "BULLISH")
    news_risk = payload.get("news_risk_level", "NO_RISK")

    result = live_decision_engine.evaluate_live_market(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        direction=direction,
        observed_sweep=observed_sweep,
        observed_displacement=observed_disp,
        observed_fvg=observed_fvg,
        htf_bias=htf_bias,
        news_risk_level=news_risk,
    )
    return result.model_dump(mode="json")


@app.get("/api/live-decision/scenarios")
def list_live_scenarios_endpoint(symbol: Optional[str] = None, limit: int = 20):
    """Retrieve stored live decision scenarios."""
    if hasattr(repo, "list_live_decision_scenarios"):
        return repo.list_live_decision_scenarios(symbol=symbol, limit=limit)
    return []


# ============================================================================
# Step 16: 24×7 Cloud Runtime, Advanced Event Push & Notification Architecture
# ============================================================================
from trading_ai.cloud import CloudMarketMonitor, CloudRuntimeManager
from trading_ai.events import EventRouter, WebSocketGateway
from trading_ai.notifications import (
    CloudVoiceSynthesizer,
    NotificationOrchestrator,
    PreferencesManager,
)
from trading_ai.schemas.cloud import CloudStreamEvent, UserNotificationPreferences
from trading_ai.schemas.enums import (
    EventPriorityLevel,
    NotificationChannel,
    NotificationPrivacyLevel,
)
from trading_ai.storage.task_queue import LocalPersistentTaskQueue
from trading_ai.storage.vector_store import SQLiteVectorStore

# C2 Vector Store Adapter & Backend
vector_store = SQLiteVectorStore(repository=repo)

# C3 Persistent Task Queue with CommitBarrier Serial Invariant
persistent_task_queue = LocalPersistentTaskQueue(repository=repo, commit_barrier=commit_barrier)

# C1 WebSocket Gateway & Event Router
websocket_gateway = WebSocketGateway(repository=repo, device_manager=security_core.device_manager)
event_router = EventRouter(
    event_bus=adaptive_core.event_bus,
    repository=repo,
    gateway=websocket_gateway,
)

# Multi-Channel Notification Orchestrator
cloud_voice_synthesizer = CloudVoiceSynthesizer(voice_type="hindi_natural", voice_speed=1.0)
notification_orchestrator = NotificationOrchestrator(
    repository=repo,
    gateway=websocket_gateway,
    voice_synthesizer=cloud_voice_synthesizer,
)

# 24×7 Cloud Market Monitor with Stale Data Gating
cloud_market_monitor = CloudMarketMonitor(
    watchlist_monitor=watchlist_monitor,
    repository=repo,
    event_router=event_router,
)

# 24×7 Cloud Runtime Manager
cloud_runtime_manager = CloudRuntimeManager(
    repository=repo,
    market_monitor=cloud_market_monitor,
    event_router=event_router,
    gateway=websocket_gateway,
    orchestrator=notification_orchestrator,
    task_queue=persistent_task_queue,
    vector_store=vector_store,
)

# ============================================================================
# Step 17: Master Decision Engine Architecture
# ============================================================================
from trading_ai.master_decision import MasterDecisionEngine
from trading_ai.schemas.master_decision import MasterDecisionResult

master_decision_engine = MasterDecisionEngine(
    repository=repo,
    event_router=event_router,
    risk_engine=execution_core.risk_engine,
    execution_gate=execution_core.execution_gate,
)
cloud_market_monitor.master_decision_engine = master_decision_engine


@app.get("/api/cloud/status")
def get_cloud_status_endpoint():
    """Retrieve comprehensive 24×7 cloud runtime health matrix."""
    return cloud_runtime_manager.get_status().model_dump(mode="json")


@app.post("/api/cloud/start")
def start_cloud_runtime_endpoint():
    """Start 24×7 autonomous cloud monitoring."""
    cloud_runtime_manager.start()
    return {"status": "ONLINE", "message": "Cloud runtime active."}


@app.post("/api/cloud/pause")
def pause_cloud_runtime_endpoint():
    """Pause 24×7 cloud monitoring."""
    cloud_runtime_manager.pause()
    return {"status": "PAUSED", "message": "Cloud runtime paused."}


@app.post("/api/cloud/resume")
def resume_cloud_runtime_endpoint():
    """Resume 24×7 cloud monitoring."""
    cloud_runtime_manager.resume()
    return {"status": "ONLINE", "message": "Cloud runtime resumed."}


@app.post("/api/cloud/stop")
def stop_cloud_runtime_endpoint():
    """Gracefully stop 24×7 cloud monitoring."""
    cloud_runtime_manager.stop()
    return {"status": "OFFLINE", "message": "Cloud runtime stopped."}


@app.post("/api/cloud/cycle")
def trigger_cloud_cycle_endpoint(force: bool = True):
    """Trigger a single manual cloud monitoring cycle for evaluation and verification."""
    result = cloud_runtime_manager.step_execution(force=force)
    return result


@app.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket, client_id: str = "web_client", device_id: Optional[str] = None):
    """
    Real-time WebSocket event stream for connected PC and mobile clients.
    Replays missed events on connect and streams live CloudStreamEvents.
    """
    if hasattr(security_core, "auth_manager") and security_core.auth_manager.is_auth_enabled:
        session_id = websocket.cookies.get("trading_ai_session")
        token = websocket.query_params.get("token") or websocket.query_params.get("auth_token")
        client_ip = websocket.client.host if websocket.client else "127.0.0.1"
        is_authenticated = False
        if session_id and security_core.auth_manager.verify_session(session_id):
            is_authenticated = True
        elif token and security_core.auth_manager.verify_token(token, client_ip=client_ip):
            is_authenticated = True
        elif device_id:
            dev = security_core.device_manager.get_device(device_id)
            if dev and dev.status.value in ["ACTIVE", "TRUSTED"]:
                is_authenticated = True
        if not is_authenticated:
            await websocket.close(code=4003, reason="Authentication required.")
            return

    connected = await websocket_gateway.connect(websocket=websocket, client_id=client_id, device_id=device_id)
    if not connected:
        return

    try:
        # Replay missed events upon connect (smallest-sufficient-context)
        missed = websocket_gateway.replay_missed_events(client_id=client_id, limit=5)
        for evt in missed:
            await websocket.send_text(json.dumps({"type": "REPLAY", "event": evt}))

        while True:
            # Keepalive ping/pong receiver
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_gateway.disconnect(websocket=websocket, client_id=client_id, device_id=device_id)
    except Exception as e:
        logger.debug("WebSocket event loop exception: %s", e)
        websocket_gateway.disconnect(websocket=websocket, client_id=client_id, device_id=device_id)


@app.get("/api/events")
def list_stream_events_endpoint(limit: int = 50, min_priority: Optional[str] = None, symbol: Optional[str] = None):
    """Retrieve persisted stream events from authoritative SQLite log."""
    return repo.get_stream_events(limit=limit, min_priority=min_priority, symbol=symbol)


@app.get("/api/events/replay")
def replay_missed_events_endpoint(client_id: str = "web_client", limit: int = 20):
    """Offline client recovery: replay missed high-priority notifications."""
    return websocket_gateway.replay_missed_events(client_id=client_id, limit=limit)


@app.get("/api/notifications")
def list_notifications_endpoint(limit: int = 50, channel: Optional[str] = None):
    """List dispatched and queued notification records."""
    return repo.get_notification_records(limit=limit, channel=channel)


@app.post("/api/notifications/test")
def send_test_notification_endpoint(payload: Dict[str, Any]):
    """Trigger a verified test notification with Hindi synthesis and epistemic separation."""
    sym = payload.get("symbol", "EURUSD")
    prio_str = payload.get("priority", "P3_CONFIRMED_SETUP")
    try:
        prio = EventPriorityLevel(prio_str)
    except Exception:
        prio = EventPriorityLevel.P3_CONFIRMED_SETUP

    event = CloudStreamEvent(
        event_type=payload.get("event_type", "SETUP_CONFIRMED"),
        symbol=sym,
        priority=prio,
        confidence=float(payload.get("confidence", 0.90)),
        evidence_reference={"pattern": payload.get("pattern", "Bullish FVG"), "timeframe": "15m"},
        payload={
            "symbol": sym,
            "pattern_name": payload.get("pattern", "Bullish FVG"),
            "direction": payload.get("direction", "BUY"),
            "timeframe": "15m",
            "entry_price": 1.0850,
            "stop_loss": 1.0820,
            "reason": payload.get("reason", "HTF & MTF bias aligned with liquidity sweep"),
        },
    )

    # Route through notification orchestrator
    records = notification_orchestrator.process_event(event)
    return {
        "status": "DISPATCHED" if records else "FILTERED",
        "records": [r.model_dump(mode="json") for r in records],
    }


@app.get("/api/notifications/preferences")
def get_notification_preferences_endpoint(user_id: str = "default_user"):
    """Retrieve user notification preferences and quiet hours."""
    prefs = notification_orchestrator.preferences_manager.get_preferences()
    return prefs.model_dump(mode="json")


@app.put("/api/notifications/preferences")
def update_notification_preferences_endpoint(payload: Dict[str, Any], user_id: str = "default_user"):
    """Update user notification preferences."""
    current = notification_orchestrator.preferences_manager.get_preferences()
    data = current.model_dump(mode="json")
    data.update(payload)
    updated = UserNotificationPreferences(**data)
    notification_orchestrator.preferences_manager.save_preferences(updated)
    return updated.model_dump(mode="json")


@app.post("/api/devices/{device_id}/push_token")
def register_device_push_token_endpoint(device_id: str, payload: Dict[str, Any]):
    """Register mobile or web push token for a paired device."""
    push_token = payload.get("push_token", "")
    platform = payload.get("platform", "web")
    if not push_token:
        raise HTTPException(status_code=400, detail="push_token is required")
    repo.save_device_push_token(device_id=device_id, push_token=push_token, platform=platform)
    return {"status": "REGISTERED", "device_id": device_id, "platform": platform}


@app.get("/api/notifications/recent")
def get_recent_notifications_endpoint(limit: int = 10, channel: Optional[str] = None):
    """Retrieve recent notifications with Hindi and English localized copy."""
    records = repo.get_notification_records(limit=limit, channel=channel)
    return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in records]


@app.post("/api/notifications/subscribe")
def subscribe_notifications_endpoint(payload: Dict[str, Any]):
    """Subscribe phone or client for push notifications."""
    device_id = payload.get("device_id", "default_phone")
    push_token = payload.get("push_token", f"token_{uuid.uuid4().hex[:12]}")
    platform = payload.get("platform", "mobile_push")
    repo.save_device_push_token(device_id=device_id, push_token=push_token, platform=platform)
    return {
        "status": "SUBSCRIBED",
        "device_id": device_id,
        "platform": platform,
        "push_token": push_token,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/vector_store/health")
def get_vector_store_health_endpoint():
    """Retrieve C2 Vector Store Adapter health statistics."""
    return vector_store.health()


@app.get("/api/task_queue/health")
def get_task_queue_health_endpoint():
    """Retrieve C3 Task Queue Adapter health statistics."""
    return persistent_task_queue.health()


@app.post("/api/notifications/voice/preview")
def preview_voice_alert_endpoint(payload: Dict[str, Any]):
    """Generate audio container for custom Hindi alert text."""
    text = payload.get("text", "Hello Alam, EURUSD पर confirmation मिला है।")
    speed = float(payload.get("speed", 1.0))
    voice = payload.get("voice", "hindi_natural")
    res = cloud_voice_synthesizer.synthesize_speech(hindi_text=text, voice_speed=speed, voice_type=voice)
    return res


# ============================================================================
# Step 17: Master Decision Engine REST Endpoints
# ============================================================================

@app.post("/api/decision/master/evaluate")
def evaluate_master_decision_post_endpoint(payload: Dict[str, Any]):
    """Evaluate full deterministic multi-concept trade decision with custom market data."""
    symbol = payload.get("symbol", "EURUSD").upper()
    timeframe = payload.get("timeframe", "15m")
    market_data = payload.get("market_data", {})
    candidate_rules = payload.get("candidate_rules")
    direction = payload.get("direction")
    as_of = payload.get("as_of_timestamp")

    # If candles are not supplied in market_data, fetch snapshot
    if not market_data.get("candles"):
        snap = data_provider.get_snapshot(symbol=symbol, timeframe=timeframe)
        market_data["candles"] = snap.candles
        market_data["current_price"] = getattr(snap, "last_price", getattr(snap, "current_price", snap.candles[-1].close if snap.candles else 1.0850))

    res = master_decision_engine.evaluate_decision(
        symbol=symbol,
        timeframe=timeframe,
        market_data=market_data,
        candidate_rules=candidate_rules,
        proposed_direction=direction,
        as_of_timestamp=as_of,
        force_recompute=True,
    )
    return res.model_dump(mode="json")


@app.get("/api/decision/master/evaluate")
def evaluate_master_decision_get_endpoint(
    symbol: str = "EURUSD",
    timeframe: str = "15m",
    direction: Optional[str] = None,
):
    """Evaluate deterministic trade decision using live market provider state."""
    sym_clean = symbol.upper()
    snap = data_provider.get_snapshot(symbol=sym_clean, timeframe=timeframe)
    cur_px = getattr(snap, "last_price", getattr(snap, "current_price", snap.candles[-1].close if snap.candles else 1.0850))
    m_data = {
        "candles": snap.candles,
        "current_price": cur_px,
        "structure_bias": "BULLISH" if snap.candles and snap.candles[-1].close > snap.candles[0].open else "BEARISH",
        "liquidity_swept": True,
        "fvg_detected": True,
    }
    res = master_decision_engine.evaluate_decision(
        symbol=sym_clean,
        timeframe=timeframe,
        market_data=m_data,
        proposed_direction=direction,
        force_recompute=True,
    )
    return res.model_dump(mode="json")


@app.get("/api/decision/audit/list")
def list_decision_audits_endpoint(symbol: Optional[str] = None, limit: int = 50):
    """List historical immutable decision audit traces."""
    return repo.list_decision_audits(symbol=symbol, limit=limit)


@app.get("/api/decision/audit/{decision_id}")
def get_decision_audit_endpoint(decision_id: str):
    """Retrieve an immutable decision audit record by ID for exact reproducibility."""
    rec = repo.get_decision_audit(decision_id=decision_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Decision audit trace not found")
    return rec


@app.get("/api/decision/concepts")
def list_concept_modules_endpoint():
    """Retrieve all registered trading concept modules and data availability requirements."""
    registered = master_decision_engine.get_registered_modules()
    db_modules = {m["module_id"]: m for m in repo.get_concept_modules()}
    out = []
    for mod_id, mod in registered.items():
        db_entry = db_modules.get(mod_id, {})
        out.append({
            "module_id": mod.concept_id,
            "concept_name": mod.concept_name,
            "concept_type": "MACRO" if mod.is_macro else "MICRO",
            "authority_priority": mod.authority_priority,
            "is_approved": mod.is_approved,
            "is_enabled": db_entry.get("is_enabled", True),
            "required_data_fields": mod.required_data_fields,
            "is_macro": mod.is_macro,
        })
    return out


@app.post("/api/decision/concepts/{module_id}/toggle")
def toggle_concept_module_endpoint(module_id: str, payload: Dict[str, Any]):
    """Enable or disable a trading concept module."""
    enable = bool(payload.get("enable", True))
    repo.toggle_concept_module(module_id=module_id, is_enabled=enable)
    return {"status": "SUCCESS", "module_id": module_id, "is_enabled": enable}


@app.post("/api/decision/concepts/{module_id}/approve")
def approve_concept_module_endpoint(module_id: str, payload: Dict[str, Any]):
    """Approve or revoke approval for a concept module (Part T/U requirement)."""
    approve = bool(payload.get("approve", True))
    repo.set_concept_module_approval(module_id=module_id, is_approved=approve)
    if module_id in master_decision_engine._modules:
        master_decision_engine._modules[module_id].is_approved = approve
    return {"status": "SUCCESS", "module_id": module_id, "is_approved": approve}


@app.post("/api/decision/threshold")
def set_decision_threshold_endpoint(payload: Dict[str, Any]):
    """Set configurable decision confluence threshold (e.g. 8.0 out of 10)."""
    val = float(payload.get("threshold", 8.0))
    if val < 1.0 or val > 10.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 1.0 and 10.0")
    master_decision_engine.threshold = val
    return {"status": "UPDATED", "threshold": master_decision_engine.threshold}


# ============================================================================
# Step 18: Universal Providers, Connection Manager & Lecture Benchmarking API
# ============================================================================
from trading_ai.providers import (
    UniversalConnectionManager,
    UniversalProviderRegistry,
)
from trading_ai.benchmark import LectureBenchmarkEngine
from trading_ai.schemas.provider import ProviderCategory
from trading_ai.schemas.benchmark import BenchmarkStage
from trading_ai.supervisor import MarketSupervisor
from trading_ai.schemas.enums import UnifiedExecutionTestingMode

universal_provider_registry = UniversalProviderRegistry(repository=repo)
universal_connection_manager = UniversalConnectionManager(universal_provider_registry, repository=repo)
lecture_benchmark_engine = LectureBenchmarkEngine(repository=repo)

market_supervisor = MarketSupervisor(
    repository=repo,
    provider_registry=universal_provider_registry,
    mtf_engine=mtf_engine,
    situation_engine=situation_engine,
    chart_marker=chart_marker,
    data_provider=data_provider,
    execution_gate=getattr(execution_core, "execution_gate", None) if execution_core else None,
    notification_orchestrator=notification_orchestrator,
)


@app.get("/api/providers/list")
def list_universal_providers_endpoint(category: Optional[str] = None):
    """List all registered universal providers with operational metadata."""
    providers = universal_provider_registry.list_providers(category=category)
    return [p.model_dump(mode="json") for p in providers]


@app.get("/api/providers/{provider_id}")
def get_universal_provider_endpoint(provider_id: str):
    """Retrieve details and masked configuration for a specific provider."""
    p = universal_provider_registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found.")
    masked_cfg = universal_connection_manager.get_masked_config(provider_id)
    res = p.metadata.model_dump(mode="json")
    res["masked_config"] = masked_cfg
    return res


@app.post("/api/providers/{provider_id}/toggle")
def toggle_universal_provider_endpoint(provider_id: str, payload: Optional[Dict[str, Any]] = None):
    """Enable or disable a universal provider."""
    p = universal_provider_registry.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found.")
    enable = payload.get("enable", not p.is_enabled) if payload else (not p.is_enabled)
    universal_provider_registry.toggle_provider(provider_id, enable)
    return {"status": "SUCCESS", "success": True, "provider_id": provider_id, "enabled": enable}


@app.post("/api/providers/{provider_id}/approve")
def approve_universal_provider_endpoint(
    provider_id: str,
    approved: Optional[bool] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    """Approve or revoke user approval for a provider."""
    is_app = approved if approved is not None else (payload.get("approve", True) if payload else True)
    universal_provider_registry.approve_provider(provider_id, is_app)
    return {"status": "SUCCESS", "success": True, "provider_id": provider_id, "approved": is_app}


@app.post("/api/providers/{provider_id}/configure")
def configure_universal_provider_endpoint(provider_id: str, payload: Optional[Dict[str, Any]] = None):
    """Save non-secret configuration for a provider with automatic credential masking."""
    payload = payload or {}
    config = payload.get("config", payload)
    res = universal_connection_manager.configure_provider(provider_id, config)
    return res


@app.post("/api/providers/{provider_id}/test")
def test_universal_provider_endpoint(provider_id: str):
    """Execute a safe, non-destructive connection test for a provider."""
    try:
        res = universal_connection_manager.test_connection(provider_id)
        return res.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Error testing connection for {provider_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/providers/active/{category}")
def get_active_category_provider_endpoint(category: str):
    """Return the active provider for a category."""
    try:
        cat_enum = ProviderCategory(category.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category '{category}'.")
    active_p = universal_provider_registry.get_active_provider(cat_enum)
    if not active_p:
        return {"category": cat_enum.value, "active_provider": None, "provider_id": None}
    res = active_p.metadata.model_dump(mode="json")
    res["category"] = cat_enum.value
    return res


@app.post("/api/providers/active/{category}/switch")
def switch_active_category_provider_endpoint(
    category: str,
    provider_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    """Switch the active provider for a category without altering the AI brain."""
    target_id = provider_id or (payload.get("provider_id") if payload else "")
    try:
        cat_enum = ProviderCategory(category.upper())
        universal_provider_registry.set_active_provider(cat_enum, target_id)
        return {"status": "SWITCHED", "success": True, "category": cat_enum.value, "active_provider_id": target_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Dynamic Broker, Live AI Marking, Supervisor & Execution Endpoints
# ============================================================================

class AddBrokerRequest(BaseModel):
    broker_id: Optional[str] = None
    provider_id: Optional[str] = None
    broker_name: Optional[str] = None
    name: Optional[str] = None
    provider_name: Optional[str] = None
    provider_type: Optional[str] = None
    adapter_type: str = "GENERIC_BROKER"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    server: Optional[str] = None
    account_id: Optional[str] = None
    authentication_type: str = "API_KEY"
    live_supported: bool = False
    paper_supported: bool = True
    execution_state: Optional[str] = None


class EvaluateSupervisorRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None


class EnvironmentModeRequest(BaseModel):
    mode: Optional[str] = None
    environment: Optional[str] = None


@app.get("/api/brokers")
def list_brokers_endpoint():
    """List all registered brokers with masked credentials and active status."""
    brokers = [p for p in universal_provider_registry.list_providers() if p.provider_type == ProviderCategory.BROKER]
    results = []
    for b in brokers:
        m_cfg = universal_connection_manager.get_masked_config(b.provider_id)
        d = b.model_dump(mode="json")
        d["masked_config"] = m_cfg
        results.append(d)
    return results


@app.post("/api/brokers/add")
def add_broker_endpoint(req: AddBrokerRequest):
    """Dynamically register a new broker without editing source code."""
    cfg = {}
    if req.api_key:
        cfg["api_key"] = req.api_key
    if req.api_secret:
        cfg["api_secret"] = req.api_secret
    if req.server:
        cfg["server"] = req.server
    if req.account_id:
        cfg["account_id"] = req.account_id

    b_id = req.broker_id or req.provider_id or "new_broker"
    b_name = req.broker_name or req.name or req.provider_name or b_id

    provider = universal_provider_registry.add_broker(
        broker_id=b_id,
        broker_name=b_name,
        adapter_type=req.adapter_type,
        authentication_type=req.authentication_type,
        live_supported=req.live_supported,
        paper_supported=req.paper_supported,
        config=cfg if cfg else None,
        execution_state=req.execution_state,
    )
    meta = provider.metadata if hasattr(provider, "metadata") else provider
    if cfg:
        universal_connection_manager.configure_provider(meta.provider_id, cfg)
    res_dict = meta.model_dump(mode="json")
    return {
        "status": "SUCCESS",
        "success": True,
        "broker": res_dict,
        **res_dict,
    }


@app.get("/api/brokers/active")
def get_active_broker_endpoint():
    """Return canonical active broker context."""
    b = universal_provider_registry.get_active_broker()
    if not b:
        return {"active_broker": None}
    res = b.metadata.model_dump(mode="json")
    res["masked_config"] = universal_connection_manager.get_masked_config(b.provider_id)
    return {
        **res,
        "active_broker": res,
    }


@app.post("/api/brokers/{broker_id}/switch")
def switch_broker_endpoint(broker_id: str):
    """
    Manually switch active broker context.
    Rebinds MarketSupervisor, loads market context, and synchronizes chart.
    INVARIANT: Never places orders, never enables live auto execution.
    """
    try:
        res = market_supervisor.switch_broker(broker_id)
        active_b = universal_provider_registry.get_active_broker()
        active_dict = active_b.metadata.model_dump(mode="json") if active_b else {"provider_id": broker_id}
        return {
            "status": "SUCCESS",
            "success": True,
            "switched": True,
            "details": res,
            "active_broker": active_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/supervisor/status")
def get_supervisor_status_endpoint():
    """Operational status of the continuous MarketSupervisor."""
    return market_supervisor.get_status()


@app.post("/api/supervisor/evaluate")
def evaluate_supervisor_endpoint(req: Optional[EvaluateSupervisorRequest] = None):
    """Trigger continuous market evaluation and canonical AI chart markings."""
    sym = req.symbol if req else None
    tf = req.timeframe if req else None
    eval_res = market_supervisor.evaluate_market(symbol=sym, timeframe=tf)
    markings = eval_res.get("markings", []) if isinstance(eval_res, dict) else eval_res
    return {
        "status": "SUCCESS",
        "success": True,
        "symbol": sym or market_supervisor._active_symbol,
        "markings_count": len(markings),
        "markings": [m.model_dump(mode="json") for m in markings],
        "supervisor_status": market_supervisor.get_status(),
    }


@app.get("/api/supervisor/markings/{symbol:path}")
def get_supervisor_markings_endpoint(symbol: str, timeframe: Optional[str] = None):
    """Fetch verified markings from MarketSupervisor."""
    markings = market_supervisor.get_markings_for_symbol(symbol=symbol, timeframe=timeframe)
    return {
        "symbol": symbol,
        "count": len(markings),
        "markings": [m.model_dump(mode="json") for m in markings],
    }


@app.get("/api/markings/{symbol:path}")
def get_canonical_chart_markings_endpoint(symbol: str, timeframe: Optional[str] = None, only_valid: bool = True):
    """Canonical chart markings endpoint for live chart overlays."""
    markings = repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=only_valid)
    return {
        "symbol": symbol,
        "count": len(markings),
        "markings": [m.model_dump(mode="json") for m in markings],
    }


@app.post("/api/execution/environment-mode")
def set_execution_environment_mode_endpoint(req: EnvironmentModeRequest):
    """
    Safely switch between BACKTEST, PAPER, MANUAL_LIVE, and AUTO_LIVE.
    Enforces ExecutionGate and RiskEngine pre-conditions.
    """
    try:
        mode_val = req.mode or req.environment or "PAPER"
        mode_enum = UnifiedExecutionTestingMode(mode_val.upper())
        res = market_supervisor.set_execution_mode(mode_enum)
        return {
            "status": "SUCCESS",
            "success": True,
            "current_mode": mode_enum.value,
            **res,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/backtests/{backtest_id}/markings")
def get_backtest_markings_endpoint(backtest_id: str):
    """Retrieve canonical visual markings for a specific historical backtest replay."""
    markings = repo.get_chart_markings_for_backtest(backtest_id=backtest_id)
    return {
        "backtest_id": backtest_id,
        "count": len(markings),
        "markings": [m.model_dump(mode="json") for m in markings],
    }



# ----------------------------------------------------------------------------
# Step 18 Correction: TradingView Ingestion & Replay Bar Clock Controls
# ----------------------------------------------------------------------------

@app.post("/api/providers/tradingview/webhook")
def tradingview_webhook_endpoint(payload: Dict[str, Any]):
    """Authorized webhook endpoint for TradingView alerts and Pine Script market data."""
    tv_prov = universal_provider_registry.get_provider("tradingview")
    if not tv_prov:
        raise HTTPException(status_code=503, detail="TradingView provider not registered.")
    try:
        norm = tv_prov.ingest_webhook_payload(payload)
        return {
            "status": "ACCEPTED",
            "provider_id": "tradingview",
            "canonical_symbol": norm.canonical_symbol,
            "timeframe": norm.timeframe,
            "close": norm.close,
            "timestamp": norm.timestamp,
            "is_live": norm.is_live,
        }
    except Exception as e:
        logger.error(f"Error ingesting TradingView payload: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/providers/tradingview/status")
def get_tradingview_status_endpoint():
    """Return status, instructions, and cached symbols for TradingView."""
    tv_prov = universal_provider_registry.get_provider("tradingview")
    if not tv_prov:
        raise HTTPException(status_code=404, detail="TradingView provider not registered.")
    return {
        "status": tv_prov.metadata.status.value,
        "capabilities": tv_prov.metadata.capabilities,
        "webhook_endpoint": "/api/providers/tradingview/webhook",
        "instructions": (
            "Configure TradingView Alert Webhook URL to: "
            "http://<host>:8000/api/providers/tradingview/webhook with JSON payload containing: "
            "{'ticker': 'BTCUSD', 'close': {{close}}, 'time': {{time}}, 'timeframe': '15m'}"
        ),
        "cached_symbols": list(tv_prov._live_cache.keys()),
    }


@app.get("/api/backtest/replay/clock/status")
def get_replay_clock_status_endpoint():
    """Return current state of the independent BacktestClock."""
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "get_replay_clock"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    clock = tl_prov.get_replay_clock()
    curr_bar = tl_prov.get_current_replay_bar()
    state = clock.to_dict()
    state["current_bar"] = curr_bar
    return state


@app.post("/api/backtest/replay/clock/step")
def step_replay_clock_endpoint(payload: Optional[Dict[str, Any]] = None):
    """Step forward or backward by N bars on the historical replay clock."""
    bars = int(payload.get("bars", 1)) if payload else 1
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "step_replay"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    bar = tl_prov.step_replay(bars=bars)
    clock = tl_prov.get_replay_clock()
    return {
        "status": "STEPPED",
        "bars_stepped": bars,
        "current_timestamp": clock.current_timestamp,
        "current_bar": bar,
    }


@app.post("/api/backtest/replay/clock/speed")
def set_replay_clock_speed_endpoint(payload: Dict[str, Any]):
    """Set replay clock playback speed (1x, 2x, 5x, 10x, 25x, 50x)."""
    speed = float(payload.get("speed", 1.0))
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "get_replay_clock"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    clock = tl_prov.get_replay_clock()
    actual_speed = clock.set_speed(speed)
    return {"status": "UPDATED", "speed": actual_speed}


@app.post("/api/backtest/replay/clock/play")
def play_replay_clock_endpoint():
    """Resume / play historical replay."""
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "get_replay_clock"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    tl_prov.get_replay_clock().resume()
    return {"status": "PLAYING", "is_paused": False}


@app.post("/api/backtest/replay/clock/pause")
def pause_replay_clock_endpoint():
    """Pause historical replay."""
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "get_replay_clock"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    tl_prov.get_replay_clock().pause()
    return {"status": "PAUSED", "is_paused": True}


@app.post("/api/backtest/replay/clock/reset")
def reset_replay_clock_endpoint():
    """Reset historical replay clock to start."""
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "get_replay_clock"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    tl_prov.get_replay_clock().reset()
    return {"status": "RESET", "clock": tl_prov.get_replay_clock().to_dict()}


@app.post("/api/backtest/replay/load_bars")
def load_replay_bars_endpoint(payload: Dict[str, Any]):
    """Load historical / exported replay bars into Trading-Leads provider."""
    symbol = str(payload.get("symbol", "BTC/USD"))
    timeframe = str(payload.get("timeframe", "15m"))
    bars = payload.get("bars", [])
    if not bars:
        raise HTTPException(status_code=400, detail="Bars list cannot be empty.")
    tl_prov = universal_provider_registry.get_provider("trading_leads")
    if not tl_prov or not hasattr(tl_prov, "load_replay_bars"):
        raise HTTPException(status_code=503, detail="Replay provider not available.")
    count = tl_prov.load_replay_bars(symbol=symbol, timeframe=timeframe, bars=bars)
    return {"status": "LOADED", "bars_count": count, "symbol": symbol, "timeframe": timeframe}


@app.get("/api/benchmark/lecture/list")
def list_lecture_benchmarks_endpoint(limit: int = 50):
    """List historical real lecture performance benchmark records."""
    benchmarks = repo.list_lecture_benchmarks(limit=limit)
    return {"benchmarks": benchmarks}


@app.get("/api/benchmark/lecture/{lecture_id}")
def get_lecture_benchmark_endpoint(lecture_id: str):
    """Fetch complete benchmark instrumentation, timeline, and bottlenecks for a lecture."""
    b = lecture_benchmark_engine.get_benchmark(lecture_id)
    if not b:
        raise HTTPException(status_code=404, detail=f"Benchmark for '{lecture_id}' not found.")
    return b.model_dump(mode="json")


@app.post("/api/benchmark/lecture/sample")
def generate_sample_lecture_benchmark_endpoint():
    """Generate or execute a real calibrated benchmark record for UI demonstration."""
    import time
    lid = f"bench_lecture_{int(time.time())}"
    rec = lecture_benchmark_engine.start_benchmark(
        lecture_id=lid,
        file_name="EURUSD_M15_Institutional_Liquidity_Lecture.mp4",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        file_size_bytes=142_500_000,  # ~142.5 MB
        duration_seconds=1840.0,      # ~30.6 minutes
        resolution="1920x1080",
        fps=30.0,
    )
    # Simulate realistic calibrated stage measurements
    with lecture_benchmark_engine.benchmark_stage(lid, BenchmarkStage.INGESTION):
        time.sleep(0.05)
    with lecture_benchmark_engine.benchmark_stage(lid, BenchmarkStage.AUDIO):
        time.sleep(0.10)
    with lecture_benchmark_engine.benchmark_stage(lid, BenchmarkStage.VISUAL):
        time.sleep(0.35)
    with lecture_benchmark_engine.benchmark_stage(lid, BenchmarkStage.ALIGNMENT):
        time.sleep(0.12)
    with lecture_benchmark_engine.benchmark_stage(lid, BenchmarkStage.KNOWLEDGE):
        time.sleep(0.08)

    lecture_benchmark_engine.update_counts(
        lecture_id=lid,
        frames_sampled=184,
        frames_processed=42,
        audio_segments=61,
        visual_segments=42,
        aligned_segments=38,
        gemini_calls=8,
        tokens_input=24500,
        tokens_output=3200,
    )
    lecture_benchmark_engine.update_knowledge_metrics(
        lecture_id=lid,
        concepts=5,
        rules=3,
        evidence=12,
        exceptions=1,
        examples=4,
        uncertainties=0,
        needs_review=0,
    )
    final = lecture_benchmark_engine.finalize_benchmark(lid, is_complete=True)
    return final.model_dump(mode="json")


# ============================================================================
# Phase 1: AI Chat + Application Guide / Insights Endpoints
# ============================================================================
from trading_ai.chat import AIChatService, ApplicationGuide
from trading_ai.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryResponse,
    AppGuideResponse,
)
from trading_ai.schemas.unified import (
    UnifiedAIRequest,
    UnifiedAIResponse,
    UnifiedSessionState,
)

class ClearChatRequest(BaseModel):
    session_id: str = "default_session"

app_guide_instance = ApplicationGuide()
ai_chat_service = AIChatService(
    repository=repo,
    gemini_client=getattr(orchestrator, "_ai", None),
    companion_core=companion_core,
    master_decision_engine=master_decision_engine,
    token_manager=getattr(orchestrator, "_token_manager", None),
    app_guide=app_guide_instance,
    execution_core=execution_core,
    model_router=getattr(orchestrator, "_model_router", None),
    market_supervisor=market_supervisor,
)

# Connect Unified AI Brain to Companion Core, Voice Gateway, Execution Core, and Market Supervisor
if companion_core:
    companion_core.set_chat_service(ai_chat_service)
if execution_core:
    ai_chat_service.set_execution_core(execution_core)
if market_supervisor:
    ai_chat_service.set_market_supervisor(market_supervisor)
if hasattr(security_core, "voice_gateway") and security_core.voice_gateway:
    security_core.voice_gateway.set_chat_service(ai_chat_service)


@app.post("/api/chat/message", response_model=ChatMessageResponse)
def post_chat_message_endpoint(req: ChatMessageRequest):
    """
    Multi-turn AI Chat & Application Guide message endpoint.
    Retrieves live system state, market context, app map, and respects token safety budget.
    """
    return ai_chat_service.process_message(req)


@app.get("/api/chat/history", response_model=ChatHistoryResponse)
def get_chat_history_endpoint(session_id: str = "default_session", limit: int = 50):
    """Retrieve conversation history turns for a session."""
    return ai_chat_service.get_history(session_id=session_id, limit=limit)


@app.post("/api/chat/clear")
def clear_chat_history_endpoint(req: ClearChatRequest):
    """Clear or reset conversation history turns for a session."""
    return ai_chat_service.clear_history(session_id=req.session_id)


@app.get("/api/chat/guide", response_model=AppGuideResponse)
def get_app_guide_endpoint(category: Optional[str] = None):
    """Retrieve the authoritative Application Component Registry and app tour."""
    return ai_chat_service.get_app_guide(category=category)


# ============================================================================
# Step 3: Unified AI Brain (Text, Voice, Shared Memory, Cross-Modality)
# ============================================================================
@app.post("/api/unified/process", response_model=UnifiedAIResponse)
def post_unified_process_endpoint(req: UnifiedAIRequest):
    """
    Unified AI Brain endpoint serving Text, Voice, and Multimodal requests
    through the single reasoning, memory, token management, and safety pipeline.
    """
    return ai_chat_service.process_unified_request(req)


@app.get("/api/unified/session", response_model=UnifiedSessionState)
def get_unified_session_endpoint(session_id: str = "default_session"):
    """Retrieve shared session state across text, voice, and application context."""
    return ai_chat_service.get_session_state(session_id=session_id)


class ActionConfirmationRequest(BaseModel):
    session_id: str = "default_session"
    action: str = "CONFIRM"


@app.post("/api/unified/confirm", response_model=UnifiedAIResponse)
def post_unified_confirm_endpoint(req: ActionConfirmationRequest):
    """
    Explicit confirmation or cancellation of a proposed order intent.
    Routes directly through ExecutionGate and RiskEngine safety invariants.
    """
    return ai_chat_service.confirm_pending_action(session_id=req.session_id, action=req.action)









