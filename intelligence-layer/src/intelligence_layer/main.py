"""Main FastAPI application for intelligence layer with LLM and RAG capabilities."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import timedelta, datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from .config import load_config, Config
from .llm_config import load_llm_config, LLMIntelligenceConfig
from .logging import configure_logging, get_logger
from .models import (
    IntelligenceState,
    MarketWindowFeatures,
    EmbeddingResponse,
    RegimeResponse,
    GraphFeaturesResponse,
    RLStateResponse,
    MarketData,
)
from .regime_detection import RegimeInferencePipeline
from .graph_analytics import MarketGraphAnalytics
from .state_assembly import CompositeStateAssembler
from .data_import import data_importer, DataSource
from .market_analytics import market_analytics
from .model_registry import model_registry, ModelCategory, UseCase

# New LLM and RAG services
from .llm_service import LLMService
from .rag_service import RAGService
from .research_service import FinancialResearchService

# Derivatives and market data API
from .derivatives_api import router as derivatives_router

# MLflow integration API
from .mlflow_routes import router as mlflow_router

# Global configuration and services
config = load_config()
llm_config = load_llm_config()
logger = get_logger(__name__)

# Global service instances
regime_pipeline: RegimeInferencePipeline = None
graph_analytics: MarketGraphAnalytics = None
state_assembler: CompositeStateAssembler = None

# New LLM and RAG service instances
llm_service: LLMService = None
rag_service: RAGService = None
research_service: FinancialResearchService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global regime_pipeline, graph_analytics, state_assembler
    global llm_service, rag_service, research_service
    
    # Startup
    configure_logging(config.logging)
    logger.info("Intelligence Layer API starting up", config=config.model_dump())
    
    # Initialize LLM and RAG services
    try:
        llm_service = LLMService(llm_config)
        await llm_service.initialize()
        logger.info("LLM Service initialized")
        
        rag_service = RAGService(llm_config)
        await rag_service.initialize()
        logger.info("RAG Service initialized")
        
        research_service = FinancialResearchService(llm_config)
        await research_service.initialize()
        logger.info("Research Service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM/RAG services: {e}")
        # Continue without LLM services if they fail to initialize
    
    try:
        # Initialize services
        regime_pipeline = RegimeInferencePipeline(config)
        graph_analytics = MarketGraphAnalytics(config)
        state_assembler = CompositeStateAssembler(config)
        
        # Set up graph projections
        graph_analytics.setup_projections()
        
        logger.info("Intelligence services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize intelligence services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Intelligence Layer API shutting down")
    try:
        if graph_analytics:
            graph_analytics.close()
        if state_assembler:
            await state_assembler.close()
        logger.info("Intelligence services closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="Algorithmic Trading Intelligence API",
    description="Intelligence layer providing market analysis, regime detection, and ML services",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default port
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include derivatives and market data API router
app.include_router(derivatives_router, prefix="/api/v1")

# Include MLflow integration API router
app.include_router(mlflow_router, prefix="/api/v1")


def get_config() -> Config:
    """Dependency to get configuration."""
    return config


def get_regime_pipeline() -> RegimeInferencePipeline:
    """Dependency to get regime pipeline."""
    if regime_pipeline is None:
        raise HTTPException(status_code=503, detail="Regime pipeline not initialized")
    return regime_pipeline


def get_graph_analytics() -> MarketGraphAnalytics:
    """Dependency to get graph analytics."""
    if graph_analytics is None:
        raise HTTPException(status_code=503, detail="Graph analytics not initialized")
    return graph_analytics


def get_state_assembler() -> CompositeStateAssembler:
    """Dependency to get state assembler."""
    if state_assembler is None:
        raise HTTPException(status_code=503, detail="State assembler not initialized")
    return state_assembler


def get_llm_service() -> LLMService:
    """Dependency to get LLM service."""
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return llm_service


def get_rag_service() -> RAGService:
    """Dependency to get RAG service."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return rag_service


def get_research_service() -> FinancialResearchService:
    """Dependency to get research service."""
    if research_service is None:
        raise HTTPException(status_code=503, detail="Research service not initialized")
    return research_service


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "intelligence-api"}


# ============================================================================
# EMERGENCY CONTROLS & QUICK ACTIONS API ENDPOINTS
# ============================================================================

# Global system state for emergency controls
class SystemState:
    def __init__(self):
        self.trading_status: str = "ACTIVE"  # ACTIVE, PAUSED, HALTED
        self.emergency_halt_active: bool = False
        self.last_status_change: datetime = datetime.now(timezone.utc)
        self.halt_reason: Optional[str] = None

system_state = SystemState()

# Request/Response models for emergency controls

class EmergencyHaltRequest(BaseModel):
    reason: Optional[str] = "Manual emergency halt"
    force: bool = False

class EmergencyHaltResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime
    previous_status: str
    new_status: str

class TradingControlRequest(BaseModel):
    action: str  # "pause" or "resume"
    reason: Optional[str] = None

class TradingControlResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime
    previous_status: str
    new_status: str

class SystemStatusResponse(BaseModel):
    trading_status: str
    emergency_halt_active: bool
    last_status_change: datetime
    halt_reason: Optional[str]
    uptime_seconds: float

class QuickChartRequest(BaseModel):
    symbol: str
    timeframe: str = "1H"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class QuickChartResponse(BaseModel):
    symbol: str
    timeframe: str
    data_points: int
    chart_url: Optional[str] = None
    message: str

class SymbolSearchRequest(BaseModel):
    query: str
    limit: int = 10

class SymbolSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int

class WatchlistItem(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: Optional[int] = None

class WatchlistResponse(BaseModel):
    items: List[WatchlistItem]
    last_updated: datetime

# Emergency Controls Endpoints
@app.post("/emergency/halt", response_model=EmergencyHaltResponse)
async def emergency_halt(request: EmergencyHaltRequest):
    """Emergency halt - immediately stop all trading activities."""
    try:
        previous_status = system_state.trading_status
        
        if system_state.emergency_halt_active and not request.force:
            raise HTTPException(
                status_code=400, 
                detail="Emergency halt already active. Use force=true to override."
            )
        
        # Set emergency halt
        system_state.trading_status = "HALTED"
        system_state.emergency_halt_active = True
        system_state.last_status_change = datetime.now(timezone.utc)
        system_state.halt_reason = request.reason
        
        logger.critical(f"EMERGENCY HALT ACTIVATED: {request.reason}")
        
        # TODO: Send halt signal to execution core
        # await execution_client.emergency_halt()
        
        return EmergencyHaltResponse(
            success=True,
            message=f"Emergency halt activated: {request.reason}",
            timestamp=system_state.last_status_change,
            previous_status=previous_status,
            new_status=system_state.trading_status
        )
        
    except Exception as e:
        logger.error(f"Emergency halt failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emergency halt failed: {str(e)}")

@app.post("/emergency/resume", response_model=EmergencyHaltResponse)
async def resume_from_halt():
    """Resume trading after emergency halt (requires manual intervention)."""
    try:
        if not system_state.emergency_halt_active:
            raise HTTPException(status_code=400, detail="No emergency halt is active")
        
        previous_status = system_state.trading_status
        
        # Resume trading
        system_state.trading_status = "ACTIVE"
        system_state.emergency_halt_active = False
        system_state.last_status_change = datetime.now(timezone.utc)
        system_state.halt_reason = None
        
        logger.warning("Emergency halt DEACTIVATED - Trading resumed")
        
        return EmergencyHaltResponse(
            success=True,
            message="Emergency halt deactivated - Trading resumed",
            timestamp=system_state.last_status_change,
            previous_status=previous_status,
            new_status=system_state.trading_status
        )
        
    except Exception as e:
        logger.error(f"Resume from halt failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resume failed: {str(e)}")

@app.post("/trading/control", response_model=TradingControlResponse)
async def trading_control(request: TradingControlRequest):
    """Pause or resume trading (non-emergency)."""
    try:
        if system_state.emergency_halt_active:
            raise HTTPException(
                status_code=400, 
                detail="Cannot control trading while emergency halt is active"
            )
        
        previous_status = system_state.trading_status
        
        if request.action == "pause":
            if system_state.trading_status == "PAUSED":
                raise HTTPException(status_code=400, detail="Trading is already paused")
            
            system_state.trading_status = "PAUSED"
            system_state.last_status_change = datetime.now(timezone.utc)
            message = f"Trading paused: {request.reason or 'Manual pause'}"
            logger.warning(message)
            
        elif request.action == "resume":
            if system_state.trading_status == "ACTIVE":
                raise HTTPException(status_code=400, detail="Trading is already active")
            
            system_state.trading_status = "ACTIVE"
            system_state.last_status_change = datetime.now(timezone.utc)
            message = f"Trading resumed: {request.reason or 'Manual resume'}"
            logger.info(message)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'pause' or 'resume'")
        
        return TradingControlResponse(
            success=True,
            message=message,
            timestamp=system_state.last_status_change,
            previous_status=previous_status,
            new_status=system_state.trading_status
        )
        
    except Exception as e:
        logger.error(f"Trading control failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trading control failed: {str(e)}")

@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system and trading status."""
    uptime = (datetime.now(timezone.utc) - system_state.last_status_change).total_seconds()
    
    return SystemStatusResponse(
        trading_status=system_state.trading_status,
        emergency_halt_active=system_state.emergency_halt_active,
        last_status_change=system_state.last_status_change,
        halt_reason=system_state.halt_reason,
        uptime_seconds=uptime
    )

# Quick Actions Endpoints
@app.post("/quick/chart", response_model=QuickChartResponse)
async def quick_chart(request: QuickChartRequest):
    """Generate quick chart data for a symbol."""
    try:
        # TODO: Implement actual chart data generation
        # For now, return mock response
        
        logger.info(f"Quick chart requested for {request.symbol} ({request.timeframe})")
        
        return QuickChartResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            data_points=100,  # Mock data points
            chart_url=None,  # TODO: Generate chart URL
            message=f"Chart data prepared for {request.symbol}"
        )
        
    except Exception as e:
        logger.error(f"Quick chart failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@app.post("/quick/symbol-search", response_model=SymbolSearchResponse)
async def symbol_search(request: SymbolSearchRequest):
    """Search for trading symbols."""
    try:
        # Mock symbol search results
        mock_results = [
            {"symbol": "EURUSD", "name": "Euro / US Dollar", "type": "Forex", "exchange": "FX"},
            {"symbol": "GBPUSD", "name": "British Pound / US Dollar", "type": "Forex", "exchange": "FX"},
            {"symbol": "USDJPY", "name": "US Dollar / Japanese Yen", "type": "Forex", "exchange": "FX"},
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "Stock", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "Stock", "exchange": "NASDAQ"},
            {"symbol": "BTC-USD", "name": "Bitcoin USD", "type": "Crypto", "exchange": "CRYPTO"},
            {"symbol": "ETH-USD", "name": "Ethereum USD", "type": "Crypto", "exchange": "CRYPTO"},
        ]
        
        # Filter results based on query
        filtered_results = [
            result for result in mock_results
            if request.query.lower() in result["symbol"].lower() or 
               request.query.lower() in result["name"].lower()
        ][:request.limit]
        
        logger.info(f"Symbol search for '{request.query}' returned {len(filtered_results)} results")
        
        return SymbolSearchResponse(
            query=request.query,
            results=filtered_results,
            total_found=len(filtered_results)
        )
        
    except Exception as e:
        logger.error(f"Symbol search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Symbol search failed: {str(e)}")

@app.get("/quick/watchlist", response_model=WatchlistResponse)
async def get_watchlist():
    """Get current watchlist with live prices."""
    try:
        # Mock watchlist data
        mock_watchlist = [
            WatchlistItem(symbol="EURUSD", price=1.0845, change=0.0012, change_percent=0.11),
            WatchlistItem(symbol="GBPUSD", price=1.2634, change=-0.0023, change_percent=-0.18),
            WatchlistItem(symbol="USDJPY", price=149.85, change=0.45, change_percent=0.30),
            WatchlistItem(symbol="AUDUSD", price=0.6523, change=0.0008, change_percent=0.12),
            WatchlistItem(symbol="USDCHF", price=0.8756, change=-0.0015, change_percent=-0.17),
        ]
        
        return WatchlistResponse(
            items=mock_watchlist,
            last_updated=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Watchlist fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Watchlist fetch failed: {str(e)}")

@app.post("/quick/reconnect")
async def force_reconnect():
    """Force reconnection to all services."""
    try:
        logger.info("Force reconnect requested")
        
        # TODO: Implement actual reconnection logic
        # - Reconnect to databases
        # - Reconnect to execution core
        # - Refresh all connections
        
        return {
            "success": True,
            "message": "Reconnection initiated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services_reconnected": ["database", "execution_core", "market_data"]
        }
        
    except Exception as e:
        logger.error(f"Force reconnect failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Force reconnect failed: {str(e)}")

# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Algorithmic Trading Intelligence API",
        "version": "0.1.0",
        "status": "running",
    }


@app.post("/intelligence/embedding", response_model=EmbeddingResponse)
async def infer_embedding(
    features: MarketWindowFeatures,
    config: Config = Depends(get_config),
) -> EmbeddingResponse:
    """
    Infer market state embedding from window features.
    
    This endpoint processes market window features and returns:
    - Embedding ID for storage/retrieval
    - Similarity context with historical embeddings
    - Confidence score for the embedding quality
    """
    try:
        # TODO: Implement embedding inference
        # 1. Validate input features
        # 2. Run through embedding model (TCN/VAE)
        # 3. Store embedding in pgvector
        # 4. Find similar historical embeddings
        # 5. Calculate confidence score
        
        # Placeholder response
        from uuid import uuid4
        from datetime import datetime, timezone
        
        return EmbeddingResponse(
            embedding_id=uuid4(),
            similarity_context=[],
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
        )
        
    except Exception as e:
        logger.error("Embedding inference failed", error=str(e), features=features.model_dump())
        raise HTTPException(status_code=500, detail=f"Embedding inference failed: {str(e)}")


@app.get("/intelligence/regime", response_model=RegimeResponse)
async def get_regime_inference(
    asset_id: str,
) -> RegimeResponse:
    """
    Get current market regime inference for an asset (simplified endpoint for testing).
    
    Returns deterministic regime probabilities for testing purposes.
    """
    try:
        from datetime import datetime, timezone
        
        # For testing purposes, return deterministic regime probabilities
        # based on the asset_id to ensure statefulness
        asset_hash = hash(asset_id) % 100
        
        # Create deterministic regime probabilities
        regime_probs = {
            "low_vol_trending": 0.3 + (asset_hash % 10) * 0.01,
            "high_vol_ranging": 0.4 + (asset_hash % 15) * 0.01,
            "crisis": 0.1 + (asset_hash % 5) * 0.01,
        }
        
        # Normalize probabilities
        total = sum(regime_probs.values())
        regime_probs = {k: v/total for k, v in regime_probs.items()}
        
        # Create deterministic transition likelihoods
        transition_probs = {
            "low_vol_trending": 0.2 + (asset_hash % 8) * 0.01,
            "high_vol_ranging": 0.3 + (asset_hash % 12) * 0.01,
            "crisis": 0.05 + (asset_hash % 3) * 0.01,
        }
        
        # Calculate entropy deterministically
        entropy = -sum(p * np.log2(p) for p in regime_probs.values() if p > 0)
        
        return RegimeResponse(
            regime_probabilities=regime_probs,
            transition_likelihoods=transition_probs,
            regime_entropy=entropy,
            confidence=0.8 + (asset_hash % 20) * 0.01,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),  # Fixed timestamp for testing
        )
        
    except Exception as e:
        logger.error("Regime inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Regime inference failed: {str(e)}")


@app.post("/intelligence/regime", response_model=RegimeResponse)
async def infer_regime(
    market_data: List[MarketData],
    pipeline: RegimeInferencePipeline = Depends(get_regime_pipeline),
) -> RegimeResponse:
    """
    Infer current market regime and transition probabilities.
    
    Returns:
    - Regime probabilities for all known regimes
    - Transition likelihoods to other regimes
    - Regime entropy (uncertainty measure)
    - Overall confidence in regime classification
    """
    try:
        if not pipeline.is_trained:
            raise HTTPException(
                status_code=400, 
                detail="Regime pipeline not trained. Train with historical data first."
            )
        
        if not market_data:
            raise HTTPException(status_code=400, detail="Market data is required")
        
        # Infer regime using the pipeline
        regime_response = pipeline.infer_regime(market_data)
        
        logger.info(f"Regime inference completed: {regime_response.regime_probabilities}")
        return regime_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regime inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Regime inference failed: {str(e)}")


@app.post("/intelligence/regime/train")
async def train_regime_model(
    historical_data: List[MarketData],
    pipeline: RegimeInferencePipeline = Depends(get_regime_pipeline),
) -> Dict[str, Any]:
    """
    Train the regime detection model with historical data.
    
    Args:
        historical_data: Historical market data for training
        
    Returns:
        Training status and regime definitions
    """
    try:
        if not historical_data:
            raise HTTPException(status_code=400, detail="Historical data is required")
        
        if len(historical_data) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data: need at least 100 data points"
            )
        
        # Train the pipeline
        pipeline.train(historical_data)
        
        # Get regime definitions
        regime_definitions = pipeline.get_regime_definitions()
        
        logger.info(f"Regime model trained with {len(historical_data)} data points")
        
        return {
            "status": "trained",
            "data_points": len(historical_data),
            "regime_definitions": regime_definitions,
            "timestamp": historical_data[-1].timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regime training failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Regime training failed: {str(e)}")


@app.get("/intelligence/graph-features", response_model=GraphFeaturesResponse)
async def get_graph_features(
    asset_id: str,
) -> GraphFeaturesResponse:
    """
    Get graph structural features for an asset.
    
    Returns deterministic graph features for testing purposes.
    """
    try:
        from datetime import datetime, timezone
        
        # For testing purposes, return deterministic graph features
        # based on the asset_id to ensure statefulness
        asset_hash = hash(asset_id) % 100
        
        return GraphFeaturesResponse(
            cluster_membership=f"cluster_{asset_hash % 5}",
            centrality_metrics={
                "centrality_score": 0.1 + (asset_hash % 50) * 0.01,
                "degree_centrality": 0.2 + (asset_hash % 30) * 0.01,
                "betweenness_centrality": 0.05 + (asset_hash % 20) * 0.01,
            },
            systemic_risk_proxies={
                "systemic_risk_proxy": 0.3 + (asset_hash % 40) * 0.01,
                "contagion_risk": 0.15 + (asset_hash % 25) * 0.01,
            },
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),  # Fixed timestamp for testing
        )
        
    except Exception as e:
        logger.error("Graph features extraction failed", error=str(e), asset_id=asset_id)
        raise HTTPException(status_code=500, detail=f"Graph features extraction failed: {str(e)}")


@app.post("/intelligence/graph/analyze")
async def run_graph_analysis(
    analysis_type: str = "asset_correlations",
    analytics: MarketGraphAnalytics = Depends(get_graph_analytics),
) -> Dict[str, Any]:
    """
    Run graph analysis algorithms.
    
    Args:
        analysis_type: Type of analysis ("asset_correlations", "regime_transitions")
        
    Returns:
        Analysis results and execution metadata
    """
    try:
        if analysis_type == "asset_correlations":
            results = analytics.analyze_asset_correlations()
        elif analysis_type == "regime_transitions":
            results = analytics.analyze_regime_transitions()
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown analysis type: {analysis_type}"
            )
        
        logger.info(f"Graph analysis '{analysis_type}' completed")
        
        return {
            "analysis_type": analysis_type,
            "status": "completed",
            "timestamp": results["timestamp"].isoformat(),
            "algorithms_executed": list(results.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Graph analysis failed", error=str(e), analysis_type=analysis_type)
        raise HTTPException(status_code=500, detail=f"Graph analysis failed: {str(e)}")


@app.get("/intelligence/state", response_model=RLStateResponse)
async def assemble_rl_state(
    asset_ids: str,  # Comma-separated list of asset IDs
    strategy_ids: str = "",  # Comma-separated list
) -> RLStateResponse:
    """
    Assemble composite RL state for strategy orchestration.
    
    Combines:
    - Market embeddings from pgvector
    - Regime labels and transitions from Neo4j
    - Graph structural features from Neo4j GDS
    - Portfolio and risk state from execution core
    - Confidence and uncertainty measures
    """
    try:
        strategy_list = [s.strip() for s in strategy_ids.split(",") if s.strip()]
        asset_list = [s.strip() for s in asset_ids.split(",") if s.strip()]
        primary_asset = asset_list[0] if asset_list else "EURUSD"  # Use first asset or default
        
        # For testing purposes, create deterministic market data
        # based on the primary asset to ensure statefulness
        from datetime import datetime, timedelta, timezone
        asset_hash = hash(primary_asset) % 1000
        
        # Generate deterministic market data
        recent_data = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Fixed base time for testing
        base_price = 1.1000 + (asset_hash % 100) * 0.0001  # Deterministic base price
        
        for i in range(100):  # 100 data points
            timestamp = base_time + timedelta(minutes=i * 15)  # 15-minute intervals
            
            # Deterministic price changes based on asset_id and index
            price_seed = (asset_hash + i) % 1000
            price_change = (price_seed - 500) * 0.0000001  # Small deterministic change
            base_price += price_change
            
            high_offset = (price_seed % 50) * 0.0000001
            low_offset = (price_seed % 30) * 0.0000001
            volume = 1000 + (price_seed % 9000)  # Deterministic volume
            
            recent_data.append(MarketData(
                timestamp=timestamp,
                asset_id=primary_asset,
                open=base_price,
                high=base_price + high_offset,
                low=base_price - low_offset,
                close=base_price,
                volume=volume
            ))
        
        # Create deterministic RL state response for testing
        from uuid import uuid4
        
        # Create deterministic intelligence state
        intelligence_state = IntelligenceState(
            embedding_similarity_context=[],
            current_regime_label=f"regime_{asset_hash % 3}",
            regime_transition_probabilities={
                "regime_0": 0.3 + (asset_hash % 10) * 0.01,
                "regime_1": 0.4 + (asset_hash % 15) * 0.01,
                "regime_2": 0.3 + (asset_hash % 12) * 0.01,
            },
            regime_confidence=0.8 + (asset_hash % 20) * 0.01,
            confidence_scores={
                "overall": 0.75 + (asset_hash % 25) * 0.01,
                "regime": 0.8 + (asset_hash % 20) * 0.01,
            },
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        rl_state = RLStateResponse(
            composite_state=intelligence_state,
            state_components={
                "market_data_points": len(recent_data),
                "strategy_count": len(strategy_list),
                "asset_hash": asset_hash,
            },
            assembly_metadata={
                "deterministic": True,
                "test_mode": True,
            },
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        logger.info(f"RL state assembled for {primary_asset} with {len(strategy_list)} strategies")
        return rl_state
        
    except Exception as e:
        logger.error("RL state assembly failed", error=str(e), asset_ids=asset_ids)
        raise HTTPException(status_code=500, detail=f"RL state assembly failed: {str(e)}")


@app.get("/data/search")
async def search_symbols(
    query: str,
    source: str = "yahoo_finance",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for asset symbols.
    
    Args:
        query: Search query (e.g., 'Apple', 'EUR', 'BTC')
        source: Data source ('yahoo_finance', 'alpha_vantage', etc.)
        limit: Maximum number of results
        
    Returns:
        List of matching symbols with metadata
    """
    try:
        data_source = DataSource(source)
        results = await data_importer.search_symbols(query, data_source, limit)
        
        logger.info(f"Symbol search completed: {len(results)} results", query=query)
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Symbol search failed", error=str(e), query=query)
        raise HTTPException(status_code=500, detail=f"Symbol search failed: {str(e)}")


@app.get("/data/symbol-info")
async def get_symbol_info(
    symbol: str,
    source: str = "yahoo_finance",
) -> Dict[str, Any]:
    """
    Get detailed information about a symbol.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        source: Data source
        
    Returns:
        Symbol information including name, type, exchange, etc.
    """
    try:
        data_source = DataSource(source)
        info = await data_importer.get_symbol_info(symbol, data_source)
        
        logger.info(f"Symbol info retrieved", symbol=symbol)
        return info
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Symbol info retrieval failed", error=str(e), symbol=symbol)
        raise HTTPException(status_code=500, detail=f"Symbol info retrieval failed: {str(e)}")


@app.post("/data/import")
async def import_external_data(
    symbol: str,
    source: str = "yahoo_finance",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    Import market data from external source.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        source: Data source ('yahoo_finance', etc.)
        start_date: Start date (ISO format, e.g., '2024-01-01')
        end_date: End date (ISO format)
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')
        
    Returns:
        Imported data and metadata
    """
    try:
        data_source = DataSource(source)
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Fetch data
        market_data = await data_importer.fetch_data(
            data_source,
            symbol,
            start_dt,
            end_dt,
            interval,
        )
        
        if not market_data:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Convert to dict for response
        data_dicts = [
            {
                'timestamp': d.timestamp.isoformat(),
                'asset_id': d.asset_id,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume,
            }
            for d in market_data
        ]
        
        logger.info(
            f"Data import completed: {len(market_data)} points",
            symbol=symbol,
            source=source,
        )
        
        return {
            'symbol': symbol,
            'source': source,
            'interval': interval,
            'data_points': len(market_data),
            'start_date': market_data[0].timestamp.isoformat() if market_data else None,
            'end_date': market_data[-1].timestamp.isoformat() if market_data else None,
            'data': data_dicts,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data import failed", error=str(e), symbol=symbol)
        raise HTTPException(status_code=500, detail=f"Data import failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# MODEL REGISTRY API ENDPOINTS
# ============================================================================

@app.get("/models/list")
async def list_models(
    category: Optional[str] = None,
    use_case: Optional[str] = None,
    production_ready: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    List available ML models with optional filtering.
    
    Args:
        category: Filter by model category (time_series, representation, graph, etc.)
        use_case: Filter by use case (price_forecasting, regime_detection, etc.)
        production_ready: Filter by production readiness
        
    Returns:
        List of models with metadata
    """
    try:
        cat = ModelCategory(category) if category else None
        uc = UseCase(use_case) if use_case else None
        
        models = model_registry.list_models(
            category=cat,
            use_case=uc,
            production_ready=production_ready
        )
        
        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category.value,
                    "use_cases": [u.value for u in m.use_cases],
                    "description": m.description,
                    "strengths": m.strengths,
                    "weaknesses": m.weaknesses,
                    "best_for": m.best_for,
                    "production_ready": m.production_ready,
                    "latency_class": m.latency_class,
                    "data_requirements": m.data_requirements,
                    "explainability": m.explainability,
                    "hyperparameters": m.hyperparameters,
                    "min_samples": m.min_samples,
                    "recommended_samples": m.recommended_samples,
                    "gpu_required": m.gpu_required,
                    "memory_mb": m.memory_mb,
                    "supports_online_learning": m.supports_online_learning,
                    "supports_transfer_learning": m.supports_transfer_learning,
                }
                for m in models
            ],
            "count": len(models),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Complete model specification
    """
    try:
        model = model_registry.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        return {
            "id": model.id,
            "name": model.name,
            "category": model.category.value,
            "use_cases": [u.value for u in model.use_cases],
            "description": model.description,
            "strengths": model.strengths,
            "weaknesses": model.weaknesses,
            "best_for": model.best_for,
            "production_ready": model.production_ready,
            "latency_class": model.latency_class,
            "data_requirements": model.data_requirements,
            "explainability": model.explainability,
            "hyperparameters": model.hyperparameters,
            "dependencies": model.dependencies,
            "min_samples": model.min_samples,
            "recommended_samples": model.recommended_samples,
            "supports_online_learning": model.supports_online_learning,
            "supports_transfer_learning": model.supports_transfer_learning,
            "gpu_required": model.gpu_required,
            "memory_mb": model.memory_mb,
            "paper_url": model.paper_url,
            "implementation_url": model.implementation_url,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model details", error=str(e), model_id=model_id)
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")


@app.get("/models/recommend")
async def recommend_models(use_case: str) -> Dict[str, Any]:
    """
    Get recommended models for a specific use case.
    
    Args:
        use_case: Use case identifier (e.g., 'price_forecasting', 'regime_detection')
        
    Returns:
        Recommended models sorted by suitability
    """
    try:
        uc = UseCase(use_case)
        models = model_registry.get_recommended_models(uc)
        
        return {
            "use_case": use_case,
            "recommendations": [
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category.value,
                    "description": m.description,
                    "strengths": m.strengths[:3],  # Top 3 strengths
                    "best_for": m.best_for,
                    "production_ready": m.production_ready,
                    "latency_class": m.latency_class,
                    "data_requirements": m.data_requirements,
                    "explainability": m.explainability,
                    "gpu_required": m.gpu_required,
                }
                for m in models
            ],
            "count": len(models),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid use case: {str(e)}")
    except Exception as e:
        logger.error("Failed to recommend models", error=str(e), use_case=use_case)
        raise HTTPException(status_code=500, detail=f"Failed to recommend models: {str(e)}")


@app.get("/models/categories")
async def get_model_categories() -> Dict[str, Any]:
    """
    Get all available model categories.
    
    Returns:
        List of model categories
    """
    try:
        categories = model_registry.get_model_categories()
        
        # Get count for each category
        category_counts = {}
        for cat in categories:
            models = model_registry.list_models(category=ModelCategory(cat))
            category_counts[cat] = len(models)
        
        return {
            "categories": [
                {
                    "id": cat,
                    "name": cat.replace("_", " ").title(),
                    "count": category_counts[cat]
                }
                for cat in categories
            ],
            "total_categories": len(categories),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error("Failed to get categories", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@app.get("/models/use-cases")
async def get_use_cases() -> Dict[str, Any]:
    """
    Get all available use cases.
    
    Returns:
        List of use cases with model counts
    """
    try:
        use_cases = model_registry.get_use_cases()
        
        # Get count for each use case
        use_case_counts = {}
        for uc in use_cases:
            models = model_registry.list_models(use_case=UseCase(uc))
            use_case_counts[uc] = len(models)
        
        return {
            "use_cases": [
                {
                    "id": uc,
                    "name": uc.replace("_", " ").title(),
                    "count": use_case_counts[uc]
                }
                for uc in use_cases
            ],
            "total_use_cases": len(use_cases),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error("Failed to get use cases", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get use cases: {str(e)}")


# ============================================================================
# STRATEGY REGISTRY ENDPOINTS
# ============================================================================

from .strategy_registry import strategy_registry, StrategyFamily, TimeHorizon, AssetClass

@app.get("/strategies/list")
async def list_strategies(
    family: Optional[str] = None,
    horizon: Optional[str] = None,
    asset_class: Optional[str] = None,
    production_ready: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    List all available trading strategies with optional filtering.
    
    Args:
        family: Filter by strategy family (trend, mean_reversion, etc.)
        horizon: Filter by time horizon (intraday, daily, etc.)
        asset_class: Filter by asset class (fx, equities, etc.)
        production_ready: Filter by production readiness
    
    Returns:
        List of strategies with metadata
    """
    try:
        fam = StrategyFamily(family) if family else None
        hor = TimeHorizon(horizon) if horizon else None
        ac = AssetClass(asset_class) if asset_class else None
        
        strategies = strategy_registry.list_strategies(
            family=fam,
            horizon=hor,
            asset_class=ac,
            production_ready=production_ready
        )
        
        return {
            "strategies": [
                {
                    "id": s.id,
                    "name": s.name,
                    "family": s.family.value,
                    "horizon": s.horizon.value,
                    "asset_classes": [ac.value for ac in s.asset_classes],
                    "description": s.description,
                    "production_ready": s.production_ready,
                    "complexity": s.complexity,
                    "typical_sharpe": s.typical_sharpe,
                    "typical_max_dd": s.typical_max_dd,
                    "typical_win_rate": s.typical_win_rate,
                }
                for s in strategies
            ],
            "count": len(strategies),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        logger.error("Failed to list strategies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list strategies: {str(e)}")


@app.get("/strategies/{strategy_id}")
async def get_strategy_details(strategy_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific strategy.
    
    Args:
        strategy_id: Strategy identifier
    
    Returns:
        Complete strategy specification
    """
    try:
        strategy = strategy_registry.get_strategy(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        
        return {
            "id": strategy.id,
            "name": strategy.name,
            "family": strategy.family.value,
            "horizon": strategy.horizon.value,
            "asset_classes": [ac.value for ac in strategy.asset_classes],
            "description": strategy.description,
            "signal_logic": strategy.signal_logic,
            "entry_rules": strategy.entry_rules,
            "exit_rules": strategy.exit_rules,
            "risk_controls": strategy.risk_controls,
            "strengths": strategy.strengths,
            "weaknesses": strategy.weaknesses,
            "best_for": strategy.best_for,
            "production_ready": strategy.production_ready,
            "complexity": strategy.complexity,
            "data_requirements": strategy.data_requirements,
            "latency_sensitivity": strategy.latency_sensitivity,
            "parameters": strategy.parameters,
            "typical_sharpe": strategy.typical_sharpe,
            "typical_max_dd": strategy.typical_max_dd,
            "typical_win_rate": strategy.typical_win_rate,
            "typical_turnover": strategy.typical_turnover,
            "max_position_size": strategy.max_position_size,
            "max_leverage": strategy.max_leverage,
            "stop_loss_pct": strategy.stop_loss_pct,
            "regime_affinity": strategy.regime_affinity,
            "paper_url": strategy.paper_url,
            "implementation_notes": strategy.implementation_notes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get strategy details", strategy_id=strategy_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get strategy details: {str(e)}")


@app.get("/strategies/recommend")
async def recommend_strategies(
    asset_class: str,
    horizon: str,
    regime: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommended strategies for given market conditions.
    
    Args:
        asset_class: Asset class (fx, equities, crypto, etc.)
        horizon: Time horizon (intraday, daily, swing, position)
        regime: Optional current market regime
    
    Returns:
        Recommended strategies sorted by suitability
    """
    try:
        ac = AssetClass(asset_class)
        hor = TimeHorizon(horizon)
        
        strategies = strategy_registry.get_recommended_strategies(
            asset_class=ac,
            horizon=hor,
            regime=regime
        )
        
        return {
            "asset_class": asset_class,
            "horizon": horizon,
            "regime": regime,
            "recommendations": [
                {
                    "id": s.id,
                    "name": s.name,
                    "family": s.family.value,
                    "description": s.description,
                    "typical_sharpe": s.typical_sharpe,
                    "typical_max_dd": s.typical_max_dd,
                    "complexity": s.complexity,
                    "regime_affinity": s.regime_affinity.get(regime, 0.0) if regime else 0.0
                }
                for s in strategies
            ],
            "count": len(strategies),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        logger.error("Failed to recommend strategies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to recommend strategies: {str(e)}")


@app.get("/strategies/families")
async def get_strategy_families() -> Dict[str, Any]:
    """
    Get all strategy families.
    
    Returns:
        List of strategy families with counts
    """
    try:
        families = strategy_registry.get_strategy_families()
        
        # Get count for each family
        family_counts = {}
        for fam in families:
            strategies = strategy_registry.list_strategies(family=StrategyFamily(fam))
            family_counts[fam] = len(strategies)
        
        return {
            "families": [
                {
                    "id": fam,
                    "name": fam.replace("_", " ").title(),
                    "count": family_counts[fam]
                }
                for fam in families
            ],
            "total_families": len(families),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error("Failed to get strategy families", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get strategy families: {str(e)}")


@app.get("/strategies/horizons")
async def get_time_horizons() -> Dict[str, Any]:
    """
    Get all time horizons.
    
    Returns:
        List of time horizons with counts
    """
    try:
        horizons = strategy_registry.get_time_horizons()
        
        # Get count for each horizon
        horizon_counts = {}
        for hor in horizons:
            strategies = strategy_registry.list_strategies(horizon=TimeHorizon(hor))
            horizon_counts[hor] = len(strategies)
        
        return {
            "horizons": [
                {
                    "id": hor,
                    "name": hor.replace("_", " ").title(),
                    "count": horizon_counts[hor]
                }
                for hor in horizons
            ],
            "total_horizons": len(horizons),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error("Failed to get time horizons", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get time horizons: {str(e)}")


# Markets API Endpoints

@app.get("/markets/live-data")
async def get_live_market_data(
    assets: str,
    include_depth: bool = False,
) -> Dict[str, Any]:
    """
    Get real-time market data.
    
    Args:
        assets: Comma-separated list of asset IDs
        include_depth: Include order book depth
        
    Returns:
        Live market data for requested assets
    """
    try:
        asset_list = [a.strip() for a in assets.split(",") if a.strip()]
        
        # Generate mock live data (in production, this would come from Rust execution core)
        live_data = []
        for asset_id in asset_list:
            # Deterministic mock data based on asset_id
            asset_hash = hash(asset_id) % 1000
            base_price = 1.0 + (asset_hash % 100) * 0.01
            
            data = {
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bid": base_price - 0.0001,
                "ask": base_price + 0.0001,
                "last": base_price,
                "volume": 1000000 + (asset_hash % 5000000),
                "spread_bps": 2.0,
                "depth_bid": 2500000.0,
                "depth_ask": 2300000.0,
                "tick_frequency": 100 + (asset_hash % 50),
            }
            
            if include_depth:
                data["order_book"] = {
                    "bids": [[base_price - i * 0.0001, 100000.0] for i in range(1, 6)],
                    "asks": [[base_price + i * 0.0001, 100000.0] for i in range(1, 6)],
                }
            
            live_data.append(data)
        
        logger.info(f"Live market data retrieved for {len(asset_list)} assets")
        return {"data": live_data, "timestamp": datetime.now(timezone.utc).isoformat()}
        
    except Exception as e:
        logger.error("Live market data retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve live market data: {str(e)}")


@app.get("/markets/correlations")
async def get_correlation_matrix(
    assets: str,
    window: str = "24H",
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Calculate rolling correlation matrix.
    
    Args:
        assets: Comma-separated list of asset IDs
        window: Rolling window (1H, 4H, 24H, 7D, 30D)
        method: Correlation method (pearson, spearman, kendall)
        
    Returns:
        Correlation matrix with significance levels
    """
    try:
        asset_list = [a.strip() for a in assets.split(",") if a.strip()]
        
        # Convert window to number of data points
        window_map = {"1H": 60, "4H": 240, "24H": 1440, "7D": 10080, "30D": 43200}
        window_points = window_map.get(window, 100)
        
        # Calculate correlation matrix
        corr_data = market_analytics.calculate_correlation_matrix(
            asset_list,
            window=min(window_points, 100),  # Limit to available data
            method=method,
        )
        
        # Convert to JSON-serializable format
        result = {
            "timestamp": corr_data.timestamp.isoformat(),
            "assets": corr_data.assets,
            "correlation_matrix": corr_data.correlation_matrix.tolist() if corr_data.correlation_matrix.size > 0 else [],
            "significance": corr_data.significance.tolist() if corr_data.significance.size > 0 else [],
            "window": window,
            "method": method,
            "clusters": corr_data.clusters or [],
        }
        
        logger.info(f"Correlation matrix calculated for {len(asset_list)} assets")
        return result
        
    except Exception as e:
        logger.error("Correlation calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to calculate correlations: {str(e)}")


@app.get("/markets/microstructure")
async def get_market_microstructure(
    asset_id: str,
) -> Dict[str, Any]:
    """
    Get real-time microstructure metrics.
    
    Args:
        asset_id: Asset identifier
        
    Returns:
        Market microstructure metrics
    """
    try:
        # Generate mock microstructure data (in production, this would come from Rust)
        asset_hash = hash(asset_id) % 1000
        base_price = 1.0 + (asset_hash % 100) * 0.01
        
        microstructure = {
            "asset_id": asset_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spread_bps": 1.8 + (asset_hash % 10) * 0.1,
            "effective_spread_bps": 1.6 + (asset_hash % 10) * 0.1,
            "quoted_spread_bps": 2.0 + (asset_hash % 10) * 0.1,
            "depth_bid": 2500000.0 + (asset_hash % 1000000),
            "depth_ask": 2300000.0 + (asset_hash % 1000000),
            "imbalance_ratio": 0.52 + (asset_hash % 20) * 0.01,
            "tick_frequency": 100.0 + (asset_hash % 50),
            "price_impact_bps": 0.9 + (asset_hash % 10) * 0.1,
        }
        
        logger.info(f"Microstructure data retrieved for {asset_id}")
        return microstructure
        
    except Exception as e:
        logger.error("Microstructure retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve microstructure: {str(e)}")


@app.get("/markets/liquidity")
async def get_liquidity_analysis(
    assets: str,
) -> Dict[str, Any]:
    """
    Analyze market liquidity.
    
    Args:
        assets: Comma-separated list of asset IDs
        
    Returns:
        Liquidity analysis for requested assets
    """
    try:
        asset_list = [a.strip() for a in assets.split(",") if a.strip()]
        
        # Generate mock liquidity data (in production, this would come from Rust)
        liquidity_data = []
        for asset_id in asset_list:
            asset_hash = hash(asset_id) % 1000
            
            data = {
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bid_liquidity_usd": 2500000.0 + (asset_hash % 2000000),
                "ask_liquidity_usd": 2300000.0 + (asset_hash % 2000000),
                "total_liquidity_usd": 4800000.0 + (asset_hash % 4000000),
                "liquidity_score": 75.0 + (asset_hash % 25),
                "resilience_score": 70.0 + (asset_hash % 30),
                "toxicity_score": 20.0 + (asset_hash % 30),
            }
            liquidity_data.append(data)
        
        logger.info(f"Liquidity analysis completed for {len(asset_list)} assets")
        return {"data": liquidity_data, "timestamp": datetime.now(timezone.utc).isoformat()}
        
    except Exception as e:
        logger.error("Liquidity analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze liquidity: {str(e)}")


@app.get("/markets/events")
async def get_market_events(
    since: Optional[str] = None,
    event_types: Optional[str] = None,
    severity_min: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Get recent market events.
    
    Args:
        since: ISO timestamp to get events since
        event_types: Comma-separated list of event types
        severity_min: Minimum severity (0-1)
        
    Returns:
        List of market events
    """
    try:
        # Generate mock events (in production, these would come from event bus)
        events = [
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
                "asset_id": "EURUSD",
                "event_type": "VOLATILITY_SPIKE",
                "severity": 0.7,
                "description": "Volatility increased by 45% above baseline",
                "recommended_action": "ALERT",
            },
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
                "asset_id": "GBPUSD",
                "event_type": "LIQUIDITY_DROP",
                "severity": 0.6,
                "description": "Liquidity decreased by 30%",
                "recommended_action": "ALERT",
            },
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                "asset_id": "BTCUSD",
                "event_type": "PRICE_SPIKE",
                "severity": 0.8,
                "description": "Price increased by 3.5% in 5 minutes",
                "recommended_action": "INVESTIGATE",
            },
        ]
        
        # Filter by severity
        events = [e for e in events if e["severity"] >= severity_min]
        
        # Filter by event types
        if event_types:
            types = [t.strip() for t in event_types.split(",")]
            events = [e for e in events if e["event_type"] in types]
        
        # Filter by time
        if since:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= since_dt]
        
        logger.info(f"Retrieved {len(events)} market events")
        return events
        
    except Exception as e:
        logger.error("Market events retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market events: {str(e)}")


# ============================================================================
# LLM AND RAG API ENDPOINTS
# ============================================================================

# Request/Response models for LLM and RAG
class LLMQueryRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None
    prefer_local: bool = True
    provider: Optional[str] = None

class LLMQueryResponse(BaseModel):
    answer: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    timestamp: datetime

class RAGQueryRequest(BaseModel):
    question: str
    context_filter: Optional[Dict[str, Any]] = None

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    model_info: Dict[str, Any]
    retrieval_info: Optional[Dict[str, Any]] = None

class DocumentIngestRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentIngestResponse(BaseModel):
    success: bool
    chunks_created: int
    document_id: str
    message: str

class ResearchRequest(BaseModel):
    query: str
    include_web: bool = True
    include_market_data: bool = True

class ResearchResponse(BaseModel):
    query: str
    timestamp: str
    rag_analysis: Optional[Dict[str, Any]]
    web_research: Optional[Dict[str, Any]]
    market_data: Optional[Dict[str, Any]]
    comprehensive_analysis: Optional[Dict[str, Any]]

class StockAnalysisRequest(BaseModel):
    symbol: str

class StockAnalysisResponse(BaseModel):
    symbol: str
    market_data: Dict[str, Any]
    news_analysis: Dict[str, Any]
    ai_analysis: str
    timestamp: str

# LLM Endpoints
@app.post("/llm/query", response_model=LLMQueryResponse)
async def llm_query(
    request: LLMQueryRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Query LLM with financial analysis capabilities."""
    try:
        response = await llm_svc.generate(
            prompt=request.query,
            system_prompt=request.system_prompt,
            prefer_local=request.prefer_local,
            provider=request.provider
        )
        
        return LLMQueryResponse(
            answer=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            response_time=response.response_time,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

@app.post("/llm/financial-analysis", response_model=LLMQueryResponse)
async def financial_analysis(
    request: LLMQueryRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Perform specialized financial analysis using LLM."""
    try:
        context = request.system_prompt if request.system_prompt else {}
        response = await llm_svc.financial_analysis(request.query, context)
        
        return LLMQueryResponse(
            answer=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            response_time=response.response_time,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Financial analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Financial analysis failed: {str(e)}")

# RAG Endpoints
@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Query RAG system with document retrieval."""
    try:
        result = await rag_svc.query(request.question, request.context_filter)
        
        return RAGQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            model_info=result["model_info"],
            retrieval_info=result.get("retrieval_info")
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/rag/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Ingest document into RAG system."""
    try:
        import tempfile
        import json
        
        # Parse metadata
        doc_metadata = json.loads(metadata) if metadata else {}
        doc_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        })
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Ingest document
            chunks_created = await rag_svc.ingest_document(tmp_file_path, doc_metadata)
            
            return DocumentIngestResponse(
                success=True,
                chunks_created=chunks_created,
                document_id=f"doc_{hash(file.filename)}",
                message=f"Successfully ingested {file.filename} with {chunks_created} chunks"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.get("/rag/stats")
async def rag_stats(rag_svc: RAGService = Depends(get_rag_service)):
    """Get RAG system statistics."""
    try:
        stats = await rag_svc.get_document_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")

# Research Endpoints
@app.post("/research/comprehensive", response_model=ResearchResponse)
async def comprehensive_research(
    request: ResearchRequest,
    research_svc: FinancialResearchService = Depends(get_research_service)
):
    """Perform comprehensive financial research."""
    try:
        result = await research_svc.comprehensive_research(
            request.query,
            include_web=request.include_web
        )
        
        return ResearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Comprehensive research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@app.post("/research/stock-analysis", response_model=StockAnalysisResponse)
async def stock_analysis(
    request: StockAnalysisRequest,
    research_svc: FinancialResearchService = Depends(get_research_service)
):
    """Perform detailed stock analysis."""
    try:
        result = await research_svc.stock_analysis(request.symbol)
        
        return StockAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Stock analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stock analysis failed: {str(e)}")

@app.get("/research/market-overview")
async def market_overview(
    research_svc: FinancialResearchService = Depends(get_research_service)
):
    """Get comprehensive market overview."""
    try:
        # Get economic indicators
        economic_data = await research_svc.market_data.get_economic_indicators()
        
        # Generate AI analysis of market conditions
        market_analysis = await research_svc.llm_service.financial_analysis(
            "Provide a comprehensive market overview based on current economic indicators and market conditions. Include key trends, risks, and opportunities.",
            {"economic_indicators": economic_data}
        )
        
        return {
            "economic_indicators": economic_data,
            "ai_analysis": market_analysis.content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market overview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "intelligence_layer.main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=config.debug,
    )


# ============================================================================
# Portfolio & Risk Management API Endpoints
# ============================================================================

@app.get("/portfolios/list")
async def list_portfolios() -> Dict[str, Any]:
    """List all portfolios."""
    # Mock implementation - replace with actual database query
    portfolios = [
        {
            "id": "PORT-001",
            "name": "MAIN TRADING PORTFOLIO",
            "baseCurrency": "USD",
            "initialCapital": 100000,
            "currentCapital": 104127.89,
            "mode": "PAPER",
            "status": "ACTIVE",
            "createdAt": "2024-01-01T00:00:00Z",
            "strategyAllocations": [
                {"strategyId": "regime_switching", "weight": 0.40, "capitalAllocated": 40000},
                {"strategyId": "momentum_rotation", "weight": 0.30, "capitalAllocated": 30000},
            ],
            "allocationModel": "VOL_TARGET",
            "rebalanceFrequency": "WEEKLY",
            "turnoverConstraint": 20,
        }
    ]
    return {"portfolios": portfolios}


@app.get("/portfolios/{portfolio_id}")
async def get_portfolio(portfolio_id: str) -> Dict[str, Any]:
    """Get portfolio by ID."""
    # Mock implementation
    return {
        "id": portfolio_id,
        "name": "MAIN TRADING PORTFOLIO",
        "baseCurrency": "USD",
        "initialCapital": 100000,
        "currentCapital": 104127.89,
        "mode": "PAPER",
        "status": "ACTIVE",
        "createdAt": "2024-01-01T00:00:00Z",
        "strategyAllocations": [],
        "allocationModel": "VOL_TARGET",
        "rebalanceFrequency": "WEEKLY",
        "turnoverConstraint": 20,
    }


@app.post("/portfolios/create")
async def create_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """Create new portfolio."""
    # Mock implementation - add to database
    return {
        **portfolio,
        "id": f"PORT-{random.randint(100, 999)}",
        "createdAt": datetime.now().isoformat(),
        "currentCapital": portfolio["initialCapital"],
    }


@app.put("/portfolios/{portfolio_id}")
async def update_portfolio(portfolio_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update portfolio."""
    # Mock implementation
    return {
        "id": portfolio_id,
        **updates,
    }


@app.delete("/portfolios/{portfolio_id}")
async def delete_portfolio(portfolio_id: str) -> Dict[str, Any]:
    """Delete portfolio."""
    return {"message": f"Portfolio {portfolio_id} deleted"}


@app.get("/portfolios/{portfolio_id}/risk-limits")
async def list_risk_limits(portfolio_id: str) -> Dict[str, Any]:
    """List risk limits for portfolio."""
    limits = [
        {
            "id": "LIMIT-001",
            "portfolioId": portfolio_id,
            "type": "HARD",
            "category": "POSITION",
            "name": "MAX POSITION SIZE",
            "threshold": 15.0,
            "currentValue": 10.5,
            "breached": False,
            "action": "BLOCK",
        }
    ]
    return {"limits": limits}


@app.post("/risk-limits/create")
async def create_risk_limit(limit: Dict[str, Any]) -> Dict[str, Any]:
    """Create risk limit."""
    return {
        **limit,
        "id": f"LIMIT-{random.randint(100, 999)}",
        "currentValue": 0,
        "breached": False,
    }


@app.put("/risk-limits/{limit_id}")
async def update_risk_limit(limit_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update risk limit."""
    return {
        "id": limit_id,
        **updates,
    }


@app.delete("/risk-limits/{limit_id}")
async def delete_risk_limit(limit_id: str) -> Dict[str, Any]:
    """Delete risk limit."""
    return {"message": f"Risk limit {limit_id} deleted"}


@app.get("/stress-scenarios/list")
async def list_stress_scenarios() -> Dict[str, Any]:
    """List stress test scenarios."""
    scenarios = [
        {
            "id": "STRESS-001",
            "name": "2008 FINANCIAL CRISIS",
            "type": "HISTORICAL",
            "description": "Lehman Brothers collapse scenario",
            "parameters": {"startDate": "2008-09-15", "duration": "90d", "volMultiplier": 3.5},
            "impact": {
                "portfolioLoss": -18500,
                "maxDrawdown": 17.8,
                "recoveryDays": 120,
            },
        }
    ]
    return {"scenarios": scenarios}


@app.post("/stress-test/run")
async def run_stress_test(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run stress test."""
    return {
        "portfolioId": request["portfolioId"],
        "scenarioId": request["scenarioId"],
        "impact": {
            "portfolioLoss": -10000 - random.random() * 10000,
            "maxDrawdown": 10 + random.random() * 10,
            "recoveryDays": 60 + random.randint(0, 60),
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/portfolios/{portfolio_id}/rebalance")
async def rebalance_portfolio(portfolio_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """Rebalance portfolio allocations."""
    return {
        "id": portfolio_id,
        "strategyAllocations": request["allocations"],
        "rebalancedAt": datetime.now().isoformat(),
    }


# ============================================================================
# Data & Models Management API Endpoints
# ============================================================================

@app.get("/models/deployed")
async def list_deployed_models() -> Dict[str, Any]:
    """List all deployed models."""
    models = [
        {
            "id": "TCN_V1.2",
            "name": "TCN EMBEDDING MODEL",
            "type": "TCN",
            "status": "ACTIVE",
            "accuracy": 92.3,
            "trained": "2024-01-10",
            "version": "1.2.0",
            "hash": "a3f5b2c8d9e1f4a6",
            "dataset": "EURUSD_2020-2023_CLEAN",
        }
    ]
    return {"models": models}


@app.post("/models/deploy/{job_id}")
async def deploy_model(job_id: str) -> Dict[str, Any]:
    """Deploy a trained model."""
    return {
        "id": f"MODEL_V{random.randint(1, 100)}",
        "name": "DEPLOYED MODEL",
        "type": "MOCK",
        "status": "TESTING",
        "accuracy": 85 + random.random() * 10,
        "trained": datetime.now().strftime("%Y-%m-%d"),
        "version": "1.0.0",
        "hash": f"{random.randint(1000000, 9999999):x}",
        "dataset": "MOCK_DATASET",
    }


@app.put("/models/{model_id}/status")
async def update_model_status(model_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """Update model status."""
    return {
        "id": model_id,
        "status": request["status"],
        "updatedAt": datetime.now().isoformat(),
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> Dict[str, Any]:
    """Delete a deployed model."""
    return {"message": f"Model {model_id} deleted"}


@app.post("/models/{model_id}/validate")
async def validate_model(model_id: str) -> Dict[str, Any]:
    """Run validation on a model."""
    return {
        "temporal_continuity": 0.85 + random.random() * 0.15,
        "regime_separability": 0.80 + random.random() * 0.15,
        "similarity_coherence": 0.88 + random.random() * 0.12,
        "prediction_accuracy": 0.82 + random.random() * 0.15,
        "precision": 0.85 + random.random() * 0.12,
        "recall": 0.83 + random.random() * 0.12,
        "f1_score": 0.84 + random.random() * 0.12,
        "confusion_matrix": [
            [850, 50, 30, 20],
            [40, 880, 45, 35],
            [35, 55, 870, 40],
            [25, 45, 55, 875],
        ],
    }


@app.get("/datasets/list")
async def list_datasets() -> Dict[str, Any]:
    """List all training datasets."""
    datasets = [
        {
            "id": "DS001",
            "name": "EURUSD_2020-2023_CLEAN",
            "records": 1250000,
            "features": 32,
            "quality": 98.5,
            "size": "2.4 GB",
            "dateRange": "2020-01-01 to 2023-12-31",
            "assets": ["EURUSD"],
            "status": "READY",
        }
    ]
    return {"datasets": datasets}


@app.post("/datasets/create")
async def create_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Create/import a new dataset."""
    return {
        **dataset,
        "id": f"DS{random.randint(100, 999)}",
    }


@app.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str) -> Dict[str, Any]:
    """Validate dataset quality."""
    return {
        "quality": 95 + random.random() * 5,
        "issues": [],
    }


@app.post("/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """Export dataset."""
    format_ext = request["format"].lower()
    return {
        "url": f"/exports/{dataset_id}.{format_ext}",
    }


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    """Delete a dataset."""
    return {"message": f"Dataset {dataset_id} deleted"}


@app.get("/training/jobs")
async def list_training_jobs() -> Dict[str, Any]:
    """List all training jobs."""
    jobs = [
        {
            "id": "JOB001",
            "model_id": "vae",
            "model_name": "Variational Autoencoder",
            "dataset_id": "DS001",
            "status": "RUNNING",
            "current_epoch": 45,
            "total_epochs": 100,
            "train_loss": 0.0234,
            "val_loss": 0.0289,
            "accuracy": 89.1,
            "started_at": "2024-01-15 12:00:00",
            "eta": "2H 15M",
        }
    ]
    return {"jobs": jobs}


@app.post("/training/start")
async def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Start a new training job."""
    return {
        "id": f"JOB{random.randint(100, 999)}",
        "model_id": config["model_id"],
        "model_name": "Mock Model",
        "dataset_id": config["dataset_id"],
        "status": "RUNNING",
        "current_epoch": 0,
        "total_epochs": config["epochs"],
        "train_loss": 0.5,
        "val_loss": 0.6,
        "accuracy": 0,
        "started_at": datetime.now().isoformat(),
        "eta": f"{config['epochs'] // 20}H {(config['epochs'] % 20) * 3}M",
    }


@app.get("/training/jobs/{job_id}")
async def get_training_job_status(job_id: str) -> Dict[str, Any]:
    """Get training job status."""
    return {
        "id": job_id,
        "status": "RUNNING",
        "current_epoch": random.randint(1, 100),
        "total_epochs": 100,
        "train_loss": 0.02 + random.random() * 0.05,
        "val_loss": 0.03 + random.random() * 0.05,
        "accuracy": 85 + random.random() * 10,
    }


@app.post("/training/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str) -> Dict[str, Any]:
    """Cancel a training job."""
    return {"message": f"Training job {job_id} cancelled"}


# ============================================================================
# Execution Management API Endpoints
# ============================================================================

@app.get("/execution/orders")
async def list_execution_orders(
    status: Optional[str] = None,
    asset: Optional[str] = None,
    strategy_id: Optional[str] = None,
    adapter_id: Optional[str] = None
) -> Dict[str, Any]:
    """List all orders with optional filtering."""
    # Mock implementation
    orders = [
        {
            "id": "ORD-001",
            "internalId": "INT-001",
            "venueId": "DERIV-12345",
            "strategyId": "regime_switching",
            "portfolioId": "PORT-001",
            "asset": "EURUSD",
            "side": "BUY",
            "size": 50000,
            "filledSize": 50000,
            "orderType": "MARKET",
            "avgFillPrice": 1.0845,
            "status": "FILLED",
            "createdAt": datetime.now().isoformat(),
            "filledAt": datetime.now().isoformat(),
            "slippageBps": 0.2,
            "latencyMs": 12,
            "adapterId": "DERIV_API",
        }
    ]
    
    # Apply filters
    if status:
        orders = [o for o in orders if o["status"] == status]
    if asset:
        orders = [o for o in orders if o["asset"] == asset]
    if strategy_id:
        orders = [o for o in orders if o["strategyId"] == strategy_id]
    if adapter_id:
        orders = [o for o in orders if o["adapterId"] == adapter_id]
    
    return {"orders": orders}


@app.get("/execution/orders/{order_id}")
async def get_execution_order(order_id: str) -> Dict[str, Any]:
    """Get order by ID."""
    return {
        "id": order_id,
        "internalId": f"INT-{order_id.split('-')[1]}",
        "strategyId": "regime_switching",
        "portfolioId": "PORT-001",
        "asset": "EURUSD",
        "side": "BUY",
        "size": 50000,
        "filledSize": 50000,
        "orderType": "MARKET",
        "status": "FILLED",
        "createdAt": datetime.now().isoformat(),
        "latencyMs": 12,
        "adapterId": "DERIV_API",
    }


@app.post("/execution/orders/create")
async def create_execution_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """Create new order."""
    return {
        **order,
        "id": f"ORD-{random.randint(100, 999)}",
        "internalId": f"INT-{random.randint(100, 999)}",
        "createdAt": datetime.now().isoformat(),
        "status": "CREATED",
        "filledSize": 0,
        "latencyMs": 0,
    }


@app.post("/execution/orders/{order_id}/cancel")
async def cancel_execution_order(order_id: str) -> Dict[str, Any]:
    """Cancel order."""
    return {
        "id": order_id,
        "status": "CANCELLED",
        "cancelledAt": datetime.now().isoformat(),
    }


@app.put("/execution/orders/{order_id}")
async def modify_execution_order(order_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Modify order."""
    return {
        "id": order_id,
        **updates,
        "modifiedAt": datetime.now().isoformat(),
    }


@app.get("/execution/adapters")
async def list_execution_adapters() -> Dict[str, Any]:
    """List all execution adapters."""
    adapters = [
        {
            "id": "DERIV_API",
            "name": "DERIV API",
            "type": "BROKER",
            "status": "CONNECTED",
            "health": "HEALTHY",
            "latencyMs": 12,
            "uptimePercent": 99.98,
            "ordersToday": 1247,
            "fillsToday": 1243,
            "rejectsToday": 4,
            "errorRate": 0.32,
            "reconnectAttempts": 0,
            "lastHeartbeat": datetime.now().isoformat(),
            "rateLimitUsage": 45,
            "supportedAssets": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "supportedOrderTypes": ["MARKET", "LIMIT", "STOP"],
            "tradingHours": "24/7",
            "minOrderSize": 1000,
            "maxOrderSize": 1000000,
            "feeSchedule": {"EURUSD": 0.0001, "GBPUSD": 0.0001},
        },
        {
            "id": "MT5_ADAPTER",
            "name": "MT5 ADAPTER",
            "type": "BROKER",
            "status": "CONNECTED",
            "health": "HEALTHY",
            "latencyMs": 18,
            "uptimePercent": 99.95,
            "ordersToday": 856,
            "fillsToday": 852,
            "rejectsToday": 4,
            "errorRate": 0.47,
            "reconnectAttempts": 1,
            "lastHeartbeat": datetime.now().isoformat(),
            "rateLimitUsage": 32,
            "supportedAssets": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
            "supportedOrderTypes": ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
            "tradingHours": "24/5",
            "minOrderSize": 1000,
            "maxOrderSize": 500000,
            "feeSchedule": {"EURUSD": 0.00015, "GBPUSD": 0.00015},
        },
        {
            "id": "SHADOW_EXEC",
            "name": "SHADOW EXEC",
            "type": "SHADOW",
            "status": "CONNECTED",
            "health": "HEALTHY",
            "latencyMs": 2,
            "uptimePercent": 100.0,
            "ordersToday": 2103,
            "fillsToday": 2103,
            "rejectsToday": 0,
            "errorRate": 0.0,
            "reconnectAttempts": 0,
            "lastHeartbeat": datetime.now().isoformat(),
            "rateLimitUsage": 0,
            "supportedAssets": ["ALL"],
            "supportedOrderTypes": ["ALL"],
            "tradingHours": "24/7",
            "minOrderSize": 1,
            "maxOrderSize": 999999999,
            "feeSchedule": {},
        },
    ]
    return {"adapters": adapters}


@app.get("/execution/adapters/{adapter_id}")
async def get_execution_adapter(adapter_id: str) -> Dict[str, Any]:
    """Get adapter by ID."""
    return {
        "id": adapter_id,
        "name": adapter_id.replace("_", " "),
        "type": "BROKER",
        "status": "CONNECTED",
        "health": "HEALTHY",
        "latencyMs": 12,
        "uptimePercent": 99.98,
        "ordersToday": 1247,
        "fillsToday": 1243,
        "rejectsToday": 4,
    }


@app.put("/execution/adapters/{adapter_id}")
async def update_execution_adapter(adapter_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update adapter configuration."""
    return {
        "id": adapter_id,
        **updates,
        "updatedAt": datetime.now().isoformat(),
    }


@app.post("/execution/adapters/{adapter_id}/reconnect")
async def reconnect_execution_adapter(adapter_id: str) -> Dict[str, Any]:
    """Reconnect adapter."""
    return {
        "id": adapter_id,
        "status": "CONNECTED",
        "reconnectedAt": datetime.now().isoformat(),
    }


@app.get("/execution/metrics")
async def get_execution_metrics() -> Dict[str, Any]:
    """Get execution metrics."""
    return {
        "avgLatencyMs": 12.5,
        "p95LatencyMs": 18.2,
        "p99LatencyMs": 24.8,
        "fillRate": 99.62,
        "rejectionRate": 0.38,
        "avgSlippageBps": 0.15,
        "implementationShortfall": 0.08,
        "priceImprovement": 0.05,
        "ordersPerSecond": 2.5,
        "peakLoad": 15.2,
    }


@app.get("/execution/tca/{order_id}")
async def get_tca_report(order_id: str) -> Dict[str, Any]:
    """Get TCA report for order."""
    return {
        "orderId": order_id,
        "expectedCost": 10.5,
        "realizedCost": 10.2,
        "spreadCapture": 0.3,
        "marketImpact": 0.15,
        "timingCost": 0.05,
        "opportunityCost": 0.1,
        "executionQuality": "GOOD",
    }


@app.get("/execution/circuit-breakers")
async def list_circuit_breakers() -> Dict[str, Any]:
    """List circuit breakers."""
    breakers = [
        {
            "id": "CB-001",
            "name": "MAX ORDER RATE",
            "type": "ORDER_RATE",
            "threshold": 100,
            "currentValue": 45,
            "breached": False,
            "action": "THROTTLE",
            "enabled": True,
        },
        {
            "id": "CB-002",
            "name": "MAX REJECTION RATE",
            "type": "REJECTION_RATE",
            "threshold": 5.0,
            "currentValue": 0.38,
            "breached": False,
            "action": "HALT",
            "enabled": True,
        },
        {
            "id": "CB-003",
            "name": "PRICE DEVIATION",
            "type": "PRICE_DEVIATION",
            "threshold": 1.0,
            "currentValue": 0.15,
            "breached": False,
            "action": "ALERT",
            "enabled": True,
        },
    ]
    return {"breakers": breakers}


@app.put("/execution/circuit-breakers/{breaker_id}")
async def update_circuit_breaker(breaker_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update circuit breaker."""
    return {
        "id": breaker_id,
        **updates,
        "updatedAt": datetime.now().isoformat(),
    }


@app.post("/execution/kill-switch")
async def execute_kill_switch() -> Dict[str, Any]:
    """Emergency kill switch - cancel all orders."""
    return {
        "cancelled": random.randint(5, 20),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/execution/reconciliation")
async def get_reconciliation_report() -> Dict[str, Any]:
    """Get reconciliation report."""
    return {
        "timestamp": datetime.now().isoformat(),
        "internalPositions": {"EURUSD": 50000, "GBPUSD": -30000},
        "brokerPositions": {"EURUSD": 50000, "GBPUSD": -30000},
        "mismatches": [],
        "cashBalance": {
            "internal": 104127.89,
            "broker": 104127.89,
            "difference": 0,
        },
        "fillMismatches": [],
    }


@app.post("/execution/reconciliation/run")
async def run_reconciliation() -> Dict[str, Any]:
    """Run reconciliation."""
    return {
        "timestamp": datetime.now().isoformat(),
        "internalPositions": {"EURUSD": 50000, "GBPUSD": -30000},
        "brokerPositions": {"EURUSD": 50000, "GBPUSD": -30000},
        "mismatches": [],
        "cashBalance": {
            "internal": 104127.89,
            "broker": 104127.89,
            "difference": 0,
        },
        "fillMismatches": [],
    }
