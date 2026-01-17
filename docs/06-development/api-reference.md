# API Reference

This document provides comprehensive API documentation for the Algorithmic Trading System's backend services.

## Services Overview

The system consists of two main backend services:

- **Intelligence Layer** (Python/FastAPI) - Port 8000
- **Execution Core** (Rust/Warp) - Port 8001

## Base URLs

- Intelligence Layer: `http://localhost:8000`
- Execution Core: `http://localhost:8001`

## Navigation Update - Option 1C

The system now implements **Option 1C** navigation sequence:
**F1:DASH | F2:WORK | F3:MLOPS | F4:MKTS | F5:INTL | F6:STRT | F7:SIMU | F8:PORT | F9:EXEC | F10:SYST**

Key change: **MLOps** (F3) replaces Data Models, providing early access to machine learning operations in the research workflow.

## Authentication

Currently, the APIs use no authentication for development. In production, implement proper authentication and authorization.

## Error Handling

All APIs return consistent error responses:

```json
{
  "detail": "Error message",
  "status": 500
}
```

## Intelligence Layer API (Port 8000)

### Health Check

#### GET /health
Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "intelligence-api"
}
```

### MLOps Endpoints

#### GET /models/list
List available ML models in the registry.

**Response:**
```json
{
  "models": [
    {
      "id": "tcn_regime_v2",
      "name": "TCN Regime Detector v2.1",
      "category": "regime_detection",
      "production_ready": true,
      "latency_class": "LOW",
      "memory_mb": 256
    }
  ]
}
```

#### GET /models/{model_id}
Get detailed information about a specific model.

**Response:**
```json
{
  "id": "tcn_regime_v2",
  "name": "TCN Regime Detector v2.1",
  "description": "Advanced regime detection using Temporal Convolutional Networks",
  "strengths": ["High accuracy", "Low latency", "Robust to noise"],
  "best_for": ["Real-time regime detection", "Multi-asset analysis"],
  "min_samples": 10000,
  "gpu_required": false,
  "supports_online_learning": true
}
```

#### GET /training/jobs
List active and recent training jobs.

**Response:**
```json
{
  "jobs": [
    {
      "id": "job_001",
      "model_id": "tcn_regime_v2",
      "status": "RUNNING",
      "current_epoch": 47,
      "total_epochs": 100,
      "train_loss": 0.0234,
      "val_loss": 0.0287,
      "accuracy": 87.3
    }
  ]
}
```

#### GET /deployment/models
List deployed models in production.

**Response:**
```json
{
  "deployed_models": [
    {
      "id": "prod_001",
      "name": "TCN Regime Detector",
      "status": "ACTIVE",
      "accuracy": 89.4,
      "version": "v2.0",
      "deployed_at": "2024-01-10T10:30:00Z"
    }
  ]
}
```

### Emergency Controls

#### POST /emergency/halt
Emergency halt - immediately stop all trading activities.

**Request Body:**
```json
{
  "reason": "Manual emergency halt",
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Emergency halt activated: Manual emergency halt",
  "timestamp": "2024-01-15T14:30:00Z",
  "previous_status": "ACTIVE",
  "new_status": "HALTED"
}
```

#### POST /emergency/resume
Resume trading after emergency halt.

**Response:**
```json
{
  "success": true,
  "message": "Emergency halt deactivated - Trading resumed",
  "timestamp": "2024-01-15T14:35:00Z",
  "previous_status": "HALTED",
  "new_status": "ACTIVE"
}
```

#### POST /trading/control
Pause or resume trading (non-emergency).

**Request Body:**
```json
{
  "action": "pause",
  "reason": "Manual pause"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Trading paused: Manual pause",
  "timestamp": "2024-01-15T14:30:00Z",
  "previous_status": "ACTIVE",
  "new_status": "PAUSED"
}
```

#### GET /system/status
Get current system and trading status.

**Response:**
```json
{
  "trading_status": "ACTIVE",
  "emergency_halt_active": false,
  "last_status_change": "2024-01-15T14:30:00Z",
  "halt_reason": null,
  "uptime_seconds": 3600.0
}
```

### Quick Actions

#### POST /quick/chart
Generate quick chart data for a symbol.

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1H",
  "start_date": "2024-01-01",
  "end_date": "2024-01-15"
}
```

**Response:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1H",
  "data_points": 100,
  "chart_url": null,
  "message": "Chart data prepared for EURUSD"
}
```

#### POST /quick/symbol-search
Search for trading symbols.

**Request Body:**
```json
{
  "query": "EUR",
  "limit": 10
}
```

**Response:**
```json
{
  "query": "EUR",
  "results": [
    {
      "symbol": "EURUSD",
      "name": "Euro / US Dollar",
      "type": "Forex",
      "exchange": "FX"
    }
  ],
  "total_found": 1
}
```

#### GET /quick/watchlist
Get current watchlist with live prices.

**Response:**
```json
{
  "items": [
    {
      "symbol": "EURUSD",
      "price": 1.0845,
      "change": 0.0012,
      "change_percent": 0.11,
      "volume": null
    }
  ],
  "last_updated": "2024-01-15T14:30:00Z"
}
```

#### POST /quick/reconnect
Force reconnection to all services.

**Response:**
```json
{
  "success": true,
  "message": "Reconnection initiated",
  "timestamp": "2024-01-15T14:30:00Z",
  "services_reconnected": ["database", "execution_core", "market_data"]
}
```

### Intelligence Endpoints

#### GET /intelligence/regime
Get current market regime inference for an asset.

**Parameters:**
- `asset_id` (string): Asset identifier

**Response:**
```json
{
  "regime_probabilities": {
    "low_vol_trending": 0.65,
    "high_vol_ranging": 0.25,
    "crisis": 0.10
  },
  "transition_likelihoods": {
    "low_vol_trending": 0.20,
    "high_vol_ranging": 0.30,
    "crisis": 0.05
  },
  "regime_entropy": 1.23,
  "confidence": 0.87,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /intelligence/graph-features
Get graph structural features for an asset.

**Parameters:**
- `asset_id` (string): Asset identifier

**Response:**
```json
{
  "cluster_membership": "cluster_2",
  "centrality_metrics": {
    "centrality_score": 0.35,
    "degree_centrality": 0.42,
    "betweenness_centrality": 0.18
  },
  "systemic_risk_proxies": {
    "systemic_risk_proxy": 0.45,
    "contagion_risk": 0.28
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /intelligence/state
Assemble composite RL state for strategy orchestration.

**Parameters:**
- `asset_ids` (string): Comma-separated list of asset IDs
- `strategy_ids` (string, optional): Comma-separated list of strategy IDs

**Response:**
```json
{
  "composite_state": {
    "embedding_similarity_context": [],
    "current_regime_label": "regime_1",
    "regime_transition_probabilities": {
      "regime_0": 0.35,
      "regime_1": 0.45,
      "regime_2": 0.20
    },
    "regime_confidence": 0.87,
    "confidence_scores": {
      "overall": 0.82,
      "regime": 0.87
    },
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "state_components": {
    "market_data_points": 100,
    "strategy_count": 2,
    "asset_hash": 456
  },
  "assembly_metadata": {
    "deterministic": true,
    "test_mode": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Execution Core API (Port 8001)

### Health Check

#### GET /health
Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "execution-core"
}
```

### System Status

#### GET /system/status
Get current system status and uptime.

**Response:**
```json
{
  "trading_status": "ACTIVE",
  "emergency_halt_active": false,
  "last_status_change": "2024-01-15T14:30:00Z",
  "halt_reason": null,
  "uptime_seconds": 3600.0
}
```

### Emergency Controls

#### POST /emergency/halt
Emergency halt - immediately stop all trading activities.

**Request Body:**
```json
{
  "reason": "Manual emergency halt",
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Emergency halt activated: Manual emergency halt",
  "timestamp": "2024-01-15T14:30:00Z",
  "previous_status": "ACTIVE",
  "new_status": "HALTED"
}
```

#### POST /emergency/resume
Resume trading after emergency halt.

**Response:**
```json
{
  "success": true,
  "message": "Emergency halt deactivated - Trading resumed",
  "timestamp": "2024-01-15T14:35:00Z",
  "previous_status": "HALTED",
  "new_status": "ACTIVE"
}
```

#### POST /trading/control
Pause or resume trading (non-emergency).

**Request Body:**
```json
{
  "action": "pause",
  "reason": "Manual pause"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Trading paused: Manual pause",
  "timestamp": "2024-01-15T14:30:00Z",
  "previous_status": "ACTIVE",
  "new_status": "PAUSED"
}
```

### Quick Orders

#### POST /orders/quick
Submit a quick order for immediate execution.

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "side": "BUY",
  "quantity": 100000.0,
  "order_type": "MARKET",
  "price": null,
  "stop_price": null
}
```

**Response:**
```json
{
  "success": true,
  "message": "Quick order submitted: BUY 100000.0 EURUSD",
  "order_id": "ORD_1705329000123",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### System Controls

#### POST /system/reconnect
Force reconnection to all external services.

**Response:**
```json
{
  "success": true,
  "message": "Reconnection initiated",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Service not initialized |

## Rate Limits

- Emergency controls: No rate limit (critical operations)
- Quick actions: 100 requests per minute
- Data endpoints: 1000 requests per minute
- Health checks: No rate limit

## Development Notes

1. All timestamps are in ISO 8601 format with UTC timezone
2. Monetary amounts are in base currency units (e.g., USD cents)
3. Percentages are expressed as decimals (0.01 = 1%)
4. Asset IDs follow standard conventions (e.g., "EURUSD", "BTC-USD")
5. Emergency controls have priority over regular trading operations
6. System maintains state consistency across both services