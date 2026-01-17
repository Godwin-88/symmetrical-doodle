use execution_core::{Config, ExecutionCoreImpl, ExecutionCore};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// System state for emergency controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub trading_status: String,  // "ACTIVE", "PAUSED", "HALTED"
    pub emergency_halt_active: bool,
    pub last_status_change: DateTime<Utc>,
    pub halt_reason: Option<String>,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            trading_status: "ACTIVE".to_string(),
            emergency_halt_active: false,
            last_status_change: Utc::now(),
            halt_reason: None,
        }
    }
}

// Request/Response models
#[derive(Debug, Deserialize)]
pub struct EmergencyHaltRequest {
    pub reason: Option<String>,
    pub force: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct EmergencyHaltResponse {
    pub success: bool,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub previous_status: String,
    pub new_status: String,
}

#[derive(Debug, Deserialize)]
pub struct TradingControlRequest {
    pub action: String,  // "pause" or "resume"
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TradingControlResponse {
    pub success: bool,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub previous_status: String,
    pub new_status: String,
}

#[derive(Debug, Deserialize)]
pub struct QuickOrderRequest {
    pub symbol: String,
    pub side: String,  // "BUY" or "SELL"
    pub quantity: f64,
    pub order_type: String,  // "MARKET", "LIMIT", "STOP"
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct QuickOrderResponse {
    pub success: bool,
    pub message: String,
    pub order_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct SystemStatusResponse {
    pub trading_status: String,
    pub emergency_halt_active: bool,
    pub last_status_change: DateTime<Utc>,
    pub halt_reason: Option<String>,
    pub uptime_seconds: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "execution_core=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Execution Core");

    // Load configuration
    let config = Config::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    // Initialize execution core
    let mut execution_core = ExecutionCoreImpl::new(&config)?;
    tracing::info!("Execution core initialized");

    // Shared system state
    let system_state = Arc::new(Mutex::new(SystemState::default()));
    let start_time = Utc::now();

    // Start API server with emergency controls and quick order endpoints
    let api_server = {
        let system_state = Arc::clone(&system_state);
        tokio::spawn(async move {
            use warp::Filter;
            use std::convert::Infallible;

            // CORS configuration
            let cors = warp::cors()
                .allow_any_origin()
                .allow_headers(vec!["content-type"])
                .allow_methods(vec!["GET", "POST", "PUT", "DELETE"]);

            // Health check endpoint
            let health = warp::path("health")
                .and(warp::get())
                .map(|| {
                    warp::reply::json(&serde_json::json!({"status": "healthy", "service": "execution-core"}))
                });

            // System status endpoint
            let system_status = {
                let system_state = Arc::clone(&system_state);
                warp::path!("system" / "status")
                    .and(warp::get())
                    .map(move || {
                        let state = system_state.lock().unwrap();
                        let uptime = Utc::now().signed_duration_since(start_time).num_seconds() as f64;
                        
                        let response = SystemStatusResponse {
                            trading_status: state.trading_status.clone(),
                            emergency_halt_active: state.emergency_halt_active,
                            last_status_change: state.last_status_change,
                            halt_reason: state.halt_reason.clone(),
                            uptime_seconds: uptime,
                        };
                        
                        warp::reply::json(&response)
                    })
            };

            // Emergency halt endpoint
            let emergency_halt = {
                let system_state = Arc::clone(&system_state);
                warp::path!("emergency" / "halt")
                    .and(warp::post())
                    .and(warp::body::json())
                    .map(move |req: EmergencyHaltRequest| {
                        let mut state = system_state.lock().unwrap();
                        let previous_status = state.trading_status.clone();
                        
                        if state.emergency_halt_active && !req.force.unwrap_or(false) {
                            return warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "error": "Emergency halt already active. Use force=true to override."
                                })),
                                warp::http::StatusCode::BAD_REQUEST
                            );
                        }
                        
                        // Set emergency halt
                        state.trading_status = "HALTED".to_string();
                        state.emergency_halt_active = true;
                        state.last_status_change = Utc::now();
                        state.halt_reason = req.reason.clone();
                        
                        tracing::error!("EMERGENCY HALT ACTIVATED: {:?}", req.reason);
                        
                        let response = EmergencyHaltResponse {
                            success: true,
                            message: format!("Emergency halt activated: {}", req.reason.unwrap_or_else(|| "Manual emergency halt".to_string())),
                            timestamp: state.last_status_change,
                            previous_status,
                            new_status: state.trading_status.clone(),
                        };
                        
                        warp::reply::with_status(
                            warp::reply::json(&response),
                            warp::http::StatusCode::OK
                        )
                    })
            };

            // Resume from halt endpoint
            let resume_halt = {
                let system_state = Arc::clone(&system_state);
                warp::path!("emergency" / "resume")
                    .and(warp::post())
                    .map(move || {
                        let mut state = system_state.lock().unwrap();
                        
                        if !state.emergency_halt_active {
                            return warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "error": "No emergency halt is active"
                                })),
                                warp::http::StatusCode::BAD_REQUEST
                            );
                        }
                        
                        let previous_status = state.trading_status.clone();
                        
                        // Resume trading
                        state.trading_status = "ACTIVE".to_string();
                        state.emergency_halt_active = false;
                        state.last_status_change = Utc::now();
                        state.halt_reason = None;
                        
                        tracing::warn!("Emergency halt DEACTIVATED - Trading resumed");
                        
                        let response = EmergencyHaltResponse {
                            success: true,
                            message: "Emergency halt deactivated - Trading resumed".to_string(),
                            timestamp: state.last_status_change,
                            previous_status,
                            new_status: state.trading_status.clone(),
                        };
                        
                        warp::reply::with_status(
                            warp::reply::json(&response),
                            warp::http::StatusCode::OK
                        )
                    })
            };

            // Trading control endpoint (pause/resume)
            let trading_control = {
                let system_state = Arc::clone(&system_state);
                warp::path!("trading" / "control")
                    .and(warp::post())
                    .and(warp::body::json())
                    .map(move |req: TradingControlRequest| {
                        let mut state = system_state.lock().unwrap();
                        
                        if state.emergency_halt_active {
                            return warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "error": "Cannot control trading while emergency halt is active"
                                })),
                                warp::http::StatusCode::BAD_REQUEST
                            );
                        }
                        
                        let previous_status = state.trading_status.clone();
                        let message = match req.action.as_str() {
                            "pause" => {
                                if state.trading_status == "PAUSED" {
                                    return warp::reply::with_status(
                                        warp::reply::json(&serde_json::json!({
                                            "error": "Trading is already paused"
                                        })),
                                        warp::http::StatusCode::BAD_REQUEST
                                    );
                                }
                                
                                state.trading_status = "PAUSED".to_string();
                                state.last_status_change = Utc::now();
                                format!("Trading paused: {}", req.reason.unwrap_or_else(|| "Manual pause".to_string()))
                            }
                            "resume" => {
                                if state.trading_status == "ACTIVE" {
                                    return warp::reply::with_status(
                                        warp::reply::json(&serde_json::json!({
                                            "error": "Trading is already active"
                                        })),
                                        warp::http::StatusCode::BAD_REQUEST
                                    );
                                }
                                
                                state.trading_status = "ACTIVE".to_string();
                                state.last_status_change = Utc::now();
                                format!("Trading resumed: {}", req.reason.unwrap_or_else(|| "Manual resume".to_string()))
                            }
                            _ => {
                                return warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "error": "Invalid action. Use 'pause' or 'resume'"
                                    })),
                                    warp::http::StatusCode::BAD_REQUEST
                                );
                            }
                        };
                        
                        tracing::info!("{}", message);
                        
                        let response = TradingControlResponse {
                            success: true,
                            message,
                            timestamp: state.last_status_change,
                            previous_status,
                            new_status: state.trading_status.clone(),
                        };
                        
                        warp::reply::with_status(
                            warp::reply::json(&response),
                            warp::http::StatusCode::OK
                        )
                    })
            };

            // Quick order endpoint
            let quick_order = warp::path!("orders" / "quick")
                .and(warp::post())
                .and(warp::body::json())
                .map(|req: QuickOrderRequest| {
                    // Validate order
                    if req.symbol.is_empty() {
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Symbol is required"
                            })),
                            warp::http::StatusCode::BAD_REQUEST
                        );
                    }
                    
                    if !["BUY", "SELL"].contains(&req.side.as_str()) {
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Side must be 'BUY' or 'SELL'"
                            })),
                            warp::http::StatusCode::BAD_REQUEST
                        );
                    }
                    
                    if req.quantity <= 0.0 {
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Quantity must be positive"
                            })),
                            warp::http::StatusCode::BAD_REQUEST
                        );
                    }
                    
                    // Generate mock order ID
                    let order_id = format!("ORD_{}", Utc::now().timestamp_millis());
                    
                    tracing::info!(
                        "Quick order submitted: {} {} {} @ {:?}",
                        req.side, req.quantity, req.symbol, req.price
                    );
                    
                    let response = QuickOrderResponse {
                        success: true,
                        message: format!("Quick order submitted: {} {} {}", req.side, req.quantity, req.symbol),
                        order_id: Some(order_id),
                        timestamp: Utc::now(),
                    };
                    
                    warp::reply::with_status(
                        warp::reply::json(&response),
                        warp::http::StatusCode::OK
                    )
                });

            // Force reconnect endpoint
            let force_reconnect = warp::path!("system" / "reconnect")
                .and(warp::post())
                .map(|| {
                    tracing::info!("Force reconnect requested");
                    
                    // TODO: Implement actual reconnection logic
                    let response = serde_json::json!({
                        "success": true,
                        "message": "Reconnection initiated",
                        "timestamp": Utc::now()
                    });
                    
                    warp::reply::json(&response)
                });

            // Combine all routes
            let routes = health
                .or(system_status)
                .or(emergency_halt)
                .or(resume_halt)
                .or(trading_control)
                .or(quick_order)
                .or(force_reconnect)
                .with(cors);

            let addr = ([0, 0, 0, 0], 8001);
            tracing::info!("Execution Core API server listening on {:?}", addr);
            
            warp::serve(routes).run(addr).await;
        })
    };

    // Main execution loop (placeholder for now)
    let main_loop = tokio::spawn(async move {
        loop {
            // Process events, check risk limits, etc.
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            // Example: Check risk limits periodically
            let risk_status = execution_core.check_risk_limits();
            if !risk_status.can_trade() {
                tracing::warn!("Risk limits breached: {:?}", risk_status);
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = api_server => {
            tracing::error!("API server terminated unexpectedly");
        }
        _ = main_loop => {
            tracing::error!("Main loop terminated unexpectedly");
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received shutdown signal");
        }
    }

    tracing::info!("Execution Core shutting down");
    Ok(())
}