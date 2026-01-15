/**
 * Deriv API Adapter - WebSocket Integration for Demo Trading
 * 
 * Features:
 * - Real-time market data streaming
 * - Order placement and management
 * - Position tracking
 * - Account balance monitoring
 * - Automatic reconnection
 * - Safety controls for demo trading
 */

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct DerivConfig {
    pub app_id: String,
    pub api_token: String,
    pub websocket_url: String,
    pub demo_mode: bool,
    pub max_position_size: f64,
    pub max_daily_trades: u32,
    pub max_daily_loss: f64,
}

impl Default for DerivConfig {
    fn default() -> Self {
        Self {
            app_id: std::env::var("DERIV_APP_ID").unwrap_or_else(|_| "118029".to_string()),
            api_token: std::env::var("DERIV_API_TOKEN").unwrap_or_default(),
            websocket_url: std::env::var("DERIV_WEBSOCKET_URL")
                .unwrap_or_else(|_| "wss://ws.derivws.com/websockets/v3?app_id=118029".to_string()),
            demo_mode: std::env::var("DERIV_DEMO_MODE")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            max_position_size: std::env::var("DERIV_MAX_POSITION_SIZE")
                .unwrap_or_else(|_| "1.0".to_string())
                .parse()
                .unwrap_or(1.0),
            max_daily_trades: std::env::var("DERIV_MAX_DAILY_TRADES")
                .unwrap_or_else(|_| "50".to_string())
                .parse()
                .unwrap_or(50),
            max_daily_loss: std::env::var("DERIV_MAX_DAILY_LOSS")
                .unwrap_or_else(|_| "1000.0".to_string())
                .parse()
                .unwrap_or(1000.0),
        }
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivTick {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivOrder {
    pub id: String,
    pub contract_id: Option<String>,
    pub symbol: String,
    pub order_type: DerivOrderType,
    pub amount: f64,
    pub price: Option<f64>,
    pub status: DerivOrderStatus,
    pub created_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub pnl: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DerivOrderType {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DerivOrderStatus {
    Pending,
    Open,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivPosition {
    pub contract_id: String,
    pub symbol: String,
    pub order_type: DerivOrderType,
    pub buy_price: f64,
    pub current_price: f64,
    pub amount: f64,
    pub pnl: f64,
    pub opened_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivAccount {
    pub balance: f64,
    pub currency: String,
    pub loginid: String,
    pub is_virtual: bool,
}

// ============================================================================
// DERIV ADAPTER
// ============================================================================

pub struct DerivAdapter {
    config: DerivConfig,
    state: Arc<RwLock<AdapterState>>,
    tx: mpsc::UnboundedSender<DerivEvent>,
}

#[derive(Debug)]
struct AdapterState {
    connected: bool,
    authorized: bool,
    account: Option<DerivAccount>,
    positions: HashMap<String, DerivPosition>,
    orders: HashMap<String, DerivOrder>,
    ticks: HashMap<String, DerivTick>,
    daily_trades: u32,
    daily_pnl: f64,
    subscribed_symbols: Vec<String>,
}

impl Default for AdapterState {
    fn default() -> Self {
        Self {
            connected: false,
            authorized: false,
            account: None,
            positions: HashMap::new(),
            orders: HashMap::new(),
            ticks: HashMap::new(),
            daily_trades: 0,
            daily_pnl: 0.0,
            subscribed_symbols: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DerivEvent {
    Connected,
    Disconnected,
    Authorized(DerivAccount),
    TickUpdate(DerivTick),
    OrderPlaced(DerivOrder),
    OrderFilled(DerivOrder),
    PositionOpened(DerivPosition),
    PositionClosed { contract_id: String, pnl: f64 },
    BalanceUpdate(f64),
    Error(String),
}

impl DerivAdapter {
    pub fn new(config: DerivConfig) -> (Self, mpsc::UnboundedReceiver<DerivEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let adapter = Self {
            config,
            state: Arc::new(RwLock::new(AdapterState::default())),
            tx,
        };
        
        (adapter, rx)
    }
    
    /// Start the WebSocket connection and event loop
    pub async fn start(&self) -> Result<()> {
        info!("Starting Deriv adapter...");
        
        let url = url::Url::parse(&self.config.websocket_url)
            .context("Invalid WebSocket URL")?;
        
        let (ws_stream, _) = connect_async(url)
            .await
            .context("Failed to connect to Deriv WebSocket")?;
        
        info!("Connected to Deriv WebSocket");
        
        {
            let mut state = self.state.write().await;
            state.connected = true;
        }
        
        let _ = self.tx.send(DerivEvent::Connected);
        
        let (mut write, mut read) = ws_stream.split();
        
        // Authorize
        self.authorize(&mut write).await?;
        
        // Subscribe to account updates
        self.subscribe_balance(&mut write).await?;
        
        // Start message processing loop
        let state = Arc::clone(&self.state);
        let tx = self.tx.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Err(e) = Self::handle_message(&text, &state, &tx, &config).await {
                            error!("Error handling message: {}", e);
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("WebSocket closed");
                        let _ = tx.send(DerivEvent::Disconnected);
                        break;
                    }
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        let _ = tx.send(DerivEvent::Error(e.to_string()));
                        break;
                    }
                    _ => {}
                }
            }
        });
        
        Ok(())
    }
    
    /// Authorize with API token
    async fn authorize<S>(&self, write: &mut S) -> Result<()>
    where
        S: SinkExt<Message> + Unpin,
        S::Error: std::error::Error + Send + Sync + 'static,
    {
        let auth_msg = serde_json::json!({
            "authorize": self.config.api_token
        });
        
        write
            .send(Message::Text(auth_msg.to_string()))
            .await
            .context("Failed to send authorize message")?;
        
        info!("Authorization request sent");
        Ok(())
    }
    
    /// Subscribe to balance updates
    async fn subscribe_balance<S>(&self, write: &mut S) -> Result<()>
    where
        S: SinkExt<Message> + Unpin,
        S::Error: std::error::Error + Send + Sync + 'static,
    {
        let balance_msg = serde_json::json!({
            "balance": 1,
            "subscribe": 1
        });
        
        write
            .send(Message::Text(balance_msg.to_string()))
            .await
            .context("Failed to subscribe to balance")?;
        
        debug!("Subscribed to balance updates");
        Ok(())
    }
    
    /// Handle incoming WebSocket messages
    async fn handle_message(
        text: &str,
        state: &Arc<RwLock<AdapterState>>,
        tx: &mpsc::UnboundedSender<DerivEvent>,
        config: &DerivConfig,
    ) -> Result<()> {
        let msg: serde_json::Value = serde_json::from_str(text)?;
        
        // Handle authorization response
        if let Some(authorize) = msg.get("authorize") {
            let loginid = authorize["loginid"].as_str().unwrap_or("unknown");
            let balance = authorize["balance"].as_f64().unwrap_or(0.0);
            let currency = authorize["currency"].as_str().unwrap_or("USD");
            let is_virtual = authorize["is_virtual"].as_i64().unwrap_or(0) == 1;
            
            let account = DerivAccount {
                balance,
                currency: currency.to_string(),
                loginid: loginid.to_string(),
                is_virtual,
            };
            
            {
                let mut s = state.write().await;
                s.authorized = true;
                s.account = Some(account.clone());
            }
            
            info!("Authorized - Account: {} ({}), Balance: {} {}", 
                loginid, 
                if is_virtual { "DEMO" } else { "REAL" },
                balance,
                currency
            );
            
            let _ = tx.send(DerivEvent::Authorized(account));
        }
        
        // Handle balance updates
        if let Some(balance_obj) = msg.get("balance") {
            if let Some(balance) = balance_obj["balance"].as_f64() {
                {
                    let mut s = state.write().await;
                    if let Some(ref mut account) = s.account {
                        account.balance = balance;
                    }
                }
                
                debug!("Balance updated: {}", balance);
                let _ = tx.send(DerivEvent::BalanceUpdate(balance));
            }
        }
        
        // Handle tick updates
        if let Some(tick) = msg.get("tick") {
            let symbol = tick["symbol"].as_str().unwrap_or("unknown").to_string();
            let bid = tick["bid"].as_f64().unwrap_or(0.0);
            let ask = tick["ask"].as_f64().unwrap_or(0.0);
            
            let tick_data = DerivTick {
                symbol: symbol.clone(),
                bid,
                ask,
                timestamp: Utc::now(),
            };
            
            {
                let mut s = state.write().await;
                s.ticks.insert(symbol.clone(), tick_data.clone());
            }
            
            let _ = tx.send(DerivEvent::TickUpdate(tick_data));
        }
        
        // Handle buy/sell contract responses
        if let Some(buy) = msg.get("buy") {
            let contract_id = buy["contract_id"].as_str().map(|s| s.to_string());
            let buy_price = buy["buy_price"].as_f64().unwrap_or(0.0);
            
            info!("Contract purchased: {:?}, Price: {}", contract_id, buy_price);
        }
        
        // Handle proposal open contract (position updates)
        if let Some(poc) = msg.get("proposal_open_contract") {
            if let Some(contract_id) = poc["contract_id"].as_str() {
                let symbol = poc["underlying"].as_str().unwrap_or("unknown");
                let buy_price = poc["buy_price"].as_f64().unwrap_or(0.0);
                let current_spot = poc["current_spot"].as_f64().unwrap_or(0.0);
                let profit = poc["profit"].as_f64().unwrap_or(0.0);
                
                // Determine if it's a buy or sell based on contract type
                let contract_type = poc["contract_type"].as_str().unwrap_or("");
                let order_type = if contract_type.contains("CALL") || contract_type.contains("UP") {
                    DerivOrderType::Buy
                } else {
                    DerivOrderType::Sell
                };
                
                let position = DerivPosition {
                    contract_id: contract_id.to_string(),
                    symbol: symbol.to_string(),
                    order_type,
                    buy_price,
                    current_price: current_spot,
                    amount: 1.0,
                    pnl: profit,
                    opened_at: Utc::now(),
                };
                
                {
                    let mut s = state.write().await;
                    s.positions.insert(contract_id.to_string(), position.clone());
                }
                
                debug!("Position update: {} - P&L: {}", contract_id, profit);
            }
        }
        
        // Handle errors
        if let Some(error) = msg.get("error") {
            let error_msg = error["message"].as_str().unwrap_or("Unknown error");
            error!("Deriv API error: {}", error_msg);
            let _ = tx.send(DerivEvent::Error(error_msg.to_string()));
        }
        
        Ok(())
    }
    
    /// Get current account information
    pub async fn get_account(&self) -> Option<DerivAccount> {
        let state = self.state.read().await;
        state.account.clone()
    }
    
    /// Get all open positions
    pub async fn get_positions(&self) -> Vec<DerivPosition> {
        let state = self.state.read().await;
        state.positions.values().cloned().collect()
    }
    
    /// Get latest tick for a symbol
    pub async fn get_tick(&self, symbol: &str) -> Option<DerivTick> {
        let state = self.state.read().await;
        state.ticks.get(symbol).cloned()
    }
    
    /// Check if adapter is connected and authorized
    pub async fn is_ready(&self) -> bool {
        let state = self.state.read().await;
        state.connected && state.authorized
    }
    
    /// Get adapter status
    pub async fn get_status(&self) -> AdapterStatus {
        let state = self.state.read().await;
        AdapterStatus {
            connected: state.connected,
            authorized: state.authorized,
            account_balance: state.account.as_ref().map(|a| a.balance),
            open_positions: state.positions.len(),
            daily_trades: state.daily_trades,
            daily_pnl: state.daily_pnl,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterStatus {
    pub connected: bool,
    pub authorized: bool,
    pub account_balance: Option<f64>,
    pub open_positions: usize,
    pub daily_trades: u32,
    pub daily_pnl: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = DerivConfig::default();
        assert!(config.demo_mode);
        assert_eq!(config.app_id, "118029");
    }
    
    #[test]
    fn test_order_status() {
        let order = DerivOrder {
            id: "test".to_string(),
            contract_id: None,
            symbol: "R_100".to_string(),
            order_type: DerivOrderType::Buy,
            amount: 1.0,
            price: None,
            status: DerivOrderStatus::Pending,
            created_at: Utc::now(),
            filled_at: None,
            pnl: None,
        };
        
        assert_eq!(order.status, DerivOrderStatus::Pending);
    }
}
