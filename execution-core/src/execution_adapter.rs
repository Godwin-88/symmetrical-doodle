use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use async_trait::async_trait;
use std::fmt;

use crate::{OrderIntent, OrderSide, OrderType};

/// Normalized order status across different execution venues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "Pending"),
            OrderStatus::PartiallyFilled => write!(f, "PartiallyFilled"),
            OrderStatus::Filled => write!(f, "Filled"),
            OrderStatus::Cancelled => write!(f, "Cancelled"),
            OrderStatus::Rejected => write!(f, "Rejected"),
            OrderStatus::Expired => write!(f, "Expired"),
        }
    }
}

/// Normalized order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,
    pub venue_order_id: String,
    pub asset_id: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub strategy_id: Option<String>,
    pub correlation_id: Option<Uuid>,
}

/// Connection status for execution adapters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Connecting,
    Error(String),
}

/// Execution adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub name: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub rate_limit_per_second: u32,
    pub sandbox_mode: bool,
    pub metadata: HashMap<String, String>,
}

/// Execution statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub orders_placed: u64,
    pub orders_filled: u64,
    pub orders_cancelled: u64,
    pub orders_rejected: u64,
    pub total_volume: f64,
    pub average_latency_ms: f64,
    pub last_activity: DateTime<Utc>,
    pub connection_uptime: f64,
}

/// Normalized execution adapter trait
#[async_trait]
pub trait ExecutionAdapter: Send + Sync {
    /// Get adapter name/identifier
    fn get_name(&self) -> &str;
    
    /// Get current connection status
    fn get_connection_status(&self) -> ConnectionStatus;
    
    /// Get execution statistics
    fn get_stats(&self) -> ExecutionStats;
    
    /// Connect to the execution venue
    async fn connect(&mut self) -> Result<()>;
    
    /// Disconnect from the execution venue
    async fn disconnect(&mut self) -> Result<()>;
    
    /// Place a new order
    async fn place_order(&mut self, intent: OrderIntent) -> Result<Order>;
    
    /// Modify an existing order
    async fn modify_order(&mut self, order_id: Uuid, new_quantity: Option<f64>, new_price: Option<f64>) -> Result<Order>;
    
    /// Cancel an existing order
    async fn cancel_order(&mut self, order_id: Uuid) -> Result<Order>;
    
    /// Get order status
    async fn get_order_status(&mut self, order_id: Uuid) -> Result<Order>;
    
    /// Get all active orders
    async fn get_active_orders(&mut self) -> Result<Vec<Order>>;
    
    /// Health check
    async fn health_check(&mut self) -> Result<bool>;
}

/// Factory for creating execution adapters
pub struct ExecutionAdapterFactory;

impl ExecutionAdapterFactory {
    pub fn create_deriv_adapter(config: AdapterConfig) -> Result<Box<dyn ExecutionAdapter>> {
        Ok(Box::new(DerivAdapter::new(config)?))
    }
    
    pub fn create_shadow_adapter(config: AdapterConfig) -> Result<Box<dyn ExecutionAdapter>> {
        Ok(Box::new(ShadowAdapter::new(config)?))
    }
}

/// Deriv API adapter implementation
pub struct DerivAdapter {
    config: AdapterConfig,
    connection_status: ConnectionStatus,
    orders: HashMap<Uuid, Order>,
    venue_order_map: HashMap<String, Uuid>,
    stats: ExecutionStats,
    client: Option<reqwest::Client>,
}

impl DerivAdapter {
    pub fn new(config: AdapterConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()?;
            
        Ok(Self {
            config,
            connection_status: ConnectionStatus::Disconnected,
            orders: HashMap::new(),
            venue_order_map: HashMap::new(),
            stats: ExecutionStats {
                orders_placed: 0,
                orders_filled: 0,
                orders_cancelled: 0,
                orders_rejected: 0,
                total_volume: 0.0,
                average_latency_ms: 0.0,
                last_activity: Utc::now(),
                connection_uptime: 0.0,
            },
            client: Some(client),
        })
    }
    
    async fn send_deriv_request(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let client = self.client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("HTTP client not initialized"))?;
            
        let start_time = std::time::Instant::now();
        
        let response = client
            .post(&self.config.endpoint)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;
            
        let latency = start_time.elapsed().as_millis() as f64;
        tracing::debug!("Deriv API request latency: {}ms", latency);
        
        if !response.status().is_success() {
            anyhow::bail!("Deriv API error: {}", response.status());
        }
        
        let result: serde_json::Value = response.json().await?;
        
        // Check for API-level errors
        if let Some(error) = result.get("error") {
            anyhow::bail!("Deriv API error: {}", error);
        }
        
        Ok(result)
    }
    
    fn create_order_from_intent(&self, intent: OrderIntent) -> Order {
        Order {
            id: Uuid::new_v4(),
            venue_order_id: String::new(), // Will be set after placement
            asset_id: intent.asset_id,
            side: intent.side,
            quantity: intent.quantity,
            filled_quantity: 0.0,
            order_type: intent.order_type,
            price: intent.price,
            status: OrderStatus::Pending,
            created_at: intent.timestamp,
            updated_at: Utc::now(),
            metadata: intent.metadata,
            strategy_id: intent.strategy_id,
            correlation_id: intent.correlation_id,
        }
    }
    
    fn map_deriv_order_type(&self, order_type: OrderType) -> &'static str {
        match order_type {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::Stop => "stop",
        }
    }
    
    fn map_deriv_side(&self, side: OrderSide) -> &'static str {
        match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        }
    }
}

#[async_trait]
impl ExecutionAdapter for DerivAdapter {
    fn get_name(&self) -> &str {
        &self.config.name
    }
    
    fn get_connection_status(&self) -> ConnectionStatus {
        self.connection_status.clone()
    }
    
    fn get_stats(&self) -> ExecutionStats {
        self.stats.clone()
    }
    
    async fn connect(&mut self) -> Result<()> {
        tracing::info!("Connecting to Deriv API at {}", self.config.endpoint);
        
        self.connection_status = ConnectionStatus::Connecting;
        
        // Test connection with a ping request
        let ping_payload = serde_json::json!({
            "ping": 1
        });
        
        match self.send_deriv_request(ping_payload).await {
            Ok(_) => {
                self.connection_status = ConnectionStatus::Connected;
                tracing::info!("Successfully connected to Deriv API");
                Ok(())
            },
            Err(e) => {
                self.connection_status = ConnectionStatus::Error(e.to_string());
                Err(e)
            }
        }
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        tracing::info!("Disconnecting from Deriv API");
        self.connection_status = ConnectionStatus::Disconnected;
        Ok(())
    }
    
    async fn place_order(&mut self, intent: OrderIntent) -> Result<Order> {
        if !matches!(self.connection_status, ConnectionStatus::Connected) {
            anyhow::bail!("Not connected to Deriv API");
        }
        
        let mut order = self.create_order_from_intent(intent);
        
        // Create Deriv API payload
        let payload = serde_json::json!({
            "buy": 1,
            "parameters": {
                "contract_type": self.map_deriv_order_type(order.order_type),
                "currency": "USD",
                "amount": order.quantity,
                "symbol": order.asset_id,
                "duration": 1,
                "duration_unit": "m",
                "basis": "stake"
            }
        });
        
        match self.send_deriv_request(payload).await {
            Ok(response) => {
                // Extract venue order ID from response
                if let Some(buy_response) = response.get("buy") {
                    if let Some(contract_id) = buy_response.get("contract_id") {
                        order.venue_order_id = contract_id.to_string();
                        order.status = OrderStatus::Pending;
                        order.updated_at = Utc::now();
                        
                        // Store order mappings
                        self.orders.insert(order.id, order.clone());
                        self.venue_order_map.insert(order.venue_order_id.clone(), order.id);
                        
                        // Update stats
                        self.stats.orders_placed += 1;
                        self.stats.total_volume += order.quantity;
                        self.stats.last_activity = Utc::now();
                        
                        tracing::info!("Order placed successfully: {} -> {}", order.id, order.venue_order_id);
                        Ok(order)
                    } else {
                        order.status = OrderStatus::Rejected;
                        self.stats.orders_rejected += 1;
                        anyhow::bail!("No contract_id in Deriv response");
                    }
                } else {
                    order.status = OrderStatus::Rejected;
                    self.stats.orders_rejected += 1;
                    anyhow::bail!("Invalid Deriv API response format");
                }
            },
            Err(e) => {
                order.status = OrderStatus::Rejected;
                self.stats.orders_rejected += 1;
                Err(e)
            }
        }
    }
    
    async fn modify_order(&mut self, order_id: Uuid, new_quantity: Option<f64>, new_price: Option<f64>) -> Result<Order> {
        let order = self.orders.get_mut(&order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
            
        // Deriv API doesn't support order modification in the same way as traditional brokers
        // For binary options, we would need to cancel and replace
        tracing::warn!("Order modification not fully supported by Deriv API, would require cancel/replace");
        
        // Update local order if modifications are provided
        if let Some(quantity) = new_quantity {
            order.quantity = quantity;
        }
        if let Some(price) = new_price {
            order.price = Some(price);
        }
        order.updated_at = Utc::now();
        
        Ok(order.clone())
    }
    
    async fn cancel_order(&mut self, order_id: Uuid) -> Result<Order> {
        let order_venue_id = {
            let order = self.orders.get(&order_id)
                .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
                
            if order.venue_order_id.is_empty() {
                anyhow::bail!("Order has no venue ID, cannot cancel");
            }
            
            order.venue_order_id.clone()
        };
        
        let payload = serde_json::json!({
            "sell": order_venue_id.parse::<u64>().unwrap_or(0)
        });
        
        match self.send_deriv_request(payload).await {
            Ok(_) => {
                let order = self.orders.get_mut(&order_id).unwrap();
                order.status = OrderStatus::Cancelled;
                order.updated_at = Utc::now();
                self.stats.orders_cancelled += 1;
                self.stats.last_activity = Utc::now();
                
                tracing::info!("Order cancelled successfully: {}", order_id);
                Ok(order.clone())
            },
            Err(e) => {
                tracing::error!("Failed to cancel order {}: {}", order_id, e);
                Err(e)
            }
        }
    }
    
    async fn get_order_status(&mut self, order_id: Uuid) -> Result<Order> {
        let order = self.orders.get(&order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
            
        // In a real implementation, we would query Deriv API for current status
        // For now, return the cached order
        Ok(order.clone())
    }
    
    async fn get_active_orders(&mut self) -> Result<Vec<Order>> {
        let active_orders: Vec<Order> = self.orders
            .values()
            .filter(|order| matches!(order.status, OrderStatus::Pending | OrderStatus::PartiallyFilled))
            .cloned()
            .collect();
            
        Ok(active_orders)
    }
    
    async fn health_check(&mut self) -> Result<bool> {
        match self.connection_status {
            ConnectionStatus::Connected => {
                // Send a ping to verify connection
                let ping_payload = serde_json::json!({"ping": 1});
                match self.send_deriv_request(ping_payload).await {
                    Ok(_) => Ok(true),
                    Err(_) => {
                        self.connection_status = ConnectionStatus::Error("Health check failed".to_string());
                        Ok(false)
                    }
                }
            },
            _ => Ok(false)
        }
    }
}

/// Shadow execution adapter for testing without real trades
pub struct ShadowAdapter {
    config: AdapterConfig,
    connection_status: ConnectionStatus,
    orders: HashMap<Uuid, Order>,
    stats: ExecutionStats,
    simulated_latency_ms: u64,
}

impl ShadowAdapter {
    pub fn new(config: AdapterConfig) -> Result<Self> {
        Ok(Self {
            config,
            connection_status: ConnectionStatus::Disconnected,
            orders: HashMap::new(),
            stats: ExecutionStats {
                orders_placed: 0,
                orders_filled: 0,
                orders_cancelled: 0,
                orders_rejected: 0,
                total_volume: 0.0,
                average_latency_ms: 50.0, // Simulated latency
                last_activity: Utc::now(),
                connection_uptime: 0.0,
            },
            simulated_latency_ms: 50,
        })
    }
    
    async fn simulate_latency(&self) {
        tokio::time::sleep(std::time::Duration::from_millis(self.simulated_latency_ms)).await;
    }
    
    fn create_order_from_intent(&self, intent: OrderIntent) -> Order {
        Order {
            id: Uuid::new_v4(),
            venue_order_id: format!("SHADOW_{}", Uuid::new_v4()),
            asset_id: intent.asset_id,
            side: intent.side,
            quantity: intent.quantity,
            filled_quantity: 0.0,
            order_type: intent.order_type,
            price: intent.price,
            status: OrderStatus::Pending,
            created_at: intent.timestamp,
            updated_at: Utc::now(),
            metadata: intent.metadata,
            strategy_id: intent.strategy_id,
            correlation_id: intent.correlation_id,
        }
    }
}

#[async_trait]
impl ExecutionAdapter for ShadowAdapter {
    fn get_name(&self) -> &str {
        &self.config.name
    }
    
    fn get_connection_status(&self) -> ConnectionStatus {
        self.connection_status.clone()
    }
    
    fn get_stats(&self) -> ExecutionStats {
        self.stats.clone()
    }
    
    async fn connect(&mut self) -> Result<()> {
        tracing::info!("Connecting to Shadow Execution (simulation mode)");
        self.simulate_latency().await;
        self.connection_status = ConnectionStatus::Connected;
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        tracing::info!("Disconnecting from Shadow Execution");
        self.connection_status = ConnectionStatus::Disconnected;
        Ok(())
    }
    
    async fn place_order(&mut self, intent: OrderIntent) -> Result<Order> {
        self.simulate_latency().await;
        
        let mut order = self.create_order_from_intent(intent);
        
        // Simulate immediate fill for market orders
        if order.order_type == OrderType::Market {
            order.status = OrderStatus::Filled;
            order.filled_quantity = order.quantity;
            self.stats.orders_filled += 1;
        } else {
            order.status = OrderStatus::Pending;
        }
        
        order.updated_at = Utc::now();
        
        // Store order
        self.orders.insert(order.id, order.clone());
        
        // Update stats
        self.stats.orders_placed += 1;
        self.stats.total_volume += order.quantity;
        self.stats.last_activity = Utc::now();
        
        tracing::info!("Shadow order placed: {} ({})", order.id, order.status);
        Ok(order)
    }
    
    async fn modify_order(&mut self, order_id: Uuid, new_quantity: Option<f64>, new_price: Option<f64>) -> Result<Order> {
        self.simulate_latency().await;
        
        let order = self.orders.get_mut(&order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
            
        if let Some(quantity) = new_quantity {
            order.quantity = quantity;
        }
        if let Some(price) = new_price {
            order.price = Some(price);
        }
        order.updated_at = Utc::now();
        
        tracing::info!("Shadow order modified: {}", order_id);
        Ok(order.clone())
    }
    
    async fn cancel_order(&mut self, order_id: Uuid) -> Result<Order> {
        self.simulate_latency().await;
        
        let order = self.orders.get_mut(&order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
            
        order.status = OrderStatus::Cancelled;
        order.updated_at = Utc::now();
        self.stats.orders_cancelled += 1;
        self.stats.last_activity = Utc::now();
        
        tracing::info!("Shadow order cancelled: {}", order_id);
        Ok(order.clone())
    }
    
    async fn get_order_status(&mut self, order_id: Uuid) -> Result<Order> {
        let order = self.orders.get(&order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found: {}", order_id))?;
            
        Ok(order.clone())
    }
    
    async fn get_active_orders(&mut self) -> Result<Vec<Order>> {
        let active_orders: Vec<Order> = self.orders
            .values()
            .filter(|order| matches!(order.status, OrderStatus::Pending | OrderStatus::PartiallyFilled))
            .cloned()
            .collect();
            
        Ok(active_orders)
    }
    
    async fn health_check(&mut self) -> Result<bool> {
        Ok(matches!(self.connection_status, ConnectionStatus::Connected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_config(name: &str) -> AdapterConfig {
        AdapterConfig {
            name: name.to_string(),
            endpoint: "https://api.deriv.com".to_string(),
            api_key: Some("test_key".to_string()),
            secret_key: Some("test_secret".to_string()),
            timeout_ms: 5000,
            retry_attempts: 3,
            rate_limit_per_second: 10,
            sandbox_mode: true,
            metadata: HashMap::new(),
        }
    }

    fn create_test_order_intent() -> OrderIntent {
        OrderIntent {
            asset_id: "EURUSD".to_string(),
            side: OrderSide::Buy,
            quantity: 1000.0,
            order_type: OrderType::Market,
            price: None,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            strategy_id: Some("test_strategy".to_string()),
            correlation_id: None,
        }
    }

    #[tokio::test]
    async fn test_shadow_adapter_creation() {
        let config = create_test_config("shadow_test");
        let adapter = ShadowAdapter::new(config);
        assert!(adapter.is_ok());
        
        let adapter = adapter.unwrap();
        assert_eq!(adapter.get_name(), "shadow_test");
        assert_eq!(adapter.get_connection_status(), ConnectionStatus::Disconnected);
    }

    #[tokio::test]
    async fn test_shadow_adapter_connection() {
        let config = create_test_config("shadow_test");
        let mut adapter = ShadowAdapter::new(config).unwrap();
        
        let result = adapter.connect().await;
        assert!(result.is_ok());
        assert_eq!(adapter.get_connection_status(), ConnectionStatus::Connected);
        
        let result = adapter.disconnect().await;
        assert!(result.is_ok());
        assert_eq!(adapter.get_connection_status(), ConnectionStatus::Disconnected);
    }

    #[tokio::test]
    async fn test_shadow_adapter_order_placement() {
        let config = create_test_config("shadow_test");
        let mut adapter = ShadowAdapter::new(config).unwrap();
        
        adapter.connect().await.unwrap();
        
        let intent = create_test_order_intent();
        let result = adapter.place_order(intent).await;
        assert!(result.is_ok());
        
        let order = result.unwrap();
        assert_eq!(order.asset_id, "EURUSD");
        assert_eq!(order.quantity, 1000.0);
        assert_eq!(order.status, OrderStatus::Filled); // Market orders fill immediately in shadow mode
        
        let stats = adapter.get_stats();
        assert_eq!(stats.orders_placed, 1);
        assert_eq!(stats.orders_filled, 1);
    }

    #[tokio::test]
    async fn test_shadow_adapter_order_cancellation() {
        let config = create_test_config("shadow_test");
        let mut adapter = ShadowAdapter::new(config).unwrap();
        
        adapter.connect().await.unwrap();
        
        // Place a limit order (won't fill immediately)
        let mut intent = create_test_order_intent();
        intent.order_type = OrderType::Limit;
        intent.price = Some(1.1000);
        
        let order = adapter.place_order(intent).await.unwrap();
        assert_eq!(order.status, OrderStatus::Pending);
        
        // Cancel the order
        let cancelled_order = adapter.cancel_order(order.id).await.unwrap();
        assert_eq!(cancelled_order.status, OrderStatus::Cancelled);
        
        let stats = adapter.get_stats();
        assert_eq!(stats.orders_cancelled, 1);
    }

    #[tokio::test]
    async fn test_shadow_adapter_health_check() {
        let config = create_test_config("shadow_test");
        let mut adapter = ShadowAdapter::new(config).unwrap();
        
        // Health check should fail when disconnected
        let health = adapter.health_check().await.unwrap();
        assert!(!health);
        
        // Health check should pass when connected
        adapter.connect().await.unwrap();
        let health = adapter.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_deriv_adapter_creation() {
        let config = create_test_config("deriv_test");
        let adapter = DerivAdapter::new(config);
        assert!(adapter.is_ok());
        
        let adapter = adapter.unwrap();
        assert_eq!(adapter.get_name(), "deriv_test");
        assert_eq!(adapter.get_connection_status(), ConnectionStatus::Disconnected);
    }

    #[test]
    fn test_execution_adapter_factory() {
        let config = create_test_config("factory_test");
        
        let shadow_adapter = ExecutionAdapterFactory::create_shadow_adapter(config.clone());
        assert!(shadow_adapter.is_ok());
        
        let deriv_adapter = ExecutionAdapterFactory::create_deriv_adapter(config);
        assert!(deriv_adapter.is_ok());
    }
}