use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use anyhow::Result;

use crate::{
    ExecutionAdapter, ExecutionAdapterFactory, AdapterConfig, Order, OrderIntent, 
    ExecutionCore, ExecutionCoreImpl, Config, Fill, LiquidityFlag
};

/// Execution manager that coordinates between adapters and the execution core
pub struct ExecutionManager {
    execution_core: Arc<RwLock<ExecutionCoreImpl>>,
    adapters: HashMap<String, Arc<RwLock<Box<dyn ExecutionAdapter>>>>,
    primary_adapter: Option<String>,
    shadow_mode: bool,
}

impl ExecutionManager {
    pub fn new(config: &Config) -> Result<Self> {
        let execution_core = ExecutionCoreImpl::new(config)?;
        
        Ok(Self {
            execution_core: Arc::new(RwLock::new(execution_core)),
            adapters: HashMap::new(),
            primary_adapter: None,
            shadow_mode: false,
        })
    }
    
    /// Add an execution adapter
    pub async fn add_adapter(&mut self, name: String, adapter: Box<dyn ExecutionAdapter>) -> Result<()> {
        let adapter = Arc::new(RwLock::new(adapter));
        self.adapters.insert(name.clone(), adapter);
        
        // Set as primary if it's the first adapter
        if self.primary_adapter.is_none() {
            self.primary_adapter = Some(name);
        }
        
        Ok(())
    }
    
    /// Add Deriv adapter with configuration
    pub async fn add_deriv_adapter(&mut self, name: String, config: AdapterConfig) -> Result<()> {
        let adapter = ExecutionAdapterFactory::create_deriv_adapter(config)?;
        self.add_adapter(name, adapter).await
    }
    
    /// Add shadow adapter with configuration
    pub async fn add_shadow_adapter(&mut self, name: String, config: AdapterConfig) -> Result<()> {
        let adapter = ExecutionAdapterFactory::create_shadow_adapter(config)?;
        self.add_adapter(name, adapter).await
    }
    
    /// Set primary adapter
    pub fn set_primary_adapter(&mut self, name: String) -> Result<()> {
        if !self.adapters.contains_key(&name) {
            anyhow::bail!("Adapter '{}' not found", name);
        }
        self.primary_adapter = Some(name);
        Ok(())
    }
    
    /// Enable or disable shadow mode
    pub fn set_shadow_mode(&mut self, enabled: bool) {
        self.shadow_mode = enabled;
    }
    
    /// Connect all adapters
    pub async fn connect_all(&mut self) -> Result<()> {
        for (name, adapter) in &self.adapters {
            let mut adapter = adapter.write().await;
            match adapter.connect().await {
                Ok(_) => tracing::info!("Connected adapter: {}", name),
                Err(e) => tracing::error!("Failed to connect adapter {}: {}", name, e),
            }
        }
        Ok(())
    }
    
    /// Disconnect all adapters
    pub async fn disconnect_all(&mut self) -> Result<()> {
        for (name, adapter) in &self.adapters {
            let mut adapter = adapter.write().await;
            match adapter.disconnect().await {
                Ok(_) => tracing::info!("Disconnected adapter: {}", name),
                Err(e) => tracing::error!("Failed to disconnect adapter {}: {}", name, e),
            }
        }
        Ok(())
    }
    
    /// Process order intent through the execution pipeline
    pub async fn process_order_intent(&mut self, intent: OrderIntent) -> Result<Uuid> {
        // First, process through execution core for risk checks and validation
        let order_id = {
            let mut core = self.execution_core.write().await;
            core.process_order_intent(intent.clone())?
        };
        
        // If in shadow mode, route to shadow adapter
        if self.shadow_mode {
            if let Some(shadow_adapter) = self.adapters.get("shadow") {
                let mut adapter = shadow_adapter.write().await;
                match adapter.place_order(intent).await {
                    Ok(order) => {
                        tracing::info!("Shadow order placed: {}", order.id);
                        // Convert to fill for portfolio update
                        if order.filled_quantity > 0.0 {
                            let fill = self.order_to_fill(&order);
                            let mut core = self.execution_core.write().await;
                            core.update_position(fill)?;
                        }
                    },
                    Err(e) => tracing::error!("Shadow order failed: {}", e),
                }
            }
        } else {
            // Route to primary adapter for live execution
            if let Some(primary_name) = &self.primary_adapter.clone() {
                if let Some(primary_adapter) = self.adapters.get(primary_name) {
                    let mut adapter = primary_adapter.write().await;
                    match adapter.place_order(intent).await {
                        Ok(order) => {
                            tracing::info!("Live order placed: {} via {}", order.id, primary_name);
                            // Convert to fill for portfolio update
                            if order.filled_quantity > 0.0 {
                                let fill = self.order_to_fill(&order);
                                let mut core = self.execution_core.write().await;
                                core.update_position(fill)?;
                            }
                        },
                        Err(e) => {
                            tracing::error!("Live order failed: {}", e);
                            return Err(e);
                        }
                    }
                } else {
                    anyhow::bail!("Primary adapter '{}' not found", primary_name);
                }
            } else {
                anyhow::bail!("No primary adapter configured");
            }
        }
        
        Ok(order_id)
    }
    
    /// Cancel order through appropriate adapter
    pub async fn cancel_order(&mut self, order_id: Uuid) -> Result<()> {
        let adapter_name = if self.shadow_mode {
            "shadow"
        } else {
            self.primary_adapter.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No primary adapter configured"))?
        };
        
        if let Some(adapter) = self.adapters.get(adapter_name) {
            let mut adapter = adapter.write().await;
            adapter.cancel_order(order_id).await?;
            tracing::info!("Order cancelled: {} via {}", order_id, adapter_name);
        } else {
            anyhow::bail!("Adapter '{}' not found", adapter_name);
        }
        
        Ok(())
    }
    
    /// Get order status from appropriate adapter
    pub async fn get_order_status(&self, order_id: Uuid) -> Result<Order> {
        let adapter_name = if self.shadow_mode {
            "shadow"
        } else {
            self.primary_adapter.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No primary adapter configured"))?
        };
        
        if let Some(adapter) = self.adapters.get(adapter_name) {
            let mut adapter = adapter.write().await;
            adapter.get_order_status(order_id).await
        } else {
            anyhow::bail!("Adapter '{}' not found", adapter_name);
        }
    }
    
    /// Get all active orders from all adapters
    pub async fn get_all_active_orders(&self) -> Result<HashMap<String, Vec<Order>>> {
        let mut all_orders = HashMap::new();
        
        for (name, adapter) in &self.adapters {
            let mut adapter = adapter.write().await;
            match adapter.get_active_orders().await {
                Ok(orders) => {
                    all_orders.insert(name.clone(), orders);
                },
                Err(e) => {
                    tracing::error!("Failed to get orders from adapter {}: {}", name, e);
                    all_orders.insert(name.clone(), Vec::new());
                }
            }
        }
        
        Ok(all_orders)
    }
    
    /// Perform health check on all adapters
    pub async fn health_check_all(&self) -> Result<HashMap<String, bool>> {
        let mut health_status = HashMap::new();
        
        for (name, adapter) in &self.adapters {
            let mut adapter = adapter.write().await;
            match adapter.health_check().await {
                Ok(healthy) => {
                    health_status.insert(name.clone(), healthy);
                },
                Err(_) => {
                    health_status.insert(name.clone(), false);
                }
            }
        }
        
        Ok(health_status)
    }
    
    /// Emergency halt all trading activity
    pub async fn emergency_halt(&mut self) -> Result<()> {
        // Halt execution core first
        {
            let mut core = self.execution_core.write().await;
            core.emergency_halt()?;
        }
        
        // Cancel all active orders across all adapters
        for (name, adapter) in &self.adapters {
            let mut adapter = adapter.write().await;
            match adapter.get_active_orders().await {
                Ok(orders) => {
                    for order in orders {
                        if let Err(e) = adapter.cancel_order(order.id).await {
                            tracing::error!("Failed to cancel order {} on adapter {}: {}", 
                                          order.id, name, e);
                        }
                    }
                },
                Err(e) => tracing::error!("Failed to get active orders from adapter {}: {}", name, e),
            }
        }
        
        tracing::warn!("Emergency halt completed across all adapters");
        Ok(())
    }
    
    /// Get execution statistics from all adapters
    pub async fn get_execution_stats(&self) -> Result<HashMap<String, crate::ExecutionStats>> {
        let mut stats = HashMap::new();
        
        for (name, adapter) in &self.adapters {
            let adapter = adapter.read().await;
            stats.insert(name.clone(), adapter.get_stats());
        }
        
        Ok(stats)
    }
    
    /// Convert Order to Fill for portfolio updates
    fn order_to_fill(&self, order: &Order) -> Fill {
        Fill {
            order_id: order.id,
            asset_id: order.asset_id.clone(),
            side: order.side.clone(),
            quantity: order.filled_quantity,
            price: order.price.unwrap_or(0.0), // This should be the actual fill price
            timestamp: order.updated_at,
            commission: 0.0, // Should be calculated based on venue rules
            metadata: order.metadata.clone(),
            execution_venue: "deriv".to_string(), // Should be dynamic based on adapter
            liquidity_flag: LiquidityFlag::Taker, // Should be determined by order type
            slippage: 0.0, // Should be calculated
        }
    }
    
    /// Get reference to execution core for direct access
    pub fn get_execution_core(&self) -> Arc<RwLock<ExecutionCoreImpl>> {
        self.execution_core.clone()
    }
    
    /// Check if system is in shadow mode
    pub fn is_shadow_mode(&self) -> bool {
        self.shadow_mode
    }
    
    /// Get list of available adapters
    pub fn get_adapter_names(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
    
    /// Get primary adapter name
    pub fn get_primary_adapter(&self) -> Option<String> {
        self.primary_adapter.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, AdapterConfig};
    use std::collections::HashMap;

    fn create_test_config() -> Config {
        Config::default()
    }

    fn create_test_adapter_config(name: &str) -> AdapterConfig {
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
            side: crate::OrderSide::Buy,
            quantity: 1000.0,
            order_type: crate::OrderType::Market,
            price: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            strategy_id: Some("test_strategy".to_string()),
            correlation_id: None,
        }
    }

    #[tokio::test]
    async fn test_execution_manager_creation() {
        let config = create_test_config();
        let manager = ExecutionManager::new(&config);
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert!(manager.get_primary_adapter().is_none());
        assert!(!manager.is_shadow_mode());
        assert_eq!(manager.get_adapter_names().len(), 0);
    }

    #[tokio::test]
    async fn test_add_shadow_adapter() {
        let config = create_test_config();
        let mut manager = ExecutionManager::new(&config).unwrap();
        
        let adapter_config = create_test_adapter_config("shadow");
        let result = manager.add_shadow_adapter("shadow".to_string(), adapter_config).await;
        assert!(result.is_ok());
        
        assert_eq!(manager.get_adapter_names().len(), 1);
        assert_eq!(manager.get_primary_adapter(), Some("shadow".to_string()));
    }

    #[tokio::test]
    async fn test_shadow_mode_execution() {
        let config = create_test_config();
        let mut manager = ExecutionManager::new(&config).unwrap();
        
        // Add shadow adapter
        let adapter_config = create_test_adapter_config("shadow");
        manager.add_shadow_adapter("shadow".to_string(), adapter_config).await.unwrap();
        
        // Enable shadow mode
        manager.set_shadow_mode(true);
        assert!(manager.is_shadow_mode());
        
        // Connect adapters
        manager.connect_all().await.unwrap();
        
        // Process order intent
        let intent = create_test_order_intent();
        let result = manager.process_order_intent(intent).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_all() {
        let config = create_test_config();
        let mut manager = ExecutionManager::new(&config).unwrap();
        
        // Add shadow adapter
        let adapter_config = create_test_adapter_config("shadow");
        manager.add_shadow_adapter("shadow".to_string(), adapter_config).await.unwrap();
        
        // Connect adapters
        manager.connect_all().await.unwrap();
        
        // Perform health check
        let health_status = manager.health_check_all().await.unwrap();
        assert_eq!(health_status.len(), 1);
        assert_eq!(health_status.get("shadow"), Some(&true));
    }

    #[tokio::test]
    async fn test_emergency_halt() {
        let config = create_test_config();
        let mut manager = ExecutionManager::new(&config).unwrap();
        
        // Add shadow adapter
        let adapter_config = create_test_adapter_config("shadow");
        manager.add_shadow_adapter("shadow".to_string(), adapter_config).await.unwrap();
        
        // Connect adapters
        manager.connect_all().await.unwrap();
        
        // Emergency halt should succeed
        let result = manager.emergency_halt().await;
        assert!(result.is_ok());
        
        // Verify execution core is halted
        let core = manager.get_execution_core();
        let core = core.read().await;
        assert!(core.is_halted);
    }

    #[tokio::test]
    async fn test_execution_stats() {
        let config = create_test_config();
        let mut manager = ExecutionManager::new(&config).unwrap();
        
        // Add shadow adapter
        let adapter_config = create_test_adapter_config("shadow");
        manager.add_shadow_adapter("shadow".to_string(), adapter_config).await.unwrap();
        
        // Get execution stats
        let stats = manager.get_execution_stats().await.unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats.contains_key("shadow"));
        
        let shadow_stats = stats.get("shadow").unwrap();
        assert_eq!(shadow_stats.orders_placed, 0);
    }
}