use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{
    ExecutionAdapter, Order, OrderIntent, OrderStatus, Portfolio, PortfolioSnapshot,
    Fill, OrderSide, LiquidityFlag
};

/// Shadow execution state comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateComparison {
    pub timestamp: DateTime<Utc>,
    pub live_portfolio: PortfolioSnapshot,
    pub shadow_portfolio: PortfolioSnapshot,
    pub differences: Vec<StateDifference>,
    pub is_synchronized: bool,
    pub drift_percentage: f64,
}

/// Difference between live and shadow states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDifference {
    pub field: String,
    pub live_value: String,
    pub shadow_value: String,
    pub difference: f64,
    pub severity: DifferenceSeverity,
}

/// Severity level of state differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DifferenceSeverity {
    Info,
    Warning,
    Critical,
}

/// Shadow execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    pub enabled: bool,
    pub sync_interval_ms: u64,
    pub max_drift_percentage: f64,
    pub alert_on_drift: bool,
    pub auto_reconcile: bool,
    pub validation_rules: ValidationRules,
}

/// Validation rules for shadow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    pub max_position_drift: f64,
    pub max_pnl_drift: f64,
    pub max_cash_drift: f64,
    pub require_order_matching: bool,
    pub validate_fills: bool,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval_ms: 1000,
            max_drift_percentage: 0.01, // 1%
            alert_on_drift: true,
            auto_reconcile: false,
            validation_rules: ValidationRules {
                max_position_drift: 0.001, // 0.1%
                max_pnl_drift: 0.01, // 1%
                max_cash_drift: 0.001, // 0.1%
                require_order_matching: true,
                validate_fills: true,
            },
        }
    }
}

/// Shadow execution manager for testing without real trades
pub struct ShadowExecutionManager {
    config: ShadowConfig,
    live_portfolio: Arc<RwLock<Portfolio>>,
    shadow_portfolio: Arc<RwLock<Portfolio>>,
    live_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>>,
    shadow_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>>,
    order_mapping: HashMap<Uuid, Uuid>, // live_order_id -> shadow_order_id
    last_sync: DateTime<Utc>,
    state_history: Vec<StateComparison>,
}

impl ShadowExecutionManager {
    pub fn new(
        config: ShadowConfig,
        live_portfolio: Arc<RwLock<Portfolio>>,
        shadow_portfolio: Arc<RwLock<Portfolio>>,
        live_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>>,
        shadow_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>>,
    ) -> Self {
        Self {
            config,
            live_portfolio,
            shadow_portfolio,
            live_adapter,
            shadow_adapter,
            order_mapping: HashMap::new(),
            last_sync: Utc::now(),
            state_history: Vec::new(),
        }
    }
    
    /// Process order intent in both live and shadow modes
    pub async fn process_order_intent(&mut self, intent: OrderIntent) -> Result<(Uuid, Uuid)> {
        let live_order_id;
        let shadow_order_id;
        
        if self.config.enabled {
            // Place order in both live and shadow adapters
            let live_order = {
                let mut adapter = self.live_adapter.write().await;
                adapter.place_order(intent.clone()).await?
            };
            
            let shadow_order = {
                let mut adapter = self.shadow_adapter.write().await;
                adapter.place_order(intent).await?
            };
            
            live_order_id = live_order.id;
            shadow_order_id = shadow_order.id;
            
            // Map the orders for tracking
            self.order_mapping.insert(live_order_id, shadow_order_id);
            
            // Update portfolios if orders are filled
            if live_order.filled_quantity > 0.0 {
                let fill = self.order_to_fill(&live_order);
                let mut portfolio = self.live_portfolio.write().await;
                portfolio.update_position(fill)?;
            }
            
            if shadow_order.filled_quantity > 0.0 {
                let fill = self.order_to_fill(&shadow_order);
                let mut portfolio = self.shadow_portfolio.write().await;
                portfolio.update_position(fill)?;
            }
            
            tracing::info!("Order processed in both live ({}) and shadow ({}) modes", 
                         live_order_id, shadow_order_id);
        } else {
            // Only process in live mode
            let live_order = {
                let mut adapter = self.live_adapter.write().await;
                adapter.place_order(intent).await?
            };
            
            live_order_id = live_order.id;
            shadow_order_id = Uuid::nil(); // No shadow order
            
            if live_order.filled_quantity > 0.0 {
                let fill = self.order_to_fill(&live_order);
                let mut portfolio = self.live_portfolio.write().await;
                portfolio.update_position(fill)?;
            }
        }
        
        Ok((live_order_id, shadow_order_id))
    }
    
    /// Cancel order in both live and shadow modes
    pub async fn cancel_order(&mut self, live_order_id: Uuid) -> Result<()> {
        // Cancel live order
        {
            let mut adapter = self.live_adapter.write().await;
            adapter.cancel_order(live_order_id).await?;
        }
        
        // Cancel corresponding shadow order if it exists
        if let Some(shadow_order_id) = self.order_mapping.get(&live_order_id) {
            let mut adapter = self.shadow_adapter.write().await;
            adapter.cancel_order(*shadow_order_id).await?;
        }
        
        tracing::info!("Order cancelled in both live and shadow modes: {}", live_order_id);
        Ok(())
    }
    
    /// Compare live and shadow portfolio states
    pub async fn compare_states(&mut self) -> Result<StateComparison> {
        let live_snapshot = {
            let portfolio = self.live_portfolio.read().await;
            portfolio.create_snapshot()
        };
        
        let shadow_snapshot = {
            let portfolio = self.shadow_portfolio.read().await;
            portfolio.create_snapshot()
        };
        
        let differences = self.calculate_differences(&live_snapshot, &shadow_snapshot);
        let drift_percentage = self.calculate_drift_percentage(&differences);
        let is_synchronized = drift_percentage <= self.config.max_drift_percentage;
        
        let comparison = StateComparison {
            timestamp: Utc::now(),
            live_portfolio: live_snapshot,
            shadow_portfolio: shadow_snapshot,
            differences,
            is_synchronized,
            drift_percentage,
        };
        
        // Store in history
        self.state_history.push(comparison.clone());
        
        // Keep only last 100 comparisons
        if self.state_history.len() > 100 {
            self.state_history.remove(0);
        }
        
        // Alert if drift is too high
        if self.config.alert_on_drift && !is_synchronized {
            tracing::warn!("Shadow execution drift detected: {:.2}% (max: {:.2}%)", 
                         drift_percentage * 100.0, self.config.max_drift_percentage * 100.0);
        }
        
        self.last_sync = Utc::now();
        Ok(comparison)
    }
    
    /// Validate shadow execution against live execution
    pub async fn validate_execution(&self) -> Result<ValidationResult> {
        let live_orders = {
            let mut adapter = self.live_adapter.write().await;
            adapter.get_active_orders().await?
        };
        
        let shadow_orders = {
            let mut adapter = self.shadow_adapter.write().await;
            adapter.get_active_orders().await?
        };
        
        let mut validation_errors = Vec::new();
        
        // Validate order count matching
        if self.config.validation_rules.require_order_matching {
            if live_orders.len() != shadow_orders.len() {
                validation_errors.push(ValidationError {
                    error_type: ValidationErrorType::OrderCountMismatch,
                    description: format!("Live orders: {}, Shadow orders: {}", 
                                       live_orders.len(), shadow_orders.len()),
                    severity: DifferenceSeverity::Warning,
                });
            }
        }
        
        // Validate individual orders
        for live_order in &live_orders {
            if let Some(shadow_order_id) = self.order_mapping.get(&live_order.id) {
                if let Some(shadow_order) = shadow_orders.iter().find(|o| o.id == *shadow_order_id) {
                    // Compare order details
                    if live_order.asset_id != shadow_order.asset_id {
                        validation_errors.push(ValidationError {
                            error_type: ValidationErrorType::OrderMismatch,
                            description: format!("Asset mismatch: {} vs {}", 
                                               live_order.asset_id, shadow_order.asset_id),
                            severity: DifferenceSeverity::Critical,
                        });
                    }
                    
                    if (live_order.quantity - shadow_order.quantity).abs() > 0.001 {
                        validation_errors.push(ValidationError {
                            error_type: ValidationErrorType::OrderMismatch,
                            description: format!("Quantity mismatch: {} vs {}", 
                                               live_order.quantity, shadow_order.quantity),
                            severity: DifferenceSeverity::Warning,
                        });
                    }
                }
            }
        }
        
        let is_valid = validation_errors.is_empty() || 
                      validation_errors.iter().all(|e| e.severity != DifferenceSeverity::Critical);
        
        Ok(ValidationResult {
            is_valid,
            errors: validation_errors,
            timestamp: Utc::now(),
        })
    }
    
    /// Reconcile shadow state with live state
    pub async fn reconcile_states(&mut self) -> Result<()> {
        if !self.config.auto_reconcile {
            return Ok(());
        }
        
        let comparison = self.compare_states().await?;
        
        if !comparison.is_synchronized {
            tracing::info!("Reconciling shadow state with live state");
            
            // Copy live portfolio state to shadow portfolio
            let live_snapshot = comparison.live_portfolio;
            let mut shadow_portfolio = self.shadow_portfolio.write().await;
            
            // This is a simplified reconciliation - in practice, you'd need more sophisticated logic
            tracing::warn!("State reconciliation not fully implemented - would require portfolio state copying");
        }
        
        Ok(())
    }
    
    /// Get shadow execution statistics
    pub fn get_statistics(&self) -> ShadowExecutionStats {
        let total_comparisons = self.state_history.len();
        let synchronized_count = self.state_history.iter()
            .filter(|c| c.is_synchronized)
            .count();
        
        let average_drift = if !self.state_history.is_empty() {
            self.state_history.iter()
                .map(|c| c.drift_percentage)
                .sum::<f64>() / self.state_history.len() as f64
        } else {
            0.0
        };
        
        let max_drift = self.state_history.iter()
            .map(|c| c.drift_percentage)
            .fold(0.0, f64::max);
        
        ShadowExecutionStats {
            enabled: self.config.enabled,
            total_comparisons,
            synchronized_count,
            synchronization_rate: if total_comparisons > 0 {
                synchronized_count as f64 / total_comparisons as f64
            } else {
                1.0
            },
            average_drift_percentage: average_drift,
            max_drift_percentage: max_drift,
            last_sync: self.last_sync,
            order_mappings: self.order_mapping.len(),
        }
    }
    
    /// Get state comparison history
    pub fn get_state_history(&self) -> &Vec<StateComparison> {
        &self.state_history
    }
    
    /// Enable or disable shadow execution
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
        tracing::info!("Shadow execution {}", if enabled { "enabled" } else { "disabled" });
    }
    
    /// Update shadow configuration
    pub fn update_config(&mut self, config: ShadowConfig) {
        self.config = config;
        tracing::info!("Shadow execution configuration updated");
    }
    
    fn calculate_differences(&self, live: &PortfolioSnapshot, shadow: &PortfolioSnapshot) -> Vec<StateDifference> {
        let mut differences = Vec::new();
        
        // Compare cash balance
        let cash_diff = (live.cash_balance - shadow.cash_balance).abs();
        if cash_diff > self.config.validation_rules.max_cash_drift {
            differences.push(StateDifference {
                field: "cash_balance".to_string(),
                live_value: live.cash_balance.to_string(),
                shadow_value: shadow.cash_balance.to_string(),
                difference: cash_diff,
                severity: if cash_diff > live.cash_balance * 0.01 {
                    DifferenceSeverity::Critical
                } else {
                    DifferenceSeverity::Warning
                },
            });
        }
        
        // Compare total P&L
        let pnl_diff = (live.total_pnl - shadow.total_pnl).abs();
        if pnl_diff > self.config.validation_rules.max_pnl_drift {
            differences.push(StateDifference {
                field: "total_pnl".to_string(),
                live_value: live.total_pnl.to_string(),
                shadow_value: shadow.total_pnl.to_string(),
                difference: pnl_diff,
                severity: if pnl_diff > live.cash_balance * 0.01 {
                    DifferenceSeverity::Critical
                } else {
                    DifferenceSeverity::Warning
                },
            });
        }
        
        // Compare exposures
        let net_exposure_diff = (live.net_exposure - shadow.net_exposure).abs();
        let gross_exposure_diff = (live.gross_exposure - shadow.gross_exposure).abs();
        
        if net_exposure_diff > self.config.validation_rules.max_position_drift {
            differences.push(StateDifference {
                field: "net_exposure".to_string(),
                live_value: live.net_exposure.to_string(),
                shadow_value: shadow.net_exposure.to_string(),
                difference: net_exposure_diff,
                severity: DifferenceSeverity::Warning,
            });
        }
        
        if gross_exposure_diff > self.config.validation_rules.max_position_drift {
            differences.push(StateDifference {
                field: "gross_exposure".to_string(),
                live_value: live.gross_exposure.to_string(),
                shadow_value: shadow.gross_exposure.to_string(),
                difference: gross_exposure_diff,
                severity: DifferenceSeverity::Warning,
            });
        }
        
        differences
    }
    
    fn calculate_drift_percentage(&self, differences: &[StateDifference]) -> f64 {
        if differences.is_empty() {
            return 0.0;
        }
        
        // Calculate weighted drift based on severity
        let total_weight: f64 = differences.iter()
            .map(|d| match d.severity {
                DifferenceSeverity::Critical => 3.0,
                DifferenceSeverity::Warning => 2.0,
                DifferenceSeverity::Info => 1.0,
            })
            .sum();
        
        let weighted_drift: f64 = differences.iter()
            .map(|d| {
                let weight = match d.severity {
                    DifferenceSeverity::Critical => 3.0,
                    DifferenceSeverity::Warning => 2.0,
                    DifferenceSeverity::Info => 1.0,
                };
                // Normalize difference as percentage
                let normalized_diff = d.difference / (d.live_value.parse::<f64>().unwrap_or(1.0).abs() + 1.0);
                weight * normalized_diff
            })
            .sum();
        
        if total_weight > 0.0 {
            weighted_drift / total_weight
        } else {
            0.0
        }
    }
    
    fn order_to_fill(&self, order: &Order) -> Fill {
        Fill {
            order_id: order.id,
            asset_id: order.asset_id.clone(),
            side: order.side.clone(),
            quantity: order.filled_quantity,
            price: order.price.unwrap_or(0.0),
            timestamp: order.updated_at,
            commission: 0.0, // Should be calculated
            metadata: order.metadata.clone(),
            execution_venue: "shadow".to_string(),
            liquidity_flag: LiquidityFlag::Taker,
            slippage: 0.0,
        }
    }
}

/// Shadow execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowExecutionStats {
    pub enabled: bool,
    pub total_comparisons: usize,
    pub synchronized_count: usize,
    pub synchronization_rate: f64,
    pub average_drift_percentage: f64,
    pub max_drift_percentage: f64,
    pub last_sync: DateTime<Utc>,
    pub order_mappings: usize,
}

/// Validation result for shadow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub timestamp: DateTime<Utc>,
}

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub description: String,
    pub severity: DifferenceSeverity,
}

/// Types of validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationErrorType {
    OrderCountMismatch,
    OrderMismatch,
    PositionMismatch,
    PnLMismatch,
    CashMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShadowAdapter, AdapterConfig, Portfolio};
    use std::collections::HashMap;

    fn create_test_shadow_config() -> ShadowConfig {
        ShadowConfig {
            enabled: true,
            sync_interval_ms: 100,
            max_drift_percentage: 0.01,
            alert_on_drift: false, // Disable for tests
            auto_reconcile: false,
            validation_rules: ValidationRules {
                max_position_drift: 0.001,
                max_pnl_drift: 0.01,
                max_cash_drift: 0.001,
                require_order_matching: true,
                validate_fills: true,
            },
        }
    }

    fn create_test_adapter_config(name: &str) -> AdapterConfig {
        AdapterConfig {
            name: name.to_string(),
            endpoint: "https://test.com".to_string(),
            api_key: None,
            secret_key: None,
            timeout_ms: 1000,
            retry_attempts: 1,
            rate_limit_per_second: 10,
            sandbox_mode: true,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_shadow_execution_manager_creation() {
        let config = create_test_shadow_config();
        let live_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        let shadow_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        
        let live_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("live")).unwrap())
        ));
        let shadow_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("shadow")).unwrap())
        ));
        
        let manager = ShadowExecutionManager::new(
            config,
            live_portfolio,
            shadow_portfolio,
            live_adapter,
            shadow_adapter,
        );
        
        assert!(manager.config.enabled);
        assert_eq!(manager.order_mapping.len(), 0);
    }

    #[tokio::test]
    async fn test_shadow_execution_state_comparison() {
        let config = create_test_shadow_config();
        let live_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        let shadow_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        
        let live_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("live")).unwrap())
        ));
        let shadow_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("shadow")).unwrap())
        ));
        
        let mut manager = ShadowExecutionManager::new(
            config,
            live_portfolio,
            shadow_portfolio,
            live_adapter,
            shadow_adapter,
        );
        
        // Connect adapters
        {
            let mut live = manager.live_adapter.write().await;
            live.connect().await.unwrap();
        }
        {
            let mut shadow = manager.shadow_adapter.write().await;
            shadow.connect().await.unwrap();
        }
        
        // Compare initial states (should be synchronized)
        let comparison = manager.compare_states().await.unwrap();
        assert!(comparison.is_synchronized);
        assert_eq!(comparison.differences.len(), 0);
        assert_eq!(comparison.drift_percentage, 0.0);
    }

    #[tokio::test]
    async fn test_shadow_execution_statistics() {
        let config = create_test_shadow_config();
        let live_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        let shadow_portfolio = Arc::new(RwLock::new(Portfolio::new()));
        
        let live_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("live")).unwrap())
        ));
        let shadow_adapter: Arc<RwLock<Box<dyn ExecutionAdapter>>> = Arc::new(RwLock::new(
            Box::new(ShadowAdapter::new(create_test_adapter_config("shadow")).unwrap())
        ));
        
        let mut manager = ShadowExecutionManager::new(
            config,
            live_portfolio,
            shadow_portfolio,
            live_adapter,
            shadow_adapter,
        );
        
        // Connect adapters
        {
            let mut live = manager.live_adapter.write().await;
            live.connect().await.unwrap();
        }
        {
            let mut shadow = manager.shadow_adapter.write().await;
            shadow.connect().await.unwrap();
        }
        
        // Perform a comparison to generate statistics
        manager.compare_states().await.unwrap();
        
        let stats = manager.get_statistics();
        assert!(stats.enabled);
        assert_eq!(stats.total_comparisons, 1);
        assert_eq!(stats.synchronized_count, 1);
        assert_eq!(stats.synchronization_rate, 1.0);
        assert_eq!(stats.order_mappings, 0);
    }

    #[test]
    fn test_shadow_config_default() {
        let config = ShadowConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sync_interval_ms, 1000);
        assert_eq!(config.max_drift_percentage, 0.01);
        assert!(config.alert_on_drift);
        assert!(!config.auto_reconcile);
    }
}