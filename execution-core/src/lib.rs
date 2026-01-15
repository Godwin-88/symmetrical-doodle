pub mod config;
pub mod deriv_adapter;
pub mod event_bus;
pub mod execution_adapter;
pub mod execution_manager;
pub mod health;
pub mod integration_example;
pub mod market_data;
pub mod portfolio;
pub mod risk;
pub mod shadow_execution;
pub mod shutdown;

#[cfg(test)]
pub mod health_property_test;

pub use config::{Config, RiskLimits};
pub use event_bus::{EventBus, Event, EventHandler, EventEnvelope, EventStore, InMemoryEventStore};
pub use execution_adapter::{ExecutionAdapter, ExecutionAdapterFactory, DerivAdapter, ShadowAdapter, Order, OrderStatus, AdapterConfig, ConnectionStatus, ExecutionStats};
pub use execution_manager::ExecutionManager;
pub use market_data::{MarketDataCollector, MarketTick, OrderBookSnapshot, MarketMicrostructure, LiquidityMetrics, MarketEvent};
pub use portfolio::{Portfolio, Position, OrderIntent, Fill, OrderSide, OrderType, LiquidityFlag, PortfolioSnapshot, PortfolioStore, InMemoryPortfolioStore};
pub use risk::{RiskManager, RiskStatus, RiskAlert, RiskAlertType, RiskAlertHandler, InMemoryAlertHandler, RiskMetrics};
pub use shadow_execution::{ShadowExecutionManager, ShadowConfig, StateComparison, StateDifference, DifferenceSeverity, ValidationResult, ValidationError, ShadowExecutionStats};

use anyhow::Result;
use uuid::Uuid;

/// Core execution interface for the trading system
pub trait ExecutionCore {
    /// Process an order intent and return an order ID
    fn process_order_intent(&mut self, intent: OrderIntent) -> Result<Uuid>;
    
    /// Update position based on a fill
    fn update_position(&mut self, fill: Fill) -> Result<()>;
    
    /// Check current risk limits
    fn check_risk_limits(&mut self) -> RiskStatus;
    
    /// Emergency halt all trading activity
    fn emergency_halt(&mut self) -> Result<()>;
}

/// Main execution core implementation
pub struct ExecutionCoreImpl {
    pub event_bus: EventBus,
    pub portfolio: Portfolio,
    pub risk_manager: RiskManager,
    pub is_halted: bool,
}

impl ExecutionCoreImpl {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            event_bus: EventBus::new(),
            portfolio: Portfolio::new(),
            risk_manager: RiskManager::new(config.risk_limits.clone()),
            is_halted: false,
        })
    }
}

impl ExecutionCore for ExecutionCoreImpl {
    fn process_order_intent(&mut self, intent: OrderIntent) -> Result<Uuid> {
        if self.is_halted {
            anyhow::bail!("System is halted - cannot process orders");
        }
        
        // Check pre-trade risk limits
        self.risk_manager.check_pre_trade_risk(&self.portfolio, &intent.asset_id, intent.quantity, intent.price.unwrap_or(0.0))?;
        
        // Check risk limits before processing
        let risk_status = self.risk_manager.check_limits(&self.portfolio);
        if !risk_status.can_trade() {
            anyhow::bail!("Risk limits breached: {:?}", risk_status);
        }
        
        let order_id = Uuid::new_v4();
        let event = Event::OrderIntentReceived { order_id, intent };
        self.event_bus.publish(event)?;
        
        Ok(order_id)
    }
    
    fn update_position(&mut self, fill: Fill) -> Result<()> {
        self.portfolio.update_position(fill.clone())?;
        let event = Event::PositionUpdated { fill };
        self.event_bus.publish(event)?;
        Ok(())
    }
    
    fn check_risk_limits(&mut self) -> RiskStatus {
        self.risk_manager.check_limits(&self.portfolio)
    }
    
    fn emergency_halt(&mut self) -> Result<()> {
        self.is_halted = true;
        self.risk_manager.emergency_halt("Manual emergency halt".to_string())?;
        let event = Event::EmergencyHalt;
        self.event_bus.publish(event)?;
        tracing::warn!("Emergency halt activated");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_core_creation() {
        let config = Config::default();
        let execution_core = ExecutionCoreImpl::new(&config);
        assert!(execution_core.is_ok());
    }

    #[test]
    fn test_emergency_halt() {
        let config = Config::default();
        let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
        
        assert!(!execution_core.is_halted);
        
        let result = execution_core.emergency_halt();
        assert!(result.is_ok());
        assert!(execution_core.is_halted);
    }

    #[test]
    fn test_order_intent_when_halted() {
        let config = Config::default();
        let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
        
        execution_core.emergency_halt().unwrap();
        
        let intent = OrderIntent {
            asset_id: "EURUSD".to_string(),
            side: OrderSide::Buy,
            quantity: 1000.0,
            order_type: OrderType::Market,
            price: None,
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            strategy_id: None,
            correlation_id: None,
        };
        
        let result = execution_core.process_order_intent(intent);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("halted"));
    }
}
    // Property-based test for kill switch effectiveness
    #[test]
    fn prop_emergency_kill_switch_effectiveness() {
        // Feature: algorithmic-trading-system, Property 3: Emergency Kill Switch Effectiveness
        let config = Config::default();
        let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
        
        // Test various scenarios where kill switch should prevent all trading
        let test_scenarios = vec![
            ("Normal operation", false),
            ("After emergency halt", true),
        ];
        
        for (scenario, should_halt) in test_scenarios {
            println!("Testing scenario: {}", scenario);
            
            if should_halt {
                // Activate emergency halt
                execution_core.emergency_halt().unwrap();
            }
            
            // Try to process various order intents
            let order_intents = vec![
                OrderIntent {
                    asset_id: "EURUSD".to_string(),
                    side: OrderSide::Buy,
                    quantity: 1000.0,
                    order_type: OrderType::Market,
                    price: None,
                    metadata: std::collections::HashMap::new(),
                    timestamp: chrono::Utc::now(),
                    strategy_id: Some("test_strategy".to_string()),
                    correlation_id: None,
                },
                OrderIntent {
                    asset_id: "GBPUSD".to_string(),
                    side: OrderSide::Sell,
                    quantity: 500.0,
                    order_type: OrderType::Limit,
                    price: Some(1.2500),
                    metadata: std::collections::HashMap::new(),
                    timestamp: chrono::Utc::now(),
                    strategy_id: Some("test_strategy".to_string()),
                    correlation_id: None,
                },
            ];
            
            for intent in order_intents {
                let result = execution_core.process_order_intent(intent);
                
                if should_halt {
                    // After emergency halt, all order processing should fail
                    assert!(result.is_err(), "Order processing should fail after emergency halt");
                    assert!(result.unwrap_err().to_string().contains("halted"), 
                           "Error should mention system is halted");
                } else {
                    // Normal operation should succeed (assuming no other risk breaches)
                    // We'll accept either success or risk-related failures, but not halt-related failures
                    if let Err(e) = result {
                        assert!(!e.to_string().contains("halted"), 
                               "Should not fail due to halt in normal operation");
                    }
                }
            }
            
            // Test that risk limit checking also reflects halt status
            let risk_status = execution_core.check_risk_limits();
            if should_halt {
                assert!(!risk_status.can_trade(), "Should not be able to trade after emergency halt");
            }
            
            // Reset for next scenario
            if should_halt {
                execution_core.is_halted = false;
                execution_core.risk_manager.reset_halt().unwrap();
            }
        }
    }

    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
        #[test]
        fn prop_kill_switch_blocks_all_orders(
            asset_id in "[A-Z]{3,6}",
            quantity in 1.0f64..10000.0f64,
            price in 0.01f64..1000.0f64,
            side in prop::sample::select(vec![OrderSide::Buy, OrderSide::Sell]),
            order_type in prop::sample::select(vec![OrderType::Market, OrderType::Limit]),
        ) {
            // Feature: algorithmic-trading-system, Property 3: Emergency Kill Switch Effectiveness
            // **Validates: Requirements 1.6**
            
            let config = Config::default();
            let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
            
            // Verify normal operation first (if within risk limits)
            let normal_intent = OrderIntent {
                asset_id: asset_id.clone(),
                side,
                quantity,
                order_type,
                price: if order_type == OrderType::Limit { Some(price) } else { None },
                metadata: std::collections::HashMap::new(),
                timestamp: chrono::Utc::now(),
                strategy_id: Some("test_strategy".to_string()),
                correlation_id: None,
            };
            
            // Try normal operation first
            let _normal_result = execution_core.process_order_intent(normal_intent.clone());
            
            // Activate emergency halt
            execution_core.emergency_halt().unwrap();
            
            // Property: For any system state, activating the global kill switch should immediately halt all trading activity
            prop_assert!(execution_core.is_halted, "System should be halted after emergency halt");
            prop_assert!(execution_core.risk_manager.is_halted(), "Risk manager should also be halted");
            
            // Try the same order after halt - should always fail
            let halt_result = execution_core.process_order_intent(normal_intent);
            prop_assert!(halt_result.is_err(), "All orders should fail after emergency halt");
            prop_assert!(halt_result.unwrap_err().to_string().contains("halted"), 
                        "Error should mention system is halted");
            
            // Risk status should prevent trading
            let risk_status = execution_core.check_risk_limits();
            prop_assert!(!risk_status.can_trade(), "Risk status should prevent trading after halt");
            prop_assert!(matches!(risk_status, RiskStatus::EmergencyHalt { .. }), 
                        "Risk status should indicate emergency halt");
        }

        #[test]
        fn prop_kill_switch_immediate_effect_across_components(
            num_orders in 1usize..10,
        ) {
            // Feature: algorithmic-trading-system, Property 3: Emergency Kill Switch Effectiveness
            // **Validates: Requirements 1.6**
            
            let config = Config::default();
            let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
            
            // Verify normal operation
            prop_assert!(!execution_core.is_halted, "System should start unhaltd");
            prop_assert!(execution_core.check_risk_limits().can_trade(), "Should be able to trade initially");
            
            // Activate kill switch
            let halt_result = execution_core.emergency_halt();
            prop_assert!(halt_result.is_ok(), "Emergency halt should succeed");
            
            // Property: Kill switch should have immediate effect across all system components
            prop_assert!(execution_core.is_halted, "Execution core should be immediately halted");
            prop_assert!(execution_core.risk_manager.is_halted(), "Risk manager should be immediately halted");
            prop_assert!(!execution_core.check_risk_limits().can_trade(), "Should immediately prevent trading");
            
            // Test multiple order attempts - all should fail immediately
            for i in 0..num_orders {
                let intent = OrderIntent {
                    asset_id: format!("TEST{}", i),
                    side: OrderSide::Buy,
                    quantity: 100.0,
                    order_type: OrderType::Market,
                    price: None,
                    metadata: std::collections::HashMap::new(),
                    timestamp: chrono::Utc::now(),
                    strategy_id: Some("test_strategy".to_string()),
                    correlation_id: None,
                };
                
                let result = execution_core.process_order_intent(intent);
                prop_assert!(result.is_err(), "Order {} should fail after halt", i);
                prop_assert!(result.unwrap_err().to_string().contains("halted"), 
                           "Order {} failure should be due to halt", i);
            }
        }

        #[test]
        fn prop_kill_switch_persists_across_operations(
            operations in 1usize..5,
        ) {
            // Feature: algorithmic-trading-system, Property 3: Emergency Kill Switch Effectiveness
            // **Validates: Requirements 1.6**
            
            let config = Config::default();
            let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
            
            // Activate kill switch
            execution_core.emergency_halt().unwrap();
            
            // Property: Kill switch state should persist across multiple operations
            for i in 0..operations {
                // Check that halt persists
                prop_assert!(execution_core.is_halted, "Halt should persist across operation {}", i);
                prop_assert!(!execution_core.check_risk_limits().can_trade(), 
                           "Should not be able to trade during operation {}", i);
                
                // Try various operations that should all fail
                let intent = OrderIntent {
                    asset_id: format!("ASSET{}", i),
                    side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
                    quantity: 100.0 * (i as f64 + 1.0),
                    order_type: OrderType::Market,
                    price: None,
                    metadata: std::collections::HashMap::new(),
                    timestamp: chrono::Utc::now(),
                    strategy_id: Some("test_strategy".to_string()),
                    correlation_id: None,
                };
                
                let result = execution_core.process_order_intent(intent);
                prop_assert!(result.is_err(), "Operation {} should fail when halted", i);
                prop_assert!(result.unwrap_err().to_string().contains("halted"), 
                           "Operation {} should fail due to halt", i);
                
                // Check risk limits again
                let risk_status = execution_core.check_risk_limits();
                prop_assert!(matches!(risk_status, RiskStatus::EmergencyHalt { .. }), 
                           "Risk status should remain emergency halt during operation {}", i);
            }
        }
    }

    #[test]
    fn prop_kill_switch_immediate_effect() {
        // Test that kill switch has immediate effect across all system components
        let config = Config::default();
        let mut execution_core = ExecutionCoreImpl::new(&config).unwrap();
        
        // Verify normal operation first
        assert!(!execution_core.is_halted);
        assert!(execution_core.check_risk_limits().can_trade());
        
        // Activate kill switch
        let result = execution_core.emergency_halt();
        assert!(result.is_ok(), "Emergency halt should succeed");
        
        // Verify immediate effect
        assert!(execution_core.is_halted, "System should be immediately halted");
        assert!(!execution_core.check_risk_limits().can_trade(), "Should immediately prevent trading");
        
        // Verify that the risk manager is also halted
        assert!(execution_core.risk_manager.is_halted(), "Risk manager should also be halted");
        
        // Test multiple order attempts - all should fail immediately
        for i in 0..5 {
            let intent = OrderIntent {
                asset_id: format!("TEST{}", i),
                side: OrderSide::Buy,
                quantity: 100.0,
                order_type: OrderType::Market,
                price: None,
                metadata: std::collections::HashMap::new(),
                timestamp: chrono::Utc::now(),
                strategy_id: Some("test_strategy".to_string()),
                correlation_id: None,
            };
            
            let result = execution_core.process_order_intent(intent);
            assert!(result.is_err(), "All orders should fail after halt");
            assert!(result.unwrap_err().to_string().contains("halted"), 
                   "All failures should be due to halt");
        }
    }
}