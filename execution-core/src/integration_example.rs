use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

use crate::{
    Config, ExecutionManager, ShadowExecutionManager, ShadowConfig, AdapterConfig,
    Portfolio, OrderIntent, OrderSide, OrderType
};

/// Example demonstrating integration of Deriv API adapter with shadow execution
pub async fn run_integration_example() -> Result<()> {
    tracing::info!("Starting execution adapter integration example");
    
    // Create system configuration
    let config = Config::default();
    
    // Create execution manager
    let mut execution_manager = ExecutionManager::new(&config)?;
    
    // Configure Deriv API adapter
    let deriv_config = AdapterConfig {
        name: "deriv_primary".to_string(),
        endpoint: "https://api.deriv.com".to_string(),
        api_key: Some("demo_api_key".to_string()),
        secret_key: Some("demo_secret".to_string()),
        timeout_ms: 5000,
        retry_attempts: 3,
        rate_limit_per_second: 10,
        sandbox_mode: true, // Use sandbox for demo
        metadata: HashMap::new(),
    };
    
    // Configure shadow adapter
    let shadow_config = AdapterConfig {
        name: "shadow_adapter".to_string(),
        endpoint: "https://shadow.local".to_string(),
        api_key: None,
        secret_key: None,
        timeout_ms: 1000,
        retry_attempts: 1,
        rate_limit_per_second: 100,
        sandbox_mode: true,
        metadata: HashMap::new(),
    };
    
    // Add adapters to execution manager (use shadow adapters for testing)
    execution_manager.add_shadow_adapter("deriv".to_string(), deriv_config).await?;
    execution_manager.add_shadow_adapter("shadow".to_string(), shadow_config).await?;
    
    // Set primary adapter
    execution_manager.set_primary_adapter("deriv".to_string())?;
    
    // Connect all adapters
    execution_manager.connect_all().await?;
    
    // Verify health of all adapters
    let health_status = execution_manager.health_check_all().await?;
    tracing::info!("Adapter health status: {:?}", health_status);
    
    // Create shadow execution manager for live/shadow comparison
    let live_portfolio = Arc::new(RwLock::new(Portfolio::new()));
    let shadow_portfolio = Arc::new(RwLock::new(Portfolio::new()));
    
    // Get adapter references (use shadow adapters for testing)
    let live_adapter = Arc::new(RwLock::new(
        crate::ExecutionAdapterFactory::create_shadow_adapter(AdapterConfig {
            name: "live_shadow".to_string(),
            endpoint: "https://live.shadow.local".to_string(),
            api_key: None,
            secret_key: None,
            timeout_ms: 1000,
            retry_attempts: 1,
            rate_limit_per_second: 100,
            sandbox_mode: true,
            metadata: HashMap::new(),
        })?
    ));
    
    let shadow_adapter = Arc::new(RwLock::new(
        crate::ExecutionAdapterFactory::create_shadow_adapter(AdapterConfig {
            name: "shadow_test".to_string(),
            endpoint: "https://shadow.local".to_string(),
            api_key: None,
            secret_key: None,
            timeout_ms: 1000,
            retry_attempts: 1,
            rate_limit_per_second: 100,
            sandbox_mode: true,
            metadata: HashMap::new(),
        })?
    ));
    
    // Connect shadow execution adapters
    {
        let mut live = live_adapter.write().await;
        live.connect().await?;
    }
    {
        let mut shadow = shadow_adapter.write().await;
        shadow.connect().await?;
    }
    
    let shadow_exec_config = ShadowConfig {
        enabled: true,
        sync_interval_ms: 1000,
        max_drift_percentage: 0.01,
        alert_on_drift: true,
        auto_reconcile: false,
        validation_rules: crate::shadow_execution::ValidationRules {
            max_position_drift: 0.001,
            max_pnl_drift: 0.01,
            max_cash_drift: 0.001,
            require_order_matching: true,
            validate_fills: true,
        },
    };
    
    let mut shadow_manager = ShadowExecutionManager::new(
        shadow_exec_config,
        live_portfolio.clone(),
        shadow_portfolio.clone(),
        live_adapter,
        shadow_adapter,
    );
    
    // Demonstrate order processing
    tracing::info!("Processing sample orders...");
    
    let sample_orders = vec![
        OrderIntent {
            asset_id: "EURUSD".to_string(),
            side: OrderSide::Buy,
            quantity: 1000.0,
            order_type: OrderType::Market,
            price: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            strategy_id: Some("demo_strategy".to_string()),
            correlation_id: None,
        },
        OrderIntent {
            asset_id: "GBPUSD".to_string(),
            side: OrderSide::Sell,
            quantity: 500.0,
            order_type: OrderType::Limit,
            price: Some(1.2500),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            strategy_id: Some("demo_strategy".to_string()),
            correlation_id: None,
        },
    ];
    
    // Process orders through shadow execution manager
    for (i, order) in sample_orders.into_iter().enumerate() {
        tracing::info!("Processing order {}: {} {} {}", 
                      i + 1, order.side, order.quantity, order.asset_id);
        
        match shadow_manager.process_order_intent(order).await {
            Ok((live_id, shadow_id)) => {
                tracing::info!("Order processed - Live: {}, Shadow: {}", live_id, shadow_id);
            },
            Err(e) => {
                tracing::error!("Order processing failed: {}", e);
            }
        }
        
        // Small delay between orders
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    
    // Compare live and shadow states
    tracing::info!("Comparing live and shadow states...");
    let state_comparison = shadow_manager.compare_states().await?;
    tracing::info!("State synchronized: {}, Drift: {:.4}%", 
                  state_comparison.is_synchronized, 
                  state_comparison.drift_percentage * 100.0);
    
    if !state_comparison.differences.is_empty() {
        tracing::warn!("Found {} state differences:", state_comparison.differences.len());
        for diff in &state_comparison.differences {
            tracing::warn!("  {}: {} vs {} (diff: {:.4})", 
                          diff.field, diff.live_value, diff.shadow_value, diff.difference);
        }
    }
    
    // Validate shadow execution
    let validation_result = shadow_manager.validate_execution().await?;
    tracing::info!("Shadow execution validation: {}", 
                  if validation_result.is_valid { "PASSED" } else { "FAILED" });
    
    if !validation_result.errors.is_empty() {
        tracing::warn!("Validation errors found:");
        for error in &validation_result.errors {
            tracing::warn!("  {:?}: {}", error.error_type, error.description);
        }
    }
    
    // Get execution statistics
    let execution_stats = execution_manager.get_execution_stats().await?;
    tracing::info!("Execution statistics:");
    for (adapter_name, stats) in execution_stats {
        tracing::info!("  {}: {} orders placed, {} filled, avg latency: {:.2}ms",
                      adapter_name, stats.orders_placed, stats.orders_filled, stats.average_latency_ms);
    }
    
    let shadow_stats = shadow_manager.get_statistics();
    tracing::info!("Shadow execution statistics:");
    tracing::info!("  Enabled: {}, Comparisons: {}, Sync rate: {:.2}%",
                  shadow_stats.enabled, shadow_stats.total_comparisons, 
                  shadow_stats.synchronization_rate * 100.0);
    
    // Demonstrate emergency halt
    tracing::info!("Testing emergency halt...");
    execution_manager.emergency_halt().await?;
    tracing::info!("Emergency halt completed");
    
    // Disconnect all adapters
    execution_manager.disconnect_all().await?;
    
    tracing::info!("Integration example completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integration_example() {
        // Initialize tracing for test output
        let _ = tracing_subscriber::fmt::try_init();
        
        // Run the integration example
        let result = run_integration_example().await;
        
        // The example should complete without errors
        assert!(result.is_ok(), "Integration example failed: {:?}", result.err());
    }
}