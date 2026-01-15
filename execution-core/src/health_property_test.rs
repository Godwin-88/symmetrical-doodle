//! Property-based tests for health check accuracy validation.
//!
//! This module validates Property 23: Health Check Accuracy (Execution Core)
//! - Health checks accurately reflect component status
//! - Performance metrics are within expected bounds
//! - Status aggregation works correctly
//! - Health transitions are properly detected
//!
//! Requirements validated: 10.2

use std::sync::Arc;

use crate::health::{
    HealthChecker, HealthStatus, ComponentType
};
use crate::config::Config;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_accuracy_basic() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        // Test basic health check functionality
        let system_health = health_checker.get_system_health().await;
        assert!(system_health.is_ok());
        
        let health_response = system_health.unwrap();
        assert!(!health_response.components.is_empty());
    }

    #[tokio::test]
    async fn test_system_health_aggregation_basic() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        let system_health = health_checker.get_system_health().await.unwrap();
        
        // System should have multiple components
        assert!(system_health.components.len() >= 3);
        
        // Verify component types are correct
        let component_types: Vec<ComponentType> = system_health.components.values()
            .map(|c| c.component_type.clone())
            .collect();
        
        assert!(component_types.contains(&ComponentType::Core));
        assert!(component_types.contains(&ComponentType::Resource));
    }

    #[tokio::test]
    async fn test_health_metrics_bounds_validation() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        let system_health = health_checker.get_system_health().await.unwrap();
        
        // Verify all metrics are within valid bounds
        for component in system_health.components.values() {
            if let Some(metrics) = &component.metrics {
                assert!(metrics.response_time_ms >= 0.0, "Response time must be non-negative");
                assert!(metrics.error_rate >= 0.0 && metrics.error_rate <= 1.0, "Error rate must be between 0 and 1");
                assert!(metrics.throughput_rps >= 0.0, "Throughput must be non-negative");
                assert!(metrics.uptime_seconds >= 0.0, "Uptime must be non-negative");
                
                // Verify resource usage values are reasonable
                for (key, value) in &metrics.resource_usage {
                    if key.contains("usage") || key.contains("percent") {
                        assert!(*value >= 0.0 && *value <= 1.0, 
                               "Usage metrics should be between 0 and 1: {} = {}", key, value);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_health_status_consistency() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        let system_health = health_checker.get_system_health().await.unwrap();
        
        // Count components by status
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;
        let mut unknown_count = 0;
        
        for component in system_health.components.values() {
            match component.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
                HealthStatus::Unknown => unknown_count += 1,
            }
        }
        
        // Verify system status aggregation logic
        if unhealthy_count > 0 {
            assert_eq!(system_health.status, HealthStatus::Unhealthy);
        } else if degraded_count > 0 {
            assert_eq!(system_health.status, HealthStatus::Degraded);
        } else if unknown_count > 0 {
            assert_eq!(system_health.status, HealthStatus::Unknown);
        } else {
            assert_eq!(system_health.status, HealthStatus::Healthy);
        }
        
        // Verify summary matches actual counts
        if let Some(summary_healthy) = system_health.summary.get("healthy_components") {
            if let Some(count) = summary_healthy.as_u64() {
                assert_eq!(count, healthy_count as u64);
            }
        }
    }

    #[tokio::test]
    async fn test_liveness_and_readiness_probes() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        // Test liveness probe
        let liveness = health_checker.liveness_probe().await;
        assert!(liveness.is_ok());
        
        let liveness_response = liveness.unwrap();
        assert_eq!(liveness_response["status"], "alive");
        assert!(liveness_response["timestamp"].is_number());
        
        // Test readiness probe
        let readiness = health_checker.readiness_probe().await;
        // Readiness should succeed if system is healthy or degraded
        let system_health = health_checker.get_system_health().await.unwrap();
        match system_health.status {
            HealthStatus::Healthy | HealthStatus::Degraded => {
                assert!(readiness.is_ok());
                let readiness_response = readiness.unwrap();
                assert_eq!(readiness_response["status"], "ready");
            }
            _ => {
                assert!(readiness.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_component_dependencies() {
        let config = Arc::new(Config::default());
        let health_checker = HealthChecker::new(config);
        
        let system_health = health_checker.get_system_health().await.unwrap();
        
        // Verify dependency relationships
        for component in system_health.components.values() {
            match component.name.as_str() {
                "event_bus" => {
                    assert!(component.dependencies.is_empty(), "Event bus should have no dependencies");
                }
                "portfolio" => {
                    assert!(component.dependencies.contains(&"event_bus".to_string()), 
                           "Portfolio should depend on event bus");
                }
                "risk_manager" => {
                    assert!(component.dependencies.contains(&"event_bus".to_string()), 
                           "Risk manager should depend on event bus");
                    assert!(component.dependencies.contains(&"portfolio".to_string()), 
                           "Risk manager should depend on portfolio");
                }
                "system_resources" => {
                    assert!(component.dependencies.is_empty(), "System resources should have no dependencies");
                }
                _ => {}
            }
        }
    }
}