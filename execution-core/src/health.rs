/*!
Health monitoring and status endpoints for the Execution Core.

This module provides comprehensive health checking capabilities including:
- Component status monitoring
- Performance metrics collection
- Dependency health validation
- System resource monitoring
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};

use crate::config::Config;
use crate::event_bus::EventBus;
use crate::portfolio::Portfolio;
use crate::risk::RiskManager;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    Database,
    Cache,
    ExternalApi,
    InternalService,
    Resource,
    Core,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub response_time_ms: f64,
    pub error_rate: f64,
    pub throughput_rps: f64,
    pub resource_usage: HashMap<String, f64>,
    pub last_check: u64, // Unix timestamp
    pub uptime_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub component_type: ComponentType,
    pub status: HealthStatus,
    pub message: String,
    pub metrics: Option<HealthMetrics>,
    pub dependencies: Vec<String>,
    pub last_updated: u64, // Unix timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthResponse {
    pub status: HealthStatus,
    pub timestamp: u64,
    pub uptime_seconds: f64,
    pub components: HashMap<String, ComponentHealth>,
    pub summary: HashMap<String, serde_json::Value>,
    pub alerts: Vec<String>,
}

pub struct HealthChecker {
    start_time: std::time::Instant,
    config: Arc<Config>,
    event_bus: Option<Arc<RwLock<EventBus>>>,
    portfolio: Option<Arc<RwLock<Portfolio>>>,
    risk_manager: Option<Arc<RwLock<RiskManager>>>,
    alert_thresholds: HashMap<String, f64>,
}

impl HealthChecker {
    pub fn new(config: Arc<Config>) -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("response_time_ms".to_string(), 1000.0);
        alert_thresholds.insert("error_rate".to_string(), 0.05);
        alert_thresholds.insert("cpu_usage".to_string(), 0.8);
        alert_thresholds.insert("memory_usage".to_string(), 0.8);
        alert_thresholds.insert("disk_usage".to_string(), 0.9);

        Self {
            start_time: std::time::Instant::now(),
            config,
            event_bus: None,
            portfolio: None,
            risk_manager: None,
            alert_thresholds,
        }
    }

    pub fn set_event_bus(&mut self, event_bus: Arc<RwLock<EventBus>>) {
        self.event_bus = Some(event_bus);
    }

    pub fn set_portfolio(&mut self, portfolio: Arc<RwLock<Portfolio>>) {
        self.portfolio = Some(portfolio);
    }

    pub fn set_risk_manager(&mut self, risk_manager: Arc<RwLock<RiskManager>>) {
        self.risk_manager = Some(risk_manager);
    }

    async fn check_event_bus_health(&self) -> ComponentHealth {
        let start_time = std::time::Instant::now();
        
        match &self.event_bus {
            Some(event_bus) => {
                let bus = event_bus.read().await;
                let response_time = start_time.elapsed().as_millis() as f64;
                
                // Check event bus metrics
                let queue_size = bus.get_queue_size();
                let processed_events = bus.get_processed_count();
                let error_count = bus.get_error_count();
                
                let error_rate = if processed_events > 0 {
                    error_count as f64 / processed_events as f64
                } else {
                    0.0
                };

                let status = if error_rate > self.alert_thresholds["error_rate"] {
                    HealthStatus::Degraded
                } else if queue_size > 1000 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                let message = match status {
                    HealthStatus::Healthy => "Event bus operating normally".to_string(),
                    HealthStatus::Degraded => format!("Event bus degraded: queue_size={}, error_rate={:.3}", queue_size, error_rate),
                    _ => "Event bus status unknown".to_string(),
                };

                let mut resource_usage = HashMap::new();
                resource_usage.insert("queue_size".to_string(), queue_size as f64);
                resource_usage.insert("processed_events".to_string(), processed_events as f64);
                resource_usage.insert("error_count".to_string(), error_count as f64);

                ComponentHealth {
                    name: "event_bus".to_string(),
                    component_type: ComponentType::Core,
                    status,
                    message,
                    metrics: Some(HealthMetrics {
                        response_time_ms: response_time,
                        error_rate,
                        throughput_rps: 0.0, // Would be calculated from metrics
                        resource_usage,
                        last_check: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        uptime_seconds: self.start_time.elapsed().as_secs_f64(),
                    }),
                    dependencies: vec![],
                    last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                }
            }
            None => ComponentHealth {
                name: "event_bus".to_string(),
                component_type: ComponentType::Core,
                status: HealthStatus::Unhealthy,
                message: "Event bus not initialized".to_string(),
                metrics: None,
                dependencies: vec![],
                last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            }
        }
    }

    async fn check_portfolio_health(&self) -> ComponentHealth {
        let start_time = std::time::Instant::now();
        
        match &self.portfolio {
            Some(portfolio) => {
                let portfolio_ref = portfolio.read().await;
                let response_time = start_time.elapsed().as_millis() as f64;
                
                // Check portfolio metrics
                let position_count = portfolio_ref.get_all_positions().len();
                let total_pnl = portfolio_ref.get_total_pnl();
                
                let status = HealthStatus::Healthy;

                let message = format!("Portfolio healthy: {} positions, PnL: {:.2}", position_count, total_pnl);

                let mut resource_usage = HashMap::new();
                resource_usage.insert("position_count".to_string(), position_count as f64);
                resource_usage.insert("total_pnl".to_string(), total_pnl);

                ComponentHealth {
                    name: "portfolio".to_string(),
                    component_type: ComponentType::Core,
                    status,
                    message,
                    metrics: Some(HealthMetrics {
                        response_time_ms: response_time,
                        error_rate: 0.0,
                        throughput_rps: 0.0,
                        resource_usage,
                        last_check: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        uptime_seconds: self.start_time.elapsed().as_secs_f64(),
                    }),
                    dependencies: vec!["event_bus".to_string()],
                    last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                }
            }
            None => ComponentHealth {
                name: "portfolio".to_string(),
                component_type: ComponentType::Core,
                status: HealthStatus::Unhealthy,
                message: "Portfolio not initialized".to_string(),
                metrics: None,
                dependencies: vec!["event_bus".to_string()],
                last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            }
        }
    }

    async fn check_risk_manager_health(&self) -> ComponentHealth {
        let start_time = std::time::Instant::now();
        
        match &self.risk_manager {
            Some(risk_manager) => {
                let manager = risk_manager.read().await;
                let response_time = start_time.elapsed().as_millis() as f64;
                
                // Check risk manager metrics
                let is_halted = manager.is_halted();
                
                let status = if is_halted {
                    HealthStatus::Unhealthy
                } else {
                    HealthStatus::Healthy
                };

                let message = match status {
                    HealthStatus::Healthy => "Risk manager healthy".to_string(),
                    HealthStatus::Unhealthy => "Kill switch activated - trading halted".to_string(),
                    _ => "Risk manager status unknown".to_string(),
                };

                let mut resource_usage = HashMap::new();
                resource_usage.insert("is_halted".to_string(), if is_halted { 1.0 } else { 0.0 });

                ComponentHealth {
                    name: "risk_manager".to_string(),
                    component_type: ComponentType::Core,
                    status,
                    message,
                    metrics: Some(HealthMetrics {
                        response_time_ms: response_time,
                        error_rate: 0.0,
                        throughput_rps: 0.0,
                        resource_usage,
                        last_check: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        uptime_seconds: self.start_time.elapsed().as_secs_f64(),
                    }),
                    dependencies: vec!["event_bus".to_string(), "portfolio".to_string()],
                    last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                }
            }
            None => ComponentHealth {
                name: "risk_manager".to_string(),
                component_type: ComponentType::Core,
                status: HealthStatus::Unhealthy,
                message: "Risk manager not initialized".to_string(),
                metrics: None,
                dependencies: vec!["event_bus".to_string(), "portfolio".to_string()],
                last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            }
        }
    }

    async fn check_system_resources(&self) -> ComponentHealth {
        let start_time = std::time::Instant::now();
        
        // Get system resource information
        let mut resource_usage = HashMap::new();
        let mut status = HealthStatus::Healthy;
        let mut messages = Vec::new();

        // Memory usage (simplified - in production would use proper system metrics)
        let memory_usage = 0.5; // Placeholder
        resource_usage.insert("memory_usage".to_string(), memory_usage);
        
        if memory_usage > self.alert_thresholds["memory_usage"] {
            status = HealthStatus::Degraded;
            messages.push(format!("High memory usage: {:.1}%", memory_usage * 100.0));
        }

        // CPU usage (simplified)
        let cpu_usage = 0.3; // Placeholder
        resource_usage.insert("cpu_usage".to_string(), cpu_usage);
        
        if cpu_usage > self.alert_thresholds["cpu_usage"] {
            status = HealthStatus::Degraded;
            messages.push(format!("High CPU usage: {:.1}%", cpu_usage * 100.0));
        }

        let message = if messages.is_empty() {
            "System resources within normal limits".to_string()
        } else {
            messages.join("; ")
        };

        let response_time = start_time.elapsed().as_millis() as f64;

        ComponentHealth {
            name: "system_resources".to_string(),
            component_type: ComponentType::Resource,
            status,
            message,
            metrics: Some(HealthMetrics {
                response_time_ms: response_time,
                error_rate: 0.0,
                throughput_rps: 0.0,
                resource_usage,
                last_check: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            }),
            dependencies: vec![],
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub async fn get_system_health(&self) -> Result<SystemHealthResponse> {
        // Run all health checks
        let event_bus_health = self.check_event_bus_health().await;
        let portfolio_health = self.check_portfolio_health().await;
        let risk_health = self.check_risk_manager_health().await;
        let resource_health = self.check_system_resources().await;

        let mut components = HashMap::new();
        components.insert(event_bus_health.name.clone(), event_bus_health.clone());
        components.insert(portfolio_health.name.clone(), portfolio_health.clone());
        components.insert(risk_health.name.clone(), risk_health.clone());
        components.insert(resource_health.name.clone(), resource_health.clone());

        // Generate alerts
        let mut alerts = Vec::new();
        for component in components.values() {
            match component.status {
                HealthStatus::Unhealthy => {
                    alerts.push(format!("{}: {}", component.name, component.message));
                }
                HealthStatus::Degraded => {
                    alerts.push(format!("{}: Performance degraded - {}", component.name, component.message));
                }
                _ => {}
            }
        }

        // Determine overall status
        let statuses: Vec<&HealthStatus> = components.values().map(|c| &c.status).collect();
        let overall_status = if statuses.contains(&&HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if statuses.contains(&&HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else if statuses.contains(&&HealthStatus::Unknown) {
            HealthStatus::Unknown
        } else {
            HealthStatus::Healthy
        };

        // Generate summary
        let mut summary = HashMap::new();
        summary.insert("total_components".to_string(), serde_json::Value::Number(serde_json::Number::from(components.len())));
        summary.insert("healthy_components".to_string(), serde_json::Value::Number(serde_json::Number::from(
            components.values().filter(|c| c.status == HealthStatus::Healthy).count()
        )));
        summary.insert("degraded_components".to_string(), serde_json::Value::Number(serde_json::Number::from(
            components.values().filter(|c| c.status == HealthStatus::Degraded).count()
        )));
        summary.insert("unhealthy_components".to_string(), serde_json::Value::Number(serde_json::Number::from(
            components.values().filter(|c| c.status == HealthStatus::Unhealthy).count()
        )));

        let avg_response_time: f64 = components.values()
            .filter_map(|c| c.metrics.as_ref())
            .map(|m| m.response_time_ms)
            .sum::<f64>() / components.len().max(1) as f64;
        
        summary.insert("average_response_time_ms".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(avg_response_time).unwrap_or(serde_json::Number::from(0))
        ));

        Ok(SystemHealthResponse {
            status: overall_status,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            components,
            summary,
            alerts,
        })
    }

    pub async fn liveness_probe(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "status": "alive",
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    pub async fn readiness_probe(&self) -> Result<serde_json::Value> {
        let health = self.get_system_health().await?;
        
        match health.status {
            HealthStatus::Healthy | HealthStatus::Degraded => {
                Ok(serde_json::json!({
                    "status": "ready",
                    "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
                }))
            }
            _ => Err(anyhow!("Service not ready"))
        }
    }
}