use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub logging: LoggingConfig,
    pub risk_limits: RiskLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub postgres_url: String,
    pub neo4j_url: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_size: f64,
    pub max_drawdown: f64,
    pub max_daily_loss: f64,
    pub position_limits: HashMap<String, f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                postgres_url: "postgresql://postgres:password@localhost:5432/trading_system".to_string(),
                neo4j_url: "bolt://localhost:7687".to_string(),
                neo4j_user: "neo4j".to_string(),
                neo4j_password: "password".to_string(),
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                max_connections: 10,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
            },
            risk_limits: RiskLimits {
                max_position_size: 100000.0,
                max_drawdown: 0.05, // 5%
                max_daily_loss: 10000.0,
                position_limits: HashMap::new(),
            },
        }
    }
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let settings = config::Config::builder()
            .add_source(config::Environment::with_prefix("TRADING"))
            .build()?;
        
        let config = settings.try_deserialize().unwrap_or_else(|_| Config::default());
        Ok(config)
    }
}