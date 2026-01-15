use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Real-time market tick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub timestamp: i64,           // Nanosecond precision
    pub asset_id: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub last_size: f64,
    pub volume: f64,
    pub sequence_number: u64,     // For gap detection
    pub exchange: String,
    pub latency_us: u32,          // Capture latency
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp: i64,
    pub asset_id: String,
    pub bids: Vec<(f64, f64)>,    // Price, Size
    pub asks: Vec<(f64, f64)>,
    pub depth_levels: usize,
    pub checksum: u64,            // Data integrity
}

/// Market microstructure metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMicrostructure {
    pub timestamp: i64,
    pub asset_id: String,
    pub spread_bps: f64,
    pub effective_spread_bps: f64,
    pub quoted_spread_bps: f64,
    pub depth_bid: f64,           // Cumulative depth
    pub depth_ask: f64,
    pub imbalance_ratio: f64,     // Bid/Ask imbalance
    pub tick_frequency: f64,      // Ticks per second
    pub price_impact_bps: f64,    // For standard size
}

/// Liquidity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    pub timestamp: i64,
    pub asset_id: String,
    pub bid_liquidity_usd: f64,
    pub ask_liquidity_usd: f64,
    pub total_liquidity_usd: f64,
    pub liquidity_score: f64,     // 0-100
    pub resilience_score: f64,    // How fast liquidity recovers
    pub toxicity_score: f64,      // Adverse selection risk
}

/// Market event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MarketEvent {
    PriceUpdate {
        asset_id: String,
        price: f64,
        change_pct: f64,
        timestamp: i64,
    },
    VolatilitySpike {
        asset_id: String,
        current_vol: f64,
        baseline_vol: f64,
        spike_factor: f64,
        timestamp: i64,
    },
    LiquidityDrop {
        asset_id: String,
        current_liquidity: f64,
        baseline_liquidity: f64,
        drop_pct: f64,
        timestamp: i64,
    },
    CorrelationBreakdown {
        asset_pair: (String, String),
        current_corr: f64,
        historical_corr: f64,
        deviation: f64,
        timestamp: i64,
    },
    CircuitBreaker {
        asset_id: String,
        reason: String,
        duration_sec: u32,
        timestamp: i64,
    },
}

/// Market data collector
pub struct MarketDataCollector {
    ticks: HashMap<String, Vec<MarketTick>>,
    microstructure: HashMap<String, MarketMicrostructure>,
    liquidity: HashMap<String, LiquidityMetrics>,
}

impl MarketDataCollector {
    pub fn new() -> Self {
        Self {
            ticks: HashMap::new(),
            microstructure: HashMap::new(),
            liquidity: HashMap::new(),
        }
    }

    /// Process incoming market tick
    pub fn process_tick(&mut self, tick: MarketTick) {
        let asset_id = tick.asset_id.clone();
        
        // Store tick
        self.ticks.entry(asset_id.clone())
            .or_insert_with(Vec::new)
            .push(tick.clone());
        
        // Calculate microstructure metrics
        let microstructure = self.calculate_microstructure(&tick);
        self.microstructure.insert(asset_id.clone(), microstructure);
        
        // Calculate liquidity metrics
        let liquidity = self.calculate_liquidity(&tick);
        self.liquidity.insert(asset_id, liquidity);
    }

    /// Calculate microstructure metrics from tick
    fn calculate_microstructure(&self, tick: &MarketTick) -> MarketMicrostructure {
        let spread = tick.ask - tick.bid;
        let mid_price = (tick.bid + tick.ask) / 2.0;
        let spread_bps = (spread / mid_price) * 10000.0;
        
        MarketMicrostructure {
            timestamp: tick.timestamp,
            asset_id: tick.asset_id.clone(),
            spread_bps,
            effective_spread_bps: spread_bps * 0.9, // Simplified
            quoted_spread_bps: spread_bps,
            depth_bid: tick.bid_size,
            depth_ask: tick.ask_size,
            imbalance_ratio: tick.bid_size / (tick.bid_size + tick.ask_size),
            tick_frequency: 100.0, // Placeholder
            price_impact_bps: spread_bps * 0.5, // Simplified
        }
    }

    /// Calculate liquidity metrics from tick
    fn calculate_liquidity(&self, tick: &MarketTick) -> LiquidityMetrics {
        let mid_price = (tick.bid + tick.ask) / 2.0;
        let bid_liquidity = tick.bid_size * tick.bid;
        let ask_liquidity = tick.ask_size * tick.ask;
        let total_liquidity = bid_liquidity + ask_liquidity;
        
        LiquidityMetrics {
            timestamp: tick.timestamp,
            asset_id: tick.asset_id.clone(),
            bid_liquidity_usd: bid_liquidity,
            ask_liquidity_usd: ask_liquidity,
            total_liquidity_usd: total_liquidity,
            liquidity_score: (total_liquidity / 1000000.0).min(100.0), // Simplified
            resilience_score: 75.0, // Placeholder
            toxicity_score: 25.0, // Placeholder
        }
    }

    /// Get latest microstructure for asset
    pub fn get_microstructure(&self, asset_id: &str) -> Option<&MarketMicrostructure> {
        self.microstructure.get(asset_id)
    }

    /// Get latest liquidity for asset
    pub fn get_liquidity(&self, asset_id: &str) -> Option<&LiquidityMetrics> {
        self.liquidity.get(asset_id)
    }

    /// Get recent ticks for asset
    pub fn get_recent_ticks(&self, asset_id: &str, count: usize) -> Vec<&MarketTick> {
        self.ticks.get(asset_id)
            .map(|ticks| ticks.iter().rev().take(count).collect())
            .unwrap_or_default()
    }
}

impl Default for MarketDataCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_collector() {
        let mut collector = MarketDataCollector::new();
        
        let tick = MarketTick {
            timestamp: 1000000000,
            asset_id: "EURUSD".to_string(),
            bid: 1.0845,
            ask: 1.0847,
            bid_size: 1000000.0,
            ask_size: 900000.0,
            last_price: 1.0846,
            last_size: 50000.0,
            volume: 5000000.0,
            sequence_number: 1,
            exchange: "TEST".to_string(),
            latency_us: 100,
        };
        
        collector.process_tick(tick.clone());
        
        // Check microstructure
        let microstructure = collector.get_microstructure("EURUSD").unwrap();
        assert!(microstructure.spread_bps > 0.0);
        assert!(microstructure.imbalance_ratio > 0.0 && microstructure.imbalance_ratio < 1.0);
        
        // Check liquidity
        let liquidity = collector.get_liquidity("EURUSD").unwrap();
        assert!(liquidity.total_liquidity_usd > 0.0);
        
        // Check ticks
        let ticks = collector.get_recent_ticks("EURUSD", 10);
        assert_eq!(ticks.len(), 1);
    }
}
