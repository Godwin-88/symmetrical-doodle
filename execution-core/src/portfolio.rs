use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    pub asset_id: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub strategy_id: Option<String>,
    pub correlation_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "Buy"),
            OrderSide::Sell => write!(f, "Sell"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: Uuid,
    pub asset_id: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub commission: f64,
    pub metadata: HashMap<String, String>,
    pub execution_venue: String,
    pub liquidity_flag: LiquidityFlag,
    pub slippage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityFlag {
    Maker,
    Taker,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub asset_id: String,
    pub quantity: f64,
    pub average_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub last_updated: DateTime<Utc>,
    pub first_opened: DateTime<Utc>,
    pub total_commission: f64,
    pub high_water_mark: f64,
    pub low_water_mark: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub positions: HashMap<String, Position>,
    pub cash_balance: f64,
    pub total_pnl: f64,
    pub net_exposure: f64,
    pub gross_exposure: f64,
    pub total_commission: f64,
    pub sequence_number: u64,
}

/// Persistent storage trait for portfolio state
pub trait PortfolioStore: Send + Sync {
    fn save_snapshot(&mut self, snapshot: &PortfolioSnapshot) -> Result<()>;
    fn load_latest_snapshot(&self) -> Result<Option<PortfolioSnapshot>>;
    fn save_fill(&mut self, fill: &Fill) -> Result<()>;
    fn load_fills(&self, from_timestamp: DateTime<Utc>) -> Result<Vec<Fill>>;
}

/// In-memory portfolio store implementation
pub struct InMemoryPortfolioStore {
    snapshots: Arc<Mutex<Vec<PortfolioSnapshot>>>,
    fills: Arc<Mutex<Vec<Fill>>>,
}

impl InMemoryPortfolioStore {
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
            fills: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl PortfolioStore for InMemoryPortfolioStore {
    fn save_snapshot(&mut self, snapshot: &PortfolioSnapshot) -> Result<()> {
        self.snapshots.lock().unwrap().push(snapshot.clone());
        Ok(())
    }
    
    fn load_latest_snapshot(&self) -> Result<Option<PortfolioSnapshot>> {
        let snapshots = self.snapshots.lock().unwrap();
        Ok(snapshots.last().cloned())
    }
    
    fn save_fill(&mut self, fill: &Fill) -> Result<()> {
        self.fills.lock().unwrap().push(fill.clone());
        Ok(())
    }
    
    fn load_fills(&self, from_timestamp: DateTime<Utc>) -> Result<Vec<Fill>> {
        let fills = self.fills.lock().unwrap();
        let filtered: Vec<Fill> = fills
            .iter()
            .filter(|f| f.timestamp >= from_timestamp)
            .cloned()
            .collect();
        Ok(filtered)
    }
}

/// Portfolio accounting with real-time position tracking
pub struct Portfolio {
    positions: HashMap<String, Position>,
    cash_balance: f64,
    total_pnl: f64,
    fills: Vec<Fill>,
    store: Box<dyn PortfolioStore>,
    sequence_number: u64,
    total_commission: f64,
    starting_capital: f64,
}

impl Portfolio {
    pub fn new() -> Self {
        Self::with_store(Box::new(InMemoryPortfolioStore::new()))
    }
    
    pub fn with_store(store: Box<dyn PortfolioStore>) -> Self {
        let starting_capital = 100000.0;
        Self {
            positions: HashMap::new(),
            cash_balance: starting_capital,
            total_pnl: 0.0,
            fills: Vec::new(),
            store,
            sequence_number: 0,
            total_commission: 0.0,
            starting_capital,
        }
    }
    
    pub fn restore_from_snapshot(&mut self) -> Result<()> {
        if let Some(snapshot) = self.store.load_latest_snapshot()? {
            self.positions = snapshot.positions;
            self.cash_balance = snapshot.cash_balance;
            self.total_pnl = snapshot.total_pnl;
            self.sequence_number = snapshot.sequence_number;
            self.total_commission = snapshot.total_commission;
            
            // Load fills since snapshot
            let fills = self.store.load_fills(snapshot.timestamp)?;
            self.fills = fills;
            
            tracing::info!("Portfolio restored from snapshot at {}", snapshot.timestamp);
        }
        Ok(())
    }
    
    pub fn create_snapshot(&self) -> PortfolioSnapshot {
        PortfolioSnapshot {
            timestamp: Utc::now(),
            positions: self.positions.clone(),
            cash_balance: self.cash_balance,
            total_pnl: self.total_pnl,
            net_exposure: self.get_net_exposure(),
            gross_exposure: self.get_gross_exposure(),
            total_commission: self.total_commission,
            sequence_number: self.sequence_number,
        }
    }
    
    pub fn save_snapshot(&mut self) -> Result<()> {
        let snapshot = self.create_snapshot();
        self.store.save_snapshot(&snapshot)?;
        tracing::debug!("Portfolio snapshot saved at {}", snapshot.timestamp);
        Ok(())
    }
    
    pub fn update_position(&mut self, fill: Fill) -> Result<()> {
        // Save fill to persistent store
        self.store.save_fill(&fill)?;
        
        let position = self.positions.entry(fill.asset_id.clone())
            .or_insert_with(|| Position {
                asset_id: fill.asset_id.clone(),
                quantity: 0.0,
                average_price: 0.0,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
                last_updated: Utc::now(),
                first_opened: fill.timestamp,
                total_commission: 0.0,
                high_water_mark: 0.0,
                low_water_mark: 0.0,
            });
        
        let fill_value = fill.quantity * fill.price;
        let commission = fill.commission;
        
        // Update commission tracking
        position.total_commission += commission;
        self.total_commission += commission;
        
        // Calculate realized P&L for position reductions
        let mut realized_pnl = 0.0;
        
        match fill.side {
            OrderSide::Buy => {
                if position.quantity < 0.0 {
                    // Covering short position - calculate realized P&L
                    let covered_quantity = fill.quantity.min(position.quantity.abs());
                    realized_pnl = covered_quantity * (position.average_price - fill.price);
                    position.realized_pnl += realized_pnl;
                }
                
                if position.quantity + fill.quantity > 0.0 && position.quantity <= 0.0 {
                    // Flipping from short to long
                    let remaining_quantity = position.quantity + fill.quantity;
                    position.average_price = fill.price;
                    position.quantity = remaining_quantity;
                } else if position.quantity >= 0.0 {
                    // Adding to long position
                    let new_quantity = position.quantity + fill.quantity;
                    if new_quantity > 0.0 {
                        position.average_price = ((position.quantity * position.average_price) + fill_value) / new_quantity;
                    }
                    position.quantity = new_quantity;
                } else {
                    // Reducing short position
                    position.quantity += fill.quantity;
                }
                
                self.cash_balance -= fill_value + commission;
            },
            OrderSide::Sell => {
                if position.quantity > 0.0 {
                    // Selling long position - calculate realized P&L
                    let sold_quantity = fill.quantity.min(position.quantity);
                    realized_pnl = sold_quantity * (fill.price - position.average_price);
                    position.realized_pnl += realized_pnl;
                }
                
                if position.quantity - fill.quantity < 0.0 && position.quantity >= 0.0 {
                    // Flipping from long to short
                    let remaining_quantity = position.quantity - fill.quantity;
                    position.average_price = fill.price;
                    position.quantity = remaining_quantity;
                } else if position.quantity <= 0.0 {
                    // Adding to short position
                    let new_quantity = position.quantity - fill.quantity;
                    if new_quantity < 0.0 {
                        position.average_price = ((position.quantity.abs() * position.average_price) + fill_value) / new_quantity.abs();
                    }
                    position.quantity = new_quantity;
                } else {
                    // Reducing long position
                    position.quantity -= fill.quantity;
                }
                
                self.cash_balance += fill_value - commission;
            }
        }
        
        position.last_updated = fill.timestamp;
        self.fills.push(fill);
        self.sequence_number += 1;
        
        // Update high/low water marks
        let current_value = position.quantity * position.average_price;
        if current_value > position.high_water_mark {
            position.high_water_mark = current_value;
        }
        if current_value < position.low_water_mark {
            position.low_water_mark = current_value;
        }
        
        // Store position info for logging before recalculating P&L
        let asset_id = position.asset_id.clone();
        let quantity = position.quantity;
        let avg_price = position.average_price;
        
        // Recalculate total P&L
        self.calculate_total_pnl();
        
        tracing::debug!("Position updated for {}: quantity={}, avg_price={}, realized_pnl={}", 
                       asset_id, quantity, avg_price, realized_pnl);
        
        Ok(())
    }
    
    pub fn get_position(&self, asset_id: &str) -> Option<&Position> {
        self.positions.get(asset_id)
    }
    
    pub fn get_all_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }
    
    pub fn get_cash_balance(&self) -> f64 {
        self.cash_balance
    }
    
    pub fn get_total_pnl(&self) -> f64 {
        self.total_pnl
    }
    
    pub fn get_net_exposure(&self) -> f64 {
        self.positions.values()
            .map(|pos| pos.quantity * pos.average_price)
            .sum()
    }
    
    pub fn get_gross_exposure(&self) -> f64 {
        self.positions.values()
            .map(|pos| (pos.quantity * pos.average_price).abs())
            .sum()
    }
    
    pub fn get_total_commission(&self) -> f64 {
        self.total_commission
    }
    
    pub fn get_equity(&self) -> f64 {
        self.cash_balance + self.total_pnl
    }
    
    pub fn get_return_on_capital(&self) -> f64 {
        if self.starting_capital > 0.0 {
            self.total_pnl / self.starting_capital
        } else {
            0.0
        }
    }
    
    pub fn get_fills(&self) -> &Vec<Fill> {
        &self.fills
    }
    
    pub fn get_fills_for_asset(&self, asset_id: &str) -> Vec<&Fill> {
        self.fills.iter()
            .filter(|fill| fill.asset_id == asset_id)
            .collect()
    }
    
    pub fn get_sequence_number(&self) -> u64 {
        self.sequence_number
    }
    
    fn calculate_total_pnl(&mut self) {
        self.total_pnl = self.positions.values()
            .map(|pos| pos.realized_pnl + pos.unrealized_pnl)
            .sum();
    }
    
    pub fn update_market_prices(&mut self, prices: &HashMap<String, f64>) {
        for (asset_id, current_price) in prices {
            if let Some(position) = self.positions.get_mut(asset_id) {
                if position.quantity != 0.0 {
                    position.unrealized_pnl = position.quantity * (current_price - position.average_price);
                }
            }
        }
        self.calculate_total_pnl();
    }
    
    pub fn calculate_drawdown(&self) -> f64 {
        let current_equity = self.get_equity();
        
        // For simplicity, use starting capital as peak equity
        // In a real system, you'd track the actual peak equity over time
        let peak_equity = self.starting_capital.max(current_equity);
        
        if peak_equity > 0.0 {
            (peak_equity - current_equity) / peak_equity
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fill(asset_id: &str, side: OrderSide, quantity: f64, price: f64) -> Fill {
        Fill {
            order_id: Uuid::new_v4(),
            asset_id: asset_id.to_string(),
            side,
            quantity,
            price,
            timestamp: Utc::now(),
            commission: 1.0,
            metadata: HashMap::new(),
            execution_venue: "test_venue".to_string(),
            liquidity_flag: LiquidityFlag::Taker,
            slippage: 0.0,
        }
    }

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new();
        assert_eq!(portfolio.get_cash_balance(), 100000.0);
        assert_eq!(portfolio.get_total_pnl(), 0.0);
        assert_eq!(portfolio.get_all_positions().len(), 0);
    }

    #[test]
    fn test_long_position_opening() {
        let mut portfolio = Portfolio::new();
        let fill = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        
        portfolio.update_position(fill).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert_eq!(position.quantity, 1000.0);
        assert_eq!(position.average_price, 1.1000);
        assert_eq!(portfolio.get_cash_balance(), 100000.0 - 1100.0 - 1.0); // price * quantity - commission
    }

    #[test]
    fn test_short_position_opening() {
        let mut portfolio = Portfolio::new();
        let fill = create_test_fill("EURUSD", OrderSide::Sell, 1000.0, 1.1000);
        
        portfolio.update_position(fill).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert_eq!(position.quantity, -1000.0);
        assert_eq!(position.average_price, 1.1000);
        assert_eq!(portfolio.get_cash_balance(), 100000.0 + 1100.0 - 1.0); // + price * quantity - commission
    }

    #[test]
    fn test_position_averaging() {
        let mut portfolio = Portfolio::new();
        
        // First buy
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill1).unwrap();
        
        // Second buy at different price
        let fill2 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1200);
        portfolio.update_position(fill2).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert_eq!(position.quantity, 2000.0);
        assert_eq!(position.average_price, 1.1100); // (1.1000 + 1.1200) / 2
    }

    #[test]
    fn test_realized_pnl_calculation() {
        let mut portfolio = Portfolio::new();
        
        // Open long position
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill1).unwrap();
        
        // Close half at profit
        let fill2 = create_test_fill("EURUSD", OrderSide::Sell, 500.0, 1.1100);
        portfolio.update_position(fill2).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        println!("Position quantity: {}, realized_pnl: {}", position.quantity, position.realized_pnl);
        assert_eq!(position.quantity, 500.0);
        assert!((position.realized_pnl - 5.0).abs() < 0.01); // 500 * (1.1100 - 1.1000) = 5.0
    }

    #[test]
    fn test_position_flip_long_to_short() {
        let mut portfolio = Portfolio::new();
        
        // Open long position
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill1).unwrap();
        
        // Sell more than position size (flip to short)
        let fill2 = create_test_fill("EURUSD", OrderSide::Sell, 1500.0, 1.1100);
        portfolio.update_position(fill2).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert_eq!(position.quantity, -500.0);
        assert_eq!(position.average_price, 1.1100);
        assert!((position.realized_pnl - 10.0).abs() < 0.01); // 1000 * (1.1100 - 1.1000) = 10.0
    }

    #[test]
    fn test_unrealized_pnl_calculation() {
        let mut portfolio = Portfolio::new();
        
        // Open long position
        let fill = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill).unwrap();
        
        // Update market prices
        let mut prices = HashMap::new();
        prices.insert("EURUSD".to_string(), 1.1050);
        portfolio.update_market_prices(&prices);
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert!((position.unrealized_pnl - 5.0).abs() < 0.01); // 1000 * (1.1050 - 1.1000) = 5.0
        assert!((portfolio.get_total_pnl() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_exposure_calculations() {
        let mut portfolio = Portfolio::new();
        
        // Long EURUSD
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill1).unwrap();
        
        // Short GBPUSD
        let fill2 = create_test_fill("GBPUSD", OrderSide::Sell, 800.0, 1.2500);
        portfolio.update_position(fill2).unwrap();
        
        let net_exposure = portfolio.get_net_exposure();
        let gross_exposure = portfolio.get_gross_exposure();
        
        assert_eq!(net_exposure, 1100.0 - 1000.0); // 1000*1.1 - 800*1.25
        assert_eq!(gross_exposure, 1100.0 + 1000.0); // |1000*1.1| + |800*1.25|
    }

    #[test]
    fn test_commission_tracking() {
        let mut portfolio = Portfolio::new();
        
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        let fill2 = create_test_fill("EURUSD", OrderSide::Sell, 500.0, 1.1100);
        
        portfolio.update_position(fill1).unwrap();
        portfolio.update_position(fill2).unwrap();
        
        assert_eq!(portfolio.get_total_commission(), 2.0);
        
        let position = portfolio.get_position("EURUSD").unwrap();
        assert_eq!(position.total_commission, 2.0);
    }

    #[test]
    fn test_portfolio_snapshot() {
        let mut portfolio = Portfolio::new();
        
        let fill = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill).unwrap();
        
        let snapshot = portfolio.create_snapshot();
        
        assert_eq!(snapshot.positions.len(), 1);
        assert!(snapshot.positions.contains_key("EURUSD"));
        assert_eq!(snapshot.cash_balance, portfolio.get_cash_balance());
        assert_eq!(snapshot.total_pnl, portfolio.get_total_pnl());
    }

    #[test]
    fn test_return_on_capital() {
        let mut portfolio = Portfolio::new();
        
        // Make some profit
        let fill1 = create_test_fill("EURUSD", OrderSide::Buy, 1000.0, 1.1000);
        portfolio.update_position(fill1).unwrap();
        
        let fill2 = create_test_fill("EURUSD", OrderSide::Sell, 1000.0, 1.1100);
        portfolio.update_position(fill2).unwrap();
        
        let position = portfolio.get_position("EURUSD").unwrap();
        let expected_profit = 10.0; // 1000 * (1.1100 - 1.1000) = 10.0
        assert!((position.realized_pnl - expected_profit).abs() < 0.01);
        
        let roc = portfolio.get_return_on_capital();
        assert!((roc - expected_profit / 100000.0).abs() < 0.000001);
    }
}
    // Property-based tests (simplified implementation)
    
    #[test]
    fn prop_complete_audit_trail_simple() {
        // Feature: algorithmic-trading-system, Property 19: Complete Audit Trail
        let mut portfolio = Portfolio::new();
        
        // Test with a variety of fills
        let test_cases = vec![
            ("EURUSD", OrderSide::Buy, 1000.0, 1.1000, 1.0),
            ("GBPUSD", OrderSide::Sell, 500.0, 1.2500, 0.5),
            ("BTCUSD", OrderSide::Buy, 0.1, 50000.0, 5.0),
            ("EURUSD", OrderSide::Sell, 500.0, 1.1050, 1.0),
        ];
        
        let mut expected_fills = Vec::new();
        
        for (asset_id, side, quantity, price, commission) in test_cases {
            let fill = Fill {
                order_id: uuid::Uuid::new_v4(),
                asset_id: asset_id.to_string(),
                side,
                quantity,
                price,
                timestamp: chrono::Utc::now(),
                commission,
                metadata: HashMap::new(),
                execution_venue: "test_venue".to_string(),
                liquidity_flag: LiquidityFlag::Taker,
                slippage: 0.0,
            };
            
            expected_fills.push(fill.clone());
            portfolio.update_position(fill).unwrap();
        }
        
        // Verify audit trail completeness
        let stored_fills = portfolio.get_fills();
        assert_eq!(stored_fills.len(), expected_fills.len());
        
        // Verify all fills are stored with complete metadata
        for (original, stored) in expected_fills.iter().zip(stored_fills.iter()) {
            assert_eq!(original.order_id, stored.order_id);
            assert_eq!(&original.asset_id, &stored.asset_id);
            assert_eq!(original.quantity, stored.quantity);
            assert_eq!(original.price, stored.price);
            assert_eq!(original.commission, stored.commission);
        }
        
        // Verify sequence number increases monotonically
        let sequence = portfolio.get_sequence_number();
        assert_eq!(sequence as usize, expected_fills.len());
    }

    #[test]
    fn prop_position_consistency_simple() {
        // Test position consistency with multiple trades
        let mut portfolio = Portfolio::new();
        let asset_id = "EURUSD";
        let mut expected_quantity = 0.0f64;
        
        let trades = vec![
            (OrderSide::Buy, 1000.0, 1.1000),
            (OrderSide::Sell, 500.0, 1.1050),
            (OrderSide::Buy, 200.0, 1.1020),
            (OrderSide::Sell, 300.0, 1.1030),
        ];
        
        // Process trades and track expected quantity
        for (side, quantity, price) in &trades {
            let fill = Fill {
                order_id: uuid::Uuid::new_v4(),
                asset_id: asset_id.to_string(),
                side: side.clone(),
                quantity: *quantity,
                price: *price,
                timestamp: chrono::Utc::now(),
                commission: 1.0,
                metadata: HashMap::new(),
                execution_venue: "test_venue".to_string(),
                liquidity_flag: LiquidityFlag::Taker,
                slippage: 0.0,
            };
            
            portfolio.update_position(fill).unwrap();
            
            match side {
                OrderSide::Buy => expected_quantity += quantity,
                OrderSide::Sell => expected_quantity -= quantity,
            }
        }
        
        // Verify final position matches expected
        if let Some(position) = portfolio.get_position(asset_id) {
            // Allow for small floating point differences
            assert!((position.quantity - expected_quantity).abs() < 0.001);
        } else if expected_quantity.abs() < 0.001 {
            // Position should not exist if quantity is effectively zero
            assert!(true);
        } else {
            panic!("Position should exist but doesn't");
        }
    }