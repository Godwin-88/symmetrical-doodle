//! Derivatives Support for Execution Core
//!
//! Provides options, futures, and structured products execution capabilities.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Option type (Call or Put)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    Call,
    Put,
}

/// Option exercise style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionStyle {
    European,
    American,
}

/// Derivative instrument type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DerivativeType {
    Option {
        option_type: OptionType,
        style: OptionStyle,
        strike: f64,
        expiry: DateTime<Utc>,
    },
    Future {
        expiry: DateTime<Utc>,
        contract_size: f64,
        tick_size: f64,
    },
    Forward {
        expiry: DateTime<Utc>,
        settlement_price: f64,
    },
}

/// Greeks for options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64, rho: f64) -> Self {
        Self { delta, gamma, theta, vega, rho }
    }

    /// Aggregate Greeks from multiple positions
    pub fn aggregate(positions: &[DerivativePosition]) -> Self {
        let mut total = Greeks::default();
        for pos in positions {
            if let Some(ref greeks) = pos.greeks {
                let multiplier = if pos.is_long { 1.0 } else { -1.0 };
                total.delta += greeks.delta * pos.quantity as f64 * multiplier;
                total.gamma += greeks.gamma * pos.quantity as f64 * multiplier;
                total.theta += greeks.theta * pos.quantity as f64 * multiplier;
                total.vega += greeks.vega * pos.quantity as f64 * multiplier;
                total.rho += greeks.rho * pos.quantity as f64 * multiplier;
            }
        }
        total
    }
}

/// Derivative contract specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeContract {
    pub id: String,
    pub underlying: String,
    pub derivative_type: DerivativeType,
    pub contract_size: f64,
    pub currency: String,
    pub exchange: Option<String>,
    pub margin_requirement: f64,
    pub created_at: DateTime<Utc>,
}

impl DerivativeContract {
    /// Check if contract has expired
    pub fn is_expired(&self) -> bool {
        match &self.derivative_type {
            DerivativeType::Option { expiry, .. } => *expiry <= Utc::now(),
            DerivativeType::Future { expiry, .. } => *expiry <= Utc::now(),
            DerivativeType::Forward { expiry, .. } => *expiry <= Utc::now(),
        }
    }

    /// Get time to expiry in years
    pub fn time_to_expiry(&self) -> f64 {
        let expiry = match &self.derivative_type {
            DerivativeType::Option { expiry, .. } => expiry,
            DerivativeType::Future { expiry, .. } => expiry,
            DerivativeType::Forward { expiry, .. } => expiry,
        };

        let duration = *expiry - Utc::now();
        duration.num_seconds() as f64 / (365.25 * 24.0 * 60.0 * 60.0)
    }

    /// Get strike price for options
    pub fn strike(&self) -> Option<f64> {
        match &self.derivative_type {
            DerivativeType::Option { strike, .. } => Some(*strike),
            _ => None,
        }
    }
}

/// Derivative position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativePosition {
    pub id: String,
    pub contract: DerivativeContract,
    pub quantity: u32,
    pub is_long: bool,
    pub entry_price: f64,
    pub current_price: f64,
    pub greeks: Option<Greeks>,
    pub margin_used: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub opened_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl DerivativePosition {
    /// Calculate position value
    pub fn market_value(&self) -> f64 {
        let multiplier = if self.is_long { 1.0 } else { -1.0 };
        self.quantity as f64 * self.current_price * self.contract.contract_size * multiplier
    }

    /// Update position with new price
    pub fn update_price(&mut self, new_price: f64, new_greeks: Option<Greeks>) {
        self.current_price = new_price;
        if new_greeks.is_some() {
            self.greeks = new_greeks;
        }

        let multiplier = if self.is_long { 1.0 } else { -1.0 };
        self.unrealized_pnl = (new_price - self.entry_price)
            * self.quantity as f64
            * self.contract.contract_size
            * multiplier;
        self.last_updated = Utc::now();
    }

    /// Calculate intrinsic value for options
    pub fn intrinsic_value(&self, spot: f64) -> f64 {
        match &self.contract.derivative_type {
            DerivativeType::Option { option_type, strike, .. } => {
                match option_type {
                    OptionType::Call => (spot - strike).max(0.0),
                    OptionType::Put => (strike - spot).max(0.0),
                }
            }
            _ => 0.0,
        }
    }
}

/// Derivative order for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeOrder {
    pub id: String,
    pub contract_id: String,
    pub side: OrderSide,
    pub order_type: DerivativeOrderType,
    pub quantity: u32,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub filled_quantity: u32,
    pub average_fill_price: Option<f64>,
    pub commission: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeOrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeInForce {
    GoodTillCanceled,
    Day,
    ImmediateOrCancel,
    FillOrKill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Derivative portfolio manager
#[derive(Debug, Default)]
pub struct DerivativePortfolio {
    pub positions: HashMap<String, DerivativePosition>,
    pub orders: HashMap<String, DerivativeOrder>,
    pub contracts: HashMap<String, DerivativeContract>,
    pub cash: f64,
    pub margin_used: f64,
    pub margin_available: f64,
}

impl DerivativePortfolio {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            positions: HashMap::new(),
            orders: HashMap::new(),
            contracts: HashMap::new(),
            cash: initial_cash,
            margin_used: 0.0,
            margin_available: initial_cash,
        }
    }

    /// Get total portfolio value
    pub fn total_value(&self) -> f64 {
        let positions_value: f64 = self.positions.values()
            .map(|p| p.market_value())
            .sum();
        self.cash + positions_value
    }

    /// Get aggregated Greeks
    pub fn net_greeks(&self) -> Greeks {
        let positions: Vec<_> = self.positions.values().cloned().collect();
        Greeks::aggregate(&positions)
    }

    /// Get total unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum()
    }

    /// Get total realized P&L
    pub fn realized_pnl(&self) -> f64 {
        self.positions.values()
            .map(|p| p.realized_pnl)
            .sum()
    }

    /// Add a new contract
    pub fn add_contract(&mut self, contract: DerivativeContract) {
        self.contracts.insert(contract.id.clone(), contract);
    }

    /// Open a new position
    pub fn open_position(&mut self, position: DerivativePosition) -> Result<(), String> {
        let margin_required = position.quantity as f64
            * position.entry_price
            * position.contract.contract_size
            * position.contract.margin_requirement;

        if margin_required > self.margin_available {
            return Err("Insufficient margin".to_string());
        }

        self.margin_used += margin_required;
        self.margin_available -= margin_required;
        self.cash -= position.quantity as f64 * position.entry_price * position.contract.contract_size;
        self.positions.insert(position.id.clone(), position);

        Ok(())
    }

    /// Close a position
    pub fn close_position(&mut self, position_id: &str, close_price: f64) -> Result<f64, String> {
        let position = self.positions.remove(position_id)
            .ok_or("Position not found")?;

        let multiplier = if position.is_long { 1.0 } else { -1.0 };
        let pnl = (close_price - position.entry_price)
            * position.quantity as f64
            * position.contract.contract_size
            * multiplier;

        // Release margin
        let margin_released = position.margin_used;
        self.margin_used -= margin_released;
        self.margin_available += margin_released;

        // Add proceeds/loss to cash
        self.cash += position.quantity as f64 * close_price * position.contract.contract_size + pnl;

        Ok(pnl)
    }

    /// Update all positions with new prices
    pub fn update_prices(&mut self, prices: &HashMap<String, (f64, Option<Greeks>)>) {
        for (contract_id, (price, greeks)) in prices {
            for position in self.positions.values_mut() {
                if position.contract.id == *contract_id {
                    position.update_price(*price, greeks.clone());
                }
            }
        }
    }

    /// Check and handle expired positions
    pub fn handle_expirations(&mut self, spot_prices: &HashMap<String, f64>) -> Vec<(String, f64)> {
        let mut expired_pnl = Vec::new();
        let expired_ids: Vec<String> = self.positions.iter()
            .filter(|(_, p)| p.contract.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired_ids {
            if let Some(position) = self.positions.get(&id) {
                let spot = spot_prices.get(&position.contract.underlying).copied().unwrap_or(0.0);
                let settlement_value = position.intrinsic_value(spot);

                if let Ok(pnl) = self.close_position(&id, settlement_value) {
                    expired_pnl.push((id, pnl));
                }
            }
        }

        expired_pnl
    }
}

/// Risk limits for derivatives trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeRiskLimits {
    pub max_delta: f64,
    pub max_gamma: f64,
    pub max_vega: f64,
    pub max_theta: f64,
    pub max_position_size: u32,
    pub max_notional: f64,
    pub max_margin_usage: f64,
}

impl Default for DerivativeRiskLimits {
    fn default() -> Self {
        Self {
            max_delta: 100.0,
            max_gamma: 10.0,
            max_vega: 50.0,
            max_theta: -100.0,
            max_position_size: 1000,
            max_notional: 1_000_000.0,
            max_margin_usage: 0.8,
        }
    }
}

impl DerivativeRiskLimits {
    /// Check if portfolio violates any limits
    pub fn check_limits(&self, portfolio: &DerivativePortfolio) -> Vec<String> {
        let mut violations = Vec::new();
        let greeks = portfolio.net_greeks();

        if greeks.delta.abs() > self.max_delta {
            violations.push(format!("Delta limit exceeded: {:.2} > {:.2}", greeks.delta.abs(), self.max_delta));
        }
        if greeks.gamma.abs() > self.max_gamma {
            violations.push(format!("Gamma limit exceeded: {:.2} > {:.2}", greeks.gamma.abs(), self.max_gamma));
        }
        if greeks.vega.abs() > self.max_vega {
            violations.push(format!("Vega limit exceeded: {:.2} > {:.2}", greeks.vega.abs(), self.max_vega));
        }
        if greeks.theta < self.max_theta {
            violations.push(format!("Theta limit exceeded: {:.2} < {:.2}", greeks.theta, self.max_theta));
        }

        let margin_ratio = portfolio.margin_used / (portfolio.margin_used + portfolio.margin_available);
        if margin_ratio > self.max_margin_usage {
            violations.push(format!("Margin usage exceeded: {:.1}% > {:.1}%", margin_ratio * 100.0, self.max_margin_usage * 100.0));
        }

        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_option_contract() {
        let expiry = Utc::now() + Duration::days(30);
        let contract = DerivativeContract {
            id: "OPT-001".to_string(),
            underlying: "XAUUSD".to_string(),
            derivative_type: DerivativeType::Option {
                option_type: OptionType::Call,
                style: OptionStyle::European,
                strike: 2000.0,
                expiry,
            },
            contract_size: 100.0,
            currency: "USD".to_string(),
            exchange: Some("CME".to_string()),
            margin_requirement: 0.2,
            created_at: Utc::now(),
        };

        assert!(!contract.is_expired());
        assert!(contract.time_to_expiry() > 0.0);
        assert_eq!(contract.strike(), Some(2000.0));
    }

    #[test]
    fn test_portfolio_operations() {
        let mut portfolio = DerivativePortfolio::new(100_000.0);

        let expiry = Utc::now() + Duration::days(30);
        let contract = DerivativeContract {
            id: "OPT-001".to_string(),
            underlying: "BTCUSD".to_string(),
            derivative_type: DerivativeType::Option {
                option_type: OptionType::Call,
                style: OptionStyle::European,
                strike: 45000.0,
                expiry,
            },
            contract_size: 1.0,
            currency: "USD".to_string(),
            exchange: None,
            margin_requirement: 0.5,
            created_at: Utc::now(),
        };

        let position = DerivativePosition {
            id: "POS-001".to_string(),
            contract: contract.clone(),
            quantity: 10,
            is_long: true,
            entry_price: 1500.0,
            current_price: 1500.0,
            greeks: Some(Greeks::new(0.5, 0.02, -10.0, 50.0, 5.0)),
            margin_used: 7500.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            opened_at: Utc::now(),
            last_updated: Utc::now(),
        };

        portfolio.add_contract(contract);
        assert!(portfolio.open_position(position).is_ok());
        assert_eq!(portfolio.positions.len(), 1);

        let greeks = portfolio.net_greeks();
        assert!((greeks.delta - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_greeks_aggregation() {
        let positions = vec![
            DerivativePosition {
                id: "1".to_string(),
                contract: DerivativeContract {
                    id: "C1".to_string(),
                    underlying: "TEST".to_string(),
                    derivative_type: DerivativeType::Option {
                        option_type: OptionType::Call,
                        style: OptionStyle::European,
                        strike: 100.0,
                        expiry: Utc::now() + Duration::days(30),
                    },
                    contract_size: 100.0,
                    currency: "USD".to_string(),
                    exchange: None,
                    margin_requirement: 0.2,
                    created_at: Utc::now(),
                },
                quantity: 10,
                is_long: true,
                entry_price: 5.0,
                current_price: 5.5,
                greeks: Some(Greeks::new(0.5, 0.05, -0.1, 0.2, 0.01)),
                margin_used: 1000.0,
                unrealized_pnl: 500.0,
                realized_pnl: 0.0,
                opened_at: Utc::now(),
                last_updated: Utc::now(),
            },
        ];

        let aggregated = Greeks::aggregate(&positions);
        assert!((aggregated.delta - 5.0).abs() < 0.01);
        assert!((aggregated.gamma - 0.5).abs() < 0.01);
    }
}
